import functools

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
import math

_LOG2E = 1.4426950408889634

@functools.lru_cache(maxsize=32)
def fused_prepare_compute_w_u_tl(
    total_chunks: int,
    total_tokens: int,
    batch: int,
    head: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float16",
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-3, -2, -1],  # output Tensor：w, u, cu_g
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DK=64, block_DV=64):
        assert dim_k % block_DK == 0, "dim_k must be divisible by block_DK"
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.macro
        def _fused_body(
            k: T.Tensor([total_tokens, head, dim_k], dtype),
            v: T.Tensor([total_tokens, head, dim_v], dtype),
            alpha: T.Tensor([total_tokens, head], dtype),
            beta: T.Tensor([total_tokens, head], dtype),
            cu_seqlens: T.Tensor([batch + 1], "int32"),
            chunk_offsets: T.Tensor([batch + 1], "int32"),
            w: T.Tensor([total_tokens, head, dim_k], dtype),
            u: T.Tensor([total_tokens, head, dim_v], dtype),
            cu_g: T.Tensor([total_tokens, head], dtype),
        ):
            with T.Kernel(total_chunks, head, threads=threads) as (tid, hid):
                batch_idx = T.int32(0)
                low = T.int32(0)
                high = T.int32(batch)
                for _ in T.serial(32):
                    if low <= high:
                        mid = (low + high) // 2
                        if chunk_offsets[mid] <= tid:
                            batch_idx = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                local_chunk_idx = tid - chunk_offsets[batch_idx]

                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]

                chunk_token_start = seq_start + local_chunk_idx * block_C
                chunk_token_end = T.min(chunk_token_start + block_C, seq_end)
                actual_len = chunk_token_end - chunk_token_start

                # share memory
                alpha_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                cu_g_shared = T.alloc_shared([block_C], dtype)
                k_shared = T.alloc_shared([block_C, block_DK], dtype)
                v_shared = T.alloc_shared([block_C, block_DV], dtype)
                k_beta_shared = T.alloc_shared([block_C, block_DK], dtype)
                v_beta_shared = T.alloc_shared([block_C, block_DV], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                w_shared = T.alloc_shared([block_C, block_DK], dtype)
                u_shared = T.alloc_shared([block_C, block_DV], dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, block_DK], accum_dtype)
                u_frag = T.alloc_fragment([block_C, block_DV], accum_dtype)

                T.copy(
                    alpha[chunk_token_start : chunk_token_start + actual_len, hid],
                    alpha_shared[:actual_len],
                )
                T.copy(
                    beta[chunk_token_start : chunk_token_start + actual_len, hid],
                    beta_shared[:actual_len],
                )

                acc = T.float32(0.0)
                for i in T.serial(block_C):
                    if i < actual_len:
                        acc = acc + T.log(alpha_shared[i])
                    cu_g_shared[i] = acc

                # --- KKT = k @ k^T ---
                T.clear(gram_frag)
                for i_k in T.Pipelined(dim_k // block_DK, num_stages=num_stages):
                    k_start = i_k * block_DK
                    T.copy(
                        k[
                            chunk_token_start : chunk_token_start + actual_len,
                            hid,
                            k_start : k_start + block_DK,
                        ],
                        k_shared[:actual_len, :],
                    )
                    T.gemm(
                        k_shared[:actual_len, :],
                        k_shared[:actual_len, :],
                        gram_frag[:actual_len, :actual_len],
                        transpose_B=True,
                    )

                # --- Compute A_g = (I + strictLower(diag(β)·(Γ⊙KK^T)))^{-1} ---
                # Paper uses a SINGLE gated matrix for both w and u.
                # P = -strictLower(diag(β)·(Γ⊙KK^T)) for Neumann: (I-P)^{-1} = (I+strictLower(...))^{-1}
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i < actual_len and j < i,
                        -gram_frag[i, j]
                        * beta_shared[i]
                        * T.exp2((cu_g_shared[i] - cu_g_shared[j]) * _LOG2E),
                        T.float32(0),
                    )

                T.clear(S_shared)
                for i in T.serial(actual_len):
                    S_shared[i, i] = 1.0

                for _r in T.serial(num_rounds):
                    T.gemm(P_shared, S_shared, temp_frag, clear_accum=True)
                    for i, j in T.Parallel(actual_len, actual_len):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.gemm(P_shared, P_shared, temp_frag, clear_accum=True)
                    T.copy(
                        temp_frag[:actual_len, :actual_len],
                        P_shared[:actual_len, :actual_len],
                    )

                for i_k in T.Pipelined(dim_k // block_DK, num_stages=num_stages):
                    k_start = i_k * block_DK
                    T.copy(
                        k[
                            chunk_token_start : chunk_token_start + actual_len,
                            hid,
                            k_start : k_start + block_DK,
                        ],
                        k_shared[:actual_len, :],
                    )
                    for i_s, i_k2 in T.Parallel(actual_len, block_DK):
                        k_beta_shared[i_s, i_k2] = (
                            k_shared[i_s, i_k2] * beta_shared[i_s]
                        )
                    T.gemm(
                        S_shared[:actual_len, :actual_len],
                        k_beta_shared[:actual_len, :],
                        w_frag[:actual_len, :],
                        clear_accum=True,
                    )
                    T.copy(w_frag[:actual_len, :], w_shared[:actual_len, :])
                    T.copy(
                        w_shared[:actual_len, :],
                        w[
                            chunk_token_start : chunk_token_start + actual_len,
                            hid,
                            k_start : k_start + block_DK,
                        ],
                    )

                for i_v in T.Pipelined(dim_v // block_DV, num_stages=num_stages):
                    v_start = i_v * block_DV
                    T.copy(
                        v[
                            chunk_token_start : chunk_token_start + actual_len,
                            hid,
                            v_start : v_start + block_DV,
                        ],
                        v_shared[:actual_len, :],
                    )
                    for i_s, i_v2 in T.Parallel(actual_len, block_DV):
                        v_beta_shared[i_s, i_v2] = (
                            v_shared[i_s, i_v2] * beta_shared[i_s]
                        )
                    T.gemm(
                        S_shared[:actual_len, :actual_len],
                        v_beta_shared[:actual_len, :],
                        u_frag[:actual_len, :],
                        clear_accum=True,
                    )
                    T.copy(u_frag[:actual_len, :], u_shared[:actual_len, :])
                    T.copy(
                        u_shared[:actual_len, :],
                        u[
                            chunk_token_start : chunk_token_start + actual_len,
                            hid,
                            v_start : v_start + block_DV,
                        ],
                    )

                T.copy(
                    cu_g_shared[:actual_len],
                    cu_g[chunk_token_start : chunk_token_start + actual_len, hid],
                )

        @T.prim_func
        def fused_prepare_compute_w_u(
            k: T.Tensor([total_tokens, head, dim_k], dtype),
            v: T.Tensor([total_tokens, head, dim_v], dtype),
            alpha: T.Tensor([total_tokens, head], dtype),
            beta: T.Tensor([total_tokens, head], dtype),
            cu_seqlens: T.Tensor([batch + 1], "int32"),
            chunk_offsets: T.Tensor([batch + 1], "int32"),
            w: T.Tensor([total_tokens, head, dim_k], dtype),
            u: T.Tensor([total_tokens, head, dim_v], dtype),
            cu_g: T.Tensor([total_tokens, head], dtype),
        ):
            _fused_body(k, v, alpha, beta, cu_seqlens, chunk_offsets, w, u, cu_g)

        return fused_prepare_compute_w_u

    return _func

@functools.lru_cache(maxsize=32)
def _h_recurrence_tl(
    total_tokens: int,
    batch: int,
    head: int,
    chunk_size: int,
    max_chunks: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float16",
):
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DV=64):
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.prim_func
        def h_recurrence_kernel(
            k: T.Tensor([total_tokens, head, dim_k], dtype),
            g: T.Tensor([total_tokens, head], dtype),
            w: T.Tensor([total_tokens, head, dim_k], dtype),
            u: T.Tensor([total_tokens, head, dim_v], dtype),
            cu_seqlens: T.Tensor([batch+1], "int32"),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, max_chunks+1, dim_k, dim_v], dtype),
            v_new: T.Tensor([total_tokens, head, dim_v], dtype),
        ):
            with T.Kernel(dim_v//block_DV, batch, head, threads=threads) as (vid, bid, hid):
                seq_start = cu_seqlens[bid]
                seq_end = cu_seqlens[bid + 1]
                seqlen = seq_end - seq_start
                num_chunks = (seqlen + block_C - 1) // block_C

                v_offset = vid * block_DV

                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, block_DV], dtype)
                h_c = T.alloc_shared([dim_k, block_DV], dtype)
                v_new_c = T.alloc_shared([block_C, block_DV], dtype)
                k_scaled_s = T.alloc_shared([block_C, dim_k], dtype)
                ws_frag = T.alloc_fragment([block_C, block_DV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, block_DV], accum_dtype)

                # initialize h
                T.copy(
                    S_0[bid, hid, :, v_offset : v_offset + block_DV], h_c, disable_tma=True
                )
                T.copy(h_c, S[bid, hid, 0, :, v_offset: v_offset+block_DV], disable_tma=True)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    chunk_token_start = seq_start + t * block_C
                    chunk_token_end = T.min(chunk_token_start + block_C, seq_end)
                    actual_len = chunk_token_end - chunk_token_start

                    T.copy(
                        k[chunk_token_start : chunk_token_start + actual_len, hid, :],
                        k_c[:actual_len, :],
                    )
                    T.copy(
                        w[chunk_token_start : chunk_token_start + actual_len, hid, :],
                        w_c[:actual_len, :],
                    )
                    T.copy(
                        u[chunk_token_start : chunk_token_start + actual_len, hid, v_offset : v_offset + block_DV],
                        u_c[:actual_len, :],
                    )
                    T.copy(
                        g[chunk_token_start : chunk_token_start + actual_len, hid],
                        g_c[:actual_len],
                    )

                    g_last_val = g_c[actual_len - 1]
                    T.gemm(w_c[:actual_len, :], h_c, ws_frag[:actual_len, :], clear_accum=True)
                    for i, j in T.Parallel(actual_len, block_DV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_last_val) * _LOG2E
                        )

                    T.copy(
                        v_new_c[:actual_len, :],
                        v_new[chunk_token_start : chunk_token_start + actual_len, hid, v_offset : v_offset + block_DV],
                    )

                    for n, kk in T.Parallel(actual_len, dim_k):
                        k_scaled_s[n, kk] = k_c[n, kk] * T.exp2(
                            (g_last_val - g_c[n]) * _LOG2E
                        )
                    for i, j in T.Parallel(dim_k, block_DV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_last_val * _LOG2E)
                    T.gemm(
                        k_scaled_s[:actual_len, :],
                        v_new_c[:actual_len, :],
                        h_next_frag,
                        transpose_A=True,
                    )
                    T.copy(h_next_frag, h_c)
                    T.copy(h_c, S[bid, hid, t+1, :, v_offset: v_offset+block_DV], disable_tma=True)
        return h_recurrence_kernel

    return _func

@functools.lru_cache(maxsize=32)
def _output_o_tl(
    total_chunks: int,
    total_tokens: int,
    batch: int,
    head_q: int,
    head_kv: int,
    group_size: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DV=64):
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.prim_func
        def output_o_kernel(
            q: T.Tensor([total_tokens, head_q, dim_k], dtype),
            k: T.Tensor([total_tokens, head_kv, dim_k], dtype),
            g: T.Tensor([total_tokens, head_kv], dtype),
            cu_seqlens: T.Tensor([batch+1], "int32"),
            chunk_offsets: T.Tensor([batch+1], "int32"),
            S: T.Tensor([total_tokens, head, dim_k, dim_v], dtype),
            v_new: T.Tensor([total_tokens, head_kv, dim_v], dtype),
            o: T.Tensor([total_tokens, head_q, dim_v], dtype),
        ):
            with T.Kernel(total_chunks, head_q, threads=threads) as (tid, q_hid):
                kv_hid = q_hid // group_size

                batch_idx = T.int32(0)
                low = T.int32(0)
                high = T.int32(batch)
                for _ in T.serial(32):
                    if low <= high:
                        mid = (low + high) // 2
                        if chunk_offsets[mid] <= tid:
                            batch_idx = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                local_chunk_idx = tid - chunk_offsets[batch_idx]

                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]

                chunk_token_start = seq_start + local_chunk_idx * block_C
                chunk_token_end = T.min(chunk_token_start + block_C, seq_end)
                actual_len = chunk_token_end - chunk_token_start

                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                T,copy(q[])
                # 1. 加载 q（使用 q_hid）
                for i in T.serial(block_C):
                    cond = i < actual_len
                    off = chunk_token_start + i
                    for d in T.serial(dim_k):
                        q_c[i, d] = T.if_then_else(cond, q[off, q_hid, d], 0.0)

                # 2. 加载 k, g（使用 kv_hid）
                for i in T.serial(block_C):
                    cond = i < actual_len
                    off = chunk_token_start + i
                    for d in T.serial(dim_k):
                        k_c[i, d] = T.if_then_else(cond, k[off, kv_hid, d], 0.0)
                    g_c[i] = T.if_then_else(cond, g[off, kv_hid], 0.0)

                # 3. 加载隐藏状态 h = S[batch_idx, kv_hid, local_chunk_idx]
                T.copy(S[batch_idx, kv_hid, local_chunk_idx, :, :], h_c, disable_tma=True)

                # 4. 加载 v_new（使用 kv_hid）
                for i in T.serial(block_C):
                    cond = i < actual_len
                    off = chunk_token_start + i
                    for d in T.serial(dim_v):
                        v_new_c[i, d] = T.if_then_else(cond, v_new[off, kv_hid, d], 0.0)

                # 5. 计算 o = (q @ h) * exp(g)
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    if i < actual_len:
                        o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                # 6. 计算块内注意力 attn
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    if i < actual_len and j < actual_len and i >= j:
                        attn[i, j] = attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E)
                    else:
                        attn[i, j] = 0.0

                # 7. o += attn @ v_new
                T.gemm(attn, v_new_c, o_frag)

                # 8. 写回 o（使用 q_hid）
                for i in T.serial(block_C):
                    cond = i < actual_len
                    off = chunk_token_start + i
                    for d in T.serial(dim_v):
                        if cond:
                            o[off, q_hid, d] = o_frag[i, d]
    return _func


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    batch = cu_seqlens.size(0) - 1
    head_q = q.size(1)
    head_kv = k.size(1)
    dim_k = k.size(2)
    dim_v = v.size(2)
    assert head_q % head_kv == 0, "head_q must be multiple of head_kv"

    # compute chunk metadata
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_batch = (seq_lens + chunk_size - 1) // chunk_size
    total_chunks = chunks_per_batch.sum().item()
    max_chunks = chunks_per_batch.max().item()
    chunk_offsets = torch.cat(
        [torch.tensor([0], device=cu_seqlens.device), chunks_per_batch.cumsum(0)]
    )

    # ---------- kernel1：w, u ----------
    fused_fn = fused_prepare_compute_w_u_tl(
        total_chunks,
        k.size(0),
        batch,
        head_kv,
        chunk_size,
        dim_k,
        dim_v,
        dtype=str(q.dtype),
    )(fused_num_stages, fused_threads)
    w, u, cu_g = fused_fn(k, v, alpha, beta, cu_seqlens, chunk_offsets)

    # ---------- kernel 2：recurrent state ----------
    h_fn = _h_recurrence_tl(
        batch, head_kv, chunk_size, max_chunks, dim_k, dim_v, dtype=str(q.dtype)
    )(h_num_stages, h_threads)
    S_0 = torch.zeros(batch, head_kv, dim_k, dim_v, dtype=q.dtype, device=q.device)
    S_buf, v_new = h_fn(k, cu_g, w, u, cu_seqlens, S_0)
    # S_buf shape: [batch, head_kv, max_chunks+1, dim_k, dim_v]

    # ---------- kernel 3：output ----------
    o_fn = _output_o_tl(
        total_chunks, head_q, head_kv, chunk_size, dim_k, dim_v, dtype=str(q.dtype)
    )(o_threads)
    o = o_fn(q, k, g, cu_seqlens, chunk_offsets, S_buf, v_new)

    return o, S_buf
