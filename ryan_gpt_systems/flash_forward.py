import torch
import math
import triton
import triton.language as tl

class FlashAttentionFunctionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        _, Nq, d = Q.shape
        _, Nk, _ = K.shape
        
        scale = 1.0 / math.sqrt(d)
        
        Bq, Bk = 16, 16
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        
        O = torch.zeros_like(Q)  # (batch, Nq, d)
        L = torch.zeros(Q.shape[0], Nq, device=Q.device, dtype=Q.dtype)  # (batch, Nq)
        
        # Loop over batch
        for b in range(Q.shape[0]):
            Qb = Q[b]  # (Nq, d)
            Kb = K[b]  # (Nk, d)
            Vb = V[b]  # (Nk, d)
            
            for i in range(Tq):
                i_start, i_end = i * Bq, min((i + 1) * Bq, Nq)
                
                Qi = Qb[i_start:i_end]  # (Bq, d)
                Oi = torch.zeros_like(Qi)
                li = torch.zeros(Qi.shape[0], device=Q.device, dtype=Q.dtype)
                mi = torch.full((Qi.shape[0],), float('-inf'), device=Q.device, dtype=Q.dtype)
                
                for j in range(Tk):
                    j_start, j_end = j * Bk, min((j + 1) * Bk, Nk)
                    
                    Kj = Kb[j_start:j_end]  # (Bk, d)
                    Vj = Vb[j_start:j_end]  # (Bk, d)
                    
                    # S_ij = Qi @ K_j^T / sqrt(d)
                    Sij = (Qi @ Kj.T) * scale  # (Bq, Bk)

                    if is_causal:
                        q_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(-1)
                        k_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                        Sij = Sij.masked_fill(q_idx < k_idx, -1e6)
                    
                    # m_new = max(m_old, rowmax(S_ij))
                    mi_new = torch.maximum(mi, Sij.max(dim=-1).values)
                    
                    # P_ij = exp(S_ij - m_new)
                    Pij = torch.exp(Sij - mi_new.unsqueeze(-1))
                    
                    # alpha = exp(m_old - m_new)
                    alpha = torch.exp(mi - mi_new)
                    
                    # l_new = alpha * l_old + rowsum(P_ij)
                    li = alpha * li + Pij.sum(dim=-1)
                    
                    # O_new = alpha * O_old + P_ij @ V_j
                    Oi = alpha.unsqueeze(-1) * Oi + Pij @ Vj
                    
                    mi = mi_new
                
                # Final normalization
                O[b, i_start:i_end] = Oi / li.unsqueeze(-1)
                L[b, i_start:i_end] = mi + torch.log(li)
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        
        batch, Nq, d = Q.shape
        _, Nk, _ = K.shape
        scale = 1.0 / math.sqrt(d)
        
        Bq, Bk = 16, 16
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        
        # Initialize gradients
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Precompute Di = rowsum(dO * O) for all positions
        Di = torch.sum(grad_output * O, dim=-1)  # (batch, Nq)
        
        for b in range(batch):
            Qb = Q[b]      # (Nq, d)
            Kb = K[b]      # (Nk, d)
            Vb = V[b]      # (Nk, d)
            Ob = O[b]      # (Nq, d)
            Lb = L[b]      # (Nq,)
            dOb = grad_output[b]  # (Nq, d)
            Dib = Di[b]    # (Nq,)
            
            for i in range(Tq):
                i_start, i_end = i * Bq, min((i + 1) * Bq, Nq)
                
                Qi = Qb[i_start:i_end]      # (Bq, d)
                Oi = Ob[i_start:i_end]      # (Bq, d)
                Li = Lb[i_start:i_end]      # (Bq,)
                dOi = dOb[i_start:i_end]    # (Bq, d)
                Dii = Dib[i_start:i_end]    # (Bq,)
                
                dQi = torch.zeros_like(Qi)  # Accumulate dQ for this tile
                
                for j in range(Tk):
                    j_start, j_end = j * Bk, min((j + 1) * Bk, Nk)
                    
                    Kj = Kb[j_start:j_end]  # (Bk, d)
                    Vj = Vb[j_start:j_end]  # (Bk, d)
                    
                    # Recompute attention scores for this tile
                    Sij = (Qi @ Kj.T) * scale  # (Bq, Bk)
                    
                    # Recompute P using saved logsumexp
                    Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (Bq, Bk)
                    
                    # dV += P^T @ dO
                    dV[b, j_start:j_end] += Pij.T @ dOi  # (Bk, d)
                    
                    # dP = dO @ V^T
                    dPij = dOi @ Vj.T  # (Bq, Bk)
                    
                    # dS = P * (dP - Di)
                    dSij = Pij * (dPij - Dii.unsqueeze(-1))  # (Bq, Bk)
                    
                    # dQ += dS @ K * scale
                    dQi += (dSij @ Kj) * scale  # (Bq, d)
                    
                    # dK += dS^T @ Q * scale
                    dK[b, j_start:j_end] += (dSij.T @ Qi) * scale  # (Bk, d)
                
                dQ[b, i_start:i_end] = dQi
        
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )
    
    # Load Q once
    Qi = tl.load(Q_block_ptr)
    
    # On-chip accumulators (float32 for precision)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    
    # Loop over key tiles
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)
        
        # Attention scores
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        
        # Causal mask
        if IS_CAUSAL:
            q_idx = query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            Sij = tl.where(q_idx[:, None] >= k_idx[None, :], Sij, -1e6)
        
        # Online softmax
        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))
        Pij = tl.exp(Sij - mi_new[:, None])
        alpha = tl.exp(mi - mi_new)
        li = alpha * li + tl.sum(Pij, axis=1)
        Oi = alpha[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj)
        mi = mi_new
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Normalize and store
    Oi = Oi / li[:, None]
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, (mi + tl.log(li)).to(L_block_ptr.type.element_ty))


class FlashAttentionFunctionTriton(torch.autograd.Function):
    
    Q_TILE_SIZE = 64
    K_TILE_SIZE = 64

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, Nq, d = Q.shape
        _, Nk, _ = K.shape
        
        O = torch.empty_like(Q)
        L = torch.empty(batch, Nq, device=Q.device, dtype=Q.dtype)
        
        grid = (triton.cdiv(Nq, FlashAttentionFunctionTriton.Q_TILE_SIZE), batch)
        
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            1.0 / math.sqrt(d),
            is_causal,
            d,
            FlashAttentionFunctionTriton.Q_TILE_SIZE,
            FlashAttentionFunctionTriton.K_TILE_SIZE,
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError