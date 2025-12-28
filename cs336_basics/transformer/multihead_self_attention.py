import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import einsum, rearrange
from ryan_gpt_basics.utility import scaled_dot_product_attention
from ryan_gpt_basics.transformer.rope import RotaryPositionalEmbedding
from ryan_gpt_basics.transformer.linear import Linear

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None = None, max_seq_len: int | None = None, with_rope: bool = False,
                 device=None, dtype=None) -> None:

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.w_q = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty((num_heads * self.d_v, d_model), device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty((d_model, num_heads * self.d_v), device=device, dtype=dtype))

        # Initialize weights
        std = math.sqrt(2 / (d_model + num_heads * self.d_k))
        init.trunc_normal_(self.w_q, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_k, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_v, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_o, mean=0.0, std=std, a=-3*std, b=3*std)

        self.with_rope = with_rope

        if self.with_rope:
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        
        # Cache causal mask to avoid recreating every forward pass
        if max_seq_len is not None:
            causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))
            self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)
        else:
            self.causal_mask = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            token_positions: Optional token positions for RoPE
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """

        # Project to Q, K, V using einsum: x @ W^T
        # x: (b s d_model), W: (h_dk d_model) -> result: (b s h_dk)
        Q = einsum(x, self.w_q, 'b s d_in, h_dk d_in -> b s h_dk')
        K = einsum(x, self.w_k, 'b s d_in, h_dk d_in -> b s h_dk')
        V = einsum(x, self.w_v, 'b s d_in, h_dv d_in -> b s h_dv')
        
        # Reshape to separate heads using rearrange
        # (batch, seq_len, num_heads * d_k) -> (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, 'b s (h d_v) -> b h s d_v', h=self.num_heads, d_v=self.d_v)
        
        _, seq_len, _ = x.size()
        
        if self.rope is not None and token_positions is not None:
            # Apply RoPE to Q and K - vectorized across all heads
            # Q, K shape: (batch, num_heads, seq_len, d_k)
            # Reshape to (batch * num_heads, seq_len, d_k) for RoPE
            b, h, s, d = Q.shape
            Q_flat = Q.reshape(b * h, s, d)
            K_flat = K.reshape(b * h, s, d)
            
            # Expand token_positions: (batch, seq) -> (batch * num_heads, seq)
            pos_expanded = token_positions.unsqueeze(1).expand(b, h, s).reshape(b * h, s)
            
            Q_flat = self.rope(Q_flat, pos_expanded)
            K_flat = self.rope(K_flat, pos_expanded)
            
            Q = Q_flat.reshape(b, h, s, d)
            K = K_flat.reshape(b, h, s, d)
        
        # Use cached causal mask (sliced to current seq_len) or create if not cached
        if self.causal_mask is not None:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        else:
            causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=x.device, dtype=torch.bool))

        # Attention: (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, d_v)
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # Concatenate heads using rearrange
        # (batch, num_heads, seq_len, d_v) -> (batch, seq_len, num_heads * d_v)
        attn_output = rearrange(attn_output, 'b h s d_v -> b s (h d_v)')
        
        # Output projection using einsum
        # (batch, seq_len, h·dv) @ W_o^T -> (batch, seq_len, d_model)
        output = einsum(attn_output, self.w_o, 'b s h_dv, d_out h_dv -> b s d_out')
        
        return output
