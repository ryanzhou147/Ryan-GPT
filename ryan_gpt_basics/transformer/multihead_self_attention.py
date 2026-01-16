import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import einsum, rearrange
from ryan_gpt_basics.utility import scaled_dot_product_attention
from ryan_gpt_basics.transformer.rope import RotaryPositionalEmbedding

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None = None, max_seq_len: int | None = None, with_rope: bool = False,
                 device=None, dtype=None, use_flash: bool = True) -> None:

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.w_q = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty((num_heads * self.d_k, d_model), device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty((num_heads * self.d_v, d_model), device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty((d_model, num_heads * self.d_v), device=device, dtype=dtype))

        std = math.sqrt(2 / (d_model + num_heads * self.d_k))
        init.trunc_normal_(self.w_q, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_k, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_v, mean=0.0, std=std, a=-3*std, b=3*std)
        init.trunc_normal_(self.w_o, mean=0.0, std=std, a=-3*std, b=3*std)

        self.with_rope = with_rope
        self.use_flash = use_flash

        try:
            from ryan_gpt_systems.flash_attention import flash_attention_triton, flash_attention_pytorch
            self._flash_available = True
            self._flash_triton = flash_attention_triton
            self._flash_pytorch = flash_attention_pytorch
        except Exception:
            self._flash_available = False
            self._flash_triton = None
            self._flash_pytorch = None

        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device) if with_rope else None

        if max_seq_len is not None:
            causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))
            self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)
        else:
            self.causal_mask = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = einsum(x, self.w_q, 'b s d_in, h_dk d_in -> b s h_dk')
        K = einsum(x, self.w_k, 'b s d_in, h_dk d_in -> b s h_dk')
        V = einsum(x, self.w_v, 'b s d_in, h_dv d_in -> b s h_dv')

        Q = rearrange(Q, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, 'b s (h d_k) -> b h s d_k', h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, 'b s (h d_v) -> b h s d_v', h=self.num_heads, d_v=self.d_v)

        _, seq_len, _ = x.size()

        if self.rope is not None and token_positions is not None:
            b, h, s, d = Q.shape
            Q_flat = Q.reshape(b * h, s, d)
            K_flat = K.reshape(b * h, s, d)
            pos_expanded = token_positions.unsqueeze(1).expand(b, h, s).reshape(b * h, s)
            Q_flat = self.rope(Q_flat, pos_expanded)
            K_flat = self.rope(K_flat, pos_expanded)
            Q = Q_flat.reshape(b, h, s, d)
            K = K_flat.reshape(b, h, s, d)

        if self.causal_mask is not None:
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        else:
            causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=x.device, dtype=torch.bool))

        attn_output = None
        if self.use_flash and self._flash_available:
            is_causal = causal_mask is not None and causal_mask.shape[-2:] == (seq_len, seq_len)
            try:
                b, h, s, d = Q.shape
                Q_flat = Q.reshape(b * h, s, d)
                K_flat = K.reshape(b * h, s, d)
                V_flat = V.reshape(b * h, s, d)
                if self._flash_pytorch is not None:
                    out_flat = self._flash_pytorch(Q_flat, K_flat, V_flat, is_causal)
                elif self._flash_triton is not None:
                    out_flat = self._flash_triton(Q_flat, K_flat, V_flat, is_causal)
                else:
                    out_flat = None
                attn_output = out_flat.reshape(b, h, s, d)
            except Exception:
                attn_output = None

        if attn_output is None:
            attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        attn_output = rearrange(attn_output, 'b h s d_v -> b s (h d_v)')
        return einsum(attn_output, self.w_o, 'b s h_dv, d_out h_dv -> b s d_out')
