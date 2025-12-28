import torch
from ryan_gpt_basics.transformer.multihead_self_attention import MultiHeadSelfAttention
from ryan_gpt_basics.transformer.swiglu import SwiGLU
from ryan_gpt_basics.transformer.rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope_theta=rope_theta, max_seq_len=max_seq_len,
                                         with_rope=with_rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the Transformer block to the input tensor.
        
        x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_model)
        token_positions: torch.Tensor | None Tensor of shape (batch_size, seq_len) indicating the position of each token
        """
        # Apply RMSNorm
        x_norm1 = self.rmsnorm1(x)
        # Apply Multi-Head Self-Attention
        mha_out = self.mha(x_norm1, token_positions)
        # Add residual connection
        x_res1 = x + mha_out

        # Apply RMSNorm
        x_norm2 = self.rmsnorm2(x_res1)
        # Apply SwiGLU Feed-Forward Network
        ff_out = self.ffn(x_norm2)
        # Add residual connection
        y = x_res1 + ff_out

        return y