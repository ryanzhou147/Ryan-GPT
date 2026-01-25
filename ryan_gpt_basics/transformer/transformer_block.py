import torch
from ryan_gpt_basics.transformer.multihead_self_attention import MultiHeadSelfAttention
from ryan_gpt_basics.transformer.swiglu import SwiGLU
from ryan_gpt_basics.transformer.rmsnorm import RMSNorm

class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None, use_flash: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, rope_theta=rope_theta, max_seq_len=max_seq_len,
                         with_rope=with_rope, device=device, dtype=dtype, use_flash=use_flash)
        self.ffn = SwiGLU(d_model, d_ff)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mha(self.rmsnorm1(x), token_positions)
        x = x + self.ffn(self.rmsnorm2(x))
        return x