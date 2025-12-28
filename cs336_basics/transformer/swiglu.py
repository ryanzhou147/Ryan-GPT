import torch
import torch.nn as nn
import torch.nn.init as init
from ryan_gpt_basics.transformer.linear import Linear


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # Round to nearest multiple of 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.shape[-1] == self.d_model, f"Expected input last dimension {self.d_model}, got {x.shape[-1]}"

        w1_out = self.w1(x) # Dimensions: (batch_size, seq_len, d_ff)
        silu_out = w1_out * torch.sigmoid(w1_out) 

        w3_out = self.w3(x) # Dimensions: (batch_size, seq_len, d_ff)
        gated = silu_out * w3_out

        return self.w2(gated)


