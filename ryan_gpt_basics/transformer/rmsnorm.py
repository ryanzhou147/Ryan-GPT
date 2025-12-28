import torch
import torch.nn as nn
import torch.nn.init as init

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        """Construct an RMSNorm module. This function should accept the following parameters:
        d_model: int Dimension of the input
        eps: float = 1e-5 Small constant for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input tensor of shape (batch_size, seq_len, dmodel) and return
        normalized tensor of the same shape."""

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        x_norm = x_norm.to(in_dtype)

        return self.weight * x_norm