import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct a RotaryPositionalEmbedding module. This function should accept the following parameters:
        theta: float theta value for RoPE
        d_k: int Dimension of query key and vectors
        max_seq_len: int Maximum sequence length from input
        device: torch.device | None = None Device to store the parameters on
        """
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(0, max_seq_len, device=device).float()
        sinusoid_inp = einsum(positions, inv_freq, 'i, j -> i j')
        self.register_buffer('cos_cached', torch.cos(sinusoid_inp), persistent=False)
        self.register_buffer('sin_cached', torch.sin(sinusoid_inp), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Positional Embedding to the input tensor.
        
        x: torch.Tensor Input tensor of shape (batch_size, seq_len, d_k)
        token_positions: torch.Tensor Tensor of shape (batch_size, seq_len) indicating the position of each token
        """
        batch_size, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected input last dimension {self.d_k}, got {d_k}"
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        cos = self.cos_cached[token_positions]  # Shape: (batch_size, seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # Shape: (batch_size, seq_len, d_k/2)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos

        return x_rotated