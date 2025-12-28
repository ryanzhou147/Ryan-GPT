import torch
import torch.nn as nn
from ryan_gpt_basics.transformer.transformer_block import TransformerBlock
from ryan_gpt_basics.transformer.linear import Linear
from ryan_gpt_basics.transformer.embedding import Embedding
from ryan_gpt_basics.transformer.rmsnorm import RMSNorm
from einops import rearrange, einsum

class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        
        # Use context_length as max_seq_len if not provided
        if max_seq_len is None:
            max_seq_len = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, with_rope, rope_theta, max_seq_len, device, dtype)
            for _ in range(num_layers)
        ])
        self.rmsnorm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply the Transformer language model to the input indices.
        
        in_indices: torch.Tensor Input tensor of shape (batch_size, sequence_length)
        """
        batch_size, seq_len = in_indices.shape
        assert seq_len <= self.context_length, f"Input sequence length {seq_len} exceeds context_length {self.context_length}"

        # Get token positions
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1) # Shape: (batch_size, seq_len)
        # Embed input indices
        x = self.token_embeddings(in_indices) # Shape: (batch_size, seq_len, d_model)
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        # Apply final RMSNorm
        x = self.rmsnorm_final(x)
        # Project to vocabulary logits
        logits = self.lm_head(x) # Shape: (batch_size, seq_len, vocab_size)
        return logits