import torch
import torch.nn as nn
from ryan_gpt_basics.transformer.transformer_block import TransformerBlock
from ryan_gpt_basics.transformer.linear import Linear
from ryan_gpt_basics.transformer.embedding import Embedding
from ryan_gpt_basics.transformer.rmsnorm import RMSNorm

class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None, use_flash: bool = True) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        max_seq_len = max_seq_len or context_length

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, with_rope, rope_theta, max_seq_len, device, dtype, use_flash=use_flash)
            for _ in range(num_layers)
        ])
        self.rmsnorm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.shape
        assert seq_len <= self.context_length

        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(in_indices)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        return self.lm_head(self.rmsnorm_final(x))