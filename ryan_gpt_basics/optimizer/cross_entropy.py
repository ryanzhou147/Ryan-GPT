import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return nll_loss.mean()

    def perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.forward(logits, targets))