import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        # Subtract max for numerical stability
        x_max = torch.max(logits, dim=-1, keepdim=True).values
        logits = logits - x_max
        # Compute log probabilities
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Gather the log probabilities of the target classes
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return nll_loss.mean()

    def perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the perplexity from the cross-entropy loss."""
        ce_loss = self.forward(logits, targets)
        perplexity = torch.exp(ce_loss)
        return perplexity