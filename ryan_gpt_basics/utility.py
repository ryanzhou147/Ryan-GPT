import torch
import math
import os
import typing
import numpy as np
from collections.abc import Iterable 
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
    assert mask.max() <= 1 and mask.min() >= 0, "Mask tensor must be binary (0s and 1s)"
    assert value.size(-2) == key.size(-2), "Key and Value must have the same sequence length"
    d_k = query.size(-1)
    scores = einsum(query, key, '... i d, ... j d -> ... i j') / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    
    outputs = einsum(attn_weights, value, '... i j, ... j d -> ... i d')
    return outputs

def learning_rate_schedule(current_iteration: int, max_learning_rate: float, minimum_learning_rate: float, warmup_iterations: int, cosine_annealing_iterations: int) -> float:
    if current_iteration < warmup_iterations:
        lr = max_learning_rate * (current_iteration / warmup_iterations)
    elif current_iteration <= cosine_annealing_iterations: 
        progress = (current_iteration - warmup_iterations) / (cosine_annealing_iterations - warmup_iterations)
        lr = minimum_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - minimum_learning_rate)
    else: 
        lr = minimum_learning_rate
    
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            param_norm = torch.linalg.vector_norm(parameter.grad.data, ord=2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data.mul_(scale)


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(x)
    starts = np.random.randint(0, n - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    
    # Single gather from memmap, then convert + transfer in one call
    seq = torch.tensor(x[starts[:, None] + offsets], dtype=torch.long, device=device)
    
    return seq[:, :-1], seq[:, 1:]

# What if the dataset is too big to load into memory? We can use a Unix systemcall named mmap which
# maps a file on disk to virtual memory, and lazily loads the file contents when that memory location is
# accessed. Thus, you can “pretend” you have the entire dataset in memory. Numpy implements this through
# np.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), which
# will return a numpy array-like object that loads the entries on-demand as you access them. When sampling
# from your dataset (i.e., a numpy array) during training, be sure load the dataset in memorymapped mode (via np.memmap or the flag mmap_mode='r' to np.load, depending on how you saved the
# array). Make sure you also specify a dtype that matches the array that you’re loading. It may be helpful
# to explicitly verify that the memory-mapped data looks correct (e.g., doesn’t contain values beyond the
# expected vocabulary size).


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> int:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Decode text from a language model.
    
    Args:
        model: Language model that outputs logits of shape [batch, seq, vocab]
        prompt_ids: Input token IDs of shape [seq_len] or [batch, seq_len]
        max_new_tokens: Maximum number of new tokens to decode
        context_length: Maximum context length the model supports
        eos_token_id: If provided, stop generation when this token is produced
        temperature: Temperature for softmax scaling (lower = more deterministic)
        top_p: Nucleus sampling threshold (if provided, sample from smallest set with cumulative prob >= top_p)
    
    Returns:
        Generated token IDs including the prompt
    """
    # Handle 1D input
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    
    generated = prompt_ids.clone()
    for _ in range(max_new_tokens):
        # Truncate to context length
        input_ids = generated[:, -context_length:]
        
        # Get logits for last position
        logits = model(input_ids)[:, -1, :]  # [batch, vocab]
        
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = softmax(logits, dim=-1)
        
        # Top-p (nucleus) sampling
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            # Find cutoff: smallest set where cumsum >= top_p
            mask = cumsum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Sample from sorted distribution then map back
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, -1, next_token)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop if EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    
    return generated.squeeze(0) if prompt_ids.size(0) == 1 else generated
