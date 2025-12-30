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
    assert value.size(-2) == key.size(-2), "Key and Value must have the same sequence length"

    # Fast-path: use flash-attention functions from ryan_gpt_systems when available.
    try:
        from ryan_gpt_systems.flash_attention import flash_attention_triton, flash_attention_pytorch
        flash_available = True
        _flash_triton = flash_attention_triton
        _flash_pytorch = flash_attention_pytorch
    except Exception:
        flash_available = False
        _flash_triton = None
        _flash_pytorch = None

    # If inputs include head dimension (batch, heads, seq, d), reshape and use flash attention
    if flash_available and query.ndim == 4:
        b, h, s, d = query.shape
        # detect simple causal mask pattern
        is_causal = False
        if mask is not None and mask.ndim >= 2:
            try:
                if mask.shape[-2:] == (s, s):
                    tril = torch.tril(torch.ones((s, s), device=mask.device, dtype=mask.dtype))
                    # compare first batch/head mask slice if possible
                    mask_slice = mask.reshape(-1, s, s)[0]
                    if torch.equal(mask_slice.to(tril.dtype), tril):
                        is_causal = True
            except Exception:
                is_causal = False

        try:
            Q = query.reshape(b * h, s, d)
            K = key.reshape(b * h, s, d)
            V = value.reshape(b * h, s, d)
            # Prefer PyTorch flash implementation first
            if _flash_pytorch is not None:
                out_flat = _flash_pytorch(Q, K, V, is_causal)
            elif _flash_triton is not None:
                out_flat = _flash_triton(Q, K, V, is_causal)
            else:
                out_flat = None

            if out_flat is not None:
                out = out_flat.reshape(b, h, s, d)
                return out
        except Exception:
            # If flash path fails, fall back to Python implementation below
            pass

    # Fallback generic implementation (works for both with/without head dim)
    assert mask is None or (mask.max() <= 1 and mask.min() >= 0), "Mask tensor must be binary (0s and 1s)"
    d_k = query.size(-1)
    scores = einsum(query, key, '... i d, ... j d -> ... i j') / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=query.device))
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

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    lr_config: dict = None) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'lr_config': lr_config,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, dict]:
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_config = checkpoint.get('lr_config', None)
    return checkpoint['iteration'], lr_config


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    banned_token_ids: list[int] | torch.Tensor | None = None,
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

        # Mask any banned tokens by setting their logits very low
        if banned_token_ids is not None:
            try:
                if isinstance(banned_token_ids, torch.Tensor):
                    banned = banned_token_ids.to(logits.device)
                else:
                    banned = torch.tensor(banned_token_ids, device=logits.device, dtype=torch.long)
                # scatter to set logits[:, banned] = -1e9
                logits.scatter_(1, banned.unsqueeze(0).expand(logits.size(0), -1), -1e9)
            except Exception:
                for bid in (banned_token_ids if not isinstance(banned_token_ids, torch.Tensor) else banned_token_ids.tolist()):
                    logits[:, bid] = -1e9

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
