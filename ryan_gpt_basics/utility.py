import torch
import math
import os
import typing
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
    min_new_tokens: int = 0,
) -> torch.Tensor:
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    
    generated = prompt_ids.clone()
    new_tokens = []
    
    for i in range(max_new_tokens):
        input_ids = generated[:, -context_length:]
        logits = model(input_ids)[:, -1, :]
        
        if temperature != 1.0:
            logits = logits / temperature

        if banned_token_ids is not None:
            try:
                if isinstance(banned_token_ids, torch.Tensor):
                    banned = banned_token_ids.to(logits.device)
                else:
                    banned = torch.tensor(banned_token_ids, device=logits.device, dtype=torch.long)
                logits.scatter_(1, banned.unsqueeze(0).expand(logits.size(0), -1), -1e9)
            except Exception:
                for bid in (banned_token_ids if not isinstance(banned_token_ids, torch.Tensor) else banned_token_ids.tolist()):
                    logits[:, bid] = -1e9

        if i < min_new_tokens and eos_token_id is not None:
            logits[:, eos_token_id] = -1e9

        probs = softmax(logits, dim=-1)
        
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, -1, next_token)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
        new_tokens.append(next_token.item())
        
        # Stop on EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        
        # Repetition detection: stop if same n-gram repeats 3 times
        if len(new_tokens) >= 9:
            for n in [3, 4, 5, 6, 7, 8]:
                if len(new_tokens) >= n * 3:
                    last = tuple(new_tokens[-n:])
                    prev = tuple(new_tokens[-2*n:-n])
                    prev_prev = tuple(new_tokens[-3*n:-2*n])
                    if last == prev == prev_prev:
                        # Remove the repeated part
                        generated = generated[:, :-(2*n)]
                        return generated.squeeze(0) if prompt_ids.size(0) == 1 else generated
    
    return generated.squeeze(0) if prompt_ids.size(0) == 1 else generated