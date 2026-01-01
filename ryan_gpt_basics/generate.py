#!/usr/bin/env python3
"""Generate text from a trained GPT model."""

import argparse
import torch
import regex as re

from ryan_gpt_basics.transformer.transformer import TransformerLM
from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor
from ryan_gpt_basics.utility import decode

PRESETS = {
    "wikipedia": {
        "vocab": "data/tokenized/vocab.json",
        "merges": "data/tokenized/merges.txt",
        "checkpoint": "runs/pretrain_less_parameters/checkpoints/ckpt_final.pt",
    },
    "chat": {
        "vocab": "data/tokenized/vocab.json",
        "merges": "data/tokenized/merges.txt",
        "checkpoint": "runs/finetune_v2/checkpoints/ckpt_final.pt",
    },
}

def clean_text(text: str) -> str:
    """Clean generated text by fixing spacing issues."""
    # Remove any special tokens that leaked through
    text = text.replace('<|endoftext|>', '')
    text = text.replace('<|user|>', '')
    text = text.replace('<|assistant|>', '')
    
    # Normalize curly apostrophes to straight
    text = text.replace('\u2019', "'")  # right single quote '
    text = text.replace('\u2018', "'")  # left single quote '
    
    # Fix contractions: "I ' m" -> "I'm"
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)
    text = re.sub(r"(\w) '(\w)", r"\1'\2", text)
    text = re.sub(r"(\w)' (\w)", r"\1'\2", text)
    
    # Fix spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    text = text.strip()
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Capitalize after periods
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    # Ensure response ends with punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

def extract_response(text: str) -> str:
    """Extract assistant response from chat-formatted text."""
    if '<|assistant|>' in text:
        text = text.split('<|assistant|>')[-1]
    
    for marker in ['<|endoftext|>', '<|user|>', '<|assistant|>']:
        text = text.replace(marker, '')
    
    return clean_text(text)


def load_model(checkpoint_path: str, device: str):
    """Load model and infer architecture from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer architecture
    vocab_size, d_model = None, None
    for k, v in state_dict.items():
        if 'token_embeddings' in k and v.ndim == 2:
            vocab_size, d_model = v.shape[0], v.shape[1]
            break
    
    num_layers = 0
    for k in state_dict.keys():
        if k.startswith('transformer_blocks.'):
            try:
                idx = int(k.split('.')[1])
                num_layers = max(num_layers, idx + 1)
            except:
                pass
    
    d_ff = None
    for k, v in state_dict.items():
        if 'ffn.w1' in k and v.ndim == 2:
            d_ff = v.shape[0]
            break
    
    # Defaults
    vocab_size = vocab_size or 10000
    d_model = d_model or 512
    num_layers = num_layers or 4
    d_ff = d_ff or 1536
    num_heads = d_model // 64
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=512,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        with_rope=True,
        rope_theta=10000.0,
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_tokenizer(vocab_path: str, merges_path: str):
    """Load tokenizer with special tokens."""
    return BPEProcessor.from_files(
        vocab_path, merges_path,
        ['<|endoftext|>', '<|user|>', '<|assistant|>']
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.5,
    top_p: float = 0.9,
    device: str = 'cuda',
) -> str:
    """Generate text from a prompt."""
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    eos_id = tokenizer.encode('<|endoftext|>')[0]
    
    generated_ids = decode(
        model, prompt_ids,
        max_new_tokens=max_tokens,
        context_length=512,
        eos_token_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        banned_token_ids=None,
    )
    
    text = tokenizer.decode(generated_ids.tolist())
    return text
def generate_response(
    model,
    tokenizer,
    user_input: str,
    max_tokens: int = 150,
    temperature: float = 0.6,
    top_p: float = 0.9,
    min_tokens: int = 10,
    device: str = 'cuda',
    chat_mode: bool = True,
) -> str:
    """Generate a response to user input."""
    
    if chat_mode:
        prompt = f'<|user|>\n{user_input}\n<|assistant|>\n'
    else:
        prompt = user_input
    
    # Get special token IDs
    eos_id = tokenizer.encode('<|endoftext|>')[0]
    user_id = tokenizer.encode('<|user|>')[0]
    
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    
    generated = []
    input_ids = prompt_ids.clone()
    
    with torch.no_grad():
        for i in range(max_tokens):
            logits = model(input_ids.unsqueeze(0))[:, -1, :]
            
            # Ban EOS and <|user|> for first min_tokens
            if i < min_tokens:
                logits[0, eos_id] = float('-inf')
                logits[0, user_id] = float('-inf')
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            
            # Stop if EOS or <|user|> (after min_tokens)
            if next_id == eos_id or next_id == user_id:
                break
            
            generated.append(next_id)
            
            # Stop on repetition (same token 5+ times in a row)
            if len(generated) >= 5 and len(set(generated[-5:])) == 1:
                generated = generated[:-4]  # Remove repeated tokens
                break
            
            input_ids = torch.cat([input_ids, torch.tensor([next_id], device=device)])
            
            # Truncate context if too long
            if len(input_ids) > 512:
                input_ids = input_ids[-512:]
    
    response_text = tokenizer.decode(generated)
    
    # Clean up garbage
    response_text = response_text.replace('�', '')
    response_text = re.sub(r'\.{3,}', '.', response_text)  # "....." -> "."
    response_text = re.sub(r'(.)\1{4,}', r'\1', response_text)  # "aaaaa" -> "a"
    
    return clean_text(response_text)

def main():
    parser = argparse.ArgumentParser(description="Generate text from GPT")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="wikipedia")
    parser.add_argument("--checkpoint", help="Override checkpoint path")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--chat_format", action="store_true", help="Treat as chat and extract response")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    checkpoint_path = args.checkpoint or preset["checkpoint"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(preset["vocab"], preset["merges"])

    text = generate_text(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    
    if args.chat_format or '<|assistant|>' in args.prompt:
        text = extract_response(text)
    else:
        text = clean_text(text)
    
    print(text)


if __name__ == "__main__":
    main()