#!/usr/bin/env python3
"""Generate text from a trained GPT model."""

import argparse
import torch

from ryan_gpt_basics.transformer.transformer import TransformerLM
from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor
from ryan_gpt_basics.utility import decode as generate

PRESETS = {
    "tinystories": {
        "vocab": "data/tokenized/vocab.json",
        "merges": "data/tokenized/merges.txt",
        "checkpoint": "runs/finetune_test/checkpoints/ckpt.pt",
        "prompt": "I am a scientist",
    },
    "owt": {
        "vocab": "data/owt/vocab.json",
        "merges": "data/owt/merges.txt",
        "checkpoint": "data/owt/main_experiment/checkpoints/checkpoint_final.pt",
        "prompt": "The scientists discovered that",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate text from GPT")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="tinystories")
    parser.add_argument("--checkpoint", help="Override checkpoint path")
    parser.add_argument("--prompt", help="Override prompt")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    checkpoint_path = args.checkpoint or preset["checkpoint"]
    prompt = args.prompt or preset["prompt"]

    # Load tokenizer (include common special tokens used in dialogue)
    tokenizer = BPEProcessor.from_files(preset["vocab"], preset["merges"], ["<|endoftext|>", "<|user|>", "<|assistant|>"])

    # Load model checkpoint first and infer architecture when possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer vocab_size and d_model from embedding or lm_head if present
    vocab_size = None
    d_model = None
    for k, v in state_dict.items():
        if 'token_embeddings' in k and v.ndim == 2:
            vocab_size, d_model = v.shape[0], v.shape[1]
            break
    if vocab_size is None or d_model is None:
        for k, v in state_dict.items():
            if 'lm_head' in k and v.ndim == 2:
                vocab_size, d_model = v.shape[0], v.shape[1]
                break

    # Infer num_layers from transformer block keys
    num_layers = 0
    for k in state_dict.keys():
        if k.startswith('transformer_blocks.'):
            try:
                idx = int(k.split('.')[1])
                num_layers = max(num_layers, idx + 1)
            except Exception:
                pass

    # Infer num_heads from mha weight shapes if possible
    num_heads = None
    for k, v in state_dict.items():
        if k.endswith('mha.w_q') or k.endswith('mha.w_q.weight') or k.endswith('mha.w_q.W'):
            out_dim = v.shape[0]
            if d_model is not None:
                for h in [1,2,4,8,16,32,64]:
                    dk = d_model // h
                    if dk > 0 and h * dk == out_dim:
                        num_heads = h
                        break
            break
    if num_heads is None:
        num_heads = 16

    # Infer d_ff from ffn weights
    d_ff = None
    for k, v in state_dict.items():
        if 'ffn.w1' in k or 'ffn.w1.W' in k or 'ffn.w1.weight' in k:
            # assume shape (d_ff, d_model)
            if v.ndim == 2:
                d_ff = v.shape[0]
                break
    if d_ff is None:
        d_ff = max(d_model * 2 if d_model is not None else 512, 512)

    # Fallbacks
    if vocab_size is None:
        vocab_size = 4000
    if d_model is None:
        d_model = 256
    if num_layers == 0:
        num_layers = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=256,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        with_rope=True,
        rope_theta=10000.0,
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    eos_id = None
    if hasattr(tokenizer, 'special_tokens_to_id'):
        eos_id = tokenizer.special_tokens_to_id.get(b"<|endoftext|>")
    else:
        eos_id = tokenizer.vocab.get(b"<|endoftext|>")
    if eos_id is None:
        print("Warning: <|endoftext|> not found in tokenizer")
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("-" * 60)

    # Build banned token list: mask special markers during generation
    banned_tokens = []
    specials = [b"<|endoftext|>", b"<|user|>", b"<|assistant|>"]
    if hasattr(tokenizer, 'special_tokens_to_id'):
        for t in specials:
            tid = tokenizer.special_tokens_to_id.get(t)
            if tid is not None:
                banned_tokens.append(tid)
    else:
        for t in specials:
            tid = tokenizer.vocab.get(t)
            if tid is not None:
                banned_tokens.append(tid)
    # debug: print(f"Banned token ids: {banned_tokens}")

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        context_length=256,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
        banned_token_ids=banned_tokens if banned_tokens else None,
    )

    text = tokenizer.decode(generated_ids.tolist())
    print(f"{text}\n")
    print(f"[{len(generated_ids)} tokens]")


if __name__ == "__main__":
    main()
