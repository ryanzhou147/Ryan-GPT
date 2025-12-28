#!/usr/bin/env python3
"""Generate text from a trained GPT model."""

import argparse
import torch

from ryan_gpt_basics.transformer.transformer import TransformerLM
from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor
from ryan_gpt_basics.utility import decode as generate

PRESETS = {
    "tinystories": {
        "vocab": "data/tinystories/vocab.json",
        "merges": "data/tinystories/merges.txt",
        "checkpoint": "data/tinystories/bs_64/checkpoints/checkpoint_final.pt",
        "prompt": "Once upon a time",
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

    # Load tokenizer
    tokenizer = BPEProcessor.from_files(preset["vocab"], preset["merges"], ["<|endoftext|>"])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        with_rope=True,
        rope_theta=10000.0,
    ).to(device)

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    eos_id = tokenizer.vocab.get(b"<|endoftext|>")
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("-" * 60)

    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        context_length=256,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    text = tokenizer.decode(generated_ids.tolist())
    print(f"{text}\n")
    print(f"[{len(generated_ids)} tokens]")


if __name__ == "__main__":
    main()
