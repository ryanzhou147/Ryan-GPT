#!/usr/bin/env python3
"""
GPT Training Script
Supports BPE tokenization and transformer language model training.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gpt.logger import Logger
from gpt.optimizer.adamw import AdamW
from gpt.optimizer.cross_entropy import CrossEntropyLoss
from gpt.transformer.transformer import TransformerLM
from gpt.utility import learning_rate_schedule, save_checkpoint, load_checkpoint


def tokenize(input_path: str, output_dir: str, vocab_size: int = 10000):
    """Train BPE tokenizer on input file and convert to token IDs."""
    from gpt.tokenizer.train_bpe import train_bpe
    from gpt.tokenizer.bpe_tokenizer import BPEProcessor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    special_tokens = ["<|endoftext|>"]

    if not vocab_path.exists():
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        with open(vocab_path, "w") as f:
            json.dump({str(k): v.hex() for k, v in vocab.items()}, f)
        with open(merges_path, "w") as f:
            for left, right in merges:
                f.write(f"{left.hex()} {right.hex()}\n")

    tokenizer = BPEProcessor.from_files(str(vocab_path), str(merges_path), special_tokens)
    
    print(f"Tokenizing {input_path}...")
    out_path = output_dir / (Path(input_path).stem + ".npy")
    
    start = time.time()
    ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            ids.append(token_id)
            if i > 0 and i % 10_000_000 == 0:
                elapsed = time.time() - start
                print(f"  {i:,} tokens ({elapsed:.1f}s, {i/elapsed:,.0f} tok/s)")
    
    ids = np.array(ids, dtype=np.uint16)
    np.save(out_path, ids)
    print(f"Saved {len(ids):,} tokens to {out_path} ({time.time()-start:.1f}s)")


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: torch.device):
    """Sample a random batch from the dataset."""
    n = len(data)
    starts = np.random.randint(0, n - seq_len, size=batch_size)
    offsets = np.arange(seq_len + 1)
    seq = torch.tensor(data[starts[:, None] + offsets], dtype=torch.long, device=device)
    return seq[:, :-1], seq[:, 1:]


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(project=args.project, name=output_dir.name, config=vars(args))

    # Data
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    print(f"Train: {len(train_data):,} tokens")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        with_rope=True,
        rope_theta=args.rope_theta,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iter {start_iter}")

    # Train
    model.train()
    for step in range(start_iter, args.max_steps + 1):
        lr = learning_rate_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = CrossEntropyLoss().forward(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log
        if step % args.log_interval == 0:
            elapsed = logger.elapsed_time()
            iters_done = step - start_iter
            msg = f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s"
            
            if iters_done > 0:
                eta = (args.max_steps - step) * elapsed / iters_done / 60
                msg += f" | ETA {eta:.1f}m"

            metrics = {"loss": loss.item(), "lr": lr}

            if val_data is not None and step % args.eval_interval == 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_steps):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, device)
                    with torch.no_grad():
                        vlogits = model(vx)
                        val_losses.append(CrossEntropyLoss().forward(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)).item())
                model.train()
                val_loss = sum(val_losses) / len(val_losses)
                metrics["val_loss"] = val_loss
                msg += f" | val {val_loss:.4f}"

            logger.log(step, metrics)
            print(msg)

        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, ckpt_dir / f"ckpt_{step}.pt")

    save_checkpoint(model, optimizer, args.max_steps, ckpt_dir / "ckpt_final.pt")
    logger.finish()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="GPT Trainer")
    subparsers = parser.add_subparsers(dest="cmd")

    # Tokenize
    tok = subparsers.add_parser("tokenize")
    tok.add_argument("--input", required=True)
    tok.add_argument("--output_dir", required=True)
    tok.add_argument("--vocab_size", type=int, default=10000)

    # Train
    tr = subparsers.add_parser("train")
    tr.add_argument("--train_data", required=True)
    tr.add_argument("--val_data", default=None)
    tr.add_argument("--output_dir", required=True)
    tr.add_argument("--project", default="gpt")
    # Model
    tr.add_argument("--vocab_size", type=int, required=True)
    tr.add_argument("--context_length", type=int, default=256)
    tr.add_argument("--num_layers", type=int, default=4)
    tr.add_argument("--d_model", type=int, default=512)
    tr.add_argument("--num_heads", type=int, default=16)
    tr.add_argument("--d_ff", type=int, default=1344)
    tr.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    tr.add_argument("--lr", type=float, default=6e-4)
    tr.add_argument("--min_lr", type=float, default=6e-5)
    tr.add_argument("--beta1", type=float, default=0.9)
    tr.add_argument("--beta2", type=float, default=0.99)
    tr.add_argument("--eps", type=float, default=1e-8)
    tr.add_argument("--weight_decay", type=float, default=0.1)
    tr.add_argument("--max_grad_norm", type=float, default=1.0)
    # Schedule
    tr.add_argument("--warmup_steps", type=int, default=1000)
    tr.add_argument("--max_steps", type=int, default=20000)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--seed", type=int, default=42)
    # Logging
    tr.add_argument("--log_interval", type=int, default=100)
    tr.add_argument("--eval_interval", type=int, default=500)
    tr.add_argument("--eval_steps", type=int, default=20)
    tr.add_argument("--save_interval", type=int, default=2000)
    tr.add_argument("--resume", default=None)

    args = parser.parse_args()
    
    if args.cmd == "tokenize":
        tokenize(args.input, args.output_dir, args.vocab_size)
    elif args.cmd == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
