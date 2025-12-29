import argparse
import json
import time
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import torch

from ryan_gpt_basics.logger import Logger
from ryan_gpt_basics.optimizer.adamw import AdamW
from ryan_gpt_basics.optimizer.cross_entropy import CrossEntropyLoss
from ryan_gpt_basics.transformer.transformer import TransformerLM
from ryan_gpt_basics.utility import learning_rate_schedule, save_checkpoint, load_checkpoint


# =============================================================================
# Tokenize
# =============================================================================

def tokenize(input_path: str, output_dir: str, vocab_size: int = 10000):
    """Train BPE tokenizer on input file and convert to token IDs."""
    from ryan_gpt_basics.tokenizer.train_bpe import train_bpe
    from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    special_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>"]

    if not vocab_path.exists():
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        with open(vocab_path, "w") as f:
            json.dump({str(k): v.hex() for k, v in vocab.items()}, f)
        with open(merges_path, "w") as f:
            for left, right in merges:
                f.write(f"{left.hex()} {right.hex()}\n")
    else:
        print(f"[CACHED] Tokenizer already exists at {vocab_path}")

    tokenizer = BPEProcessor.from_files(str(vocab_path), str(merges_path), special_tokens)
    
    print(f"Tokenizing {input_path}...")
    out_path = output_dir / (Path(input_path).stem + ".npy")
    
    if out_path.exists():
        print(f"[CACHED] Tokens already exist at {out_path}")
        return
    
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


def tokenize_file(input_path: str, output_path: str, tokenizer_dir: str):
    """Tokenize a single file using an existing tokenizer."""
    from ryan_gpt_basics.tokenizer.bpe_tokenizer import BPEProcessor
    
    vocab_path = Path(tokenizer_dir) / "vocab.json"
    merges_path = Path(tokenizer_dir) / "merges.txt"
    special_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
    
    tokenizer = BPEProcessor.from_files(str(vocab_path), str(merges_path), special_tokens)
    
    print(f"Tokenizing {input_path}...")
    start = time.time()
    
    ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            ids.append(token_id)
            if i > 0 and i % 1_000_000 == 0:
                elapsed = time.time() - start
                print(f"  {i:,} tokens ({elapsed:.1f}s)")
    
    ids = np.array(ids, dtype=np.uint16)
    np.save(output_path, ids)
    print(f"Saved {len(ids):,} tokens to {output_path} ({time.time()-start:.1f}s)")


# =============================================================================
# Data Loading
# =============================================================================

def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: torch.device):
    """Sample a random batch from the dataset."""
    n = len(data)
    starts = np.random.randint(0, n - seq_len, size=batch_size)
    offsets = np.arange(seq_len + 1)
    seq = torch.tensor(data[starts[:, None] + offsets], dtype=torch.long, device=device)
    return seq[:, :-1], seq[:, 1:]


# =============================================================================
# Training
# =============================================================================

def train(args):
    """Main pretraining loop."""
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
    if val_data is not None:
        print(f"Val: {len(val_data):,} tokens")

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
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

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
    loss_fn = CrossEntropyLoss()
    
    for step in range(start_iter, args.max_steps + 1):
        lr = learning_rate_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn.forward(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

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

            if val_data is not None and step % args.eval_interval == 0 and step > 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_steps):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, device)
                    with torch.no_grad():
                        vlogits = model(vx)
                        val_losses.append(loss_fn.forward(
                            vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)
                        ).item())
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
    print("Pretraining done.")


# =============================================================================
# Fine-tuning
# =============================================================================

def finetune(args):
    """Fine-tuning loop - loads pretrained checkpoint."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(project=args.project, name=output_dir.name, config=vars(args))

    # Data
    # Support multiple training sources for on-the-fly mixing: comma-separated paths
    train_paths = [p.strip() for p in args.train_data.split(',')]
    train_datasets = [np.load(p, mmap_mode='r') for p in train_paths]
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    total_tokens = sum(len(d) for d in train_datasets)
    print(f"Train sources: {train_paths}")
    print(f"Total train tokens: {total_tokens:,} tokens")
    if val_data is not None:
        print(f"Val: {len(val_data):,} tokens")

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
    
    # Load pretrained weights
    print(f"Loading pretrained checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Fresh optimizer for fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Train
    model.train()
    loss_fn = CrossEntropyLoss()
    
    # Prepare mixing probabilities for multiple datasets (if provided)
    if len(train_datasets) > 1:
        if args.mix:
            try:
                probs = [float(x) for x in args.mix.split(',')]
                assert len(probs) == len(train_datasets)
                probs = np.array(probs, dtype=float)
                probs = probs / probs.sum()
            except Exception:
                print("Invalid --mix format; falling back to uniform mixing.")
                probs = np.ones(len(train_datasets), dtype=float) / len(train_datasets)
        else:
            probs = np.ones(len(train_datasets), dtype=float) / len(train_datasets)
    else:
        probs = np.array([1.0], dtype=float)

    for step in range(args.max_steps + 1):
        lr = learning_rate_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Sample which dataset to draw this batch from according to probs
        ds_idx = int(np.random.choice(len(train_datasets), p=probs))
        x, y = get_batch(train_datasets[ds_idx], args.batch_size, args.context_length, device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn.forward(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log
        if step % args.log_interval == 0:
            elapsed = logger.elapsed_time()
            msg = f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s"
            
            if step > 0:
                eta = (args.max_steps - step) * elapsed / step / 60
                msg += f" | ETA {eta:.1f}m"

            metrics = {"loss": loss.item(), "lr": lr}

            if val_data is not None and step % args.eval_interval == 0 and step > 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_steps):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, device)
                    with torch.no_grad():
                        vlogits = model(vx)
                        val_losses.append(loss_fn.forward(
                            vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)
                        ).item())
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
    print("Fine-tuning done.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT Trainer")
    subparsers = parser.add_subparsers(dest="cmd")

    # -------------------------------------------------------------------------
    # Tokenize (train tokenizer + tokenize file)
    # -------------------------------------------------------------------------
    tok = subparsers.add_parser("tokenize", help="Train BPE tokenizer and tokenize data")
    tok.add_argument("--input", required=True, help="Input text file")
    tok.add_argument("--output_dir", required=True, help="Output directory for vocab and tokens")
    tok.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")

    # -------------------------------------------------------------------------
    # Tokenize file (using existing tokenizer)
    # -------------------------------------------------------------------------
    tokfile = subparsers.add_parser("tokenize_file", help="Tokenize a file using existing tokenizer")
    tokfile.add_argument("--input", required=True, help="Input text file")
    tokfile.add_argument("--output", required=True, help="Output .npy file")
    tokfile.add_argument("--tokenizer_dir", required=True, help="Directory with vocab.json and merges.txt")

    # -------------------------------------------------------------------------
    # Train (Pretrain)
    # -------------------------------------------------------------------------
    tr = subparsers.add_parser("train", help="Pretrain the model")
    tr.add_argument("--train_data", required=True, help="Path to training .npy file")
    tr.add_argument("--val_data", default=None, help="Path to validation .npy file")
    tr.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    tr.add_argument("--project", default="gpt-pretrain", help="Project name for logging")
    # Model
    tr.add_argument("--vocab_size", type=int, required=True)
    tr.add_argument("--context_length", type=int, default=256)
    tr.add_argument("--num_layers", type=int, default=4)
    tr.add_argument("--d_model", type=int, default=512)
    tr.add_argument("--num_heads", type=int, default=8)
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
    tr.add_argument("--resume", default=None, help="Resume from checkpoint")

    # -------------------------------------------------------------------------
    # Fine-tune
    # -------------------------------------------------------------------------
    ft = subparsers.add_parser("finetune", help="Fine-tune a pretrained model")
    ft.add_argument("--train_data", required=True, help="Path(s) to training .npy file(s). For multiple files, separate with commas")
    ft.add_argument("--val_data", default=None, help="Path to validation .npy file")
    ft.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    ft.add_argument("--checkpoint", required=True, help="Pretrained checkpoint to load")
    ft.add_argument("--project", default="gpt-finetune", help="Project name for logging")
    # Model (must match pretrained)
    ft.add_argument("--vocab_size", type=int, required=True)
    ft.add_argument("--context_length", type=int, default=256)
    ft.add_argument("--num_layers", type=int, default=4)
    ft.add_argument("--d_model", type=int, default=512)
    ft.add_argument("--num_heads", type=int, default=8)
    ft.add_argument("--d_ff", type=int, default=1344)
    ft.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer (lower LR for fine-tuning)
    ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--min_lr", type=float, default=1e-5)
    ft.add_argument("--beta1", type=float, default=0.9)
    ft.add_argument("--beta2", type=float, default=0.99)
    ft.add_argument("--eps", type=float, default=1e-8)
    ft.add_argument("--weight_decay", type=float, default=0.1)
    ft.add_argument("--max_grad_norm", type=float, default=1.0)
    # Schedule (shorter for fine-tuning)
    ft.add_argument("--warmup_steps", type=int, default=100)
    ft.add_argument("--max_steps", type=int, default=5000)
    ft.add_argument("--batch_size", type=int, default=32)
    ft.add_argument("--seed", type=int, default=42)
    # Logging
    ft.add_argument("--log_interval", type=int, default=50)
    ft.add_argument("--eval_interval", type=int, default=250)
    ft.add_argument("--eval_steps", type=int, default=20)
    ft.add_argument("--save_interval", type=int, default=1000)
    ft.add_argument("--mix", default=None, help="Comma-separated mixing proportions for multiple train_data files, e.g. '0.7,0.3' for two sources")

    # -------------------------------------------------------------------------
    # Parse and dispatch
    # -------------------------------------------------------------------------
    args = parser.parse_args()
    
    if args.cmd == "tokenize":
        tokenize(args.input, args.output_dir, args.vocab_size)
    elif args.cmd == "tokenize_file":
        tokenize_file(args.input, args.output, args.tokenizer_dir)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "finetune":
        finetune(args)
    elif args.cmd == "flash_test":
        flash_test(args)
    # Generation functionality moved to `generate.py`.
    else:
        parser.print_help()


def flash_test(args):
    """Run small forward/backward tests for flash-attention (model-level and direct functions)."""
    import torch
    from ryan_gpt_basics.transformer.transformer import TransformerLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running flash-attention test on device: {device}")

    # Try importing direct flash functions
    try:
        from ryan_gpt_systems.flash_attention import flash_attention_triton, flash_attention_pytorch
        triton_ok = True
        print("Found flash_attention_triton and flash_attention_pytorch")
    except Exception as e:
        triton_ok = False
        flash_attention_triton = None
        flash_attention_pytorch = None
        print("Flash attention functions not available:", e)

    # Build a tiny model with use_flash=True
    vocab_size = 100
    seq_len = args.seq_len
    batch = args.batch_size
    num_heads = args.num_heads
    d_model = args.d_model
    num_layers = max(1, args.num_layers)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=max(d_model * 2, 4),
        with_rope=False,
        use_flash=True,
    ).to(device)

    model.train()

    # Random input indices
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.long)
    logits = model(input_ids)
    loss = logits.view(-1, logits.size(-1)).float().softmax(-1).mean()
    loss.backward()
    print("Model forward+backward completed (use_flash=True)")

    # Direct flash function tests (if available)
    if triton_ok:
        head_dim = d_model // num_heads
        b = batch * num_heads
        Q = torch.randn(b, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        K = torch.randn(b, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        V = torch.randn(b, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)

        # Triton implementation
        try:
            out_tr = flash_attention_triton(Q, K, V, is_causal=False)
            out_tr.sum().backward()
            print("Triton flash attention forward+backward succeeded")
        except Exception as e:
            print("Triton flash attention failed:", e)

        # PyTorch reference implementation
        try:
            out_py = flash_attention_pytorch(Q, K, V, is_causal=False)
            out_py.sum().backward()
            print("PyTorch flash attention forward+backward succeeded")
        except Exception as e:
            print("PyTorch flash attention failed:", e)

    print("Flash-attention test finished.")


if __name__ == "__main__":
    main()