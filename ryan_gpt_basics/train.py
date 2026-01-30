import argparse
import json
import os
import time
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import torch
import torch.distributed as dist

from ryan_gpt_basics.logger import Logger
from ryan_gpt_basics.optimizer.adamw import AdamW
from ryan_gpt_basics.optimizer.cross_entropy import CrossEntropyLoss
from ryan_gpt_basics.transformer.transformer import TransformerLM
from ryan_gpt_basics.utility import learning_rate_schedule, save_checkpoint, load_checkpoint

def setup_distributed(backend: str = "auto"):

    if "RANK" not in os.environ:
        return {"rank": 0, "world_size": 1, "local_rank": 0, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    num_gpus = torch.cuda.device_count()
    
    # Auto-detect backend
    if backend == "auto":
        # Use gloo if not enough GPUs (for CPU-based simulation)
        backend = "nccl" if local_rank < num_gpus and num_gpus > 0 else "gloo"
    
    if backend == "nccl":
        if local_rank >= num_gpus:
            raise RuntimeError(
                f"Local rank {local_rank} requested but only {num_gpus} GPU(s) available. "
                f"Use --nproc_per_node={num_gpus} or fewer, or use --backend=gloo for CPU testing."
            )
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # gloo backend for CPU-based distributed training (testing)
        dist.init_process_group(backend="gloo")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"Distributed: {world_size} processes ({backend} backend)")
    
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": device}


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce_mean(tensor):
    """Average tensor across all processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def get_distributed_batch(data, batch_size, seq_len, device, rank=0, world_size=1, rng=None):
    """Get a batch with proper sharding across ranks."""
    if rng is None:
        rng = np.random
    
    # Each rank samples from a different portion of the data
    n = len(data)
    shard_size = n // world_size
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size if rank < world_size - 1 else n
    
    # Sample from this rank's shard
    shard_n = end_idx - start_idx - seq_len
    starts = rng.integers(0, shard_n, size=batch_size) + start_idx
    offsets = np.arange(seq_len + 1)
    seq = torch.tensor(data[starts[:, None] + offsets], dtype=torch.long, device=device)
    return seq[:, :-1], seq[:, 1:]

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


def train(args):
    """Main pretraining loop with optional distributed training."""
    # Setup distributed if using torchrun
    backend = getattr(args, 'backend', 'auto')
    dist_info = setup_distributed(backend=backend)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]
    
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    
    if is_main_process():
        print(f"Device: {device}" + (f" (rank {rank}/{world_size})" if world_size > 1 else ""))

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create dirs
    if world_size > 1:
        dist.barrier()

    logger = Logger(project=args.project, name=output_dir.name, config=vars(args)) if is_main_process() else None

    # Data
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    if is_main_process():
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
    if is_main_process():
        print(f"Parameters: {num_params:,}")
    
    # Wrap model for distributed training
    use_manual_grad_sync = False  # For custom DDP implementations
    if world_size > 1:
        # Choose strategy based on args
        strategy = getattr(args, 'strategy', 'ddp')
        backend_used = getattr(args, 'backend', 'auto')
        
        # Gloo backend use manual gradient sync instead of DDP
        if backend_used == 'gloo':
            use_manual_grad_sync = True
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if is_main_process():
                print("Using: Manual gradient sync (gloo)")
        elif strategy == 'ddp':
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[dist_info["local_rank"]])
            if is_main_process():
                print("Using: DistributedDataParallel (DDP)")
        elif strategy == 'ddp_bucketed':
            from ryan_gpt_systems.ddp_bucket import DDPBucketed
            model = DDPBucketed(model, bucket_size_mb=getattr(args, 'bucket_size_mb', 25.0))
            if is_main_process():
                print("Using: DDPBucketed")
        elif strategy == 'ddp_flat':
            from ryan_gpt_systems.ddp_flat import DDPIndividualParameters
            model = DDPIndividualParameters(model)
            if is_main_process():
                print("Using: DDPIndividualParameters")
        elif strategy == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
            mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
            model = FSDP(model, mixed_precision=mp_policy, device_id=device)
            if is_main_process():
                print("Using: FSDP (Fully Sharded Data Parallel)")
    
    # Gradient accumulation info
    grad_accum = args.gradient_accumulation_steps
    effective_batch = args.batch_size * grad_accum * world_size
    if is_main_process():
        print(f"Batch size: {args.batch_size} x {grad_accum} accum x {world_size} GPUs = {effective_batch} effective")

    # Optimizer - use sharded optimizer if requested
    model_params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
    
    if world_size > 1 and getattr(args, 'strategy', 'ddp') == 'zero':
        from ryan_gpt_systems.optimizer_state_sharding import ShardedOptimizer
        optimizer = ShardedOptimizer(model_params, AdamW, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        if is_main_process():
            print("Using: ShardedOptimizer (ZeRO-style)")
    else:
        optimizer = AdamW(model_params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    # Resume handling
    start_iter = 0
    lr_config = None
    
    if args.resume:
        start_iter, lr_config = load_checkpoint(args.resume, model.module if hasattr(model, 'module') else model, optimizer)
        if is_main_process():
            print(f"Resumed from iter {start_iter}")
        torch.manual_seed(args.seed + start_iter + rank)
        np.random.seed(args.seed + start_iter + rank)
        rng = np.random.default_rng(args.seed + start_iter + rank)
        if lr_config and is_main_process():
            print(f"Restored LR config: max_lr={lr_config['max_lr']:.2e}, min_lr={lr_config['min_lr']:.2e}, max_steps={lr_config['max_steps']}")
    
    # Use saved LR config if resuming, other use inputs
    if lr_config is None:
        lr_config = {
            'max_lr': args.lr,
            'min_lr': args.min_lr,
            'warmup_steps': args.warmup_steps,
            'max_steps': args.max_steps,
        }
        if args.resume and is_main_process():
            print(f"No lr_config in checkpoint, using defaults")

    # Train
    model.train()
    loss_fn = CrossEntropyLoss()
    strategy = getattr(args, 'strategy', 'ddp')
    
    for step in range(start_iter, args.max_steps + 1):
        lr = learning_rate_schedule(
            step, 
            lr_config['max_lr'], 
            lr_config['min_lr'], 
            lr_config['warmup_steps'], 
            lr_config['max_steps']
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(grad_accum):
            x, y = get_distributed_batch(train_data, args.batch_size, args.context_length, device, rank, world_size, rng)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn.forward(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / grad_accum
            
            loss.backward()
            accum_loss += loss.item()
        
        # Manual gradient sync for custom DDP implementations or gloo backend
        if world_size > 1 and strategy in ['ddp_bucketed', 'ddp_flat']:
            model.finish_gradient_synchronization()
        elif use_manual_grad_sync:
            # Manual all-reduce for gloo backend
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)
        
        model_for_clip = model.module if hasattr(model, 'module') else model
        torch.nn.utils.clip_grad_norm_(model_for_clip.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log
        if step % args.log_interval == 0:
            # All-reduce loss for accurate reporting
            loss_tensor = torch.tensor([accum_loss], device=device)
            all_reduce_mean(loss_tensor)
            avg_loss = loss_tensor.item()
            
            # Validation (all ranks participate)
            val_loss = None
            if val_data is not None and step % args.eval_interval == 0 and step > 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_steps):
                    vx, vy = get_distributed_batch(val_data, args.batch_size, args.context_length, device, rank, world_size, rng)
                    with torch.no_grad():
                        vlogits = model(vx)
                        val_losses.append(loss_fn.forward(
                            vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)
                        ).item())
                model.train()
                val_loss = sum(val_losses) / len(val_losses)
                
                # All-reduce val loss
                val_tensor = torch.tensor([val_loss], device=device)
                all_reduce_mean(val_tensor)
                val_loss = val_tensor.item()
            
            if is_main_process():
                elapsed = logger.elapsed_time()
                iters_done = step - start_iter
                msg = f"step {step} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                
                if iters_done > 0:
                    eta = (args.max_steps - step) * elapsed / iters_done / 60
                    msg += f" | ETA {eta:.1f}m"

                metrics = {"loss": avg_loss, "lr": lr}

                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                    msg += f" | val {val_loss:.4f}"

                logger.log(step, metrics)
                print(msg)

        if step > 0 and step % args.save_interval == 0 and is_main_process():
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint(model_to_save, optimizer, step, ckpt_dir / f"ckpt_{step}.pt", lr_config)

    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model
        save_checkpoint(model_to_save, optimizer, args.max_steps, ckpt_dir / "ckpt_final.pt", lr_config)
        logger.finish()
        print("Pretraining done.")
    
    cleanup_distributed()


# Fine-tuning
def finetune(args):
    """Fine-tuning loop with optional distributed training."""
    # Setup distributed if using torchrun
    backend = getattr(args, 'backend', 'auto')
    dist_info = setup_distributed(backend=backend)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]
    
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    
    if is_main_process():
        print(f"Device: {device}" + (f" (rank {rank}/{world_size})" if world_size > 1 else ""))

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    logger = Logger(project=args.project, name=output_dir.name, config=vars(args)) if is_main_process() else None

    # Data
    train_paths = [p.strip() for p in args.train_data.split(',')]
    train_datasets = [np.load(p, mmap_mode='r') for p in train_paths]
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    total_tokens = sum(len(d) for d in train_datasets)
    if is_main_process():
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
    if is_main_process():
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    num_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Parameters: {num_params:,}")

    # Wrap model for distributed training
    use_manual_grad_sync = False
    if world_size > 1:
        strategy = getattr(args, 'strategy', 'ddp')
        backend_used = getattr(args, 'backend', 'auto')
        
        # For gloo backend (CPU testing), use manual gradient sync
        if backend_used == 'gloo':
            use_manual_grad_sync = True
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if is_main_process():
                print("Using: Manual gradient sync (gloo backend)")
        elif strategy == 'ddp':
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[dist_info["local_rank"]])
            if is_main_process():
                print("Using: DistributedDataParallel (DDP)")
        elif strategy == 'ddp_bucketed':
            from ryan_gpt_systems.ddp_bucket import DDPBucketed
            model = DDPBucketed(model, bucket_size_mb=getattr(args, 'bucket_size_mb', 25.0))
            if is_main_process():
                print("Using: DDPBucketed")
        elif strategy == 'ddp_flat':
            from ryan_gpt_systems.ddp_flat import DDPIndividualParameters
            model = DDPIndividualParameters(model)
            if is_main_process():
                print("Using: DDPIndividualParameters")
        elif strategy == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
            mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
            model = FSDP(model, mixed_precision=mp_policy, device_id=device)
            if is_main_process():
                print("Using: FSDP (Fully Sharded Data Parallel)")

    # Gradient accumulation info
    grad_accum = args.gradient_accumulation_steps
    effective_batch = args.batch_size * grad_accum * world_size
    if is_main_process():
        print(f"Batch size: {args.batch_size} x {grad_accum} accum x {world_size} GPUs = {effective_batch} effective")

    # Fresh optimizer for fine-tuning
    model_params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
    
    if world_size > 1 and getattr(args, 'strategy', 'ddp') == 'zero':
        from ryan_gpt_systems.optimizer_state_sharding import ShardedOptimizer
        optimizer = ShardedOptimizer(model_params, AdamW, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        if is_main_process():
            print("Using: ShardedOptimizer (ZeRO-style)")
    else:
        optimizer = AdamW(model_params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    # Train
    model.train()
    loss_fn = CrossEntropyLoss()
    strategy = getattr(args, 'strategy', 'ddp')
    
    # Prepare mixing probabilities for multiple datasets
    if len(train_datasets) > 1:
        if args.mix:
            try:
                probs = [float(x) for x in args.mix.split(',')]
                assert len(probs) == len(train_datasets)
                probs = np.array(probs, dtype=float)
                probs = probs / probs.sum()
            except Exception:
                if is_main_process():
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

        # Gradient accumulation loop
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(grad_accum):
            ds_idx = int(rng.choice(len(train_datasets), p=probs))
            x, y = get_distributed_batch(train_datasets[ds_idx], args.batch_size, args.context_length, device, rank, world_size, rng)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn.forward(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / grad_accum  # Scale loss for accumulation
            
            loss.backward()
            accum_loss += loss.item()
        
        # Manual gradient sync for custom DDP implementations or gloo backend
        if world_size > 1 and strategy in ['ddp_bucketed', 'ddp_flat']:
            model.finish_gradient_synchronization()
        elif use_manual_grad_sync:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)
        
        model_for_clip = model.module if hasattr(model, 'module') else model
        torch.nn.utils.clip_grad_norm_(model_for_clip.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log
        if step % args.log_interval == 0:
            # All-reduce loss
            loss_tensor = torch.tensor([accum_loss], device=device)
            all_reduce_mean(loss_tensor)
            avg_loss = loss_tensor.item()
            
            # Validation (all ranks participate)
            val_loss = None
            if val_data is not None and step % args.eval_interval == 0 and step > 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_steps):
                    vx, vy = get_distributed_batch(val_data, args.batch_size, args.context_length, device, rank, world_size, rng)
                    with torch.no_grad():
                        vlogits = model(vx)
                        val_losses.append(loss_fn.forward(
                            vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)
                        ).item())
                model.train()
                val_loss = sum(val_losses) / len(val_losses)
                
                val_tensor = torch.tensor([val_loss], device=device)
                all_reduce_mean(val_tensor)
                val_loss = val_tensor.item()
            
            if is_main_process():
                elapsed = logger.elapsed_time()
                msg = f"step {step} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
                
                if step > 0:
                    eta = (args.max_steps - step) * elapsed / step / 60
                    msg += f" | ETA {eta:.1f}m"

                metrics = {"loss": avg_loss, "lr": lr}

                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                    msg += f" | val {val_loss:.4f} | diff {(val_loss - avg_loss):.4f}"

                logger.log(step, metrics)
                print(msg)

        if step > 0 and step % args.save_interval == 0 and is_main_process():
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint(model_to_save, optimizer, step, ckpt_dir / f"ckpt_{step}.pt")

    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model
        save_checkpoint(model_to_save, optimizer, args.max_steps, ckpt_dir / "ckpt_final.pt")
        logger.finish()
        print("Fine-tuning done.")
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="GPT Trainer")
    subparsers = parser.add_subparsers(dest="cmd")

    # Tokenize (train tokenizer + tokenize file)
    tok = subparsers.add_parser("tokenize", help="Train BPE tokenizer and tokenize data")
    tok.add_argument("--input", required=True, help="Input text file")
    tok.add_argument("--output_dir", required=True, help="Output directory for vocab and tokens")
    tok.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")

    # Tokenize file (using existing tokenizer)
    tokfile = subparsers.add_parser("tokenize_file", help="Tokenize a file using existing tokenizer")
    tokfile.add_argument("--input", required=True, help="Input text file")
    tokfile.add_argument("--output", required=True, help="Output .npy file")
    tokfile.add_argument("--tokenizer_dir", required=True, help="Directory with vocab.json and merges.txt")
    # Train (Pretrain)
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
    tr.add_argument("--gradient_accumulation_steps", type=int, default=1)
    tr.add_argument("--seed", type=int, default=42)
    # Logging
    tr.add_argument("--log_interval", type=int, default=100)
    tr.add_argument("--eval_interval", type=int, default=500)
    tr.add_argument("--eval_steps", type=int, default=20)
    tr.add_argument("--save_interval", type=int, default=2000)
    tr.add_argument("--resume", default=None, help="Resume from checkpoint")
    # Distributed training
    tr.add_argument("--strategy", type=str, default="ddp", 
                    choices=["ddp", "ddp_bucketed", "ddp_flat", "zero", "fsdp"],
                    help="Distributed training strategy")
    tr.add_argument("--bucket_size_mb", type=float, default=25.0, help="Bucket size for ddp_bucketed")
    tr.add_argument("--backend", type=str, default="auto", choices=["auto", "nccl", "gloo"],
                    help="Distributed backend: nccl (GPU), gloo (CPU/testing), auto (detect)")

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
    ft.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ft.add_argument("--seed", type=int, default=42)
    # Logging
    ft.add_argument("--log_interval", type=int, default=50)
    ft.add_argument("--eval_interval", type=int, default=250)
    ft.add_argument("--eval_steps", type=int, default=20)
    ft.add_argument("--save_interval", type=int, default=1000)
    ft.add_argument("--mix", default=None, help="Comma-separated mixing proportions for multiple train_data files")
    # Distributed training
    ft.add_argument("--strategy", type=str, default="ddp", 
                    choices=["ddp", "ddp_bucketed", "ddp_flat", "zero", "fsdp"],
                    help="Distributed training strategy")
    ft.add_argument("--bucket_size_mb", type=float, default=25.0, help="Bucket size for ddp_bucketed")
    ft.add_argument("--backend", type=str, default="auto", choices=["auto", "nccl", "gloo"],
                    help="Distributed backend: nccl (GPU), gloo (CPU/testing), auto (detect)")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()