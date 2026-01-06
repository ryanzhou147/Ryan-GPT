#!/usr/bin/env python
"""
Benchmarks multi-GPU training on a single GPU using gloo backend.
- DDPIndividualParameters (ddp_flat.py) - async per-param all-reduce
- DDPBucketed (ddp_bucket.py) - bucketed gradient sync  
- ShardedOptimizer (optimizer_state_sharding.py) - ZeRO-style optimizer
- TensorParallelTransformerLM (tensor_parallelism.py) - model parallelism
- Plus PyTorch baselines for comparison.
    
One GPU: torchrun --nproc_per_node=4 benchmark_strategies.py --quick
For actual multi-GPU: torchrun --nproc_per_node=4 benchmark_strategies.py --backend nccl
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchmarkConfig:
    name: str
    strategy: str
    use_sharded_optimizer: bool = False
    use_mixed_precision: bool = True
    bucket_size_mb: float = 25.0
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_steps: int = 100


@dataclass 
class BenchmarkResult:
    name: str
    strategy: str
    use_sharded_optimizer: bool
    use_mixed_precision: bool
    total_time_sec: float
    steps_per_sec: float
    tokens_per_sec: float
    peak_memory_mb: float
    allocated_memory_mb: float
    final_loss: float
    weights_synchronized: bool
    error: Optional[str] = None


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ctx_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)).view(1, 1, ctx_len, ctx_len))

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, S, D))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, ctx_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, num_heads, ctx_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, ctx_len, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, d_ff, ctx_len) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def setup_distributed(backend="auto"):
    if "RANK" not in os.environ:
        return {"rank": 0, "world_size": 1, "local_rank": 0, 
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "backend": "single"}
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gpus = torch.cuda.device_count()
    
    # Auto-select backend based on GPU availability
    if backend == "auto":
        backend = "nccl" if num_gpus >= world_size else "gloo"
    
    if backend == "nccl" and local_rank < num_gpus:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        # Gloo: all processes share GPU 0 (simulation mode)
        dist.init_process_group(backend="gloo")
        if num_gpus > 0:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        if rank == 0:
            print(f"[INFO] Simulating {world_size} ranks on {num_gpus} GPU(s) with gloo backend")
        backend = "gloo"
    
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": device, "backend": backend}

def sync_barrier(dist_info):
    """Synchronization barrier compatible with both backends."""
    if dist_info["world_size"] > 1:
        if dist_info["backend"] == "nccl":
            dist.barrier(device_ids=[dist_info["local_rank"]])
        else:
            dist.barrier()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def run_benchmark(cfg: BenchmarkConfig, dist_info: Dict) -> BenchmarkResult:
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    device = dist_info["device"]
    backend = dist_info["backend"]
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    gc.collect()
    
    sync_barrier(dist_info)
    
    try:
        wrapper = None
        needs_sync = False
        
        if cfg.strategy == "tensor_parallel":
            from ryan_gpt_systems.tensor_parallelism import TensorParallelGroup, TensorParallelTransformerLM
            tp = TensorParallelGroup()
            model = TensorParallelTransformerLM(
                cfg.vocab_size, cfg.context_length, cfg.num_layers,
                cfg.d_model, cfg.num_heads, cfg.d_ff, tp
            ).to(device)
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
        else:
            model = TransformerLM(
                cfg.vocab_size, cfg.context_length, cfg.num_layers,
                cfg.d_model, cfg.num_heads, cfg.d_ff
            ).to(device)
            
            if cfg.strategy == "ddp_flat":
                from ryan_gpt_systems.ddp_flat import DDPIndividualParameters
                wrapper = DDPIndividualParameters(model)
                needs_sync = True
                
            elif cfg.strategy == "ddp_bucketed":
                from ryan_gpt_systems.ddp_bucket import DDPBucketed
                wrapper = DDPBucketed(model, bucket_size_mb=cfg.bucket_size_mb)
                needs_sync = True
                
            elif cfg.strategy == "torch_ddp":
                from torch.nn.parallel import DistributedDataParallel as DDP
                for p in model.parameters():
                    dist.broadcast(p.data, src=0)
                if backend == "nccl":
                    wrapper = DDP(model, device_ids=[local_rank])
                else:
                    wrapper = DDP(model)  # gloo doesn't use device_ids
                
            elif cfg.strategy == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
                mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
                if backend == "nccl":
                    wrapper = FSDP(model, mixed_precision=mp, device_id=device)
                else:
                    wrapper = FSDP(model, mixed_precision=mp)
        
        fwd_model = wrapper if wrapper else model
        
        # Create optimizer        
        if cfg.use_sharded_optimizer and cfg.strategy not in ["fsdp", "tensor_parallel"]:
            from ryan_gpt_systems.optimizer_state_sharding import ShardedOptimizer
            params = fwd_model.module.parameters() if hasattr(fwd_model, 'module') else model.parameters()
            opt = ShardedOptimizer(params, torch.optim.AdamW, lr=3e-4, weight_decay=0.1)
        else:
            params = fwd_model.module.parameters() if hasattr(fwd_model, 'module') else model.parameters()
            opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.1)
        
        # Training        
        grad_accum = cfg.gradient_accumulation_steps
        tokens_per_step = cfg.batch_size * grad_accum * world_size * cfg.context_length
        use_amp = cfg.use_mixed_precision and device.type == "cuda" and cfg.strategy != "fsdp"
        
        # Warmup
        for _ in range(3):
            opt.zero_grad()
            for _ in range(grad_accum):
                x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
                y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
                if use_amp:
                    with torch.autocast("cuda", torch.bfloat16):
                        loss = F.cross_entropy(fwd_model(x).view(-1, cfg.vocab_size), y.view(-1))
                else:
                    loss = F.cross_entropy(fwd_model(x).view(-1, cfg.vocab_size), y.view(-1))
                (loss / grad_accum).backward()
            if needs_sync:
                wrapper.finish_gradient_synchronization()
            opt.step()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        sync_barrier(dist_info)
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        # Timed run
        t0 = time.perf_counter()
        total_loss = 0.0
        
        for step in range(cfg.num_steps):
            opt.zero_grad()
            step_loss = 0.0
            for _ in range(grad_accum):
                x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
                y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
                if use_amp:
                    with torch.autocast("cuda", torch.bfloat16):
                        loss = F.cross_entropy(fwd_model(x).view(-1, cfg.vocab_size), y.view(-1))
                else:
                    loss = F.cross_entropy(fwd_model(x).view(-1, cfg.vocab_size), y.view(-1))
                (loss / grad_accum).backward()
                step_loss += loss.item() / grad_accum
            
            if needs_sync:
                wrapper.finish_gradient_synchronization()
            
            base = fwd_model.module if hasattr(fwd_model, 'module') else model
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            opt.step()
            total_loss += step_loss
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        sync_barrier(dist_info)
        
        elapsed = time.perf_counter() - t0
        
        # Metrics
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
        alloc_mem = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
        
        # Check weight sync
        base = fwd_model.module if hasattr(fwd_model, 'module') else model
        wsum = sum(p.sum().item() for p in base.parameters())
        if world_size > 1:
            wsums = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(wsums, torch.tensor([wsum], device=device))
            synced = all(abs(wsums[0].item() - w.item()) < 0.5 for w in wsums)
        else:
            synced = True
        
        del model, opt
        if wrapper:
            del wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return BenchmarkResult(
            name=cfg.name, strategy=cfg.strategy,
            use_sharded_optimizer=cfg.use_sharded_optimizer,
            use_mixed_precision=cfg.use_mixed_precision,
            total_time_sec=elapsed,
            steps_per_sec=cfg.num_steps / elapsed,
            tokens_per_sec=cfg.num_steps * tokens_per_step / elapsed,
            peak_memory_mb=peak_mem, allocated_memory_mb=alloc_mem,
            final_loss=total_loss / cfg.num_steps,
            weights_synchronized=synced,
        )
    
    except Exception as e:
        import traceback
        return BenchmarkResult(
            name=cfg.name, strategy=cfg.strategy,
            use_sharded_optimizer=cfg.use_sharded_optimizer,
            use_mixed_precision=cfg.use_mixed_precision,
            total_time_sec=0, steps_per_sec=0, tokens_per_sec=0,
            peak_memory_mb=0, allocated_memory_mb=0,
            final_loss=0, weights_synchronized=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def print_table(results: List[BenchmarkResult], world_size: int, backend: str):
    print(f"Results for ({world_size} simulated GPUs, {backend} backend)")
    print(f"{'Strategy':<35} {'ZeRO':>5} {'MP':>4} | {'Steps/s':>8} {'MTok/s':>8} {'Time':>7} | {'Peak MB':>9} | {'Loss':>7} {'Sync':>5}")
    
    for r in results:
        if r.error:
            err_line = r.error.split('\n')[0][:60]
            print(f"{r.name:<35} ERROR: {err_line}")
            continue
        print(f"{r.name:<35} {'Y' if r.use_sharded_optimizer else 'N':>5} "
              f"{'Y' if r.use_mixed_precision else 'N':>4} | "
              f"{r.steps_per_sec:>8.2f} {r.tokens_per_sec/1e6:>8.3f} {r.total_time_sec:>6.1f}s | "
              f"{r.peak_memory_mb:>9.1f} | {r.final_loss:>7.4f} {'✓' if r.weights_synchronized else '✗':>5}")
    
    print("\nRyan-GPT Implementations:")
    print("  ddp_flat      : DDPIndividualParameters - async per-param all-reduce")
    print("  ddp_bucketed  : DDPBucketed - bucketed gradient sync")
    print("  + ZeRO        : ShardedOptimizer - optimizer state sharding")
    print("  tensor_parallel: TensorParallelTransformerLM - model parallelism")
    print("\nPytorch Baselines: torch_ddp, fsdp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--single", choices=["ddp_flat", "ddp_bucketed", "torch_ddp", "fsdp", "tensor_parallel"])
    args = parser.parse_args()
    
    dist_info = setup_distributed(args.backend)
    world_size = dist_info["world_size"]
    device = dist_info["device"]
    backend = dist_info["backend"]
    
    if is_main():
        print(f"Distributed Training Benchmark")
        print(f"Backend: {backend}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Steps: {'20 (quick)' if args.quick else args.steps}")
        print(f"Batch: {args.batch_size} x {args.grad_accum} accum x {world_size} ranks")
        print(f"{'#'*70}\n", flush=True)
    
    steps = 20 if args.quick else args.steps
    base = dict(vocab_size=10000, context_length=256, num_layers=6, d_model=512, num_heads=8, d_ff=2048,
                batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, num_steps=steps)
    
    configs = [
        BenchmarkConfig(name="DDPFlat (your ddp_flat.py)", strategy="ddp_flat", **base),
        BenchmarkConfig(name="DDPBucketed (your ddp_bucket.py)", strategy="ddp_bucketed", **base),
        BenchmarkConfig(name="DDPFlat + ZeRO", strategy="ddp_flat", use_sharded_optimizer=True, **base),
        BenchmarkConfig(name="DDPBucketed + ZeRO", strategy="ddp_bucketed", use_sharded_optimizer=True, **base),
        BenchmarkConfig(name="TensorParallel", strategy="tensor_parallel", **base),
        BenchmarkConfig(name="PyTorch DDP (baseline)", strategy="torch_ddp", **base),
        BenchmarkConfig(name="PyTorch FSDP (baseline)", strategy="fsdp", **base),
        BenchmarkConfig(name="PyTorch DDP + ZeRO", strategy="torch_ddp", use_sharded_optimizer=True, **base),
    ]
    
    if args.single:
        configs = [c for c in configs if c.strategy == args.single]
    
    results = []
    for i, cfg in enumerate(configs):
        if is_main():
            print(f"[{i+1}/{len(configs)}] {cfg.name}", flush=True)
        
        sync_barrier(dist_info)
        r = run_benchmark(cfg, dist_info)
        results.append(r)
        
        if is_main():
            if r.error:
                print(f"    Error: {r.error.split(chr(10))[0][:70]}", flush=True)
            else:
                print(f"    {r.steps_per_sec:.2f} steps/s, {r.tokens_per_sec/1e6:.3f} MTok/s, "
                      f"{r.peak_memory_mb:.0f} MB, sync={'✓' if r.weights_synchronized else '✗'}", flush=True)
        
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    if is_main():
        print_table(results, world_size, backend)
        
        with open(args.output, 'w') as f:
            json.dump({"world_size": world_size, "backend": backend, "results": [asdict(r) for r in results]}, f, indent=2)
        print(f"\nSaved: {args.output}")
    
    cleanup()