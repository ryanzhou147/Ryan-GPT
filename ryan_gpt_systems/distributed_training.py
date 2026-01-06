"""
Large-scale distributed training strategies for ryan-gpt.py
- Multi-GPU training with DDP (Data Distributed Parallel)
- TPU training with PyTorch XLA
- ZeRO-style optimizer sharding
- FSDP (Fully Sharded Data Parallel)
- Gradient checkpointing for memory efficiency
"""

import os
import functools
from typing import Optional, Callable, Any, Type
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from ryan_gpt_systems.ddp_bucket import DDPBucketed
from ryan_gpt_systems.ddp_flat import DDPIndividualParameters
from ryan_gpt_systems.optimizer_state_sharding import ShardedOptimizer


def is_tpu_available() -> bool:
    """Check if TPU is available via torch_xla."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False


def is_distributed() -> bool:
    """Check if we're in a distributed environment."""
    return dist.is_initialized()


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> dict:
    """
    Initialize distributed training environment.
    
    Returns dict with: rank, world_size, local_rank, device
    """
    # Already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device,
        }
    
    # Check for torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        
        # Set CUDA device
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        if rank == 0:
            print(f"Initialized distributed: {world_size} processes")
        
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device,
        }
    
    # Single GPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "device": device,
    }


def setup_tpu(tpu_cores: int = 8) -> dict:
    """
    Initialize TPU training environment.
    
    Args:
        tpu_cores: Number of TPU cores (1, 8, or more for pods)
    
    Returns dict with: rank, world_size, device
    """
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    
    if rank == 0:
        print(f"Initialized TPU: {world_size} cores")
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": rank,
        "device": device,
    }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedStrategy:
    """Base class for distributed training strategies."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        raise NotImplementedError
    
    def wrap_optimizer(
        self, 
        optimizer_cls: Type[Optimizer], 
        model: nn.Module, 
        **optimizer_kwargs
    ) -> Optimizer:
        """Create optimizer, potentially with sharding."""
        return optimizer_cls(model.parameters(), **optimizer_kwargs)
    
    def sync_gradients(self, model: nn.Module):
        """Synchronize gradients across processes (if needed)."""
        pass
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """All-reduce a tensor across processes."""
        if get_world_size() == 1:
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if op == "mean":
            tensor.div_(get_world_size())
        return tensor
    
    @contextmanager
    def no_sync(self, model: nn.Module):
        """Context manager to disable gradient sync (for gradient accumulation)."""
        yield


class SingleGPUStrategy(DistributedStrategy):
    """Single GPU training (no distribution)."""
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)


class DDPStrategy(DistributedStrategy):
    """PyTorch DistributedDataParallel strategy."""
    
    def __init__(self, device: torch.device, find_unused_parameters: bool = False):
        super().__init__(device)
        self.find_unused_parameters = find_unused_parameters
        self._model = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._model = DDP(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=self.find_unused_parameters,
        )
        return self._model
    
    @contextmanager
    def no_sync(self, model: nn.Module):
        """Disable gradient sync during gradient accumulation."""
        if hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield


class DDPBucketedStrategy(DistributedStrategy):
    """Custom bucketed DDP strategy (from ddp_bucket.py)."""
    
    def __init__(self, device: torch.device, bucket_size_mb: float = 25.0):
        super().__init__(device)
        self.bucket_size_mb = bucket_size_mb
        self._wrapper = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        self._wrapper = DDPBucketed(model, bucket_size_mb=self.bucket_size_mb)
        return self._wrapper
    
    def sync_gradients(self, model: nn.Module):
        """Must be called after backward() to sync gradients."""
        if self._wrapper is not None:
            self._wrapper.finish_gradient_synchronization()


class DDPFlatStrategy(DistributedStrategy):
    """Individual parameter async all-reduce strategy (from ddp_flat.py)."""
    
    def __init__(self, device: torch.device):
        super().__init__(device)
        self._wrapper = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        self._wrapper = DDPIndividualParameters(model)
        return self._wrapper
    
    def sync_gradients(self, model: nn.Module):
        """Must be called after backward() to sync gradients."""
        if self._wrapper is not None:
            self._wrapper.finish_gradient_synchronization()


class ZeROStrategy(DistributedStrategy):
    """
    ZeRO-style optimizer state sharding.
    Combines DDP for gradients with sharded optimizer states.
    """
    
    def __init__(self, device: torch.device):
        super().__init__(device)
        self._model = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        return self._model
    
    def wrap_optimizer(
        self, 
        optimizer_cls: Type[Optimizer], 
        model: nn.Module, 
        **optimizer_kwargs
    ) -> Optimizer:
        """Create sharded optimizer."""
        # Get the underlying module if wrapped in DDP
        if hasattr(model, 'module'):
            params = model.module.parameters()
        else:
            params = model.parameters()
        
        return ShardedOptimizer(params, optimizer_cls, **optimizer_kwargs)
    
    @contextmanager
    def no_sync(self, model: nn.Module):
        if hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield


class FSDPStrategy(DistributedStrategy):
    """
    Fully Sharded Data Parallel (FSDP) strategy.
    Shards model parameters, gradients, AND optimizer states.
    Best for very large models that don't fit on a single GPU.
    """
    
    def __init__(
        self, 
        device: torch.device,
        sharding_strategy: str = "full",  # "full", "shard_grad_op", "no_shard"
        mixed_precision: bool = True,
        activation_checkpointing: bool = False,
        cpu_offload: bool = False,
    ):
        super().__init__(device)
        self.sharding_strategy = sharding_strategy
        self.mixed_precision = mixed_precision
        self.activation_checkpointing = activation_checkpointing
        self.cpu_offload = cpu_offload
        self._model = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
        )
        
        model = model.to(self.device)
        
        # Sharding strategy
        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding = strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        # Mixed precision
        mp_policy = None
        if self.mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        
        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.cpu_offload else None
        
        # Auto wrap policy - wrap transformer layers
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1e6,  # Wrap layers with >1M params
        )
        
        self._model = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
        )
        
        # Optional activation checkpointing
        if self.activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )
            # Apply checkpointing to transformer blocks
            # This needs to be customized based on your model architecture
            pass
        
        return self._model
    
    @contextmanager
    def no_sync(self, model: nn.Module):
        if hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield


class TPUStrategy(DistributedStrategy):
    """
    TPU training strategy using PyTorch XLA.
    
    Requirements:
        pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    """
    
    def __init__(self, device=None):
        import torch_xla.core.xla_model as xm
        if device is None:
            device = xm.xla_device()
        super().__init__(device)
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)
    
    def sync_gradients(self, model: nn.Module):
        """Reduce gradients across TPU cores."""
        import torch_xla.core.xla_model as xm
        xm.reduce_gradients(model)
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """All-reduce using XLA."""
        import torch_xla.core.xla_model as xm
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
        if op == "mean":
            reduced = reduced / xm.xrt_world_size()
        return reduced
    
    def optimizer_step(self, optimizer: Optimizer):
        """TPU-specific optimizer step."""
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
    
    def mark_step(self):
        """Mark step for TPU (triggers execution)."""
        import torch_xla.core.xla_model as xm
        xm.mark_step()


# =============================================================================
# Strategy Factory
# =============================================================================

def get_strategy(
    strategy_name: str = "auto",
    device: Optional[torch.device] = None,
    **kwargs,
) -> DistributedStrategy:
    """
    Get the appropriate distributed strategy.
    
    Args:
        strategy_name: One of "auto", "single", "ddp", "ddp_bucketed", 
                       "ddp_flat", "zero", "fsdp", "tpu"
        device: Target device (auto-detected if None)
        **kwargs: Strategy-specific arguments
    
    Returns:
        Configured DistributedStrategy instance
    """
    strategies = {
        "single": SingleGPUStrategy,
        "ddp": DDPStrategy,
        "ddp_bucketed": DDPBucketedStrategy,
        "ddp_flat": DDPFlatStrategy,
        "zero": ZeROStrategy,
        "fsdp": FSDPStrategy,
        "tpu": TPUStrategy,
    }
    
    if strategy_name == "auto":
        if is_tpu_available():
            strategy_name = "tpu"
        elif is_distributed():
            strategy_name = "ddp"
        else:
            strategy_name = "single"
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
    
    # Auto-detect device
    if device is None:
        if strategy_name == "tpu":
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        elif is_distributed():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return strategies[strategy_name](device, **kwargs)


# =============================================================================
# Distributed Data Loading
# =============================================================================

class DistributedDataLoader:
    """
    Data loader that handles distributed sampling.
    Each rank gets a different portion of the data.
    """
    
    def __init__(
        self,
        data,  # numpy array or similar
        batch_size: int,
        seq_len: int,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        import numpy as np
        
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # Shard data among ranks
        total_len = len(data)
        shard_size = total_len // world_size
        self.start_idx = rank * shard_size
        self.end_idx = self.start_idx + shard_size if rank < world_size - 1 else total_len
        
        self.rng = np.random.default_rng(seed + rank)
    
    def get_batch(self):
        """Get a batch of data for this rank."""
        import numpy as np
        
        # Sample from this rank's shard
        n = self.end_idx - self.start_idx - self.seq_len
        if n <= 0:
            raise ValueError("Data shard too small for sequence length")
        
        starts = self.rng.integers(0, n, size=self.batch_size) + self.start_idx
        offsets = np.arange(self.seq_len + 1)
        seq = torch.tensor(
            self.data[starts[:, None] + offsets], 
            dtype=torch.long, 
            device=self.device
        )
        return seq[:, :-1], seq[:, 1:]
    
    def set_epoch(self, epoch: int):
        """Reset RNG for new epoch (for reproducibility)."""
        import numpy as np
        self.rng = np.random.default_rng(self.seed + self.rank + epoch * 1000)


# =============================================================================
# Training Loop Helper
# =============================================================================

class DistributedTrainer:
    """
    High-level distributed training helper.
    
    Example:
        trainer = DistributedTrainer(
            model=model,
            strategy="ddp",
            optimizer_cls=torch.optim.AdamW,
            lr=1e-4,
        )
        
        for step in range(max_steps):
            loss = trainer.train_step(batch)
            if step % log_interval == 0:
                print(f"Step {step}, Loss: {trainer.all_reduce_scalar(loss):.4f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "auto",
        optimizer_cls: Type[Optimizer] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        **optimizer_kwargs,
    ):
        # Setup distributed
        self.dist_info = setup_distributed()
        self.rank = self.dist_info["rank"]
        self.world_size = self.dist_info["world_size"]
        self.device = self.dist_info["device"]
        
        # Get strategy
        self.strategy = get_strategy(strategy, device=self.device)
        
        # Wrap model
        self.model = self.strategy.wrap_model(model)
        
        # Create optimizer
        if optimizer_cls is None:
            optimizer_cls = torch.optim.AdamW
        self.optimizer = self.strategy.wrap_optimizer(
            optimizer_cls, self.model, **optimizer_kwargs
        )
        
        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        
        # GradScaler for AMP
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        
        # Accumulation tracking
        self._accum_step = 0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> float:
        """
        Perform a single training step with gradient accumulation.
        
        Returns loss value (accumulated).
        """
        self.model.train()
        
        # Should we sync gradients on this step?
        sync_gradients = (self._accum_step + 1) % self.gradient_accumulation_steps == 0
        
        # Context for disabling sync during accumulation
        if sync_gradients or not hasattr(self.strategy, 'no_sync'):
            context = contextmanager(lambda: iter([None]))()
        else:
            context = self.strategy.no_sync(self.model)
        
        with context:
            # Forward pass
            if self.mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    loss = loss / self.gradient_accumulation_steps
            else:
                logits = self.model(input_ids)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        self._accum_step += 1
        
        # Optimizer step
        if sync_gradients:
            # Manual gradient sync for custom strategies
            self.strategy.sync_gradients(self.model)
            
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            if isinstance(self.strategy, TPUStrategy):
                self.strategy.optimizer_step(self.optimizer)
            elif self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self._accum_step = 0
        
        return loss.item() * self.gradient_accumulation_steps
    
    def all_reduce_scalar(self, value: float) -> float:
        """All-reduce a scalar value across all processes."""
        if self.world_size == 1:
            return value
        
        tensor = torch.tensor([value], device=self.device)
        tensor = self.strategy.all_reduce(tensor, op="mean")
        return tensor.item()
    
    def save_checkpoint(self, path: str, step: int, **extra_state):
        """Save checkpoint (only on rank 0)."""
        if self.rank != 0:
            return
        
        # Get model state
        model_state = self.model.state_dict()
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            **extra_state,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint."""
        map_location = {"cuda:0": f"cuda:{self.dist_info['local_rank']}"}
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint


# =============================================================================
# TPU-specific utilities
# =============================================================================

def tpu_spawn(fn: Callable, args: tuple = (), nprocs: int = 8):
    """
    Spawn function across TPU cores.
    
    Example:
        def train_fn(index, args):
            trainer = DistributedTrainer(model, strategy="tpu")
            for batch in dataloader:
                trainer.train_step(batch)
        
        tpu_spawn(train_fn, args=(config,), nprocs=8)
    """
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(fn, args=args, nprocs=nprocs)


# =============================================================================
# Example usage
# =============================================================================
def test_strategies():
    """Test all strategies work without crashing."""
    import torch.nn as nn
    
    # Tiny model
    model = nn.Linear(32, 32)
    
    # Test single GPU
    strategy = get_strategy("single")
    wrapped = strategy.wrap_model(model)
    x = torch.randn(2, 32, device=strategy.device)
    out = wrapped(x)
    print(f"✓ single: {out.shape}")
    
    # Test trainer (single mode) - use raw loss, skip reshape
    model2 = nn.Linear(32, 32)
    trainer = DistributedTrainer(
        model=model2,
        strategy="single",
        lr=1e-3,
        gradient_accumulation_steps=2,
    )
    
    loss_fn = nn.MSELoss()
    for i in range(4):
        x = torch.randn(2, 32, device=trainer.device)
        y = torch.randn(2, 32, device=trainer.device)
        
        # Manual forward/backward instead of train_step
        trainer.model.train()
        out = trainer.model(x)
        loss = loss_fn(out, y)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        print(f"  step {i}: loss={loss.item():.4f}")
    
    print("✓ trainer works")

if __name__ == "__main__":
    test_strategies()
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--strategy", type=str, default="auto",
#                         choices=["auto", "single", "ddp", "ddp_bucketed", "ddp_flat", "zero", "fsdp", "tpu"])
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--seq_len", type=int, default=512)
#     args = parser.parse_args()
    
#     # Setup
#     dist_info = setup_distributed()
#     print(f"Rank {dist_info['rank']}/{dist_info['world_size']} on {dist_info['device']}")
    
#     # Create dummy model
#     model = nn.Sequential(
#         nn.Embedding(10000, 256),
#         nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True),
#             num_layers=4,
#         ),
#         nn.Linear(256, 10000),
#     )
    
#     # Setup trainer
#     trainer = DistributedTrainer(
#         model=model,
#         strategy=args.strategy,
#         lr=1e-4,
#         weight_decay=0.01,
#         gradient_accumulation_steps=4,
#     )
    
#     # Dummy training loop
#     loss_fn = nn.CrossEntropyLoss()
#     for step in range(10):
#         # Random batch
#         x = torch.randint(0, 10000, (args.batch_size, args.seq_len), device=trainer.device)
#         y = torch.randint(0, 10000, (args.batch_size, args.seq_len), device=trainer.device)
        
#         loss = trainer.train_step(x, y, loss_fn)
        
#         if step % 2 == 0 and trainer.rank == 0:
#             avg_loss = trainer.all_reduce_scalar(loss)
#             print(f"Step {step}: loss = {avg_loss:.4f}")
    
#     cleanup_distributed()
#     print("Done!")
