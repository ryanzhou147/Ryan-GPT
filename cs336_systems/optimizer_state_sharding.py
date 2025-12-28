import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any

class ShardedOptimizer(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.all_params = []  # (param, owner_rank)
        
        # Normalize params to list
        params_list = [params] if isinstance(params, torch.Tensor) else list(params)
        
        # Shard parameters among ranks
        sharded_params = []
        for i, p in enumerate(params_list):
            owner_rank = i % self.world_size
            self.all_params.append((p, owner_rank))
            if owner_rank == self.rank:
                sharded_params.append(p)
        
        # Initialize wrapped optimizer
        self.optimizer = optimizer_cls(sharded_params, **kwargs) if sharded_params else None
        
        # Initialize parent
        super().__init__([{'params': []}], kwargs)
        self.param_groups.clear()
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure) if self.optimizer else None
        
        # Broadcast updated params from owners
        for param, owner_rank in self.all_params:
            dist.broadcast(param.data, src=owner_rank)
        
        return loss
    
    def zero_grad(self, set_to_none=False):
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def add_param_group(self, param_group: dict[str, Any]):
        sharded_params = []
        for i, p in enumerate(param_group['params']):
            owner_rank = (len(self.all_params) + i) % self.world_size
            self.all_params.append((p, owner_rank))
            if owner_rank == self.rank:
                sharded_params.append(p)
        
        if sharded_params and self.optimizer:
            new_group = {k: v for k, v in param_group.items() if k != 'params'}
            new_group['params'] = sharded_params
            self.optimizer.add_param_group(new_group)

def get_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)

def profile_training(model_config, simulate_sharding=False):
    
    import gc
    import time
    from torch import nn

    # Setup
    device = torch.device("cuda:0")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    from ryan_gpt_basics.transformer.transformer import TransformerLM
    
    d_model, d_ff, num_layers, num_heads = model_config
    batch_size = 4
    context_length = 512
    vocab_size = 10000
    world_size = 2
    
    # Print header
    mode = "SHARDED" if simulate_sharding else "STANDARD"
    print(f"{mode} OPTIMIZER")
    
    # Create model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    ).to(device)
    
    mem_after_model = get_memory_mb()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model memory: {mem_after_model:.2f} MB")
    print(f"Parameters: {num_params:,}")
    
    # Create optimizer (sharded or standard)
    if simulate_sharding:
        all_params = list(model.parameters())
        sharded_params = [p for i, p in enumerate(all_params) if i % world_size == 0]
        optimizer = torch.optim.Adam(sharded_params, lr=1e-4)
        print(f"Sharded: {len(sharded_params)}/{len(all_params)} param tensors")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    target_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    # Run one step to allocate all memory
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
    loss.backward()
    
    mem_before_step = get_memory_mb()
    optimizer.step()
    mem_after_step = get_memory_mb()
    
    optimizer_mem = mem_after_step - mem_before_step
    print(f"Optimizer state: {optimizer_mem:.2f} MB")
    
    # Benchmark timing
    torch.cuda.reset_peak_memory_stats()
    times = []
    
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    print(f"Peak memory: {peak_mem:.2f} MB")
    print(f"Avg step time: {avg_time:.2f} ms")
    
    return {
        "optimizer_mem": optimizer_mem,
        "peak_mem": peak_mem,
        "avg_time": avg_time,
        "num_params": num_params,
    }


if __name__ == "__main__":
    # Small model config
    config = (768, 3072, 12, 12)  # d_model, d_ff, layers, heads
    
    print("Profiling small model")
    
    standard = profile_training(config, simulate_sharding=False)
    sharded = profile_training(config, simulate_sharding=True)
    
    # Summary
    saved = standard['optimizer_mem'] - sharded['optimizer_mem']
    pct = saved / standard['optimizer_mem'] * 100
    print(f"Memory saved: {saved:.0f} MB ({pct:.0f}%)")