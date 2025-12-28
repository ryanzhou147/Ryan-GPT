import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class DDPIndividualParameters:
    
    def __init__(self, module: nn.Module):
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        
        # Broadcast parameters from rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Register hooks for async all-reduce
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._hook)
    
    def _hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, param))
    
    def __call__(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad.data /= self.world_size
        self.handles.clear()
    
    def __getattr__(self, name):
        # Forward any attribute access to the wrapped module
        return getattr(self.module, name)

if __name__ == "__main__":
    mp.spawn(train, args=(world_size,), nprocs=world_size)