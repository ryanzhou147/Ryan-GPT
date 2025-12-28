import torch
import torch.nn as nn
import torch.distributed as dist

class DDPBucketed:
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        # Avoid __getattr__ recursion
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "world_size", dist.get_world_size())
        object.__setattr__(self, "bucket_size_bytes", int(bucket_size_mb * 1024 * 1024))

        object.__setattr__(self, "buckets", [])
        object.__setattr__(self, "param_to_bucket", {})

        # Broadcast parameters from rank 0
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        dist.barrier()

        self._create_buckets()

    def _create_buckets(self):
        params = list(reversed(list(self.module.parameters())))

        current_bucket = []
        current_size = 0

        for p in params:
            p_size = p.numel() * p.element_size()

            if current_bucket and current_size + p_size > self.bucket_size_bytes:
                self._finalize_bucket(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(p)
            current_size += p_size

        if current_bucket:
            self._finalize_bucket(current_bucket)

    def _finalize_bucket(self, params):
        bucket_idx = len(self.buckets)

        total_numel = sum(p.numel() for p in params)
        flat_buffer = torch.zeros(
            total_numel,
            dtype=params[0].dtype,
            device=params[0].device,
        )

        offset = 0
        for p in params:
            size = p.numel()
            self.param_to_bucket[p] = (bucket_idx, offset, size)
            offset += size

        self.buckets.append((flat_buffer, params))

    def finish_gradient_synchronization(self):
        """
        Call this AFTER loss.backward()
        """
        for flat_buffer, params in self.buckets:
            # Pack grads
            offset = 0
            for p in params:
                size = p.numel()
                if p.grad is None:
                    flat_buffer[offset:offset + size].zero_()
                else:
                    flat_buffer[offset:offset + size].copy_(
                        p.grad.data.view(-1)
                    )
                offset += size

            # All-reduce
            dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM)
            flat_buffer.div_(self.world_size)

            # Unpack grads
            offset = 0
            for p in params:
                size = p.numel()
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                p.grad.data.copy_(
                    flat_buffer[offset:offset + size].view_as(p)
                )
                offset += size

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "module"), name)
