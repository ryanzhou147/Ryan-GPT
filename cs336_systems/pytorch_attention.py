import torch
import timeit

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    head_dims = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    for d_model in head_dims:
        for seq_len in seq_lengths:
            print(f"\nBenchmarking d_model={d_model}, seq_len={seq_len}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                from ryan_gpt_basics.transformer.multihead_self_attention import MultiHeadSelfAttention
                mha = MultiHeadSelfAttention(d_model=d_model, num_heads=1, device=device).to(device)
                mha = torch.compile(mha)
                Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)

                # Warmup
                for _ in range(20):
                    out = mha(Q)
                    out.mean().backward()
                    mha.zero_grad()
                torch.cuda.synchronize()

                # Time forward passes
                torch.cuda.synchronize()
                start = timeit.default_timer()
                for _ in range(100):
                    out = mha(Q)
                    torch.cuda.synchronize()
                forward_time = timeit.default_timer() - start
                print(f"Forward (100 passes): {forward_time:.4f}s")

                # Measure memory after forward, before backward
                mem_before_backward = torch.cuda.max_memory_allocated()
                print(f"Memory before backward: {mem_before_backward / (1024**2):.2f} MB")

                # Time backward passes
                torch.cuda.synchronize()
                start = timeit.default_timer()
                for _ in range(100):
                    out = mha(Q)
                    torch.cuda.synchronize()
                    out.mean().backward()
                    torch.cuda.synchronize()
                    mha.zero_grad()
                backward_time = timeit.default_timer() - start
                print(f"Forward+Backward (100 passes): {backward_time:.4f}s")

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"OOM for d_model={d_model}, seq_len={seq_len}")
                    torch.cuda.empty_cache()