import timeit
import torch
from contextlib import nullcontext
import ryan_gpt_basics.transformer.transformer as transformer
import ryan_gpt_basics.optimizer.cross_entropy as cross_entropy
import ryan_gpt_basics.optimizer.adamw as adamw
import torch.cuda.nvtx as nvtx


def benchmark_transformer(vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, with_rope: bool = False,
                 rope_theta: float | None = None, max_seq_len: int | None = None,
                 device=None, dtype=None, batch_size: int = 4, warmup_steps: int = 10, benchmark_steps: int = 50, backward: bool = False,
                 use_bf16: bool = False, memory_snapshot: bool = False, snapshot_path: str = "memory_snapshot.pickle", compile: bool = False) -> None:
    
    print(f"  Config: batch={batch_size}, context={context_length}, layers={num_layers}")

    # Initialize the model
    model = transformer.TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, with_rope,
                 rope_theta, max_seq_len, device, dtype).to(device)
    
    if compile:
        model = torch.compile(model)

    model.train()
    # Generate a random batch of data
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    target_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    criterion = cross_entropy.CrossEntropyLoss()
    optimizer = adamw.AdamW(model.parameters())

    # choose autocast context: bf16 when requested and running on CUDA, else no-op
    if use_bf16 and torch.cuda.is_available():
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    # Warm-up steps
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            with amp_ctx:
                optimizer.zero_grad()
                outputs = model(input_data)
                if backward:
                    loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
                    loss.backward()
                    optimizer.step()
    
    # Benchmark steps
    times = []
    if torch.cuda.is_available() and hasattr(torch.cuda.memory, '_record_memory_history'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(benchmark_steps):

        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        with amp_ctx:
            with nvtx.range("zero_grad"):
                optimizer.zero_grad()
            
            with nvtx.range("forward"):
                outputs = model(input_data)
            
            if backward:
                with nvtx.range("loss"):
                    loss = criterion(outputs.view(-1, vocab_size), target_data.view(-1))
                
                with nvtx.range("backward"):
                    loss.backward()
                
                with nvtx.range("optimizer_step"):
                    optimizer.step()
            
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak memory: {peak_memory_gb:.2f} GB")
    if memory_snapshot:
        if torch.cuda.is_available() and hasattr(torch.cuda.memory, '_dump_snapshot'):

            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
        else:
            print("Warning: CUDA memory snapshot API not available; no snapshot written")

    avg_time_per_step = sum(times) / benchmark_steps
    std_time_per_step = (sum((t - avg_time_per_step) ** 2 for t in times) / benchmark_steps) ** 0.5
    print(f"Average time per step ({'forward + backward' if backward else 'forward only'}): {avg_time_per_step:.6f} seconds +/- {std_time_per_step:.6f} seconds")
    return avg_time_per_step

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-snapshot', action='store_true', help='Record CUDA memory history and dump a snapshot')
    parser.add_argument('--snapshot-path', type=str, default='./memory_snapshot.pickle', help='Path to write memory snapshot')
    parser.add_argument('--bf16', action='store_true', help='Run benchmarks with BF16 autocast when available')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--sizes', type=str, nargs='*', default=['small','medium','large','xl','2.7B'],
                        help='Which model sizes to benchmark')
    parser.add_argument('--forward-only', action='store_true', help='Skip backward pass')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_sizes = {
        "small": (768, 3072, 12, 12),
        "medium": (1024, 4096, 24, 16),
        "large": (1280, 5120, 36, 20),
        "xl": (1600, 6400, 48, 25),
        "2.7B": (2560, 10240, 32, 32)
    }
    vocab_size = 10000
    context_length = args.context
    batch_size = args.batch
    warmup_steps = args.warmup
    benchmark_steps = args.steps
    backward = not args.forward_only
    compile = args.compile

    for size_name in args.sizes:
        if size_name not in model_sizes:
            print(f"Unknown model size: {size_name}, skipping")
            continue
        d_model, d_ff, num_layers, num_heads = model_sizes[size_name]
        print(f"Benchmarking {size_name} model:")
        bf16_options = [False, True] if args.bf16 else [False]
        for use_bf16 in bf16_options:
            print(f"  Mixed BF16: {use_bf16}")
            try:
                benchmark_transformer(vocab_size, context_length, num_layers, d_model, num_heads, d_ff,
                                      device=device, batch_size=batch_size,
                                      warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
                                      backward=backward, use_bf16=use_bf16,
                                      memory_snapshot=args.memory_snapshot, snapshot_path=args.snapshot_path, compile=compile)
            except RuntimeError as e:
                print(f"  Skipping (OOM or other error): {e}")
                torch.cuda.empty_cache()