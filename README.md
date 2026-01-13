# Ryan-GPT

TLDR; 12.5M parameter transformer LM. Trained in 12 hours on a single RTX 3060 (12GB).

## Quick Start
```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn torch

# Download checkpoints
chmod +x download_models.sh && ./download_models.sh

# Run
python webapp/app.py --host 127.0.0.1 --port 8080
```

**Controls**: `temperature` (0.01–0.99), `min_tokens` (5 for finetune, 10 for pretrain)

## Example Prompts

**Pretrain**: `The capital of France is`, `Albert Einstein was born in`

**Finetune**: `Hello`, `How are you?`, `What is your name?`

## Architecture

| | |
|-|-|
| Layers / d_model / heads | 6 / 320 / 5 (head_dim=64) |
| FFN | 1280 (4×d) |
| Context | 512 |
| Vocab | 16,000 BPE |
| Params | 12.5M |
| Attention | Flash Attention v2 |
| Positional encoding | RoPE (θ=10000) |

**Param breakdown**: embeddings 5.1M + 6×(attn 0.4M + ffn 0.8M) = 12.5M

## Training Data

| Dataset | Raw | Tokens | 
|---------|-----|--------|
| Wikipedia | 182 MB | ~160M |
| C4 (cleaned) | 165 MB | ~137M |
| **Pretrain total** | **347 MB** | **297M** |
| DailyDialog train | 6.5 MB | 1.61M |
| DailyDialog val | 0.6 MB | 0.15M |
| **Finetune total** | **7.1 MB** | **1.75M** |
| **Tokens/param** | 297/12.5 = **23.8x**|
| **Chinchilla optimal** | **20x** |

## Training Config

| | Pretrain | Finetune |
|-|----------|----------|
| Data | Wiki + C4 | DailyDialog |
| Steps | 80,000 | 1,000 |
| Batch × accum | 32 × 2 | 32 × 2 |
| Tokens/step | 32,768 | 32,768 |
| LR (cosine) | 6e-4 → 6e-5 | 3e-5 → 3e-6 |
| Warmup | 2,000 | 50 |
| Loss | 3.44 | 2.31 |
| Val loss | — | 2.49 |

<p align="center">
  <img src="assets/pretrain_loss.svg" width="100%" alt="Pretrain Loss">
  <img src="assets/finetune_loss.svg" width="100%" alt="Finetune Loss">
</p>

## Memory

**Inference (fp32)**:
- Weights: 50 MB
- KV cache (512 ctx): 6 layers × 2 × 320 × 512 × 4B = 7.9 MB
- Total: < 500 MB

**Training**: ~6-8 GB peak (batch=32, grad_accum=2)

**Flash Attention**: 2-4× reduction in attention activation memory. O(N) memory vs O(N²) for standard attention. Does not affect KV cache size.

## Pipeline
```bash
# 1. Extract
python ryan_gpt_data/extract_wikipedia.py --out data/wikipedia/wiki_text.txt
python ryan_gpt_data/extract_C4_news.py --out data/c4/c4.txt

# 2. Combine
cat data/wikipedia/wiki_text.txt data/c4/c4.txt > data/combined/all_text.txt

# 3. Tokenize (BPE)
python ryan_gpt_basics/train.py tokenize \
  --input data/combined/all_text.txt \
  --output_dir data/tokenized \
  --vocab_size 16000

# 4. Pretrain
python ryan_gpt_basics/train.py train \
  --train_data data/tokenized/all_text.npy \
  --output_dir runs/pretrain \
  --vocab_size 16000 --context_length 512 \
  --num_layers 6 --d_model 320 --num_heads 5 --d_ff 1280 \
  --batch_size 32 --gradient_accumulation_steps 2 \
  --max_steps 80000 --lr 6e-4 --min_lr 6e-5 --warmup_steps 2000

# 5. Finetune
python ryan_gpt_basics/train.py finetune \
  --checkpoint runs/pretrain/checkpoints/ckpt_final.pt \
  --train_data data/dailydialog/train.npy \
  --val_data data/dailydialog/val.npy \
  --output_dir runs/finetune \
  --max_steps 1000 --lr 3e-5 --min_lr 3e-6 --warmup_steps 50

# 6. Deploy
cp runs/finetune/checkpoints/ckpt_final.pt models/finetune/ckpt_final.pt
python webapp/app.py
```

# Distributed Training Primitives

This repository also contains the implementation code for the **[Distributed Training series](https://rzhou.me/thoughts/distributed-training)** on my blog.

The goal is not to replace PyTorch DDP, but to understand *how* they work by building them component by component and validating them on a single consumer GPU (RTX 3060).

### Supported Strategies

| Strategy | Description | Blog Post |
|----------|-------------|-----------|
| `single` | Standard single-process training | [Part 1](https://ryanzhou.com/thoughts/distributed-training1) |
| `ddp_flat` | Naive all-reduce per parameter | [Part 2](https://ryanzhou.com/thoughts/distributed-training2) |
| `ddp_bucketed` | Coalesced gradient buckets | [Part 3](https://ryanzhou.com/thoughts/distributed-training3) |
| `zero` | Optimizer state sharding + DDP | [Part 3](https://ryanzhou.com/thoughts/distributed-training3) |
| `tensor_parallel` | Column/Row parallel layers | [Part 4](https://ryanzhou.com/thoughts/distributed-training4) |

### Running the Distributed Code

The codebase supports switching between different parallelism strategies via the command line.

**Note:** If you run `torchrun` on a machine with fewer GPUs than the requested `nproc_per_node`, the code automatically falls back to **Simulation Mode** (using `gloo` backend on a single device) to verify logic correctness.

#### 1. Distributed Data Parallelism (DDP)
Spin up 4 processes to train a model using data parallelism. 

```bash
# Standard PyTorch DDP
torchrun --nproc_per_node=4 train.py --strategy ddp

# Custom Bucketed Implementation (from Part 3)
torchrun --nproc_per_node=4 train.py --strategy ddp_bucketed
```

#### 2. Optimizer Sharding (ZeRO)
Train using optimizer state sharding to reduce memory usage:

```bash
torchrun --nproc_per_node=4 train.py --strategy zero
```

#### 3. Tensor Parallelism
Train with layer-wise model partitioning:

```bash
torchrun --nproc_per_node=4 train.py --strategy tensor_parallel
```

### Benchmarking
To validate correctness and compare memory usage across strategies:

```bash
torchrun --nproc_per_node=4 benchmark_strategies.py --quick
```
This outputs **Correctness Checks** (weight sync verification), **Peak VRAM usage**, and **Throughput**.

## Acknowledgments
Project inspired by Stanford's [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/). Implementation, training pipeline, and model are my own.