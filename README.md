# RyanGPT

TL;DR
- Proof of concept model. Trained in 12 hours on a single GPU (NVIDIA GeForce RTX 3060, 12 GB) with AMD Ryzen 5 7600 and 32 GB RAM.

Two models: Pretrain vs Fine-tune
- Pretrain: a base language model trained on large general corpora (Wikipedia + C4 here).
- Fine-tune: a short, task-specific adaptation of the pretrain checkpoint that specializes behavior for dialogue (Dailydialog).

Data sizes used for training
- Wikipedia: 182 MB
- C4: 165 MB
- DailyDialog: 7.4 MB

1) Run & use

Install and start the web UI:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn torch   # install correct torch build for your CUDA
python webapp/app.py --host 127.0.0.1 --port 8080
```

Open: http://127.0.0.1:8080

Controls
- Model: selects `models/<name>/ckpt_final.pt` (the app lists all `models/*/ckpt_final.pt`).
- temperature: sampling randomness, float in (0.01-0.99) exclusive; lower → deterministic, higher → more random.
- min_tokens: minimum tokens before EOS is allowed. (finetune model: 5 and pretrain model: 10).

Try these prompts:

pretrain_prompts = [
  'Albert Einstein was born in',
  'The capital of France is',
  'The Roman Empire was',
]

finetune_prompts = [
  'Hello',
  'How are you?',
  'What is your name?',
]

2) Training overview (steps)

Step 1 — Extract raw text

Example commands (use `ryan_gpt_data/` extractors or your own scripts):

```bash
# Wikipedia
python ryan_gpt_data/extract_wikipedia.py --out data/wikipedia/wiki_text.txt

# C4 news
python ryan_gpt_data/extract_C4_news.py --out data/c4/c4.txt
```

Step 2 — Clean / inspect data

No fixed command; use editor/tools or custom scripts in `ryan_gpt_data/` to clean and combine raw files. Example:

```bash
cat data/wikipedia/wiki_text.txt data/c4/c4.txt > data/combined/all_text.txt
```

Step 3 — Tokenize & build BPE vocabulary

```bash
python ryan_gpt_basics/train.py tokenize --input data/combined/all_text.txt --output_dir data/tokenized --vocab_size 16000
```

Step 4 — Pretrain (base model)

```bash
PYTHONPATH=. python ryan_gpt_basics/train.py train \
  --train_data data/tokenized/all_text.npy \
  --output_dir runs/pretrain \
  --vocab_size 16000 --context_length 512 \
  --num_layers 6 --d_model 320 --num_heads 5 --d_ff 1280 \
  --batch_size 32 --gradient_accumulation_steps 2 \
  --max_steps 80000 --lr 6e-4 --min_lr 6e-5 --warmup_steps 2000 \
  --log_interval 100 --save_interval 5000
```

Step 5 — Finetune (DailyDialog example)

```bash
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
  --train_data data/dailydialog/train.npy \
  --val_data data/dailydialog/val.npy \
  --checkpoint runs/pretrain/checkpoints/ckpt_final.pt \
  --output_dir runs/finetune \
  --vocab_size 16000 --context_length 512 \
  --num_layers 6 --d_model 320 --num_heads 5 --d_ff 1280 \
  --batch_size 32 --gradient_accumulation_steps 2 \
  --max_steps 1000 --lr 3e-5 --min_lr 3e-6 --warmup_steps 50 \
  --log_interval 50 --save_interval 250 --eval_interval 100
```

Step 6 — Deploy to web UI

```bash
# Example: deploy a finetune model (DailyDialog)
mkdir -p models/finetune_dailydialog
cp runs/finetune/checkpoints/ckpt_final.pt models/finetune_dailydialog/ckpt_final.pt

# Example: deploy a pretrain model
mkdir -p models/pretrain
cp runs/pretrain/checkpoints/ckpt_final.pt models/pretrain/ckpt_final.pt

# Start the web UI
python webapp/app.py --host 127.0.0.1 --port 8080
```

Model hyperparameters & rationale

Why these values
- `vocab_size=16000`: compact BPE vocabulary to reduce embedding size and memory while preserving expressivity for English-style corpora.
- `context_length=512`: balances useful context with KV-cache size; 512 tokens is sufficient for dialog and fits cache constraints.
- `num_layers=6`, `d_model=320`, `num_heads=5`, `d_ff=1280`: small, efficient transformer chosen to keep inference well under 12GB VRAM on an RTX3060 while producing reasonable quality.
- `batch_size=32` with `gradient_accumulation_steps=2`: training-time configuration to keep per-GPU batch compute stable while fitting GPU memory.

Quick parameter math (approximate)
- Embedding params: vocab_size * d_model = 16,000 * 320 = 5,120,000 params (~5.1M).
- Per-layer weights (self-attention + output + FFN):
  - attention weights ≈ 3*d*d + d*d = 4 * d^2 = 4 * 320^2 = 409,600
  - FFN weights ≈ 2 * d * d_ff = 2 * 320 * 1280 = 819,200
  - per-layer total ≈ 1,228,800
- Total model params ≈ embedding + num_layers * per-layer ≈ 5,120,000 + 6 * 1,228,800 ≈ 12,492,800 params (~12.5M).

Memory estimate (inference)
- Model weights (float32): ~12.5M * 4B ≈ 50 MB.
- KV cache (per token): 2 * d_model * 4B per layer. For context=512 and 6 layers: 6 * 2 * 320 * 512 * 4B ≈ 7.9 MB.
- FlashAttention: reduces attention activation / workspace memory compared with standard attention (typical implementations report ~2–4x lower temporary peak memory during attention computation). That lowers transient activation needs but does not change KV cache size.
- Activation buffers and framework overhead: reserve ~100–300 MB depending on runtime and whether FlashAttention is used.
- Total runtime footprint (weights + KV + overhead) remains well under 12 GiB — FlashAttention widens the safety margin on RTX3060 (12GB).

Scaling note
- Doubling `d_model` or `num_layers` increases parameter count and runtime memory nonlinearly (params scale ≈ O(num_layers * d^2) and KV cache scales with num_layers * d * context_length). The chosen dims (6×320) were selected to remain comfortably below 12GB for single-GPU inference and training on your machine while preserving quality.

3) Contribute

- Add data: place raw text under `data/` or add extractors in `ryan_gpt_data/`.
- Train: use `ryan_gpt_basics/train.py` for tokenize / train / finetune.
- Deploy model: copy `runs/<run>/checkpoints/ckpt_final.pt` → `models/<name>/ckpt_final.pt` and restart `webapp/app.py`.
- Verify: open the web UI and select your model.

Appendix — exact pipeline used is available in `ryan_gpt_basics/pipeline.bash`.