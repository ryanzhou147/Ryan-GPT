#!/bin/bash
set -e
cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 0: Data extraction & combine"
echo "=========================================="

# Wikipedia
mkdir -p data/wikipedia
python ryan_gpt_data/extract_wikipedia.py --out data/wikipedia/wiki_text.txt

# C4 (news subset / extractor)
mkdir -p data/c4
python ryan_gpt_data/extract_C4_news.py --out data/c4/c4.txt

# DailyDialog (extracts train/val/test into data/dailydialog/)
mkdir -p data/dailydialog
python ryan_gpt_data/extract_dailydialog.py --out_dir data/dailydialog

# Combine wiki + c4 into a single corpus for tokenization
mkdir -p data/combined
cat data/wikipedia/wiki_text.txt data/c4/c4.txt > data/combined/all_text.txt

# Tokenize & build BPE vocab
python ryan_gpt_basics/train.py tokenize --input data/combined/all_text.txt --output_dir data/tokenized --vocab_size 16000


echo "=========================================="
echo "STEP 1: Pretrain (~8-10 hours)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/all_text.npy \
    --output_dir runs/pretrain \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 6 \
    --d_model 320 \
    --num_heads 5 \
    --d_ff 1280 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_steps 80000 \
    --lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 2000 \
    --log_interval 100 \
    --save_interval 5000

echo "=========================================="
echo "STEP 2: Fine-tune on DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
    --train_data data/dailydialog/train.npy \
    --val_data data/dailydialog/val.npy \
    --checkpoint runs/pretrain/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 6 \
    --d_model 320 \
    --num_heads 5 \
    --d_ff 1280 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_steps 1000 \
    --lr 3e-5 \
    --min_lr 3e-6 \
    --warmup_steps 50 \
    --log_interval 50 \
    --save_interval 250 \
    --eval_interval 100

echo "=========================================="
echo "STEP 3: Deploy checkpoints to models/"
echo "=========================================="

# Ensure models/ layout expected by the web UI and copy final checkpoints there
mkdir -p models/pretrain
if [ -f runs/pretrain/checkpoints/ckpt_final.pt ]; then
    cp runs/pretrain/checkpoints/ckpt_final.pt models/pretrain/ckpt_final.pt
fi

mkdir -p models/finetune_dailydialog
if [ -f runs/finetune/checkpoints/ckpt_final.pt ]; then
    cp runs/finetune/checkpoints/ckpt_final.pt models/finetune_dailydialog/ckpt_final.pt
fi

echo "Deployed checkpoints to models/"
