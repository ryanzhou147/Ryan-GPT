#!/bin/bash
set -e
cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 1: Tokenize Combined Data"
echo "=========================================="
rm -rf data/tokenized
rm -f data/dailydialog/*.npy

PYTHONPATH=. python ryan_gpt_basics/train.py tokenize \
    --input data/combined/all_text.txt \
    --output_dir data/tokenized \
    --vocab_size 16000

python -c "
import numpy as np
data = np.load('data/tokenized/all_text.npy')
print(f'Total tokens: {len(data):,}')
"

echo "=========================================="
echo "STEP 2: Extract & Tokenize DailyDialog"
echo "=========================================="
python ryan_gpt_data/extract_dailydialog.py \
    --output_dir data/dailydialog

PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/train.txt \
    --output data/dailydialog/train.npy \
    --tokenizer_dir data/tokenized

PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/validation.txt \
    --output data/dailydialog/val.npy \
    --tokenizer_dir data/tokenized

echo "=========================================="
echo "STEP 3: Pretrain (~3-4 hours)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/all_text.npy \
    --output_dir runs/pretrain_chinchilla_style \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 4 \
    --d_model 384 \
    --num_heads 6 \
    --d_ff 1536 \
    --batch_size 64 \
    --max_steps 40000 \
    --lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 1000 \
    --log_interval 100 \
    --save_interval 5000

echo "=========================================="
echo "STEP 4: Fine-tune on DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
    --train_data data/dailydialog/train.npy \
    --val_data data/dailydialog/val.npy \
    --checkpoint runs/pretrain_chinchilla_style/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune_chinchilla_style \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 4 \
    --d_model 384 \
    --num_heads 6 \
    --d_ff 1536 \
    --batch_size 64 \
    --max_steps 2000 \
    --lr 3e-5 \
    --min_lr 1e-6 \
    --warmup_steps 100 \
    --log_interval 50 \
    --save_interval 500

echo "=========================================="
echo "STEP 5: Chat"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/chat.py \
    --checkpoint runs/finetune_chinchilla_style/checkpoints/ckpt_final.pt \
    --temperature 0.5

echo "=========================================="
echo "DONE!"
echo "=========================================="