#!/bin/bash
set -e
cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 1: Pretrain (~3-4 hours)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/all_text.npy \
    --output_dir runs/pretrain_chinchilla_style \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 6 \
    --d_model 320 \
    --num_heads 5 \
    --d_ff 1280 \
    --batch_size 64 \
    --max_steps 40000 \
    --lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 1000 \
    --log_interval 100 \
    --save_interval 5000

echo "=========================================="
echo "STEP 2: Fine-tune on DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
    --train_data data/dailydialog/train.npy \
    --val_data data/dailydialog/val.npy \
    --checkpoint runs/pretrain_chinchilla_style/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune_chinchilla_style \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 6 \
    --d_model 320 \
    --num_heads 5 \
    --d_ff 1280 \
    --batch_size 64 \
    --max_steps 2000 \
    --lr 3e-5 \
    --min_lr 3e-6 \
    --warmup_steps 100 \
    --log_interval 50 \
    --save_interval 500

echo "=========================================="
echo "STEP 3: Chat"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/chat.py \
    --checkpoint runs/finetune_chinchilla_style/checkpoints/ckpt_final.pt \
    --temperature 0.5

echo "=========================================="
echo "DONE!"
echo "=========================================="