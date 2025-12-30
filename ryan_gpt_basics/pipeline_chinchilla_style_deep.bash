#!/bin/bash
set -e
cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 1: Pretrain (~8-10 hours)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/all_text.npy \
    --output_dir runs/pretrain_chinchilla_style_deep \
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
    --checkpoint runs/pretrain_chinchilla_style_deep/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune_chinchilla_style_deep \
    --vocab_size 16000 \
    --context_length 512 \
    --num_layers 6 \
    --d_model 320 \
    --num_heads 5 \
    --d_ff 1280 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_steps 500 \
    --lr 3e-5 \
    --min_lr 3e-6 \
    --warmup_steps 50 \
    --log_interval 50 \
    --save_interval 250 \
    --eval_interval 100

echo "=========================================="
echo "STEP 3: Chat"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/chat.py \
    --checkpoint runs/finetune_deep/checkpoints/ckpt_final.pt \
    --temperature 0.5

echo "=========================================="
echo "DONE!"
echo "=========================================="