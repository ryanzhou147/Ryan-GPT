#!/bin/bash
set -e

cd ~/Downloads/ryan-gpt

echo "=========================================="
echo "STEP 1: Extract Wikipedia (1000 articles)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_data/extract_wikipedia.py \
    --max_articles 1000

echo "=========================================="
echo "STEP 2: Tokenize Wikipedia"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py tokenize \
    --input data/wikipedia/wiki_text.txt \
    --output_dir data/tokenized \
    --vocab_size 10000

echo "=========================================="
echo "STEP 3: Extract DailyDialog"
echo "=========================================="
python ryan_gpt_data/extract_dailydialog.py \
    --output_dir data/dailydialog \
    --max_dialogues 1000

echo "=========================================="
echo "STEP 4: Tokenize DailyDialog"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/train.txt \
    --output data/dailydialog/train.npy \
    --tokenizer_dir data/tokenized

PYTHONPATH=. python ryan_gpt_basics/train.py tokenize_file \
    --input data/dailydialog/validation.txt \
    --output data/dailydialog/val.npy \
    --tokenizer_dir data/tokenized

echo "=========================================="
echo "STEP 5: Pretrain (Wikipedia)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py train \
    --train_data data/tokenized/wiki_text.npy \
    --output_dir runs/pretrain \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 12 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 32 \
    --max_steps 10000 \
    --lr 3e-4 \
    --warmup_steps 1000 \
    --log_interval 100 \
    --save_interval 1000

echo "=========================================="
echo "STEP 6: Finetune (70% Dialog / 30% Wiki)"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py finetune \
    --train_data data/dailydialog/train.npy,data/tokenized/wiki_text.npy \
    --val_data data/dailydialog/val.npy \
    --mix 0.7,0.3 \
    --checkpoint runs/pretrain/checkpoints/ckpt_final.pt \
    --output_dir runs/finetune \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 12 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 32 \
    --max_steps 5000 \
    --lr 5e-5 \
    --warmup_steps 300 \
    --log_interval 100 \
    --save_interval 500

echo "=========================================="
echo "STEP 7: Chat Test"
echo "=========================================="
PYTHONPATH=. python ryan_gpt_basics/train.py chat \
    --checkpoint runs/finetune/checkpoints/ckpt_final.pt \
    --tokenizer_dir data/tokenized \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 12 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 50

echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
