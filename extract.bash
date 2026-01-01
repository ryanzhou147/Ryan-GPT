mkdir -p models/pretrain models/finetune && \
curl -L -o models/pretrain/ckpt_final.pt "https://huggingface.co/ryanzhou147/ryan-gpt/resolve/main/pretrain_wikipedia/ckpt_final.pt" && \
curl -L -o models/finetune/ckpt_final.pt "https://huggingface.co/ryanzhou147/ryan-gpt/resolve/main/finetune_dailydialog/ckpt_final.pt" && \
