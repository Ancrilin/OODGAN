#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192 47790 10464 28349 48533 28602 16850 35085"

for seed in ${seeds} ; do
  python -m app.run_oodp_gan \
  --seed=${seed}  \
  --D_lr=2e-5 \
  --G_lr=2e-5 \
  --bert_lr=2e-5 \
  --fine_tune \
  --n_epoch=500 \
  --patience=10 \
  --train_batch_size=32 \
  --bert_type=bert-base-uncased \
  --dataset=oos-eval \
  --data_file=binary_undersample \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --G_z_dim=1024  \
  -alpha=0.95 \
  --logarithm \
  --result=base-oos-gan-alpha0.95/base-oos-gan-alpha0.95 \
  --output_dir=base-oos-gan-alpha0.95/base-oos-gan-alpha0.95_${seed}

done
exit 0