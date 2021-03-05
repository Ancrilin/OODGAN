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
  --bert_type=bert-base-chinese \
  --dataset=smp \
  --data_file=binary_true_smp_full_v2 \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --alpha=0.85  \
  --stopwords \
  --remove_punctuation  \
  --result=base-gan-oodp-alpha0.85-psw/base-gan-oodp-alpha0.85-psw \
  --output_dir=base-gan-oodp-alpha0.85-psw/base-gan-oodp-alpha0.85-psw-smp_${seed}

done
exit 0