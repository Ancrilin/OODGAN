#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192 47790 10464 28349 48533 28602 16850 35085"

for seed in ${seeds} ; do
  python -m app.run_gan \
  --seed=${seed}  \
  --fine_tune \
  --n_epoch=500  \
  --patience=10 \
  --fake_sample_weight=1.0  \
  --D_lr=2e-5 \
  --G_lr=2e-5 \
  --bert_lr=4e-5  \
  --G_z_dim=1024  \
  --n_epoch=500 \
  --patience=10 \
  --dataset=smp \
  --data_file=binary_true_smp_full_v2 \
  --bert_type=bert-base-chinese \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --result=base-gan-true_v2/base-gan-true_v2 \
  --output_dir=base-gan-true_v2/base-gan-true_v2-smp_${seed}

done
exit 0