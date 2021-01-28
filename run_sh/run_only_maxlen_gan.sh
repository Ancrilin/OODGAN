#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"

# $1 maxlen

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
  --n_epoch=500 \
  --patience=10 \
  --dataset=smp \
  --data_file=binary_smp_full \
  --bert_type=bert-base-chinese \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --output_dir=only-maxlen_${1}-gan/only-maxlen_${1}-gan-smp_${seed}  \
  --maxlen=${1}

done
exit 0