#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192 47790 10464 28349 48533 28602 16850 35085"

for seed in ${seeds} ; do
  python -m app.run_bert \
  --seed=${seed}  \
  --fine_tune \
  --n_epoch=500  \
  --patience=10 \
  --lr=4e-5  \
  --n_epoch=500 \
  --patience=10 \
  --dataset=smp \
  --data_file=binary_true_smp_full_v2 \
  --bert_type=bert-base-chinese \
  --do_train \
  --do_eval \
  --do_test \
  --result=base-bert/base-bert \
  --output_dir=base-bert/base-bert-smp_${seed}

done
exit 0