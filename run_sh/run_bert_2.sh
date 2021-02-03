#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192 47790 10464 28349 48533 28602 16850 35085"

for seed in ${seeds} ; do
  python -m app.run_bert_2 \
  --seed=${seed}  \
  --fine_tune \
  --n_epoch=500  \
  --patience=10 \
  --lr=4e-5  \
  --n_epoch=500 \
  --patience=10 \
  --dataset=smp \
  --data_file=binary_smp_full \
  --bert_type=bert-base-chinese \
  --do_train \
  --do_eval \
  --do_test \
  --result=base-bert_2/base-bert_2 \
  --output_dir=base-bert_2/base-bert_2-smp_${seed}

done
exit 0