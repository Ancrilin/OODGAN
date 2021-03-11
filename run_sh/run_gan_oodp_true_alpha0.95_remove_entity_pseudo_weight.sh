
#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192 47790 10464 28349 48533 28602 16850 35085"

# $1 pseudo_sample_weight

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
  --alpha=0.95  \
  --manual_knowledge  \
  --remove_entity \
  --entity_mode=2 \
  --pseudo_sample_weight=${1} \
  --result=gan-oodp-alpha0.95-remove_entity-pseudo/gan-oodp-alpha0.95-remove_entity-pseudo${1}/gan-oodp-alpha0.95-remove_entity-pseudo${1} \
  --output_dir=gan-oodp-alpha0.95-remove_entity-pseudo/gan-oodp-alpha0.95-remove_entity-pseudo${1}/gan-oodp-alpha0.95-remove_entity-pseudo${1}-smp_${seed}

done
exit 0