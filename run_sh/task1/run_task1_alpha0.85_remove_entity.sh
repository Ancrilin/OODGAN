#! /bin/bash

seeds="16"

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
  --data_file=binary_task1 \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --alpha=0.85  \
  --remove_entity \
  --entity_mode=2 \
  --result=task1_alpha0.85_remove_entity_mode2/task1_alpha0.85_remove_entity_mode2 \
  --output_dir=task1_alpha0.85_remove_entity_mode2/task1_alpha0.85_remove_entity_mode2-${seed}

done
exit 0