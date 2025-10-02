#!/bin/bash

python3 main_parallel.py \
  --data custom:./data/dataset_10episodes.pt \
  --action_space 6 \
  --img_size 64 \
  --num_steps 32 \
  --bs 6 \
  --nfilterG 32 \
  --num_gpu 1 \
  --warm_up 16 \
  --warmup_decay_epoch 100 \
  --do_memory False \
  --num_components 1 \
  --config_temporal 24 \
  --save_epoch 20 \
  --seed 111111 \
  --end_bias 0.5
