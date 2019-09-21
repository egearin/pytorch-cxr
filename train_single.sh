#!/bin/bash -l

datamode="iid_max_single"
dataset="stanford"
mode="per_study"
desc="custom"
runtime_dir="20190920_${datamode}_${dataset}_${mode}_${desc}"

rm -rf runtime/$runtime_dir

python train.py \
  --cuda 0 \
  --runtime-dir $runtime_dir \
  --tensorboard \
