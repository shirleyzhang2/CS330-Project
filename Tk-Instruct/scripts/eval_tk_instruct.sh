#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/zhan4532/.cache/huggingface

python src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path ../output/finetune_v5/checkpoint-15000 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ../splits \
    --task_dir ../tk-instruct-train-classfication-tasks \
    --output_dir output/finetuned-v5-original-instruction \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4
