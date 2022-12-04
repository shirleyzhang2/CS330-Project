#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/zhan4532/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port src/run_s2s.py \
    --do_train \
    --do_eval \
    --model_name_or_path allenai/tk-instruct-small-def-pos \
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
    --task_dir ../gpt3-paraphrase-tasks-tk-instruct-train \
    --output_dir ../output/finetune_v1 \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 5000 \
    --deepspeed ds_configs/stage2.config \
    --run_name tk-finetune-experiment-v1
