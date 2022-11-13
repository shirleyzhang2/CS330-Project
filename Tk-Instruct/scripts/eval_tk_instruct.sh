#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/violetyao/.cache/huggingface

python src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
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
    --data_dir ../eval \
#    --task_dir ../tasks \
    --task_dir ../gpt3-tasks \
    --output_dir output/textual_entailment/tk-instruct-small-def-pos/ \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4
