#!/bin/bash

# 定义参数列表
# model_sizes=(7 13 30 65 130 260 520 1040 2080)
model_sizes=(7 13)
# total_gpus=(8 16 32 64 128 256 512 1024)
total_gpus=(8 16 32 64 128)
# gpu_names=("a100-pcie-40gb" "a100-sxm-80gb" "a100-sxm-40gb" "h100-sxm-80gb" "mi100-32gb" "mi250-128gb" "v100-sxm-32gb")
gpu_names=("a100-sxm-80gb")
# base_output_dir="/home/hzeng/prj/llm-analysis/results"
for model_size in "${model_sizes[@]}"; do
    for total_gpu in "${total_gpus[@]}"; do
        for gpu_name in "${gpu_names[@]}"; do
            echo "${model_size} ${total_gpu} ${gpu_name}" 
            bash /home/hzeng/prj/llm-analysis/examples/llama/model_num_name.sh ${model_size} ${total_gpu} ${gpu_name}
        done
    done
done