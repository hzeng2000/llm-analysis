#!/bin/bash

# 定义参数列表
# total_gpus=(8 16 32 64)
# model_sizes=(7 13 30 65)
# gpu_names=("a100-pcie-40gb" "a100-sxm-80gb" "a100-sxm-40gb" "h100-sxm-80gb" "mi100-32gb" "mi250-128gb" "v100-sxm-32gb")
gpu_names=("a100-sxm-80gb" "a100-sxm-40gb" "h100-sxm-80gb")
for ((i=0;i<=2;i+=1));
do
    bash /home/hzeng/prj/llm-analysis/examples/llama/model_num_gpus.sh  ${gpu_names[i]}
done
