#!/bin/bash

# 定义参数列表
model_sizes=(7 13 30 65 130 260 520 1040 2080)
total_gpus=(8 16 32 64 128 256 512 1024)
# gpu_name="a100-sxm-80gb"
for ((i=0;i<=8;i+=1));
do
for ((j=0;j<=7;j+=1))
do
    echo "${model_sizes[i]} ${total_gpus[j]} "
    # bash /home/hzeng/prj/llm-analysis/examples/llama/qi_train.sh ${model_sizes[i]} ${total_gpus[j]} ${gpu_name}
    bash /home/hzeng/prj/llm-analysis/examples/llama/qi_train.sh ${model_sizes[i]} ${total_gpus[j]}
done
done
