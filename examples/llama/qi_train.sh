#!/bin/bash

model_size=$1
total_num_gpus=$2
# gpu_name=$3
gpu_name="a100-sxm-80gb"
model_name=decapoda-research_llama-${model_size}b-hf
global_batch_size=2048
seq_len=1024
max_seq_len=2048
total_num_tokens=32768

flops_efficiency=0.5
hbm_memory_efficiency=0.9
# output_dir="/home/hzeng/prj/llm-analysis/results/results_"${model_name}""${total_num_gpus}"gpus_"${gpu_name}"_gb"
output_dir="/home/hzeng/prj/llm-analysis/results_a100sxm80g/results_"${model_name}""${total_num_gpus}"gpus"
if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

for ((tp_size=1;tp_size<=8;tp_size*=2))
do
max_pp=$((total_num_gpus / tp_size))
for ((pp_size=1;pp_size<=${max_pp};pp_size*=2))
do
for sp_size in 1 $tp_size
do
for ((act_rec=0;act_rec<=4;act_rec+=1))
do
max_batch=$((global_batch_size / total_num_gpus))
for ((micro_batch=2;micro_batch<$((max_batch));micro_batch*=2))
do
    echo "Running command:"
    echo "python3 /home/hzeng/prj/llm-analysis/llm_analysis/analysis.py train \\
    --model_name ${model_name} \\
    --gpu_name ${gpu_name} \\
    --activation_recomputation ${act_rec} \\
    --tp_size ${tp_size} \\
    --pp_size ${pp_size} \\
    --sp_size ${sp_size} \\
    --global_batch_size ${global_batch_size} \\
    --total_num_tokens ${total_num_tokens} \\
    --seq_len ${seq_len} \\
    --total_num_gpus ${total_num_gpus} \\
    --flops_efficiency ${flops_efficiency} \\
    --hbm_memory_efficiency ${hbm_memory_efficiency} \\
    --output_dir ${output_dir} \\
    --batch_size_per_gpu ${micro_batch} \\
    --log_level \"WARNING\" \\
    --output_file_suffix \"${micro_batch}\""

    python3 /home/hzeng/prj/llm-analysis/llm_analysis/analysis.py train \
    --model_name ${model_name} \
    --gpu_name ${gpu_name} \
    --activation_recomputation ${act_rec} \
    --tp_size ${tp_size} \
    --pp_size ${pp_size} \
    --sp_size ${sp_size} \
    --global_batch_size ${global_batch_size} \
    --total_num_tokens ${total_num_tokens} \
    --seq_len ${seq_len} \
    --total_num_gpus ${total_num_gpus} \
    --flops_efficiency ${flops_efficiency} \
    --hbm_memory_efficiency ${hbm_memory_efficiency} \
    --output_dir ${output_dir} \
    --batch_size_per_gpu ${micro_batch} \
    --log_level "WARNING" \
    --output_file_suffix "\"${micro_batch}\"" \
    | tee output.log 2>&1
done
done
done
done
done

# python3 /home/hzeng/prj/llm-analysis/llm_analysis/analysis.py train \
#     --model_name decapoda-research_llama-7b-hf \
#     --gpu_name a100-pcie-40gb \
#     --activation_recomputation 0 \
#     --tp_size 8 \
#     --pp_size 1 \
#     --sp_size 1 \
#     --global_batch_size 1024 \
#     --total_num_tokens 10240 \
#     --seq_len 20 \
#     --total_num_gpus 32 \
#     --flops_efficiency 0.9 \
#     --hbm_memory_efficiency 0.9 \
#     --output_dir /home/hzeng/prj/llm-analysis/results_128gpus_a100-pcie-40gb_gb16384 \
#     --batch_size_per_gpu 1 \
#     --log_level WARNING \
#     --output_file_suffix 1 
