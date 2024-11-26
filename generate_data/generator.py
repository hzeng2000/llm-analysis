import os
import subprocess
import json

# 定义参数列表
model_sizes = [7, 13]
total_gpus = [8, 16, 32, 64, 128]
gpu_names = ["a100-sxm-80gb"]

# 定义常量
global_batch_size = 2048
seq_len = 1024
max_seq_len = 2048
total_num_tokens = 32768
flops_efficiency = 0.5
hbm_memory_efficiency = 0.9
base_output_dir = "/home/hzeng/prj/llm-analysis/results"

def run_analysis(model_size, total_num_gpus, gpu_name):
    model_name = f"decapoda-research_llama-{model_size}b-hf"
    output_dir = os.path.join(base_output_dir, f"results_{model_name}_{total_num_gpus}gpus_{gpu_name}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        print(f"{output_dir} already exists but is not a directory")
        return

    for tp_size in [1, 2, 4, 8]:
        max_pp = total_num_gpus // tp_size
        for pp_size in [2**i for i in range(int(max_pp).bit_length()) if 2**i <= max_pp]:
            for sp_size in [1, tp_size]:
                for act_rec in [0]:
                    max_batch = global_batch_size // total_num_gpus
                    for micro_batch in [2**i for i in range(1, int(max_batch).bit_length())]:
                        command = [
                            "python3", "/home/hzeng/prj/llm-analysis/llm_analysis/analysis.py", "train",
                            "--model_name", model_name,
                            "--gpu_name", gpu_name,
                            "--activation_recomputation", str(act_rec),
                            "--tp_size", str(tp_size),
                            "--pp_size", str(pp_size),
                            "--sp_size", str(sp_size),
                            "--global_batch_size", str(global_batch_size),
                            "--total_num_tokens", str(total_num_tokens),
                            "--seq_len", str(seq_len),
                            "--total_num_gpus", str(total_num_gpus),
                            "--flops_efficiency", str(flops_efficiency),
                            "--hbm_memory_efficiency", str(hbm_memory_efficiency),
                            "--output_dir", output_dir,
                            "--batch_size_per_gpu", str(micro_batch),
                            "--log_level", "WARNING",
                            # "--output_file_suffix", str(micro_batch)
                        ]
                        
                        print("Running command:")
                        print(" ".join(command))
                        
                        result = subprocess.run(command, capture_output=True, text=True)
                        output = result.stdout
                        print(output)
                        
                        try:
                            output_json = json.loads(output)
                            if output_json.get("max_batch_size_per_gpu", 0) < output_json.get("batch_size_per_gpu", 0):
                                print("batch_size_per_gpu is bigger than max_batch_size_per_gpu, skipping further micro_batch iterations.")
                                break
                        except json.JSONDecodeError:
                            print("Failed to parse JSON output.")

for model_size in model_sizes:
    for total_gpu in total_gpus:
        for gpu_name in gpu_names:
            print(f"{model_size} {total_gpu} {gpu_name}")
            run_analysis(model_size, total_gpu, gpu_name)