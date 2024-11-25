from math import inf
import os
import json

def find_min_latency_file(directory):
    min_latency = inf
    min_latency_file = None
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                latency_per_iter = data.get('latency_per_iter', -1)
                dp_size = data.get('dp_size', -1)
                rdp_size = data.get('rdp_size', -1)
                tp_size = data.get('tp_size', -1)
                pp_size = data.get('pp_size', -1)
                sp_size = data.get('sp_size', -1)
                ep_size = data.get('ep_size', -1)
                # print(latency_per_iter)
                if latency_per_iter < min_latency:
                    min_latency = latency_per_iter
                    min_latency_file = filename
    
    return min_latency_file, min_latency, dp_size, rdp_size, tp_size, pp_size, sp_size, ep_size

def main():
    model_sizes=[7, 13, 30, 65, 130, 260, 520, 1040, 2080]
    total_gpus=[8, 16, 32, 64, 128, 256, 512, 1024]
    gpu_names = ["a100-pcie-40gb", "a100-sxm-80gb", "a100-sxm-40gb", "h100-sxm-80gb", "mi100-32gb", "mi250-128gb", "v100-sxm-32gb"]
    result = []
    
    for i in range(len(total_gpus)):
        for gpu_name in gpu_names:
            for model_size in model_sizes:
                # model_size = model_sizes[i]
                model_name=f'decapoda-research_llama-{model_size}b-hf'
                # gpu_name = "a100-sxm-80gb"
                output_dir=f"results_{model_name}{total_gpus[i]}gpus_{gpu_name}_gb"
                # output_dir=f"results_{model_name}{total_gpus[i]}gpus"
                # base_dir = "/home/hzeng/prj/llm-analysis/results_3xsxm"
                # base_dir = "/home/hzeng/prj/llm-analysis/results_3xsxm"
                base_dir = "/home/hzeng/prj/llm-analysis/results"
                dir_path = os.path.join(base_dir, output_dir)
                if os.path.isdir(dir_path):
                    # print(f"Directory '{dir_path}' exists.")
                    min_latency_file, min_latency, dp_size, rdp_size, tp_size, pp_size, sp_size, ep_size = find_min_latency_file(dir_path)
                    if min_latency_file:
                        result.append((model_name, gpu_name, f"{total_gpus[i]}gpus", min_latency, dp_size, rdp_size, tp_size, pp_size, sp_size, ep_size))
                else:
                    print(f"Directory '{dir_path}' does not exist.")
    
    for model_name, gpu_name, total_gpus, min_latency, dp_size, rdp_size, tp_size, pp_size, sp_size, ep_size in result:
        print(f"{model_name}, {gpu_name}, {total_gpus}, {min_latency}, {dp_size}, {rdp_size}, {tp_size}, {pp_size}, {sp_size}, {ep_size}")

if __name__ == "__main__":
    main()