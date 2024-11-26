import os
import json
import re



def extract_info(subdir_name):
    pattern = re.compile(r'results_(?P<model_name>[^_]+_[^_]+)_(?P<total_num_gpus>\d+)gpus_(?P<gpu_name>[^_]+)')
    match = pattern.match(subdir_name)
    if match:
        return match.group('model_name'), match.group('total_num_gpus'), match.group('gpu_name')
    return None, None, None

# 定义根目录
root_dir = '/home/hzeng/prj/llm-analysis/results'

# 初始化有效和无效数据的计数器
valid_count = 0
invalid_count = 0

# 初始化一个列表来存储所有有效的JSON数据
valid_data = []

# 遍历根目录及其子文件夹
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(subdir, file)
            
            # 读取JSON文件
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 检查是否为有效数据
                if data['max_batch_size_per_gpu'] > data['batch_size_per_gpu']:
                    valid_count += 1
                    
                    # 提取子文件夹名的三个信息
                    subdir_name = os.path.basename(subdir)
                    model_name, total_num_gpus, gpu_name = extract_info(subdir_name)
                    
                    if model_name and total_num_gpus and gpu_name:
                        # 添加子文件夹名的三个信息到数据中
                        data['model_name'] = model_name
                        data['total_num_gpus'] = int(total_num_gpus)
                        data['gpu_name'] = gpu_name
                        
                        # 将有效数据添加到列表中
                        valid_data.append(data)
                        invalid_count += 1
                else :
                    invalid_count += 1

# 将所有有效数据整合成一个大的JSON文件
output_file = 'combined_results.json'
with open(output_file, 'w') as f:
    json.dump(valid_data, f, indent=4)

# 输出有效和无效数据的个数
print(f"Valid data count: {valid_count}")
print(f"Invalid data count: {invalid_count}")