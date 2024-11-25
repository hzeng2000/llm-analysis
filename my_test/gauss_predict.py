import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 读取GPU配置文件函数
def load_gpu_config(gpu_name, config_path="/home/hzeng/prj/llm-analysis/llm_analysis/gpu_configs"):
    file_path = f"{config_path}/{gpu_name}.json"
    with open(file_path, 'r') as f:
        gpu_data = json.load(f)
    return gpu_data
# 模型配置读取
def load_model_config(model_name, config_path="/home/hzeng/prj/llm-analysis/llm_analysis/model_configs"):
    file_path = f"{config_path}/{model_name}.json"
    with open(file_path, 'r') as f:
        model_data = json.load(f)
    return model_data
# 预处理数据
def preprocess_data(raw_data, gpu_config_path, model_config_path):
    X, y = [], []
    for entry in raw_data:
        model_name, gpu_name, num_gpus, delay, *params = entry
        gpu_config = load_gpu_config(gpu_name, gpu_config_path)
        # 提取GPU相关特征
        gpu_features = [
            gpu_config["mem_per_GPU_in_GB"],
            gpu_config["hbm_bandwidth_in_GB_per_sec"],
            gpu_config["peak_fp16_TFLOPS"],
            gpu_config["intra_node_bandwidth_in_GB_per_sec"]
        ]
        # 加载模型配置
        model_config = load_model_config(model_name, model_config_path)
        model_features = [
            model_config["param_count"],
            model_config["num_layers"],
            model_config["hidden_dim"]
        ]
        # 合并所有特征
        features = [num_gpus] + gpu_features + model_features
        X.append(features)
        y.append(delay)
    return np.array(X), np.array(y)

# 初始化高斯过程回归模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

def train_model(X, y):
    gpr.fit(X, y)

def predict(data_point):
    y_pred, sigma = gpr.predict(data_point.reshape(1, -1), return_std=True)
    return y_pred[0], sigma[0]

# 示例数据处理和训练
raw_data = [
    ["decapoda-research_llama-7b-hf", "a100-pcie-40gb", 8, 92.34454805553614, 4, 1, 1, 2, 1, 1],
    ["decapoda-research_llama-7b-hf", "a100-sxm-80gb", 8, 92.1530651775333, 2, 1, 2, 2, 1, 1],
    ["decapoda-research_llama-7b-hf", "a100-sxm-40gb", 8, 92.34454805553614, 8, 1, 1, 1, 1, 1],
    ["decapoda-research_llama-7b-hf", "h100-sxm-80gb", 8, 29.25191111386631, 8, 1, 1, 1, 1, 1],
    ["decapoda-research_llama-7b-hf", "mi100-32gb", 8, 155.73271233989874, 1, 1, 4, 2, 1, 1],
    ["decapoda-research_llama-7b-hf", "mi250-128gb", 8, 79.26472376676084, 2, 1, 1, 4, 1, 1],
    ["decapoda-research_llama-7b-hf", "v100-sxm-32gb", 8, 229.87228170117626, 1, 1, 2, 4, 1, 1],
    ["decapoda-research_llama-13b-hf", "a100-pcie-40gb", 16, 71.26732352048644, 2, 1, 8, 1, 8, 1],
    ["decapoda-research_llama-13b-hf", "a100-sxm-80gb", 16, 71.14301081772697, 4, 1, 1, 4, 1, 1],
    ["decapoda-research_llama-13b-hf", "a100-sxm-40gb", 16, 71.26732352048644, 2, 1, 2, 4, 1, 1],
    ["decapoda-research_llama-13b-hf", "h100-sxm-80gb", 16, 22.560594529785018, 1, 1, 2, 8, 1, 1],
    ["decapoda-research_llama-13b-hf", "mi100-32gb", 16, 120.22940775749724, 2, 1, 1, 8, 1, 1],
    ["decapoda-research_llama-13b-hf", "mi250-128gb", 16, 61.21004954251783, 8, 1, 1, 2, 1, 1],
    ["decapoda-research_llama-13b-hf", "v100-sxm-32gb", 16, 177.48091708362497, 2, 1, 8, 1, 8, 1],
    ["decapoda-research_llama-30b-hf", "a100-pcie-40gb", 32, 59.564678660470626, 2, 1, 2, 8, 1, 1],
    ["decapoda-research_llama-30b-hf", "a100-sxm-80gb", 32, 59.472771451739895, 2, 1, 2, 8, 1, 1],
    ["decapoda-research_llama-65b-hf", "a100-sxm-80gb", 32, 133.80137341666662, 2, 1, 4, 4, 1, 1],
    ["decapoda-research_llama-65b-hf", "a100-sxm-40gb", 32, 133.9640320729317, 1, 1, 8, 4, 1, 1],
    ["decapoda-research_llama-65b-hf", "h100-sxm-80gb", 32, 42.363559021325614, 2, 1, 8, 2, 1, 1]
]
test_data = [
    ["decapoda-research_llama-65b-hf", "a100-sxm-80gb", 64, 59.472771451739895, 2, 1, 2, 8, 1, 1]
]

if __name__ == "__main__":
    scaler = StandardScaler()
    # 数据预处理
    gpu_config_path = "/home/hzeng/prj/llm-analysis/llm_analysis/gpu_configs"
    model_config_path = "/home/hzeng/prj/llm-analysis/llm_analysis/model_configs"
    X, y = preprocess_data(raw_data, gpu_config_path, model_config_path)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    train_model(X_train, y_train)

    # 新数据预测
    x_test, _ = preprocess_data(test_data, gpu_config_path, model_config_path)
    pred, uncertainty = predict(x_test)
    print(f"预测值: {pred:.2f}, 不确定性: {uncertainty:.2f}")
