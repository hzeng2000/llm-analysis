import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

# 示例数据
data = {
    "model_size": [7, 13, 30, 65],  # 模型大小（单位：B）
    "gpu_num": [8, 16, 32, 64],     # GPU数量
    "training_delay": [100, 200, 400, 800]  # 训练延迟（单位：秒）
}

# 将数据转换为numpy数组
model_size = np.array(data["model_size"])
gpu_num = np.array(data["gpu_num"])
training_delay = np.array(data["training_delay"])

# 构建贝叶斯回归模型
with pm.Model() as model:
    # 定义共享变量
    model_size_shared = pm.Data('model_size', model_size)
    gpu_num_shared = pm.Data('gpu_num', gpu_num)
    
    # 定义先验分布
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_size = pm.Normal('beta_size', mu=0, sigma=10)
    beta_gpu = pm.Normal('beta_gpu', mu=0, sigma=10)
    beta_size_sq = pm.Normal('beta_size_sq', mu=0, sigma=10)
    beta_gpu_sq = pm.Normal('beta_gpu_sq', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # 定义多项式模型
    mu = alpha + beta_size * model_size_shared + beta_gpu * gpu_num_shared + \
         beta_size_sq * model_size_shared**2 + beta_gpu_sq * gpu_num_shared**2
    
    # 定义似然函数
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=training_delay)
    
    # 进行MCMC采样
    trace = pm.sample(2000, tune=2000, cores=1, target_accept=0.9)

# 查看模型参数的后验分布
az.plot_trace(trace)
plt.show()

# 预测新的模型大小和GPU数量下的训练延迟
new_model_size = np.array([7])
new_gpu_num = np.array([16, 32, 64, 128])
new_model_size_expanded = np.tile(new_model_size, len(new_gpu_num))
with model:
    # 更新共享变量的值
    pm.set_data({'model_size': new_model_size_expanded, 'gpu_num': new_gpu_num})
    
    # 构建预测模型
    posterior_predictive = pm.sample_posterior_predictive(trace, samples=2000)

# 提取预测结果
predicted_delay = posterior_predictive['likelihood'].mean(axis=0)

# 输出预测结果
for gpu, delay in zip(new_gpu_num, predicted_delay):
    print(f"Predicted training delay for model size 7B with {gpu} GPUs: {delay:.2f} seconds")