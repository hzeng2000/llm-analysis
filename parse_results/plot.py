import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# hidden_size = [64, 128, 256, 512, 1024]
# layer = [0.5, 1, 2, 4, 8]
# ratio = [128, 128, 128, 256, 512]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(hidden_size, layer, ratio, c=ratio, cmap='viridis')
# ax.set_xlabel('hidden_size')
# ax.set_ylabel('layer')
# ax.set_zlabel('ratio')

# # 添加平面表示比值为128
# xx, yy = np.meshgrid(hidden_size, layer)
# z = 128 * np.ones_like(xx)
# ax.plot_surface(xx, yy, z, alpha=0.5, color='r')

# plt.show()
models = [
    {"Model": "Llama-2-7b-hf", "Model_size(b)": 7, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "Llama-2-13b-hf", "Model_size(b)": 13, "hidden_layer": 40, "hidden_size": 5120, "ratio": 128},
    {"Model": "Llama-2-70b-hf", "Model_size(b)": 70, "hidden_layer": 80, "hidden_size": 8192, "ratio": 102.4},
    {"Model": "Llama-3.1-8B", "Model_size(b)": 8, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "Llama-3.1-70B", "Model_size(b)": 70, "hidden_layer": 80, "hidden_size": 8192, "ratio": 102.4},
    {"Model": "Llama-3.1-405B", "Model_size(b)": 405, "hidden_layer": 126, "hidden_size": 16384, "ratio": 130.031746},
    {"Model": "Qwen2.5-7B", "Model_size(b)": 7, "hidden_layer": 28, "hidden_size": 3584, "ratio": 128},
    {"Model": "Qwen2.5-14B", "Model_size(b)": 14, "hidden_layer": 48, "hidden_size": 5120, "ratio": 106.6666667},
    {"Model": "Qwen2.5-32B", "Model_size(b)": 32, "hidden_layer": 64, "hidden_size": 5120, "ratio": 80},
    {"Model": "Qwen2.5-72B", "Model_size(b)": 72, "hidden_layer": 80, "hidden_size": 8192, "ratio": 102.4},
    {"Model": "opt-6.7b", "Model_size(b)": 6.7, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "opt-13b", "Model_size(b)": 13, "hidden_layer": 40, "hidden_size": 5120, "ratio": 128},
    {"Model": "opt-30b", "Model_size(b)": 30, "hidden_layer": 48, "hidden_size": 7168, "ratio": 149.3333333},
    {"Model": "opt-66b", "Model_size(b)": 66, "hidden_layer": 64, "hidden_size": 9216, "ratio": 144},
    {"Model": "opt-175b", "Model_size(b)": 175, "hidden_layer": 96, "hidden_size": 12288, "ratio": 128},
    {"Model": "APUS-xDAN-4.0-MOE-136B", "Model_size(b)": 136, "hidden_layer": 60, "hidden_size": 7168, "ratio": 119.4666667},
    {"Model": "XVERSE-MoE-A36B", "Model_size(b)": 136, "hidden_layer": 50, "hidden_size": 6144, "ratio": 122.88},
    {"Model": "openbuddy-mixtral-22bx8-65k", "Model_size(b)": 65, "hidden_layer": 56, "hidden_size": 6144, "ratio": 109.7142857},
    {"Model": "GPT-J 6B", "Model_size(b)": 6, "hidden_layer": 28, "hidden_size": 4096, "ratio": 146.2857143},
    {"Model": "glm-4-9b", "Model_size(b)": 9, "hidden_layer": 40, "hidden_size": 4096, "ratio": 102.4},
    {"Model": "telechat-7B", "Model_size(b)": 7, "hidden_layer": 30, "hidden_size": 4096, "ratio": 136.5333333},
    {"Model": "TeleChat-12B", "Model_size(b)": 12, "hidden_layer": 38, "hidden_size": 5120, "ratio": 134.7368421},
    {"Model": "TeleChat-52B", "Model_size(b)": 52, "hidden_layer": 64, "hidden_size": 8192, "ratio": 128},
    {"Model": "TeleChat2-115B", "Model_size(b)": 115, "hidden_layer": 96, "hidden_size": 8192, "ratio": 85.33333333},
    {"Model": "gpt-neox-20b", "Model_size(b)": 20, "hidden_layer": 44, "hidden_size": 6144, "ratio": 139.6363636},
    {"Model": "pygmalion-2-7b", "Model_size(b)": 7, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "pygmalion-2-13b", "Model_size(b)": 13, "hidden_layer": 40, "hidden_size": 5120, "ratio": 128},
    {"Model": "bloom-176b", "Model_size(b)": 176, "hidden_layer": 70, "hidden_size": 14336, "ratio": 204.8},
    {"Model": "cpm-bee-10b", "Model_size(b)": 10, "hidden_layer": 48, "hidden_size": 4096, "ratio": 85.33333333},
    {"Model": "BioMedGPT-LM-7B", "Model_size(b)": 7, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "chatglm2-6b", "Model_size(b)": 6, "hidden_layer": 28, "hidden_size": 4096, "ratio": 146.2857143},
    {"Model": "Baichuan2-7B", "Model_size(b)": 7, "hidden_layer": 32, "hidden_size": 4096, "ratio": 128},
    {"Model": "Baichuan2-13B", "Model_size(b)": 13, "hidden_layer": 40, "hidden_size": 5120, "ratio": 128},
    {"Model": "falcon-7b", "Model_size(b)": 7, "hidden_layer": 64, "hidden_size": 4096, "ratio": 64},
    {"Model": "falcon-11B", "Model_size(b)": 11, "hidden_layer": 60, "hidden_size": 4096, "ratio": 68.26666667},
    {"Model": "falcon-180B", "Model_size(b)": 180, "hidden_layer": 80, "hidden_size": 14848, "ratio": 185.6},
    {"Model": "falcon-40b", "Model_size(b)": 40, "hidden_layer": 60, "hidden_size": 8192, "ratio": 136.5333333}
]

import numpy as np
from adjustText import adjust_text
# hidden_size = [64, 128, 256, 1024, 1024, 1600]
# layer = [0.5, 1, 2, 4, 8, 16]
# ratio = [128, 128, 128, 256, 128, 100]
hidden_size = [model["hidden_size"] for model in models]
layer = [model["hidden_layer"] for model in models]
ratio = [model["ratio"] for model in models]
model_name = [model["Model"] for model in models]
# 绘制散点图
scatter = plt.scatter(layer, hidden_size, c=ratio, cmap='viridis')
# plt.axhline(y=128, color='r', linestyle='--')  # 添加水平线表示比值为128
hidden_size_range = np.linspace(min(hidden_size), max(hidden_size), 100)
layer_line = hidden_size_range / 128  
plt.plot(layer_line, hidden_size_range, color='r', linestyle='--')
texts = []
for i, txt in enumerate(hidden_size):
    texts.append(plt.text(layer[i], hidden_size[i], model_name[i], ha='center'))

# 调整注释位置
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
plt.ylabel('hidden_size')
plt.xlabel('layer')
plt.colorbar(label='ratio')
plt.show()