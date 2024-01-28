import torch.nn as nn
from GAT import TF
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

validation_data = pd.read_csv('boruta_selected_testA_fill.csv')
validation_answer = pd.read_csv('dataset_answerA.csv', names=['ID', 'Value'])

# 提取验证集特征和目标值
X_val = validation_data.iloc[:, :41].values
y_val = validation_answer.iloc[:, -1].values
adj = torch.Tensor(pd.read_csv('testA_adj.csv', header=None).values).to(device)

max_vals = np.max(X_val, axis=0)
min_vals = np.min(X_val, axis=0)
# 归一化每列的值到0到1之间
X_val = (X_val - min_vals) / (max_vals - min_vals)

in_features = X_val.shape[1]
# 创建模型对象
model = TF(in_features).to(device)
#model = torch.jit.load('trained_model.pt')

# 定义模型并加载保存的模型
model.load_state_dict(torch.load("GAT_1.h5"))

# 加载模型参数
#model.load_state_dict(torch.load('trained_model.pt'))

# 设置模型为评估模式
model.eval()
# # 使用模型进行推断
# output = model(X_val, adj)

# mse_val = mean_squared_error(y_val, output)
# print("Validation Mean Squared Error:", mse_val)

# 使用模型进行推断
with torch.no_grad():
    output = model(torch.Tensor(X_val).to(device),adj).cpu()

# 将输出结果转换为一维数组
output = output.squeeze().numpy()

mse_val = mean_squared_error(y_val, output)
print("Validation Mean Squared Error:", mse_val)