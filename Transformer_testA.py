import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Transformer_one import Mlp, TF
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载验证集数据集
validation_data = pd.read_csv('boruta_selected_testA_fill.csv')
validation_answer = pd.read_csv('dataset_answerA.csv', names=['ID', 'Value'])

# 提取验证集特征和目标值
X_val = validation_data.iloc[:, :].values
y_val = validation_answer.iloc[:, -1].values

max_vals = np.max(X_val, axis=0)
min_vals = np.min(X_val, axis=0)
# 归一化每列的值到0到1之间
X_val = (X_val - min_vals) / (max_vals - min_vals)

adj_df = pd.read_csv('testA_adj.csv', header=None)

# 将adj_df转换为NumPy数组
adj_df = adj_df.values

# 使用np.newaxis创建一个新的维度，将adj矩阵从[800, 800]变为[1, 800, 800]
adj_expanded = adj_df[np.newaxis, :, :]

# 通过np.repeat函数复制adj矩阵两次，维度变为[3, 800, 800]
adj = torch.Tensor(np.repeat(adj_expanded, 3, axis=0)).to(device)

# 定义模型并加载保存的模型
model = TF(in_features=X_val.shape[1])
model=model.to(device)
model.load_state_dict(torch.load("transformer_1.h5"))

# 创建验证集数据集和数据加载器
val_dataset = CustomDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)

# 将模型设置为评估模式
model.eval()

# 进行验证集数据的预测
y_pred_val = []
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs,adj)
        y_pred_val.extend(outputs.cpu().detach().numpy())
y_pred_val = np.array(y_pred_val)
print(y_pred_val)


# 计算验证集的MSE
mse_val = mean_squared_error(y_val, y_pred_val)
print("Validation Mean Squared Error:", mse_val)