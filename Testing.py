import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from GAT import TF  # 从您的模型模块中导入自定义模型
from sklearn.metrics import mean_squared_error

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 指定使用GPU设备
    print("Running on the GPU")
else:
    device = torch.device("cpu")  # 指定使用CPU设备
    print("Running on the CPU")

# 载入数据集
data = pd.read_csv('boruta_selected_features.csv')

# 提取特征列和目标列
X_train = data.iloc[:, :-1].values
y_train = data.iloc[:, -1].values
adj_train = torch.Tensor(pd.read_csv('adj.csv', header=None).values).to(device)

max_vals = np.max(X_train, axis=0)
min_vals = np.min(X_train, axis=0)
# 归一化每列的值到0到1之间
X_train = (X_train - min_vals) / (max_vals - min_vals)
#  定义模型参数
in_features=X_train.shape[1]

validation_data = pd.read_csv('boruta_selected_testA_fill.csv')
validation_answer = pd.read_csv('dataset_answerA.csv', names=['ID', 'Value'])

# 提取验证集特征和目标值
X_test = validation_data.iloc[:, :41].values
y_test = validation_answer.iloc[:, -1].values
adj_test = torch.Tensor(pd.read_csv('testA_adj.csv', header=None).values).to(device)

max_vals = np.max(X_test, axis=0)
min_vals = np.min(X_test, axis=0)
# 归一化每列的值到0到1之间
X_test = (X_test - min_vals) / (max_vals - min_vals)

num_epochs_range = [50,100, 200, 300, 400, 500]
Ir_range = [ 0.01,0.001,0.005]

parameters = {
    'hidden_features': [16, 32, 64],
    'num_heads': [1,2, 3, 4, 5, 6],
    'dropout': [0.0, 0.001, 0.005],
    'num_epochs': num_epochs_range,
    'Ir': Ir_range
}

results = [] 
# 声明变量以保存最小的mse_val和对应的参数
min_mse_val = float("inf")  # 初始化为正无穷大
best_param = None


# 进行参数网格搜索和训练
for param in ParameterGrid(parameters):
    #  定义模型参数
    in_features=X_train.shape[1]
    num_epochs = param['num_epochs']
    Ir = param['Ir']

    model = TF(in_features, param['hidden_features'], param['num_heads'], param['dropout'])
    model.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=Ir)
    criterion = nn.MSELoss()

    # 转换数据为Tensor
    X_train_tensor = torch.Tensor(X_train).to(device)
    y_train_tensor = torch.Tensor(y_train).view(-1, 1).to(device)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor, adj_train)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        #print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss.item()))

    print("Parameters:", param)

    model.eval()

    # 使用模型进行推断
    with torch.no_grad():
        output = model(torch.Tensor(X_test).to(device),adj_test).cpu()

    output = output.squeeze().numpy()

    if np.isnan(output).any():
        print("Output contains NAN")
    else:
        mse_val = mean_squared_error(y_test, output)
        print("Validation Mean Squared Error:", mse_val)
        result = {'param': param, 'mse_val': mse_val}
        results.append(result)

        if mse_val < min_mse_val:
            min_mse_val = mse_val
            best_param = param

    torch.cuda.empty_cache()

# 输出最小mse_val和对应参数
print("Best Parameters:", best_param)
print("Minimum Validation Mean Squared Error:", min_mse_val)

# for result in results:
#     print("Parameters:", result['param'])
#     print("Validation Mean Squared Error:", result['mse_val'])