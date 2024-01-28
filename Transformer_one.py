import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F
import random
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

seed = 1234  # 设置随机数种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Mlp(nn.Module):
    def __init__(self, in_features, dropout, hidden_features=None,hidden_dim=41, act_layer=nn.GELU, pred=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.model_attention = MultiheadAttention(in_features=in_features,dropout=dropout,
                               num_heads=3,head_dim=19)
        self.act = act_layer()
        self.pred = pred
        #self.norm = nn.BatchNorm1d(hidden_dim)
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x,adj):
        x0 = x
        x=self.model_attention(x,adj)
        x1 = x0 + x
        #x = self.norm(x1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x2 = x1 + x
            #x = self.norm(x2)
        x = x.squeeze(0)
        return x


class TF(nn.Module):
    def size(self):
        # 定义size方法，返回结果和预期相同
        return torch.Size([])

    def __init__(self, in_features, drop=0.0):
        super().__init__()
        #self.Block1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block1_1 = Mlp(in_features=in_features, dropout=drop,  hidden_features=64, act_layer=nn.GELU, pred=False)
        self.Block1_2 = Mlp(in_features=in_features, dropout=drop, hidden_features=64, act_layer=nn.GELU,pred=False)
        # self.Block1_3 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, dropout=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, dropout=drop, pred=False)
        self.Block2_1 = Mlp(in_features=in_features, dropout=drop, hidden_features=64, act_layer=nn.GELU, pred=False)
        self.Block2_2 = Mlp(in_features=in_features, dropout=drop, hidden_features=64, act_layer=nn.GELU, pred=True)
        #self.Block2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=True)

    def forward(self, x , adj):
        #return self.Block2_3(self.Block2_2(self.Block2_1(self.Block1_3(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj),adj),adj)
        #return self.Block2_4(self.Block2_3(self.Block2_2(self.Block2_1(self.Block1_4(self.Block1_3(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj),adj),adj),adj),adj)
        return self.Block2_2(self.Block2_1(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj)
        #return self.Block2(self.Block1(x,adj),adj)



class MultiheadAttention(nn.Module):
    def __init__(self, in_features, num_heads, head_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_linear = nn.Linear(in_features, num_heads * head_dim)
        self.k_linear = nn.Linear(in_features, num_heads * head_dim)
        self.v_linear = nn.Linear(in_features, num_heads * head_dim)

        self.fc = nn.Linear(num_heads * head_dim, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,adj):
        seq_length,in_features= x.size()

        q = self.q_linear(x).view(seq_length, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_linear(x).view(seq_length, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_linear(x).view(seq_length, self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        zero_vec = -9e15*torch.ones_like(attn_weights)
        attn_weights = torch.where(adj > 0, attn_weights, zero_vec)

        # # 将attn_weights分为三个[1, 800, 800]的矩阵
        # attn_weights_1 = attn_weights[:1]
        # attn_weights_2 = attn_weights[1:2]
        # attn_weights_3 = attn_weights[2:]

        # # 针对adj矩阵的0和1位置，更新对应的attn_weights的数值
        # attn_weights_1[adj == 0] = -9e15
        # attn_weights_2[adj == 0] = -9e15
        # attn_weights_3[adj == 0] = -9e15
        # attn_weights = torch.cat([attn_weights_1, attn_weights_2, attn_weights_3], dim=0)

        x = (attn_weights @ v).transpose(1, 2).contiguous().view(seq_length,self.num_heads * self.head_dim)
        x = self.fc(x)

        return x
    
if __name__ == '__main__':
    import pandas as pd

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 加载你的CSV文件
    data = pd.read_csv('boruta_selected_features.csv')

    # 提取特征和目标
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    max_vals = np.max(X, axis=0)
    min_vals = np.min(X, axis=0)
    # 归一化每列的值到0到1之间
    X = (X - min_vals) / (max_vals - min_vals)

    #adj = torch.Tensor(pd.read_csv('adj.csv', header=None).values).to(device)

    # 加载 adj.csv 文件为 pandas DataFrame
    adj_df = pd.read_csv('adj.csv', header=None)

    # 转换为 torch Tensor，并添加一个维度
    #adj = torch.unsqueeze(torch.Tensor(adj_df.values), 0).to(device)


    # 将adj_df转换为NumPy数组
    adj_df = adj_df.values

    # 使用np.newaxis创建一个新的维度，将adj矩阵从[800, 800]变为[1, 800, 800]
    adj_expanded = adj_df[np.newaxis, :, :]

    # 通过np.repeat函数复制adj矩阵两次，维度变为[3, 800, 800]
    adj = torch.Tensor(np.repeat(adj_expanded, 3, axis=0)).to(device)


    # 定义自定义数据集类
    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.Tensor(X).to(device)
            self.y = torch.Tensor(y).to(device)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # 初始化模型、损失函数和优化器
    in_features=X.shape[1]
    model = TF(in_features)
    model.to(device)
    loss_fn = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建数据集和数据加载器
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(dataset,  batch_size=len(X), shuffle=True)

    # 训练模型
    num_epochs = 200
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs,adj)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss.item()))

    # 计算预测值

    # y_pred = model(torch.Tensor(X)).detach().numpy()
    #print(y_pred)

    # 保存训练好的模型
    torch.save(model.state_dict(), "transformer_1.h5")

    # # 计算 MSE
    # mse = mean_squared_error(y, y_pred)
    # print("Mean Squared Error:", mse)