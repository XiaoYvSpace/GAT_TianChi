import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np
import torch

seed = 123

# 设置Python随机数种子
random.seed(seed)

# 设置NumPy随机数种子
np.random.seed(seed)

# 设置PyTorch随机数种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
    #def __init__(self, in_features, out_features, alpha, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.alpha = alpha
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout):
        super(GraphAttentionNetwork, self).__init__()
        self.hidden_features = hidden_features
        self.num_heads = num_heads

        self.attentions = nn.ModuleList([GraphAttentionLayer(in_features, hidden_features, dropout=dropout) for _ in range(num_heads)])

        self.out_att = GraphAttentionLayer(num_heads*hidden_features, out_features, dropout=dropout)

    def forward(self, input, adj):
        x = input
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        return F.elu(x)

# 载入数据集
data = pd.read_csv('boruta_selected_features.csv')

# 提取特征列和目标列
data_features = data.iloc[:, :-1].values
data_labels = data.iloc[:, -1].values
adj = torch.Tensor(pd.read_csv('adj.csv', header=None).values)

#  定义模型参数
num_nodes = data_features.shape[0]
num_features = data_features.shape[1]
hidden_features = 32
num_heads = 4
dropout = 0.5
num_classes = 1

# 创建图注意力网络模型
model = GraphAttentionNetwork(num_features, hidden_features, num_classes, num_heads, dropout)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 转换数据为Tensor
data_features = torch.Tensor(data_features)
data_labels = torch.Tensor(data_labels).view(-1, 1)

# 模型训练
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(data_features, adj)
    loss = criterion(output, data_labels)
    loss.backward()
    optimizer.step()
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss.item()))

# 进行多轮训练
num_epochs = 100
for epoch in range(num_epochs):
    train(epoch)