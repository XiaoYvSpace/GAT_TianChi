import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np
import torch

seed = 1234

# 设置Python随机数种子
random.seed(seed)

# 设置NumPy随机数种子
np.random.seed(seed)

# 设置PyTorch随机数种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 测试git

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
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout, hidden_dim,act_layer=nn.GELU, pred=True):
        super(GraphAttentionNetwork, self).__init__()
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([GraphAttentionLayer(in_features, hidden_features, dropout=dropout) for _ in range(num_heads)])
        self.out_att = GraphAttentionLayer(num_heads*hidden_features, out_features, dropout=dropout)
        #self.norm = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, adj):
        x0 = input
        x = torch.cat([att(x0, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
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

    def __init__(self, in_features,hidden_features,num_heads,dropout):
        super().__init__()
        #self.Block1 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block1_1 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block1_2 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block1_3 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block1_4 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block2_1 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block2_2 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block2_3 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=False)
        self.Block2_4 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=True)
        #self.Block2 = GraphAttentionNetwork(in_features=in_features, hidden_features=hidden_features, out_features=41,num_heads=num_heads,dropout=dropout,hidden_dim=41,act_layer=nn.GELU, pred=True)

    def forward(self, x , adj):
        #return self.Block2_3(self.Block2_2(self.Block2_1(self.Block1_3(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj),adj),adj)
        return self.Block2_4(self.Block2_3(self.Block2_2(self.Block2_1(self.Block1_4(self.Block1_3(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj),adj),adj),adj),adj)
        #return self.Block2_2(self.Block2_1(self.Block1_2(self.Block1_1(x,adj),adj),adj),adj)
        #return self.Block2(self.Block1(x,adj),adj)

if __name__ =='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # 载入数据集
    data = pd.read_csv('boruta_selected_features.csv')

    # 提取特征列和目标列
    data_features = data.iloc[:, :-1].values
    data_labels = data.iloc[:, -1].values
    adj = torch.Tensor(pd.read_csv('adj.csv', header=None).values).to(device)

    max_vals = np.max(data_features, axis=0)
    min_vals = np.min(data_features, axis=0)
    # 归一化每列的值到0到1之间
    data_features = (data_features - min_vals) / (max_vals - min_vals)

    #  定义模型参数
    in_features=data_features.shape[1]
    hidden_features = 64
    num_heads = 3
    dropout = 0.0

    # 创建图注意力网络模型
    #model = GraphAttentionNetwork(in_features, hidden_features, num_classes, num_heads, dropout)
    model = TF(in_features,hidden_features,num_heads,dropout)
    model.to(device)
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 转换数据为Tensor
    data_features = torch.Tensor(data_features).to(device)
    data_labels = torch.Tensor(data_labels).view(-1, 1).to(device)

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
    num_epochs = 400
    for epoch in range(num_epochs):
        train(epoch)

    # model_path = 'trained_model.pt'
    # torch.save(model.state_dict(), model_path)

    # model_path = 'trained_model.pt'
    # torch.jit.save(torch.jit.script(model), model_path)

    # 保存训练好的模型
    torch.save(model.state_dict(), "GAT_1.h5")