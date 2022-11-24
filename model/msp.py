import torch
import torch.nn as nn
import numpy as np

class feedForward(nn.Module):
    def __init__(self, dim):
        super(feedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )
        
    def forward(self, x):
        value = torch.relu(self.fc(x)) + x
        return value

class selfAttention(nn.Module):
    def __init__(self, dim, adj):
        super(selfAttention, self).__init__()
        self.qfc = nn.Linear(dim, dim)
        self.kfc = nn.Linear(dim, dim)
        self.vfc = nn.Linear(dim, dim)
        
        self.ff = feedForward(dim)
        
        self.d = dim
        self.adj = adj
        
    def forward(self, x):
        q, k, v = self.qfc(x), self.kfc(x), self.vfc(x)
        corr = torch.matmul(q, k.transpose(-2,-1)) * self.d ** (-0.5)
        corr = torch.softmax(corr+torch.eye(self.adj.shape[0]).to(device)+self.adj, -1)
        out = torch.relu(torch.matmul(corr, v)) + x
        return self.ff(out)

class marketSharePrediction(nn.Module):
    def __init__(self, indim, dim, outdim, adj, dev):
        super(marketSharePrediction, self).__init__()
        global device
        device = dev
        
        self.adj = torch.from_numpy(adj).to(device)
        
        self.input_fc = nn.Linear(indim, dim)
        self.attention = nn.Sequential(
            selfAttention(dim, self.adj),
            selfAttention(dim, self.adj),
            selfAttention(dim, self.adj),
            selfAttention(dim, self.adj),
        )
        self.output_fc = nn.Linear(dim, outdim)
        
    def forward(self, x):
        '''
        x:[batch size, carries, routes, features]
        '''
        x = self.input_fc(x)
        x = self.attention(x)
        x = self.output_fc(x)
        return x.squeeze(-1) / x.squeeze(-1).sum(1).unsqueeze(1)
        