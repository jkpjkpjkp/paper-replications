"""Independent PyTorch reconstruction; no author implementation was found."""
import torch
from torch import nn

class HGA(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.conv=nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(3)])
        self.gate=nn.ModuleList([nn.Linear(2*width,1) for _ in range(3)])
    def forward(self,h,adj):
        out=[]
        for a,w,g in zip(adj,self.conv,self.gate):
            deg=a.sum(-1).clamp_min(1)
            norm=a*deg.rsqrt()[:,None]*deg.rsqrt()[None,:]
            neighbor=torch.relu(norm@w(h))
            m=torch.cat((neighbor,h),dim=-1)
            out.append(torch.sigmoid(g(m))*m)
        return sum(out)

class Model(nn.Module):
    def __init__(self,features,hidden=100,head=(64,32),dropout=.2,graph_layers=0):
        super().__init__()
        self.encoder=nn.LSTM(features,hidden,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        width=2*hidden
        self.graph=nn.ModuleList()
        for _ in range(graph_layers):
            self.graph.append(HGA(width));width*=2
        def task():return nn.Sequential(nn.Linear(width,head[0]),nn.ReLU(),nn.Linear(head[0],head[1]),nn.ReLU(),nn.Linear(head[1],1))
        self.return_head=task();self.risk_head=task()
    def forward(self,x,adj=None):
        # x: batch of dates, stocks, lookback, input features.
        b,n,t,f=x.shape
        _,(hidden,_)=self.encoder(x.reshape(b*n,t,f))
        h=self.dropout(torch.cat((hidden[-2],hidden[-1]),dim=-1)).reshape(b,n,-1)
        if len(self.graph) and adj is None:raise ValueError('Typed adjacency is required; missing graphs must not be fabricated')
        for layer in self.graph:h=layer(h,adj)
        return torch.cat((self.return_head(h),self.risk_head(h)),dim=-1),h
