"""Set-Sequence equations (1)-(3), with a causal LongConv backbone.

LongConv is adapted from Elliotepsteino/safari, commit
b385a6ab0a6a9fdd2fbf8810e10e2d68000599a9 (Apache-2.0; LICENSE-safari).
Changes: compact dependencies, explicit causal FFT padding, float32 FFT under
autocast, and [stock,time,channel] layout. Set layers are our reconstruction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LongConv(nn.Module):
    def __init__(self, dim, length=30, channels=2, threshold=.003):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(channels,dim,length)*.002)
        self.skip = nn.Parameter(torch.randn(channels,dim))
        self.mix = nn.Linear(channels*dim,2*dim)
        self.threshold = threshold

    def convolve(self, x):
        # Linear convolution, never circular wrap-around or centered padding.
        u=x.transpose(1,2).float()
        k=self.kernel.float()
        k=torch.sign(k)*F.relu(k.abs()-self.threshold)
        n=1 << (u.shape[-1]+k.shape[-1]-2).bit_length()
        uf=torch.fft.rfft(u,n=n)
        kf=torch.fft.rfft(k,n=n)
        y=torch.fft.irfft(uf[:,None]*kf[None],n=n)[...,:u.shape[-1]]
        y=y+u[:,None]*self.skip[None,:,:,None]
        return y.flatten(1,2).transpose(1,2).to(x.dtype)

    def forward(self,x):
        return F.glu(self.mix(F.gelu(self.convolve(x))),dim=-1)


def ffn(input_dim,hidden,output_dim,dropout):
    return nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,output_dim))


class Block(nn.Module):
    def __init__(self,cfg,use_set):
        super().__init__()
        d=cfg['hidden_dim'];h=cfg['ffn_hidden_dim'];p=cfg['dropout']
        self.use_set=use_set
        self.chunk=cfg['chunk_size']
        if use_set:
            self.phi=ffn(d*self.chunk,h,cfg['phi_dim'],p)
            self.rho=ffn(cfg['phi_dim'],h,cfg['summary_dim'],p)
            self.psi=ffn(d+cfg['summary_dim'],h,d,p)
        self.seq=LongConv(d,cfg['kernel_length'],cfg['kernel_channels'],cfg['kernel_threshold'])
        self.drop=nn.Dropout(p)
        self.norm=nn.LayerNorm(d)

    def forward(self,x,return_summary=False):
        z=x;summary=None
        if self.use_set:
            # Exactly three observations [t-2,t-1,t], padded with zero on the left.
            chunks=[F.pad(x,(0,0,lag,0))[:,:x.shape[1]] for lag in range(self.chunk-1,-1,-1)]
            embedded=self.phi(torch.cat(chunks,dim=-1))
            summary=self.rho(embedded.mean(dim=0,keepdim=True))
            z=self.psi(torch.cat([x,summary.expand(x.shape[0],-1,-1)],dim=-1))
        y=self.norm(x+self.drop(self.seq(z)))
        return (y,summary) if return_summary else y


class PortfolioModel(nn.Module):
    def __init__(self,cfg,model='set_sequence'):
        super().__init__()
        if model not in ('set_sequence','longconv'):
            raise ValueError(model)
        self.encoder=nn.Linear(79,cfg['hidden_dim'])
        self.blocks=nn.ModuleList([Block(cfg,model=='set_sequence') for _ in range(cfg['layers'])])
        self.head=nn.Linear(cfg['hidden_dim'],1)
        nn.init.zeros_(self.head.bias)
        self.receptive_history=cfg['layers']*(cfg['kernel_length']-1+(cfg['chunk_size']-1 if model=='set_sequence' else 0))

    def forward(self,x):
        """Input [time,stock,feature], return [time,stock] signed scores."""
        z=self.encoder(x.transpose(0,1))
        for layer in self.blocks:
            z=layer(z)
        return self.head(z).squeeze(-1).transpose(0,1).float()

    def weights(self,x):
        s=self(x)
        denominator=s.abs().sum(dim=-1,keepdim=True).clamp_min(1e-8)
        return s/denominator


def optimizer(model,cfg):
    kernel=[];other=[]
    for n,p in model.named_parameters():
        (kernel if n.endswith('.kernel') else other).append(p)
    return torch.optim.AdamW([
        {'params':kernel,'weight_decay':cfg['kernel_weight_decay']},
        {'params':other,'weight_decay':cfg['other_weight_decay']}],lr=cfg['learning_rate'])


def sharpe_loss(weights,returns):
    r=(weights*returns).sum(-1)
    return -r.mean()/r.std(unbiased=False).clamp_min(1e-6)
