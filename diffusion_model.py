"""Original compact 1D U-Net reconstruction of STABLE's CDG and MLG.

Architecture, cosine schedule, and optimizer are explicit reconstruction choices.
The macro-only and full-condition branches share all denoiser parameters.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, channels, cond_dim=64):
        super().__init__()
        self.norm1=nn.GroupNorm(8,channels)
        self.conv1=nn.Conv1d(channels,channels,3,padding=1)
        self.condition=nn.Linear(cond_dim,channels)
        self.norm2=nn.GroupNorm(8,channels)
        self.conv2=nn.Conv1d(channels,channels,3,padding=1)

    def forward(self,x,h):
        z=self.conv1(F.silu(self.norm1(x)))+self.condition(h)[:,:,None]
        return x+self.conv2(F.silu(self.norm2(z)))


class ConditionalUNet(nn.Module):
    def __init__(self,macro_dim,stock_dim,dim=16,width=32,gate_cap=2.):
        super().__init__()
        self.gate_cap=gate_cap
        self.macro=nn.Linear(macro_dim,dim)
        self.stock=nn.Linear(stock_dim,dim)
        self.gate=nn.Linear(2*dim,1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.cond=nn.Sequential(nn.Linear(2*dim,64),nn.SiLU(),nn.Linear(64,64))
        self.time=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,64))
        self.input=nn.Conv1d(1,width,3,padding=1)
        self.r1=Residual(width)
        self.down=nn.Conv1d(width,width*2,4,stride=2,padding=1)
        self.r2=Residual(width*2)
        self.down2=nn.Conv1d(width*2,width*2,4,stride=2,padding=1)
        self.mid=Residual(width*2)
        self.up2=nn.Conv1d(width*4,width*2,3,padding=1)
        self.r3=Residual(width*2)
        self.up=nn.Conv1d(width*3,width,3,padding=1)
        self.r4=Residual(width)
        self.out=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,1,3,padding=1))

    def denoise(self,x,t,c):
        f=torch.exp(torch.arange(16,device=x.device)*(-math.log(10000)/15))
        a=t.float()[:,None]*f[None,:]
        h=self.time(torch.cat([a.sin(),a.cos()],dim=1))+self.cond(c)
        a=self.r1(self.input(x[:,None,:]),h)
        b=self.r2(self.down(a),h)
        c=self.mid(self.down2(b),h)
        c=F.interpolate(c,size=b.shape[-1],mode='nearest')
        b=self.r3(self.up2(torch.cat([c,b],dim=1)),h)
        b=F.interpolate(b,size=a.shape[-1],mode='nearest')
        return self.out(self.r4(self.up(torch.cat([b,a],dim=1)),h))[:,0,:]

    def forward(self,x,t,macro,stock,gate_override=None):
        hm=self.macro(macro); hc=self.stock(stock)
        full=torch.cat([hm,hc],dim=1)
        macro_only=torch.cat([hm,torch.zeros_like(hc)],dim=1)
        out=self.denoise(torch.cat([x,x]),torch.cat([t,t]),torch.cat([macro_only,full]))
        em,ef=out.chunk(2)
        gate=self.gate_cap*torch.sigmoid(self.gate(full))
        if gate_override is not None: gate=torch.full_like(gate,gate_override)
        return em+gate*(ef-em)


def cosine_schedule(steps=200):
    t=torch.arange(steps+1,dtype=torch.float64)/steps
    a=torch.cos((t+.008)/1.008*math.pi/2).square()
    beta=(1-a[1:]/a[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta,dim=0).float()


@torch.no_grad()
def sample(model,macro,stock,initial_noise,schedule,steps=30,eta=0.,generator=None):
    """DDIM sampling with a fixed, reported clean-state clipping threshold."""
    x=initial_noise.clone()
    times=torch.linspace(len(schedule)-1,0,steps).round().long().tolist()
    clip_count=0
    for i,t in enumerate(times):
        at=schedule[t]
        ap=schedule[times[i+1]] if i+1<len(times) else x.new_tensor(1.)
        e=model(x,torch.full((len(x),),t,device=x.device,dtype=torch.long),macro,stock)
        clean=(x-(1-at).sqrt()*e)/at.sqrt()
        clip_count+=(clean.abs()>8).sum().item()
        clean=clean.clamp(-8,8)
        sigma=eta*torch.sqrt(((1-ap)/(1-at)*(1-at/ap)).clamp_min(0))
        x=ap.sqrt()*clean+(1-ap-sigma*sigma).clamp_min(0).sqrt()*e
        if eta:
            x=x+sigma*torch.randn(x.shape,device=x.device,generator=generator)
    return x,clip_count/(len(times)*initial_noise.numel())
