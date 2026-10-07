"""Small full-text cross-attention x0 diffusion diagnostic, NOT released InkDiT.

Frozen polyphase40 codec, all384 channels modeled. No prefix/trajectory teacher
conditioning. Masks and latent whitening are explicit; NULL never has zero keys.
"""
import math
import torch
from torch import nn


def positions(p,width):
    f=torch.exp(torch.arange(width//2,device=p.device,dtype=p.dtype)*(-math.log(10000)/(width//2-1)))
    v=p[...,None]*f
    return torch.cat((v.sin(),v.cos()),-1)


class CrossBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.self_attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,memory,mask,text_mask):
        h=self.norms[0](x);x=x+self.self_attention(h,h,h,key_padding_mask=~mask,need_weights=False)[0]
        h=self.norms[1](x);x=x+self.cross_attention(h,memory,memory,key_padding_mask=~text_mask,need_weights=False)[0]
        return (x+self.ff(self.norms[2](x))).masked_fill(~mask[...,None],0.)


class TextLatentDenoiser(nn.Module):
    def __init__(self,channels=384,vocab_size=81,width=128,depth=4,heads=4):
        super().__init__()
        if width<8 or width%heads or width%2 or channels<1 or depth<1:raise ValueError('valid dimensions required')
        self.config=dict(channels=channels,vocab_size=vocab_size,width=width,depth=depth,heads=heads)
        self.project=nn.Linear(channels,width);self.text=nn.Embedding(vocab_size+2,width,padding_idx=0)
        self.time=nn.Sequential(nn.Linear(width,width*4),nn.SiLU(),nn.Linear(width*4,width))
        self.blocks=nn.ModuleList([CrossBlock(width,heads) for _ in range(depth)])
        self.final=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,channels))
        nn.init.zeros_(self.final[-1].weight);nn.init.zeros_(self.final[-1].bias)
    def forward(self,x,time,text,mask,drop_text=None):
        # Input guard outside training hot loop: mask is Boolean B,L, text right
        # padded -1. No character truncation to latent length and no reference z.
        width=self.config['width'];b,l,_=x.shape
        drop=torch.zeros(b,device=x.device,dtype=torch.bool) if drop_text is None else drop_text
        tokens=torch.cat((torch.ones(b,1,dtype=torch.long,device=x.device),torch.where(text>=0,text+2,0)),1)
        text_mask=tokens!=0;text_mask[:,1:] &= ~drop[:,None];tokens[:,1:]=torch.where(drop[:,None],0,tokens[:,1:])
        memory=(self.text(tokens)+positions(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype),width)[None]).masked_fill(~text_mask[...,None],0.)
        p=torch.arange(l,device=x.device,dtype=x.dtype)
        relative=p[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
        x=self.project(x.masked_fill(~mask[...,None],0.))+positions(p,width)[None]+positions(relative*100,width)
        x=x+self.time(positions(time.to(x.dtype)*1000,width))[:,None]
        x=x.masked_fill(~mask[...,None],0.)
        for block in self.blocks:x=block(x,memory,mask,text_mask)
        return self.final(x).masked_fill(~mask[...,None],0.)


def fit_whitening(latents,train_ids,floor=.1):
    if not train_ids or len(set(train_ids))!=len(train_ids) or not 0<floor<=1:raise ValueError('unique TRAIN IDs and bounded positive floor required')
    values=torch.cat([latents[i] for i in train_ids],0).double()
    if values.ndim!=2 or not torch.isfinite(values).all():raise ValueError('finite L,C TRAIN means required')
    mean=values.mean(0).float();std=values.std(0,unbiased=False).clamp_min(floor).float()
    return dict(mean=mean,std=std,floor=floor,train_ids=list(train_ids),points=int(values.shape[0]),definition='TRAIN-only channel mean/population std, floor0.1, all channels; no latent channel masking')


def transform(x,stats,inverse=False):
    return x*stats['std']+stats['mean'] if inverse else (x-stats['mean'])/stats['std']


def masked_mse(pred,target,mask):
    if pred.shape!=target.shape or pred.ndim!=3 or mask.shape!=pred.shape[:2]:raise ValueError('matching B,L,C and B,L required')
    residual=torch.where(mask[...,None],pred-target,0.)
    return residual.square().sum()/(mask.sum().clamp_min(1)*pred.shape[-1])


def cosine_schedule(steps=1000,device='cpu'):
    if steps<2:raise ValueError('at least two diffusion steps')
    a=torch.cos((torch.arange(steps+1,device=device,dtype=torch.float32)/steps+.008)/1.008*math.pi*.5).square();a/=a[0].clone()
    beta=(1-a[1:]/a[:-1]).clamp(.0001,.9999)
    return (1-beta).cumprod(0)


def forward_noise(clean,epsilon,t,alpha,mask):
    a=alpha[t][:,None,None];return (a.sqrt()*clean+(1-a).sqrt()*epsilon).masked_fill(~mask[...,None],0.)


@torch.no_grad()
def ddim_sample(model,noise,text,mask,alpha,steps=50,guidance=1.,drop_text=False):
    """Pure-noise start. Only text + oracle latent length; NO target/prefix input.

    Predict x0; deterministic eta0 DDIM. Text CFG explicitly drops TEXT, unlike
    upstream prefix-only CFG when no handwriting prefix is provided.
    """
    if not 2<=steps<=len(alpha) or guidance<0 or model.training:raise ValueError('bounded eval-mode sampling required')
    if mask.dtype!=torch.bool or mask.shape!=noise.shape[:2] or not mask.any(1).all():raise ValueError('nonempty Boolean latent masks required')
    times=torch.linspace(len(alpha)-1,0,steps,device='cpu').round().long().tolist();times.append(-1)
    x=noise.clone().masked_fill(~mask[...,None],0.);drop=torch.full((x.shape[0],),drop_text,device=x.device,dtype=torch.bool)
    for current,next_time in zip(times[:-1],times[1:]):
        t=torch.full((x.shape[0],),current/(len(alpha)-1),device=x.device)
        start=model(x,t,text,mask,drop)
        if guidance!=1 and not drop_text:
            null=model(x,t,text,mask,torch.ones_like(drop));start=null+guidance*(start-null)
        if next_time<0:x=start;continue
        a,b=alpha[current],alpha[next_time];eps=(x-a.sqrt()*start)/(1-a).clamp_min(1e-12).sqrt()
        x=(b.sqrt()*start+(1-b).sqrt()*eps).masked_fill(~mask[...,None],0.)
    if not torch.isfinite(x).all():raise FloatingPointError('nonfinite unconditional-start sample')
    return x
