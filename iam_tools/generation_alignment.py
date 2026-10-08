"""Soft TrInk-inspired alignment intervention, NOT TrInk or original InkDiT.

TrInk (EMNLP2025,section2.2/2.3) learns PE *amplitudes* and uses TRAIN-average
points/character for the Gaussian center. Here queries are packed8-point blocks;
three heads receive a finite log-Gaussian prior, one remains global for delayed
marks. No exact character boundaries, source trajectories or target-derived
alignment enter generation. This is an intentionally partial, controlled bundle.
"""
import math
import torch
from .generation_coverage import WriterTextDenoiser
from .latent_diffusion import positions


def gaussian_bias(text_mask, length, heads, blocks_per_character, *, sigma=2.5, cap=12., dtype=torch.float32):
    if text_mask.ndim!=2 or text_mask.dtype!=torch.bool or not text_mask[:,0].all():raise ValueError('B,S mask with always-valid BOS required')
    if type(length)!=int or length<1 or type(heads)!=int or heads<2 or not math.isfinite(blocks_per_character) or blocks_per_character<=0:raise ValueError('positive dimensions/TRAIN ratio required')
    if not math.isfinite(sigma) or sigma<=0 or not math.isfinite(cap) or cap<=0:raise ValueError('finite positive prior settings required')
    b,s=text_mask.shape;counts=text_mask[:,1:].sum(1)
    center=(torch.arange(length,device=text_mask.device,dtype=dtype)/blocks_per_character)[None].expand(b,-1)
    center=torch.minimum(center,(counts-1).clamp_min(0).to(dtype)[:,None])
    keys=torch.arange(s-1,device=text_mask.device,dtype=dtype)
    local=(-.5*((keys[None,None]-center[:,:,None])/sigma).square()).clamp_min(-cap)
    bias=torch.zeros(b,heads,length,s,device=text_mask.device,dtype=dtype)
    bias[:,:heads-1,:,1:]=local[:,None]
    # Do not give unconditional BOS a perpetual advantage over local characters.
    bias[:,:heads-1,:,0]=torch.where(counts[:,None,None]>0,-2.,0.).to(dtype)
    bias.masked_fill_(~text_mask[:,None,None],float('-inf'))
    return bias.reshape(b*heads,length,s)


class AlignedWriterDenoiser(WriterTextDenoiser):
    def __init__(self,alignment=False,blocks_per_character=1.,prior_sigma=2.5,prior_cap=12.,**config):
        if type(alignment)!=bool or not math.isfinite(blocks_per_character) or blocks_per_character<=0:raise ValueError('explicit alignment flag/positive TRAIN ratio required')
        super().__init__(**config)
        self.text_pe_scale=torch.nn.Parameter(torch.ones(()));self.ink_pe_scale=torch.nn.Parameter(torch.ones(()))
        self.alignment=alignment;self.blocks_per_character=float(blocks_per_character);self.prior_sigma=float(prior_sigma);self.prior_cap=float(prior_cap)
        if not math.isfinite(prior_sigma) or prior_sigma<=0 or not math.isfinite(prior_cap) or prior_cap<=0:raise ValueError('positive finite Gaussian parameters required')
        self.config=dict(self.config,alignment=alignment,blocks_per_character=blocks_per_character,prior_sigma=prior_sigma,prior_cap=prior_cap)
    def forward(self,x,time,text,mask,drop_text=None,writer_ids=None):
        # Exact original behavior for the global arm; added scalar state is unused.
        if not self.alignment:return super().forward(x,time,text,mask,drop_text,writer_ids)
        if writer_ids is None or writer_ids.shape!=(len(x),) or writer_ids.dtype!=torch.long:raise ValueError('explicit B-long writer IDs required')
        width=self.config['width'];b,l,_=x.shape
        drop=torch.zeros(b,device=x.device,dtype=torch.bool) if drop_text is None else drop_text
        tokens=torch.cat((torch.ones(b,1,dtype=torch.long,device=x.device),torch.where(text>=0,text+2,0)),1)
        text_mask=tokens!=0;text_mask[:,1:] &= ~drop[:,None];tokens[:,1:]=torch.where(drop[:,None],0,tokens[:,1:])
        memory=(self.text(tokens)+self.text_pe_scale*positions(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype),width)[None]).masked_fill(~text_mask[...,None],0.)
        p=torch.arange(l,device=x.device,dtype=x.dtype);relative=p[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
        x=self.project(x.masked_fill(~mask[...,None],0.))+self.ink_pe_scale*(positions(p,width)[None]+positions(relative*100,width))
        x=x+self.time(positions(time.to(x.dtype)*1000,width))[:,None]+self.writer(writer_ids)[:,None]
        x=x.masked_fill(~mask[...,None],0.)
        bias=gaussian_bias(text_mask,l,self.config['heads'],self.blocks_per_character,sigma=self.prior_sigma,cap=self.prior_cap,dtype=x.dtype)
        for block in self.blocks:
            h=block.norms[0](x);x=x+block.self_attention(h,h,h,key_padding_mask=~mask,need_weights=False)[0]
            h=block.norms[1](x);x=x+block.cross_attention(h,memory,memory,attn_mask=bias,need_weights=False)[0]
            x=(x+block.ff(block.norms[2](x))).masked_fill(~mask[...,None],0.)
        return self.final(x).masked_fill(~mask[...,None],0.)
