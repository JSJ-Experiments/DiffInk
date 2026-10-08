"""Soft local text hint, NOT target character boundaries or smoothing.

Linear interpolation of requested character embeddings along oracle-duration
fraction. This is an approximate inductive bias; global cross-attention remains
unrestricted, important for delayed dots/crossbars and nonuniform stroke spacing.
"""
import torch
from .generation_coverage import WriterTextDenoiser
from .latent_diffusion import positions


def local_text_hint(embedding,text,mask):
    if text.ndim!=2 or mask.ndim!=2 or len(text)!=len(mask) or mask.dtype!=torch.bool:raise ValueError('B,Stext/B,Lbool mask required')
    lengths=(text>=0).sum(1)
    if not (lengths>0).all() or not mask.any(1).all():raise ValueError('nonempty text and duration required')
    # Never truncate text to latent length; interpolate all real characters.
    tokens=torch.where(text>=0,text+2,0);memory=embedding(tokens);b,l=mask.shape
    fraction=torch.arange(l,device=mask.device)[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
    index=(fraction*(lengths-1)[:,None]).clamp_min(0);left=index.floor().long().clamp_max(text.shape[1]-1);right=(left+1).clamp_max(text.shape[1]-1)
    left=torch.minimum(left,(lengths-1)[:,None]);right=torch.minimum(right,(lengths-1)[:,None]);mix=(index-index.floor())[...,None]
    rows=torch.arange(b,device=text.device)[:,None];hint=memory[rows,left]*(1-mix)+memory[rows,right]*mix
    return hint.masked_fill(~mask[...,None],0.)


class AnchoredWriterDenoiser(WriterTextDenoiser):
    def __init__(self,local_text_scale=0.,**config):
        if not 0<=local_text_scale<=.25:raise ValueError('small bounded local text hint required')
        super().__init__(**config);self.local_text_scale=float(local_text_scale);self.config=dict(self.config,local_text_scale=self.local_text_scale)
    def forward(self,x,time,text,mask,drop_text=None,writer_ids=None):
        if self.local_text_scale==0:return super().forward(x,time,text,mask,drop_text,writer_ids)
        if writer_ids is None or writer_ids.shape!=(len(x),) or writer_ids.dtype!=torch.long:raise ValueError('explicit B-long writer IDs required')
        width=self.config['width'];b,l,_=x.shape;drop=torch.zeros(b,device=x.device,dtype=torch.bool) if drop_text is None else drop_text
        tokens=torch.cat((torch.ones(b,1,dtype=torch.long,device=x.device),torch.where(text>=0,text+2,0)),1)
        text_mask=tokens!=0;text_mask[:,1:] &= ~drop[:,None];tokens[:,1:]=torch.where(drop[:,None],0,tokens[:,1:])
        memory=(self.text(tokens)+positions(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype),width)[None]).masked_fill(~text_mask[...,None],0.)
        p=torch.arange(l,device=x.device,dtype=x.dtype);relative=p[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
        x=self.project(x.masked_fill(~mask[...,None],0.))+positions(p,width)[None]+positions(relative*100,width)
        x=x+self.time(positions(time.to(x.dtype)*1000,width))[:,None]+self.writer(writer_ids)[:,None]
        hint=local_text_hint(self.text,text,mask).masked_fill(drop[:,None,None],0.);x=x+self.local_text_scale*hint
        x=x.masked_fill(~mask[...,None],0.)
        for block in self.blocks:x=block(x,memory,mask,text_mask)
        return self.final(x).masked_fill(~mask[...,None],0.)
