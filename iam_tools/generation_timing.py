"""Frozen-checkpoint timing interventions: diagnostics, not usable oracle generation.

position_lengths changes ONLY denominator of relative-progress PE. mask continues
controlling self-attention and emitted sequence. Same trained parameter state.
No architecture/weights/data updates, no implicit smoothing or forced termination.
"""
import torch
from .generation_alignment import AlignedWriterDenoiser,gaussian_bias
from .latent_diffusion import positions


class TimingProbe(AlignedWriterDenoiser):
    def forward(self,x,time,text,mask,drop_text=None,writer_ids=None,position_lengths=None):
        if position_lengths is None:return super().forward(x,time,text,mask,drop_text,writer_ids)
        if position_lengths.shape!=(len(x),) or position_lengths.dtype!=torch.long or (position_lengths<1).any():raise ValueError('positive B-long diagnostic PE lengths required')
        if writer_ids is None or writer_ids.shape!=(len(x),) or writer_ids.dtype!=torch.long:raise ValueError('B-long writer IDs required')
        width=self.config['width'];b,l,_=x.shape
        drop=torch.zeros(b,device=x.device,dtype=torch.bool) if drop_text is None else drop_text
        tokens=torch.cat((torch.ones(b,1,dtype=torch.long,device=x.device),torch.where(text>=0,text+2,0)),1)
        text_mask=tokens!=0;text_mask[:,1:] &= ~drop[:,None];tokens[:,1:]=torch.where(drop[:,None],0,tokens[:,1:])
        alpha_t=self.text_pe_scale if self.alignment else 1.;alpha_i=self.ink_pe_scale if self.alignment else 1.
        memory=(self.text(tokens)+alpha_t*positions(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype),width)[None]).masked_fill(~text_mask[...,None],0.)
        p=torch.arange(l,device=x.device,dtype=x.dtype);relative=p[None]/(position_lengths-1).clamp_min(1)[:,None]
        x=self.project(x.masked_fill(~mask[...,None],0.))+alpha_i*(positions(p,width)[None]+positions(relative*100,width))
        x=x+self.time(positions(time.to(x.dtype)*1000,width))[:,None]+self.writer(writer_ids)[:,None];x=x.masked_fill(~mask[...,None],0.)
        if self.alignment:bias=gaussian_bias(text_mask,l,self.config['heads'],self.blocks_per_character,sigma=self.prior_sigma,cap=self.prior_cap,dtype=x.dtype)
        for block in self.blocks:
            if not self.alignment:x=block(x,memory,mask,text_mask);continue
            h=block.norms[0](x);x=x+block.self_attention(h,h,h,key_padding_mask=~mask,need_weights=False)[0]
            h=block.norms[1](x);x=x+block.cross_attention(h,memory,memory,attn_mask=bias,need_weights=False)[0]
            x=(x+block.ff(block.norms[2](x))).masked_fill(~mask[...,None],0.)
        return self.final(x).masked_fill(~mask[...,None],0.)


def intervention_lengths(oracle,delta,mode):
    if type(oracle)!=int or not 2<=oracle<=255 or delta not in (-1,1):raise ValueError('bounded oracle and ±1block diagnostic required')
    if mode not in ('baseline','pe_only','mask_only','both'):raise ValueError('explicit timing intervention required')
    context=oracle+delta if mode in ('mask_only','both') else oracle
    position=oracle+delta if mode in ('pe_only','both') else oracle
    return context,position
