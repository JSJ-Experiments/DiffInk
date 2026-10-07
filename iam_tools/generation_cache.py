"""Tiny bounded pool: vectorizedGPUgathers, exact minimal batch padding preserved."""
import torch
from .generation_study import collate

class CachedLatentPool:
    def __init__(self,latents,records,vocab,ids,stats,device='cuda',max_lines=256):
        if not ids or len(ids)>max_lines or len(set(ids))!=len(ids):raise ValueError('bounded unique tiny pool required')
        self.ids=list(ids);self.index={s:i for i,s in enumerate(ids)};self.latent_lengths={s:len(latents[s]) for s in ids};self.text_lengths={s:len(records[s]['text']) for s in ids}
        self.clean,self.mask,self.text=collate(latents,records,vocab,ids,stats,device)
        counts=torch.tensor([records[s]['points'] for s in ids],device=device)
        self.point_mask=torch.arange(self.clean.shape[1]*8,device=device)[None]<counts[:,None]
    def select(self,ids):
        if not ids:raise ValueError('nonempty minibatch required')
        index=torch.tensor([self.index[s] for s in ids],device=self.clean.device)
        length=max(self.latent_lengths[s] for s in ids);chars=max(self.text_lengths[s] for s in ids)
        return (self.clean.index_select(0,index)[:,:length],self.mask.index_select(0,index)[:,:length],self.text.index_select(0,index)[:,:chars],self.point_mask.index_select(0,index)[:,:length*8])
