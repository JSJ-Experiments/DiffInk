"""Text-coverage isolation with explicit learned writer IDs, NOT released InkDiT."""
import hashlib
from collections import Counter
import torch
from .latent_diffusion import TextLatentDenoiser,positions
from .ocr_pool import normalized_text,round_robin


def select_coverage(manifest,original,broad_size=1024,writer_limit=32):
    records=manifest['records'];small=list(original['train']);held=list(original['unseen_prompt'])
    families={records[i]['prompt_family'] for i in held};texts={normalized_text(records[i]['text']) for i in held}
    blocked=set(manifest['test_writers'])|set(manifest['dev_writers'])
    safe=[i for i in manifest['splits']['large_train'] if records[i]['writer_id'] not in blocked and records[i]['prompt_family'] not in families and normalized_text(records[i]['text']) not in texts]
    if len(small)!=32 or len(held)!=8 or not set(small)<=set(safe):raise ValueError('pinned32/8 scope and safe train required')
    writer=records[small[0]]['writer_id']
    if any(records[i]['writer_id']!=writer for i in small+held):raise ValueError('original scope must be singlewriter')
    same=sorted(i for i in safe if records[i]['writer_id']==writer)
    counts=Counter(records[i]['writer_id'] for i in safe)
    writers=[writer]+[w for w in sorted(counts,key=lambda w:(-counts[w],w)) if w!=writer][:writer_limit-1]
    broad=list(small)+[i for i in same if i not in small]
    remaining=[dict(id=i,writer_id=records[i]['writer_id']) for i in safe if records[i]['writer_id'] in writers and i not in broad]
    broad+= [r['id'] for r in round_robin(remaining,seed=8142)][:broad_size-len(broad)]
    if len(broad)!=broad_size or len(set(broad))!=len(broad):raise ValueError('insufficient bounded broader data')
    if not set(same)<=set(broad) or set(broad)&set(held):raise ValueError('nested train/held guard')
    writers=sorted({records[i]['writer_id'] for i in broad});added=[i for i in broad if i not in same]
    probe=sorted(added,key=lambda i:hashlib.sha256(('probe:'+i).encode()).hexdigest())[:24]
    return dict(arms={'small32':small,'writer_all':same,'broad1024':broad},evaluation={'retained_train':small,'added_same_writer': [i for i in same if i not in small],'added_other_writers':probe,'unseen_prompt':held},writers=writers,
                checks=dict(held_forms_globally_excluded=True,held_normalized_texts_globally_excluded=True,reserved_writers_excluded=True,nested_train=True),coverage_counts={a:dict(lines=len(v),writers=len({records[i]['writer_id'] for i in v}),unique_texts=len({normalized_text(records[i]['text']) for i in v})) for a,v in [('small32',small),('writer_all',same),('broad1024',broad)]})


class WriterTextDenoiser(TextLatentDenoiser):
    """Add zero-initialized writer bias. No trajectory/reference input; no core edits."""
    def __init__(self,writer_count,**config):
        if type(writer_count)!=int or writer_count<1:raise ValueError('positive writer vocabulary required')
        super().__init__(**config);self.writer=torch.nn.Embedding(writer_count,self.config['width']);torch.nn.init.zeros_(self.writer.weight)
        self.config=dict(self.config,writer_count=writer_count)
    def forward(self,x,time,text,mask,drop_text=None,writer_ids=None):
        if writer_ids is None or writer_ids.shape!=(len(x),) or writer_ids.dtype!=torch.long:raise ValueError('explicit B-long writer IDs required')
        width=self.config['width'];b,l,_=x.shape
        drop=torch.zeros(b,device=x.device,dtype=torch.bool) if drop_text is None else drop_text
        tokens=torch.cat((torch.ones(b,1,dtype=torch.long,device=x.device),torch.where(text>=0,text+2,0)),1)
        text_mask=tokens!=0;text_mask[:,1:] &= ~drop[:,None];tokens[:,1:]=torch.where(drop[:,None],0,tokens[:,1:])
        memory=(self.text(tokens)+positions(torch.arange(tokens.shape[1],device=x.device,dtype=x.dtype),width)[None]).masked_fill(~text_mask[...,None],0.)
        p=torch.arange(l,device=x.device,dtype=x.dtype);relative=p[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
        x=self.project(x.masked_fill(~mask[...,None],0.))+positions(p,width)[None]+positions(relative*100,width)
        x=x+self.time(positions(time.to(x.dtype)*1000,width))[:,None]+self.writer(writer_ids)[:,None]
        x=x.masked_fill(~mask[...,None],0.)
        for block in self.blocks:x=block(x,memory,mask,text_mask)
        return self.final(x).masked_fill(~mask[...,None],0.)
