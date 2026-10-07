"""CPU CTC prefix-beam diagnostic: sum paths, no lexicon/language model.

All alphabet columns are considered (no token pruning). Beam width bounds kept
prefixes, so this is approximate search except exhaustive-width small tests.
Changing this OCR readout never changes the handwriting decoder or training.
"""
import math
import numpy as np


def _add(a,b):
    if a<b:a,b=b,a
    return a if b==-math.inf else a+math.log1p(math.exp(b-a))


def prefix_beam(log_probs,vocab,width=10):
    p=np.asarray(log_probs,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=len(vocab)+1 or not isinstance(width,int) or isinstance(width,bool) or not 1<=width<=100:
        raise ValueError('T×(alphabet+blank) log probabilities and width1–100 required')
    if len(set(vocab))!=len(vocab) or any(not isinstance(c,str) or len(c)!=1 for c in vocab):raise ValueError('unique single-character alphabet required')
    if np.isnan(p).any() or np.isposinf(p).any() or not np.allclose(np.exp(p).sum(1),1,atol=1e-5,rtol=1e-5):raise ValueError('normalized log probabilities required')
    beams={():[0.,-math.inf]}
    for frame in p:
        following={}
        def accumulate(prefix,state,value):
            if value==-math.inf:return
            row=following.setdefault(prefix,[-math.inf,-math.inf]);row[state]=_add(row[state],value)
        for prefix,(blank,nonblank) in beams.items():
            total=_add(blank,nonblank);accumulate(prefix,0,total+float(frame[0]))
            for token in range(1,len(vocab)+1):
                value=float(frame[token])
                if prefix and token==prefix[-1]:
                    accumulate(prefix,1,nonblank+value)
                    accumulate(prefix+(token,),1,blank+value)
                else:accumulate(prefix+(token,),1,total+value)
        beams=dict(sorted(following.items(),key=lambda x:(-_add(*x[1]),x[0]))[:width])
    best,score=min(beams.items(),key=lambda x:(-_add(*x[1]),x[0]))
    return dict(text=''.join(vocab[t-1] for t in best),log_probability=_add(*score),width=width,
                search='CTC prefix beam, all tokens, no LM/lexicon; approximate prefix pruning')


def audit_head(head,cache,texts,groups,vocab,width=10):
    """Mean-only eval on declared DEV/report; same batched logits for both readers."""
    import torch
    from .frozen_ocr_study import collate_latents
    from .inkvae import greedy_ctc,edit_distance
    ids=sorted(set(i for values in groups.values() for i in values),key=lambda i:(cache[i]['mu'].shape[-1],i))
    rows=[];was=head.training;head.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(ids),16):
                batch=ids[start:start+16];z,_,mask=collate_latents(cache,batch);logits=head(z,padding_mask=~mask)
                for j,sid in enumerate(batch):
                    values=logits[:int(mask[j].sum()),j].cpu();greedy=greedy_ctc(values.argmax(-1).tolist(),vocab)
                    result=prefix_beam(values.log_softmax(-1).numpy(),vocab,width);beam=result['text']
                    rows.append(dict(sample_id=sid,text=texts[sid],characters=len(texts[sid]),greedy=greedy,beam=beam,
                        greedy_errors=edit_distance(texts[sid],greedy),beam_errors=edit_distance(texts[sid],beam),beam_log_probability=result['log_probability']))
    finally:head.train(was)
    by_id={r['sample_id']:r for r in rows};summaries={}
    for group,group_ids in groups.items():
        selected=[by_id[i] for i in group_ids];chars=sum(r['characters'] for r in selected)
        summaries[group]=dict(lines=len(selected),greedy_cer=sum(r['greedy_errors'] for r in selected)/chars,beam_cer=sum(r['beam_errors'] for r in selected)/chars,
            improved=sum(r['beam_errors']<r['greedy_errors'] for r in selected),same=sum(r['beam_errors']==r['greedy_errors'] for r in selected),worsened=sum(r['beam_errors']>r['greedy_errors'] for r in selected),
            beam_exact_lines=sum(r['beam_errors']==0 for r in selected))
    return dict(width=width,no_language_model=True,no_lexicon=True,no_checkpoint_selection_using_beam=True,
        latent='mu only, no sampled-z beam evaluation',logits='raw head logits, stable log_softmax; NOT loss-only [-30,30] clamp',groups=summaries,lines=rows)
