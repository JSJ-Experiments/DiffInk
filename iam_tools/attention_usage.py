"""Frozen last-layer interventions: correct attention weights != semantic usage.

Keep every preceding layer/text/key/query unchanged. Rotate ONLY character token
embeddings in selected last-layer value heads, keeping BOS/positions fixed, or
zero selected head contexts. This is an ablation, not valid new-text generation.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .weak_alignment import attention_log_probs

MODES=('manual_control','local_token_rotate','global_token_rotate','local_zero','global_zero','all_zero')


def rotated_character_delta(model,text):
    if text.ndim!=2 or text.dtype!=torch.long:raise ValueError('B,S padded integer text required')
    tokens=torch.where(text>=0,text+2,0);rotated=tokens.clone()
    for j in range(len(text)):
        n=int((text[j]>=0).sum())
        if n:rotated[j,:n]=tokens[j,:n].roll(1)
    delta=model.text(rotated)-model.text(tokens)
    return F.pad(delta,(0,0,1,0))  # BOS value unchanged, positional PE unchanged.


def intervened_attention(module,query,key,value,mask,mode,delta):
    if mode not in MODES:raise ValueError('explicit last-layer frozen intervention required')
    b,l,e=query.shape;s=value.shape[1];h=module.num_heads;d=e//h
    p=attention_log_probs(module,query,key,attn_mask=mask).exp()
    w=module.in_proj_weight[2*e:];bias=None if module.in_proj_bias is None else module.in_proj_bias[2*e:]
    v=F.linear(value,w,bias).reshape(b,s,h,d).transpose(1,2);context=p@v
    selected=slice(0,h-1) if mode.startswith('local_') else slice(h-1,h)
    if mode.endswith('token_rotate'):
        if delta is None or delta.shape!=value.shape:raise ValueError('character-embedding-only B,S,E value delta required')
        alt=F.linear(value+delta,w,bias).reshape(b,s,h,d).transpose(1,2)
        context=context.clone();context[:,selected]=p[:,selected]@alt[:,selected]
    elif mode.endswith('zero'):
        context=context.clone()
        context[:,slice(None) if mode=='all_zero' else selected]=0.
    merged=context.transpose(1,2).reshape(b,l,e)
    return F.linear(merged,module.out_proj.weight,module.out_proj.bias)


@contextmanager
def last_layer_intervention(model,text,mode):
    if mode not in MODES:raise ValueError('explicit frozen last-layer intervention required')
    delta=rotated_character_delta(model,text) if mode.endswith('token_rotate') else None
    def after(module,args,kwargs,output):
        return (intervened_attention(module,args[0],args[1],args[2],kwargs.get('attn_mask'),mode,delta),None)
    hook=model.blocks[-1].cross_attention.register_forward_hook(after,with_kwargs=True)
    try:yield
    finally:hook.remove()
