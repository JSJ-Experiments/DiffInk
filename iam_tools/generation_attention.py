"""Evaluation-only cross-attention observation; not ground-truth character alignment.

need_weights=True chooses a different PyTorch attention kernel. Preserve hooks,
report inference drift, and never use this path for training or checkpoint choice.
"""
import numpy as np
import torch


@torch.no_grad()
def capture(model,*args,**kwargs):
    if model.training:raise ValueError('eval-only observer required')
    observed={};hooks=[]
    def request(module,args,kw):
        kw=dict(kw,need_weights=True,average_attn_weights=False);return args,kw
    def keep(name):
        def record(module,args,result):observed[name]=result[1].detach().cpu().numpy()
        return record
    try:
        for name,module in model.named_modules():
            if name.endswith('cross_attention'):
                hooks.extend([module.register_forward_pre_hook(request,with_kwargs=True),module.register_forward_hook(keep(name))])
        result=model(*args,**kwargs)
    finally:
        for hook in hooks:hook.remove()
    if not observed:raise ValueError('cross-attention modules required')
    return result,observed


def describe(weights,real_length,text_length):
    """Per-head token barycenter. No reference character/point boundaries assumed."""
    w=np.asarray(weights)
    if w.ndim!=3 or not 1<=real_length<=w.shape[1] or not 1<=text_length<w.shape[2]:raise ValueError('H,L,S with BOS+realtext required')
    real=w[:,:real_length,1:text_length+1];mass=real.sum(-1);p=real/np.maximum(mass[...,None],1e-12)
    token_fraction=np.arange(text_length)/max(1,text_length-1);center=(p*token_fraction).sum(-1)
    entropy=-(np.where(p>0,p*np.log(np.maximum(p,1e-12)),0.)).sum(-1)/max(1.,np.log(text_length))
    stats=[]
    for h,c in enumerate(center):
        corr=None
        if len(c)>1 and c.std()>1e-8:corr=float(np.corrcoef(np.linspace(0,1,len(c)),c)[0,1])
        stats.append(dict(head=h,time_token_barycenter_correlation=corr,backward_barycenter_step_fraction=float(np.mean(np.diff(c)<-.001)) if len(c)>1 else None,normalized_char_attention_entropy=float(entropy[h].mean()),char_attention_mass=float(mass[h].mean())))
    return dict(heads=stats,mean_weights=real.mean(0).tolist(),definition='descriptive soft attention over text, excluding BOS/padding; NOT actual alignment accuracy or causality')
