"""Actual released DiT x0 diffusion helpers with explicit English contracts.

The backbone is model.dit.DiT, NOT the standalone deterministic writer. Timesteps
are normalized explicitly before F5-style scale1000 sinusoidal embedding. Text
CFG drops TEXT (prefix-only upstream CFG is ineffective with no prefix). This
baseline learns generic handwriting, not explicit writer-ID style control.
"""
import math
import torch
from .corpus_dit_contract import duration


def call(model,x,text,t,mask,divisor,drop_text=False):
    return model(x=x,noise=x,text=text,time=t.float()/divisor,mask=mask,drop_text=drop_text,drop_cond=True)


def training_loss(model,clean,text,mask,prefix_blocks,t,diffusion,divisor,keep_prefix,drop_text):
    """Matching original sampled-x0 MSE and retained-prefix noise policy.

    Every valid suffix contributes; dropped prefixes supervise the entire line.
    Neither padding nor retained reference positions contribute to x0 loss.
    """
    if clean.ndim!=3 or clean.shape[-1]!=384 or mask.shape!=clean.shape[:2] or mask.dtype!=torch.bool or prefix_blocks.shape!=(len(clean),) or not math.isfinite(divisor) or divisor<=0:
        raise ValueError('B,L,384 sampled latent and Boolean mask/prefix contract required')
    noisy,_=diffusion.noise_images(clean.transpose(1,2),t)
    retained=(torch.arange(clean.shape[1],device=clean.device)[None]<prefix_blocks[:,None])&mask if keep_prefix and not drop_text else torch.zeros_like(mask)
    active=mask&~retained
    if not active.any():raise ValueError('nonempty actually denoised valid suffix required')
    cond=torch.where(retained[...,None],clean,noisy)
    pred=model(x=cond,noise=noisy,text=text,time=t.float()/divisor,mask=mask,drop_text=drop_text,drop_cond=not bool(retained.any()))
    error=(pred-clean)[active].square()
    return dict(loss=error.mean(),active40=error[:,:40].mean(),unused344=error[:,40:].mean(),active_mask=active,prediction=pred)


@torch.no_grad()
def sample(model,noise,text,mask,alpha,steps=50,guidance=1.,divisor=1000.,null=False):
    """Genuine target-free DDIM: only caller's random noise/text/requested mask.

    No reference, target z/points/length/pens, endpoint correction or smoothing.
    Caller derives mask from TRAIN-only text-duration predictor, never GT length.
    """
    if model.training or noise.ndim!=3 or noise.shape[-1]!=384 or mask.shape!=noise.shape[:2] or mask.dtype!=torch.bool or not mask.any(1).all() or not 2<=steps<=len(alpha) or guidance<0 or divisor<=0:
        raise ValueError('bounded eval-mode target-free diffusion inputs required')
    times=torch.linspace(len(alpha)-1,0,steps).round().long().tolist()+[-1]
    x=noise.masked_fill(~mask[...,None],0.)
    for current,nxt in zip(times[:-1],times[1:]):
        t=torch.full((len(x),),current,device=x.device,dtype=torch.long);start=call(model,x,text,t,mask,divisor,null)
        if guidance!=1. and not null:
            unconditional=call(model,x,text,t,mask,divisor,True);start=unconditional+guidance*(start-unconditional)
        if nxt<0:x=start
        else:
            a,b=alpha[current],alpha[nxt];eps=(x-a.sqrt()*start)/(1-a).clamp_min(1e-8).sqrt();x=b.sqrt()*start+(1-b).sqrt()*eps
        x=x.masked_fill(~mask[...,None],0.)
    if not torch.isfinite(x).all():raise FloatingPointError('nonfinite actual diffusion sample')
    return x


def requested_inputs(texts,vocab,duration_model,seed,ids,device):
    if not texts or len(texts)!=len(ids) or any(set(t)-set(vocab) for t in texts):raise ValueError('covered nonempty requested text and identity only for noise RNG required')
    lengths=[duration(duration_model,t) for t in texts];length=max(lengths);mask=torch.arange(length,device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    labels=torch.full((len(texts),max(map(len,texts))),-1,device=device,dtype=torch.long);noise=torch.zeros(len(texts),length,384,device=device)
    import hashlib
    for j,(sid,text,n) in enumerate(zip(ids,texts,lengths)):
        labels[j,:len(text)]=torch.tensor([vocab.index(c) for c in text],device=device)
        g=torch.Generator(device=device);g.manual_seed(int(hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()[:15],16));noise[j,:n]=torch.randn(n,384,generator=g,device=device)
    return noise,labels,mask,lengths


def fit_posterior_whitening(items):
    """TRAIN-only total latent variance, including the sampled posterior noise."""
    total=None;second=None;variance=None;count=0
    for mu,lv in items:
        if mu.shape!=lv.shape or mu.ndim!=2 or mu.shape[1]!=384 or not torch.isfinite(mu).all() or not torch.isfinite(lv).all():raise ValueError('finite TRAIN L,384 posterior means/logvars required')
        a=mu.double();v=lv.double().exp();total=a.sum(0) if total is None else total+a.sum(0);second=a.square().sum(0) if second is None else second+a.square().sum(0);variance=v.sum(0) if variance is None else variance+v.sum(0);count+=len(mu)
    if not count:raise ValueError('nonempty TRAIN posterior required')
    mean=total/count;var=(second/count-mean.square()).clamp_min(0)+variance/count
    return dict(mean=mean.float(),std=var.sqrt().clamp_min(.1).float(),points=count,definition='TRAIN-only total posterior variance: variance(mu)+mean(exp(logvar)); all384 channels, stdfloor.1')


def learning_rate(step,cfg):
    if step<=cfg['warmup_updates']:return cfg['lr']*(.1+.9*step/cfg['warmup_updates'])
    progress=min(1.,(step-cfg['warmup_updates'])/(cfg['max_updates']-cfg['warmup_updates']))
    return cfg['min_lr']+(cfg['lr']-cfg['min_lr'])*.5*(1+math.cos(math.pi*progress))
