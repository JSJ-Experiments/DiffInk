"""Geometry anchoring ONLY for the explicitly identified polyphase40 codec.

Raw index differences match target segments, NOT physical velocities or generic
smoothing. Pen-up jumps and padding never supervise segment matching.
"""
import torch
from .latent_diffusion import transform


def physical_terms(pred, target, stats, point_mask, *, codec_contract):
    if codec_contract!='polyphase40':raise ValueError('not valid for generic learned semantic VAE')
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[-1]!=384:raise ValueError('B,L,384 required')
    p=transform(pred,stats,True)[...,:40].reshape(pred.shape[0],-1,5)
    q=transform(target,stats,True)[...,:40].reshape(pred.shape[0],-1,5)
    if point_mask.shape!=p.shape[:2] or point_mask.dtype!=torch.bool:raise ValueError('real Boolean point mask required')
    xy=torch.where(point_mask[...,None],p[...,:2]-q[...,:2],0.).square().sum()/(2*point_mask.sum().clamp_min(1))
    links=point_mask[:,:-1]&point_mask[:,1:]&(q[:,:-1,2:].argmax(-1)==0)
    delta=(p[:,1:,:2]-p[:,:-1,:2])-(q[:,1:,:2]-q[:,:-1,:2])
    difference=torch.where(links[...,None],delta,0.).square().sum()/(2*links.sum().clamp_min(1))
    return dict(xy=xy,first_difference=difference)


def gradient_norm(loss, parameters):
    grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    terms=[g.detach().square().sum() for g in grads if g is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def coefficients(base_norm, xy_norm, difference_norm, fractions=(.25,.10)):
    if min(base_norm,xy_norm,difference_norm)<=1e-12 or not all(torch.isfinite(torch.tensor(x)) for x in (base_norm,xy_norm,difference_norm)):
        raise ValueError('nondegenerate finite gradient calibration required')
    if not all(0<f<=1 for f in fractions):raise ValueError('bounded positive gradient fractions')
    return dict(xy=fractions[0]*base_norm/xy_norm,first_difference=fractions[1]*base_norm/difference_norm)
