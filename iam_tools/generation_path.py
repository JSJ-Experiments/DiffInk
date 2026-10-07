"""Controlled text-conditioning versus diffusion diagnostic, unchanged architecture.

Terminal/direct generation takes ONLY text, mask (oracle duration), and Gaussian
noise/zero input. It never receives the target trajectory. Direct is a regression
capacity probe, not a paper diffusion sampler.
"""
import torch
from .latent_diffusion import forward_noise,ddim_sample

ARMS=('uniform','terminal','direct')
NATIVE={'uniform':'ddim50','terminal':'terminal','direct':'zero'}


def training_input(arm,clean,epsilon,draw_t,alpha,mask):
    if arm not in ARMS or clean.shape!=epsilon.shape or mask.shape!=clean.shape[:2]:raise ValueError('known arm,matching B,L,C and mask required')
    if arm=='uniform':
        return forward_noise(clean,epsilon,draw_t,alpha,mask),draw_t.float()/(len(alpha)-1)
    # No sqrt(alpha999)*target contamination: terminal is exact pure Gaussian.
    x=epsilon if arm=='terminal' else torch.zeros_like(epsilon)
    return x.masked_fill(~mask[...,None],0.),torch.ones(clean.shape[0],device=clean.device)


@torch.no_grad()
def generate(model,epsilon,text,mask,alpha,mode,drop_text=False):
    if model.training or mode not in ('zero','terminal','ddim2','ddim10','ddim50','ddim100'):raise ValueError('eval model and explicit readout required')
    if epsilon.ndim!=3 or mask.shape!=epsilon.shape[:2] or mask.dtype!=torch.bool or not mask.any(1).all():raise ValueError('nonempty Boolean duration mask')
    if mode.startswith('ddim'):
        return ddim_sample(model,epsilon,text,mask,alpha,steps=int(mode[4:]),drop_text=drop_text)
    x=epsilon if mode=='terminal' else torch.zeros_like(epsilon)
    drop=torch.full((len(x),),drop_text,device=x.device,dtype=torch.bool)
    result=model(x.masked_fill(~mask[...,None],0.),torch.ones(len(x),device=x.device),text,mask,drop)
    if not torch.isfinite(result).all():raise FloatingPointError('nonfinite direct/terminal prediction')
    return result.masked_fill(~mask[...,None],0.)


def native_train_score(evaluation,arm):
    """Fixed TRAIN/native trajectory score, never unseen or cherry-picked seed."""
    group=evaluation['aggregate']['train'][NATIVE[arm]+'_correct']
    return group['x_rmse']+group['y_rmse']+.25*group['segment_vector_rmse']+.1*(1-group['pen_f1_min'])
