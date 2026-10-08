"""Controlled meaningful-channel / diffusion-parameterization interventions.

The initialized transport stores eight (XY + 3 pen) points in its first40
channels. Compact40 is a representation ablation, NOT learned semantic InkVAE.
`v` is the diffusion rotated-noise target, not trajectory velocity. It gives
x0 = sqrt(alpha)*xt - sqrt(1-alpha)*v, a low-noise identity path. Uniform v-MSE
also changes effective SNR weighting; do not attribute results to the skip alone.
"""
import math
import torch
from torch import nn

ARMS = {'full_x0': (384, 'x0'), 'compact_x0': (40, 'x0'),
        'full_v': (384, 'v'), 'compact_v': (40, 'v')}


def coefficients(alpha, t, like):
    if t.shape != (len(like),) or t.dtype != torch.long:
        raise ValueError('one integer diffusion timestep per batch item required')
    if alpha.ndim != 1 or (t < 0).any() or (t >= len(alpha)).any():
        raise ValueError('bounded one-dimensional diffusion schedule required')
    a = alpha[t].to(like)[:, None, None]
    if not torch.isfinite(a).all() or (a <= 0).any() or (a > 1).any():
        raise ValueError('finite alpha in (0,1] required')
    return a.sqrt(), (1-a).sqrt()


def to_x0(output, noisy, a, b, prediction):
    if prediction == 'x0': return output
    if prediction == 'v': return a*noisy-b*output
    raise ValueError('x0 or diffusion-v prediction required')


def loss(model, clean384, text, mask, prefix, t, alpha, divisor,
         keep_prefix, drop_text, channels, prediction, eps384=None):
    """Shared 384-wide posterior/noise RNG, then slice for compact ablation.

    Same retained-prefix suffix mask as corpus baseline. v uses actual noisy
    suffix, not the clean retained input. No padding/reference contributes.
    Returns training-objective and comparable reconstructed-x0 field MSE.
    """
    if channels not in (40,384) or prediction not in ('x0','v'):
        raise ValueError('declared factorial arm required')
    if clean384.ndim!=3 or clean384.shape[-1]!=384 or mask.shape!=clean384.shape[:2] or mask.dtype!=torch.bool or prefix.shape!=(len(clean384),) or not math.isfinite(divisor) or divisor<=0:
        raise ValueError('finite divisor and shared B,L,384 masked posterior required')
    a,b=coefficients(alpha,t,clean384)
    eps384=torch.randn_like(clean384.transpose(1,2)).transpose(1,2) if eps384 is None else eps384
    if eps384.shape!=clean384.shape:raise ValueError('shared full-channel diffusion noise required')
    noisy384=a*clean384+b*eps384
    clean=clean384[...,:channels];noisy=noisy384[...,:channels];eps=eps384[...,:channels]
    retained=(torch.arange(clean.shape[1],device=clean.device)[None]<prefix[:,None])&mask if keep_prefix and not drop_text else torch.zeros_like(mask)
    active=mask&~retained
    if not active.any():raise ValueError('nonempty denoised suffix required')
    cond=torch.where(retained[...,None],clean,noisy)
    out=model(x=cond,noise=noisy,text=text,time=t.float()/divisor,mask=mask,
              drop_text=drop_text,drop_cond=not bool(retained.any()))
    if out.shape!=clean.shape:raise ValueError('arm output channel mismatch')
    target=clean if prediction=='x0' else a*eps-b*clean
    error=(out-target)[active].square()
    x0=to_x0(out,noisy,a,b,prediction)
    xerror=(x0-clean)[active].square()
    return dict(loss=error.mean(),active40_objective=error[:,:40].mean(),
                unused344_objective=error[:,40:].mean() if channels==384 else None,
                xy16_x0_mse=xerror[:,[j for j in range(40) if j%5<2]].mean(),
                pen24_x0_mse=xerror[:,[j for j in range(40) if j%5>=2]].mean(),
                active40_x0_mse=xerror[:,:40].mean(),active_mask=active,
                prediction=x0,noisy=noisy,raw_output=out)


class X0Adapter(nn.Module):
    """384-wide evaluation boundary; all unused compact output = TRAIN mean.

    Evaluator supplies target-free noise384/mask and performs identical DDIM/CFG.
    Conversion is linear, so text CFG on x0 equals CFG on v. Does NOT introduce
    an endpoint/target/smoothing heuristic. At compact decode, unused whitened
    channels are zero. Their irrelevance is independently gated in the study.
    """
    def __init__(self, backbone, alpha, channels, prediction, divisor=1000.):
        super().__init__()
        if channels not in (40,384) or prediction not in ('x0','v') or divisor<=0:
            raise ValueError('declared arm contract required')
        self.backbone=backbone;self.register_buffer('alpha',alpha.clone())
        self.channels=channels;self.prediction=prediction;self.divisor=divisor

    def forward(self,x,noise,text,time,mask,drop_text,drop_cond):
        t=(time*self.divisor).round().long()
        a,b=coefficients(self.alpha,t,noise)
        raw=self.backbone(x=x[...,:self.channels],noise=noise[...,:self.channels],
                          text=text,time=time,mask=mask,drop_text=drop_text,
                          drop_cond=drop_cond)
        decoded=to_x0(raw,noise[...,:self.channels],a,b,self.prediction)
        out=torch.zeros_like(noise);out[...,:self.channels]=decoded
        return out.masked_fill(~mask[...,None],0.)


def matched_initial_state(full_state, compact_state):
    """Exactly same backbone/text/time initialization, sliced latent IO weights.

    Input projection concatenates [latent,text]; retain first40 latent and ALL
    text columns. No architecture width/depth/attention change in compact arm.
    """
    result={}
    for key,value in compact_state.items():
        original=full_state[key]
        if key=='input_embed.proj.weight':
            result[key]=torch.cat((original[:,:40],original[:,384:]),1).clone()
        elif key in ('proj_out.weight','proj_out.bias'):
            result[key]=original[:40].clone()
        else: result[key]=original.clone()
        if result[key].shape!=value.shape:raise ValueError('unexpected compact architecture difference '+key)
    return result
