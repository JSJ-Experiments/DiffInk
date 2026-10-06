"""Opt-in research conditioning; no change to default/released InkVAE contract."""
import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm1d(nn.Module):
    """Normalize channels at each point, never across points or batch items.

    Same affine parameter names/shapes as GroupNorm(1,C). Not numerically
    equivalent to GroupNorm: a checkpoint MUST record this research choice.
    """
    def __init__(self, channels, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(channels,device=device,dtype=dtype))
        self.bias=nn.Parameter(torch.zeros(channels,device=device,dtype=dtype))
        self.eps=eps

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=len(self.weight):
            raise ValueError('B x C x T input required')
        return F.layer_norm(x.transpose(1,2),(len(self.weight),),self.weight,self.bias,self.eps).transpose(1,2)


def install_channel_norm(module):
    """Copy affine parameters; replace only explicit GroupNorm(1,C) layers."""
    count=0
    for name,child in list(module.named_children()):
        if isinstance(child,nn.GroupNorm):
            if child.num_groups!=1 or not child.affine:raise ValueError('only affine GroupNorm(1,C) supported')
            layer=ChannelNorm1d(child.num_channels,child.eps,child.weight.device,child.weight.dtype)
            with torch.no_grad():layer.weight.copy_(child.weight);layer.bias.copy_(child.bias)
            setattr(module,name,layer);count+=1
        else:count+=install_channel_norm(child)
    return count


def condition_batches(batches,mode,scale=.01):
    """Physical microbatches only; offsets use input real points, not dataset fits.

    Input/output centering is reversible, aspect/spacing/states unchanged.
    Validation input can be transformed but never enters training statistics.
    Padded XY stays sentinel-zero, unless the explicit edge_pad probe is chosen.
    """
    if mode not in ('control','center','channel','edge_pad') or not 0<scale<=1:
        raise ValueError('known mode and positive model scale required')
    result={};offsets={}
    for sid,(raw,mask,labels) in batches.items():
        if raw.ndim!=3 or raw.shape[:2]!=(1,5) or mask.shape!=(1,raw.shape[-1]) or mask.dtype!=torch.bool:
            raise ValueError('one-line B x 5 x T batch and boolean real-point mask required')
        n=int(mask.sum())
        if not n or not mask[0,:n].all() or mask[0,n:].any():raise ValueError('nonempty contiguous real prefix required')
        x=raw.clone();offset=raw.new_zeros(2)
        if mode=='center':
            xy=raw[0,:2,:n];center=(xy.amin(-1)+xy.amax(-1))/2
            x[0,:2,:n]-=center[:,None];offset=center*scale
        elif mode=='edge_pad':x[0,:2,n:]=raw[0,:2,n-1,None]
        result[sid]=(x,mask,labels);offsets[sid]=offset
    return result,offsets


def restore_xy(xy,offset):
    """Restore model-space positions after a reversible input transform."""
    if xy.shape[-1]!=2 or offset.shape!=(2,):raise ValueError('XY last dimension and two-component offset required')
    return xy+offset
