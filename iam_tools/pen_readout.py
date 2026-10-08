"""Frozen-feature pen readout refit; only three final logit rows may change.

Teacher geometry must remain identical. Free geometry may change because newly
predicted pen states are fed into history; that is the intended causal diagnostic.
"""
import torch
from torch import nn
from torch.nn import functional as F


def extract_pen_features(model,feedback,text,writer_ids,mask):
    if mask.dtype!=torch.bool or mask.shape!=(*feedback.shape[:2],8):raise ValueError('real B,L,8 point mask required')
    captured=[]
    hook=model.point_readout.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    try:
        with torch.no_grad():outputs=model.teacher(feedback,text,writer_ids)
    finally:hook.remove()
    if len(captured)!=1:raise ValueError('one fused teacher point readout required')
    values=captured[0].reshape(*mask.shape,model.config['width'])[mask]
    return values,outputs


def make_pen_readout(model):
    readout=model.point_readout
    if readout.out_features!=5:raise ValueError('two XY rows followed by three pen logits required')
    head=nn.Linear(readout.in_features,3,device=readout.weight.device,dtype=readout.weight.dtype)
    with torch.no_grad():head.weight.copy_(readout.weight[2:]);head.bias.copy_(readout.bias[2:])
    return head


def install_pen_readout(model,head):
    r=model.point_readout
    if head.weight.shape!=(3,r.in_features) or head.bias.shape!=(3,) or not torch.isfinite(head.weight).all() or not torch.isfinite(head.bias).all():raise ValueError('finite compatible three-row head required')
    with torch.no_grad():r.weight[2:].copy_(head.weight);r.bias[2:].copy_(head.bias)


def assert_only_pen_rows_changed(before,after):
    if set(before)!=set(after):raise ValueError('same state keys required')
    for key in before:
        a,b=before[key],after[key]
        if key in ['point_readout.weight','point_readout.bias']:a,b=a[:2],b[:2]
        if not torch.equal(a,b):raise ValueError('non-pen parameter changed: '+key)
    return True


def pen_focal(logits,states,weights):
    if logits.ndim!=2 or logits.shape[1]!=3 or states.shape!=(len(logits),) or states.dtype!=torch.long or weights.shape!=(3,) or not len(logits):raise ValueError('real N,3 logits/N class targets/3 bounded weights required')
    ce=F.cross_entropy(logits,states,reduction='none')
    return (weights[states]*(1-ce.neg().exp()).square()*ce).mean()
