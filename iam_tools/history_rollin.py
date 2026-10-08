"""Explicit block-history interventions for the no-intra-point-feedback writer.

Own XY and own pen histories can be selected independently, strictly AFTER the
current block has been predicted. Selected predicted feedback is detached;
recurrent hidden states retain ordinary BPTT. This is scheduled sampling, not an
unbiased likelihood objective, and source-index targets can be ambiguous after a
bad rollout. Diagnostic source-length interventions are NOT free generation.
"""
import torch
from torch.nn import functional as F


def history_forward(model,feedback,text,writer_ids,own_xy,own_pen):
    if model.config['point_feedback']:
        raise ValueError('no-intra-point-feedback checkpoint required')
    if feedback.ndim!=3 or feedback.shape[-1]!=40 or not feedback.shape[1]:
        raise ValueError('B,L,40 history fields required')
    b,length,_=feedback.shape
    for mask in [own_xy,own_pen]:
        if mask.dtype!=torch.bool or mask.shape!=(b,length-1) or mask.device!=feedback.device:
            raise ValueError('explicit device-matched B,L-1 Boolean history transition masks required')
    memory,valid,writer=model.memory(text,writer_ids);state=model.initial_state(memory)
    previous=feedback.new_zeros(b,40);offsets=[];pens=[];traces=[];histories=[]
    for j in range(length):
        histories.append(previous)
        state,trace=model.coarse_step(previous,state,memory,valid,writer,feedback.new_full((b,),float(j==0)))
        inputs=feedback.new_zeros(b,8,6)
        inputs[:,0,:5]=torch.cat((previous[:,14:16],previous[:,37:40]),-1)
        inputs[:,0,5]=float(j==0)
        decoded,_=model.point_gru(inputs,state[1][None]);out=model.point_readout(decoded)
        o,p=out[...,:2],out[...,2:];offsets.append(o);pens.append(p);traces.append(trace)
        if j<length-1:
            own_o=o.detach().flatten(1)
            own_p=F.one_hot(p.detach().argmax(-1),3).to(o.dtype).flatten(1)
            previous=torch.cat((torch.where(own_xy[:,j,None],own_o,feedback[:,j,:16]),torch.where(own_pen[:,j,None],own_p,feedback[:,j,16:])), -1).detach()
    trace={k:torch.stack([t[k] for t in traces],1) for k in traces[0]}
    trace['used_history']=torch.stack(histories,1)
    return torch.stack(offsets,1),torch.stack(pens,1),trace


def rollin_probability(step,max_probability=.2,ramp_steps=500):
    if type(step)!=int or step<1 or not 0<=max_probability<=1 or type(ramp_steps)!=int or ramp_steps<1:
        raise ValueError('positive update/ramp and bounded probability required')
    return max_probability*min(step/ramp_steps,1.)


def transition_mask(draws,point_mask,probability):
    """Only transitions leading to another REAL block are eligible; draws fixed."""
    if point_mask.ndim!=3 or point_mask.shape[-1]!=8 or point_mask.dtype!=torch.bool or draws.shape!=(point_mask.shape[0],point_mask.shape[1]-1) or draws.device!=point_mask.device or not torch.isfinite(draws).all() or (draws<0).any() or (draws>=1).any() or not 0<=probability<=1:
        raise ValueError('matching uniform[0,1) draws, real B,L,8 mask and bounded probability required')
    return (draws<probability)&point_mask[:,1:].any(-1)
