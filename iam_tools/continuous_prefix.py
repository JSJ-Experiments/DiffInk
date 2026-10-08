"""Closed-loop rollout retaining gradients through continuous coordinate feedback.

Unlike generated_prefix.own_prefix_forward, predicted XY is NOT detached before
the next recurrent update. Hard argmax pen decisions remain nondifferentiable.
Forward values are identical; this changes only the derivative used to optimize
an own-history trajectory loss. It is not generic smoothing or a model redesign.
"""
import torch
from torch.nn import functional as F
from .history_rollin import history_forward
from .generated_prefix import own_prefix_forward
from .autoregressive_strokes import reconstruction_terms
from .cumulative_xy import cumulative_xy_loss


def continuous_prefix_forward(model,text,writer_ids,blocks):
    if model.config['point_feedback'] or type(blocks) is not int or not 1<=blocks<=256:
        raise ValueError('bounded no-intra-point-feedback rollout required')
    memory,valid,writer=model.memory(text,writer_ids);state=model.initial_state(memory)
    previous=memory.new_zeros(len(text),40);offsets=[];pens=[];traces=[];histories=[]
    for j in range(blocks):
        histories.append(previous)
        state,tr=model.coarse_step(previous,state,memory,valid,writer,memory.new_full((len(text),),float(j==0)))
        inputs=memory.new_zeros(len(text),8,6)
        inputs[:,0,:5]=torch.cat((previous[:,14:16],previous[:,37:40]),-1)
        inputs[:,0,5]=float(j==0)
        decoded,_=model.point_gru(inputs,state[1][None]);out=model.point_readout(decoded)
        xy,pen=out[...,:2],out[...,2:];offsets.append(xy);pens.append(pen);traces.append(tr)
        previous=torch.cat((xy.flatten(1),F.one_hot(pen.detach().argmax(-1),3).to(xy.dtype).flatten(1)),-1)
    trace={key:torch.stack([t[key] for t in traces],1) for key in traces[0]}
    trace['used_history']=torch.stack(histories,1)
    return torch.stack(offsets,1),torch.stack(pens,1),trace


def paired_feedback_losses(model,feedback,text,writer_ids,targets,states,real_mask,cfg,continuous):
    if type(continuous) is not bool or feedback.shape[:2]!=real_mask.shape[:2]:
        raise ValueError('explicit derivative policy and matched training extent required')
    zero=torch.zeros(len(text),feedback.shape[1]-1,dtype=torch.bool,device=feedback.device)
    pred,pen,trace=history_forward(model,feedback,text,writer_ids,zero,zero)
    terms=reconstruction_terms(pred,pen,targets,states,real_mask,pred.new_tensor(cfg['pen_weights']))
    teacher_xy=cumulative_xy_loss(pred,targets,real_mask,cfg['offset_stats'])
    base=terms['offset']+cfg['pen_weight']*terms['pen']+cfg['teacher_anchor_weight']*teacher_xy
    forward=continuous_prefix_forward if continuous else own_prefix_forward
    own,own_pen,own_trace=forward(model,text,writer_ids,feedback.shape[1])
    own_xy=cumulative_xy_loss(own,targets,real_mask,cfg['offset_stats'])
    own_terms=reconstruction_terms(own,own_pen,targets,states,real_mask,pred.new_tensor(cfg['pen_weights']))
    return dict(base=base,teacher_offset=terms['offset'],teacher_pen=terms['pen'],teacher_xy=teacher_xy,
                own_xy=own_xy,own_pen=own_terms['pen'],own_offset_diagnostic=own_terms['offset'],
                teacher_trace=trace,own_trace=own_trace)
