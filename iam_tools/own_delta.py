"""Own-history target displacement matching, NOT generic smoothing.

Normalized chronological offsets correspond to target index differences. Ink-only
uses both real endpoints and GT previous continue, excluding origin/pen-up jumps.
It spans eight-point block boundaries. Nonuniform IAM/RDP spacing means these are
NOT physical velocity or curvature. No GT field reaches the generated graph.
"""
import torch
from .continuous_prefix import continuous_prefix_forward
from .history_rollin import history_forward
from .autoregressive_strokes import reconstruction_terms
from .cumulative_xy import cumulative_xy_loss,anchor_coefficient


def displacement_mask(states,real_mask,mode):
    if states.shape!=real_mask.shape or states.ndim!=3 or states.shape[2]!=8 or states.dtype!=torch.long or real_mask.dtype!=torch.bool or mode not in ('ink','all'):
        raise ValueError('B,L,8 integer states, Boolean real mask and explicit ink/all mode required')
    flat=real_mask.flatten(1);pen=states.flatten(1)
    if not flat.any(1).all() or ((~flat[:,:-1])&flat[:,1:]).any() or ((pen[flat]<0)|(pen[flat]>2)).any():
        raise ValueError('nonempty prefix-valid real points and valid pen classes required')
    if mode=='all':return real_mask
    ink=torch.zeros_like(flat);ink[:,1:]=flat[:,:-1]&flat[:,1:]&(pen[:,:-1]==0)
    return ink.reshape_as(real_mask)


def displacement_loss(prediction,target,states,real_mask,mode):
    if prediction.shape!=target.shape or prediction.shape!=(*real_mask.shape,2):
        raise ValueError('matched B,L,8,2 normalized chronological offsets required')
    selected=displacement_mask(states,real_mask,mode)
    if not selected.any():raise ValueError('at least one supervised target displacement required')
    return (prediction-target)[selected].square().mean()


def paired_delta_losses(model,feedback,text,writer_ids,targets,states,real_mask,cfg):
    """Same paired teacher/full-own graphs as existing continuous-feedback path."""
    if feedback.shape[:2]!=real_mask.shape[:2]:raise ValueError('same supervised training unroll extent required')
    zero=torch.zeros(len(text),feedback.shape[1]-1,dtype=torch.bool,device=feedback.device)
    pred,pen,trace=history_forward(model,feedback,text,writer_ids,zero,zero)
    teacher=reconstruction_terms(pred,pen,targets,states,real_mask,pred.new_tensor(cfg['pen_weights']))
    teacher_xy=cumulative_xy_loss(pred,targets,real_mask,cfg['offset_stats'])
    base=teacher['offset']+cfg['pen_weight']*teacher['pen']+cfg['teacher_anchor_weight']*teacher_xy
    own,own_pen,own_trace=continuous_prefix_forward(model,text,writer_ids,feedback.shape[1])
    own_xy=cumulative_xy_loss(own,targets,real_mask,cfg['offset_stats'])
    own_terms=reconstruction_terms(own,own_pen,targets,states,real_mask,pred.new_tensor(cfg['pen_weights']))
    complete=base+cfg['own_xy_weight']*own_xy+cfg['own_pen_weight']*own_terms['pen']
    return dict(base=base,complete=complete,teacher_offset=teacher['offset'],teacher_pen=teacher['pen'],teacher_xy=teacher_xy,own_xy=own_xy,own_pen=own_terms['pen'],own_offset_diagnostic=own_terms['offset'],own_ink_delta=displacement_loss(own,targets,states,real_mask,'ink'),own_all_delta=displacement_loss(own,targets,states,real_mask,'all'),teacher_trace=trace,own_trace=own_trace)


def delta_coefficients(complete_norm,ink_norm,all_norm):
    return dict(ink=anchor_coefficient(complete_norm,ink_norm,.25),all=anchor_coefficient(complete_norm,all_norm,.25))
