"""Shared actual InkVAE training losses and decoder-gradient anchor calibration."""
import math
import torch
import torch.nn.functional as F
from .gmm import get_mixture_coef, get_loss


def mixture_expectation(output):
    pi=output[:,3:23].softmax(1)
    return torch.stack([(pi*output[:,23:43]).sum(1),(pi*output[:,43:63]).sum(1)],-1)


def target_difference_loss(prediction, target, states, mask, order=1):
    """Match target index differences INSIDE strokes, never smooth toward zero.

    RDP points are unevenly spaced: these are not time derivatives or geometric
    curvature. Select complete valid windows before arithmetic so NaN padding
    cannot contaminate values or gradients. Average coordinates and windows.
    """
    if (prediction.shape != target.shape or prediction.shape != (*mask.shape, 2)
            or states.shape != mask.shape or mask.dtype != torch.bool or order not in (1, 2)):
        raise ValueError('matching B x T x 2 coordinates, states, boolean mask; order 1/2 required')
    if not mask.any():
        raise ValueError('nonempty real-point mask required')
    if prediction.shape[1] <= order:
        return prediction[mask].sum()*0
    valid = mask[:, :-order].clone()
    for offset in range(1, order+1):
        valid &= mask[:, offset:mask.shape[1]-order+offset]
    for offset in range(order):
        valid &= states[:, offset:mask.shape[1]-order+offset] == 0
    if not valid.any():
        return prediction[mask].sum()*0
    residual = []
    for offset in range(order+1):
        sl = slice(offset, prediction.shape[1]-order+offset)
        residual.append(prediction[:, sl][valid]-target[:, sl][valid])
    difference = residual[1]-residual[0] if order == 1 else residual[2]-2*residual[1]+residual[0]
    return difference.square().mean()


def target_tangent_loss(prediction, target, states, mask):
    """Target segment-direction matching, not smoothing; short edges count equally.

    Half squared unit-vector distance equals 1-cos(angle) for nondegenerate
    segments. Exclude zero-length target segments. Clamp predicted length to
    1% of that target segment's length to keep collapsed-edge gradients finite.
    This deliberate short-edge sensitivity is experimental, not a default loss.
    """
    if (prediction.shape != target.shape or prediction.shape != (*mask.shape, 2)
            or states.shape != mask.shape or mask.dtype != torch.bool):
        raise ValueError('matching coordinates, states and boolean mask required')
    valid = mask[:, :-1] & mask[:, 1:] & (states[:, :-1] == 0)
    if not valid.any(): return prediction[mask].sum()*0
    dp = prediction[:, 1:][valid]-prediction[:, :-1][valid]
    dt = target[:, 1:][valid]-target[:, :-1][valid]
    length = torch.linalg.vector_norm(dt, dim=-1)
    nonzero = length > 1e-8
    if not nonzero.any(): return prediction[mask].sum()*0
    dp, dt, length = dp[nonzero], dt[nonzero], length[nonzero]
    predicted_length = torch.linalg.vector_norm(dp, dim=-1).clamp_min(length*.01)
    return .5*(dp/predicted_length[:, None]-dt/length[:, None]).square().sum(-1).mean()


def stable_nll(pi,mx,my,sx,sy,rho,x,y):
    rho=rho.clamp(-1+1e-5,1-1e-5);one=(1-rho.square()).clamp_min(1e-8)
    dx=(x-mx)/sx;dy=(y-my)/sy
    logpdf=-math.log(2*math.pi)-sx.log()-sy.log()-.5*one.log()-.5*((dx-rho*dy).square()/one+dy.square())
    return -torch.logsumexp(pi.clamp_min(torch.finfo(pi.dtype).tiny).log()+logpdf,1)


def bounded_pen(logits,targets,mask,gamma=2,cap=8,policy='bounded_three_state'):
    losses=[]
    for i in range(len(logits)):
        labels=targets[i,mask[i]];real_logits=logits[i,mask[i]]
        if policy=='binary_forced_final':
            if labels[-1]!=2 or (labels[:-1]>1).any():raise ValueError('binary policy needs final-only EOC')
            real_logits=real_logits[:-1,:2];labels=labels[:-1]
        elif policy!='bounded_three_state':raise ValueError('unknown bounded pen policy')
        counts=labels.bincount(minlength=real_logits.shape[-1]).float()
        weights=(counts[0]/counts.clamp_min(1)).sqrt().clamp(max=cap);weights[counts==0]=0
        ce=F.cross_entropy(real_logits,labels,reduction='none')
        losses.append((weights[labels]*(1-ce.neg().exp()).pow(gamma)*ce).mean())
    return torch.stack(losses).mean()


def loss_terms(output,target_model_space,mask,config,ctc,kl,style):
    pi,mx,my,sx,sy,rho,pen,logits=get_mixture_coef(output,20)
    channel=target_model_space
    element=stable_nll(pi,mx,my,sx,sy,rho,channel[:,:1],channel[:,1:2])
    if not getattr(config,'gmm_logspace',False):
        element,legacy_pen=get_loss(pi,mx,my,sx,sy,rho,pen,logits,channel[:,:1],channel[:,1:2],channel[:,2:])
    else:legacy_pen=None
    policy=getattr(config,'pen_policy','inverse_frequency')
    if policy=='inverse_frequency':
        if legacy_pen is None:
            _,legacy_pen=get_loss(pi,mx,my,sx,sy,rho,pen,logits,channel[:,:1],channel[:,1:2],channel[:,2:])
        pen_loss=legacy_pen
    else:
        pen_loss=bounded_pen(logits.transpose(1,2),channel[:,2:].argmax(1),mask.bool(),
                             getattr(config,'pen_focal_gamma',2),getattr(config,'pen_weight_cap',8),policy)
    xy=mixture_expectation(output);target=channel[:,:2].transpose(1,2)
    mse=(xy[mask.bool()]-target[mask.bool()]).square().mean()
    terms = {'gmm':element[mask.bool()].mean(),'pen':pen_loss,'expected_xy':mse,
             'ctc':ctc.mean(),'kl':kl.mean(),'style':style.mean()}
    if getattr(config, 'target_delta_weight', 0):
        terms['target_delta'] = target_difference_loss(xy, target, channel[:, 2:].argmax(1), mask.bool())
    return terms


def weighted_loss(terms,config):
    return sum(v*float(getattr(config,k+'_weight',0)) for k,v in terms.items())


def calibrate_anchor(model,batches,config,device,fraction=.15):
    """Median per-line decoder gradient ratio, on SAME sampled forward per loss.

    No optimizer steps and no .grad accumulation. Logs actual weighted ratio per
    line; a scalar calibrated on one sample is not assumed valid for every line.
    """
    from utils.mask import downsample_mask
    parameters=list(model.decoder.parameters())+list(model.transformer_decoder.parameters())
    rows=[]
    for batch in batches:
        data,mask,text,_,writer=batch
        data=data.to(device).transpose(1,2);mask=mask.to(device)
        output,ctc,kl,style=model(data,downsample_mask(mask,8),text.to(device),writer.to(device),False,False,point_mask=mask)
        terms=loss_terms(output,model.to_model_space(data),mask,config,ctc,kl,style)
        a=torch.autograd.grad(terms['gmm']*config.gmm_weight,parameters,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(terms['expected_xy'],parameters,allow_unused=True)
        norm=lambda gradients:float(torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt())
        gn,mn=norm(a),norm(b)
        if not math.isfinite(gn+mn) or gn<=0 or mn<=1e-12:raise ValueError('invalid anchor gradient calibration')
        rows.append({'gmm_decoder_norm':gn,'mse_decoder_norm':mn,'suggested_weight':fraction*gn/mn})
    weight=float(torch.tensor([r['suggested_weight'] for r in rows],dtype=torch.float64).quantile(.5))
    for row in rows:row['actual_initial_weighted_ratio']=weight*row['mse_decoder_norm']/row['gmm_decoder_norm']
    return {'weight':weight,'target_fraction':fraction,'method':'median matched-per-line decoder gradient ratio','lines':rows}
