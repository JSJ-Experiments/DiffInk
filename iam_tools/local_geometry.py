"""Target-relative segment matching on unevenly spaced stroke points, not smoothing."""
import torch


def target_relative_difference_loss(prediction,target,states,mask,quantile=.25):
    """Match ΔXY / max(target edge length, target q25), per real stroke edge.

    Dimensionless vector disagreement constrains BOTH target direction and length.
    The per-line floor prevents tiny/degenerate edges dominating. Zero target
    edges are excluded; point/raw-Δ objectives still constrain their positions.
    Every target corner remains a valid target, not a curvature-to-zero penalty.
    Select real connected windows before arithmetic, so NaN padding is irrelevant.
    """
    if (prediction.shape!=target.shape or prediction.shape!=(*mask.shape,2) or states.shape!=mask.shape
            or mask.dtype!=torch.bool or not 0<quantile<1 or not mask.any()):
        raise ValueError('matching XY, states, nonempty boolean mask and bounded quantile required')
    per_line=[]
    for p,t,s,m in zip(prediction,target,states,mask):
        valid=m[:-1]&m[1:]&(s[:-1]==0)
        if not valid.any():continue
        dp=p[1:][valid]-p[:-1][valid];dt=t[1:][valid]-t[:-1][valid]
        length=torch.linalg.vector_norm(dt,dim=-1);keep=length>1e-8
        if not keep.any():continue
        length=length[keep];floor=torch.quantile(length.detach(),quantile)
        residual=(dp[keep]-dt[keep])/length.clamp_min(floor)[:,None]
        per_line.append(residual.square().mean())
    return torch.stack(per_line).mean() if per_line else prediction[mask].sum()*0


def calibrate_relative(model,train_batches,fraction=.2):
    """Fixed full-TRAIN aggregate decoder gradient calibration, no .grad writes."""
    from .curve_study import forward_xy
    from .latent_integration import DELTA_WEIGHT
    from model.losses import target_difference_loss
    if not train_batches or not 0<fraction<=.5:raise ValueError('bounded nonempty calibration required')
    params=list(model.decoder.parameters())+list(model.transformer_decoder.parameters())
    sums={k:[torch.zeros_like(p) for p in params] for k in ('geometry','relative')}
    values=dict(geometry=0.,relative=0.)
    for raw,mask,_ in train_batches:
        xy,target,_=forward_xy(model,raw,mask);states=raw[:,2:].argmax(1)
        g=(xy[mask]-target[mask]).square().mean()+DELTA_WEIGHT*target_difference_loss(xy,target,states,mask)
        r=target_relative_difference_loss(xy,target,states,mask)
        for k,loss in [('geometry',g),('relative',r)]:
            gs=torch.autograd.grad(loss/len(train_batches),params,retain_graph=k=='geometry')
            for acc,v in zip(sums[k],gs):acc.add_(v.detach())
            values[k]+=float(loss.detach())/len(train_batches)
    norms={k:float(torch.stack([v.square().sum() for v in gs]).sum().sqrt()) for k,gs in sums.items()}
    if min(norms.values())<1e-12:raise ValueError('degenerate relative geometry calibration')
    weight=fraction*norms['geometry']/norms['relative']
    cosine=float(torch.stack([(a*b).sum() for a,b in zip(sums['geometry'],sums['relative'])]).sum()/(norms['geometry']*norms['relative']))
    return dict(target_fraction=fraction,relative_weight=weight,decoder_gradient_norms=norms,cosine=cosine,initial_losses=values,
                target_length_floor='per-line nonzero connected target edge q25',
                method='aggregate full-training-set decoder gradients; fixed scalar; no validation gradients or search')
