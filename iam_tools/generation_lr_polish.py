"""Controlled LR restoration and index-phase diagnostics (not curve smoothing)."""
import copy, math
import numpy as np
import torch


def restore_adam(model, state, lr):
    """Keep all saved Adam moments/settings, changing every group's actual LR."""
    if not isinstance(lr,(float,int)) or isinstance(lr,bool) or not math.isfinite(lr) or not 0<lr<=1e-3:
        raise ValueError('bounded finite positive learning rate required')
    if not state['state']:
        raise ValueError('nonempty completed Adam state required')
    # Preserve grouping, even if a future pinned model has multiple groups.
    parameters=list(model.parameters()); groups=[];used=[]
    for g in state['param_groups']:
        indices=g['params'];used.extend(indices)
        if any(type(i)!=int or not 0<=i<len(parameters) for i in indices):
            raise ValueError('unsupported optimizer parameter indexing')
        groups.append(dict(params=[parameters[i] for i in indices]))
    if sorted(used)!=list(range(len(parameters))):
        raise ValueError('optimizer must cover each model parameter exactly once')
    opt=torch.optim.AdamW(groups)
    opt.load_state_dict(copy.deepcopy(state))
    for g in opt.param_groups:
        g['lr']=float(lr)
        if 'initial_lr' in g:g['initial_lr']=float(lr)
    return opt


def phase_statistics(samples, period=8):
    """Pool real within-true-stroke links by starting point index mod period.

    Samples are (predicted XY, target XY, target states); per-index displacement
    is not physical velocity. Target corners and nonuniform spacing are retained.
    Empty angle sets are reported as None, not a fictitious perfect angle score.
    """
    if type(period)!=int or period<1:raise ValueError('positive integer period required')
    bins=[dict(errors=[],target=[],angles=[]) for _ in range(period)]
    for pred,target,states in samples:
        pred,target,states=np.asarray(pred),np.asarray(target),np.asarray(states)
        if pred.shape!=target.shape or target.ndim!=2 or target.shape[1]!=2 or states.shape!=(len(target),):
            raise ValueError('aligned N,2 coordinates and N states required')
        if not np.isfinite(pred).all() or not np.isfinite(target).all() or not np.isin(states,[0,1,2]).all():
            raise ValueError('finite coordinates and three-state labels required')
        a,b=np.diff(pred,axis=0),np.diff(target,axis=0)
        for phase,entry in enumerate(bins):
            take=(states[:-1]==0)&(np.arange(len(a))%period==phase)
            pa,tb=a[take],b[take];entry['errors'].extend((pa-tb).tolist());entry['target'].extend(tb.tolist())
            valid=(np.linalg.norm(pa,axis=1)>1e-8)&(np.linalg.norm(tb,axis=1)>1e-8)
            cosine=(pa[valid]*tb[valid]).sum(1)/(np.linalg.norm(pa[valid],axis=1)*np.linalg.norm(tb[valid],axis=1))
            entry['angles'].extend(np.degrees(np.arccos(np.clip(cosine,-1,1))).tolist())
    rows=[]
    for phase,entry in enumerate(bins):
        a,t=np.asarray(entry['errors']).reshape(-1,2),np.asarray(entry['target']).reshape(-1,2)
        rows.append(dict(phase=phase,links=len(a),
            segment_error_rmse=float(np.sqrt(np.square(a).sum(1).mean())) if len(a) else None,
            target_segment_rms=float(np.sqrt(np.square(t).sum(1).mean())) if len(t) else None,
            tangent_error_p90_degrees=float(np.percentile(entry['angles'],90)) if entry['angles'] else None))
    return rows
