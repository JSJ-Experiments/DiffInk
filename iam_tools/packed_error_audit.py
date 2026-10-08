"""Decompose aligned coordinate errors; diagnostic only, NEVER a readout filter."""
import numpy as np


def block_error_statistics(samples, period=8):
    """Per-block mean position error + zero-mean shape error, on real points.

    The decomposition uses targets and cannot be used as a generation correction.
    Pooled squared vector errors retain signed cross terms. Index differences
    match sampled target geometry, not physical velocity or geometric curvature.
    """
    if type(period)!=int or period<1:raise ValueError('positive integer period required')
    point={k:[] for k in ('error','block_mean','shape')}
    links={name:{k:[] for k in ('error','block_mean','shape','left','right','target')} for name in ('seam','within')}
    for pred,true,states in samples:
        pred,true,states=np.asarray(pred),np.asarray(true),np.asarray(states)
        if pred.shape!=true.shape or true.ndim!=2 or true.shape[1]!=2 or states.shape!=(len(true),):raise ValueError('matching real N,2 coordinates and states required')
        if not len(true) or not np.isfinite(pred).all() or not np.isfinite(true).all() or not np.isin(states,[0,1,2]).all():raise ValueError('nonempty finite samples required')
        e=pred-true;mean=np.empty_like(e)
        for start in range(0,len(e),period):mean[start:start+period]=e[start:start+period].mean(0)
        shape=e-mean
        for key,values in [('error',e),('block_mean',mean),('shape',shape)]:point[key].extend(values.tolist())
        de,dm,ds=np.diff(e,axis=0),np.diff(mean,axis=0),np.diff(shape,axis=0)
        indices=np.arange(len(de));connected=states[:-1]==0
        for name in links:
            take=connected & ((indices%period==period-1) if name=='seam' else (indices%period!=period-1))
            for key,values in [('error',de),('block_mean',dm),('shape',ds),('left',e[:-1]),('right',e[1:]),('target',np.diff(true,axis=0))]:links[name][key].extend(values[take].tolist())
    def array(values):return np.asarray(values).reshape(-1,2)
    def energy(values):return float(np.square(array(values)).sum(1).mean()) if len(values) else None
    pe={k:energy(v) for k,v in point.items()};result=dict(definition='target-dependent orthogonal per-block mean-error/shape-error decomposition, NOT an inference fix; squared vector energy, not per-axis MSE',points=len(point['error']),point_energy=pe,
        point_block_mean_energy_fraction=pe['block_mean']/pe['error'] if pe['error'] else None,links={})
    for name,entry in links.items():
        ee,bb,ss=energy(entry['error']),energy(entry['block_mean']),energy(entry['shape'])
        left,right=array(entry['left']),array(entry['right']);denom=np.sqrt(np.square(left).sum(1).mean()*np.square(right).sum(1).mean()) if len(left) else 0
        row=dict(links=len(entry['error']),segment_error_energy=ee,block_mean_difference_energy=bb,shape_difference_energy=ss,
            twice_cross_term=float(2*(array(entry['block_mean'])*array(entry['shape'])).sum(1).mean()) if ee is not None else None,
            adjacent_error_uncentered_correlation=float((left*right).sum(1).mean()/denom) if denom else None)
        # RDP link lengths are nonuniform. Bin by target length, NEVER index/time velocity.
        lengths=np.linalg.norm(array(entry['target']),axis=1)
        if len(lengths):
            bounds=np.quantile(lengths,[0,.25,.5,.75,1]);labels=np.searchsorted(bounds[1:-1],lengths,side='right');errors=array(entry['error'])
            row['target_length_quartile_bounds']=bounds.tolist();row['target_length_bins']=[dict(bin=i,links=int((labels==i).sum()),segment_error_rmse=float(np.sqrt(np.square(errors[labels==i]).sum(1).mean())) if (labels==i).any() else None) for i in range(4)]
        result['links'][name]=row
    return result
