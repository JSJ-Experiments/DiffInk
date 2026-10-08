"""Decompose chronological displacement error by true pen topology.

Raw index displacements are NOT physical velocities. Pen-up jumps matter for
where writing resumes, but are not drawn curves. This is descriptive accounting,
not proof that reweighting improves handwriting and not generic smoothing.
"""
import numpy as np


def offset_error_groups(target,predicted_xy,stats):
    target=np.asarray(target,dtype=float)
    if target.ndim!=2 or target.shape[1]!=5 or not len(target) or not np.isfinite(target).all():raise ValueError('finite nonempty targetN,5 required')
    states=target[:,2:].argmax(-1)
    if not np.array_equal(target[:,2:],np.eye(3)[states]) or states[-1]!=2 or (states[:-1]==2).any():raise ValueError('true English one-hot final-only EOC contract required')
    mean=np.asarray(stats['mean'],dtype=float);std=np.asarray(stats['std'],dtype=float)
    if mean.shape!=(2,) or std.shape!=(2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('finite2-axis normalization, positive std required')
    delta=np.diff(np.r_[np.zeros((1,2)),target[:,:2]],axis=0)
    if predicted_xy is None:error=(delta-mean)/std
    else:
        p=np.asarray(predicted_xy,dtype=float)
        if p.shape!=(len(target),2) or not np.isfinite(p).all():raise ValueError('finite matching predicted absolute XY required')
        error=(np.diff(np.r_[np.zeros((1,2)),p],axis=0)-delta)/std
    energy=np.square(error).sum(-1);first=np.arange(len(target))==0;previous=np.r_[2,states[:-1]]
    masks={'origin':first,'within_stroke':~first&(previous==0),'pen_up_jump':~first&(previous==1),'block_phase0':np.arange(len(target))%8==0}
    if not np.all(masks['origin']|masks['within_stroke']|masks['pen_up_jump']):raise ValueError('exhaustive real point topology required')
    return {name:dict(points=int(mask.sum()),energy=float(energy[mask].sum())) for name,mask in masks.items()}


def aggregate_groups(rows):
    if not rows:raise ValueError('nonempty scope required')
    names=('origin','within_stroke','pen_up_jump','block_phase0')
    if any(set(r)!=set(names) for r in rows):raise ValueError('consistent group schema required')
    result={k:dict(points=sum(r[k]['points'] for r in rows),energy=sum(r[k]['energy'] for r in rows)) for k in names}
    total=sum(result[k]['energy'] for k in names[:3]);count=sum(result[k]['points'] for k in names[:3])
    for row in result.values():
        if row['points']<0 or row['energy']<0 or not np.isfinite(row['energy']):raise ValueError('finite nonnegative error accounting required')
        row['fraction_of_total_energy']=row['energy']/total if total else 0.
        row['per_axis_mse']=row['energy']/row['points']/2 if row['points'] else None
    return dict(groups=result,points=count,total_energy=total,normalized_per_axis_mse=total/count/2,
                partition='origin / inside true stroke / move after truepen-up are disjoint and exhaustive; block_phase0 overlaps them',
                terminology='normalized index-displacement squared error, not physical velocity; pen-up movements are not inked segments; no causal reweighting claim')
