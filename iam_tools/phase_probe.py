"""Train-only fixed mod8 residual-bias probe (not smoothing/model training)."""
import json
from pathlib import Path
import numpy as np
import h5py
from .trajectory_geometry import geometry_metrics
from .report_curve_study import aggregate


def fit_phase_bias(lines):
    if not lines:raise ValueError('nonempty training lines required')
    sums=np.zeros((8,2));weights=np.zeros(8)
    for line in lines:
        phases=line['mu']['geometry']['point_index_mod8'];n=sum(p['count'] for p in phases)
        for p in phases:
            w=p['count']/n;sums[p['phase']]+=w*np.array(p['mean_residual']);weights[p['phase']]+=w
    if (weights==0).any():raise ValueError('all phases must be represented')
    bias=sums/weights[:,None];global_bias=(bias*weights[:,None]).sum(0)/weights.sum()
    return bias-global_bias,global_bias


def run(directory,root='data'):
    directory=Path(directory);root=Path(root);info=json.loads((directory/'report/summary.json').read_text())
    step=info['selected_step'];splits=info['provenance']['splits']
    row=json.loads((directory/f'eval-{step}.json').read_text())
    bias,global_bias=fit_phase_bias([r for r in row['lines'] if r['sample_id'] in splits['train']])
    out=directory/'phase-bias-probe';out.mkdir(exist_ok=True);rows=[]
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5') as tr,h5py.File(root/'diffink/iam_overfit/tiny_val.h5') as va:
        for line in row['lines']:
            sid=line['sample_id'];a=np.load(directory/f'step-{step}/{sid}/mu.npy');truth=(tr if sid in tr else va)[sid]['point_seq'][:]
            xy=a[:,:2]-bias[np.arange(len(a))%8];t=truth[:,:2]*.01;states=truth[:,2:].argmax(1)
            np.save(out/f'{sid}.npy',np.column_stack([xy,a[:,2:]]))
            rows.append(dict(sample_id=sid,baseline=dict(geometry=line['mu']['geometry']),
                             corrected=dict(geometry=geometry_metrics(xy,t,states))))
    result=dict(source_checkpoint_sha256=info['selected_checkpoint_sha256'],bias=bias.tolist(),global_bias_not_removed=global_bias.tolist(),
                method='fixed train-only mod8 position bias, zero-centered; no smoothing or model changes; mean-only diagnostic',groups={})
    for group,ids in splits.items():result['groups'][group]={k:aggregate([r[k] for r in rows if r['sample_id'] in ids]) for k in ('baseline','corrected')}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
