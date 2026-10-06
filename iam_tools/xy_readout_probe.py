"""CPU frozen-feature linear XY probe; TRAIN equations only, no model updates.

Matches point and true-stroke first differences. Diagnoses readout vs feature
limitations; not a probabilistic decoder or proposed production checkpoint.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .writer_expansion import load,device_batches
from .curve_study import forward_xy
from .trajectory_geometry import geometry_metrics
from .report_curve_study import aggregate
from .latent_integration import DELTA_WEIGHT


def fit(features,targets,states,weight=DELTA_WEIGHT):
    if not (len(features)==len(targets)==len(states)) or not features or weight<0:raise ValueError('matching nonempty training equations required')
    aa=[];bb=[]
    for f,t,s in zip(features,targets,states):
        f=np.asarray(f,dtype=np.float64);t=np.asarray(t,dtype=np.float64);s=np.asarray(s)
        if f.ndim!=2 or t.shape!=(len(f),2) or s.shape!=(len(f),) or len(f)<3:raise ValueError('feature/target/state shapes')
        if not np.isfinite(f).all() or not np.isfinite(t).all() or not np.isin(s,[0,1,2]).all():raise ValueError('finite coordinates/features and valid states required')
        a=np.column_stack([f,np.ones(len(f))]);edge=s[:-1]==0
        aa.append(a/np.sqrt(len(a)));bb.append(t/np.sqrt(len(a)))
        if edge.any() and weight:
            aa.append(np.sqrt(weight/edge.sum())*np.diff(a,axis=0)[edge]);bb.append(np.sqrt(weight/edge.sum())*np.diff(t,axis=0)[edge])
    # QR/SVD least squares in float64; no held-out equations or regularization search.
    a=np.concatenate(aa);b=np.concatenate(bb);coef,residual,rank,singular=np.linalg.lstsq(a,b,rcond=1e-6)
    return coef,dict(equations=len(a),columns=a.shape[1],rank=int(rank),delta_weight=weight,
                     effective_condition_number=float(singular[0]/singular[rank-1]),rcond=1e-6)


def run(config,repo,root,source_rel,source_sha,output):
    torch.set_num_threads(4)
    model,samples,raw,cfg,vocab,provenance=load(config,repo,root,source_rel,source_sha)
    before={k:v.clone() for k,v in model.state_dict().items()};cache={};captured=[]
    hook=model.transformer_decoder.fc.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach()))
    try:
        with torch.no_grad():
            for sid,(raw,mask,_) in device_batches(raw,'cpu').items():
                xy,truth,out=forward_xy(model,raw,mask);n=int(mask.sum())
                cache[sid]=dict(features=captured.pop()[0,:n].numpy(),target=truth[0,:n].numpy(),
                                states=raw[0,2:,:n].argmax(0).numpy(),xy=xy[0,:n].numpy(),pens=out[0,:3,:n].argmax(0).numpy())
    finally:hook.remove()
    ids=provenance['splits']['train'];coef,diag=fit([cache[i]['features'] for i in ids],[cache[i]['target'] for i in ids],[cache[i]['states'] for i in ids])
    rows=[];output=Path(output);output.mkdir(parents=True,exist_ok=True);np.save(output/'linear-coefficients.npy',coef)
    for sid,e in cache.items():
        xy=np.column_stack([e['features'],np.ones(len(e['features']))])@coef
        np.save(output/f'{sid}.npy',np.column_stack([xy,np.eye(3)[e['pens']]]))
        rows.append(dict(sample_id=sid,baseline=dict(geometry=geometry_metrics(e['xy'],e['target'],e['states'])),
                         probe=dict(geometry=geometry_metrics(xy,e['target'],e['states']))))
    report=dict(source_sha256=source_sha,provenance=provenance,diagnostics=diag,model_bitwise_unchanged=all(torch.equal(before[k],v) for k,v in model.state_dict().items()),groups={},lines=rows)
    for group,ids in provenance['splits'].items():
        report['groups'][group]={kind:aggregate([r[kind] for r in rows if r['sample_id'] in ids]) for kind in ('baseline','probe')}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
