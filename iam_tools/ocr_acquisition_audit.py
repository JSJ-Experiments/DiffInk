"""Descriptive pen-boundary backward-jump proxy; NOT character alignment labels."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np
from .ocr_pool_study import load_pool
from .pen_ab import file_sha


def backward_jumps(points,scale=.01,threshold=.5):
    a=np.asarray(points)
    if a.ndim!=2 or a.shape[1]!=5 or not np.isfinite(a).all() or not np.isfinite([scale,threshold]).all() or scale<=0 or threshold<=0:
        raise ValueError('finite N×5 points and positive model scale/threshold required')
    states=a[:,2:].argmax(1);boundary=states[:-1]!=0;dx=np.diff(a[:,0])*scale
    return int(((dx < -threshold)&boundary).sum())


def audit(root,pool_sha,evaluation,out):
    pool,m,_=load_pool(root,pool_sha);evaluation=Path(evaluation);row=json.loads(evaluation.read_text());by={r['sample_id']:r for r in row['lines']}
    if set(by)!=set(m['records']) or len(by)!=len(row['lines']):raise ValueError('complete unique pool mean evaluation required')
    result=dict(definition='count pen-boundary jumps nextX-previousX < -0.5 line-height, chronological RDP index. Descriptive non-monotonic acquisition proxy, NOT proof of delayed dots/label errors/causation.',
        threshold_model_units=.5,model_scale=.01,pool_manifest_sha256=pool_sha,evaluation_sha256=file_sha(evaluation),step=row['step'],groups={},lines=[])
    with h5py.File(pool/'lines.h5') as hf:
        for group in ('large_train','dev','held_out'):
            bins={0:[],1:[]}
            for sid in m['splits'][group]:
                a=hf[sid]['point_seq'][:]
                if hashlib.sha256(a.tobytes()).hexdigest()!=m['records'][sid]['points_sha256']:raise ValueError('point fingerprint changed: '+sid)
                count=backward_jumps(a);r=by[sid]['mu'];bins[int(count>0)].append(r)
                result['lines'].append(dict(sample_id=sid,large_backward_pen_jumps=count,errors=r['errors'],characters=r['characters']))
            result['groups'][group]={str(k):dict(lines=len(v),errors=sum(r['errors'] for r in v),characters=sum(r['characters'] for r in v),
                cer=sum(r['errors'] for r in v)/sum(r['characters'] for r in v) if v else None) for k,v in bins.items()}
    Path(out).write_text(json.dumps(result,indent=2)+'\n');return result
