"""Reconstruct exact CUDA text-drop draws from parent state without new training."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np,torch
from .writer_expansion import training_schedule
from .pen_ab import file_sha


def summarize_rows(rows,counts):
    groups={}
    for label,wanted in [('no_null',False),('has_null',True)]:
        group=[r for r in rows if (counts[r['step']]>0)==wanted]
        groups[label]=dict(updates=len(group),**{k:float(np.mean([r[k] for r in group])) if group else None for k in ['base_mse','physical_xy_mse','segment_mse','gradient_norm']})
    return groups


def audit(directory,root='/data',output_name='dropout-replay.json'):
    if Path(output_name).name!=output_name or not output_name.endswith('.json'):raise ValueError('bounded diagnostic filename required')
    if not torch.cuda.is_available():raise ValueError('CUDA RNG replay requires CUDA; CPU same seed is not equivalent')
    directory=Path(directory);root=Path(root);cfg=json.loads((directory/'config.json').read_text())
    if file_sha(root/cfg['parent'])!=cfg['parent_sha256']:raise ValueError('immutable parent drift')
    saved=torch.load(root/cfg['parent'],map_location='cpu',weights_only=False);rng=torch.Generator(device='cuda');rng.set_state(saved['noise_rng_state'])
    with h5py.File(root/cfg['data_parent']/'source.h5') as f:lengths={i:f[i]['latent_mean'].shape[0] for i in cfg['splits']['train']}
    schedule=list(training_schedule(cfg['splits']['train'],cfg['schedule_offset']+cfg['max_updates'],8,seed=cfg['schedule_seed']))[cfg['schedule_offset']:]
    counts={};shas={}
    for step,ids in enumerate(schedule,1):
        epsilon=torch.randn((8,max(lengths[i] for i in ids),384),device='cuda',generator=rng);t=torch.randint(0,1000,(8,),device='cuda',generator=rng);drop=torch.rand(8,device='cuda',generator=rng)<cfg['text_drop_probability']
        counts[step]=int(drop.sum())
        if step==1 or step%200==0:shas[step]=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest()
    result={};verified=0
    for arm in cfg['arms']:
        rows=[json.loads(l) for l in (directory/arm/'metrics.jsonl').read_text().splitlines()]
        for r in rows:
            if r['sample_ids']!=schedule[r['step']-1]:raise ValueError('batch replay mismatch')
            if 'noise_rng_sha256' in r:
                if r['noise_rng_sha256']!=shas[r['step']]:raise ValueError('GPU RNG replay mismatch')
                verified+=1
        result[arm]=summarize_rows(rows,counts)
    out=directory/'diagnostics';out.mkdir(exist_ok=True);payload=dict(groups=result,periodic_gpu_rng_checks=verified,scope='exact GPUdropoutdraw replay; association between null-containing batches and batch loss/gradient, NOT alone proof of harmful gradient interference',counts_by_step=counts)
    p=out/output_name
    if p.exists():raise ValueError('refuse overwrite existing replay')
    p.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k!='counts_by_step'},indent=2),flush=True);return payload
