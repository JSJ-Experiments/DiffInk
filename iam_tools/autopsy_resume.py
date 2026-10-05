"""Strict source/config identity checks and AdamW-state restoration for geometry continuation."""
import hashlib
from pathlib import Path
import statistics
import torch

RESUME_ONLY_KEYS={'base_lr','initialization','output_base','resume_checkpoint','resume_checkpoint_sha256','resume_step','preflight_file'}


def validate_resume(checkpoint,cfg,digest,sample_digest):
    if digest!=cfg['resume_checkpoint_sha256']:raise ValueError('source checkpoint SHA256 mismatch')
    if checkpoint['step']!=cfg['resume_step'] or checkpoint['step']!=1000:raise ValueError('resume must start at step 1000')
    if checkpoint.get('sample_sha256')!=sample_digest:raise ValueError('resume sample identity mismatch')
    if cfg['base_lr']!=1e-5 or checkpoint['config']['base_lr']!=1.5e-4:raise ValueError('authorized LR change is 1.5e-4 -> 1e-5')
    original=checkpoint['config']
    keys=(set(cfg)|set(original))-RESUME_ONLY_KEYS
    changes={key:(original.get(key),cfg.get(key)) for key in keys if original.get(key)!=cfg.get(key)}
    if changes:raise ValueError(f'resume changes more than LR/metadata: {changes}')
    if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint:raise ValueError('model AND optimizer state required')
    states=checkpoint['optimizer_state_dict']['state']
    if not states or any(float(s['step'])!=1000 for s in states.values()):raise ValueError('source AdamW state is incomplete or not at step 1000')


def load_resume(cfg,sample_digest,override=None):
    if cfg['initialization']=='fresh-seed-42':return None
    path=Path(override or cfg['resume_checkpoint'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    validate_resume(checkpoint,cfg,digest,sample_digest)
    return checkpoint


def restore_optimizer(optimizer,checkpoint,new_lr):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # load_state_dict restores the OLD LR too; override only after restoring moments/steps.
    for group in optimizer.param_groups:group['lr']=new_lr
    return {'adam_step_min':min(float(s['step']) for s in optimizer.state.values()),
            'adam_step_max':max(float(s['step']) for s in optimizer.state.values()),
            'state_entries':len(optimizer.state),'learning_rates':[g['lr'] for g in optimizer.param_groups]}


def clipping_summary(rows,cap):
    if not rows:return {'steps':0,'fraction':None,'median_raw_norm':None,'mean_raw_norm':None,'max_raw_norm':None}
    norms=[r['gradient_norm'] for r in rows]
    return {'steps':len(rows),'fraction':sum(x>cap for x in norms)/len(norms),
            'median_raw_norm':statistics.median(norms),'mean_raw_norm':statistics.mean(norms),'max_raw_norm':max(norms)}
