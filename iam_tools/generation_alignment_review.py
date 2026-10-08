"""Fail-closed protocol validation for the fresh attention intervention."""
import collections, hashlib, json, tarfile
from pathlib import Path
import numpy as np
import torch
from .generation_alignment_study import ARMS, REFERENCE, REFERENCE_SHA
from .generation_capacity import DATA, SOURCE_H5_SHA, DATASET_SHA, WHITENING_SHA, select_capacity
from .generation_composition import fit_duration, learning_rate, seal_confirmation, seal_synthetic
from .generation_coverage_study import score
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha

AS_RUN = ('iam_tools/generation_alignment.py','iam_tools/generation_alignment_study.py',
          'iam_tools/generation_composition.py','iam_tools/generation_duration_eval.py',
          'iam_tools/generation_coverage.py','iam_tools/generation_coverage_study.py',
          'iam_tools/generation_cache.py','iam_tools/generation_study.py',
          'iam_tools/generation_geometry.py','iam_tools/latent_diffusion.py',
          'model/vae.py','model/losses.py')


def verify(directory, cfg, results, data, *, continuation=False):
    """Check actual logs/weights/evaluations, not just declarative config flags."""
    p=Path(directory); trains=data['training_ids']; schedules={}; initial={}; rng={}; exposure={}
    if set(results)!=set(ARMS) or set(trains)!=set(ARMS):raise ValueError('exact two-arm guard')
    if not continuation and (not cfg['fresh'] or cfg['parent_step']!=0):raise ValueError('fresh model guard')
    if continuation and (cfg['fresh'] or cfg['parent_step']!=24000):raise ValueError('bounded continuation guard')
    if trains['global']!=trains['soft_gaussian']:raise ValueError('matched TRAIN guard')
    models=cfg['models']
    if models['global']['alignment'] is not False or models['soft_gaussian']['alignment'] is not True or {k:v for k,v in models['global'].items() if k!='alignment'}!={k:v for k,v in models['soft_gaussian'].items() if k!='alignment'}:raise ValueError('only intended alignment bundle changes')
    for a,r in results.items():
        if r['last_step']!=cfg['max_updates'] or not r['codec_reader_unchanged'] or r['stop']!='budget_completed':raise ValueError('complete matched budget/frozen guard')
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()]
        if len(rows)!=r['last_step'] or [x['step'] for x in rows]!=list(range(1,r['last_step']+1)):raise ValueError('actual update sequence guard')
        if any(len(x['sample_ids'])!=cfg['batch'] or not set(x['sample_ids'])<=set(trains[a]) for x in rows):raise ValueError('actual TRAIN scope guard')
        if any(x['learning_rates']!=[cfg['lr'] if continuation else learning_rate(x['step'],cfg['max_updates'])] for x in rows):raise ValueError('actual learning-rate guard')
        if any(not np.isfinite(x[k]) for x in rows for k in ('loss','gradient_norm')):raise ValueError('finite optimizer guard')
        schedules[a]=[x['sample_ids'] for x in rows]
        if hashlib.sha256(json.dumps(schedules[a]).encode()).hexdigest()!=r['schedule_sha256']:raise ValueError('schedule hash drift')
        counts=collections.Counter(i for row in rows for i in row['sample_ids']); values=[counts[i] for i in trains[a]]
        exposure[a]=dict(min=min(values),median=float(np.median(values)),max=max(values),total=sum(values))
        if not min(values):raise ValueError('all TRAIN exposure required')
        saved=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        opt=saved['optimizer_state_dict']
        if saved['step']!=0 or bool(opt['state'])!=continuation:raise ValueError('step0 empty/restored Adam guard')
        if any(g['lr']!=1e-5 or list(g['betas'])!=cfg['betas'] or g['weight_decay']!=cfg['weight_decay'] for g in opt['param_groups']):raise ValueError('initial actual Adam groups mismatch')
        initial[a]=tensor_digest(saved['model_state_dict'])
        if initial[a]!=r['initial_state_sha256']:raise ValueError('initial digest drift')
        rng[a]={k:saved[k] for k in ('torch_rng_state','cuda_rng_state')}
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(p/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
        history=r['history']
        if [h['step'] for h in history]!=cfg['eval_steps']:raise ValueError('all scheduled evaluations required')
        if r['best_step']!=min(history,key=lambda h:h['train_score'])['step']:raise ValueError('TRAIN-only selection guard')
        for h in history:
            ev=json.loads((p/a/f'eval-{h["step"]}.json').read_text());ids=[row['sample_id'] for row in ev['lines'] if row['policy']=='correct' and row['sample_id'] in trains[a]]
            if len(ids)!=len(trains[a]) or set(ids)!=set(trains[a]) or abs(score(ev,set(trains[a]))-h['train_score'])>1e-9:raise ValueError('EVERY actual TRAIN required for selection')
            if file_sha(p/a/f'evaluation-{h["step"]}.h5')!=ev['packed_h5_sha256']:raise ValueError('packed evaluation SHA drift')
            de=json.loads((p/a/f'duration-eval-{h["step"]}.json').read_text())
            if de['ids']!=data['splits']['unseen_prompt'] or file_sha(p/a/f'duration-evaluation-{h["step"]}.h5')!=de['packed_h5_sha256']:raise ValueError('duration evaluation integrity drift')
    if schedules['global']!=schedules['soft_gaussian'] or (not continuation and len(set(initial.values()))!=1):raise ValueError('identical minibatches/fresh weights required')
    if any(not torch.equal(rng['global'][k],rng['soft_gaussian'][k]) for k in rng['global']):raise ValueError('identical initial RNG required')
    return dict(complete_matched_budgets=True,identical_fresh_weights_empty_adam_rng=not continuation,identical_actual_minibatches_lrs=True,full_actual_train_selection=True,frozen_codec_reader=True,exposure=exposure,initial_state_sha256=initial)


def verify_sources(directory,repo,root,cfg,data,results):
    p=Path(directory);root=Path(root)
    if cfg.get('continuation_of'):return verify_continuation_sources(p,repo,root,cfg,data,results)
    source=root/DATA
    for name,sha in [('source.h5',SOURCE_H5_SHA),('dataset.json',DATASET_SHA),('whitening.pt',WHITENING_SHA)]:
        if file_sha(source/name)!=sha:raise ValueError('immutable source drift: '+name)
    if cfg['reference_config_only']!=REFERENCE or cfg['reference_sha256']!=REFERENCE_SHA or file_sha(root/REFERENCE)!=REFERENCE_SHA:raise ValueError('reference config drift')
    reference=torch.load(root/REFERENCE,map_location='cpu',weights_only=False)
    if cfg['auxiliary_weights']!=reference['config']['auxiliary_weights']:raise ValueError('fixed inherited anchors required')
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:raise ValueError('as-run archive drift')
    with tarfile.open(p/'as-run-source.tar.gz') as tar:
        for name in AS_RUN:
            if tar.extractfile(name).read()!=(Path(repo)/name).read_bytes():raise ValueError('as-run implementation drift: '+name)
    original=json.loads((source/'dataset.json').read_text());scope=select_capacity(original);scope['training_ids']={a:scope['training_ids']['larger256'] for a in ARMS}
    if data['splits']!=scope['splits'] or data['training_ids']!=scope['training_ids'] or any(r!=original['records'][i] for i,r in data['records'].items()):raise ValueError('reproducible TRAIN/development selector drift')
    duration=fit_duration(data['records'],data['training_ids']['global'],cfg['writers'])
    for k,v in duration.items():
        if isinstance(v,list) and v and isinstance(v[0],(int,float)):
            if not np.allclose(v,cfg['duration_model'][k],rtol=1e-10,atol=1e-10):raise ValueError('TRAIN-only duration fit drift')
        elif v!=cfg['duration_model'][k]:raise ValueError('TRAIN-only duration scope drift')
    from .ocr_pool_study import load_pool
    from .ocr_convergence import POOL_SHA
    pool,manifest,vocab=load_pool(root,POOL_SHA)
    if vocab!=cfg['vocab']:raise ValueError('clean alphabet drift')
    seal=seal_confirmation(manifest,original['records'],scope['training_ids']['global'],cfg['writers'],development_ids=scope['splits']['unseen_prompt'])
    if seal!=data['confirmation_seal'] or seal_synthetic(original['records'],scope['training_ids']['global'],cfg['writers'])!=data['synthetic_seal']:raise ValueError('predeclared metadata seals drift')
    guards=verify(p,cfg,results,data)
    from .generation_alignment import AlignedWriterDenoiser
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['model_seed']);fresh=AlignedWriterDenoiser(**cfg['models']['global'])
    # CPU-only versus CUDA wheels can differ ~1.6e-6 in normal-init math.
    # Actual arm-to-arm weights remain independently BITWISE equal above.
    differences=[]
    for a in ARMS:
        initial=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)['model_state_dict']
        if set(initial)!=set(fresh.state_dict()) or any(not torch.allclose(v,initial[k],atol=2e-6,rtol=1e-6) for k,v in fresh.state_dict().items()):raise ValueError('independently regenerated fresh seed/weights mismatch')
        differences.append(max(float((v-initial[k]).abs().max()) for k,v in fresh.state_dict().items()))
    guards['fresh_initialization_independently_regenerated']=True
    guards['fresh_regeneration_max_difference']=max(differences)
    guards['fresh_regeneration_tolerance']='atol2e-6,rtol1e-6 across CPU/CUDA wheel initialization; actual arm equality remains bitwise'
    guards.update(as_run_source_bytes_verified=True,train_only_duration_refit_verified=True,sealed_prompts_reselected_identically=True,historical_confirmation_form_exposure_disclosed=True)
    return guards,pool


def verify_continuation_sources(p,repo,root,cfg,data,results):
    rel=cfg['continuation_of']
    if not rel.startswith('checkpoints/iam_generation_alignment/') or '..' in Path(rel).parts:raise ValueError('bounded fresh parent path')
    parent=Path(root)/rel;pc=json.loads((parent/'config.json').read_text());pd=json.loads((parent/'dataset.json').read_text());pr=json.loads((parent/'result.json').read_text());original,pool=verify_sources(parent,repo,root,pc,pd,pr)
    if file_sha(parent/'result.json')!=cfg['parent_result_sha256'] or data!=pd:raise ValueError('immutable parent/data/seals required')
    if any(cfg[k]!=pc[k] for k in ('models','auxiliary_weights','duration_model','vocab','writers','source_rel','source_sha256','reader_rel','reader_sha256')):raise ValueError('continuation contract drift')
    if not any(r['history'][-1]['aggregate']['all_train256']['correct']['free_cer']>.1 for r in pr.values()):raise ValueError('TRAIN-only extension trigger')
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:raise ValueError('continuation archive SHA drift')
    with tarfile.open(p/'as-run-source.tar.gz') as tar:
        for name in AS_RUN+('iam_tools/generation_alignment_continuation.py',):
            if tar.extractfile(name).read()!=(Path(repo)/name).read_bytes():raise ValueError('continuation as-run source drift: '+name)
    guards=verify(p,cfg,results,data,continuation=True)
    def moments(opt):return tensor_digest({f'{i}:{k}':v for i,state in opt['state'].items() for k,v in state.items()})
    for a in ARMS:
        old=torch.load(parent/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);new=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        if file_sha(parent/a/'checkpoint-last.pt')!=cfg['parent_checkpoints'][a] or tensor_digest(old['model_state_dict'])!=tensor_digest(new['model_state_dict']) or moments(old['optimizer_state_dict'])!=moments(new['optimizer_state_dict']):raise ValueError('own-arm source weights/full Adam equality required')
        if any(not torch.equal(new[k],old[k]) for k in ('torch_rng_state','cuda_rng_state')):raise ValueError('own-arm source RNG equality required')
    guards.update(original_fresh_study_guards=original,own_arm_model_adam_rng_restored=True,seals_unchanged=True,confirmation_not_used_for_training=True,as_run_source_bytes_verified=True)
    return guards,pool
