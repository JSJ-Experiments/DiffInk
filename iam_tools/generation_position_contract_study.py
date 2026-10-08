"""Matched fresh positional-contract study, 48000 updates/arm on separate T4s.

No confirmation evaluated here. Exposed development is descriptive only. All
sources remain packed; each GPU writes ONLY its own arm directory on the Volume.
"""
import copy
import hashlib
import json
import tarfile
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from .generation_position_contract import PositionContractWriter, contract_lr
from .generation_timing_study import PARENT
from .generation_capacity import DATA, SOURCE_H5_SHA, DATASET_SHA, WHITENING_SHA, select_capacity
from .generation_composition import fit_duration
from .generation_cache import CachedLatentPool
from .generation_coverage_study import evaluate, score, writer_tensor
from .generation_duration_eval import evaluate_duration
from .generation_geometry import physical_terms
from .latent_diffusion import masked_mse
from .writer_expansion import load, training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor

ARMS = ('relative100', 'absolute')


def verify_duration_refit(refit, original, records, ids):
    """BLAS refits differ at ~1e-14; metadata and actual durations remain exact."""
    from .generation_composition import predict_duration
    if set(refit) != set(original):
        raise ValueError('TRAIN-only duration keys differ')
    for k, v in refit.items():
        other = original[k]
        if isinstance(v, list) and v and type(v[0]) in (float, int):
            if len(v) != len(other) or not np.allclose(v, other, rtol=1e-10, atol=1e-10):
                raise ValueError('TRAIN-only numerical duration refit drift: '+k)
        elif v != other:
            raise ValueError('TRAIN-only duration metadata drift: '+k)
    if any(predict_duration(refit, records[s]['text'], records[s]['writer_id']) !=
           predict_duration(original, records[s]['text'], records[s]['writer_id']) for s in ids):
        raise ValueError('actual rounded duration requests differ')


def checked_path(relative, root):
    p = Path(relative)
    if not relative.startswith('checkpoints/iam_generation_position_contract/') or p.is_absolute() or '..' in p.parts:
        raise ValueError('bounded positional-contract experiment path required')
    return Path(root) / p


def prepare(repo, root='/data'):
    root = Path(root); source = root / DATA
    for name, sha in [('source.h5', SOURCE_H5_SHA), ('dataset.json', DATASET_SHA), ('whitening.pt', WHITENING_SHA)]:
        if file_sha(source/name) != sha:
            raise ValueError('immutable coverage source drift: '+name)
    original = json.loads((source/'dataset.json').read_text())
    old = json.loads((root/PARENT/'config.json').read_text())
    scope = select_capacity(original)
    train = scope['training_ids']['larger256']
    scope['training_ids'] = {a:list(train) for a in ARMS}
    ids = sorted(set(scope['splits']['all_train256'] + scope['splits']['unseen_prompt']))
    records = {sid: original['records'][sid] for sid in ids}
    writers = original['writers']
    duration = fit_duration(records, train, writers)
    verify_duration_refit(duration, old['duration_model'], records, ids)
    duration = copy.deepcopy(old['duration_model'])  # Retain pinned reference bytes after independent refit.
    from .generation_alignment_study import REFERENCE, REFERENCE_SHA
    if file_sha(root/REFERENCE) != REFERENCE_SHA:
        raise ValueError('fixed config/anchor reference required, not neural initialization')
    reference = torch.load(root/REFERENCE, map_location='cpu', weights_only=False)['config']
    spec = dict(reference['model'], alignment=True, blocks_per_character=duration['blocks_per_character'], prior_sigma=2.5, prior_cap=12.)
    out = root/'checkpoints/iam_generation_position_contract'/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz', 'w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base = Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for p in sorted(base.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'):
                    tar.add(p, arcname=directory+'/'+str(p.relative_to(base)))
    cfg = dict(profile='fresh paired relative100 vs absolute positional contract; standalone mapper NOT semantic InkVAE/InkDiT/TrInk',
        fresh=True,parent_step=0,neural_checkpoint_initialization=False,
        data_parent=DATA,source_h5_sha256=SOURCE_H5_SHA,data_manifest_sha256=DATASET_SHA,whitening_sha256=WHITENING_SHA,
        reference_config_only=REFERENCE,reference_sha256=REFERENCE_SHA,
        vocab=old['vocab'],writers=writers,models={a:dict(spec,position_policy=a) for a in ARMS},
        source_rel=old['source_rel'],source_sha256=old['source_sha256'],reader_rel=old['reader_rel'],reader_sha256=old['reader_sha256'],
        batch=8,betas=[.9,.99],weight_decay=.01,clip=1.,max_updates=48000,
        eval_steps=[0,8000,16000,24000,36000,48000],max_train_wall_seconds_per_arm=2400,
        model_seed=28142,schedule_seeds=[29142,39142],lr_schedule='same fresh24000 warmup/hold/cosine then24000 constant1e-5, full Adam retained',
        auxiliary_weights=reference['auxiliary_weights'],duration_model=duration,
        source_archive_sha256=file_sha(out/'as-run-source.tar.gz'),selection='all256 native TRAIN geometry only',
        confirmation_policy='No opened paired/synthetic confirmation reused; exposed olddev8 descriptive only; fresh blind gate required after TRAIN-selected timing evaluation',
        hypothesis='remove length-dependent high-frequency query PE at initialization, keep absolute index PE/soft Gaussian prior/text amplitudes/architecture/targets fixed',
        limitations='absolute query features length-independent but self-attention remains context-dependent; oracle TRAIN mask still leaks duration; corpus-familiar reader; one seed; only256TRAIN; no generic smoothing or target resampling',
        torch_cpu_threads=2,not_promoted=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'dataset.json').write_text(json.dumps(dict(records=records,writers=writers,**scope),indent=2)+'\n')
    return str(out.relative_to(root))


def run(arm, relative, repo, root='/data'):
    if arm not in ARMS or not torch.cuda.is_available():
        raise ValueError('explicit matched positional arm and T4 required')
    root=Path(root);out=checked_path(relative,root);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text())
    if cfg['max_updates']!=48000 or cfg['models'][arm]['position_policy']!=arm:
        raise ValueError('immutable bounded protocol required')
    if file_sha(out/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:
        raise ValueError('immutable as-run source required')
    torch.set_num_threads(cfg['torch_cpu_threads'])
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda')
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    latents={};targets={};records=data['records'];source=root/DATA
    if file_sha(source/'source.h5')!=SOURCE_H5_SHA or file_sha(source/'whitening.pt')!=WHITENING_SHA:
        raise ValueError('frozen data drift')
    with h5py.File(source/'source.h5') as f:
        for sid in records:
            latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(source/'whitening.pt',weights_only=True)
    stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    pool=CachedLatentPool(latents,records,cfg['vocab'],sorted(records),stats,max_lines=264)
    torch.manual_seed(cfg['model_seed']);model=PositionContractWriter(**cfg['models'][arm]).cuda()
    torch.manual_seed(cfg['model_seed'])  # Same post-prototype RNG as previous fresh study.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
    initial=tensor_digest(model.state_dict());folder=out/arm;folder.mkdir(exist_ok=False)
    train=data['training_ids'][arm]
    schedule=[batch for seed in cfg['schedule_seeds'] for batch in training_schedule(train,24000,cfg['batch'],seed=seed)]
    history=[];durations=[];seconds=0.;clipped=0;best_step=0;stop='budget_completed'
    with ResourceMonitor(folder,interval=5,sustained_seconds=30) as monitor:
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=arm,model=cfg['models'][arm]),step=step,
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),folder/name)
        def assess(step):
            with monitor.in_phase('eval/'+arm):
                ev=evaluate(model,codec,reader,pool,latents,records,cfg['vocab'],stats,data['splits'],targets,folder,step,cfg['writers'],controls=step==48000)
                de=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['all_train256'],folder,step,cfg['writers'],cfg['duration_model'],controls=step==48000)
                dev=folder/'exposed-development-duration';dev.mkdir(exist_ok=True)
                dd=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['unseen_prompt'],dev,step,cfg['writers'],cfg['duration_model'])
                s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']))
                durations.append(dict(step=step,train=de['aggregate'],exposed_development=dd['aggregate']))
            return s
        best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        monitor.set_phase('train/'+arm);model.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic()
                for g in opt.param_groups:g['lr']=contract_lr(step)
                clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],'cuda')
                pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi)
                terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);w=cfg['auxiliary_weights']
                loss=mse+w['xy']*terms['xy']+w['first_difference']*terms['first_difference']
                opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,sample_ids=batch,learning_rates=[g['lr'] for g in opt.param_groups],loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'],text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()))
                log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped']
                if step%2000==0:
                    print(dict(arm=arm,**row,train_seconds=seconds),flush=True);log.flush()
                if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite positional study')
                limit=seconds>cfg['max_train_wall_seconds_per_arm']
                if step in cfg['eval_steps'] or limit:
                    s=assess(step);save('checkpoint-last.pt',step)
                    if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):
            raise ValueError('frozen codec/reader drift')
    result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,duration_history=durations,stop=stop,train_seconds=seconds,
        clip_fraction=clipped/step,initial_state_sha256=initial,codec_reader_unchanged=True,
        last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),
        schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(output=str(folder),arm=arm,train=result['history'][-1]['aggregate']['all_train256']['correct'],duration=result['duration_history'][-1]['train'])
