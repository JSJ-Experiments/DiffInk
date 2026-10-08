"""Matched bounded continuation: TRAIN fit gate failed in supervised48k arm.

No held sources/results opened. Preserve original48k protocol/artifacts; restore
own-arm48000 model/FULLAdam/CPU+CUDA RNG, retain original fixed coefficients.
Both receive12000additional updates at1e-5 on SAME new order. Not fresh weights.
"""
import json,time,tarfile,hashlib,copy
from pathlib import Path
import h5py,numpy as np,torch
from .generation_weak_alignment_study import (ARMS,load_teacher,assess_alignment)
from .generation_prefix_contract import PrefixContractWriter
from .generation_capacity import DATA,SOURCE_H5_SHA,WHITENING_SHA
from .generation_cache import CachedLatentPool
from .generation_coverage_study import evaluate,score,writer_tensor
from .generation_duration_eval import evaluate_duration
from .generation_geometry import physical_terms
from .latent_diffusion import masked_mse
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor
from .weak_alignment import PackedAlignmentPool,capture_last_alignment,alignment_loss
from .generation_weak_confirmation import resolved_seal

PARENT='checkpoints/iam_generation_weak_alignment/20261008-130852'


def checked_path(relative,root):
    p=Path(relative)
    if not relative.startswith('checkpoints/iam_generation_weak_continuation/') or p.is_absolute() or '..' in p.parts:
        raise ValueError('bounded continuation path required')
    return Path(root)/p


def prepare(repo,root='/data'):
    root=Path(root);p=root/PARENT;old=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text())
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS}
    if (p/'confirmation').exists() or any(r['last_step']!=48000 or r['stop']!='budget_completed' for r in results.values()):raise ValueError('complete matched parent with UNOPENED confirmation required')
    if not any(next(h for h in r['history'] if h['step']==r['best_step'])['aggregate']['all_train256']['correct']['free_cer']>.05 for r in results.values()):raise ValueError('extension must be triggered by native TRAIN fit, not held performance')
    resolved_seal(p,old,data,root)
    out=root/'checkpoints/iam_generation_weak_continuation'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for q in sorted(base.rglob('*')):
                if q.is_file() and '__pycache__' not in q.parts and q.suffix in ('.py','.yaml','.json'):tar.add(q,arcname=directory+'/'+str(q.relative_to(base)))
    for directory in ['reserved-confirmation','reserved-confirmation-corrected']:
        for f in (p/directory).rglob('*'):
            if f.is_file():dest=out/directory/f.relative_to(p/directory);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    cfg=copy.deepcopy(old);cfg.update(profile='matched12000update extension ONLY because supervised native TRAIN fit6.44% fails5% gate; standalone mapper NOT InkDiT',fresh=False,parent_step=48000,neural_checkpoint_initialization=True,
        continuation_of=PARENT,parent_config_sha256=file_sha(p/'config.json'),parent_dataset_sha256=file_sha(p/'dataset.json'),parent_results={a:file_sha(p/a/'result.json') for a in ARMS},
        parent_checkpoints={a:results[a]['last_sha256'] for a in ARMS},max_updates=60000,continuation_updates=12000,continuation_schedule_seed=49142,
        eval_steps=[48000,54000,60000],max_train_wall_seconds_per_arm=1500,source_archive_sha256=file_sha(out/'as-run-source.tar.gz'),
        extension_trigger='native TRAIN fit gate not met, no confirmation opened, no objective/model/weights reconfiguration; retain same reserved metadata including documented repair')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'dataset.json').write_bytes((p/'dataset.json').read_bytes())
    return str(out.relative_to(root))

def run(arm, relative, repo, root='/data'):
    if arm not in ARMS or not torch.cuda.is_available():
        raise ValueError('explicit matched positional arm and T4 required')
    root=Path(root);out=checked_path(relative,root);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text())
    if cfg['continuation_updates']!=12000 or cfg['parent_step']!=48000 or cfg['max_updates']!=60000:
        raise ValueError('bounded matched12000update fullAdam continuation required')
    if file_sha(out/'reserved-confirmation/seal.json')!=cfg['reserved_confirmation_sha256']:
        raise ValueError('pre-training fresh seal guard')
    if file_sha(out/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:
        raise ValueError('immutable as-run source required')
    torch.set_num_threads(cfg['torch_cpu_threads'])
    parent=root/cfg['continuation_of']
    labels=load_teacher(parent,cfg,data['training_ids'][arm],data['records'])
    aligned=PackedAlignmentPool(labels,data['training_ids'][arm],device='cuda')
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
    source=parent/arm/'checkpoint-last.pt'
    if file_sha(source)!=cfg['parent_checkpoints'][arm]:raise ValueError('immutable own-arm source checkpoint guard')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved['step']!=48000:raise ValueError('exact own-arm source step48000 required')
    model=PrefixContractWriter(**cfg['models'][arm]).cuda();model.load_state_dict(saved['model_state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
    for g in opt.param_groups:g['lr']=1e-5
    torch.set_rng_state(saved['torch_rng_state']);torch.cuda.set_rng_state(saved['cuda_rng_state'])
    initial=tensor_digest(model.state_dict());folder=out/arm;folder.mkdir(exist_ok=False)
    train=data['training_ids'][arm]
    schedule=list(training_schedule(train,12000,cfg['batch'],seed=cfg['continuation_schedule_seed']))
    history=[];durations=[];alignment_history=[];alignment_weight=saved['alignment_weight'];calibration=saved['calibration'];seconds=0.;clipped=0;best_step=48000;stop='budget_completed'
    with ResourceMonitor(folder,interval=5,sustained_seconds=30) as monitor:
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=arm,model=cfg['models'][arm]),step=step,
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),alignment_weight=alignment_weight,calibration=calibration),folder/name)
        def assess(step):
            with monitor.in_phase('eval/'+arm):
                ev=evaluate(model,codec,reader,pool,latents,records,cfg['vocab'],stats,data['splits'],targets,folder,step,cfg['writers'],controls=step==60000)
                de=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['all_train256'],folder,step,cfg['writers'],cfg['duration_model'],controls=step==60000)
                dev=folder/'exposed-development-duration';dev.mkdir(exist_ok=True)
                dd=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['unseen_prompt'],dev,step,cfg['writers'],cfg['duration_model'])
                alignment=assess_alignment(model,pool,aligned,records,cfg,train)
                alignment_history.append(dict(step=step,**alignment))
                (folder/f'alignment-eval-{step}.json').write_text(json.dumps(alignment_history[-1],indent=2)+'\n')
                s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']))
                durations.append(dict(step=step,train=de['aggregate'],exposed_development=dd['aggregate']))
            return s
        best=assess(48000);save('checkpoint-initial.pt',48000);save('checkpoint-best.pt',48000)
        monitor.set_phase('train/'+arm);model.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for update,batch in enumerate(schedule,1):
                step=48000+update
                start=time.monotonic()
                for g in opt.param_groups:g['lr']=1e-5
                clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],'cuda')
                with capture_last_alignment(model) as captured:
                    pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi)
                auxiliary=alignment_loss(captured['log_probs'],aligned.select(batch),mask,text,validate=update==1)
                terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);w=cfg['auxiliary_weights']
                base_loss=mse+w['xy']*terms['xy']+w['first_difference']*terms['first_difference']
                loss=base_loss+alignment_weight*auxiliary if alignment_weight else base_loss
                opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,sample_ids=batch,alignment_cross_entropy=float(auxiliary.detach()),alignment_weight=alignment_weight,learning_rates=[g['lr'] for g in opt.param_groups],loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'],text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()))
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
        alignment_history=alignment_history,calibration=calibration,alignment_weight=alignment_weight,
        actual_updates=update,clip_fraction=clipped/update,initial_state_sha256=initial,codec_reader_unchanged=True,
        last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),
        schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(output=str(folder),arm=arm,train=result['history'][-1]['aggregate']['all_train256']['correct'],duration=result['duration_history'][-1]['train'])
