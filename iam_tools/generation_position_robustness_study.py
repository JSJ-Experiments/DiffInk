"""Controlled nuisance-jitter continuation of the winning TRAIN-fitting arm.
Targets, mask, strokes and point spacing remain unchanged; only relativePE
denominator is jittered on half TRAIN rows. NOT smoothing or time resampling.
"""
import copy,hashlib,json,tarfile,time
from pathlib import Path
import h5py,numpy as np,torch
ARMS=('control','pe_jitter')
from .generation_timing import TimingProbe
from .generation_position_robustness import jitter_lengths
from .generation_timing_study import PARENT,PARENT_SHA
from .generation_alignment_review import verify_sources
from .generation_capacity import DATA
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


def run(repo,root='/data',steps=8000):
    relative=PARENT
    if type(steps)!=int or steps!=8000 or not torch.cuda.is_available():raise ValueError('one bounded matched8000-update T4 continuation required')
    if not relative.startswith('checkpoints/iam_generation_alignment_continuation/') or '..' in Path(relative).parts:raise ValueError('bounded completed fresh study path')
    torch.set_num_threads(4);root=Path(root);parent=root/relative;cfg=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());old=json.loads((parent/'result.json').read_text())
    guards,_=verify_sources(parent,repo,root,cfg,data,old)
    # Previously opened confirmation now stays excluded; do not present it as fresh.
    if file_sha(parent/'soft_gaussian/checkpoint-last.pt')!=PARENT_SHA['soft_gaussian']:raise ValueError('pinned winning arm required')
    cfg['models']={a:copy.deepcopy(cfg['models']['soft_gaussian']) for a in ARMS}
    data['training_ids']={a:data['training_ids']['soft_gaussian'] for a in ARMS}
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    latents={};targets={};records=data['records'];vocab=cfg['vocab'];writers=cfg['writers'];source=root/DATA
    with h5py.File(source/'source.h5') as f:
        for sid in records:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(source/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()};pool=CachedLatentPool(latents,records,vocab,sorted(records),stats,max_lines=264)
    out=root/'checkpoints/iam_generation_position_robustness'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(cfg);cfg.update(fresh=False,parent_step=48000,continuation_of=relative,parent_result_sha256=file_sha(parent/'result.json'),parent_checkpoints={a:PARENT_SHA['soft_gaussian'] for a in ARMS},max_updates=steps,eval_steps=[0,2000,4000,8000],schedule_seed=49142,lr=1e-5,lr_schedule='constant1e-5, inherited full Adam moments; matched TRAIN-only-triggered continuation',initialization='identical winning soft_gaussian48000weights/full Adam/RNG, no new model parameters',trigger='frozen256TRAIN intervention identified PE-only ±1block CER56–58% versus native2.4%; never optimize from confirmation',jitter_probability=.5,jitter_fraction=.2,jitter_seed=50142,scope='ONLY relativePE denominator changed; input tensor size/self-attention mask/target packed sequence/XY/pen/point spacing unchanged; inference native mask denominator, NO oracle PE override',selection='all256TRAIN native geometry only; no development/confirmation selection',eval_train_duration=True,original_guards=guards)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for p in sorted(base.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'):tar.add(p,arcname=directory+'/'+str(p.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz');(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'dataset.json').write_text(json.dumps(data,indent=2)+'\n');results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        for a in ARMS:
            saved=torch.load(parent/'soft_gaussian/checkpoint-last.pt',map_location='cuda',weights_only=False);model=TimingProbe(**cfg['models'][a]).cuda();model.load_state_dict(saved['model_state_dict']);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
            for g in opt.param_groups:g['lr']=cfg['lr']
            torch.set_rng_state(saved['torch_rng_state'].cpu());torch.cuda.set_rng_state(saved['cuda_rng_state'].cpu());folder=out/a;folder.mkdir();jrng=torch.Generator().manual_seed(cfg['jitter_seed']);initial=tensor_digest(model.state_dict());train=data['training_ids'][a];schedule=list(training_schedule(train,steps,cfg['batch'],seed=cfg['schedule_seed']));history=[];dh=[];seconds=0.;clipped=0;best_step=0;stop='budget_completed'
            def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=a,model=cfg['models'][a]),step=step,absolute_step=cfg['parent_step']+step,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),jitter_rng_state=jrng.get_state()),folder/name)
            def assess(step):
                with monitor.in_phase('eval/'+a):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,data['splits'],targets,folder,step,writers,controls=step==steps);de=evaluate_duration(model,codec,reader,records,vocab,stats,data['splits']['all_train256'],folder,step,writers,cfg['duration_model'],controls=False);s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']));dh.append(dict(step=step,aggregate=de['aggregate']));return s
            best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+a);model.train()
            with (folder/'metrics.jsonl').open('w') as log:
                for step,batch in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,'cuda');pe_lengths=jitter_lengths(mask.sum(1),jrng,probability=cfg['jitter_probability'],fraction=cfg['jitter_fraction']);position=pe_lengths if a=='pe_jitter' else mask.sum(1);pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi,position_lengths=position);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);w=cfg['auxiliary_weights'];loss=mse+w['xy']*terms['xy']+w['first_difference']*terms['first_difference'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                    row=dict(position_lengths=position.tolist(),native_lengths=mask.sum(1).tolist(),step=step,absolute_step=cfg['parent_step']+step,sample_ids=batch,learning_rates=[g['lr'] for g in opt.param_groups],loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'],text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()));log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped']
                    if step%1000==0:print(dict(arm=a,**row,train_seconds=seconds),flush=True);log.flush()
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite continuation')
                    limit=seconds>1800
                    if step in cfg['eval_steps'] or limit:
                        s=assess(step);save('checkpoint-last.pt',step)
                        if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                    if limit:stop='wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise ValueError('frozen evaluator drift')
            r=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,duration_history=dh,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,codec_reader_unchanged=True,last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),not_promoted=True);results[a]=r;(folder/'result.json').write_text(json.dumps(r,indent=2)+'\n');del model,opt,saved;torch.cuda.empty_cache();monitor.set_phase('cleanup')
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),train={a:r['history'][-1]['aggregate']['all_train256']['correct']['free_cer'] for a,r in results.items()})
