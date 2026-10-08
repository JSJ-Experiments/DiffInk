"""Matched TRAIN-triggered convergence continuation, before sealed confirmation."""
import copy,hashlib,json,tarfile,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_alignment_study import ARMS
from .generation_alignment import AlignedWriterDenoiser
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


def run(relative,repo,root='/data',steps=24000):
    if type(steps)!=int or steps!=24000 or not torch.cuda.is_available():raise ValueError('one bounded matched24000-update T4 continuation required')
    if not relative.startswith('checkpoints/iam_generation_alignment/') or '..' in Path(relative).parts:raise ValueError('bounded completed fresh study path')
    torch.set_num_threads(4);root=Path(root);parent=root/relative;cfg=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());old=json.loads((parent/'result.json').read_text())
    guards,_=verify_sources(parent,repo,root,cfg,data,old)
    if (parent/'confirmation').exists():raise ValueError('do not retrain these arms after opening confirmation')
    if not any(r['history'][-1]['aggregate']['all_train256']['correct']['free_cer']>.1 for r in old.values()):raise ValueError('predeclared TRAIN underfit trigger not met')
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    latents={};targets={};records=data['records'];vocab=cfg['vocab'];writers=cfg['writers'];source=root/DATA
    with h5py.File(source/'source.h5') as f:
        for sid in records:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(source/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()};pool=CachedLatentPool(latents,records,vocab,sorted(records),stats,max_lines=264)
    out=root/'checkpoints/iam_generation_alignment_continuation'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(cfg);cfg.update(fresh=False,parent_step=old['global']['last_step'],continuation_of=relative,parent_result_sha256=file_sha(parent/'result.json'),parent_checkpoints={a:r['last_sha256'] for a,r in old.items()},max_updates=steps,eval_steps=[0,8000,16000,24000],schedule_seed=39142,lr=1e-5,lr_schedule='constant1e-5, inherited full Adam moments; matched TRAIN-only-triggered continuation',initialization='each arm restores its own fresh-study last model/Adam/RNG; NO crossover or polished prototype weights',trigger='predeclared before soft arm finished: either matched-final TRAIN CER>10% => BOTH arms24000 more updates at1e-5 before unsealing; no held/confirmation tuning',original_guards=guards)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for p in sorted(base.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'):tar.add(p,arcname=directory+'/'+str(p.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz');(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'dataset.json').write_bytes((parent/'dataset.json').read_bytes());results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        for a in ARMS:
            saved=torch.load(parent/a/'checkpoint-last.pt',map_location='cuda',weights_only=False);model=AlignedWriterDenoiser(**cfg['models'][a]).cuda();model.load_state_dict(saved['model_state_dict']);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
            for g in opt.param_groups:g['lr']=cfg['lr']
            torch.set_rng_state(saved['torch_rng_state'].cpu());torch.cuda.set_rng_state(saved['cuda_rng_state'].cpu());folder=out/a;folder.mkdir();initial=tensor_digest(model.state_dict());train=data['training_ids'][a];schedule=list(training_schedule(train,steps,cfg['batch'],seed=cfg['schedule_seed']));history=[];dh=[];seconds=0.;clipped=0;best_step=0;stop='budget_completed'
            def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=a,model=cfg['models'][a]),step=step,absolute_step=cfg['parent_step']+step,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),folder/name)
            def assess(step):
                with monitor.in_phase('eval/'+a):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,data['splits'],targets,folder,step,writers,controls=step==steps);de=evaluate_duration(model,codec,reader,records,vocab,stats,data['splits']['unseen_prompt'],folder,step,writers,cfg['duration_model'],controls=step==steps);s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']));dh.append(dict(step=step,aggregate=de['aggregate']));return s
            best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+a);model.train()
            with (folder/'metrics.jsonl').open('w') as log:
                for step,batch in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,'cuda');pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);w=cfg['auxiliary_weights'];loss=mse+w['xy']*terms['xy']+w['first_difference']*terms['first_difference'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                    row=dict(step=step,absolute_step=cfg['parent_step']+step,sample_ids=batch,learning_rates=[g['lr'] for g in opt.param_groups],loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'],text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()));log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped']
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
