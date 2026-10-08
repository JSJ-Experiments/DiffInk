"""Matched broad1024 continuation: unrestricted attention vs small local text hint."""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_text_anchor import AnchoredWriterDenoiser,local_text_hint
from .generation_coverage_study import evaluate,score,writer_tensor
from .generation_cache import CachedLatentPool
from .generation_geometry import physical_terms
from .latent_diffusion import masked_mse,positions
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor
PARENT='checkpoints/iam_generation_coverage/20261008-045818/broad1024/checkpoint-last.pt'
PARENT_SHA='f527a73d15576fba26a3c1af5fe787382a12b25d761f25d4834327019c3714b1'


def validate_budget(steps):
    if type(steps)!=int or not 1000<=steps<=6000:raise ValueError('bounded1000–6000updates required')


def run(config,repo,root='/data',steps=4000):
    validate_budget(steps)
    if not torch.cuda.is_available():raise ValueError('T4 required')
    torch.set_num_threads(4);root=Path(root);parent=root/Path(PARENT).parent.parent
    if file_sha(root/PARENT)!=PARENT_SHA:raise ValueError('pinned broad8000parent required')
    cfg=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());saved=torch.load(root/PARENT,map_location='cuda',weights_only=False)
    if saved['step']!=26000 or file_sha(parent/'source.h5')!=cfg['source_h5_sha256'] or file_sha(parent/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('parent/data integrity failure')
    codec,_,_,cc,_,_=load(config,repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    records=data['records'];splits=data['splits'];train=data['training_ids']['broad1024'];writers=data['writers'];vocab=cfg['vocab'];latents={};targets={}
    with h5py.File(parent/'source.h5') as f:
        for sid in records:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(parent/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()};pool=CachedLatentPool(latents,records,vocab,sorted(records),stats,max_lines=1100)
    out=root/'checkpoints/iam_generation_text_anchor'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    for name in ['generation_text_anchor.py','generation_text_anchor_study.py','generation_coverage.py','generation_coverage_study.py','latent_diffusion.py','generation_study.py','generation_path_study.py','generation_geometry.py','generation_cache.py','resource_monitor.py','ocr_joint_adapter.py','ocr_recurrent.py','ocr_context_features.py']:
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['vae.py','blocks.py','losses.py']:
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    schedule=list(training_schedule(train,8000+steps,8,seed=9142))[8000:]
    scales={'control':0.,'local_hint':.1};settings=dict(cfg,profile='matched localtextpositionhint diagnostic,NOTactualcharacteralignment or paperreproduction',parent=PARENT,parent_sha256=PARENT_SHA,data_parent=str(parent.relative_to(root)),parent_step=26000,arms=list(scales),local_text_scales=scales,max_updates=steps,eval_steps=[0,steps//2,steps],schedule_offset=8000,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),intervention='only localcharacterembedding inputhint;linear normalizedduration interpolation; allcharsused,noglyphboundaries/targetteacher;globalattentionunrestricted',selection='same actualbroadTRAIN65probe,never8held',same_source_model_optimizer=True,not_promoted=True)
    # Descriptive hint magnitude BEFORE any optimizer updates, original32 TRAIN only.
    probe=AnchoredWriterDenoiser(local_text_scale=.1,**cfg['model']).cuda().eval();probe.load_state_dict(saved['model_state_dict']);clean,mask,text,_=pool.select(splits['retained_train'][:8]);wi=writer_tensor(splits['retained_train'][:8],records,writers,'cuda')
    with torch.no_grad():
        width=probe.config['width'];p=torch.arange(clean.shape[1],device='cuda',dtype=clean.dtype);relative=p[None]/(mask.sum(1)-1).clamp_min(1)[:,None]
        hidden=probe.project(torch.zeros_like(clean))+positions(p,width)[None]+positions(relative*100,width)+probe.time(positions(torch.ones(8,device='cuda')*1000,width))[:,None]+probe.writer(wi)[:,None]
        hint=.1*local_text_hint(probe.text,text,mask);settings['initial_local_hint_rms_fraction']=float(hint[mask].square().mean().sqrt()/hidden[mask].square().mean().sqrt())
    del probe;(out/'config.json').write_text(json.dumps(settings,indent=2)+'\n');results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        for arm,scale in scales.items():
            model=AnchoredWriterDenoiser(local_text_scale=scale,**cfg['model']).cuda();model.load_state_dict(saved['model_state_dict']);base=[p for n,p in model.named_parameters() if n!='writer.weight'];opt=torch.optim.AdamW([dict(params=base),dict(params=[model.writer.weight])]);opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']));folder=out/arm;folder.mkdir();best_step=0;history=[];seconds=0.;stop='budget_completed'
            def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(settings,arm=arm,model=model.config),step=26000+step),folder/name)
            def assess(step):
                with monitor.in_phase('eval/'+arm):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,writers,controls=step==steps);s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']));return s
            best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm)
            with (folder/'metrics.jsonl').open('w') as log:
                for step,batch in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,'cuda');pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');base_loss=masked_mse(pred,clean,mask);loss=base_loss+cfg['auxiliary_weights']['xy']*terms['xy']+cfg['auxiliary_weights']['first_difference']*terms['first_difference']
                    opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();row=dict(step=step,sample_ids=batch,loss=float(loss.detach()),base_mse=float(base_loss.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm);log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,8)
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinitehintobjective')
                    if step%500==0:print(dict(arm=arm,**row,train_seconds=seconds),flush=True);log.flush()
                    limit=seconds>900
                    if step in settings['eval_steps'] or limit:
                        s=assess(step);save('checkpoint-last.pt',step)
                        if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                    if limit:stop='wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozenstate drift')
            result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,stop=stop,train_seconds=seconds,codec_reader_unchanged=True,last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,opt;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),arms={a:dict(last=r['last_step'],groups={s:d['correct']['free_cer'] for s,d in r['history'][-1]['aggregate'].items()}) for a,r in results.items()})
