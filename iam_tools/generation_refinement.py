"""Matched frozen-codec continuation: whitened MSE / physicalXY / XY+segments."""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_study import collate,evaluate,fixed_denoising,SOURCE,SHA,READER_SHA,READER_REL
from .generation_benchmark import PARENT
from .generation_geometry import physical_terms,gradient_norm,coefficients
from .latent_diffusion import TextLatentDenoiser,cosine_schedule,forward_noise,masked_mse
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor
from .generation_cache import CachedLatentPool

PARENT_SHA='ddf0720e29c8692eea2bd31a2f0f28444dfa230fee271ceaab0a48080f2bfceb'
ARMS=('uniform','physical_xy','physical_xy_segments')


def run(config,repo,root='/data',steps=2000,threads=4):
    if type(steps)!=int or not 1000<=steps<=4000 or threads not in (1,4) or not torch.cuda.is_available():raise ValueError('bounded T4,1000–4000updates,1/4threads')
    torch.set_num_threads(threads);root=Path(root);parent=root/PARENT
    if file_sha(parent/'text/checkpoint-last.pt')!=PARENT_SHA:raise ValueError('pinned step8000 parent required')
    settings=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());splits=data['splits'];records=data['records'];vocab=settings['vocab']
    codec,_,_,cfg,_,_=load(config,repo,root,SOURCE,SHA,writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cfg,'cuda')
    codec_digest=tensor_digest(codec.state_dict());reader_digest=tensor_digest(reader.state_dict());latents={};targets={}
    with h5py.File(parent/'source.h5') as f:
        for sid in records:
            latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
            direct=latents[sid][...,:40].reshape(-1,5)[:records[sid]['points'],:2].cpu().numpy()
            if np.abs(direct-targets[sid][:,:2]).max()>.0005:raise ValueError('polyphase40 direct geometry contract drift')
    stats=torch.load(parent/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    saved=torch.load(parent/'text/checkpoint-last.pt',map_location='cuda',weights_only=False)
    out=root/'checkpoints/iam_generation_refinement'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    for name in ['generation_refinement.py','generation_geometry.py','generation_study.py','latent_diffusion.py','resource_monitor.py','generation_cache.py']:
        p=out/'source-code'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    schedule=list(training_schedule(splits['train'],8000+steps,8,seed=5142))[8000:];alpha=cosine_schedule(device='cuda')
    pool=CachedLatentPool(latents,records,vocab,sorted(records),stats)
    prepare=pool.select
    prototype=TextLatentDenoiser(**settings['model']).cuda();prototype.load_state_dict(saved['model_state_dict']);prototype.train()
    params=list(prototype.parameters());norms={k:[] for k in ('base','xy','first_difference')}
    calibration_rng=torch.Generator(device='cuda').manual_seed(11242)
    for group in schedule[:4]:
        clean,mask,text,pm=prepare(group);eps=torch.randn(clean.shape,device='cuda',generator=calibration_rng);t=torch.randint(0,1000,(8,),device='cuda',generator=calibration_rng)
        pred=prototype(forward_noise(clean,eps,t,alpha,mask),t.float()/999,text,mask,torch.zeros(8,device='cuda',dtype=torch.bool))
        terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');base=masked_mse(pred,clean,mask)
        for key,loss in dict(base=base,**terms).items():norms[key].append(gradient_norm(loss,params))
    averaged={k:float(np.mean(v)) for k,v in norms.items()};weights=coefficients(averaged['base'],averaged['xy'],averaged['first_difference'])
    settings=dict(settings,parent=PARENT,parent_sha256=PARENT_SHA,arms=ARMS,max_updates=steps,max_wall_seconds_per_arm=600,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),initial_state_sha256=tensor_digest(prototype.state_dict()),resume_rng_sha256=hashlib.sha256(saved['noise_rng_state'].cpu().numpy().tobytes()).hexdigest(),parent_step=8000,lr=1e-4,threads=threads,auxiliary_weights=weights,calibration_gradient_norms=norms,calibration_mean_norms=averaged,
                  calibration='TRAIN first4 continuation minibatches,seed11242,XY25% and firstdifference10% of base full-model gradient norm; fixed thereafter',
                  target_derivative='within true stroke index difference, NOT physical velocity/curvature; no generic smoothing',selection='TRAIN t999XY+pen only',resource_monitor='phase-aware2s sampling/30s alerts; request4CPU/16GiB,T4',noise_provenance='inherited GPU RNG state and matched deterministic draws; RNGSHA at0/every200/final; no per-updateGPU-toCPUepsilon copy',eval_steps=[0,steps],not_promoted=True)
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n');results={}
    with ResourceMonitor(out,interval=2,sustained_seconds=30) as monitor:
        for arm in ARMS:
            monitor.set_phase('prepare/'+arm);model=copy.deepcopy(prototype);optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            for g in optimizer.param_groups:g['lr']=1e-4
            rng=torch.Generator(device='cuda');rng.set_state(saved['noise_rng_state'].cpu());folder=out/arm;folder.mkdir();history=[];best=None;best_step=0
            def assessment(step):
                with monitor.in_phase('eval/'+arm):
                    score=fixed_denoising(model,latents,records,vocab,stats,splits['train'],alpha,'text')
                    result=evaluate(model,codec,reader,latents,records,vocab,stats,splits,targets,folder,step,alpha)
                    history.append(dict(step=step,denoising=score,groups=result['groups']));return score['999']['xy']+score['999']['pen']
            def save(name,step):
                torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=dict(settings,arm=arm),step=8000+step,noise_rng_state=rng.get_state()),folder/name)
            best=assessment(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm);start=time.monotonic();stop='budget_completed'
            with (folder/'metrics.jsonl').open('w') as log:
                for step,group in enumerate(schedule,1):
                    began=time.monotonic();clean,mask,text,pm=prepare(group)
                    eps=torch.randn(clean.shape,device='cuda',generator=rng);t=torch.randint(0,1000,(8,),device='cuda',generator=rng);drop=torch.rand(8,device='cuda',generator=rng)<.1
                    pred=model(forward_noise(clean,eps,t,alpha,mask),t.float()/999,text,mask,drop);base=masked_mse(pred,clean,mask);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40')
                    loss=base
                    if arm!='uniform':loss=loss+weights['xy']*terms['xy']
                    if arm=='physical_xy_segments':loss=loss+weights['first_difference']*terms['first_difference']
                    optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                    values=dict(step=step,sample_ids=group,loss=float(loss.detach()),base_mse=float(base.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm)
                    if step%200==0 or step==1:
                        values['noise_rng_sha256']=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest();print(dict(arm=arm,step=step,**{k:values[k] for k in ('base_mse','physical_xy_mse','segment_mse')}),flush=True);log.flush()
                    log.write(json.dumps(values)+'\n');monitor.step(time.monotonic()-began,8)
                    if not np.isfinite(values['loss']):raise FloatingPointError('nonfinite continuation')
                    if time.monotonic()-start>settings['max_wall_seconds_per_arm']:stop='wall_limit';break
            score=assessment(step);save('checkpoint-last.pt',step)
            if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
            if tensor_digest(codec.state_dict())!=codec_digest or tensor_digest(reader.state_dict())!=reader_digest or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozen source drift')
            result=dict(last_step=step,best_step=best_step,history=history,stop=stop,codec_reader_unchanged=True,final_noise_rng_sha256=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),last_sha256=file_sha(folder/'checkpoint-last.pt'),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,optimizer;torch.cuda.empty_cache()
    if len({r['final_noise_rng_sha256'] for r in results.values()})!=1:raise AssertionError('matched RNG guard failed')
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),weights=weights,groups={k:r['history'][-1]['groups'] for k,r in results.items()})
