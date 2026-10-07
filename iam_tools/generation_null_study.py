"""Does NULL-example supervision interfere with faithful conditional geometry?"""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_path import generate
from .generation_path_study import aggregate,validate_budget
from .generation_study import noise_for,decode_sample,WRITER,SOURCE,SHA,READER_REL,READER_SHA
from .generation_benchmark import PARENT as DATA_PARENT
from .latent_diffusion import TextLatentDenoiser,cosine_schedule,transform,masked_mse
from .generation_geometry import physical_terms
from .generation_cache import CachedLatentPool
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .ocr_joint_study import local_geometry
from .generation_dropout_audit import audit
from .resource_monitor import ResourceMonitor

PARENT='checkpoints/iam_generation_path/20261007-163247/direct/checkpoint-last.pt'
PARENT_SHA='561a96b6e0646fd25500b92a6f71925c82e961ab2e3885da9ea4af5c09f6774e'
ARMS=('drop10','drop0');PARENT_STEP=14000


@torch.no_grad()
def evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,alpha,label):
    was=model.training;model.eval();folder=Path(folder);file=folder/f'evaluation-{step}.h5';ids=sorted(records);rows=[]
    with h5py.File(file,'w') as f:
        for condition in ('correct','null','swapped'):
            policy='zero_'+condition
            for seed in (9142,9143):
                for start in range(0,len(ids),8):
                    group=ids[start:start+8];_,mask,text,_=pool.select(group)
                    if condition=='swapped':_,_,text,_=pool.select([ids[(ids.index(i)+1)%len(ids)] for i in group])
                    epsilon=noise_for(group,latents,seed,mask.device);z=transform(generate(model,epsilon,text,mask,alpha,'zero',condition=='null'),stats,True)
                    for j,sid in enumerate(group):
                        latent=z[j,:len(latents[sid])];points,metrics=decode_sample(codec,reader,latent,records[sid],vocab);true=targets[sid];n=len(true)
                        row=dict(sample_id=sid,text=records[sid]['text'],writer_id=WRITER,policy=policy,seed=seed,geometry=local_geometry(points[:n,:2],true[:,:2],true[:,2:].argmax(1)),pen_aligned_reference=boundary_metrics(points[:n,2:].argmax(1),true[:,2:].argmax(1)),**metrics)
                        rows.append(row);g=f.create_group(f'{policy}/{seed}/{sid}');g.create_dataset('points',data=points,compression='gzip');g.create_dataset('latent',data=latent.cpu().numpy(),compression='gzip');g.create_dataset('initial_noise',data=epsilon[j,:len(latents[sid])].cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    result=dict(step=step,lines=rows,aggregate=aggregate(rows,splits),packed_h5_sha256=file_sha(file),native_policy='zero_correct')
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');model.train(was);print(dict(arm=label,step=step,native={s:g['zero_correct'] for s,g in result['aggregate'].items()}),flush=True);return result


def score(ev):
    r=ev['aggregate']['train']['zero_correct'];return r['x_rmse']+r['y_rmse']+.25*r['segment_vector_rmse']+.1*(1-r['pen_f1_min'])


def run(config,repo,root='/data',steps=4000):
    validate_budget(steps)
    if not torch.cuda.is_available():raise ValueError('bounded T4 required')
    torch.set_num_threads(4);root=Path(root)
    if file_sha(root/PARENT)!=PARENT_SHA:raise ValueError('pinned direct14000 source required')
    # Complete the earlier provisional3999-row replay in a NEWartifact,neveroverwrite.
    replay=audit(root/'checkpoints/iam_generation_path/20261007-163247',root,'dropout-replay-final.json')
    saved=torch.load(root/PARENT,map_location='cuda',weights_only=False);base=saved['config'];data_root=root/DATA_PARENT;data=json.loads((data_root/'dataset.json').read_text());records=data['records'];splits=data['splits'];vocab=base['vocab']
    if saved['step']!=PARENT_STEP or file_sha(data_root/'source.h5')!=base['source_h5_sha256'] or file_sha(data_root/'whitening.pt')!=base['whitening_sha256']:raise ValueError('parent/data identity drift')
    codec,_,_,cc,_,_=load(config,repo,root,SOURCE,SHA,writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict());latents={};targets={}
    with h5py.File(data_root/'source.h5') as f:
        for sid in records:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(data_root/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()};pool=CachedLatentPool(latents,records,vocab,sorted(records),stats);alpha=cosine_schedule(device='cuda')
    out=root/'checkpoints/iam_generation_null'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    names=['generation_null_study.py','generation_path.py','generation_path_study.py','generation_dropout_audit.py','latent_diffusion.py','generation_study.py','generation_geometry.py','generation_cache.py','resource_monitor.py','ocr_joint_adapter.py','ocr_recurrent.py','ocr_context_features.py']
    for name in names:
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['vae.py','blocks.py','losses.py']:
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    (out/'dropout-replay.json').write_text(json.dumps(replay,indent=2)+'\n')
    prototype=TextLatentDenoiser(**base['model']).cuda();prototype.load_state_dict(saved['model_state_dict']);prototype.train();schedule=list(training_schedule(splits['train'],PARENT_STEP+steps,8,seed=5142))[PARENT_STEP:]
    settings=dict(base,profile='matched deterministic geometry NULL-loss interference diagnostic',parent=PARENT,parent_sha256=PARENT_SHA,parent_step=PARENT_STEP,arms=ARMS,native_readouts={a:'zero' for a in ARMS},source_readout='zero',max_updates=steps,eval_steps=sorted(set([0,steps//2,steps])),schedule_offset=PARENT_STEP,
                  schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),initial_state_sha256=tensor_digest(prototype.state_dict()),resume_rng_sha256=hashlib.sha256(saved['noise_rng_state'].cpu().numpy().tobytes()).hexdigest(),
                  text_drop_by_arm={'drop10':.1,'drop0':0.},intervention='only dropmask applied to text changes; identical randomdrop draws consumed by both; zeros/t999/batch8/LR1e-4/Adam/weights/data fixed',
                  null_evaluation_note='drop0NULLbranch untrained/OOD; report as causal ablation,not usefulunconditionalgeneration',selection='TRAINzero XRMSE+YRMSE+.25segmentRMSE+.1(1-minpenF1)',not_promoted=True)
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n');results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        probe_folder=out/'source_probe';probe_folder.mkdir();monitor.set_phase('eval/source');evaluate(prototype,codec,reader,pool,latents,records,vocab,stats,splits,targets,probe_folder,0,alpha,'source')
        for arm in ARMS:
            model=copy.deepcopy(prototype);optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            for g in optimizer.param_groups:g['lr']=1e-4
            rng=torch.Generator(device='cuda');rng.set_state(saved['noise_rng_state'].cpu());folder=out/arm;folder.mkdir();history=[];best_step=0;train_seconds=0.;stop='budget_completed'
            def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=dict(settings,arm=arm),step=PARENT_STEP+step,noise_rng_state=rng.get_state()),folder/name)
            def assessment(step):
                with monitor.in_phase('eval/'+arm):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,alpha,arm);s=score(ev);history.append(dict(step=step,score=s,aggregate=ev['aggregate']));return s
            best=assessment(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm)
            with (folder/'metrics.jsonl').open('w') as log:
                for step,ids in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(ids);epsilon=torch.randn(clean.shape,device='cuda',generator=rng);draw_t=torch.randint(0,1000,(8,),device='cuda',generator=rng);draw_drop=torch.rand(8,device='cuda',generator=rng)<.1;drop=draw_drop if arm=='drop10' else torch.zeros_like(draw_drop)
                    pred=model(torch.zeros_like(epsilon),torch.ones(8,device='cuda'),text,mask,drop);base_loss=masked_mse(pred,clean,mask);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');loss=base_loss+settings['auxiliary_weights']['xy']*terms['xy']+settings['auxiliary_weights']['first_difference']*terms['first_difference']
                    optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                    row=dict(step=step,sample_ids=ids,loss=float(loss.detach()),base_mse=float(base_loss.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,draw_null_count=int(draw_drop.sum()),applied_null_count=int(drop.sum()))
                    if step==1 or step%200==0:row['noise_rng_sha256']=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest();print(dict(arm=arm,**row),flush=True);log.flush()
                    log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;train_seconds+=elapsed;monitor.step(elapsed,8)
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite NULLablation')
                    limit=train_seconds>settings['max_train_wall_seconds_per_arm']
                    if step in settings['eval_steps'] or limit:
                        s=assessment(step);save('checkpoint-last.pt',step)
                        if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                    if limit:stop='train_wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozen codec/reader changed')
            result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,stop=stop,train_seconds=train_seconds,codec_reader_unchanged=True,final_noise_rng_sha256=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,optimizer;torch.cuda.empty_cache()
    if len({r['final_noise_rng_sha256'] for r in results.values()})!=1:raise AssertionError('matchedRNGguard')
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),arms={a:dict(last=r['last_step'],native_train=r['history'][-1]['aggregate']['train']['zero_correct'],native_unseen=r['history'][-1]['aggregate']['unseen_prompt']['zero_correct']) for a,r in results.items()})
