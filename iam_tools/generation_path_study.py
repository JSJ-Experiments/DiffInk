"""Bounded matched conditioning / terminal denoising / sampling isolation."""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_path import ARMS,NATIVE,training_input,generate,native_train_score
from .generation_study import collate,noise_for,decode_sample,summarize,WRITER,SOURCE,SHA,READER_REL,READER_SHA
from .generation_benchmark import PARENT as DATA_PARENT
from .latent_diffusion import TextLatentDenoiser,transform,cosine_schedule,masked_mse
from .generation_geometry import physical_terms
from .generation_cache import CachedLatentPool
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .ocr_joint_study import local_geometry

PARENT='checkpoints/iam_generation_refinement/20261007-160458/physical_xy_segments/checkpoint-last.pt'
PARENT_SHA='011adfc6050ee22677db31023183d913b9632c53d392674967054a1ac3959db4'
PARENT_STEP=10000


def validate_budget(steps):
    if type(steps)!=int or not 1000<=steps<=6000:raise ValueError('bounded1000–6000 updates per arm required')


def aggregate(rows,splits):
    groups=summarize(rows,splits)
    for split,ids in splits.items():
        for policy,g in groups[split].items():
            q=[r for r in rows if r['sample_id'] in ids and r['policy']==policy]
            def avg(k,sub=None):return float(np.mean([r['geometry'][k][sub] if sub else r['geometry'][k] for r in q]))
            g.update(x_rmse=avg('x_rmse'),y_rmse=avg('y_rmse'),segment_vector_rmse=avg('first_difference','vector_rmse'),second_difference_vector_rmse=avg('second_difference','vector_rmse'),
                     tangent_p90_mean=avg('tangent_angle_error_degrees','p90'),turn_p90_mean=avg('turn_angle_error_degrees','p90'),corner_turn_p90_mean=avg('target_corner_turn_error_degrees','p90'),
                     pen_f1_min=min(r['pen_aligned_reference']['pen_up_f1'] for r in q),nonfinal_false_eoc=sum(r['pen_aligned_reference']['non_final_false_eoc_count'] for r in q))
    return groups


@torch.no_grad()
def evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,alpha,arm,*,probe=False):
    was_training=model.training;model.eval();ids=sorted(records);rows=[];folder=Path(folder);file=folder/f'evaluation-{step}.h5';native=NATIVE[arm]
    modes=['zero','terminal','ddim2','ddim10','ddim50','ddim100'] if probe else ['zero','terminal','ddim50']
    policies=[(mode,'correct') for mode in modes]+[(native,'null'),(native,'swapped')]
    with h5py.File(file,'w') as hf:
        for mode,condition in policies:
            policy=mode+'_'+condition
            for seed in (9142,9143):
                for start in range(0,len(ids),8):
                    group=ids[start:start+8];_,mask,text,_=pool.select(group)
                    if condition=='swapped':
                        other=[ids[(ids.index(i)+1)%len(ids)] for i in group];_,_,text,_=pool.select(other)
                    epsilon=noise_for(group,latents,seed,mask.device)
                    sample=transform(generate(model,epsilon,text,mask,alpha,mode,condition=='null'),stats,True)
                    for j,sid in enumerate(group):
                        z=sample[j,:len(latents[sid])];points,metrics=decode_sample(codec,reader,z,records[sid],vocab);true=targets[sid];n=len(true)
                        row=dict(sample_id=sid,text=records[sid]['text'],writer_id=WRITER,policy=policy,readout=mode,conditioning=condition,seed=seed,
                                 geometry=local_geometry(points[:n,:2],true[:,:2],true[:,2:].argmax(1)),pen_aligned_reference=boundary_metrics(points[:n,2:].argmax(1),true[:,2:].argmax(1)),**metrics)
                        row['conditioning_text']=records[ids[(ids.index(sid)+1)%len(ids)]]['text'] if condition=='swapped' else ('' if condition=='null' else records[sid]['text'])
                        rows.append(row);g=hf.create_group(f'{policy}/{seed}/{sid}');g.create_dataset('points',data=points,compression='gzip');g.create_dataset('latent',data=z.cpu().numpy(),compression='gzip');g.create_dataset('initial_noise',data=epsilon[j,:len(latents[sid])].cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    result=dict(step=step,lines=rows,aggregate=aggregate(rows,splits),packed_h5_sha256=file_sha(file),native_policy=native+'_correct',direct_seed_note='zero input is deterministic; two seed labels are repeated evaluations, NOT independent draws')
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(arm=arm,step=step,native=native,metrics={s:g[native+'_correct'] for s,g in result['aggregate'].items()}),flush=True)
    model.train(was_training);return result


def run(config,repo,root='/data',steps=4000):
    validate_budget(steps)
    if not torch.cuda.is_available():raise ValueError('bounded T4 required')
    torch.set_num_threads(4);root=Path(root);parent=root/PARENT;data_parent=root/DATA_PARENT
    if file_sha(parent)!=PARENT_SHA:raise ValueError('pinned step10000 source required')
    saved=torch.load(parent,map_location='cuda',weights_only=False)
    if saved['step']!=PARENT_STEP:raise ValueError('wrong parent update count')
    cfg=saved['config'];data=json.loads((data_parent/'dataset.json').read_text());records=data['records'];splits=data['splits'];vocab=cfg['vocab']
    if file_sha(data_parent/'source.h5')!=cfg['source_h5_sha256'] or file_sha(data_parent/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('immutable data/whitening drift')
    codec,_,_,cc,_,_=load(config,repo,root,SOURCE,SHA,writer_id=None);codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');codec_digest=tensor_digest(codec.state_dict());reader_digest=tensor_digest(reader.state_dict())
    latents={};targets={}
    with h5py.File(data_parent/'source.h5') as f:
        for sid in records:latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(data_parent/'whitening.pt',weights_only=True);stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()};pool=CachedLatentPool(latents,records,vocab,sorted(records),stats)
    out=root/'checkpoints/iam_generation_path'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    for name in ['generation_path.py','generation_path_study.py','latent_diffusion.py','generation_study.py','generation_geometry.py','generation_cache.py','resource_monitor.py','ocr_joint_adapter.py','ocr_recurrent.py','ocr_context_features.py']:
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['vae.py','blocks.py','losses.py']:
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    schedule=list(training_schedule(splits['train'],PARENT_STEP+steps,8,seed=5142))[PARENT_STEP:];alpha=cosine_schedule(device='cuda');prototype=TextLatentDenoiser(**cfg['model']).cuda();prototype.load_state_dict(saved['model_state_dict']);prototype.train()
    settings=dict(profile='same-architecture training-distribution diagnostic, NOT paper reproduction',parent=PARENT,parent_sha256=PARENT_SHA,parent_step=PARENT_STEP,data_parent=DATA_PARENT,
                  source_rel=SOURCE,source_sha256=SHA,reader_rel=READER_REL,reader_sha256=READER_SHA,source_h5_sha256=file_sha(data_parent/'source.h5'),whitening_sha256=file_sha(data_parent/'whitening.pt'),
                  model=prototype.config,initial_state_sha256=tensor_digest(prototype.state_dict()),resume_rng_sha256=hashlib.sha256(saved['noise_rng_state'].cpu().numpy().tobytes()).hexdigest(),
                  splits=splits,vocab=vocab,writer_id=WRITER,arms=ARMS,native_readouts=NATIVE,max_updates=steps,eval_steps=sorted(set([0,steps//2,steps])),max_train_wall_seconds_per_arm=900,
                  schedule_seed=5142,schedule_offset=PARENT_STEP,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),batch=8,lr=1e-4,betas=[.9,.99],weight_decay=.01,clip=1,dtype='FP32',
                  text_drop_probability=.1,auxiliary_weights=cfg['auxiliary_weights'],objective='all384 whitenedMSE + identical inherited physicalXY/true-stroke target firstdifference weights',
                  training_inputs={'uniform':'cosine forward-noise clean at uniform random t0–999','terminal':'exact pure Gaussian, t999; no alpha999 target contamination','direct':'zeros,t999; no noise or target input'},
                  matching='all arms restore same model+Adam+GPU RNG,consume identical epsilon/t/drop draws,exact minibatches; only training input and t supplied to model differ',
                  selection='TRAIN/native XRMSE+YRMSE+.25segmentRMSE+.1(1-minpenF1); heldout never selects/stops',
                  length='oracle ceil(trueN/8),not independent duration prediction',reader_scope='heldform8generator-unseen but reader-corpus-familiar; one writer32TRAIN,not generalization certification',
                  direct_seed_note='deterministic zero predictions repeated under two seed labels; not independent draws',not_promoted=True)
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n');results={}
    from .resource_monitor import ResourceMonitor
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        # Full no-training source probe distinguishes sampler step count from
        # terminal/readout errors. Only this source probe includes2/10/100DDIM.
        probe_folder=out/'source_probe';probe_folder.mkdir();monitor.set_phase('eval/source_probe');evaluate(prototype,codec,reader,pool,latents,records,vocab,stats,splits,targets,probe_folder,0,alpha,'uniform',probe=True)
        for arm in ARMS:
            monitor.set_phase('prepare/'+arm);model=copy.deepcopy(prototype);optimizer=torch.optim.AdamW(model.parameters());optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            for g in optimizer.param_groups:g['lr']=settings['lr']
            rng=torch.Generator(device='cuda');rng.set_state(saved['noise_rng_state'].cpu());folder=out/arm;folder.mkdir();history=[];best_step=0;stop='budget_completed';train_seconds=0.
            def save(name,step):
                torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=dict(settings,arm=arm),step=PARENT_STEP+step,noise_rng_state=rng.get_state()),folder/name)
            def assessment(step):
                with monitor.in_phase('eval/'+arm):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,alpha,arm)
                    score=native_train_score(ev,arm);history.append(dict(step=step,score=score,aggregate=ev['aggregate']));return score
            best=assessment(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm)
            with (folder/'metrics.jsonl').open('w') as log:
                for step,ids in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(ids);epsilon=torch.randn(clean.shape,device='cuda',generator=rng);draw_t=torch.randint(0,1000,(8,),device='cuda',generator=rng);drop=torch.rand(8,device='cuda',generator=rng)<.1
                    x,t=training_input(arm,clean,epsilon,draw_t,alpha,mask);pred=model(x,t,text,mask,drop);base=masked_mse(pred,clean,mask);terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40')
                    loss=base+settings['auxiliary_weights']['xy']*terms['xy']+settings['auxiliary_weights']['first_difference']*terms['first_difference']
                    optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip'],error_if_nonfinite=True));optimizer.step()
                    row=dict(step=step,sample_ids=ids,loss=float(loss.detach()),base_mse=float(base.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm)
                    if step==1 or step%200==0:
                        row['noise_rng_sha256']=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest();print(dict(arm=arm,**row),flush=True);log.flush()
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite objective')
                    log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;train_seconds+=elapsed;monitor.step(elapsed,8)
                    wall_limit=train_seconds>settings['max_train_wall_seconds_per_arm']
                    if step in settings['eval_steps'] or wall_limit:
                        score=assessment(step);save('checkpoint-last.pt',step)
                        if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if wall_limit:stop='train_wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=codec_digest or tensor_digest(reader.state_dict())!=reader_digest or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('immutable codec/reader changed')
            result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,stop=stop,train_seconds=train_seconds,codec_reader_unchanged=True,
                        final_noise_rng_sha256=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,optimizer;torch.cuda.empty_cache()
    if all(r['last_step']==steps for r in results.values()) and len({r['final_noise_rng_sha256'] for r in results.values()})!=1:raise AssertionError('matched RNG continuation failed')
    if file_sha(root/SOURCE)!=SHA or file_sha(root/READER_REL)!=READER_SHA or file_sha(parent)!=PARENT_SHA:raise AssertionError('immutable checkpoint drift')
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),arms={a:dict(last=r['last_step'],best=r['best_step'],native_train=r['history'][-1]['aggregate']['train'][NATIVE[a]+'_correct'],native_unseen=r['history'][-1]['aggregate']['unseen_prompt'][NATIVE[a]+'_correct']) for a,r in results.items()})
