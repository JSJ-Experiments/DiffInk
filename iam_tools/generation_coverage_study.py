"""Bounded nested32/41/1024 text coverage, faithful frozen codec, explicit style IDs."""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .generation_coverage import WriterTextDenoiser,select_coverage
from .generation_study import decode_sample,read_sequence
from .generation_path_study import aggregate
from .generation_geometry import physical_terms
from .generation_cache import CachedLatentPool
from .latent_diffusion import transform,masked_mse
from .ocr_pool_study import load_pool,single_batch
from .ocr_convergence import POOL_SHA
from .ocr_joint_adapter import load_reader
from .ocr_joint_study import local_geometry
from .ocr_context_study import tensor_digest
from .writer_expansion import load,training_schedule
from .latent_integration import encoded
from .kl_tradeoff import SOURCE,SHA
from .pen_ab import file_sha,boundary_metrics
from .inkvae import edit_distance
from .resource_monitor import ResourceMonitor

PARENT='checkpoints/iam_generation_null/20261007-164542/drop0/checkpoint-last.pt'
PARENT_SHA='a63c20b70f7167680d391a6d07650879eb965a072b5d3f11e76b99aa12205439'
DATA='checkpoints/iam_generation_gate/20261007-153404'


def validate_budget(steps):
    if type(steps)!=int or not 1000<=steps<=12000:raise ValueError('bounded1000–12000 updates required')


def writer_tensor(ids,records,writers,device):
    return torch.tensor([writers.index(records[i]['writer_id']) for i in ids],device=device,dtype=torch.long)


@torch.no_grad()
def evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,writers,controls=False):
    was=model.training;model.eval();ids=sorted(set(i for v in splits.values() for i in v));rows=[];file=Path(folder)/f'evaluation-{step}.h5'
    with h5py.File(file,'w') as f:
        for policy in (['correct','swapped_text','swapped_writer'] if controls else ['correct']):
            for start in range(0,len(ids),8):
                batch=ids[start:start+8];clean,mask,text,_=pool.select(batch);wi=writer_tensor(batch,records,writers,clean.device)
                if policy=='swapped_text':_,_,text,_=pool.select([ids[(ids.index(i)+1)%len(ids)] for i in batch])
                if policy=='swapped_writer':wi=(wi+1)%len(writers)
                pred=model(torch.zeros_like(clean),torch.ones(len(batch),device=clean.device),text,mask,writer_ids=wi);z=transform(pred,stats,True)
                for j,sid in enumerate(batch):
                    points,metrics=decode_sample(codec,reader,z[j,:len(latents[sid])],records[sid],vocab);true=targets[sid];n=len(true)
                    row=dict(sample_id=sid,text=records[sid]['text'],writer_id=records[sid]['writer_id'],policy=policy,seed=9142,geometry=local_geometry(points[:n,:2],true[:,:2],true[:,2:].argmax(1)),pen_aligned_reference=boundary_metrics(points[:n,2:].argmax(1),true[:,2:].argmax(1)),**metrics)
                    rows.append(row);g=f.create_group(policy+'/'+sid);g.create_dataset('points',data=points,compression='gzip');g.create_dataset('latent',data=z[j,:len(latents[sid])].cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    result=dict(step=step,lines=rows,aggregate=aggregate(rows,splits),packed_h5_sha256=file_sha(file))
    (Path(folder)/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');model.train(was)
    print(dict(step=step,folder=str(folder),groups={s:r['correct']['free_cer'] for s,r in result['aggregate'].items()}),flush=True);return result


def score(ev,train):
    # Fixed64TRAIN probe, restricted to each arm's actual training IDs. No held/adaptation selection.
    rows=[r for r in ev['lines'] if r['sample_id'] in train and r['policy']=='correct']
    if not rows:raise ValueError('nonempty TRAIN-only selection required')
    return float(np.mean([r['geometry']['x_rmse']+r['geometry']['y_rmse']+.25*r['geometry']['first_difference']['vector_rmse']+.1*(1-r['pen_aligned_reference']['pen_up_f1']) for r in rows]))


def run(config,repo,root='/data',steps=8000):
    validate_budget(steps)
    if not torch.cuda.is_available():raise ValueError('T4 required')
    torch.set_num_threads(4);root=Path(root);data_root=root/DATA;old=json.loads((data_root/'dataset.json').read_text());pool_path,m,vocab=load_pool(root,POOL_SHA)
    saved=torch.load(root/PARENT,map_location='cuda',weights_only=False);parent_sha=file_sha(root/PARENT)
    # SHA tied to immutable completed parent result, never a mutable best alias.
    parent_result=json.loads((root/Path(PARENT).parent/'result.json').read_text())
    if parent_sha!=PARENT_SHA or parent_sha!=parent_result['last_sha256'] or saved['step']!=18000 or saved['config']['vocab']!=vocab:raise ValueError('pinned drop0step18000/vocab required')
    scope=select_coverage(m,old['splits']);records={i:dict(m['records'][i]) for i in set(scope['arms']['broad1024'])|set(old['splits']['unseen_prompt'])};splits=scope['evaluation'];ids=sorted(records);writers=scope['writers']
    codec,_,_,cc,original_vocab,_=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if original_vocab!=vocab:raise ValueError('fixedcodecalphabet drift')
    codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda');cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    out=root/'checkpoints/iam_generation_coverage'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    snapshots=['generation_coverage.py','generation_coverage_study.py','latent_diffusion.py','generation_study.py','generation_path_study.py','generation_geometry.py','generation_cache.py','resource_monitor.py','ocr_joint_adapter.py','ocr_recurrent.py','ocr_context_features.py']
    for name in snapshots:
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['vae.py','blocks.py','losses.py']:
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    latents={};targets={};preflight=[]
    with h5py.File(pool_path/'lines.h5') as f,h5py.File(data_root/'source.h5') as old_f,h5py.File(out/'source.h5','w') as dest,torch.no_grad():
        from model.losses import mixture_expectation
        for sid in ids:
            rawpoints=f[sid]['point_seq'][:];r=records[sid]
            if hashlib.sha256(rawpoints.tobytes()).hexdigest()!=r['points_sha256']:raise ValueError('immutablepoints drift')
            raw,pm,_=single_batch(rawpoints,r['text'],vocab,'cuda');target,mu,_,_=encoded(codec,raw,pm);n=len(rawpoints);latent=mu[0].T.detach();truth=torch.cat((target[0,:n],raw[0,2:,:n].T),1).cpu().numpy()
            decoded=codec.decode(mu,padding_mask=~pm);xy=mixture_expectation(decoded)[0,:n];pen=decoded[0,:3,:n].argmax(0);err=float((xy-target[0,:n]).square().mean().sqrt())
            if err>.0005 or not torch.equal(pen,raw[0,2:,:n].argmax(0)):raise ValueError('frozen codec fidelity failure')
            if sid in old_f and (not np.array_equal(truth,old_f[sid]['target'][:]) or not np.array_equal(latent.cpu().numpy(),old_f[sid]['latent_mean'][:])):raise ValueError('original data identity changed')
            reader_errors=None
            if sid in {i for v in splits.values() for i in v}:
                text=read_sequence(reader,torch.cat((xy,torch.nn.functional.one_hot(pen,3).float()),1),vocab);reader_errors=edit_distance(r['text'],text)
            preflight.append(dict(sample_id=sid,codec_rmse=err,reader_errors=reader_errors));latents[sid]=latent;targets[sid]=truth
            g=dest.create_group(sid);g.create_dataset('target',data=truth,compression='gzip');g.create_dataset('latent_mean',data=latent.cpu().numpy(),compression='gzip')
    stats=torch.load(data_root/'whitening.pt',weights_only=True);torch.save(stats,out/'whitening.pt');stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    # Retain original TRAIN32 whitening for all arms to avoid a target/scaling confound.
    pool=CachedLatentPool(latents,records,vocab,ids,stats,max_lines=1100);eval_ids=sorted(set(i for v in splits.values() for i in v))
    dataset=dict(records=records,splits=splits,training_ids=scope['arms'],writers=writers,scope=scope,preflight=preflight)
    (out/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
    torch.manual_seed(8142);prototype=WriterTextDenoiser(len(writers),**saved['config']['model']).cuda();keys=prototype.load_state_dict(saved['model_state_dict'],strict=False)
    if keys.missing_keys!=['writer.weight'] or keys.unexpected_keys:raise ValueError('onlynewzero writer embedding allowed')
    cfg=dict(profile='nested textcoverage + explicitwriter diagnostic,NOTsemanticVAE/releasedInkDiT',parent=PARENT,parent_sha256=parent_sha,parent_step=18000,model=prototype.config,scope=scope,vocab=vocab,writers=writers,pool_sha256=POOL_SHA,source_rel=SOURCE,source_sha256=SHA,reader_rel=saved['config']['reader_rel'],reader_sha256=saved['config']['reader_sha256'],
             source_h5_sha256=file_sha(out/'source.h5'),whitening_sha256=file_sha(out/'whitening.pt'),whitening_policy='originalTRAIN32statsfixed across arms,no held statistics',max_updates=steps,eval_steps=sorted(set([0,steps//4,steps//2,steps])),max_train_wall_seconds_per_arm=900,lr=1e-4,batch=8,betas=[.9,.99],weight_decay=.01,clip=1.,text_dropout=0.,input='zeros,t999,oracleceil(N/8)',auxiliary_weights=saved['config']['auxiliary_weights'],schedule_seed=9142,selection='meanaxisRMSE+.25segmentRMSE+.1penerror;each armsactualTRAINwithinfixed73probe;neverunseen',not_promoted=True,
             caveats='broad changes textcoverageANDwriterdiversity;32->41isolates samewriter addition;oracle duration leaksidentity;reader corpusfamiliar;no independentreplications;zero deterministicnotstochasticgenerator',initial_state_sha256=tensor_digest(prototype.state_dict()))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        for arm,train_ids in scope['arms'].items():
            model=copy.deepcopy(prototype);base_parameters=[p for name,p in model.named_parameters() if name!='writer.weight'];opt=torch.optim.AdamW(base_parameters);opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            for g in opt.param_groups:g['lr']=1e-4
            opt.add_param_group(dict(params=[model.writer.weight],lr=1e-4));rng=torch.Generator(device='cuda');rng.set_state(saved['noise_rng_state'].cpu());schedule=list(training_schedule(train_ids,steps,8,seed=9142));folder=out/arm;folder.mkdir();history=[];seconds=0.;best_step=0;stop='budget_completed'
            def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=arm),step=18000+step,noise_rng_state=rng.get_state()),folder/name)
            def assess(step):
                with monitor.in_phase('eval/'+arm):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,splits,targets,folder,step,writers,controls=step==steps);s=score(ev,set(train_ids));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']));return s
            best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm)
            with (folder/'metrics.jsonl').open('w') as log:
                for step,batch in enumerate(schedule,1):
                    start=time.monotonic();clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,'cuda');pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi)
                    terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);loss=mse+cfg['auxiliary_weights']['xy']*terms['xy']+cfg['auxiliary_weights']['first_difference']*terms['first_difference'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
                    row=dict(step=step,sample_ids=batch,loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm)
                    log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch))
                    if step%500==0:print(dict(arm=arm,**row,train_seconds=seconds),flush=True);log.flush()
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinitecoverage objective')
                    limit=seconds>900
                    if step in cfg['eval_steps'] or limit:
                        s=assess(step);save('checkpoint-last.pt',step)
                        if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                    if limit:stop='wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozenstate drift')
            result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,stop=stop,train_seconds=seconds,codec_reader_unchanged=True,last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),training_ids_sha256=hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,opt;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),arms={a:dict(last=r['last_step'],last_groups={s:d['correct']['free_cer'] for s,d in r['history'][-1]['aggregate'].items()}) for a,r in results.items()})
