"""Bounded paired genuine autoregressive offset experiment on established TRAIN256.

Teacher forcing is explicitly shifted previous8-point blocks. Compare identical
fresh weights, order and loss: fixed text clock versus adaptive positive clock.
No new blind sources opened; DEV8 is already exposed and never selects a model.
"""
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
import h5py
import numpy as np
import torch
from .autoregressive_strokes import MonotonicStrokeWriter,StrokePool,fit_offset_stats,decode_offsets,reconstruction_terms
from .generation_capacity import DATA,DATASET_SHA,SOURCE_H5_SHA
from .writer_expansion import training_schedule
from .generation_geometry import gradient_norm
from .generation_study import read_sequence
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .ocr_joint_study import local_geometry
from .inkvae import edit_distance
from .resource_monitor import ResourceMonitor

PARENT='checkpoints/iam_generation_weak_continuation/20261008-134416'
ARMS=('fixed','adaptive')


def checked_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not relative.startswith('checkpoints/iam_autoregressive_study/'):
        raise ValueError('bounded autoregressive study path required')
    return Path(root)/p


def validate(cfg,data):
    if cfg['max_updates']!=8000 or cfg['batch']!=16 or cfg['model_seed']!=58142 or cfg['schedule_seed']!=58143 or cfg['max_train_wall_seconds']!=3600:
        raise ValueError('fixed bounded paired pilot required')
    a,b=cfg['models']['fixed'],cfg['models']['adaptive']
    if a['adaptive'] or not b['adaptive'] or {k:v for k,v in a.items() if k!='adaptive'}!={k:v for k,v in b.items() if k!='adaptive'}:
        raise ValueError('ONLY fixed/adaptive clock policy differs')
    train=data['training_ids'];dev=data['exposed_dev_ids'];probe=data['fixed_train_ids']
    if len(set(train))!=256 or len(set(dev))!=8 or len(set(probe))!=8 or not set(probe)<=set(train) or set(train)&set(dev):
        raise ValueError('established unique TRAIN256, fixedTRAIN8, exposedDEV8 required')
    if set(data['records'])!=set(train+dev):raise ValueError('no new blind source metadata allowed')
    if set(cfg['offset_stats']['train_ids'])!=set(train):raise ValueError('TRAIN-only offset calibration required')


def prepare(repo,root='/data'):
    root=Path(root);p=root/PARENT;source=root/DATA
    for filename,sha in [('dataset.json',DATASET_SHA),('source.h5',SOURCE_H5_SHA)]:
        if file_sha(source/filename)!=sha:raise ValueError('pinned established source drift')
    old=json.loads((p/'config.json').read_text());scope=json.loads((p/'dataset.json').read_text());train=scope['training_ids']['control'];dev=scope['splits']['unseen_prompt']
    if train!=scope['training_ids']['weak_alignment']:raise ValueError('identical established TRAIN required')
    records={s:scope['records'][s] for s in train+dev}
    with h5py.File(source/'source.h5') as f:targets={s:torch.tensor(f[s]['target'][:]) for s in train}
    stats=fit_offset_stats(targets,train);states=torch.cat([q[:,2:].argmax(-1) for q in targets.values()]);counts=torch.bincount(states,minlength=3).double()
    weights=(counts[0]/counts.clamp_min(1)).sqrt().clamp_max(8.).tolist()
    advance=sum(len(records[s]['text']) for s in train)/sum((records[s]['points']+7)//8 for s in train)
    base=dict(vocab_size=len(old['vocab']),writer_count=len(old['writers']),initial_advance=advance,width=192,text_width=64,writer_width=16)
    probe=sorted(train,key=lambda s:hashlib.sha256(('ar-fixed-train:'+s).encode()).hexdigest())[:8]
    out=root/'checkpoints/iam_autoregressive_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model']:
            basepath=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for f in sorted(basepath.rglob('*.py')):
                if '__pycache__' not in f.parts:archive.add(f,arcname=directory+'/'+str(f.relative_to(basepath)))
    data=dict(training_ids=train,exposed_dev_ids=dev,fixed_train_ids=probe,records=records)
    cfg=dict(profile='standalone autoregressive8-point index-offset/pen baseline; NOT learned semanticInkVAE, releasedInkDiT or exact Graves reproduction',parent=PARENT,
             source_data=DATA,source_h5_sha256=SOURCE_H5_SHA,source_manifest_sha256=DATASET_SHA,reader_relative=READER_REL,reader_sha256=READER_SHA,
             vocab=old['vocab'],writers=old['writers'],models={a:dict(base,adaptive=a=='adaptive') for a in ARMS},
             model_seed=58142,schedule_seed=58143,max_updates=8000,max_train_wall_seconds=3600,batch=16,lr=2e-4,lr_final=5e-5,lr_drop_step=6000,warmup_steps=200,
             betas=[.9,.99],weight_decay=1e-4,clip=5.,torch_cpu_threads=2,offset_stats=stats,pen_weights=weights,pen_gamma=2.,pen_gradient_fraction=.25,calibration_batches=8,
             eval_steps=[0,1000,2000,4000,8000],control_steps=[8000],free_max_blocks=256,
             loss='normalized chronological XY index-offset MSE + fixed gradient-calibrated bounded focal pen; valid points only; no stop-padding supervision',
             selection='fixed8TRAIN free-running CER then all256 teacher-offsetMSE; NO exposedDEV selection; teacher path has source history and source length, never call it generation',
             feedback='previous eight chronological normalized index displacements and hard pen states; BOS zero vector+flag; teacher histories at training, own predicted history at generation',
             attention='single Gaussian center starts -advance/2; positive advances .01+softplus(history GRU clock); sigma=.5+softplus clipped4; fixed arm advance TRAINratio, sigma1; character embeddings only, explicit textEOS; no global text summary/absolute queryPE',
             generation='only text/writer/normalization and common2048point cap; learned firstEOC; no target length, timing teacher, stroke reference, duration predictor or forced stop',
             caveats='Deterministic MSE may average multiple valid continuations; exposure bias; jumps/delayed marks; block8 coarsens progress; one seed and256TRAIN; frozen corpus-familiar reader; no independent new composition gate',
             fresh=True,no_neural_initialization=True,no_kl=True,no_style=True,no_ocr_training_loss=True,no_new_confirmation=True,not_promoted=True,
             source_archive_sha256=file_sha(out/'as-run-source.tar.gz'))
    validate(cfg,data);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'dataset.json').write_text(json.dumps(data,indent=2)+'\n')
    return str(out.relative_to(root))


def writer_ids(ids,data,cfg,device):return torch.tensor([cfg['writers'].index(data['records'][s]['writer_id']) for s in ids],dtype=torch.long,device=device)


def inputs(texts,vocab,device):
    a=torch.full((len(texts),max(map(len,texts))),-1,dtype=torch.long,device=device)
    for j,t in enumerate(texts):a[j,:len(t)]=torch.tensor([vocab.index(c) for c in t],device=device)
    return a


def calibrate(model,pool,data,cfg,schedule):
    digest=tensor_digest(model.state_dict());rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    parameters=list(model.parameters());rows=[]
    for batch in schedule[:cfg['calibration_batches']]:
        f,o,p,m,t=pool.select(batch);w=writer_ids(batch,data,cfg,f.device);pred,pen,_=model.teacher(f,t,w)
        terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']))
        on=gradient_norm(terms['offset'],parameters);pn=gradient_norm(terms['pen'],parameters)
        if min(on,pn)<=1e-12 or not np.isfinite([on,pn]).all():raise ValueError('nondegenerate initial objective gradients required')
        rows.append(dict(sample_ids=batch,offset_norm=on,pen_norm=pn,offset_loss=float(terms['offset'].detach()),pen_loss=float(terms['pen'].detach())))
    weight=cfg['pen_gradient_fraction']*float(np.median([r['offset_norm'] for r in rows]))/float(np.median([r['pen_norm'] for r in rows]))
    if tensor_digest(model.state_dict())!=digest or not torch.equal(rng,torch.get_rng_state()) or (crng is not None and not torch.equal(crng,torch.cuda.get_rng_state())):raise ValueError('calibration state/RNG mutation')
    return dict(weight=weight,fraction=cfg['pen_gradient_fraction'],rows=rows,state_rng_unchanged=True,
                definition='.25 * median initial full-model offset gradient / median bounded focal pen gradient over first8 TRAIN batches; same initial forward botharms; fixed thereafter')


def aggregate_teacher(rows):
    return dict(lines=len(rows),offset_mse=float(np.average([r['offset_mse'] for r in rows],weights=[r['points'] for r in rows])),
                x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in rows])),y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in rows])),
                mean_line_first_difference_rmse=float(np.mean([r['geometry']['first_difference']['vector_rmse'] or 0 for r in rows])),
                mean_line_turn_p90=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] or 0 for r in rows])),
                min_pen_f1=min(r['pen']['pen_up_f1'] for r in rows),false_internal_eoc=sum(r['pen']['non_final_false_eoc_count'] for r in rows),final_eoc_correct=sum(r['pen']['final_eoc_correct'] for r in rows),
                mean_final_attention_center=float(np.mean([r['last_center'] for r in rows])))


def aggregate_free(rows):
    return dict(lines=len(rows),cer=sum(r['errors'] for r in rows)/sum(r['characters'] for r in rows),exact=sum(r['errors']==0 for r in rows),missing_eoc=sum(not r['found_eoc'] for r in rows),
                median_generated_points=float(np.median([r['generated_points'] for r in rows])))


@torch.no_grad()
def evaluate(model,pool,targets,reader,data,cfg,folder,step,controls=False):
    was=model.training;model.eval();device=next(model.parameters()).device;teacher=[];free=[]
    with h5py.File(folder/f'evaluation-{step}.h5','w') as hf:
        for start in range(0,len(data['training_ids']),16):
            batch=data['training_ids'][start:start+16];f,o,p,m,t=pool.select(batch);wi=writer_ids(batch,data,cfg,device);pred,pen,tr=model.teacher(f,t,wi)
            xy=decode_offsets(pred,cfg['offset_stats']).reshape(len(batch),-1,2).cumsum(1);hard=pen.argmax(-1).reshape(len(batch),-1)
            for j,sid in enumerate(batch):
                q=targets[sid];n=len(q);points=torch.cat((xy[j,:n],torch.nn.functional.one_hot(hard[j,:n],3).float()),-1).cpu().numpy();truth=q.cpu().numpy()
                row=dict(sample_id=sid,points=n,offset_mse=float((pred[j]-o[j])[m[j]].square().mean()),geometry=local_geometry(points[:,:2],truth[:,:2],truth[:,2:].argmax(-1)),pen=boundary_metrics(points[:,2:].argmax(-1),truth[:,2:].argmax(-1)),last_center=float(tr['center'][j,(n-1)//8]),
                         source_history=True,source_length=True,not_generation=True)
                teacher.append(row);g=hf.create_group('teacher/'+sid);g.create_dataset('points',data=points,compression='gzip');g.create_dataset('centers',data=tr['center'][j,:(n+7)//8].cpu().numpy());g.attrs['row']=json.dumps(row)
        for split,ids in [('fixed_train8',data['fixed_train_ids']),('exposed_dev8',data['exposed_dev_ids'])]:
            for policy in ['correct','swapped'] if controls else ['correct']:
                text=[data['records'][ids[(j+1)%len(ids)]]['text'] if policy=='swapped' else data['records'][s]['text'] for j,s in enumerate(ids)]
                t=inputs(text,cfg['vocab'],device);wi=writer_ids(ids,data,cfg,device);generated=model.generate(t,wi,cfg['offset_stats'],max_blocks=cfg['free_max_blocks'])
                for j,sid in enumerate(ids):
                    n=int(generated['stops'][j]);points=generated['points'][j,:n];decoded=read_sequence(reader,points,cfg['vocab']);r=data['records'][sid]
                    row=dict(sample_id=sid,split=split,policy=policy,text=r['text'],conditioning_text=text[j],writer_id=r['writer_id'],decoded=decoded,characters=len(r['text']),errors=edit_distance(r['text'],decoded),generated_points=n,found_eoc=bool(generated['found_eoc'][j]),
                             final_center=float(generated['traces']['center'][j,(n-1)//8]),no_source_history=True,no_source_length=True)
                    free.append(row);g=hf.create_group('free/'+split+'/'+policy+'/'+sid);g.create_dataset('points',data=points.cpu().numpy(),compression='gzip');g.create_dataset('centers',data=generated['traces']['center'][j,:(n+7)//8].cpu().numpy());g.attrs['row']=json.dumps(row)
    model.train(was);groups={s:{p:aggregate_free([r for r in free if r['split']==s and r['policy']==p]) for p in sorted({r['policy'] for r in free})} for s in ['fixed_train8','exposed_dev8']}
    result=dict(step=step,teacher=aggregate_teacher(teacher),free=groups,teacher_lines=teacher,free_lines=free,packed_h5_sha256=file_sha(folder/f'evaluation-{step}.h5'))
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(step=step,teacher=result['teacher'],free=groups),flush=True)
    return result


def learning_rate(step,cfg):
    if not 1<=step<=cfg['max_updates']:raise ValueError('bounded actual update required')
    if step<=cfg['warmup_steps']:return cfg['lr']*(.2+.8*step/cfg['warmup_steps'])
    return cfg['lr'] if step<cfg['lr_drop_step'] else cfg['lr_final']


def run(arm,relative,root='/data',on_checkpoint=None):
    if arm not in ARMS or not torch.cuda.is_available():raise ValueError('explicit arm/T4 required')
    root=Path(root);out=checked_path(root,relative);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text());validate(cfg,data)
    if file_sha(out/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA:raise ValueError('immutable source archive/data guard')
    torch.set_num_threads(cfg['torch_cpu_threads']);folder=out/arm;folder.mkdir(exist_ok=False)
    reader_saved=torch.load(root/READER_REL,map_location='cpu',weights_only=True);reader,_=load_reader(root,reader_saved['config']['cfg'],'cuda');reader.requires_grad_(False);rd=tensor_digest(reader.state_dict());del reader_saved
    with h5py.File(root/DATA/'source.h5') as f:targets={s:torch.tensor(f[s]['target'][:],device='cuda') for s in data['training_ids']+data['exposed_dev_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats'],device='cuda')
    torch.manual_seed(cfg['model_seed']);model=MonotonicStrokeWriter(**cfg['models'][arm]).cuda();initial=tensor_digest(model.state_dict())
    schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],accumulation=cfg['batch'],seed=cfg['schedule_seed']))
    calibration=calibrate(model,pool,data,cfg,schedule);(folder/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    # Init clock weights zero gives same forward and same shared calibration in
    # both arms. Fixed arm has zero clock-only gradients, so measure coefficient
    # on adaptive surrogate with identical weights to isolate objective scale.
    if arm=='fixed':
        surrogate=MonotonicStrokeWriter(**cfg['models']['adaptive']).cuda();surrogate.load_state_dict(model.state_dict());calibration=calibrate(surrogate,pool,data,cfg,schedule);del surrogate
        (folder/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    torch.manual_seed(cfg['model_seed']+1) # matched post-calibration seed, no stochastic dropout/noise
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
    history=[];best=(float('inf'),float('inf'));best_step=0;clipped=0;seconds=0.;stop='budget_completed'
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,config=cfg,arm=arm,calibration=calibration,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
    def assess(step,monitor):
        with monitor.in_phase('eval/'+arm):r=evaluate(model,pool,targets,reader,data,cfg,folder,step,controls=step in cfg['control_steps'])
        history.append(dict(step=step,teacher=r['teacher'],free=r['free']));return (r['free']['fixed_train8']['correct']['cer'],r['teacher']['offset_mse'])
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        best=assess(0,monitor);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        if on_checkpoint:on_checkpoint()
        model.train();monitor.set_phase('train/'+arm)
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic()
                for g in opt.param_groups:g['lr']=learning_rate(step,cfg)
                f,o,p,m,t=pool.select(batch);wi=writer_ids(batch,data,cfg,'cuda');pred,pen,tr=model.teacher(f,t,wi);terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']))
                loss=terms['offset']+calibration['weight']*terms['pen'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,sample_ids=batch,loss=float(loss.detach()),offset_mse=float(terms['offset'].detach()),pen_loss=float(terms['pen'].detach()),pen_weight=calibration['weight'],raw_grad_norm=norm,clipped=norm>cfg['clip'],lr=opt.param_groups[0]['lr'],mean_advance=float(tr['advance'].detach().mean()),mean_sigma=float(tr['sigma'].detach().mean()))
                elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped'];log.write(json.dumps(row)+'\n')
                if not math.isfinite(row['loss']):raise FloatingPointError('nonfinite autoregressive loss')
                if step%250==0:log.flush();print(dict(arm=arm,train_seconds=seconds,**row),flush=True)
                limit=seconds>cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step,monitor);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in reader.parameters()):raise ValueError('frozen reader drift')
    result=dict(arm=arm,last_step=step,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,
                selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),calibration=calibration,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),reader_unchanged=True,not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
