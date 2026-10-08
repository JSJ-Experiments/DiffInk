"""Eight TRAIN-line capacity gate for controlled intra-block point feedback.

No exposedDEV or new blind sources, no full256 run until this new model earns its
own fidelity gate. Only point feedback differs between matched fresh arms.
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
from .point_feedback_strokes import PointFeedbackStrokeWriter
from .autoregressive_strokes import StrokePool,decode_offsets,reconstruction_terms
from .autoregressive_study import writer_ids,inputs,calibrate,aggregate_teacher,aggregate_free
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA
from .writer_expansion import training_schedule
from .generation_study import read_sequence
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .ocr_joint_study import local_geometry
from .inkvae import edit_distance
from .resource_monitor import ResourceMonitor

PARENT='checkpoints/iam_autoregressive_study/20261008-152547'
ARMS=('no_feedback','point_feedback')


def checked_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not relative.startswith('checkpoints/iam_point_feedback/'):
        raise ValueError('bounded point-feedback path required')
    return Path(root)/p


def validate(cfg,data):
    if cfg['max_updates']!=3000 or cfg['max_train_wall_seconds']!=1800 or cfg['batch']!=8 or cfg['model_seed']!=59142 or cfg['schedule_seed']!=59143:
        raise ValueError('fixed bounded capacity protocol required')
    a,b=cfg['models']['no_feedback'],cfg['models']['point_feedback']
    if a['point_feedback'] or not b['point_feedback'] or {k:v for k,v in a.items() if k!='point_feedback'}!={k:v for k,v in b.items() if k!='point_feedback'}:
        raise ValueError('ONLY intra-block feedback differs')
    ids=data['training_ids']
    if len(set(ids))!=8 or data['fixed_train_ids']!=ids or data['exposed_dev_ids'] or set(data['records'])!=set(ids) or not set(ids)<=set(cfg['offset_stats']['train_ids']):
        raise ValueError('exact8 established TRAIN capacity gate, no DEV/confirmation')
    if cfg['source_h5_sha256']!=SOURCE_H5_SHA or cfg['reader_sha256']!=READER_SHA:raise ValueError('pinned sources required')


def prepare(repo,root='/data'):
    root=Path(root);parent=root/PARENT
    old=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text())
    for filename,sha in [('dataset.json',DATASET_SHA),('source.h5',SOURCE_H5_SHA)]:
        if file_sha(root/DATA/filename)!=sha:raise ValueError('pinned TRAIN source drift')
    ids=data['fixed_train_ids'];data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:data['records'][s] for s in ids})
    cfg={k:old[k] for k in ['vocab','writers','offset_stats','pen_weights','pen_gradient_fraction','reader_relative','reader_sha256','source_data','source_h5_sha256','source_manifest_sha256','betas','clip','torch_cpu_threads']}
    base={k:v for k,v in old['models']['adaptive'].items()}
    cfg.update(parent=PARENT,profile='standalone intra-block point autoregressive capacity test, NOT InkDiT or geometry transport',models={a:dict(base,point_feedback=a=='point_feedback') for a in ARMS},model_seed=59142,schedule_seed=59143,max_updates=3000,max_train_wall_seconds=1800,batch=8,lr=3e-4,lr_final=5e-5,lr_drop_step=2400,warmup_steps=100,weight_decay=0.,calibration_batches=1,eval_steps=[0,500,1500,3000],control_steps=[3000],free_max_blocks=256,
               selection='minimum8TRAIN teacher normalized offsetMSE then negative minimum penF1; no DEV; final outputs shown regardless of selection',
               loss='same normalized index-offsetMSE + bounded gamma2 focal pen; fixed shared coefficient calibrated on point_feedback full-model gradients; valid real points only',
               attention='same adaptive eight-point history Gaussian clock both arms; previousblock history and previousblock lastpoint unchanged; ONLY points1..7 receive own/teacher predecessor within block in feedback arm',
               teacher='coarse shifted previousblock recurrence; independent fused8step pointGRU perblock starts from coarse decoder state; each prediction sees strictly previouspoint, never current/future target',
               generation='requested text/writer, own previousblock and own intra-block point history, common2048point cap, learned firstEOC; no reference/source length/oracleclock',
               caveats='eight TRAIN-line capacity diagnostic only, not compositional generalization; deterministic MSE can average continuations; one seed; frozen corpus-familiar reader',no_new_confirmation=True,not_promoted=True,no_kl=True,no_style=True,no_ocr_training_loss=True)
    out=root/'checkpoints/iam_point_feedback'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            basepath=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for f in sorted(basepath.rglob('*.py')):
                if '__pycache__' not in f.parts:archive.add(f,arcname=directory+'/'+str(f.relative_to(basepath)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz');validate(cfg,data)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'dataset.json').write_text(json.dumps(data,indent=2)+'\n')
    return str(out.relative_to(root))


def learning_rate(step,cfg):
    if not 1<=step<=cfg['max_updates']:raise ValueError('bounded update required')
    if step<=cfg['warmup_steps']:return cfg['lr']*(.2+.8*step/cfg['warmup_steps'])
    return cfg['lr'] if step<cfg['lr_drop_step'] else cfg['lr_final']


@torch.no_grad()
def evaluate(model,pool,targets,reader,data,cfg,folder,step):
    was=model.training;model.eval();device=next(model.parameters()).device;ids=data['training_ids'];teacher=[];free=[]
    f,o,p,m,t=pool.select(ids);wi=writer_ids(ids,data,cfg,device);pred,pen,tr=model.teacher(f,t,wi)
    xy=decode_offsets(pred,cfg['offset_stats']).reshape(len(ids),-1,2).cumsum(1);hard=pen.argmax(-1).reshape(len(ids),-1)
    with h5py.File(folder/f'evaluation-{step}.h5','w') as hf:
        for j,sid in enumerate(ids):
            q=targets[sid];n=len(q);truth=q.cpu().numpy();points=torch.cat((xy[j,:n],torch.nn.functional.one_hot(hard[j,:n],3).float()),-1).cpu().numpy()
            row=dict(sample_id=sid,points=n,offset_mse=float((pred[j]-o[j])[m[j]].square().mean()),geometry=local_geometry(points[:,:2],truth[:,:2],truth[:,2:].argmax(-1)),pen=boundary_metrics(points[:,2:].argmax(-1),truth[:,2:].argmax(-1)),last_center=float(tr['center'][j,(n-1)//8]),source_history=True,source_length=True,not_generation=True)
            teacher.append(row);g=hf.create_group('teacher/'+sid);g.create_dataset('points',data=points,compression='gzip');g.attrs['row']=json.dumps(row)
        for policy in ['correct','swapped'] if step in cfg['control_steps'] else ['correct']:
            texts=[data['records'][ids[(j+1)%len(ids)]]['text'] if policy=='swapped' else data['records'][s]['text'] for j,s in enumerate(ids)]
            generated=model.generate(inputs(texts,cfg['vocab'],device),wi,cfg['offset_stats'],cfg['free_max_blocks'])
            for j,sid in enumerate(ids):
                n=int(generated['stops'][j]);points=generated['points'][j,:n];decoded=read_sequence(reader,points,cfg['vocab']);r=data['records'][sid]
                row=dict(sample_id=sid,policy=policy,text=r['text'],conditioning_text=texts[j],writer_id=r['writer_id'],decoded=decoded,characters=len(r['text']),errors=edit_distance(r['text'],decoded),generated_points=n,found_eoc=bool(generated['found_eoc'][j]),no_source_history=True,no_source_length=True)
                free.append(row);g=hf.create_group('free/'+policy+'/'+sid);g.create_dataset('points',data=points.cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    # Per-phase displacements distinguish point-feedback gain from drift accumulation.
    previous_state=torch.cat((torch.full_like(p.reshape(len(ids),-1)[:,:1],2),p.reshape(len(ids),-1)[:,:-1]),1).reshape_as(p)
    within=m&(previous_state==0);phase=[]
    for j in range(8):
        err=(pred-o)[:,:,j];phase.append(dict(phase=j,real_points=int(m[:,:,j].sum()),within_stroke_points=int(within[:,:,j].sum()),all_per_axis_mse=float(err[m[:,:,j]].square().mean()),within_per_axis_mse=float(err[within[:,:,j]].square().mean())))
    result=dict(step=step,teacher=aggregate_teacher(teacher),teacher_lines=teacher,free={policy:aggregate_free([r for r in free if r['policy']==policy]) for policy in sorted({r['policy'] for r in free})},free_lines=free,block_phase=phase,packed_h5_sha256=file_sha(folder/f'evaluation-{step}.h5'))
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(step=step,teacher=result['teacher'],free=result['free']),flush=True);model.train(was);return result


def run(arm,relative,root='/data',on_checkpoint=None):
    if arm not in ARMS or not torch.cuda.is_available():raise ValueError('explicit T4 arm required')
    root=Path(root);out=checked_path(root,relative);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text());validate(cfg,data)
    if file_sha(out/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA:raise ValueError('source/data drift')
    torch.set_num_threads(cfg['torch_cpu_threads']);folder=out/arm;folder.mkdir(exist_ok=False)
    saved=torch.load(root/READER_REL,map_location='cpu',weights_only=True);reader,_=load_reader(root,saved['config']['cfg'],'cuda');reader.requires_grad_(False);rd=tensor_digest(reader.state_dict());del saved
    with h5py.File(root/DATA/'source.h5') as f:targets={s:torch.tensor(f[s]['target'][:],device='cuda') for s in data['training_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats'],device='cuda')
    torch.manual_seed(cfg['model_seed']);model=PointFeedbackStrokeWriter(**cfg['models'][arm]).cuda();initial=tensor_digest(model.state_dict())
    schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],accumulation=cfg['batch'],seed=cfg['schedule_seed']))
    # Shared initial-gradient coefficient on feedback surrogate, regardless of arm.
    torch.manual_seed(cfg['model_seed']);surrogate=PointFeedbackStrokeWriter(**cfg['models']['point_feedback']).cuda();surrogate.load_state_dict(model.state_dict());calibration=calibrate(surrogate,pool,data,cfg,schedule);del surrogate
    calibration['definition']='.25 * initial full-model offset gradient / bounded focalpen gradient on identical feedback surrogate and same first TRAIN8 batch; fixed shared coefficient both arms'
    (folder/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n');torch.manual_seed(cfg['model_seed']+1)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);history=[];seconds=0.;clipped=0;best=(float('inf'),float('inf'));best_step=0;stop='budget_completed'
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,config=cfg,arm=arm,calibration=calibration,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
    def assess(step,monitor):
        with monitor.in_phase('eval/'+arm):r=evaluate(model,pool,targets,reader,data,cfg,folder,step)
        history.append(dict(step=step,teacher=r['teacher'],free=r['free']));return (r['teacher']['offset_mse'],-r['teacher']['min_pen_f1'])
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        best=assess(0,monitor);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        if on_checkpoint:on_checkpoint()
        model.train();monitor.set_phase('train/'+arm)
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic()
                for g in opt.param_groups:g['lr']=learning_rate(step,cfg)
                f,o,p,m,t=pool.select(batch);pred,pen,tr=model.teacher(f,t,writer_ids(batch,data,cfg,'cuda'));terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']))
                loss=terms['offset']+calibration['weight']*terms['pen'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,sample_ids=batch,loss=float(loss.detach()),offset_mse=float(terms['offset'].detach()),pen_loss=float(terms['pen'].detach()),pen_weight=calibration['weight'],raw_grad_norm=norm,clipped=norm>cfg['clip'],lr=opt.param_groups[0]['lr'],mean_advance=float(tr['advance'].detach().mean()),mean_sigma=float(tr['sigma'].detach().mean()))
                elapsed=time.monotonic()-start;seconds+=elapsed;clipped+=row['clipped'];monitor.step(elapsed,len(batch));log.write(json.dumps(row)+'\n')
                if not math.isfinite(row['loss']):raise FloatingPointError('nonfinite objective')
                if step%250==0:log.flush();print(dict(arm=arm,train_seconds=seconds,**row),flush=True)
                limit=seconds>cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step,monitor);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in reader.parameters()):raise ValueError('reader mutated')
    result=dict(arm=arm,last_step=step,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),calibration=calibration,parameters=sum(p.numel() for p in model.parameters()),reader_unchanged=True,not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
