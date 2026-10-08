"""Matched low-LR continuation: true history vs modest generated block roll-in.

Eight previously exposed TRAIN capacity lines, not a composition/generation gate.
Frozen reader is evaluator only. Same model/optimizer/source/order/objective;
roll-in selects detached predicted XY+hardpens together after each current block.
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
from .history_rollin import history_forward,rollin_probability,transition_mask
from .point_feedback_strokes import PointFeedbackStrokeWriter
from .autoregressive_strokes import StrokePool,reconstruction_terms
from .autoregressive_study import writer_ids
from .point_feedback_study import evaluate as evaluate_capacity
from .writer_expansion import training_schedule
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .generation_study import read_sequence
from .inkvae import edit_distance
from .pen_ab import file_sha
from .ocr_context_study import tensor_digest
from .resource_monitor import ResourceMonitor
from .launch_ledger import study_path

PARENT='checkpoints/iam_point_feedback/20261008-160806'
ARMS=('teacher','rollin')
PARENT_SHA='8786c3a2a37bd85f19a93bb527c2293a019567137c3f52864aebcb4f2437d959'


def checked_path(root,relative):return study_path(root,relative,'checkpoints/iam_history_rollin/')


def validate(cfg,data):
    if cfg['parent_checkpoint_sha256']!=PARENT_SHA or cfg['source_h5_sha256']!=SOURCE_H5_SHA or cfg['reader_sha256']!=READER_SHA:
        raise ValueError('exact pinned parent/source/reader required')
    if cfg['max_updates']!=1000 or cfg['max_train_wall_seconds']!=1800 or cfg['batch']!=8 or cfg['lr']!=1e-5 or cfg['schedule_seed']!=60142 or cfg['rollin_seed']!=60143 or cfg['rollin_max_probability']!=.2 or cfg['rollin_ramp_steps']!=500:
        raise ValueError('fixed bounded matched low-LR continuation required')
    if cfg['models']['teacher']!=cfg['models']['rollin'] or cfg['models']['teacher']['point_feedback']:
        raise ValueError('same no-point-feedback model both arms required')
    ids=data['training_ids']
    if len(set(ids))!=8 or data['fixed_train_ids']!=ids or data['exposed_dev_ids'] or set(data['records'])!=set(ids):
        raise ValueError('exact8 established TRAIN, no DEV/confirmation required')
    if cfg['pen_weight']!=.024860149190817294 or not cfg['optimizer_restored'] or not cfg['no_ocr_training_loss'] or not cfg['not_promoted']:
        raise ValueError('same optimizer/fixed objective, evaluator-only reader and no promotion required')


def prepare(repo,root='/data'):
    root=Path(root);parent=root/PARENT;checkpoint=parent/'no_feedback/checkpoint-last.pt'
    if file_sha(checkpoint)!=PARENT_SHA or file_sha(root/DATA/'source.h5')!=SOURCE_H5_SHA or file_sha(root/DATA/'dataset.json')!=DATASET_SHA:raise ValueError('pinned continuation input drift')
    old=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());pr=json.loads((parent/'no_feedback/result.json').read_text());baseline=json.loads((parent/'no_feedback/eval-3000.json').read_text())
    if pr['last_step']!=3000 or pr['last_sha256']!=PARENT_SHA:raise ValueError('completed original control parent required; never unrequested repeat')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['step']!=3000 or saved['arm']!='no_feedback' or saved['calibration']['weight']!=pr['calibration']['weight']:raise ValueError('model/optimizer parent schema drift')
    cfg={k:old[k] for k in ['vocab','writers','offset_stats','pen_weights','reader_relative','reader_sha256','source_data','source_h5_sha256','source_manifest_sha256','betas','clip','torch_cpu_threads']}
    cfg.update(parent=PARENT,parent_checkpoint_relative=PARENT+'/no_feedback/checkpoint-last.pt',parent_checkpoint_sha256=PARENT_SHA,parent_step=3000,models={a:old['models']['no_feedback'] for a in ARMS},profile='standalone stroke-writer8TRAIN history-robustness continuation, NOT EnglishInkDiT or newtext generalization',schedule_seed=60142,rollin_seed=60143,max_updates=1000,max_train_wall_seconds=1800,batch=8,lr=1e-5,weight_decay=0.,optimizer_restored=True,pen_weight=pr['calibration']['weight'],rollin_max_probability=.2,rollin_ramp_steps=500,eval_steps=[0,250,500,1000],control_steps=[0,1000],free_max_blocks=256,
               loss='same normalized chronological index-offsetMSE + fixed bounded focalpen gamma2; real points only; no cumulative-position loss, smoothing, KL/style/OCR objective',
               training='both arms same blockwise8stepGRU graph; teacher p0, rollin joint ownXY+hardpens p=.2*min(update/500,1); identical randomdraws/order, only applied policy differs; predictedfeedback detached, hidden BPTT retained',
               caveats='scheduled sampling is not unbiased likelihood training; source-index labels may be ambiguous after diverging histories; only8TRAIN/7writers, one continuation seed; frozen corpus-familiar reader; not semantic composition proof',
               selection='minimum genuine freeTRAIN8 CER then teacherMSE ONLY among teacher-capacity-preserving evaluations; baseline eligible; no newtext/visual selection',
               teacher_capacity_gate='perline axisRMSE<=max(2*parent,.02); first-index-difference vectorRMSE<=max(1.5*parent,.012); turnp90<=max(1.5*parent,30deg); penF1>=parent-.05; zero internalfalseEOC, correct finalEOC and teacherTRUE-PEN CER0 across8',
               no_kl=True,no_style=True,no_ocr_training_loss=True,no_new_confirmation=True,not_promoted=True)
    out=root/'checkpoints/iam_history_rollin'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for f in sorted(base.rglob('*.py')):
                if '__pycache__' not in f.parts:archive.add(f,arcname=directory+'/'+str(f.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz');cfg['parent_initial_model_digest']=tensor_digest(saved['model_state_dict'])
    # Strict CPU probe of the actual restore and shifted graph before allocating GPU.
    torch.set_num_threads(2);model=PointFeedbackStrokeWriter(**cfg['models']['teacher']);model.load_state_dict(saved['model_state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
    if not opt.state or tensor_digest(model.state_dict())!=cfg['parent_initial_model_digest']:raise ValueError('exact initialized parameters/optimizer required')
    validate(cfg,data)
    for name,obj in [('config.json',cfg),('dataset.json',data),('parent-eval.json',baseline)]: (out/name).write_text(json.dumps(obj,indent=2)+'\n')
    return str(out.relative_to(root))


def teacher_gate(result,reading,parent):
    reference={r['sample_id']:r for r in parent['teacher_lines']};rows=result['teacher_lines'];failures=[]
    if len(rows)!=8 or set(r['sample_id'] for r in rows)!=set(reference) or set(r['sample_id'] for r in reading['rows'])!=set(reference):raise ValueError('complete8 perline capacity gate required')
    for r in rows:
        sid=r['sample_id'];base=reference[sid];g,h=r['geometry'],base['geometry']
        for k in ['x_rmse','y_rmse']:
            if g[k]>max(2*h[k],.02):failures.append(sid+':'+k)
        if g['first_difference']['vector_rmse']>max(1.5*h['first_difference']['vector_rmse'],.012):failures.append(sid+':first_index_difference')
        if g['turn_angle_error_degrees']['p90']>max(1.5*h['turn_angle_error_degrees']['p90'],30.):failures.append(sid+':turn_p90')
        pen=r['pen']
        if pen['pen_up_f1']<base['pen']['pen_up_f1']-.05 or pen['non_final_false_eoc_count'] or not pen['final_eoc_correct']:failures.append(sid+':pen')
    if reading['true_pen_cer']!=0.:failures.append('teacher_truepen_reading')
    return dict(passed=not failures,failures=failures,definition='declared perline parent-relative position/local-geometry/pen safeguards AND all8teacherTRUE-PEN CER0; NOT promotion/generation gate')


@torch.no_grad()
def evaluate(model,pool,targets,reader,data,cfg,folder,step,parent):
    result=evaluate_capacity(model,pool,targets,reader,data,cfg,folder,step);rows=[]
    with h5py.File(folder/f'evaluation-{step}.h5') as hf:
        for sid in data['training_ids']:
            q=torch.tensor(hf['teacher/'+sid]['points'][:],device=next(model.parameters()).device);truepen=torch.cat((q[:,:2],targets[sid][:,2:]),-1);text=data['records'][sid]['text'];a=read_sequence(reader,truepen,cfg['vocab']);b=read_sequence(reader,q,cfg['vocab'])
            rows.append(dict(sample_id=sid,text=text,characters=len(text),true_pen_reader=a,predicted_pen_reader=b,true_pen_errors=edit_distance(text,a),predicted_pen_errors=edit_distance(text,b)))
    reading=dict(rows=rows,true_pen_cer=sum(r['true_pen_errors'] for r in rows)/sum(r['characters'] for r in rows),predicted_pen_cer=sum(r['predicted_pen_errors'] for r in rows)/sum(r['characters'] for r in rows),scope='ALL8 TRUE-HISTORY reconstructions, not generation')
    gate=teacher_gate(result,reading,parent);reading['capacity_gate']=gate
    # Conditioning-swap control error against original and actually supplied text.
    if 'swapped' in result['free']:
        rr=[r for r in result['free_lines'] if r['policy']=='swapped'];reading['swap_conditioning_cer']=sum(edit_distance(r['conditioning_text'],r['decoded']) for r in rr)/sum(len(r['conditioning_text']) for r in rr)
    (folder/f'teacher-reading-{step}.json').write_text(json.dumps(reading,indent=2)+'\n')
    result['reading']=reading;result['capacity_gate']=gate;print(dict(step=step,teacher_true_pen_cer=reading['true_pen_cer'],teacher_predicted_pen_cer=reading['predicted_pen_cer'],capacity_gate=gate),flush=True);return result


def run(arm,relative,root='/data',on_checkpoint=None):
    if arm not in ARMS or not torch.cuda.is_available():raise ValueError('explicit T4 arm required')
    root=Path(root);out=checked_path(root,relative);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text());parent=json.loads((out/'parent-eval.json').read_text());validate(cfg,data)
    for p,sha in [(out/'as-run-source.tar.gz',cfg['source_archive_sha256']),(root/cfg['parent_checkpoint_relative'],PARENT_SHA),(root/DATA/'source.h5',SOURCE_H5_SHA)]:
        if file_sha(p)!=sha:raise ValueError('immutable continuation source drift')
    torch.set_num_threads(cfg['torch_cpu_threads']);folder=out/arm;folder.mkdir(exist_ok=False)
    saved_reader=torch.load(root/READER_REL,map_location='cpu',weights_only=True);reader,_=load_reader(root,saved_reader['config']['cfg'],'cuda');reader.requires_grad_(False);rd=tensor_digest(reader.state_dict());del saved_reader
    with h5py.File(root/DATA/'source.h5') as hf:targets={s:torch.tensor(hf[s]['target'][:],device='cuda') for s in data['training_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats'],device='cuda')
    saved=torch.load(root/cfg['parent_checkpoint_relative'],map_location='cpu',weights_only=False);model=PointFeedbackStrokeWriter(**cfg['models'][arm]).cuda();model.load_state_dict(saved['model_state_dict']);initial=tensor_digest(model.state_dict())
    if initial!=cfg['parent_initial_model_digest']:raise ValueError('same exact parent parameters required')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict']);optimizer_initial=tensor_digest({str(k)+':'+str(n):v for k,s in opt.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v)})
    for group in opt.param_groups:group['lr']=cfg['lr']
    torch.manual_seed(cfg['rollin_seed']);rng=torch.Generator(device='cuda').manual_seed(cfg['rollin_seed']);schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));history=[];best=(float('inf'),float('inf'));best_step=0;seconds=0.;clipped=0;stop='budget_completed'
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,parent_step=3000,total_step=3000+step,config=cfg,arm=arm,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rollin_rng=rng.get_state()),folder/name)
    def assess(step,monitor):
        with monitor.in_phase('eval/'+arm):r=evaluate(model,pool,targets,reader,data,cfg,folder,step,parent)
        history.append(dict(step=step,teacher=r['teacher'],free=r['free'],reading=r['reading'],capacity_gate=r['capacity_gate']))
        return (r['free']['correct']['cer'],r['teacher']['offset_mse']) if r['capacity_gate']['passed'] else None
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        best=assess(0,monitor)
        if best is None:raise ValueError('restored initial capacity gate must PASS')
        save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        if on_checkpoint:on_checkpoint()
        model.train();monitor.set_phase('train/'+arm)
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic();f,o,p,m,t=pool.select(batch);draws=torch.rand(len(batch),f.shape[1]-1,device='cuda',generator=rng);prob=rollin_probability(step,cfg['rollin_max_probability'],cfg['rollin_ramp_steps']) if arm=='rollin' else 0.;selected=transition_mask(draws,m,prob)
                pred,pen,tr=history_forward(model,f,t,writer_ids(batch,data,cfg,'cuda'),selected,selected);terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']));loss=terms['offset']+cfg['pen_weight']*terms['pen'];opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                eligible=int(m[:,1:].any(-1).sum());chosen=int(selected.sum());row=dict(step=step,total_step=3000+step,sample_ids=batch,loss=float(loss.detach()),offset_mse=float(terms['offset'].detach()),pen_loss=float(terms['pen'].detach()),pen_weight=cfg['pen_weight'],lr=cfg['lr'],raw_grad_norm=norm,clipped=norm>cfg['clip'],rollin_probability=prob,eligible_transitions=eligible,selected_transitions=chosen,selected_fraction=chosen/eligible if eligible else 0.,draws_sha256=hashlib.sha256(draws.cpu().numpy().tobytes()).hexdigest(),selected_sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest(),mean_advance=float(tr['advance'].detach().mean()),mean_sigma=float(tr['sigma'].detach().mean()))
                elapsed=time.monotonic()-start;seconds+=elapsed;clipped+=row['clipped'];monitor.step(elapsed,len(batch));log.write(json.dumps(row)+'\n')
                if not math.isfinite(row['loss']):raise FloatingPointError('nonfinite history continuation objective')
                if step%100==0:log.flush();print(dict(arm=arm,train_seconds=seconds,**row),flush=True)
                limit=seconds>cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step,monitor);save('checkpoint-last.pt',step)
                    if score is not None and score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in reader.parameters()):raise ValueError('frozen reader drift')
    result=dict(arm=arm,last_step=step,total_step=3000+step,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,initial_optimizer_tensor_digest=optimizer_initial,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),reader_unchanged=True,not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
