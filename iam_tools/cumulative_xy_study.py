"""Accumulated-coordinate anchor study, still an eight-TRAIN-line research toy.

Head-refitted source isolates continuous-history drift. Fresh, identical AdamW
state in all three arms (the independently refitted pen head has no compatible
whole-model optimizer). No oracle information enters actual free generation.
"""
import hashlib,json,math,tarfile,time
from pathlib import Path
import h5py,numpy as np,torch
from .cumulative_xy import cumulative_xy_loss,anchor_coefficient
from .generation_geometry import gradient_norm
from .history_rollin import history_forward,rollin_probability,transition_mask
from .history_rollin_study import teacher_gate
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
HEAD=PARENT+'/history-interventions/pen-readout-refit-continued-recovered'
PARENT_SHA='38102f10d4d42d33171bcd53124771587ead94d79899eadcf25af28c2c1efdaf'
ARMS=('teacher','teacher_anchor','rollin_anchor')

def checked_path(root,relative):return study_path(root,relative,'checkpoints/iam_cumulative_xy/')

def validate(cfg,data):
    if cfg['parent_checkpoint_sha256']!=PARENT_SHA or cfg['source_h5_sha256']!=SOURCE_H5_SHA or cfg['reader_sha256']!=READER_SHA:raise ValueError('exact pinned head-refitted source required')
    if cfg['max_updates']!=1000 or cfg['batch']!=8 or cfg['max_train_wall_seconds']!=1800 or cfg['lr']!=1e-5 or cfg['schedule_seed']!=62142 or cfg['rollin_seed']!=62143 or cfg['rollin_max_probability']!=.2 or cfg['rollin_ramp_steps']!=500:raise ValueError('fixed bounded matched protocol required')
    if set(cfg['models'])!=set(ARMS) or any(m!=cfg['models']['teacher'] or m['point_feedback'] for m in cfg['models'].values()):raise ValueError('identical no-intra-block-feedback architecture required')
    ids=data['training_ids']
    if len(set(ids))!=8 or data['fixed_train_ids']!=ids or data['exposed_dev_ids'] or set(data['records'])!=set(ids):raise ValueError('exact8 establishedTRAIN, no new DEV/confirmation')
    c=cfg['anchor_calibration']
    if not c['state_rng_unchanged'] or c['gradient_fraction']!=.15 or not math.isclose(c['weight'],anchor_coefficient(c['offset_norm'],c['anchor_norm'],.15),rel_tol=1e-12):raise ValueError('fixed modest initial teacher-gradient calibration required')
    if cfg['optimizer_restored'] or cfg['pen_weight']!=.024860149190817294 or not cfg['no_ocr_training_loss'] or not cfg['not_promoted']:raise ValueError('shared fresh optimizer and fixed pen/evaluator contract required')

def prepare(repo,root='/data'):
    root=Path(root);parent=root/PARENT;head=root/HEAD;checkpoint=head/'checkpoint-selected.pt'
    for f,sha in [(checkpoint,PARENT_SHA),(root/DATA/'source.h5',SOURCE_H5_SHA),(root/DATA/'dataset.json',DATASET_SHA),(root/READER_REL,READER_SHA)]:
        if file_sha(f)!=sha:raise ValueError('immutable source drift: '+str(f))
    old=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());hr=json.loads((head/'result.json').read_text());baseline=json.loads((head/'eval-2000.json').read_text());saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['head_updates']!=6000 or not hr['teacher_xy_bit_identical'] or not hr['only_pen_rows_changed'] or hr['teacher_predicted_pen_cer']!=0.:raise ValueError('completed frozen geometry pen-refit source required')
    cfg={k:old[k] for k in ['vocab','writers','offset_stats','pen_weights','reader_relative','reader_sha256','source_data','source_h5_sha256','source_manifest_sha256','betas','clip','torch_cpu_threads']}
    cfg.update(parent=PARENT,parent_checkpoint_relative=HEAD+'/checkpoint-selected.pt',parent_checkpoint_sha256=PARENT_SHA,parent_body_step=3000,parent_head_updates=6000,models={a:old['models']['no_feedback'] for a in ARMS},profile='standalone stroke-writer8TRAIN cumulative-path anchor study, NOT EnglishInkDiT or newtext generalization',schedule_seed=62142,rollin_seed=62143,max_updates=1000,max_train_wall_seconds=1800,batch=8,lr=1e-5,weight_decay=0.,optimizer_restored=False,optimizer_note='identical fresh AdamW(.9,.99) allarms; independently refitted pen head lacks compatible whole-model optimizer; not directly optimizer-matched to previous study',pen_weight=.024860149190817294,rollin_max_probability=.2,rollin_ramp_steps=500,eval_steps=[0,250,500,1000],control_steps=[0,1000],free_max_blocks=256,
        loss='normalized chronological displacementMSE + original bounded focalpen gamma2; anchor arms also .15-initial-gradient calibrated cumulative-error model-unit XY MSE on all real points incl pen-up jumps/origin; no endpoint correction/smoothing',
        training='all arms same blockwise8stepGRU graph; teacher/teacher_anchor p0; rollin_anchor joint detached ownXY/hardpens p=.2*min(update/500,1); common draws/order, normal hidden BPTT',
        caveats='source-index targets after generated-history drift may be ambiguous; scheduled sampling is not unbiased likelihood; only8TRAIN/7writers one seed and corpus-familiar frozen reader; not composition proof',
        selection='minimum genuine freeTRAIN CER then teacheroffsetMSE ONLY among preserved-capacity evaluations; baselineeligible; no newtext/visual selection',
        teacher_capacity_gate='same perline parent-relative position/localgeometry/pen safeguards + teacherTRUE-PEN CER0; now headrefit parent pen baseline; no promotion',
        no_kl=True,no_style=True,no_ocr_training_loss=True,no_new_confirmation=True,not_promoted=True)
    torch.set_num_threads(2);model=PointFeedbackStrokeWriter(**cfg['models']['teacher']);model.load_state_dict(saved['model_state_dict']);model.train();digest=tensor_digest(model.state_dict());rng=torch.get_rng_state().clone()
    with h5py.File(root/DATA/'source.h5') as hf:targets={s:torch.tensor(hf[s]['target'][:]) for s in data['training_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats']);batch=list(training_schedule(data['training_ids'],1,8,cfg['schedule_seed']))[0];f,o,p,m,t=pool.select(batch);zero=torch.zeros(len(batch),f.shape[1]-1,dtype=torch.bool);pred,pen,_=history_forward(model,f,t,writer_ids(batch,data,cfg,'cpu'),zero,zero);terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']));anchor=cumulative_xy_loss(pred,o,m,cfg['offset_stats']);parameters=list(model.parameters());on=gradient_norm(terms['offset'],parameters);an=gradient_norm(anchor,parameters)
    cfg['anchor_calibration']=dict(weight=anchor_coefficient(on,an),gradient_fraction=.15,offset_norm=on,anchor_norm=an,offset_mse=float(terms['offset'].detach()),cumulative_xy_mse=float(anchor.detach()),sample_ids=batch,scope='one initial fullTRAIN8 true-history blockwise graph; full-model gradient norm; frozen shared coefficient allanchorarms',state_rng_unchanged=digest==tensor_digest(model.state_dict()) and torch.equal(rng,torch.get_rng_state()))
    cfg['parent_initial_model_digest']=digest;validate(cfg,data)
    out=root/'checkpoints/iam_cumulative_xy'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for file in sorted(base.rglob('*.py')):
                if '__pycache__' not in file.parts:archive.add(file,arcname=directory+'/'+str(file.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz')
    for name,value in [('config.json',cfg),('dataset.json',data),('parent-eval.json',baseline)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return str(out.relative_to(root))

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
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);optimizer_initial=tensor_digest({str(k)+':'+str(n):v for k,s in opt.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v)})
    for group in opt.param_groups:group['lr']=cfg['lr']
    torch.manual_seed(cfg['rollin_seed']);rng=torch.Generator(device='cuda').manual_seed(cfg['rollin_seed']);schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));history=[];best=(float('inf'),float('inf'));best_step=0;seconds=0.;clipped=0;stop='budget_completed'
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,parent_body_step=3000,parent_head_updates=6000,config=cfg,arm=arm,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rollin_rng=rng.get_state()),folder/name)
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
                start=time.monotonic();f,o,p,m,t=pool.select(batch);draws=torch.rand(len(batch),f.shape[1]-1,device='cuda',generator=rng);prob=rollin_probability(step,cfg['rollin_max_probability'],cfg['rollin_ramp_steps']) if arm=='rollin_anchor' else 0.;selected=transition_mask(draws,m,prob)
                pred,pen,tr=history_forward(model,f,t,writer_ids(batch,data,cfg,'cuda'),selected,selected);terms=reconstruction_terms(pred,pen,o,p,m,pred.new_tensor(cfg['pen_weights']));anchor=cumulative_xy_loss(pred,o,m,cfg['offset_stats']);aw=cfg['anchor_calibration']['weight'] if arm!='teacher' else 0.;loss=terms['offset']+cfg['pen_weight']*terms['pen']+aw*anchor;opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                eligible=int(m[:,1:].any(-1).sum());chosen=int(selected.sum());row=dict(step=step,parent_body_step=3000,parent_head_updates=6000,sample_ids=batch,loss=float(loss.detach()),cumulative_xy_mse=float(anchor.detach()),anchor_weight=aw,offset_mse=float(terms['offset'].detach()),pen_loss=float(terms['pen'].detach()),pen_weight=cfg['pen_weight'],lr=cfg['lr'],raw_grad_norm=norm,clipped=norm>cfg['clip'],rollin_probability=prob,eligible_transitions=eligible,selected_transitions=chosen,selected_fraction=chosen/eligible if eligible else 0.,draws_sha256=hashlib.sha256(draws.cpu().numpy().tobytes()).hexdigest(),selected_sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest(),mean_advance=float(tr['advance'].detach().mean()),mean_sigma=float(tr['sigma'].detach().mean()))
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
    result=dict(arm=arm,last_step=step,parent_body_step=3000,parent_head_updates=6000,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,initial_optimizer_tensor_digest=optimizer_initial,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),reader_unchanged=True,not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
