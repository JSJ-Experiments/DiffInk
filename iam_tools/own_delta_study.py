"""Matched own-history displacement loss control, isolated v2 outputs, no new text gate."""
import hashlib,json,math,tarfile,time
from pathlib import Path
import h5py,numpy as np,torch
from .own_delta import paired_delta_losses,delta_coefficients
from .generated_prefix_study import evaluate
from .generation_geometry import gradient_norm
from .history_rollin_study import teacher_gate
from .point_feedback_strokes import PointFeedbackStrokeWriter
from .autoregressive_strokes import StrokePool,decode_offsets
from .autoregressive_study import writer_ids,aggregate_teacher
from .point_feedback_study import evaluate as evaluate_capacity
from .writer_expansion import training_schedule
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .generation_study import read_sequence
from .inkvae import edit_distance
from .pen_ab import file_sha,boundary_metrics
from .ocr_joint_study import local_geometry
from .ocr_context_study import tensor_digest
from .resource_monitor import ResourceMonitor
from .launch_ledger import study_path
PARENT='checkpoints/iam_continuous_prefix/20261008-183348-v2-recovery'
PARENT_SHA='ce63620dabecdf80d192ac1e73d38ee823ea1bb4d39c7323729c48dcdff00f0a'
ARMS=('control','ink_delta','all_delta')

def checked_path(root,relative):return study_path(root,relative,'checkpoints/iam_own_delta/')

def optimizer_digest(state):return tensor_digest({str(k)+':'+str(n):v for k,s in state['state'].items() for n,v in s.items() if torch.is_tensor(v)})

def arm_weight(arm,cfg):
    if arm not in ARMS:raise ValueError('declared matched own-displacement arm required')
    c=cfg['delta_calibration']['weights']
    return 0. if arm=='control' else c['ink' if arm=='ink_delta' else 'all']


def validate(cfg,data):
    if cfg['parent_checkpoint_sha256']!=PARENT_SHA or cfg['source_h5_sha256']!=SOURCE_H5_SHA or cfg['reader_sha256']!=READER_SHA:raise ValueError('pinned selected continuous1000 source/reader required')
    if cfg['max_updates']!=2000 or cfg['batch']!=8 or cfg['max_train_wall_seconds']!=1800 or cfg['lr']!=1e-5 or cfg['schedule_seed']!=65142 or cfg['training_seed']!=65143:raise ValueError('bounded matched displacement protocol required')
    if set(cfg['models'])!=set(ARMS) or any(m!=cfg['models']['control'] or m['point_feedback'] for m in cfg['models'].values()):raise ValueError('same source no-intrapoint-feedback models required')
    ids=data['training_ids']
    if len(set(ids))!=8 or data['fixed_train_ids']!=ids or data['exposed_dev_ids'] or set(data['records'])!=set(ids):raise ValueError('exact8 established TRAIN no new confirmation source required')
    c=cfg['delta_calibration']
    if not c['state_rng_unchanged'] or c['fraction']!=.25 or c['weights']!=delta_coefficients(c['complete_norm'],c['ink_norm'],c['all_norm']):raise ValueError('fixed modest complete-parent-gradient calibration required')
    if cfg['eval_steps']!=[0,250,500,1000,2000] or cfg['control_steps']!=[0,2000] or cfg['free_max_blocks']!=256 or cfg['parent_joint_step']!=1000 or cfg['parent_checkpoint_relative']!=PARENT+'/continuous_xy/checkpoint-best.pt' or cfg['output_volume']!='diffink-experiments-v2' or not cfg['source_volume_input_only'] or not all(cfg[k] for k in ('no_kl','no_style','no_new_confirmation')):raise ValueError('fixed evaluation/free cap/source/output and no incidental losses required')
    if not cfg['optimizer_restored'] or cfg['pen_weight']!=.024860149190817294 or cfg['teacher_anchor_weight']!=.009549097811244269 or cfg['own_xy_weight']!=5.006895593114276e-5 or cfg['own_pen_weight']!=3.159880562284457e-6 or cfg['weight_decay']!=0. or cfg['clip']!=5 or not cfg['no_ocr_training_loss'] or not cfg['not_promoted']:raise ValueError('same full parent objective/moments, evaluator-only and no promotion required')


def prepare(repo,root='data'):
    root=Path(root);parent=root/PARENT;checkpoint=parent/'continuous_xy/checkpoint-best.pt'
    for f,sha in [(checkpoint,PARENT_SHA),(root/DATA/'source.h5',SOURCE_H5_SHA),(root/DATA/'dataset.json',DATASET_SHA),(root/READER_REL,READER_SHA)]:
        if file_sha(f)!=sha:raise ValueError('immutable selected source drift: '+str(f))
    old=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());pr=json.loads((parent/'continuous_xy/result.json').read_text());baseline=json.loads((parent/'continuous_xy/eval-1000.json').read_text());saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['step']!=1000 or saved['arm']!='continuous_xy' or pr['best_step']!=1000 or not pr['history'][-1]['capacity_gate']['passed']:raise ValueError('selected preserved-capacity continuous1000 source required')
    cfg={k:old[k] for k in ['vocab','writers','offset_stats','pen_weights','reader_relative','reader_sha256','source_data','source_h5_sha256','source_manifest_sha256','betas','clip','torch_cpu_threads']}
    cfg.update(parent=PARENT,parent_checkpoint_relative=PARENT+'/continuous_xy/checkpoint-best.pt',parent_checkpoint_sha256=PARENT_SHA,parent_joint_step=1000,models={a:old['models']['continuous_xy'] for a in ARMS},profile='standalone8TRAIN own-history target-displacement control; NOT releasedInkDiT/newtext composition',schedule_seed=65142,training_seed=65143,max_updates=2000,max_train_wall_seconds=1800,batch=8,lr=1e-5,weight_decay=0.,optimizer_restored=True,pen_weight=.024860149190817294,teacher_anchor_weight=.009549097811244269,own_xy_weight=5.006895593114276e-5,own_pen_weight=3.159880562284457e-6,eval_steps=[0,250,500,1000,2000],control_steps=[0,2000],free_max_blocks=256,
        loss='same parent teacher displacement/focalpen/cumulativeXY and full-own cumulativeXY/focalpen + small own target-displacement MSE; no generic smoothing',
        training='same teacher/full-continuous-XY own-history graphs; hardpens argmax nondifferentiable; source extent ONLY supervised TRAIN unroll/mask, own path continues after predicted EOC, NOT real free generation',
        caveats='ink_delta uses GT previous continue and both real endpoints across block boundaries; excludes origin/penup jumps; all_delta includes origin and penup jumps. Normalized chronological index offsets, not physical velocity. Source-index targets may be ambiguous after a diverging path; one seed8TRAIN/7writers is capacity, not composition proof',
        selection='minimum actual freeTRAIN8 CER then teacher-offsetMSE among same predeclared perline preserved teacher-capacity guards; baseline eligible; final AND selected retained; no held/visual reselection',
        teacher_capacity_gate='same perline source-relative geometry/pen safeguards + teacherTRUE-PEN readerCER0',
        no_kl=True,no_style=True,no_ocr_training_loss=True,no_new_confirmation=True,not_promoted=True,output_volume='diffink-experiments-v2',source_volume_input_only=True)
    torch.set_num_threads(2);model=PointFeedbackStrokeWriter(**cfg['models']['control']);model.load_state_dict(saved['model_state_dict']);model.train();digest=tensor_digest(model.state_dict());rng=torch.get_rng_state().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
    if not opt.state:raise ValueError('original full-model AdamW tensors required')
    cfg['parent_initial_optimizer_digest']=optimizer_digest(opt.state_dict())
    with h5py.File(root/DATA/'source.h5') as hf:targets={sid:torch.tensor(hf[sid]['target'][:]) for sid in data['training_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats']);batch=list(training_schedule(data['training_ids'],1,8,cfg['schedule_seed']))[0];f,o,p,m,t=pool.select(batch);terms=paired_delta_losses(model,f,t,writer_ids(batch,data,cfg,'cpu'),o,p,m,cfg);params=list(model.parameters());bn=gradient_norm(terms['complete'],params);ink=gradient_norm(terms['own_ink_delta'],params);all_=gradient_norm(terms['own_all_delta'],params)
    cfg['delta_calibration']=dict(weights=delta_coefficients(bn,ink,all_),fraction=.25,complete_norm=bn,ink_norm=ink,all_norm=all_,sample_ids=batch,losses={k:float(v.detach()) for k,v in terms.items() if isinstance(v,torch.Tensor)},scope='one initial common scheduled all8 TRAIN batch; new delta gradient25% of COMPLETE parent objective; retain parent ownXY/ownpen coefficients unchanged; no perarm updates during calibration',state_rng_unchanged=digest==tensor_digest(model.state_dict()) and torch.equal(rng,torch.get_rng_state()))
    cfg['parent_initial_model_digest']=digest;validate(cfg,data)
    out=root/'checkpoints/iam_own_delta'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            folder=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for file in sorted(folder.rglob('*.py')):
                if '__pycache__' not in file.parts:archive.add(file,arcname=directory+'/'+str(file.relative_to(folder)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz')
    for name,value in [('config.json',cfg),('dataset.json',data),('parent-eval.json',baseline)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return str(out.relative_to(root))


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
    if optimizer_initial!=cfg['parent_initial_optimizer_digest']:raise ValueError('exact same restored AdamW tensor state required')
    for group in opt.param_groups:group['lr']=cfg['lr']
    torch.manual_seed(cfg['training_seed']);schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));history=[];best=(float('inf'),float('inf'));best_step=0;seconds=0.;clipped=0;stop='budget_completed'
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,parent_joint_step=1000,config=cfg,arm=arm,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
    def assess(step,monitor):
        with monitor.in_phase('eval/'+arm):r=evaluate(model,pool,targets,reader,data,cfg,folder,step,parent)
        history.append(dict(step=step,teacher=r['teacher'],free=r['free'],source_rollout=r['source_rollout'],reading=r['reading'],capacity_gate=r['capacity_gate']))
        return (r['free']['correct']['cer'],r['teacher']['offset_mse']) if r['capacity_gate']['passed'] else None
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        best=assess(0,monitor)
        if best is None:raise ValueError('restored source must pass initial teacher guard')
        save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        if on_checkpoint:
            with monitor.in_phase('persist/'+arm):on_checkpoint()
        model.train();monitor.set_phase('train/'+arm)
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic();f,o,p,m,t=pool.select(batch);terms=paired_delta_losses(model,f,t,writer_ids(batch,data,cfg,'cuda'),o,p,m,cfg);wd=arm_weight(arm,cfg);delta=terms['own_ink_delta'] if arm=='ink_delta' else terms['own_all_delta']
                loss=terms['complete']+wd*delta;opt.zero_grad(set_to_none=True)
                gradient_check={}
                if step==1 or step%250==0:
                    bn=gradient_norm(terms['complete'],list(model.parameters()));dn=gradient_norm(delta,list(model.parameters()))
                    gradient_check=dict(complete_norm=bn,delta_norm=dn,weighted_delta_ratio=wd*dn/bn if bn else None)
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,continuous_feedback_gradient=True,parent_joint_step=1000,sample_ids=batch,loss=float(loss.detach()),base_loss=float(terms['base'].detach()),complete_loss=float(terms['complete'].detach()),own_ink_delta=float(terms['own_ink_delta'].detach()),own_all_delta=float(terms['own_all_delta'].detach()),delta_weight=wd,teacher_offset=float(terms['teacher_offset'].detach()),teacher_pen=float(terms['teacher_pen'].detach()),teacher_xy=float(terms['teacher_xy'].detach()),own_xy=float(terms['own_xy'].detach()),own_pen=float(terms['own_pen'].detach()),own_offset_diagnostic=float(terms['own_offset_diagnostic'].detach()),own_xy_weight=cfg['own_xy_weight'],own_pen_weight=cfg['own_pen_weight'],teacher_anchor_weight=cfg['teacher_anchor_weight'],pen_weight=cfg['pen_weight'],lr=cfg['lr'],raw_grad_norm=norm,clipped=norm>cfg['clip'],gradient_check=gradient_check,mean_teacher_advance=float(terms['teacher_trace']['advance'].detach().mean()),mean_own_advance=float(terms['own_trace']['advance'].detach().mean()))
                elapsed=time.monotonic()-start;seconds+=elapsed;clipped+=row['clipped'];monitor.step(elapsed,len(batch));log.write(json.dumps(row)+'\n')
                if not math.isfinite(row['loss']):raise FloatingPointError('nonfinite history continuation objective')
                if step%100==0:log.flush();print(dict(arm=arm,train_seconds=seconds,**row),flush=True)
                limit=seconds>cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step,monitor);save('checkpoint-last.pt',step)
                    if score is not None and score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:
                        log.flush()
                        with monitor.in_phase('persist/'+arm):on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in reader.parameters()):raise ValueError('frozen reader drift')
    result=dict(arm=arm,continuous_feedback_gradient=True,last_step=step,parent_joint_step=1000,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,initial_optimizer_tensor_digest=optimizer_initial,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),reader_unchanged=True,not_promoted=True,parent_checkpoint_sha256=PARENT_SHA)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
