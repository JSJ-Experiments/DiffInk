"""Compare missing versus retained continuous-feedback Jacobian on identical paths.

Baseline preserves the existing teacher objective. New arms add small fixed
initial-gradient calibrated losses on full own-history rollouts. Teacher source
length supplies supervised unroll/loss masks ONLY; actual free evaluation remains
text/writer only, common cap and learned firstEOC. No posthoc stop gate or OCR loss.
"""
import hashlib,json,math,tarfile,time
from pathlib import Path
import h5py,numpy as np,torch
from .generated_prefix import auxiliary_coefficients
from .continuous_prefix import paired_feedback_losses
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
PARENT='checkpoints/iam_cumulative_xy/20261008-171928'
PARENT_SHA='34474c6b84ae4d9865fef7737ac84af2be99faa0f7f3c4b6d7b27a1ad046bdb9'
ARMS=('detached_xy','continuous_xy')

def checked_path(root,relative):return study_path(root,relative,'checkpoints/iam_continuous_prefix/')

def optimizer_digest(state):return tensor_digest({str(k)+':'+str(n):v for k,s in state['state'].items() for n,v in s.items() if torch.is_tensor(v)})

def arm_weights(arm,cfg):
    if arm not in ARMS:raise ValueError('declared three-arm own-prefix control required')
    c=cfg['auxiliary_calibration']['weights']
    return c['xy'],c['pen'] # SAME scalar loss botharms; only continuous XY derivative differs

def validate(cfg,data):
    if cfg['parent_checkpoint_sha256']!=PARENT_SHA or cfg['source_h5_sha256']!=SOURCE_H5_SHA or cfg['reader_sha256']!=READER_SHA:raise ValueError('pinned previous selected1000 source required')
    if cfg['max_updates']!=1000 or cfg['batch']!=8 or cfg['max_train_wall_seconds']!=1800 or cfg['lr']!=1e-5 or cfg['schedule_seed']!=64142 or cfg['training_seed']!=64143:raise ValueError('bounded matched protocol required')
    if set(cfg['models'])!=set(ARMS) or any(m!=cfg['models']['detached_xy'] or m['point_feedback'] for m in cfg['models'].values()):raise ValueError('identical source no-point-feedback models required')
    ids=data['training_ids']
    if len(set(ids))!=8 or data['fixed_train_ids']!=ids or data['exposed_dev_ids'] or set(data['records'])!=set(ids):raise ValueError('exact8 establishedTRAIN, no new DEV/confirmation')
    c=cfg['auxiliary_calibration']
    if not c['state_rng_unchanged'] or not c['reference_continuous_feedback'] or c['fractions']!=dict(xy=.25,pen=.10) or c['weights']!=auxiliary_coefficients(c['base_norm'],c['own_xy_norm'],c['own_pen_norm']):raise ValueError('fixed modest gradient-calibrated generated-prefix coefficients required')
    if not cfg['optimizer_restored'] or cfg['pen_weight']!=.024860149190817294 or cfg['teacher_anchor_weight']!=.009549097811244269 or not cfg['no_ocr_training_loss'] or not cfg['not_promoted']:raise ValueError('restored optimizer/fixed teacher objective/evaluator-only/no promotion contract')

def prepare(repo,root='/data'):
    root=Path(root);parent=root/PARENT;checkpoint=parent/'teacher_anchor/checkpoint-best.pt'
    for f,sha in [(checkpoint,PARENT_SHA),(root/DATA/'source.h5',SOURCE_H5_SHA),(root/DATA/'dataset.json',DATASET_SHA),(root/READER_REL,READER_SHA)]:
        if file_sha(f)!=sha:raise ValueError('immutable source drift: '+str(f))
    old=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());pr=json.loads((parent/'teacher_anchor/result.json').read_text());baseline=json.loads((parent/'teacher_anchor/eval-1000.json').read_text());saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['step']!=1000 or saved['arm']!='teacher_anchor' or pr['best_step']!=1000 or not pr['history'][-1]['capacity_gate']['passed']:raise ValueError('selected preserved-capacity1000 source required')
    cfg={k:old[k] for k in ['vocab','writers','offset_stats','pen_weights','reader_relative','reader_sha256','source_data','source_h5_sha256','source_manifest_sha256','betas','clip','torch_cpu_threads']}
    cfg.update(parent=PARENT,parent_checkpoint_relative=PARENT+'/teacher_anchor/checkpoint-best.pt',parent_checkpoint_sha256=PARENT_SHA,parent_body_step=4000,parent_head_updates=6000,models={a:old['models']['teacher_anchor'] for a in ARMS},profile='standalone stroke-writer8TRAIN continuous XY feedback gradient control; NOT EnglishInkDiT or newtext composition',schedule_seed=64142,training_seed=64143,max_updates=1000,max_train_wall_seconds=1800,batch=8,lr=1e-5,weight_decay=0.,optimizer_restored=True,pen_weight=.024860149190817294,teacher_anchor_weight=old['anchor_calibration']['weight'],eval_steps=[0,250,500,1000],control_steps=[0,1000],free_max_blocks=256,
        loss='SAME teacher displacement/focalpen/cumulativeXY + own cumulativeXY + own focalpen scalar loss botharms; shared coefficients calibrated25%/10% initial full-model gradients on CONTINUOUS-feedback surrogate, not eacharm separately',
        training='same paired truehistory/ownhistory graphs; noGT trajectory fields in own path, source extent ONLY supervised training unroll/mask; both continue after predictedEOC; detached_xy stops continuous feedback gradients; continuous_xy retains them; BOTH hardpens argmax nondifferentiable, normal hidden BPTT; NO forward/model/inference difference',
        caveats='paired source-index trajectory capacity objective, not unbiased handwriting likelihood; fullown continuation after predictedEOC uses source training length and is NOT free generation; eight familiarTRAIN across7writers one seed corpus-familiar reader; not composition proof',
        selection='minimum genuine freeTRAIN8 CER then teacheroffsetMSE among perline preserved-capacity checks; baselineeligible; no DEV/newtext or visual selection',
        teacher_capacity_gate='same perline source-relative geometry/pen safeguards and teacherTRUE-PEN readerCER0',
        no_kl=True,no_style=True,no_ocr_training_loss=True,no_new_confirmation=True,not_promoted=True)
    torch.set_num_threads(2);model=PointFeedbackStrokeWriter(**cfg['models']['detached_xy']);model.load_state_dict(saved['model_state_dict']);model.train();digest=tensor_digest(model.state_dict());rng=torch.get_rng_state().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);opt.load_state_dict(saved['optimizer_state_dict'])
    if not opt.state:raise ValueError('nonempty original whole-model AdamW state required')
    cfg['parent_initial_optimizer_digest']=optimizer_digest(opt.state_dict())
    with h5py.File(root/DATA/'source.h5') as hf:targets={s:torch.tensor(hf[s]['target'][:]) for s in data['training_ids']}
    pool=StrokePool(targets,data['records'],cfg['vocab'],data['training_ids'],cfg['offset_stats']);batch=list(training_schedule(data['training_ids'],1,8,cfg['schedule_seed']))[0];f,o,p,m,t=pool.select(batch);terms=paired_feedback_losses(model,f,t,writer_ids(batch,data,cfg,'cpu'),o,p,m,cfg,True);parameters=list(model.parameters());bn=gradient_norm(terms['base'],parameters);xn=gradient_norm(terms['own_xy'],parameters);pn=gradient_norm(terms['own_pen'],parameters)
    cfg['auxiliary_calibration']=dict(weights=auxiliary_coefficients(bn,xn,pn),fractions=dict(xy=.25,pen=.10),base_norm=bn,own_xy_norm=xn,own_pen_norm=pn,sample_ids=batch,losses={k:float(v.detach()) for k,v in terms.items() if isinstance(v,torch.Tensor)},scope='one initial fullTRAIN8 CONTINUOUS-feedback surrogate, full-model gradient norms; SAME shared coefficients botharms; NO perarm recalibration',reference_continuous_feedback=True,state_rng_unchanged=digest==tensor_digest(model.state_dict()) and torch.equal(rng,torch.get_rng_state()))
    cfg['parent_initial_model_digest']=digest;validate(cfg,data)
    out=root/'checkpoints/iam_continuous_prefix'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for file in sorted(base.rglob('*.py')):
                if '__pycache__' not in file.parts:archive.add(file,arcname=directory+'/'+str(file.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz')
    for name,value in [('config.json',cfg),('dataset.json',data),('parent-eval.json',baseline)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return str(out.relative_to(root))


def run(arm,relative,root='/data',on_checkpoint=None,resume_source=None):
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
    def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,parent_body_step=4000,parent_head_updates=6000,config=cfg,arm=arm,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
    def assess(step,monitor):
        with monitor.in_phase('eval/'+arm):r=evaluate(model,pool,targets,reader,data,cfg,folder,step,parent)
        history.append(dict(step=step,teacher=r['teacher'],free=r['free'],source_rollout=r['source_rollout'],reading=r['reading'],capacity_gate=r['capacity_gate']))
        return (r['free']['correct']['cer'],r['teacher']['offset_mse']) if r['capacity_gate']['passed'] else None
    resume_step=0
    if resume_source is not None:
        from .continuous_recovery import snapshot_info,validate_checkpoint,copy_snapshot
        info=snapshot_info(resume_source,cfg,data,arm,parent)
        recovered=torch.load(Path(resume_source)/'checkpoint-last.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(recovered,cfg,arm,info['step'])
        model.load_state_dict(recovered['model_state_dict']);opt.load_state_dict(recovered['optimizer_state_dict'])
        for group in opt.param_groups:group['lr']=cfg['lr']
        torch.set_rng_state(recovered['rng_cpu']);torch.cuda.set_rng_state_all(recovered['rng_cuda'])
        copy_snapshot(resume_source,folder);resume_step=info['step'];history=info['history'];best=info['best'];best_step=info['best_step'];seconds=info['seconds'];clipped=info['clipped']
        print(dict(arm=arm,resumed_exact_step=resume_step,remaining_updates=cfg['max_updates']-resume_step,restored_model_optimizer_rng=True),flush=True)
    elif 'recovery' in cfg:raise ValueError('recovery study must resume, never restart from parent')
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        if not resume_step:
            best=assess(0,monitor)
            if best is None:raise ValueError('restored initial capacity gate must PASS')
            save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
            if on_checkpoint:
                with monitor.in_phase('persist/'+arm):on_checkpoint()
        model.train();monitor.set_phase('train/'+arm)
        with (folder/'metrics.jsonl').open('a' if resume_step else 'w') as log:
            for step,batch in enumerate(schedule[resume_step:],resume_step+1):
                start=time.monotonic();f,o,p,m,t=pool.select(batch);terms=paired_feedback_losses(model,f,t,writer_ids(batch,data,cfg,'cuda'),o,p,m,cfg,arm=='continuous_xy');wx,wp=arm_weights(arm,cfg)
                loss=terms['base']+wx*terms['own_xy']+wp*terms['own_pen'];opt.zero_grad(set_to_none=True)
                gradient_check={}
                if step==1 or step%250==0:
                    bn=gradient_norm(terms['base'],list(model.parameters()));xn=gradient_norm(terms['own_xy'],list(model.parameters()));pn=gradient_norm(terms['own_pen'],list(model.parameters()))
                    gradient_check=dict(base_norm=bn,own_xy_norm=xn,own_pen_norm=pn,weighted_xy_ratio=wx*xn/bn if bn else None,weighted_pen_ratio=wp*pn/bn if bn else None)
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,continuous_feedback_gradient=arm=='continuous_xy',parent_body_step=4000,parent_head_updates=6000,sample_ids=batch,loss=float(loss.detach()),base_loss=float(terms['base'].detach()),teacher_offset=float(terms['teacher_offset'].detach()),teacher_pen=float(terms['teacher_pen'].detach()),teacher_xy=float(terms['teacher_xy'].detach()),own_xy=float(terms['own_xy'].detach()),own_pen=float(terms['own_pen'].detach()),own_offset_diagnostic=float(terms['own_offset_diagnostic'].detach()),own_xy_weight=wx,own_pen_weight=wp,teacher_anchor_weight=cfg['teacher_anchor_weight'],pen_weight=cfg['pen_weight'],lr=cfg['lr'],raw_grad_norm=norm,clipped=norm>cfg['clip'],gradient_check=gradient_check,mean_teacher_advance=float(terms['teacher_trace']['advance'].detach().mean()),mean_own_advance=float(terms['own_trace']['advance'].detach().mean()))
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
    result=dict(arm=arm,continuous_feedback_gradient=arm=='continuous_xy',last_step=step,parent_body_step=4000,parent_head_updates=6000,best_step=best_step,best_train_score=list(best),history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_state_sha256=initial,initial_optimizer_tensor_digest=optimizer_initial,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),reader_unchanged=True,not_promoted=True,resume_step=resume_step,recovery=cfg.get('recovery'))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return dict(output=str(folder),arm=arm,best_step=best_step,final=history[-1])
