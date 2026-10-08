"""Matched fresh TRAIN-only weak character-attention study, 48000 updates/arm on separate T4s.

No confirmation evaluated here. Exposed development is descriptive only. All
sources remain packed; each GPU writes ONLY its own arm directory on the Volume.
"""
import copy
import hashlib
import json
import tarfile
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from .generation_prefix_contract import PrefixContractWriter, contract_lr
from .generation_timing_study import PARENT
from .generation_capacity import DATA, SOURCE_H5_SHA, DATASET_SHA, WHITENING_SHA, select_capacity
from .generation_composition import fit_duration
from .generation_cache import CachedLatentPool
from .generation_coverage_study import evaluate, score, writer_tensor
from .generation_duration_eval import evaluate_duration
from .generation_geometry import physical_terms
from .latent_diffusion import masked_mse
from .writer_expansion import load, training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor
from .weak_alignment import packed_labels,PackedAlignmentPool,capture_last_alignment,alignment_loss,alignment_summary
from .generation_geometry import gradient_norm

ARMS = ('control', 'weak_alignment')


PROBE = 'checkpoints/iam_generation_alignment_probe/20261008-112944'
PROBE_H5_SHA = '2cc1fc2c9c9c1856f238fe18fbb746ab2c73aa96242091298ce2aa6e625f1205'


def prepare_teacher(root,out,train,records,cfg):
    p=Path(root)/PROBE;summary=json.loads((p/'summary.json').read_text())
    if file_sha(p/'alignment.h5')!=PROBE_H5_SHA or summary['alignment_h5_sha256']!=PROBE_H5_SHA or summary['reader_sha256']!=cfg['reader_sha256'] or summary['source_h5_sha256']!=SOURCE_H5_SHA:
        raise ValueError('immutable TRAIN-only teacher provenance required')
    rows={r['sample_id']:r for r in summary['lines']}
    if set(rows)!=set(train):raise ValueError('teacher scope must equal actual TRAIN, no held paths')
    coverage=[];counts=[]
    with h5py.File(p/'alignment.h5') as src,h5py.File(out/'timing-targets.h5','w') as dest:
        if set(src)!=set(train):raise ValueError('exact TRAIN-only packed teacher paths required')
        for sid in train:
            r=records[sid];row=rows[sid]
            if any(row[k]!=r[k] for k in ('text','writer_id')) or row['frames']!=(r['points']+3)//4:raise ValueError('teacher transcript/point pairing drift')
            frames=src[sid]['forced_token_indices'][:]
            if set(frames[frames>=0])!=set(range(len(r['text']))):raise ValueError('forced transcript path dropped a token')
            labels=packed_labels(frames,len(r['text']),r['points'])
            dest.create_dataset(sid,data=labels,compression='gzip')
            coverage.append(float((labels>=0).any(1).mean()));counts.append(int(((labels[:,0]>=0)&(labels[:,1]>=0)&(labels[:,0]!=labels[:,1])).sum()))
    return dict(probe_relative=PROBE,probe_h5_sha256=PROBE_H5_SHA,probe_summary_sha256=file_sha(p/'summary.json'),
        packed_h5_sha256=file_sha(out/'timing-targets.h5'),ids=list(train),lines=len(train),source_reader_exact=sum(rows[s]['reader_errors']==0 for s in train),
        block_coverage_quantiles=np.quantile(coverage,[0,.5,1]).tolist(),queries_with_two_distinct_characters=sum(counts),
        caveats='Corpus-familiar bidirectional reader forced through TRAIN transcript; weak four-index timing, not exact IAM character borders. Delayed marks, blanks, nonuniform spacing and lookahead remain. No held paths or teacher at inference.')


def load_teacher(out,cfg,train,records):
    if cfg['teacher']['ids']!=train or file_sha(out/'timing-targets.h5')!=cfg['teacher']['packed_h5_sha256']:
        raise ValueError('TRAIN-only packed timing target guard')
    with h5py.File(out/'timing-targets.h5') as f:
        if set(f)!=set(train):raise ValueError('no held target groups allowed')
        labels={s:f[s][:] for s in train}
    for s,a in labels.items():
        if a.shape!=((records[s]['points']+7)//8,2) or (a>=len(records[s]['text'])).any():raise ValueError('timing query/transcript mapping drift')
    return labels


def calibrate(model,pool,aligned,records,cfg,batches,stats):
    # Fresh readout is zero-initialized: body base gradients are ZERO at step0.
    # Both arms therefore receive IDENTICAL1000base-only updates first. Measure
    # on that shared warmup state, without an optimizer/RNG/state mutation.
    digest=tensor_digest(model.state_dict());cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    parameters=[p for n,p in model.named_parameters() if not n.startswith('final.')]
    rows=[]
    for batch in batches:
        clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],clean.device)
        with capture_last_alignment(model) as captured:
            pred=model(torch.zeros_like(clean),torch.ones(len(batch),device=clean.device),text,mask,writer_ids=wi)
        terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');w=cfg['auxiliary_weights']
        base=masked_mse(pred,clean,mask)+w['xy']*terms['xy']+w['first_difference']*terms['first_difference']
        aux=alignment_loss(captured['log_probs'],aligned.select(batch),mask,text)
        bn=gradient_norm(base,parameters);an=gradient_norm(aux,parameters)
        if min(bn,an)<=1e-12 or not np.isfinite([bn,an]).all():raise ValueError('nondegenerate alignment body gradient calibration required')
        rows.append(dict(sample_ids=batch,base_norm=bn,auxiliary_norm=an,base_loss=float(base.detach()),auxiliary_loss=float(aux.detach())))
    coefficient=cfg['alignment_fraction']*float(np.median([r['base_norm'] for r in rows]))/float(np.median([r['auxiliary_norm'] for r in rows]))
    if tensor_digest(model.state_dict())!=digest or not torch.equal(cpu_rng,torch.get_rng_state()) or (gpu_rng is not None and not torch.equal(gpu_rng,torch.cuda.get_rng_state())):raise ValueError('calibration cannot alter model or RNG')
    return dict(step=cfg['calibration_step'],state_sha256=digest,coefficient=coefficient,gradient_fraction=cfg['alignment_fraction'],batches=rows,
        definition='0.10 * median total base-objective body gradient norm / median auxiliary body gradient norm across first8 TRAIN minibatches; exclude final readout, fixed coefficient thereafter; no updates during measurement',state_rng_unchanged=True)


@torch.no_grad()
def assess_alignment(model,pool,aligned,records,cfg,ids):
    was=model.training;model.eval();rows=[]
    try:
        for start in range(0,len(ids),8):
            batch=ids[start:start+8];clean,mask,text,_=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],clean.device)
            with capture_last_alignment(model) as c:model(torch.zeros_like(clean),torch.ones(len(batch),device=clean.device),text,mask,writer_ids=wi)
            labels=aligned.select(batch)
            for j,sid in enumerate(batch):
                rows.append(dict(sample_id=sid,**alignment_summary(c['log_probs'][j:j+1],labels[j:j+1],mask[j:j+1],text[j:j+1])))
    finally:model.train(was)
    total=sum(r['supervised_queries'] for r in rows)
    return dict(lines=rows,aggregate=dict(lines=len(rows),supervised_queries=total,
        cross_entropy=sum(r['cross_entropy']*r['supervised_queries'] for r in rows)/total,
        target_mass=sum(r['target_mass']*r['supervised_queries'] for r in rows)/total,
        mean_line_index_error={k:float(np.mean([r['mean_index_error'][k] for r in rows])) for k in ('median','p90','p99')}),
        caveat='Attention means/forced timing fit do not themselves prove character composition; reserve one-shot new-text confirmation.')


def verify_duration_refit(refit, original, records, ids):
    """BLAS refits differ at ~1e-14; metadata and actual durations remain exact."""
    from .generation_composition import predict_duration
    if set(refit) != set(original):
        raise ValueError('TRAIN-only duration keys differ')
    for k, v in refit.items():
        other = original[k]
        if isinstance(v, list) and v and type(v[0]) in (float, int):
            if len(v) != len(other) or not np.allclose(v, other, rtol=1e-10, atol=1e-10):
                raise ValueError('TRAIN-only numerical duration refit drift: '+k)
        elif v != other:
            raise ValueError('TRAIN-only duration metadata drift: '+k)
    if any(predict_duration(refit, records[s]['text'], records[s]['writer_id']) !=
           predict_duration(original, records[s]['text'], records[s]['writer_id']) for s in ids):
        raise ValueError('actual rounded duration requests differ')


def validate_protocol(cfg,data):
    if cfg['max_updates']!=48000 or not cfg['fresh'] or cfg['parent_step']!=0 or cfg['neural_checkpoint_initialization']:
        raise ValueError('fresh matched bounded48000 protocol required')
    if set(cfg['models'])!=set(ARMS) or set(data['training_ids'])!=set(ARMS) or cfg['models']['control']!=cfg['models']['weak_alignment']:
        raise ValueError('ONLY the auxiliary may differ, not architecture or TRAIN scope')
    if data['training_ids']['control']!=data['training_ids']['weak_alignment'] or len(set(data['training_ids']['control']))!=256:
        raise ValueError('identical256TRAIN required')
    if not cfg['models']['control']['causal_queries'] or cfg['models']['control']['position_policy']!='absolute':
        raise ValueError('causal absolute-query contract required')
    if (cfg['alignment_fraction'],cfg['alignment_start_step'],cfg['calibration_step'],cfg['calibration_batches'])!=(.10,1001,1000,8):
        raise ValueError('fixed small post-warmup gradient-calibrated auxiliary required')
    if cfg['model_seed']!=28142 or cfg['schedule_seeds']!=[29142,39142] or cfg['batch']!=8:
        raise ValueError('matched weights/RNG/order contract required')


def checked_path(relative, root):
    p = Path(relative)
    if not relative.startswith('checkpoints/iam_generation_weak_alignment/') or p.is_absolute() or '..' in p.parts:
        raise ValueError('bounded positional-contract experiment path required')
    return Path(root) / p


def prepare(repo, root='/data'):
    root = Path(root); source = root / DATA
    for name, sha in [('source.h5', SOURCE_H5_SHA), ('dataset.json', DATASET_SHA), ('whitening.pt', WHITENING_SHA)]:
        if file_sha(source/name) != sha:
            raise ValueError('immutable coverage source drift: '+name)
    original = json.loads((source/'dataset.json').read_text())
    old = json.loads((root/PARENT/'config.json').read_text())
    scope = select_capacity(original)
    train = scope['training_ids']['larger256']
    scope['training_ids'] = {a:list(train) for a in ARMS}
    ids = sorted(set(scope['splits']['all_train256'] + scope['splits']['unseen_prompt']))
    records = {sid: original['records'][sid] for sid in ids}
    writers = original['writers']
    duration = fit_duration(records, train, writers)
    verify_duration_refit(duration, old['duration_model'], records, ids)
    duration = copy.deepcopy(old['duration_model'])  # Retain pinned reference bytes after independent refit.
    from .generation_alignment_study import REFERENCE, REFERENCE_SHA
    if file_sha(root/REFERENCE) != REFERENCE_SHA:
        raise ValueError('fixed config/anchor reference required, not neural initialization')
    reference = torch.load(root/REFERENCE, map_location='cpu', weights_only=False)['config']
    spec = dict(reference['model'], alignment=True, blocks_per_character=duration['blocks_per_character'], prior_sigma=2.5, prior_cap=12.)
    out = root/'checkpoints/iam_generation_weak_alignment'/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz', 'w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base = Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for p in sorted(base.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'):
                    tar.add(p, arcname=directory+'/'+str(p.relative_to(base)))
    cfg = dict(profile='fresh causal control vs TRAIN-only weak final-local-head character alignment; standalone mapper NOT semantic InkVAE/InkDiT/TrInk',
        fresh=True,parent_step=0,neural_checkpoint_initialization=False,
        data_parent=DATA,source_h5_sha256=SOURCE_H5_SHA,data_manifest_sha256=DATASET_SHA,whitening_sha256=WHITENING_SHA,
        reference_config_only=REFERENCE,reference_sha256=REFERENCE_SHA,
        vocab=old['vocab'],writers=writers,models={a:dict(spec,position_policy='absolute',causal_queries=True) for a in ARMS},
        source_rel=old['source_rel'],source_sha256=old['source_sha256'],reader_rel=old['reader_rel'],reader_sha256=old['reader_sha256'],
        batch=8,betas=[.9,.99],weight_decay=.01,clip=1.,max_updates=48000,
        eval_steps=[0,8000,16000,24000,36000,48000],max_train_wall_seconds_per_arm=5400,
        model_seed=28142,schedule_seeds=[29142,39142],lr_schedule='same fresh24000 warmup/hold/cosine then24000 constant1e-5, full Adam retained',
        auxiliary_weights=reference['auxiliary_weights'],duration_model=duration,
        source_archive_sha256=file_sha(out/'as-run-source.tar.gz'),selection='all256 native TRAIN geometry only',
        confirmation_policy='Fresh metadata-only16paired/16synthetic sealed at preparation, excludes all THREE exposed confirmations; do not open sources/score until both frozen TRAIN-selected candidates satisfy native5%CER fit gate',
        hypothesis='only small TRAIN attention auxiliary changes after identical1000updates; same causal absolute-query architecture, fresh weights/RNG, full-text Gaussian cross-attention, targets/order and base objective',
        limitations='noncausal hidden queries see requested-window context; causal latent prefixes must be budget-independent to FP32 tolerance, codec drift independently measured; native TRAIN masks still supply oracle supervision, no teacher-forced stroke autoregression; corpus-familiar reader; one seed; only256TRAIN; no generic smoothing or target resampling',
        torch_cpu_threads=2,not_promoted=True,
        alignment_fraction=.10,alignment_start_step=1001,calibration_step=1000,calibration_batches=8,
        alignment_policy='final three local heads, equal valid-query CE against actual forced nonblank emissions; fourth global head free; blanks omitted, spaces included, two-frame targets averaged; no interpolation or exact IAM boundaries',
        auxiliary_kernel_policy='separate differentiable QK/logsoftmax capture, unchanged need_weights=False main forward in BOTH arms; inference unchanged')
    validate_protocol(cfg,dict(records=records,writers=writers,**scope))
    teacher=prepare_teacher(root,out,train,records,cfg)
    cfg['teacher']=teacher
    from .generation_weak_confirmation import reserve
    seal=reserve(root,cfg,dict(records=records,writers=writers,**scope))
    reserved=out/'reserved-confirmation';reserved.mkdir()
    (reserved/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    cfg['reserved_confirmation_sha256']=file_sha(reserved/'seal.json')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'dataset.json').write_text(json.dumps(dict(records=records,writers=writers,**scope),indent=2)+'\n')
    return str(out.relative_to(root))


def run(arm, relative, repo, root='/data'):
    if arm not in ARMS or not torch.cuda.is_available():
        raise ValueError('explicit matched positional arm and T4 required')
    root=Path(root);out=checked_path(relative,root);cfg=json.loads((out/'config.json').read_text());data=json.loads((out/'dataset.json').read_text())
    validate_protocol(cfg,data)
    if file_sha(out/'reserved-confirmation/seal.json')!=cfg['reserved_confirmation_sha256']:
        raise ValueError('pre-training fresh seal guard')
    if file_sha(out/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:
        raise ValueError('immutable as-run source required')
    torch.set_num_threads(cfg['torch_cpu_threads'])
    labels=load_teacher(out,cfg,data['training_ids'][arm],data['records'])
    aligned=PackedAlignmentPool(labels,data['training_ids'][arm],device='cuda')
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cc,'cuda')
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    latents={};targets={};records=data['records'];source=root/DATA
    if file_sha(source/'source.h5')!=SOURCE_H5_SHA or file_sha(source/'whitening.pt')!=WHITENING_SHA:
        raise ValueError('frozen data drift')
    with h5py.File(source/'source.h5') as f:
        for sid in records:
            latents[sid]=torch.tensor(f[sid]['latent_mean'][:],device='cuda');targets[sid]=f[sid]['target'][:]
    stats=torch.load(source/'whitening.pt',weights_only=True)
    stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    pool=CachedLatentPool(latents,records,cfg['vocab'],sorted(records),stats,max_lines=264)
    torch.manual_seed(cfg['model_seed']);model=PrefixContractWriter(**cfg['models'][arm]).cuda()
    torch.manual_seed(cfg['model_seed'])  # Same post-prototype RNG as previous fresh study.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
    initial=tensor_digest(model.state_dict());folder=out/arm;folder.mkdir(exist_ok=False)
    train=data['training_ids'][arm]
    schedule=[batch for seed in cfg['schedule_seeds'] for batch in training_schedule(train,24000,cfg['batch'],seed=seed)]
    history=[];durations=[];alignment_history=[];alignment_weight=0.;calibration=None;seconds=0.;clipped=0;best_step=0;stop='budget_completed'
    with ResourceMonitor(folder,interval=5,sustained_seconds=30) as monitor:
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=arm,model=cfg['models'][arm]),step=step,
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),alignment_weight=alignment_weight,calibration=calibration),folder/name)
        def assess(step):
            with monitor.in_phase('eval/'+arm):
                ev=evaluate(model,codec,reader,pool,latents,records,cfg['vocab'],stats,data['splits'],targets,folder,step,cfg['writers'],controls=step==48000)
                de=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['all_train256'],folder,step,cfg['writers'],cfg['duration_model'],controls=step==48000)
                dev=folder/'exposed-development-duration';dev.mkdir(exist_ok=True)
                dd=evaluate_duration(model,codec,reader,records,cfg['vocab'],stats,data['splits']['unseen_prompt'],dev,step,cfg['writers'],cfg['duration_model'])
                alignment=assess_alignment(model,pool,aligned,records,cfg,train)
                alignment_history.append(dict(step=step,**alignment))
                (folder/f'alignment-eval-{step}.json').write_text(json.dumps(alignment_history[-1],indent=2)+'\n')
                s=score(ev,set(train));history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']))
                durations.append(dict(step=step,train=de['aggregate'],exposed_development=dd['aggregate']))
            return s
        best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        monitor.set_phase('train/'+arm);model.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(schedule,1):
                start=time.monotonic()
                for g in opt.param_groups:g['lr']=contract_lr(step)
                clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],'cuda')
                with capture_last_alignment(model) as captured:
                    pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi)
                auxiliary=alignment_loss(captured['log_probs'],aligned.select(batch),mask,text,validate=step==1)
                terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask);w=cfg['auxiliary_weights']
                base_loss=mse+w['xy']*terms['xy']+w['first_difference']*terms['first_difference']
                loss=base_loss+alignment_weight*auxiliary if alignment_weight else base_loss
                opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                row=dict(step=step,sample_ids=batch,alignment_cross_entropy=float(auxiliary.detach()),alignment_weight=alignment_weight,learning_rates=[g['lr'] for g in opt.param_groups],loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'],text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()))
                log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped']
                if step==cfg['calibration_step']:
                    calibration=calibrate(model,pool,aligned,records,cfg,schedule[:cfg['calibration_batches']],stats)
                    (folder/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
                    alignment_weight=calibration['coefficient'] if arm=='weak_alignment' else 0.
                    save('checkpoint-pre-intervention.pt',step)
                    print(dict(arm=arm,calibration=calibration),flush=True)
                if step%2000==0:
                    print(dict(arm=arm,**row,train_seconds=seconds),flush=True);log.flush()
                if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite positional study')
                limit=seconds>cfg['max_train_wall_seconds_per_arm']
                if step in cfg['eval_steps'] or limit:
                    s=assess(step);save('checkpoint-last.pt',step)
                    if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):
            raise ValueError('frozen codec/reader drift')
    result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,duration_history=durations,stop=stop,train_seconds=seconds,
        alignment_history=alignment_history,calibration=calibration,alignment_weight=alignment_weight,
        clip_fraction=clipped/step,initial_state_sha256=initial,codec_reader_unchanged=True,
        last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),
        schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),not_promoted=True)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(output=str(folder),arm=arm,train=result['history'][-1]['aggregate']['all_train256']['correct'],duration=result['duration_history'][-1]['train'])
