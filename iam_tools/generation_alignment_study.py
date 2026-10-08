"""Fresh matched global versus soft-Gaussian alignment composition study."""
import copy, hashlib, json, tarfile, time
from pathlib import Path
import h5py, numpy as np, torch
from .generation_capacity import (DATA, SOURCE_H5_SHA, DATASET_SHA, WHITENING_SHA,
                                  select_capacity, model_config, validate_budget)
from .generation_alignment import AlignedWriterDenoiser
from .generation_composition import fit_duration,seal_confirmation,seal_synthetic,learning_rate
from .generation_duration_eval import evaluate_duration
from .generation_coverage_study import evaluate, score, writer_tensor
from .generation_geometry import physical_terms
from .generation_cache import CachedLatentPool
from .latent_diffusion import masked_mse
from .writer_expansion import load, training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .generation_study import read_sequence
from .inkvae import edit_distance
from .pen_ab import file_sha
from .resource_monitor import ResourceMonitor


REFERENCE='checkpoints/iam_generation_lr_polish/20261008-070802/polish/checkpoint-last.pt'
REFERENCE_SHA='c4ffb8fb93c0008ef88645eeb3ba41ad45c13b728e342c0a39b7f7a141065890'
ARMS=('global','soft_gaussian')


def run(config, repo, root='/data', steps=24000):
    if type(steps) is not int or not 16000<=steps<=32000:raise ValueError('bounded16000–32000 fresh matched updates required')
    if not torch.cuda.is_available(): raise ValueError('T4 required')
    torch.set_num_threads(4); root = Path(root); source = root / DATA
    for name, sha in [('source.h5', SOURCE_H5_SHA), ('dataset.json', DATASET_SHA), ('whitening.pt', WHITENING_SHA)]:
        if file_sha(source/name) != sha: raise ValueError('immutable coverage source drift: '+name)
    data = json.loads((source/'dataset.json').read_text()); parent_cfg = json.loads((source/'config.json').read_text())
    if file_sha(root/REFERENCE)!=REFERENCE_SHA:raise ValueError('pinned reference config/anchors required; NO neural weights reused')
    reference=torch.load(root/REFERENCE,map_location='cpu',weights_only=False)
    scope = select_capacity(data); scope['training_ids']={a:scope['training_ids']['larger256'] for a in ARMS}; ids = sorted(set(scope['splits']['all_train256'] + scope['splits']['unseen_prompt']))
    records = {i:data['records'][i] for i in ids}; writers = data['writers']; vocab = parent_cfg['vocab']
    # Check reserved writers against clean pool, not just an inherited Boolean flag.
    from .ocr_pool_study import load_pool
    from .ocr_convergence import POOL_SHA
    _, manifest, pvocab = load_pool(root, POOL_SHA)
    blocked = set(manifest['test_writers']) | set(manifest['dev_writers'])
    if pvocab != vocab or any(records[i]['writer_id'] in blocked or records[i] != manifest['records'][i] for i in ids):
        raise ValueError('reserved writer/clean record drift')
    codec,_,_,cc,cvocab,_ = load(config, repo, root, parent_cfg['source_rel'], parent_cfg['source_sha256'], writer_id=None)
    if cvocab != vocab: raise ValueError('codec alphabet drift')
    codec = codec.cuda().eval().requires_grad_(False); reader,_ = load_reader(root,cc,'cuda')
    cd, rd = tensor_digest(codec.state_dict()), tensor_digest(reader.state_dict())
    latents, targets, preflight = {}, {}, []
    with h5py.File(source/'source.h5') as f, torch.no_grad():
        from model.losses import mixture_expectation
        for sid in ids:
            truth = f[sid]['target'][:]; z = torch.tensor(f[sid]['latent_mean'][:],device='cuda'); n=len(truth)
            decoded=codec.decode(z.T[None],padding_mask=torch.zeros(1,len(z)*8,dtype=torch.bool,device='cuda'))
            xy=mixture_expectation(decoded)[0,:n]; pen=decoded[0,:3,:n].argmax(0)
            err=float((xy-torch.tensor(truth[:,:2],device='cuda')).square().mean().sqrt())
            pen_errors=int((pen!=torch.tensor(truth[:,2:].argmax(1),device='cuda')).sum())
            if err>.0005 or pen_errors: raise ValueError('codec preflight failure')
            text=read_sequence(reader,torch.cat((xy,torch.nn.functional.one_hot(pen,3).float()),1),vocab)
            preflight.append(dict(sample_id=sid,codec_rmse=err,pen_errors=pen_errors,reader_errors=edit_distance(records[sid]['text'],text),reader_text=text))
            latents[sid]=z; targets[sid]=truth
    stats=torch.load(source/'whitening.pt',weights_only=True); stats={k:v.cuda() if torch.is_tensor(v) else v for k,v in stats.items()}
    pool=CachedLatentPool(latents,records,vocab,ids,stats,max_lines=264)
    out=root/'checkpoints/iam_generation_alignment'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    # Packed archive preserves the complete as-run implementation without a file-per-module Volume cost.
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as tar:
        for directory in ['iam_tools','model','dataset','utils','trainer','configs']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for p in sorted(base.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json'):
                    tar.add(p,arcname=directory+'/'+str(p.relative_to(base)))
    duration=fit_duration(records,scope['training_ids']['global'],writers)
    seal=seal_confirmation(manifest,data['records'],scope['training_ids']['global'],writers,development_ids=scope['splits']['unseen_prompt'])
    synthetic=seal_synthetic(data['records'],scope['training_ids']['global'],writers)
    torch.manual_seed(28142)
    model_spec=dict(reference['config']['model'],blocks_per_character=duration['blocks_per_character'],prior_sigma=2.5,prior_cap=12.)
    prototype=AlignedWriterDenoiser(alignment=False,**model_spec).cuda()
    anchors=reference['config']['auxiliary_weights'];del reference
    cfg=dict(profile='fresh global vs soft alignment bundle, standalone deterministic text mapper, NOT TrInk/semantic InkVAE/released InkDiT',
             reference_config_only=REFERENCE,reference_sha256=REFERENCE_SHA,fresh=True,parent_step=0,
             data_parent=DATA,source_h5_sha256=SOURCE_H5_SHA,data_manifest_sha256=DATASET_SHA,whitening_sha256=WHITENING_SHA,
             vocab=vocab,writers=writers,models={a:dict(model_spec,alignment=a=='soft_gaussian') for a in ARMS},
             source_rel=parent_cfg['source_rel'],source_sha256=parent_cfg['source_sha256'],reader_rel=parent_cfg['reader_rel'],reader_sha256=parent_cfg['reader_sha256'],
             max_updates=steps,eval_steps=sorted(set([0,steps//3,2*steps//3,steps])),max_train_wall_seconds_per_arm=1800,
             batch=8,betas=[.9,.99],weight_decay=.01,clip=1.,text_dropout=0.,trajectory_dropout=0.,
             model_seed=28142,schedule_seed=29142,initialization='identical FRESH neural weights and empty Adam; not a memorized checkpoint continuation',
             input='zeros,t999,oracleceil(N/8) for training; BOTH oracle and TRAIN-only text/writer-estimated durations evaluated',
             auxiliary_weights=anchors,duration_model=duration,
             lr_schedule='1000-update linear1e-5→5e-5 warmup; hold through2/3 budget; cosine→1e-5 final1/3; identical actual groups/order',
             whitening_policy='original TRAIN32 statistics fixed, no held statistics',
             selection='all actual256TRAIN axisRMSE+.25segmentRMSE+.1penerror; never dev or sealed confirmation',
             alignment_definition='TRAIN average blocks/character center=min(query/r,text_length-1); sigma2.5characters; log bias floor-12;3of4heads biased,one global; finite BOS bias-2; trainable PE AMPLITUDES(alpha),NOT a rescaled position-index grid',
             citation='https://aclanthology.org/2025.emnlp-main.244/ sections2.2/2.3; inspired PARTIAL adaptation: no contextual encoder/autoregression/MDN or600k-example training',
             caveats='one fresh initialization; oracle duration leaks identity; corpus-familiar reader; only256TRAIN; alignment+PE amplitudes bundled not separate causal ablations; fixed inherited physical loss weights not fresh gradient recalibration; development8 repeatedly inspected; fresh16 paired +16 unpaired generator-confirmation held sealed until TRAIN-selected models frozen',
             confirmation_policy=seal['policy'],not_promoted=True,source_archive_sha256=file_sha(out/'as-run-source.tar.gz'))
    dataset=dict(records=records,writers=writers,**scope,preflight=preflight,confirmation_seal=seal,synthetic_seal=synthetic)
    (out/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n'); (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    results={}
    with ResourceMonitor(out,interval=5,sustained_seconds=30) as monitor:
        for arm in ARMS:
            train=scope['training_ids'][arm];torch.manual_seed(cfg['model_seed'])
            model=copy.deepcopy(prototype).train();model.alignment=arm=='soft_gaussian';model.config=cfg['models'][arm]
            opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
            schedule=list(training_schedule(train,steps,8,seed=cfg['schedule_seed']));folder=out/arm;folder.mkdir()
            history=[];duration_history=[];seconds=0.;best_step=0;clipped=0;stop='budget_completed';initial_digest=tensor_digest(model.state_dict())
            def save(name,step):
                torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),config=dict(cfg,arm=arm,model=cfg['models'][arm]),step=step,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),folder/name)
            def assess(step):
                with monitor.in_phase('eval/'+arm):
                    ev=evaluate(model,codec,reader,pool,latents,records,vocab,stats,scope['splits'],targets,folder,step,writers,controls=step==steps)
                    de=evaluate_duration(model,codec,reader,records,vocab,stats,scope['splits']['unseen_prompt'],folder,step,writers,duration,controls=step==steps)
                    duration_history.append(dict(step=step,aggregate=de['aggregate']))
                    s=score(ev,set(train)); history.append(dict(step=step,train_score=s,aggregate=ev['aggregate']));return s
            best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);monitor.set_phase('train/'+arm)
            with (folder/'metrics.jsonl').open('w') as log:
                for step,batch in enumerate(schedule,1):
                    start=time.monotonic()
                    for g in opt.param_groups:g['lr']=learning_rate(step,steps)
                    clean,mask,text,pm=pool.select(batch);wi=writer_tensor(batch,records,writers,'cuda')
                    pred=model(torch.zeros_like(clean),torch.ones(len(batch),device='cuda'),text,mask,writer_ids=wi)
                    terms=physical_terms(pred,clean,stats,pm,codec_contract='polyphase40');mse=masked_mse(pred,clean,mask)
                    loss=mse+anchors['xy']*terms['xy']+anchors['first_difference']*terms['first_difference']
                    opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));opt.step()
                    row=dict(step=step,text_pe_scale=float(model.text_pe_scale.detach()),ink_pe_scale=float(model.ink_pe_scale.detach()),learning_rates=[g['lr'] for g in opt.param_groups],sample_ids=batch,loss=float(loss.detach()),base_mse=float(mse.detach()),physical_xy_mse=float(terms['xy'].detach()),segment_mse=float(terms['first_difference'].detach()),gradient_norm=norm,clipped=norm>cfg['clip'])
                    log.write(json.dumps(row)+'\n');elapsed=time.monotonic()-start;seconds+=elapsed;monitor.step(elapsed,len(batch));clipped+=row['clipped']
                    if step%1000==0:print(dict(arm=arm,**row,train_seconds=seconds),flush=True);log.flush()
                    if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite capacity loss')
                    limit=seconds>cfg['max_train_wall_seconds_per_arm']
                    if step in cfg['eval_steps'] or limit:
                        s=assess(step);save('checkpoint-last.pt',step)
                        if s<best:best=s;best_step=step;save('checkpoint-best.pt',step)
                    if limit:stop='wall_limit';break
            save('checkpoint-last.pt',step)
            if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozen codec/reader drift')
            result=dict(last_step=step,best_step=best_step,best_train_score=best,history=history,duration_history=duration_history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial_digest,codec_reader_unchanged=True,last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),not_promoted=True)
            results[arm]=result;(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');monitor.set_phase('cleanup');del model,opt;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(last=r['last_step'],train=r['history'][-1]['aggregate']['all_train256']['correct']) for a,r in results.items()})
