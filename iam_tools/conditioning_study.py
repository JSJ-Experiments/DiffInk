"""Matched 24-line T4 conditioning ablations; held-out input is evaluation-only."""
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .writer_expansion import load,device_batches,evaluate,train_score,training_schedule,adam_step_with_pen_lr
from .latent_integration import terms,runtime_metadata,DELTA_WEIGHT
from .conditioning import install_channel_norm,condition_batches
from .pen_ab import file_sha

SOURCE_REL='checkpoints/iam_writer_expansion/20261006-022925/checkpoint-best.pt'
SOURCE_SHA='8d591cfe41b5fa73f16877c34d7b3e62bcf3349d109b114aece0e9c681c4ebee'
PEN_WEIGHT=.0016613753687545062


def apply_mode(model,batches,mode):
    count=0
    if mode=='channel':
        count=install_channel_norm(model.encoder)+install_channel_norm(model.decoder)
        if count!=48:raise AssertionError('expected 48 residual normalization layers')
    result,offsets=condition_batches(batches,mode)
    return result,offsets,count


def run(config,repo,root='/data',steps=1000,modes=('control','center','channel'),lr=5e-5,source_rel=SOURCE_REL,source_sha=SOURCE_SHA,writer_id='checkpoint',family='iam_conditioning_study',accelerated=False,metric_pool=None,draw_batch_size=1):
    if not 1<=steps<=2000 or not 0<lr<=1e-4 or not modes or len(modes)>3 or len(set(modes))!=len(modes) or any(m not in ('control','center','channel','edge_pad') for m in modes):
        raise ValueError('bounded unique conditioning arms required')
    if family not in ('iam_conditioning_study','iam_geometry_diversity'):raise ValueError('known research family required')
    if accelerated and tuple(modes)!=('control',):raise ValueError('captured Adam currently supports validated control mode only')
    if not torch.cuda.is_available():raise RuntimeError('T4 required')
    torch.set_num_threads(4);torch.manual_seed(42)
    base,samples,raw,cfg,vocab,provenance=load(config,repo,root,source_rel,source_sha,writer_id=writer_id)
    initial_state=copy.deepcopy(base.state_dict());splits=provenance['splits'];root=Path(root)
    if family=='iam_geometry_diversity':
        train_writers={provenance['samples'][i]['writer'] for i in splits['train']}
        val_writers={provenance['samples'][i]['writer'] for i in splits['held_out']}
        if len(splits['train'])!=192 or len(splits['held_out'])!=32 or len(splits.get('source_trained',[]))!=24 or len(train_writers)!=8 or train_writers!=val_writers:
            raise ValueError('pinned 192/32, eight-writer pilot with retained24 reference required')
    directory=root/'checkpoints'/family/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    output={}
    for mode in modes:
        # Fresh model avoids stale normalization type between arms. Same source/RNG.
        model=copy.deepcopy(base);model.load_state_dict(initial_state);model.to('cuda').eval()
        batches,offsets,count=apply_mode(model,device_batches(raw,'cuda'),mode)
        arm=directory/mode;arm.mkdir()
        calibration=dict(pen_weight=PEN_WEIGHT,method='fixed matched 24-line study coefficient')
        if family=='iam_geometry_diversity':
            from .writer_expansion import balanced_calibration
            calibration=balanced_calibration(model,[batches[i] for i in splits['train']])
        pen_weight=calibration['pen_weight']
        (arm/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        settings=dict(cfg,profile=family,conditioning_mode=mode,
                      research_contract='use conditioning_study loader; not standard production trainer',
                      base_lr=lr,final_lr=lr*.06,optimizer='fresh AdamW; cosine over bounded run',betas=[.9,.99],weight_decay=0,
                      physical_batch=1,gradient_accumulation_steps=8,max_optimizer_updates=steps,eval_every=250,
                      mean_xy_anchor_weight=1.,expected_xy_weight=.1,target_delta_weight=DELTA_WEIGHT,
                      pen_weight=pen_weight,pen_head_lr_ratio=10,pen_on_mean=True,grad_clip=5,
                      kl_weight=0.,ctc_weight=0.,style_weight=0.,gmm_weight=0.,trans_dropout=0.,model_input_scale=.01,
                      sampled_z_evaluations=20,seed=4042,max_wall_seconds=1200,normalization_layers_replaced=count,
                      checkpoint_selection='train_score only; validation excluded',**runtime_metadata())
        settings.update(cuda_graphs=accelerated,metric_workers=getattr(metric_pool,'_max_workers',0),eval_draw_batch_size=draw_batch_size)
        model.config.__dict__.update(settings)
        (arm/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        (arm/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        for name in ('conditioning_study.py','conditioning.py','writer_expansion.py','latent_integration.py','trajectory_geometry.py','fast_geometry.py','metric_workers.py'):
            p=arm/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
        for rel in ('model/vae.py','model/blocks.py','model/losses.py'):
            p=arm/'source-code'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/rel).read_bytes())
        frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.'))}
        model.requires_grad_(True);model.ocr_model.requires_grad_(False);model.style_classifier.requires_grad_(False)
        hooks=[]
        for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
            active=torch.zeros_like(p);active[:63]=1;hooks.append(p.register_hook(lambda g,m=active:g*m))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,betas=(.9,.99),weight_decay=0)
        graphs=None
        if accelerated:
            from .fast_geometry import GeometryGraphs
            graphs=GeometryGraphs(model,pen_weight,sampled_weight=.1)
            capture_started=time.monotonic();graphs.prepare([batches[i] for i in splits['train']])
            settings.update(graph_capture_seconds=time.monotonic()-capture_started,captured_lengths=list(graphs.cache),capture_memory_reserved_gb=torch.cuda.memory_reserved()/2**30)
            (arm/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        initial=evaluate(model,samples,batches,splits,vocab,arm,0,xy_offsets=offsets,metric_pool=metric_pool,draw_batch_size=draw_batch_size)
        best_score=train_score(initial);best_step=0;history=[];started=time.monotonic();torch.manual_seed(4042)
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=settings,
                            sample_ids=splits['train'],optimizer_updates=step,source_sha256=source_sha,provenance=provenance,
                            rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),arm/name)
        save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);stop='budget_completed'
        with (arm/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(training_schedule(splits['train'],steps),1):
                current_lr=lr*(.06+.94*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1))))
                for g in optimizer.param_groups:g['lr']=current_lr
                optimizer.zero_grad(set_to_none=graphs is None);totals=dict(mean_geometry=0.,sampled_geometry=0.,pen=0.)
                captured_totals=torch.zeros(3,device='cuda') if graphs is not None else None
                for sid in ids:
                    if graphs is not None:
                        captured_totals+=graphs.replay(batches[sid],divisor=8)/8
                        continue
                    t=terms(model,batches[sid],pen_on_mean=True)
                    loss=(t['mean_geometry']+.1*t['sampled_geometry']+pen_weight*t['pen'])/8
                    if not torch.isfinite(loss):
                        (arm/'failure.json').write_text(json.dumps(dict(step=step,phase='loss',sample_id=sid,mode=mode,reason='nonfinite loss'),indent=2)+'\n')
                        raise FloatingPointError('nonfinite conditioning loss')
                    loss.backward()
                    for k in totals:totals[k]+=float(t[k].detach())/8
                if captured_totals is not None:
                    if not torch.isfinite(captured_totals).all():
                        (arm/'failure.json').write_text(json.dumps(dict(step=step,phase='captured_loss',sample_ids=ids,reason='nonfinite captured loss; no optimizer update applied'),indent=2)+'\n')
                        raise FloatingPointError('nonfinite captured conditioning loss')
                    totals=dict(zip(('mean_geometry','sampled_geometry','pen'),captured_totals.cpu().tolist()))
                try:
                    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True))
                except RuntimeError as exc:
                    gradients=[p.grad for p in model.parameters() if p.grad is not None]
                    report=dict(step=step,phase='gradient_norm',mode=mode,reason=str(exc),sample_ids=ids,
                                gradients_elementwise_finite=all(torch.isfinite(g).all().item() for g in gradients),
                                note='No update applied. A nonfinite FP32 aggregate norm is not necessarily elementwise nonfinite gradients.')
                    (arm/'failure.json').write_text(json.dumps(report,indent=2)+'\n')
                    raise
                adam_step_with_pen_lr(optimizer,model.transformer_decoder.fc,10)
                log.write(json.dumps(dict(step=step,**totals,raw_gradient_norm=norm,was_clipped=norm>5,lr=current_lr))+'\n');log.flush()
                if step%250==0 or step==steps:
                    row=evaluate(model,samples,batches,splits,vocab,arm,step,xy_offsets=offsets,metric_pool=metric_pool,draw_batch_size=draw_batch_size);score=train_score(row)
                    history.append(dict(step=step,score=score,groups=row['groups']));save(f'checkpoint-{step}.pt',step)
                    if score<best_score:best_score=score;best_step=step;save('checkpoint-best.pt',step)
                if time.monotonic()-started>1200:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        for h in hooks:h.remove()
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items())
        assert all(torch.equal(initial_state[k][63:],model.state_dict()[k][63:].cpu()) for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'))
        result=dict(output=str(arm),mode=mode,source_sha256=source_sha,initial=initial['groups'],history=history,
                    best_step=best_step,best_training_score=best_score,last_step=step,stop_reason=stop,
                    elapsed_seconds=time.monotonic()-started,ocr_style_bitwise_unchanged=True,sigma_rho_rows_unchanged=True,
                    source_unchanged=file_sha(root/source_rel)==source_sha)
        (arm/'result.json').write_text(json.dumps(result,indent=2)+'\n');output[mode]=result
        del optimizer,model,batches;torch.cuda.empty_cache()
    (directory/'result.json').write_text(json.dumps(output,indent=2)+'\n')
    return dict(output=str(directory),arms={m:dict(best_step=r['best_step'],last_step=r['last_step'],stop_reason=r['stop_reason']) for m,r in output.items()})


def verify(directory,repo,root='/data'):
    """Independent CPU reload respects explicit research conditioning contract."""
    from .curve_study import forward_xy
    torch.set_num_threads(4);directory=Path(directory);root=Path(root)
    selected=directory/'checkpoint-best.pt';saved=torch.load(selected,map_location='cpu',weights_only=True)
    model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,str(selected.relative_to(root)),file_sha(selected),allow_research_conditioning=True)
    original=json.loads((directory/'provenance.json').read_text())
    if prov['samples']!=original['samples'] or prov['splits']!=original['splits']:raise AssertionError('provenance mismatch')
    mode=saved['config']['conditioning_mode'];batches,offsets,count=apply_mode(model,device_batches(raw,'cpu'),mode)
    step=saved['optimizer_updates'];rows=[]
    with torch.no_grad():
        for sid,(x,mask,_) in batches.items():
            xy,_,out=forward_xy(model,x,mask);n=int(mask.sum());cpu=(xy[0,:n]+offsets[sid]).numpy()
            arr=np.load(directory/f'step-{step}/{sid}/mu.npy')
            rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(cpu-arr[:,:2]).max()),
                             pen_mismatches=int((out[0,:3,:n].argmax(0).numpy()!=arr[:,2:].argmax(1)).sum())))
    parent=torch.load(root/original['source_rel'],map_location='cpu',weights_only=True);state=model.state_dict()
    report=dict(mode=mode,selected_step=step,selected_sha256=file_sha(selected),
                logvar_head_required_frozen='LBFGS' in saved['config'].get('optimizer',''),
                logvar_head_unchanged=all(torch.equal(v,parent['model_state_dict'][k]) for k,v in state.items() if k.startswith('conv_logvar.')),
                max_cpu_gpu_xy_difference=max(r['max_xy_difference'] for r in rows),pen_argmax_mismatches=sum(r['pen_mismatches'] for r in rows),
                ocr_style_bitwise_unchanged=all(torch.equal(v,parent['model_state_dict'][k]) for k,v in state.items() if k.startswith(('ocr_model.','style_classifier.'))),
                sigma_rho_rows_unchanged=all(torch.equal(state[k][63:],parent['model_state_dict'][k][63:]) for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias')),
                finite=all(torch.isfinite(v).all().item() for v in state.values()),source_unchanged=file_sha(root/original['source_rel'])==original['source_sha256'],lines=rows)
    if report['max_cpu_gpu_xy_difference']>1e-4 or report['pen_argmax_mismatches'] or not all(report[k] for k in ('ocr_style_bitwise_unchanged','sigma_rho_rows_unchanged','finite','source_unchanged')):raise AssertionError(report)
    if report['logvar_head_required_frozen'] and not report['logvar_head_unchanged']:raise AssertionError('frozen LBFGS logvar head changed')
    (directory/'cpu-reload-check.json').write_text(json.dumps(report,indent=2)+'\n');return report
