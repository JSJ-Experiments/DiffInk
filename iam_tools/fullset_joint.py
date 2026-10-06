"""Deterministic full-set geometry+pen polish; zero held-out training access."""
import json,time
from pathlib import Path
import torch
from .writer_expansion import load,device_batches,evaluate,train_score
from .curve_study import forward_xy
from .conditioning_study import SOURCE_REL,SOURCE_SHA,PEN_WEIGHT
from .latent_integration import DELTA_WEIGHT,runtime_metadata
from .pen_refit import refit_loss
from .pen_ab import file_sha


def joint_loss(model,batch,relative=False):
    from model.losses import target_difference_loss
    raw,mask,_=batch;xy,target,out=forward_xy(model,raw,mask);states=raw[:,2:].argmax(1)
    point=(xy[mask]-target[mask]).square().mean()
    delta=target_difference_loss(xy,target,states,mask)
    pen=refit_loss(out[:,:3].transpose(1,2)[mask],states[mask],'bounded_three_state')
    result=dict(point=point,delta=delta,pen=pen)
    if relative:
        from .local_geometry import target_relative_difference_loss
        result['relative']=target_relative_difference_loss(xy,target,states,mask)
    return result


def run(config,repo,root='/data',steps=240,relative_fraction=0.,source_rel=SOURCE_REL,source_sha=SOURCE_SHA,experiment_name='diffink-english-fullset-joint',
        accelerated=False,writer_id='checkpoint',metric_pool=None,max_wall_seconds=2400,eval_every=40,pen_weight=PEN_WEIGHT,family='iam_fullset_joint'):
    if not 1<=steps<=300 or not 0<=relative_fraction<=.5:raise ValueError('bounded full-set joint diagnostic required')
    if family not in ('iam_fullset_joint','iam_geometry_fullset') or not 1<=max_wall_seconds<=2400 or not 1<=eval_every<=40 or pen_weight<0:
        raise ValueError('bounded known full-set research contract required')
    if accelerated and relative_fraction:raise ValueError('relative loss not yet supported by captured runner')
    if not torch.cuda.is_available():raise RuntimeError('T4 required')
    torch.set_num_threads(4);torch.manual_seed(42);root=Path(root)
    model,samples,raw,cfg,vocab,prov=load(config,repo,root,source_rel,source_sha,allow_research_conditioning=True,writer_id=writer_id)
    model.to('cuda').eval();batches=device_batches(raw,'cuda');splits=prov['splits']
    from .conditioning_study import apply_mode
    batches,offsets,_=apply_mode(model,batches,cfg['conditioning_mode'])
    from .local_geometry import calibrate_relative
    calibration=calibrate_relative(model,[batches[i] for i in splits['train']],relative_fraction) if relative_fraction else dict(relative_weight=0.,target_fraction=0.)
    relative_weight=calibration['relative_weight']
    directory=root/'checkpoints'/family/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    cfg.update(profile='seen-writer-full-set-joint-LBFGS-diagnostic',experiment_name=experiment_name,research_contract='fullset_joint runner; deterministic mean objective, not standard sampled trainer',training_latent='mean only; sampled-z evaluated but not optimized',
               physical_batch=1,gradient_accumulation_steps=len(splits['train']),optimizer='fresh LBFGS',base_lr=1.,max_outer_steps=steps,
               lbfgs_max_iter=10,lbfgs_max_eval=15,lbfgs_history_size=20,lbfgs_line_search='strong_wolfe',
               tolerance_grad=1e-9,tolerance_change=1e-13,expected_xy_weight=1.,target_delta_weight=DELTA_WEIGHT,
               pen_weight=pen_weight,target_relative_delta_weight=relative_weight,relative_gradient_fraction=relative_fraction,mean_xy_anchor_weight=0.,gmm_weight=0.,kl_weight=0.,ctc_weight=0.,style_weight=0.,
               trans_dropout=0.,grad_clip=None,eval_every=eval_every,sampled_z_evaluations=20,max_wall_seconds=max_wall_seconds,
               cuda_graphs=accelerated,metric_workers=getattr(metric_pool,'_max_workers',0),
               checkpoint_selection='train_score only; no held-out selection',**runtime_metadata())
    model.config.__dict__.update(cfg)
    (directory/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(directory/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    for name in ('fullset_joint.py','writer_expansion.py','curve_study.py','latent_integration.py','conditioning_study.py','pen_refit.py','local_geometry.py','conditioning.py','fast_geometry.py','metric_workers.py'):
        p=directory/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    for rel in ('model/vae.py','model/blocks.py','model/losses.py'):
        p=directory/'source-code'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/rel).read_bytes())
    model.requires_grad_(True)
    for module in (model.conv_logvar,model.ocr_model,model.style_classifier):module.requires_grad_(False)
    protected={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('conv_logvar.','ocr_model.','style_classifier.'))}
    fc_initial={k:model.state_dict()[k][63:].cpu().clone() for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias')}
    hooks=[]
    for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
        mask=torch.zeros_like(p);mask[:63]=1;hooks.append(p.register_hook(lambda g,m=mask:g*m))
    optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=1.,max_iter=10,max_eval=15,history_size=20,
                               line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-13)
    graphs=None
    if accelerated:
        from .fast_geometry import GeometryGraphs
        graphs=GeometryGraphs(model,pen_weight,sampled_weight=0.)
        capture_start=time.monotonic();graphs.prepare([batches[i] for i in splits['train']])
        cfg['graph_capture_seconds']=time.monotonic()-capture_start
        cfg['captured_lengths']=list(graphs.cache)
        cfg['capture_memory_reserved_gb']=torch.cuda.memory_reserved()/2**30
        (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print(dict(graphs=len(graphs.cache),capture_seconds=cfg['graph_capture_seconds'],memory_gb=cfg['capture_memory_reserved_gb']),flush=True)
    initial=evaluate(model,samples,batches,splits,vocab,directory,0,xy_offsets=offsets,metric_pool=metric_pool);best=train_score(initial);best_step=0;history=[];calls=0;last={};started=time.monotonic()
    def save(name,step,include_optimizer=False):
        payload=dict(model_state_dict=model.state_dict(),config=cfg,
                        sample_ids=splits['train'],optimizer_updates=step,outer_optimizer_steps=step,closure_calls=calls,
                        source_sha256=source_sha,provenance=prov)
        if include_optimizer:payload['optimizer_state_dict']=optimizer.state_dict()
        torch.save(payload,directory/name)
    save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
    def closure():
        nonlocal calls,last
        if graphs is not None:
            optimizer.zero_grad(set_to_none=False);totals=torch.zeros(3,device='cuda')
            for sid in splits['train']:totals+=graphs.replay(batches[sid],divisor=len(splits['train']))
            totals/=len(splits['train']);loss=totals[0]+pen_weight*totals[2]
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite captured full-set joint objective')
            values=totals.cpu().tolist();calls+=1;last=dict(mean_geometry=values[0],pen=values[2])
            return loss
        optimizer.zero_grad(set_to_none=True);totals=dict(point=0.,delta=0.,pen=0.)
        if relative_weight:totals['relative']=0.
        for sid in splits['train']:
            t=joint_loss(model,batches[sid],relative=relative_weight>0);loss=(t['point']+DELTA_WEIGHT*t['delta']+pen_weight*t['pen']+relative_weight*t.get('relative',0))/len(splits['train'])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite full-set joint objective')
            loss.backward()
            for k in totals:totals[k]+=float(t[k].detach())/len(splits['train'])
        calls+=1;last=totals
        return loss.new_tensor(totals['point']+DELTA_WEIGHT*totals['delta']+pen_weight*totals['pen']+relative_weight*totals.get('relative',0))
    stop='budget_completed'
    with (directory/'metrics.jsonl').open('w') as log:
        for step in range(1,steps+1):
            optimizer.step(closure)
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            log.write(json.dumps(dict(step=step,closure_calls=calls,**last,
                  gradient_norm=float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt()),
                  gradient_abs_max=max(float(g.abs().max()) for g in gradients)))+'\n');log.flush()
            optimizer.zero_grad(set_to_none=graphs is None)
            if step%eval_every==0 or step==steps:
                row=evaluate(model,samples,batches,splits,vocab,directory,step,xy_offsets=offsets,metric_pool=metric_pool);score=train_score(row)
                history.append(dict(step=step,score=score,groups=row['groups']));save(f'checkpoint-{step}.pt',step)
                if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
            if time.monotonic()-started>max_wall_seconds:stop='wall_limit';break
    save('checkpoint-last.pt',step,include_optimizer=True)
    for h in hooks:h.remove()
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in protected.items())
    assert all(torch.equal(v,model.state_dict()[k][63:].cpu()) for k,v in fc_initial.items())
    result=dict(output=str(directory),initial=initial['groups'],history=history,best_step=best_step,last_step=step,closure_calls=calls,
                best_training_score=best,stop_reason=stop,elapsed_seconds=time.monotonic()-started,
                protected_parameters_bitwise_unchanged=True,source_unchanged=file_sha(root/source_rel)==source_sha)
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
