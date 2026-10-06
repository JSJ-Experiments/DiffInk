"""Bounded deterministic 24-line optimizer diagnostic; held-out evaluation only.

Mean XY + target differences, not smoothing. Fresh full-training-set L-BFGS;
pen rows refitted afterwards against frozen mean and four sampled features.
Not a production optimizer or stochastic/KL/OCR training claim.
"""
import json
from pathlib import Path
import time
import torch
from .writer_expansion import load, device_batches, evaluate, train_score
from .curve_study import losses, forward_xy
from .pen_refit import refit_loss
from .pen_ab import file_sha
from .latent_integration import DELTA_WEIGHT, runtime_metadata


def refit_pen(model,batches,folder,updates=2000):
    """Only supplied TRAIN batches are cached. XY/body bitwise unchanged."""
    if not batches or not 1<=updates<=2000: raise ValueError('bounded nonempty pen refit required')
    before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    fc=model.transformer_decoder.fc;captured=[]
    hook=fc.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach()))
    features=[];device=fc.weight.device;devices=[device.index or 0] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            torch.manual_seed(5042)
            for raw,mask,_ in batches:
                draws=[]
                for k in range(5):
                    forward_xy(model,raw,mask,sampled=k>0);draws.append(captured.pop()[0,mask[0]])
                features.append((draws,raw[0,2:,mask[0]].argmax(0)))
    finally:hook.remove()
    head=torch.nn.Linear(fc.in_features,3,device=device)
    with torch.no_grad():head.weight.copy_(fc.weight[:3]);head.bias.copy_(fc.bias[:3])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.99),weight_decay=0)
    torch.manual_seed(42);started=time.monotonic();history=[]
    for step in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.stack([refit_loss(head(draws[(step-1)%5]),labels,'bounded_three_state',gamma=2,cap=8)
                          for draws,labels in features]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite pen refit')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if step%200==0:history.append(dict(step=step,loss=float(loss.detach())))
    with torch.no_grad():fc.weight[:3].copy_(head.weight);fc.bias[:3].copy_(head.bias)
    for k,v in model.state_dict().items():
        old=before[k]
        if k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'):assert torch.equal(old[3:],v[3:].cpu())
        else:assert torch.equal(old,v.cpu()),k
    result=dict(updates=updates,lr=.001,training_lines=len(batches),cached_features='mean + four fixed sampled-z per training line; cycling',
                non_pen_parameters_bitwise_unchanged=True,elapsed_seconds=time.monotonic()-started,history=history)
    folder.mkdir();(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def run(config,repo,root,source_rel,source_sha,steps=80):
    if not 1<=steps<=100 or not source_sha or not source_rel:raise ValueError('pinned bounded polish required')
    if not torch.cuda.is_available():raise RuntimeError('T4 required')
    torch.set_num_threads(4);torch.manual_seed(42)
    model,samples,raw,cfg,vocab,provenance=load(config,repo,root,source_rel,source_sha)
    model.to('cuda').eval();batches=device_batches(raw,'cuda');splits=provenance['splits']
    model.requires_grad_(True)
    for module in (model.style_classifier,model.ocr_model,model.conv_logvar):module.requires_grad_(False)
    protected={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.','conv_logvar.'))}
    directory=Path(root)/'checkpoints/iam_writer_polish'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    cfg.update(sample_ids=splits['train'],profile='seen-writer-24-line-full-set-LBFGS-diagnostic',optimizer='fresh LBFGS',
               base_lr=1.,lbfgs_max_iter=10,lbfgs_max_eval=15,lbfgs_history_size=10,lbfgs_line_search='strong_wolfe',
               tolerance_grad=1e-8,tolerance_change=1e-12,physical_batch=1,gradient_accumulation_steps=len(splits['train']),
               training_latent='mean',model_input_scale=.01,trans_dropout=0,expected_xy_weight=1.,target_delta_weight=DELTA_WEIGHT,
               mean_xy_anchor_weight=0.,pen_weight=0.,kl_weight=0.,ctc_weight=0.,style_weight=0.,gmm_weight=0.,grad_clip=None,
               max_outer_updates=steps,max_wall_seconds=1200,eval_every=20,pen_refit='2000 head-only updates; train only; mean + four z features',
               checkpoint_selection='training-only score after same pen refit; held-out never used',**runtime_metadata())
    (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for rel in ('writer_polish.py','writer_expansion.py','curve_study.py','trajectory_geometry.py'):
        dest=directory/'source-code/iam_tools'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).with_name(rel).read_bytes())
    for rel in ('model/vae.py','model/losses.py','model/ocr.py'):
        dest=directory/'source-code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path(repo)/rel).read_bytes())
    initial=evaluate(model,samples,batches,splits,vocab,directory,0)
    hooks=[]
    for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
        active=torch.zeros_like(p);active[3:63]=1;hooks.append(p.register_hook(lambda g,m=active:g*m))
    optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=1.,max_iter=10,max_eval=15,history_size=10,
                               line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
    calls=0;last={};history=[];started=time.monotonic();stop='budget_completed'
    def save(name,snapshot_step,outer_steps=None):
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=cfg,sample_ids=splits['train'],
                        outer_optimizer_steps=snapshot_step if outer_steps is None else outer_steps,
                        snapshot_step=snapshot_step,pen_refit_updates=0 if outer_steps is None else 2000,
                        closure_calls=calls,source_sha256=source_sha,provenance=provenance),directory/name)
    save('checkpoint-initial.pt',0)
    def closure():
        nonlocal calls,last
        optimizer.zero_grad(set_to_none=True);totals=dict(point=0.,delta=0.)
        for sid in splits['train']:
            raw,mask,_=batches[sid];t=losses(model,raw,mask,'delta');loss=(t['point']+DELTA_WEIGHT*t['delta'])/len(splits['train'])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite LBFGS objective')
            loss.backward()
            for k in totals:totals[k]+=float(t[k].detach())/len(splits['train'])
        calls+=1;last=totals
        return raw.new_tensor(totals['point']+DELTA_WEIGHT*totals['delta'])
    with (directory/'metrics.jsonl').open('w') as log:
        for step in range(1,steps+1):
            optimizer.step(closure);optimizer.zero_grad(set_to_none=True)
            log.write(json.dumps(dict(step=step,closure_calls=calls,**last))+'\n');log.flush()
            if step%20==0 or step==steps:
                row=evaluate(model,samples,batches,splits,vocab,directory,step);save(f'checkpoint-{step}.pt',step)
                history.append(dict(step=step,groups=row['groups'],score=train_score(row)))
            if time.monotonic()-started>1200:stop='wall_limit';break
    for hook in hooks:hook.remove()
    save('checkpoint-geometry.pt',step)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in protected.items())
    pen=refit_pen(model,[batches[sid] for sid in splits['train']],directory/'pen-refit')
    final_step=step+2000;final=evaluate(model,samples,batches,splits,vocab,directory,final_step)
    # Source baseline remains available; no held-out model selection.
    passed=train_score(final)<train_score(initial);save('checkpoint-final.pt',final_step,outer_steps=step)
    selected=final_step if passed else 0
    torch.save(torch.load(directory/('checkpoint-final.pt' if passed else 'checkpoint-initial.pt'),map_location='cpu',weights_only=True),directory/'checkpoint-best.pt')
    result=dict(output=str(directory),source_sha256=source_sha,initial=initial['groups'],history=history,final=final['groups'],
                best_step=selected,last_step=final_step,optimizer_outer_steps=step,closure_calls=calls,stop_reason=stop,
                elapsed_seconds=time.monotonic()-started,pen_refit=pen,geometry_improved_by_training_score=passed,
                ocr_style_logvar_parameters_bitwise_unchanged=True,source_unchanged=file_sha(Path(root)/source_rel)==source_sha)
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
