"""Non-paper deterministic eight-line geometry capacity/optimizer diagnostic.

This deliberately changes optimizer, latent readout and removes pen supervision;
it is not a one-variable GMM ablation. Stochastic reconstruction is evaluated,
but only latent-mean expected XY participates in this experiment's objective.
"""
import json
from pathlib import Path
import time
import torch
from .objective_study import load
from .eightline import evaluate
from .pen_ab import file_sha


def geometry_loss(model, raw, mask):
    from model.losses import mixture_expectation
    target=model.to_model_space(raw)
    mu=model.conv_mu(model.encoder(target))
    output=model.decode(mu,padding_mask=~mask.bool())
    xy=mixture_expectation(output);truth=target[:,:2].transpose(1,2)
    return (xy[mask]-truth[mask]).square().mean()


def run(config, repo, source, sha, steps=50):
    if not 1 <= steps <= 100: raise ValueError('LBFGS diagnostic chunk 1..100 outer steps')
    if file_sha(source) != sha: raise ValueError('pinned source required')
    model,samples,batches,cfg,hashes,_ = load(config, repo)
    parent=torch.load(source,map_location='cpu',weights_only=True)
    assert parent['sample_ids']==cfg['sample_ids']
    model.load_state_dict(parent['model_state_dict'],strict=True);model.apply_checkpoint_contract(parent)
    model.to('cuda').eval();model.conv_logvar.requires_grad_(False)
    cfg.update(profile='deterministic-LBFGS-capacity-diagnostic',training_latent='mean',
               gmm_weight=0,pen_weight=0,expected_xy_weight=1,anchor_gradient_fraction=0,
               ctc_weight=0,style_weight=0,kl_weight=0,source_checkpoint=source,source_sha256=sha,
               optimizer='LBFGS',lbfgs_history_size=10,lbfgs_max_iter=10,lbfgs_outer_steps=steps)
    cfg.update(base_lr=1.,grad_clip=None,weight_decay=0.,max_optimizer_updates=steps,
               max_wall_seconds=1800,eval_every=10,output_base='/data/checkpoints/iam_lbfgs_geometry')
    model.config.__dict__.update(cfg)
    directory=Path('/data/checkpoints/iam_lbfgs_geometry')/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    directory.mkdir(parents=True,exist_ok=False)
    before={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.','conv_logvar.'))}
    fc=model.transformer_decoder.fc
    # Pen/sigma/rho rows are not trained in this geometry-only diagnostic.
    masks=[]
    for p in (fc.weight,fc.bias):
        mask=torch.zeros_like(p);mask[3:63]=1
        masks.append(p.register_hook(lambda grad,m=mask:grad*m))
    optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=1.,
                               max_iter=10,max_eval=15,history_size=10,line_search_fn='strong_wolfe',
                               tolerance_grad=1e-8,tolerance_change=1e-10)
    device_batches=[(b[0].to('cuda').transpose(1,2),b[1].to('cuda')) for b in batches]
    closure_calls=0;last_loss=None;started=time.monotonic()
    fixed=[evaluate(model,samples,batches,cfg,'cuda',0,directory)]
    def closure():
        nonlocal closure_calls,last_loss
        optimizer.zero_grad(set_to_none=True);value=0.
        for raw,mask in device_batches:
            loss=geometry_loss(model,raw,mask)/len(batches)
            if not torch.isfinite(loss):raise FloatingPointError('LBFGS nonfinite loss')
            loss.backward();value+=float(loss.detach())
        closure_calls+=1;last_loss=value
        return raw.new_tensor(value)
    log=(directory/'metrics.jsonl').open('w')
    def save(step):
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                        config=cfg,outer_optimizer_steps=step,closure_calls=closure_calls,
                        source_sha256=sha,sample_ids=cfg['sample_ids']),directory/'checkpoint.pt')
    try:
        for step in range(1,steps+1):
            optimizer.step(closure);optimizer.zero_grad(set_to_none=True)
            row=dict(outer_step=step,closure_calls=closure_calls,last_closure_loss=last_loss)
            log.write(json.dumps(row)+'\n');log.flush()
            if step%10==0 or step==steps:
                save(step);fixed.append(evaluate(model,samples,batches,cfg,'cuda',step,directory));print(row,flush=True)
            if time.monotonic()-started>1800:break
    finally:
        log.close()
        for h in masks:h.remove()
    save(step)
    if fixed[-1]['step']!=step:fixed.append(evaluate(model,samples,batches,cfg,'cuda',step,directory))
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
    assert file_sha(source)==sha
    for key in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'):
        actual=model.state_dict()[key].cpu();original=parent['model_state_dict'][key]
        assert torch.equal(actual[:3],original[:3]) and torch.equal(actual[63:],original[63:])
    result=dict(gpu=torch.cuda.get_device_name(),output=str(directory),outer_steps=step,
                closure_calls=closure_calls,initial=fixed[0],final=fixed[-1],config=cfg,
                source_sha256=sha,frozen_parameter_invariants=True,elapsed_seconds=time.monotonic()-started,
                training_samples_only=True,interpretation='multi-knob capacity diagnostic; not production/paper training')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    page=['<!doctype html><meta charset="utf-8"><h1>Deterministic LBFGS geometry diagnostic</h1><p>Mean latent + expected XY only. No GMM/pen/KL/CTC/style objective. Pen rows frozen but hidden features change, so predicted pens may need refitting. This is a multi-knob capacity diagnostic, not a paper reproduction.</p>']
    for row in fixed:
        page.append(f'<h2>Outer LBFGS step {row["step"]}</h2>')
        for line in row['lines']:
            for label in ('mu','sampled-median','sampled-worst'):
                page.append(f'<img width="100%" src="step-{row["step"]}/{line["sample_id"]}/{label}-comparison.png">')
    (directory/'index.html').write_text('\n'.join(page))
    return {k:v for k,v in result.items() if k not in ('initial','final')}
