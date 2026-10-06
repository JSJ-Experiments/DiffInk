"""Seen-writer, line-disjoint reconstruction expansion (not an IAM benchmark).

Held-out lines are evaluation-only: no calibration, gradients, OCR warmup or
checkpoint selection. Physical batches contain one minimally padded line.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .inkvae import setup, greedy_ctc, edit_distance
from .autopsy import selected_sample, sample_sha
from .pen_ab import file_sha, boundary_metrics
from .trajectory_geometry import geometry_metrics
from .latent_integration import terms, encoded, summary, runtime_metadata, DELTA_WEIGHT

SOURCE_REL = 'checkpoints/iam_latent_integration/20261005-133620/ocr/checkpoint-best.pt'
SOURCE_SHA = 'fffe1405db10f8f6c2ce6ec1e030706b7947c93d83fa4eaeffec3b6a8c5b08f9'
MANIFEST_SHA = '6d1fafea62af6c5ddae48983bb9699d93016e4af7a11df2f8dd1743128c34c8a'


def partition(train_ids, val_ids, old_ids):
    train_ids, val_ids, old_ids = list(train_ids), list(val_ids), list(old_ids)
    for values in (train_ids, val_ids, old_ids):
        if len(values) != len(set(values)): raise ValueError('duplicate sample ID')
    if set(train_ids) & set(val_ids): raise ValueError('held-out leakage')
    if not set(old_ids) <= set(train_ids): raise ValueError('old samples not in training')
    if not train_ids or not val_ids or not old_ids: raise ValueError('empty split')
    return dict(old=old_ids, new=[i for i in train_ids if i not in old_ids], train=train_ids, held_out=val_ids)


def resolve_writer_scope(parent, writer_id='checkpoint'):
    """Explicit None selects all prepared writers; old checkpoints stay single-writer."""
    value=parent.get('config',{}).get('research_writer_id','10174') if writer_id=='checkpoint' else writer_id
    if value is not None and (not isinstance(value,str) or not value.isdecimal()):
        raise ValueError('numeric writer ID, explicit None, or checkpoint scope required')
    return value


def retention_groups(parent, train_ids, val_writers):
    """Keep the original retained reference stable across selected reloads/resumes."""
    parent_splits=parent.get('provenance',{}).get('splits',{})
    source_ids=parent_splits.get('source_trained',parent_splits.get('train',parent['sample_ids']))
    if not source_ids or len(source_ids)!=len(set(source_ids)) or not set(source_ids)<=set(train_ids) or set(train_ids)&val_writers.keys():
        raise ValueError('disjoint prepared splits and retained training-only reference required')
    groups=dict(source_trained=list(source_ids),added=[i for i in train_ids if i not in source_ids],
                same_writer_held_out=[i for i,w in val_writers.items() if w=='10174'],
                other_writers_held_out=[i for i,w in val_writers.items() if w!='10174'])
    return {k:v for k,v in groups.items() if v}


def load(config, repo, root, source_rel=SOURCE_REL, source_sha=SOURCE_SHA, allow_research_conditioning=False, writer_id='checkpoint'):
    root = Path(root); source = root/source_rel
    if file_sha(source) != source_sha: raise ValueError('source SHA mismatch')
    if file_sha(root/'diffink/iam_overfit/manifest.json') != MANIFEST_SHA: raise ValueError('manifest SHA mismatch')
    model, train, val, cfg, data_root = setup(config, repo, root/'diffink/iam_overfit')
    try:
        parent = torch.load(source, map_location='cpu', weights_only=True)
        old = parent.get('provenance', {}).get('splits', {}).get('old', parent['sample_ids']); writer = resolve_writer_scope(parent,writer_id)
        ids = {name: sorted(i for i in ds.keys if writer is None or ds.hf[i]['writer_id'][()].decode() == writer)
               for name, ds in [('train', train), ('val', val)]}
        splits = partition(ids['train'], ids['val'], old)
        if writer is None:
            splits.update(retention_groups(parent,ids['train'],{i:val.hf[i]['writer_id'][()].decode() for i in ids['val']}))
        samples = {i: selected_sample(train if i in ids['train'] else val, i) for i in ids['train']+ids['val']}
        batches = {i: (train if i in ids['train'] else val).collate_fn([s]) for i,s in samples.items()}
        model.load_state_dict(parent['model_state_dict'], strict=True); model.apply_checkpoint_contract(parent, allow_research_conditioning=allow_research_conditioning); model.eval()
        cfg.update(sample_ids=ids['train'], model_input_scale=.01, trans_dropout=0,
                   research_writer_id=writer,
                   conditioning_mode=parent.get('config',{}).get('conditioning_mode','control'))
        vocab = list(json.loads((data_root/'chars.json').read_text()))
        provenance = dict(source_rel=source_rel, source_sha256=source_sha, manifest_sha256=MANIFEST_SHA,
                          samples={i: dict(sha256=sample_sha(s), writer=str(s[0]), text=s[2], points=len(s[1]),
                                          padded_points=batches[i][0].shape[1]) for i,s in samples.items()},
                          splits=splits, split_policy='seen-writer line-disjoint; forms overlap; NOT writer/form-independent')
        return model, samples, batches, cfg, vocab, provenance
    finally: train.hf.close(); val.hf.close()


def device_batches(batches, device):
    return {i: (b[0].to(device).transpose(1,2), b[1].to(device), b[2].to(device)) for i,b in batches.items()}


def evaluate(model, samples, batches, splits, vocab, folder, step, draws=20, xy_offsets=None):
    """Fixed paired noise on CPU or GPU; evaluation never advances training RNG."""
    folder = Path(folder); rows=[]
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for j, sid in enumerate(sorted(samples)):
            raw, mask, labels = batches[sid]; sample=samples[sid]
            torch.manual_seed(1042+j*100); variants=[]
            destination=folder/f'step-{step}'/sid; destination.mkdir(parents=True, exist_ok=True)
            truth, mu, lv, lm = encoded(model, raw, mask)
            n=int(mask.sum()); target=truth[0,:n].cpu().numpy(); offset=np.zeros(2) if xy_offsets is None else xy_offsets[sid].detach().cpu().numpy(); target=target+offset; states=raw[0,2:,:n].argmax(0).cpu().numpy()
            from model.losses import mixture_expectation
            for k in range(-1, draws):
                z=mu if k<0 else mu+torch.randn_like(mu)*(.5*lv).exp()
                output=model.decode(z,padding_mask=~mask); xy=mixture_expectation(output)[0,:n].cpu().numpy()+offset
                predicted=output[0,:3,:n].argmax(0).cpu().numpy(); kind='mu' if k<0 else f'z-{k}'
                np.save(destination/f'{kind}.npy', np.column_stack([xy,np.eye(3)[predicted]]))
                decoded=greedy_ctc(model.ocr_model(z)[:int(lm.sum()),0].argmax(-1).tolist(),vocab)
                variants.append(dict(kind=kind,geometry=geometry_metrics(xy,target,states),pen=boundary_metrics(predicted,states),
                                     decoded=decoded,ocr_errors=edit_distance(sample[2],decoded),text_characters=len(sample[2])))
            rows.append(dict(sample_id=sid,text=sample[2],mu=variants[0],sampled=variants[1:]))
    result=dict(step=step,lines=rows,groups={})
    for name, ids in splits.items():
        subset=[r for r in rows if r['sample_id'] in ids]
        stats=summary(dict(lines=subset))
        for variant in ('mu','sampled'):
            selected=[r['mu'] for r in subset] if variant=='mu' else [v for r in subset for v in r['sampled']]
            stats[variant+'_cer']=sum(v['ocr_errors'] for v in selected)/sum(v['text_characters'] for v in selected)
            stats[variant+'_macro_pen_f1']=float(np.mean([v['pen']['pen_up_f1'] for v in selected]))
        result['groups'][name]=stats
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(step=step,groups={g:dict(x=s['mu']['mean_per_line_x_rmse'],y=s['mu']['mean_per_line_y_rmse'],
                                      turn=s['mu']['turn_angle_error_degrees']['p90'],pen=s['mu_macro_pen_f1'],cer=s['mu_cer']) for g,s in result['groups'].items()}),flush=True)
    return result


def training_schedule(train_ids, steps, accumulation=8, seed=42):
    """Shuffled complete epochs; never accepts a validation dataset or sampler."""
    if len(set(train_ids)) != len(train_ids) or not train_ids:
        raise ValueError('unique nonempty training IDs required')
    if steps < 1 or accumulation < 1: raise ValueError('positive schedule bounds required')
    rng = np.random.default_rng(seed); queue = []
    for _ in range(steps):
        update = []
        for _ in range(accumulation):
            if not queue: queue = list(rng.permutation(train_ids))
            update.append(str(queue.pop()))
        yield update


def train_score(row):
    """Never select on held-out metrics. Position + local geometry, same units."""
    s=row['groups']['train']; g=s['mu']; z=s['sampled']
    return g['mean_per_line_x_rmse']+g['mean_per_line_y_rmse']+.02*g['mean_per_line_first_difference_relative']+.1*(1-s['mu_macro_pen_f1'])+.1*(z['mean_per_line_x_rmse']+z['mean_per_line_y_rmse'])


def balanced_calibration(model,train_batches,fraction=.2):
    """Initial aggregate encoder gradients; capped scalar, train only."""
    if not train_batches or not 0<fraction<=1:raise ValueError('bounded nonempty calibration required')
    params=list(model.encoder.parameters())+list(model.conv_mu.parameters())
    grads={k:[torch.zeros_like(p) for p in params] for k in ('geometry','pen')}
    device=next(model.parameters()).device;devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(6042)
        for batch in train_batches:
            t=terms(model,batch,pen_on_mean=True)
            for name,loss in [('geometry',t['mean_geometry']+.1*t['sampled_geometry']),('pen',t['pen'])]:
                values=torch.autograd.grad(loss/len(train_batches),params,retain_graph=True,allow_unused=True)
                for acc,g in zip(grads[name],values):
                    if g is not None:acc.add_(g.detach())
    norms={k:float(torch.stack([g.square().sum() for g in values]).sum().sqrt()) for k,values in grads.items()}
    if min(norms.values())<=1e-12:raise ValueError('degenerate calibration')
    weight=min(1.,fraction*norms['geometry']/norms['pen'])
    return dict(encoder_gradient_norms=norms,pen_weight=weight,target_fraction=fraction,
                actual_fraction=weight*norms['pen']/norms['geometry'],mean_geometry_weight=1.,sampled_geometry_weight=.1,
                method='fixed initial full-training-set encoder gradient norm; pen scalar capped1; validation excluded')


def adam_step_with_pen_lr(optimizer,fc,ratio):
    """Per-row Adam LR for the checkpoint-compatible shared FC tensor.

    Rescale the optimizer's proposed pen-row displacement, not its gradients
    (Adam would cancel constant gradient scaling). Moment states remain valid.
    Used only with weight_decay0; all other rows retain the normal group LR.
    """
    if not 1<=ratio<=100 or any(g.get('weight_decay',0)!=0 for g in optimizer.param_groups):
        raise ValueError('bounded ratio and zero weight decay required')
    with torch.no_grad():before=[p[:3].clone() for p in (fc.weight,fc.bias)]
    optimizer.step()
    with torch.no_grad():
        for p,old in zip((fc.weight,fc.bias),before):p[:3].copy_(old+ratio*(p[:3]-old))


def validate_resume(parent,balanced,lr,new_steps):
    cfg=parent['config'];step=parent['optimizer_updates']
    expected='seen-writer-24-line-balanced-joint' if balanced else 'seen-writer-24-line-expansion'
    if cfg.get('profile')!=expected or cfg['base_lr']!=lr:
        raise ValueError('resume objective/LR must match saved experiment')
    if step<1 or not 1<=new_steps or step+new_steps!=cfg['max_optimizer_updates']:
        raise ValueError('resume must complete the originally configured bounded budget')
    if not all(k in parent for k in ('optimizer_state_dict','rng_state_cpu','rng_state_cuda','provenance')):
        raise ValueError('exact optimizer/RNG/provenance required')


def run(config,repo,root='/data',steps=1200,lr=1e-5,source_rel=SOURCE_REL,source_sha=SOURCE_SHA,balanced=False,resume=False):
    if not 1<=steps<=2000 or not 0<lr<=5e-5: raise ValueError('bounded expansion: 1..2000 updates, LR <=5e-5')
    if not torch.cuda.is_available(): raise RuntimeError('T4 required')
    torch.set_num_threads(4); torch.manual_seed(42)
    model,samples,raw,cfg,vocab,provenance=load(config,repo,root,source_rel,source_sha)
    parent=torch.load(Path(root)/source_rel,map_location='cpu',weights_only=True) if resume else None
    start_step=int(parent['optimizer_updates']) if parent else 0
    if parent:
        validate_resume(parent,balanced,lr,steps)
        cfg=dict(parent['config'])
        provenance['resume_original_provenance']=parent['provenance']
    model.to('cuda').eval(); batches=device_batches(raw,'cuda'); splits=provenance['splits']
    directory=Path(root)/'checkpoints/iam_writer_expansion'/time.strftime('%Y%m%d-%H%M%S',time.gmtime()); directory.mkdir(parents=True,exist_ok=False)
    mean_weight,sampled_weight= (1.,.1) if balanced else (1000.,100.)
    cal=json.loads((Path(root)/source_rel).with_name('calibration.json').read_text()) if parent else (balanced_calibration(model,[batches[i] for i in splits['train']]) if balanced else dict(pen_weight=1.))
    cfg.update(profile='seen-writer-24-line-balanced-joint' if balanced else 'seen-writer-24-line-expansion',base_lr=lr,physical_batch=1,gradient_accumulation_steps=8,
               max_optimizer_updates=start_step+steps,new_optimizer_updates=steps,resume_saved_update=start_step,eval_every=200,sampled_z_evaluations=20,expected_xy_weight=sampled_weight,mean_xy_anchor_weight=mean_weight,
               target_delta_weight=DELTA_WEIGHT,pen_weight=cal['pen_weight'],pen_on_mean=balanced,pen_head_lr_ratio=10 if balanced else 1,kl_weight=0.,ctc_weight=0.,style_weight=0.,gmm_weight=0.,
               model_input_scale=.01,trans_dropout=0,grad_clip=5,weight_decay=0,betas=[.9,.99],
               optimizer='fresh AdamW; independent full-train shuffled epochs; eight microbatches/update',
               checkpoint_selection='training-only train_score; held-out never used',max_wall_seconds=1800,seed=4042,**runtime_metadata())
    (directory/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n'); (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for rel in ('model/vae.py','model/losses.py','model/ocr.py'):
        dest=directory/'source-code'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((Path(repo)/rel).read_bytes())
    for rel in ('writer_expansion.py','latent_integration.py','trajectory_geometry.py'):
        dest=directory/'source-code/iam_tools'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(Path(__file__).with_name(rel).read_bytes())
    protected={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('style_classifier.','ocr_model.'))}
    model.requires_grad_(True);model.style_classifier.requires_grad_(False);model.ocr_model.requires_grad_(False)
    hooks=[]
    for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
        active=torch.zeros_like(p);active[:63]=1;hooks.append(p.register_hook(lambda g,m=active:g*m))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,betas=(.9,.99),weight_decay=0)
    if parent:
        optimizer.load_state_dict(parent['optimizer_state_dict'])
        torch.set_rng_state(parent['rng_state_cpu']);torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
    initial=evaluate(model,samples,batches,splits,vocab,directory,start_step)
    best_score=train_score(initial);best_step=start_step;history=[];started=time.monotonic()
    def save(name,step):
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=cfg,
                        sample_ids=splits['train'],optimizer_updates=step,source_sha256=source_sha,provenance=provenance,
                        rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),directory/name)
    save('checkpoint-initial.pt',start_step);save('checkpoint-best.pt',start_step)
    if not parent:torch.manual_seed(4042)
    stop='budget_completed'
    with (directory/'metrics.jsonl').open('w') as log:
        for step, update_ids in enumerate(training_schedule(splits['train'], start_step+steps), 1):
            if step<=start_step:continue
            if step==int(.75*(start_step+steps)):
                for group in optimizer.param_groups:group['lr']*=.3
            optimizer.zero_grad(set_to_none=True);totals={k:0. for k in ('mean_geometry','sampled_geometry','pen')}
            for sid in update_ids:
                t=terms(model,batches[sid],False,pen_on_mean=balanced)
                loss=(sampled_weight*t['sampled_geometry']+mean_weight*t['mean_geometry']+cal['pen_weight']*t['pen'])/8
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite expansion')
                loss.backward()
                for k in totals:totals[k]+=float(t[k].detach())/8
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True))
            if balanced:adam_step_with_pen_lr(optimizer,model.transformer_decoder.fc,10)
            else:optimizer.step()
            log.write(json.dumps(dict(step=step,**totals,raw_gradient_norm=norm,was_clipped=norm>5,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
            if step%200==0 or step==start_step+steps:
                row=evaluate(model,samples,batches,splits,vocab,directory,step);score=train_score(row)
                history.append(dict(step=step,score=score,groups=row['groups']));save(f'checkpoint-{step}.pt',step)
                if score<best_score:best_score=score;best_step=step;save('checkpoint-best.pt',step)
            if time.monotonic()-started>1800:stop='wall_limit';break
    for h in hooks:h.remove()
    save('checkpoint-last.pt',step)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in protected.items())
    result=dict(output=str(directory),source_sha256=source_sha,initial=initial['groups'],history=history,
                initial_step=start_step,resumed_model_optimizer_rng=bool(parent),best_step=best_step,best_training_score=best_score,last_step=step,stop_reason=stop,
                elapsed_seconds=time.monotonic()-started,ocr_style_bitwise_unchanged=True,
                source_unchanged=file_sha(Path(root)/source_rel)==source_sha)
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
