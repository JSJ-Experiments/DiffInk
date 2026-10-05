"""Bounded eight-line posterior/KL/OCR integration, gated against frozen geometry.

Engineering experiment: no GMM/style/augmentation. Target differences preserve
corners, not smoothing. Original source is immutable; failed stages never promote.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .objective_study import load
from .curve_study import snapshot
from .pen_ab import file_sha, boundary_metrics
from .pen_refit import refit_loss
from .inkvae import greedy_ctc, edit_distance
from .report_curve_study import aggregate

SOURCE_REL = 'checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt'
SOURCE_SHA = '52f2417e218b21736d46f1502156b4d31f4cd273c1ef669ec8ede51ee759255d'
OCR_REL = 'checkpoints/iam_ctc_head_ab/20261005-114305/blank_zero/checkpoint.pt'
OCR_SHA = 'b4c78fe5277bb8884e7e0203aa0c655cb1cfc08d51f0a07de2bac48cabbac0bd'
DELTA_WEIGHT = .20471838744633777
MEAN_WEIGHT = 1000.


def runtime_metadata():
    # TorchVersion is a str subclass but is NOT weights_only-safe to pickle.
    return dict(torch_version=str(torch.__version__), deterministic_algorithms=False)


def summary(row):
    mu = aggregate([r['mu'] for r in row['lines']])
    sampled = aggregate([s for r in row['lines'] for s in r['sampled']])
    pens = [r['mu']['pen'] for r in row['lines']]
    zpens = [s['pen'] for r in row['lines'] for s in r['sampled']]
    return dict(mu=mu, sampled=sampled, mu_min_pen_f1=min(p['pen_up_f1'] for p in pens),
                sampled_min_pen_f1=min(p['pen_up_f1'] for p in zpens),
                all_final_eoc_correct=all(p['final_eoc_correct'] for p in pens+zpens),
                false_internal_eoc=sum(p['non_final_false_eoc_count'] for p in pens+zpens))


def geometry_gate(row, reference):
    """Per-line fidelity + all-draw pen gate, never just aggregate RMSE."""
    failures = []
    refs = {r['sample_id']: r for r in reference['lines']}
    for line in row['lines']:
        ref = refs[line['sample_id']]['mu']['geometry']; current = line['mu']['geometry']
        for key in ('x_rmse', 'y_rmse'):
            if current[key] > max(ref[key]*2, .0025): failures.append(line['sample_id']+':'+key)
        if current['turn_angle_error_degrees']['p90'] > max(ref['turn_angle_error_degrees']['p90']*1.5, 9.):
            failures.append(line['sample_id']+':turn_p90')
        for variant in [line['mu']]+line['sampled']:
            p = variant['pen']
            minimum = .99 if variant['kind'] == 'mu' else .95
            if p['pen_up_f1'] < minimum or not p['final_eoc_correct'] or p['non_final_false_eoc_count']:
                failures.append(line['sample_id']+':'+variant['kind']+':pen')
    current, ref = summary(row)['sampled'], summary(reference)['sampled']
    for key in ('mean_per_line_x_rmse', 'mean_per_line_y_rmse'):
        if current[key] > ref[key]*1.2: failures.append('sampled:'+key)
    if current['turn_angle_error_degrees']['p90'] > ref['turn_angle_error_degrees']['p90']*1.2:
        failures.append('sampled:turn_p90')
    return dict(passed=not failures, failures=failures,
                thresholds=dict(mu_axis_rmse='per-line max(2*reference, .0025)',
                                mu_turn_p90='per-line max(1.5*reference, 9 degrees)',
                                sampled_mean_rmse_and_turn='<=1.2*reference',
                                mu_pen_min_f1=.99, sampled_pen_min_f1=.95,
                                final_eoc='all correct', false_internal_eoc=0))


def encoded(model, raw, mask):
    from utils.mask import downsample_mask
    target = model.to_model_space(raw)
    features = model.encoder(target)
    mu, logvar = model.conv_mu(features), model.conv_logvar(features)
    return target[:, :2].transpose(1, 2), mu, logvar, downsample_mask(mask, 8).bool()


def terms(model, batch, use_ocr=False, epsilon=None):
    from model.losses import mixture_expectation, target_difference_loss
    raw, mask, labels = batch
    truth, mu, logvar, lm = encoded(model, raw, mask)
    z = mu + torch.randn_like(mu)*(.5*logvar).exp() if epsilon is None else mu+epsilon*(.5*logvar).exp()
    out = model.decode(z, padding_mask=~mask)
    mean_out = model.decode(mu, padding_mask=~mask)
    states = raw[:, 2:].argmax(1)
    def geo(output):
        xy = mixture_expectation(output)
        return (xy[mask]-truth[mask]).square().mean()+DELTA_WEIGHT*target_difference_loss(xy, truth, states, mask)
    return dict(sampled_geometry=geo(out), mean_geometry=geo(mean_out),
                pen=refit_loss(out[:, :3].transpose(1, 2)[mask], states[mask], 'bounded_three_state'),
                kl=model.kl_divergence_new(mu, logvar, lm),
                ctc=model.get_ocr_loss(z, labels, lm) if use_ocr else mu.sum()*0)


def ocr_eval(model, batches, samples, vocab, sampled_count=20):
    lines = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for j, ((raw, mask, labels), sample) in enumerate(zip(batches, samples)):
            torch.manual_seed(1042+j*100)
            _, mu, lv, lm = encoded(model, raw, mask); variants=[]
            for k in range(-1, sampled_count):
                z = mu if k < 0 else mu+torch.randn_like(mu)*(.5*lv).exp()
                logits=model.ocr_model(z)[:int(lm.sum()), 0]
                decoded=greedy_ctc(logits.argmax(-1).tolist(), vocab)
                variants.append(dict(kind='mu' if k < 0 else f'z-{k}', decoded=decoded,
                                     errors=edit_distance(sample[2], decoded), characters=len(sample[2]),
                                     ctc=float(model.get_ocr_loss(z, labels, lm))))
            lines.append(dict(text=sample[2], variants=variants))
    out=dict(lines=lines)
    for name, variants in [('mu',[l['variants'][0] for l in lines]),('sampled',[v for l in lines for v in l['variants'][1:]])]:
        out[name]=dict(cer=sum(v['errors'] for v in variants)/sum(v['characters'] for v in variants),
                       exact=sum(v['errors']==0 for v in variants), count=len(variants),
                       mean_ctc=float(np.mean([v['ctc'] for v in variants])))
    return out


def fixed_eval(model, samples, batches, cfg, folder, step, vocab, with_ocr=False):
    row = snapshot(model, samples, [(b[0], b[1]) for b in batches], cfg, folder, step, sampled_count=20)
    row['summary'] = summary(row)
    with torch.no_grad():
        stats=[]
        for raw, mask, _ in batches:
            _, mu, lv, lm=encoded(model,raw,mask)
            std=(.5*lv).exp()[lm[:,None].expand_as(lv)]
            stats.append(dict(kl=float(model.kl_divergence_new(mu,lv,lm)), std_median=float(std.median()),std_mean=float(std.mean())))
    row['posterior']=stats
    if with_ocr: row['ocr']=ocr_eval(model,batches,samples,vocab)
    (folder/f'eval-{step}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def calibration(model, batches, use_ocr=False):
    params=list(model.encoder.parameters())+list(model.conv_mu.parameters())
    totals={k:[torch.zeros_like(p) for p in params] for k in ('geometry','pen','ctc','kl')}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(2042)
        for batch in batches:
            t=terms(model,batch,use_ocr)
            objectives=dict(geometry=100*t['sampled_geometry']+MEAN_WEIGHT*t['mean_geometry'],pen=t['pen'],ctc=t['ctc'],kl=t['kl'])
            for name,loss in objectives.items():
                gs=torch.autograd.grad(loss/len(batches),params,retain_graph=True,allow_unused=True)
                for acc,g in zip(totals[name],gs):
                    if g is not None: acc.add_(g.detach())
    norms={k:float(torch.stack([g.square().sum() for g in gs]).sum().sqrt()) for k,gs in totals.items()}
    # Modest auxiliary ENCODER gradient. Pen has its own head; never amplify
    # already-tiny pen losses merely to force an arbitrary fraction.
    pen_weight=min(1., .1*norms['geometry']/max(norms['pen'],1e-12))
    ctc_weight=min(.1, .1*norms['geometry']/max(norms['ctc'],1e-12)) if use_ocr else 0.
    return dict(encoder_gradient_norms=norms,pen_weight=pen_weight,ctc_weight=ctc_weight,
                pen_encoder_fraction=pen_weight*norms['pen']/norms['geometry'],
                ctc_encoder_fraction=ctc_weight*norms['ctc']/norms['geometry'],
                kl_weight=1e-6,method='fixed initial full-eight-line encoder gradient calibration; caps pen1/CTC.1')


def warm_ocr(model,batches,samples,vocab,folder,max_steps=500):
    """Repair old CER-zero head on NEW encoder, without touching geometry."""
    from utils.mask import downsample_mask
    before={k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('ocr_model.')}
    model.requires_grad_(False);model.ocr_model.requires_grad_(True)
    cache=[]
    with torch.no_grad():
        for raw,mask,labels in batches:
            _,mu,lv,lm=encoded(model,raw,mask);cache.append((mu,lv,labels,lm))
    optimizer=torch.optim.AdamW(model.ocr_model.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=0)
    initial=ocr_eval(model,batches,samples,vocab);history=[dict(step=0,mu=initial['mu'],sampled=initial['sampled'])]
    torch.manual_seed(3042);step=0
    if initial['mu']['cer'] or initial['sampled']['cer']:
        for step in range(1,max_steps+1):
            optimizer.zero_grad(set_to_none=True)
            for mu,lv,labels,lm in cache:
                z=mu+torch.randn_like(mu)*(.5*lv).exp()
                loss=(model.get_ocr_loss(mu,labels,lm)+model.get_ocr_loss(z,labels,lm))/(2*len(cache))
                if not torch.isfinite(loss): raise FloatingPointError('CTC warmup nonfinite')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ocr_model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            if step%50==0:
                row=ocr_eval(model,batches,samples,vocab);history.append(dict(step=step,mu=row['mu'],sampled=row['sampled']))
                print(dict(stage='ocr_warm',**history[-1]),flush=True)
                if row['mu']['cer']==row['sampled']['cer']==0:break
    final=ocr_eval(model,batches,samples,vocab)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
    result=dict(initial=initial,final=final,updates=step,geometry_bitwise_unchanged=True,history=history)
    folder.mkdir();(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(dict(ocr_state_dict=model.ocr_model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    updates=step, geometry_bitwise_unchanged=True), folder/'head-checkpoint.pt')
    return result


def run(config,repo,root='/data',steps=400,lr=5e-7,source_rel=SOURCE_REL,source_sha=SOURCE_SHA,stages=('sampled','kl','ocr')):
    if not 1<=steps<=1000 or not 0<lr<=1e-5 or not stages or any(s not in ('sampled','kl','ocr') for s in stages):
        raise ValueError('bounded stages/steps/LR required')
    if not torch.cuda.is_available(): raise RuntimeError('T4 required')
    torch.set_num_threads(4);torch.manual_seed(42)
    root=Path(root);source=root/source_rel
    if file_sha(source)!=source_sha:raise ValueError('pinned source SHA mismatch')
    model,samples,raw_batches,cfg,hashes,_=load(config,repo,root/'diffink/iam_overfit',root/'checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    parent=torch.load(source,map_location='cpu',weights_only=True)
    if parent['sample_ids']!=cfg['sample_ids']:raise ValueError('wrong sample identities')
    model.load_state_dict(parent['model_state_dict'],strict=True);model.apply_checkpoint_contract(parent);model.to('cuda').eval()
    protected={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith('style_classifier.')}
    cfg.update(profile='bounded-posterior-integration',model_input_scale=.01,trans_dropout=0,gmm_weight=0,style_weight=0,
               expected_xy_weight=100.,mean_xy_anchor_weight=MEAN_WEIGHT,target_delta_weight=DELTA_WEIGHT,
               rotation_degrees=0,training_latent='sampled plus mean anchor',pen_policy='bounded_three_state',
               max_optimizer_updates=steps,base_lr=lr,physical_batch=1,gradient_accumulation_steps=8,
               grad_clip=5,weight_decay=0,betas=[.9,.99],source_checkpoint=str(source),source_sha256=source_sha,
               sampler_eval_seeds='1042+100*line index;20 draws',optimizer='fresh AdamW each stage',
               **runtime_metadata())
    model.config.__dict__.update(cfg)
    batches=[(b[0].to('cuda').transpose(1,2),b[1].to('cuda'),b[2].to('cuda')) for b in raw_batches]
    vocab=list(json.loads(Path(cfg['text_file']).read_text()))
    directory=root/'checkpoints/iam_latent_integration'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    for relative in ('model/vae.py','model/ocr.py','model/losses.py'):
        dest=directory/'source-code'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path(repo)/relative).read_bytes())
    dest=directory/'source-code/iam_tools/latent_integration.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).read_bytes())
    refdir=directory/'reference';refdir.mkdir()
    reference=fixed_eval(model,samples,batches,cfg,refdir,0,vocab)
    outputs={};reason='all_requested_stages_completed'
    for stage in stages:
        use_ocr=stage=='ocr'
        if use_ocr:
            old=root/OCR_REL
            if file_sha(old)!=OCR_SHA:raise ValueError('pinned OCR head SHA required')
            saved=torch.load(old,map_location='cpu',weights_only=True)
            model.ocr_model.load_state_dict({k.removeprefix('ocr_model.'):v for k,v in saved['model_state_dict'].items() if k.startswith('ocr_model.')})
            warm=warm_ocr(model,batches,samples,vocab,directory/'ocr_warm')
            if warm['final']['mu']['cer'] or warm['final']['sampled']['cer']:
                reason='OCR_warmup_failed';break
        model.requires_grad_(True);model.style_classifier.requires_grad_(False);model.ocr_model.requires_grad_(use_ocr);model.eval()
        hooks=[]
        for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
            active=torch.zeros_like(p);active[:63]=1
            hooks.append(p.register_hook(lambda g,m=active:g*m))
        folder=directory/stage;folder.mkdir()
        initial=fixed_eval(model,samples,batches,cfg,folder,0,vocab,use_ocr)
        cal=calibration(model,batches,use_ocr);kl_weight=1e-6 if stage in ('kl','ocr') else 0
        stagecfg=dict(cfg,stage=stage,pen_weight=cal['pen_weight'],ctc_weight=cal['ctc_weight'],kl_weight=kl_weight,
                      training_seed=4042,ocr_dropout='eval/off',max_wall_seconds=900,eval_every=100)
        (folder/'config.json').write_text(json.dumps(stagecfg,indent=2)+'\n');(folder/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n')
        print(dict(stage=stage,calibration=cal),flush=True)
        geom=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('ocr_model.')]
        groups=[dict(params=geom,lr=lr)]
        if use_ocr:groups.append(dict(params=list(model.ocr_model.parameters()),lr=1e-5))
        optimizer=torch.optim.AdamW(groups,betas=(.9,.99),weight_decay=0)
        torch.manual_seed(4042);rng=np.random.default_rng(42);started=time.monotonic();history=[];latest=initial
        best_score=float('inf');best_step=None
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=stagecfg,
                            optimizer_updates=step,sample_ids=cfg['sample_ids'],source_sha256=source_sha,
                            rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),folder/name)
        save('checkpoint-initial.pt',0)
        with (folder/'metrics.jsonl').open('w') as log:
            for step in range(1,steps+1):
                optimizer.zero_grad(set_to_none=True);totals={k:0. for k in ('sampled_geometry','mean_geometry','pen','kl','ctc')}
                if step==int(.75*steps):
                    for group in optimizer.param_groups:group['lr']*=.3
                for j in rng.permutation(8):
                    t=terms(model,batches[int(j)],use_ocr)
                    loss=(100*t['sampled_geometry']+MEAN_WEIGHT*t['mean_geometry']+cal['pen_weight']*t['pen']+kl_weight*t['kl']+cal['ctc_weight']*t['ctc'])/8
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
                    loss.backward()
                    for key in totals:totals[key]+=float(t[key].detach())/8
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True));optimizer.step()
                row=dict(step=step,**totals,raw_gradient_norm=norm,was_clipped=norm>5,learning_rates=[g['lr'] for g in optimizer.param_groups])
                log.write(json.dumps(row)+'\n');log.flush()
                if step%100==0 or step==steps:
                    latest=fixed_eval(model,samples,batches,cfg,folder,step,vocab,use_ocr)
                    gate=geometry_gate(latest,reference)
                    if use_ocr and (latest['ocr']['mu']['cer'] or latest['ocr']['sampled']['cer']):
                        gate['passed']=False;gate['failures'].append('OCR_CER_nonzero')
                    score=latest['summary']['sampled']['mean_per_line_first_difference_relative']
                    history.append(dict(step=step,gate=gate,summary=latest['summary'],ocr=latest.get('ocr')))
                    save(f'checkpoint-{step}.pt',step)
                    if gate['passed'] and score<best_score:
                        best_score=score;best_step=step;save('checkpoint-best.pt',step)
                    print(dict(stage=stage,step=step,gate=gate,summary=latest['summary'],ocr=latest.get('ocr',{}).get('sampled')),flush=True)
                    if not gate['passed'] and step>=200:
                        reason=stage+'_fidelity_gate_failed';break
                if time.monotonic()-started>900:reason=stage+'_wall_limit';break
        for h in hooks:h.remove()
        save('checkpoint-last.pt',step)
        selected=folder/'checkpoint-best.pt' if best_step is not None else folder/'checkpoint-initial.pt'
        selected_state=torch.load(selected,map_location='cpu',weights_only=True);model.load_state_dict(selected_state['model_state_dict'])
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in protected.items())
        result=dict(stage=stage,config=stagecfg,calibration=cal,initial=initial,history=history,
                    last_step=step,selected_step=best_step,selected_checkpoint=str(selected),selected_sha256=file_sha(selected),
                    gate_passed=best_step is not None,elapsed_seconds=time.monotonic()-started,
                    source_unchanged=file_sha(source)==source_sha,style_unchanged=True)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');outputs[stage]=dict(output=str(folder),selected_step=best_step,selected_checkpoint=str(selected),gate_passed=best_step is not None)
        (directory/'study.json').write_text(json.dumps(dict(outputs=outputs,reference=str(refdir),source_sha256=source_sha,provenance=hashes,stop_reason=reason),indent=2)+'\n')
        if best_step is None or reason!='all_requested_stages_completed':break
    result=dict(output=str(directory),outputs=outputs,stop_reason=reason,source_sha256=source_sha,reference_summary=reference['summary'])
    (directory/'study.json').write_text(json.dumps(dict(result,provenance=hashes),indent=2)+'\n')
    return result
