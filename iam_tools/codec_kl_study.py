"""Matched KL continuations of the faithful initialized transport codec.

No geometry reset, fresh optimizer, OCR, style, GMM or dropout. This tests a
prior/reconstruction trade-off, NOT whether these latents are generative-ready.
"""
import copy
import json
import time
from pathlib import Path
import torch
from .writer_expansion import load,device_batches,evaluate,training_schedule,train_score
from .initialization_study import optimizer_groups,PEN
from .fast_geometry import GeometryGraphs,fast_terms
from .latent_integration import encoded
from .pen_ab import file_sha

SOURCE='checkpoints/iam_initialization_study/20261006-170142/protected_noise/checkpoint-best.pt'
SHA='5ba90c388946e7a19693b8cb578c0218686ddb0637eebcecb36c5705d68ae948'
ARMS={'kl0':0.,'kl1e-6':1e-6,'kl1e-5':1e-5}


def restore_optimizer(model,parent):
    optimizer=torch.optim.AdamW(optimizer_groups(model,'protected_noise'),betas=(.9,.99),weight_decay=0)
    groups=parent['optimizer_state_dict']['param_groups']
    if [(g['name'],len(g['params']),g['lr']) for g in groups] != [(g['name'],len(g['params']),g['lr']) for g in optimizer.param_groups]:
        raise ValueError('source optimizer parameter order/group contract mismatch')
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer_state_dict']))
    return optimizer


def candidate_valid(row):
    """Train-only mandatory boundary/EOC gate; pen-up F1 alone misses EOCs."""
    g=row['groups']['train']
    return (g['all_final_eoc_correct'] and g['false_internal_eoc']==0
            and g['mu_min_pen_f1']==1. and g['sampled_min_pen_f1']==1.)


def protect_pen_variance_weights(model,optimizer):
    """Freeze24 routed pen weight rows, retain bias learning and other moments.

    Masking gradients alone would NOT stop restored Adam momentum. Zero those
    rows' moments as well. Requires wd0; geometry/unused rows stay unmodified.
    """
    p=model.conv_logvar.weight
    if p.shape[0]<40 or any(g.get('weight_decay',0)!=0 for g in optimizer.param_groups):
        raise ValueError('initialized40-field posterior and zero weight decay required')
    index=torch.arange(p.shape[0],device=p.device);rows=(index<40)&(index%5>=2)
    mask=torch.ones_like(p);mask[rows]=0.
    before=p.detach()[rows].clone();reset={}
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if key in optimizer.state[p]:
            value=optimizer.state[p][key];reset[key]=float(value[rows].abs().max());value[rows]=0.
    hook=p.register_hook(lambda grad:grad*mask)
    return rows,before,hook,reset


@torch.no_grad()
def posterior_stats(model,batches,splits):
    rows={}
    for sid,(raw,mask,_) in batches.items():
        _,mu,lv,lm=encoded(model,raw,mask)
        valid=lm[:,None].expand_as(mu);c=torch.arange(mu.shape[1],device=mu.device)[None,:,None]
        subsets={'xy':valid&(c<40)&(c%5<2),'pen':valid&(c<40)&(c%5>=2),'unused':valid&(c>=40)}
        kl=-.5*(1+lv-mu.square()-lv.exp());std=(.5*lv).exp()
        rows[sid]={'kl_per_element':float(model.kl_divergence_new(mu,lv,lm))}
        for name,m in subsets.items():
            rows[sid][name]=dict(std_median=float(std[m].median()),std_mean=float(std[m].mean()),
                mu_rms=float(mu[m].square().mean().sqrt()),kl_contribution_per_total_element=float(kl[m].sum()/valid.sum()))
    groups={}
    for name,ids in splits.items():
        groups[name]={'kl_per_element':sum(rows[i]['kl_per_element'] for i in ids)/len(ids)}
        for subset in ('xy','pen','unused'):
            groups[name][subset]={k:sum(rows[i][subset][k] for i in ids)/len(ids) for k in rows[ids[0]][subset]}
    return dict(groups=groups,lines=rows,definition='arithmetic mean of per-line statistics; KL per valid channel×time element; initialized polyphase XY16/pen24/unused344 channels')


def capture_parity(model,graphs,batches,ids,kl_weight):
    """Fixed noise; eager reference uses core Boolean-indexed corrected KL."""
    from model.vae import VAE
    from utils.mask import downsample_mask
    params=graphs.parameters;weights=torch.tensor([1.,.1,PEN]+([kl_weight] if kl_weight else []),device='cuda')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(7042)
        noises=[torch.randn(1,model.config.latent_dim,batches[i][0].shape[-1]//8,device='cuda') for i in ids]
        model.zero_grad(set_to_none=False);reference=torch.zeros_like(weights)
        kl_diffs=[]
        for sid,noise in zip(ids,noises):
            raw,mask,_=batches[sid];values=fast_terms(model,raw,mask,noise,include_kl=bool(kl_weight))
            if kl_weight:
                features=model.encoder(model.to_model_space(raw));mu=model.conv_mu(features);lv=model.conv_logvar(features)
                core=VAE.kl_divergence_new(model,mu,lv,downsample_mask(mask,8))
                kl_diffs.append(float((values[-1]-core).abs().detach()))
                values=torch.cat((values[:3],core[None]))
            (values*weights/len(ids)).sum().backward();reference+=values.detach()/len(ids)
        expected=[p.grad.clone() for p in params]
        model.zero_grad(set_to_none=False);actual=torch.zeros_like(weights)
        for sid,noise in zip(ids,noises):actual+=graphs.replay(batches[sid],divisor=len(ids),epsilon=noise)/len(ids)
        diff=torch.stack([(p.grad-g).square().sum() for p,g in zip(params,expected)]).sum().sqrt()
        norm=torch.stack([g.square().sum() for g in expected]).sum().sqrt()
        relative=float(diff/norm.clamp_min(1e-20));values_diff=float((reference-actual).abs().max())
        result=dict(relative_gradient_l2=relative,max_term_difference=values_diff,max_core_kl_difference=max(kl_diffs,default=0.),sample_ids=ids,seed=7042,no_optimizer_updates=True)
        if relative>1e-4 or values_diff>1e-5 or result['max_core_kl_difference']>1e-5:raise AssertionError(result)
        model.zero_grad(set_to_none=False)
    return result


def run(config,repo,root='/data',steps=200,metric_pool=None,protected_pen=False):
    if not torch.cuda.is_available() or not 1<=steps<=300:raise ValueError('CUDA and 1–300 continuation updates required')
    torch.set_num_threads(4);root=Path(root)
    base,samples,raw,cfg,vocab,provenance=load(config,repo,root,SOURCE,SHA,writer_id=None)
    parent=torch.load(root/SOURCE,map_location='cpu',weights_only=True)
    if parent['optimizer_updates']!=200 or parent['config']['initialization_mode']!='protected_noise':raise ValueError('protected update200 source required')
    initial=copy.deepcopy(base.state_dict());splits=provenance['splits'];batches=device_batches(raw,'cuda')
    directory=root/'checkpoints/iam_codec_kl_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    results={};schedule=list(training_schedule(splits['train'],200+steps))[200:]
    arms={'pen_bias_kl1e-6':1e-6} if protected_pen else ARMS
    for arm,weight in arms.items():
        model=copy.deepcopy(base).cuda().eval();out=directory/arm;out.mkdir()
        settings=dict(parent['config'],profile='initialized-codec-kl-continuation',source_checkpoint=str(root/SOURCE),source_sha256=SHA,
            output_base=str(directory),kl_weight=weight,ctc_weight=0.,style_weight=0.,gmm_weight=0.,source_optimizer_updates=200,
            max_optimizer_updates=steps,optimizer='restored AdamW moments/groups/steps',seed='restored CPU/CUDA checkpoint RNG',
            training_schedule='seed42; skip first200 effective batches; paired across arms',eval_steps=[0,100,steps],
            stop_gate='train shuffled mean geometry >1e-4 after10 continuation updates; finite gradients; no held-out selection',
            torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
            posterior_pen_policy='frozen feature weights; trainable bias' if protected_pen else 'full affine logvar',
            checkpoint_selection='train_score AND exact training mean/sampled pen/F1/EOC eligibility',
            research_contract='initialized transport; compatibility study, NOT semantic/generative readiness')
        model.config.__dict__.update(settings);optimizer=restore_optimizer(model,parent)
        protected=None
        if protected_pen:
            protected=protect_pen_variance_weights(model,optimizer);settings['protected_pen_moment_reset_max']=protected[3]
        graphs=GeometryGraphs(model,PEN,.1,kl_weight=weight)
        t=time.monotonic();graphs.prepare([batches[i] for i in splits['train']])
        settings.update(capture_seconds=time.monotonic()-t,captured_lengths=list(graphs.cache))
        parity=capture_parity(model,graphs,batches,splits['old'],weight)
        (out/'capture-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        for name in ('codec_kl_study.py','fast_geometry.py','writer_expansion.py','latent_integration.py','trajectory_geometry.py','initialization_study.py'):
            p=out/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
        for name in ('vae.py','blocks.py','losses.py'):
            p=out/'source-code/model'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model'/name).read_bytes())
        torch.set_rng_state(parent['rng_state_cpu']);torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=settings,
                sample_ids=splits['train'],optimizer_updates=200+step,continuation_updates=step,source_sha256=SHA,provenance=provenance,
                rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),out/name)
        def inspect(step):
            row=evaluate(model,samples,batches,splits,vocab,out,step,metric_pool=metric_pool,draw_batch_size=21)
            (out/f'posterior-{step}.json').write_text(json.dumps(posterior_stats(model,batches,splits),indent=2)+'\n')
            return row
        first=inspect(0)
        if not candidate_valid(first):raise AssertionError('source fails training boundary/EOC gate')
        best_score=train_score(first);best_step=0;save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        history=[];stop='budget_completed';started=time.monotonic()
        with (out/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(schedule,1):
                optimizer.zero_grad(set_to_none=False);total=torch.zeros(4 if weight else 3,device='cuda')
                for sid in ids:total+=graphs.replay(batches[sid],divisor=8)/8
                if not torch.isfinite(total).all():raise FloatingPointError('nonfinite terms: no update applied')
                norm=float(torch.nn.utils.clip_grad_norm_(graphs.parameters,5,error_if_nonfinite=True));optimizer.step()
                values=total.cpu().tolist()
                log.write(json.dumps(dict(step=step,absolute_step=step+200,sample_ids=ids,mean_geometry=values[0],sampled_geometry=values[1],pen=values[2],
                    kl=values[3] if weight else None,raw_gradient_norm=norm,was_clipped=norm>5))+'\n');log.flush()
                broken=step>=10 and values[0]>1e-4;limit=time.monotonic()-started>600
                if step%100==0 or step==steps or broken or limit:
                    row=inspect(step);history.append(dict(step=step,score=train_score(row),candidate_valid=candidate_valid(row),groups=row['groups']))
                    save(f'checkpoint-{step}.pt',step)
                    if candidate_valid(row) and train_score(row)<best_score:best_score=train_score(row);best_step=step;save('checkpoint-best.pt',step)
                if broken:stop='train_geometry_gate';break
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        unchanged=all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith(('transformer_decoder.','ocr_model.','style_classifier.')))
        assert unchanged and file_sha(root/SOURCE)==SHA
        if protected is not None:
            assert torch.equal(model.conv_logvar.weight.detach()[protected[0]],protected[1])
            protected[2].remove()
        result=dict(output=str(out),kl_weight=weight,initial=first['groups'],history=history,last_step=step,best_step=best_step,stop_reason=stop,
            elapsed_seconds=time.monotonic()-started,protected_pen_weights_unchanged=protected is not None,readout_ocr_style_unchanged=unchanged,source_unchanged=True,not_promoted=True,
            selected_sha256=file_sha(out/'checkpoint-best.pt'),final_sha256=file_sha(out/'checkpoint-last.pt'))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result
        del model,optimizer,graphs;torch.cuda.empty_cache()
    (directory/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(directory),arms={k:dict(last_step=v['last_step'],best_step=v['best_step'],stop_reason=v['stop_reason']) for k,v in results.items()})


def verify(directory,repo,root='/data'):
    """Independent final CPU reload + frozen/source checks and posterior metrics."""
    import numpy as np
    from .curve_study import forward_xy
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    saved=torch.load(directory/'checkpoint-last.pt',map_location='cpu',weights_only=True)
    parent=torch.load(root/SOURCE,map_location='cpu',weights_only=True)
    sha=file_sha(directory/'checkpoint-last.pt');step=saved['continuation_updates']
    model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,str((directory/'checkpoint-last.pt').relative_to(root)),sha,writer_id=None)
    original=json.loads((directory/'provenance.json').read_text())
    if original['samples']!=prov['samples'] or original['splits']!=prov['splits']:raise AssertionError('split/sample drift')
    batches=device_batches(raw,'cpu');rows=[]
    with torch.no_grad():
        for sid,(x,mask,_) in batches.items():
            xy,_,out=forward_xy(model,x,mask);n=int(mask.sum());a=np.load(directory/f'step-{step}/{sid}/mu.npy')
            rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(xy[0,:n].numpy()-a[:,:2]).max()),
                pen_mismatches=int((out[0,:3,:n].argmax(0).numpy()!=a[:,2:].argmax(1)).sum())))
    unchanged=all(torch.equal(v,parent['model_state_dict'][k]) for k,v in model.state_dict().items() if k.startswith(('transformer_decoder.','ocr_model.','style_classifier.')))
    check=dict(final_sha256=sha,final_step=step,max_cpu_gpu_xy_difference=max(r['max_xy_difference'] for r in rows),
        pen_mismatches=sum(r['pen_mismatches'] for r in rows),readout_ocr_style_unchanged=unchanged,source_unchanged=file_sha(root/SOURCE)==SHA,
        finite=all(torch.isfinite(v).all().item() for v in model.state_dict().values()),lines=rows)
    if check['max_cpu_gpu_xy_difference']>1e-4 or check['pen_mismatches'] or not all(check[k] for k in ('readout_ocr_style_unchanged','source_unchanged','finite')):raise AssertionError(check)
    if saved['config'].get('posterior_pen_policy')=='frozen feature weights; trainable bias':
        index=torch.arange(model.conv_logvar.weight.shape[0]);rows=(index<40)&(index%5>=2)
        check['protected_pen_weights_unchanged']=torch.equal(model.conv_logvar.weight.detach()[rows],parent['model_state_dict']['conv_logvar.weight'][rows])
        if not check['protected_pen_weights_unchanged']:raise AssertionError('protected variance rows drifted')
    (directory/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n');return check


def packed_std_to_points(std):
    """Inspect the initializer's first40 polyphase fields, NOT decoder Jacobian."""
    if std.ndim!=3 or std.shape[0]!=1 or std.shape[1]<40:raise ValueError('one-line initialized40-channel route required')
    return std[0,:40].transpose(0,1).reshape(-1,5)


def tail_audit(directory,repo,root='/data'):
    """CPU-only posterior tail/failure audit; no optimization or posterior draws."""
    import numpy as np
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());audits={}
    for arm,r in results.items():
        folder=directory/arm;sha=r['final_sha256'];step=r['last_step']
        model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,
            str((folder/'checkpoint-last.pt').relative_to(root)),sha,writer_id=None)
        batches=device_batches(raw,'cpu');evaluation=json.loads((folder/f'eval-{step}.json').read_text());rows={};failures=[]
        with torch.no_grad():
            for line in evaluation['lines']:
                sid=line['sample_id'];x,mask,_=batches[sid];n=int(mask.sum());_,mu,lv,lm=encoded(model,x,mask)
                std=packed_std_to_points((.5*lv).exp())[:n];states=x[0,2:,:n].argmax(0).numpy();xy=x[0,:2,:n].T*.01
                penstd=std[:,2:].flatten()
                rows[sid]=dict(pen_std_p90=float(penstd.quantile(.9)),pen_std_p99=float(penstd.quantile(.99)),pen_std_max=float(penstd.max()))
                for v in line['sampled']:
                    if v['pen']['pen_up_f1']==1 and v['pen']['final_eoc_correct'] and not v['pen']['non_final_false_eoc_count']:continue
                    array=np.load(folder/f'step-{step}/{sid}/{v["kind"]}.npy');pred=array[:,2:].argmax(1)
                    indices=np.flatnonzero(states!=pred)
                    failures.append(dict(sample_id=sid,kind=v['kind'],pen=v['pen'],points=[dict(point_index=int(i),index_fraction=float(i/max(n-1,1)),
                        target_state=int(states[i]),predicted_state=int(pred[i]),target_xy=xy[i].tolist(),
                        routed_pen_std=std[i,2:].tolist(),routed_xy_std=std[i,:2].tolist()) for i in indices]))
        audits[arm]=dict(lines=rows,failures=failures,failing_draws=len(failures),
            train_max_routed_pen_std=max(rows[i]['pen_std_max'] for i in prov['splits']['train']),final_sha256=sha,
            caveat='First40 polyphase-channel posterior std. Learned body may introduce cross-field coupling; this is NOT an exact propagated output variance or causal intervention.')
    output=dict(arms=audits,source_sha256=SHA,definition='CPU posterior diagnostics, no optimization; every failing final draw, not selected by RMSE or pen F1 alone')
    (directory/'posterior-tail-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    from .report_codec_kl import tail_report
    tail_report(directory,root,output)
    return dict(output=str(directory/'posterior-tail-audit.json'),arms={a:dict(failing_draws=v['failing_draws'],max_routed_pen_std=v['train_max_routed_pen_std']) for a,v in audits.items()})
