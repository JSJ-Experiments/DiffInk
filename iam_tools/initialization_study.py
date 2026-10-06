"""Controlled, opt-in initialized-geometry optimizer stability research.

Not paper reproduction, semantic-latent readiness, or a production promotion.
Source checkpoint supplies identical dormant weights/OCR; geometry is explicitly
reinitialized, then all arms use the same train-only schedule and posterior RNG.
"""
import copy
import json
import math
import time
from pathlib import Path
import torch
from .identity_geometry_probe import initialize_identity_geometry
from .writer_expansion import load, device_batches, evaluate, training_schedule, train_score
from .fast_geometry import GeometryGraphs
from .pen_ab import file_sha

SOURCE='checkpoints/iam_fullset_joint/20261006-140945/checkpoint-best.pt'
SHA='89ec459de6496275a3712c08629daea10d8f4f03311712c77f49e652bca915a3'
PEN=.02099049935353879
MODES=('uniform','scaled_readout','frozen_readout','protected_noise')


def optimizer_groups(model, mode, lr=5e-5, scale=512.):
    """Freeze only specified modules; lower LR is an actual Adam displacement LR.

    Scaling gradients would largely cancel inside Adam and is not equivalent.
    Sigma/rho readout rows remain protected by a separately returned hook mask.
    """
    if mode not in MODES or not 0<lr<=5e-5 or not math.isfinite(scale) or scale<128:
        raise ValueError('known bounded initialization arm required')
    model.requires_grad_(True)
    model.ocr_model.requires_grad_(False);model.style_classifier.requires_grad_(False)
    if mode in ('frozen_readout','protected_noise'):model.transformer_decoder.requires_grad_(False)
    groups=[]
    for label in ('body','readout','posterior'):
        params=[]
        for name,p in model.named_parameters():
            category='readout' if name.startswith('transformer_decoder.') else 'posterior' if name.startswith('conv_logvar.') else 'body'
            if p.requires_grad and category==label:params.append(p)
        if params:
            rate=lr
            if label=='readout' and mode=='scaled_readout':rate/=scale
            if mode=='protected_noise':rate=lr*20 if label=='posterior' else lr/500
            groups.append(dict(params=params,lr=rate,name=label))
    return groups


def run(config, repo, root='/data', steps=100, modes=MODES[:3], metric_pool=None):
    if not torch.cuda.is_available():raise RuntimeError('T4 required')
    if not 1<=steps<=300 or not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):
        raise ValueError('bounded unique initializer arms required')
    torch.set_num_threads(4)
    base,samples,raw,cfg,vocab,provenance=load(config,repo,root,SOURCE,SHA,writer_id=None)
    init=initialize_identity_geometry(base,std=.002)
    provenance.update(initialization=init,source_role='explicit geometry reinitialization; dormant weights and frozen OCR/style from source; NOT continuation')
    initial=copy.deepcopy(base.state_dict());splits=provenance['splits'];root=Path(root)
    directory=root/'checkpoints/iam_initialization_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    directory.mkdir(parents=True,exist_ok=False);results={}
    batches=device_batches(raw,'cuda')
    for mode in modes:
        model=copy.deepcopy(base).cuda().eval();arm=directory/mode;arm.mkdir()
        settings=dict(cfg,profile='initialization-stability',initialization=init,initialization_mode=mode,
                      source_checkpoint=str(root/SOURCE),source_sha256=SHA,output_base=str(directory),max_wall_seconds=600,
                      torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                      optimizer='fresh AdamW',base_lr=5e-5,readout_lr=5e-5/512 if mode=='scaled_readout' else 0. if mode in ('frozen_readout','protected_noise') else 5e-5,
                      body_lr=1e-7 if mode=='protected_noise' else 5e-5,posterior_lr=1e-3 if mode=='protected_noise' else 5e-5,
                      betas=[.9,.99],weight_decay=0,grad_clip=5,physical_batch=1,gradient_accumulation_steps=8,
                      seed=4042,max_optimizer_updates=steps,pen_weight=PEN,mean_xy_anchor_weight=1.,
                      expected_xy_weight=.1,kl_weight=0.,ctc_weight=0.,style_weight=0.,gmm_weight=0.,
                      target_delta_weight=.20471838744633777,eval_draws=20,eval_draw_batch_size=1,
                      stop_gate='after update10: shuffled mean geometry >1e-3; no held-out stop/selection',
                      research_contract='handcrafted identity initialization; not learned semantics or DiT-ready',
                      checkpoint_selection='train_score only, but final evaluated even if initial is best')
        model.config.__dict__.update(settings)
        # Validate the NEW initialized checkpoint before enabling optional same-
        # line posterior batching. Never combine different physical lengths.
        from .writer_expansion import posterior_outputs
        from .latent_integration import encoded
        import numpy as np
        checks=[]
        for sid in splits['old']:
            x,mask,_=batches[sid]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                _,mu,lv,lm=encoded(model,x,mask);n=int(mask.sum())
                torch.manual_seed(9901);a=list(posterior_outputs(model,mu,lv,mask,n,int(lm.sum()),20,1))
                torch.manual_seed(9901);b=list(posterior_outputs(model,mu,lv,mask,n,int(lm.sum()),20,21))
            difference=max(float(np.abs(v[1]-w[1]).max()) for v,w in zip(a,b))
            pen_mismatch=sum(int((v[2]!=w[2]).sum()) for v,w in zip(a,b))
            ocr_mismatch=sum(v[3]!=w[3] for v,w in zip(a,b))
            checks.append(dict(sample_id=sid,max_xy_difference=difference,pen_mismatches=pen_mismatch,ocr_mismatches=ocr_mismatch))
        if any(c['max_xy_difference']>1e-4 or c['pen_mismatches'] or c['ocr_mismatches'] for c in checks):
            raise AssertionError('initialized posterior batching parity failed')
        (arm/'posterior-batching-parity.json').write_text(json.dumps(checks,indent=2)+'\n')
        settings['eval_draw_batch_size']=21

        (arm/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        (arm/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        for name in ('initialization_study.py','identity_geometry_probe.py','fast_geometry.py','writer_expansion.py','trajectory_geometry.py','latent_integration.py'):
            p=arm/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
        for name in ('vae.py','blocks.py','losses.py'):
            p=arm/'source-code/model'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model'/name).read_bytes())
        optimizer=torch.optim.AdamW(optimizer_groups(model,mode),betas=(.9,.99),weight_decay=0)
        hooks=[]
        if mode not in ('frozen_readout','protected_noise'):
            for p in (model.transformer_decoder.fc.weight,model.transformer_decoder.fc.bias):
                active=torch.zeros_like(p);active[:63]=1
                hooks.append(p.register_hook(lambda g,k=active:g*k))
        graphs=GeometryGraphs(model,PEN,sampled_weight=.1)
        start=time.monotonic();graphs.prepare([batches[sid] for sid in splits['train']])
        settings.update(capture_seconds=time.monotonic()-start,captured_lengths=list(graphs.cache),
                        memory_reserved_gib=torch.cuda.memory_reserved()/2**30)
        (arm/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=settings,
                            sample_ids=splits['train'],optimizer_updates=step,source_sha256=SHA,provenance=provenance,
                            rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),arm/name)
        first=evaluate(model,samples,batches,splits,vocab,arm,0,metric_pool=metric_pool,draw_batch_size=21)
        best_score=train_score(first);best_step=0;save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        torch.manual_seed(4042);started=time.monotonic();history=[];stop='budget_completed'
        with (arm/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(training_schedule(splits['train'],steps),1):
                optimizer.zero_grad(set_to_none=False);total=torch.zeros(3,device='cuda')
                for sid in ids:total+=graphs.replay(batches[sid],divisor=8)/8
                if not torch.isfinite(total).all():raise FloatingPointError('nonfinite geometry, no update applied')
                norm=float(torch.nn.utils.clip_grad_norm_(graphs.parameters,5,error_if_nonfinite=True))
                optimizer.step();values=total.cpu().tolist()
                log.write(json.dumps(dict(step=step,sample_ids=ids,mean_geometry=values[0],sampled_geometry=values[1],pen=values[2],
                                          raw_gradient_norm=norm,was_clipped=norm>5))+'\n');log.flush()
                broken=step>=10 and values[0]>1e-3
                limit=time.monotonic()-started>600
                if step in (1,10,50,100) or step==steps or broken or limit:
                    row=evaluate(model,samples,batches,splits,vocab,arm,step,metric_pool=metric_pool,draw_batch_size=21)
                    history.append(dict(step=step,score=train_score(row),groups=row['groups']))
                    save(f'checkpoint-{step}.pt',step)
                    if train_score(row)<best_score:best_score=train_score(row);best_step=step;save('checkpoint-best.pt',step)
                if broken:stop='train_geometry_gate';break
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        frozen=all(torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items() if k.startswith(('ocr_model.','style_classifier.')))
        sigma=all(torch.equal(initial[k][63:],model.state_dict()[k][63:].cpu()) for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'))
        assert frozen and sigma
        if mode in ('frozen_readout','protected_noise'):assert all(torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items() if k.startswith('transformer_decoder.'))
        lv=model.conv_logvar.bias.detach().cpu()
        result=dict(output=str(arm),mode=mode,initial=first['groups'],initial_step=0,history=history,best_step=best_step,
                    last_step=step,stop_reason=stop,elapsed_seconds=time.monotonic()-started,ocr_style_unchanged=frozen,
                    sigma_rho_unchanged=sigma,active_bias_std_median=float((.5*lv[:40]).exp().median()),
                    selected_sha256=file_sha(arm/'checkpoint-best.pt'),final_sha256=file_sha(arm/'checkpoint-last.pt'),
                    source_unchanged=file_sha(root/SOURCE)==SHA,not_promoted=True)
        assert result['source_unchanged']
        (arm/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[mode]=result
        for h in hooks:h.remove()
        del graphs,optimizer,model;torch.cuda.empty_cache()
    (directory/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(directory),arms={k:dict(last_step=v['last_step'],best_step=v['best_step'],stop_reason=v['stop_reason']) for k,v in results.items()})


def verify(directory,repo,root='/data'):
    """Reload FINAL on CPU; compare initialized rows, not pre-reset sigma rows."""
    import numpy as np
    from .curve_study import forward_xy
    from .latent_integration import encoded
    root=Path(root);directory=Path(directory);torch.set_num_threads(4)
    saved=torch.load(directory/'checkpoint-last.pt',map_location='cpu',weights_only=True)
    initial=torch.load(directory/'checkpoint-initial.pt',map_location='cpu',weights_only=True)
    parent=torch.load(root/SOURCE,map_location='cpu',weights_only=True)
    step=saved['optimizer_updates'];sha=file_sha(directory/'checkpoint-last.pt')
    model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,
                                      str((directory/'checkpoint-last.pt').relative_to(root)),sha,writer_id=None)
    original=json.loads((directory/'provenance.json').read_text())
    if original['samples']!=prov['samples'] or original['splits']!=prov['splits']:raise AssertionError('split/provenance drift')
    batches=device_batches(raw,'cpu');state=model.state_dict();rows=[];posterior={}
    with torch.no_grad():
        for sid,(x,mask,_) in batches.items():
            xy,_,out=forward_xy(model,x,mask);n=int(mask.sum());arr=np.load(directory/f'step-{step}/{sid}/mu.npy')
            rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(xy[0,:n].numpy()-arr[:,:2]).max()),
                             pen_mismatches=int((out[0,:3,:n].argmax(0).numpy()!=arr[:,2:].argmax(1)).sum())))
            _,mu,lv,lm=encoded(model,x,mask)
            valid=lm[:,None].expand_as(mu);active=valid.clone();active[:,40:]=False
            channels=torch.arange(mu.shape[1])[None,:,None]
            xy_active=active & (channels%5<2);pen_active=active & (channels%5>=2)
            inactive=valid&~active;std=(.5*lv).exp()
            def stats(a):return dict(median=float(a.median()),p90=float(a.quantile(.9)),max=float(a.max()))
            posterior[sid]=dict(active_std=stats(std[active]),xy_active_std=stats(std[xy_active]),pen_active_std=stats(std[pen_active]),unused_std=stats(std[inactive]),
                                kl_per_valid_element=float(model.kl_divergence_new(mu,lv,lm)),
                                unused_mu_rms=float(mu[inactive].square().mean().sqrt()))
    frozen=all(torch.equal(v,parent['model_state_dict'][k]) for k,v in state.items() if k.startswith(('ocr_model.','style_classifier.')))
    sigma=all(torch.equal(state[k][63:],initial['model_state_dict'][k][63:]) for k in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'))
    readout=all(torch.equal(v,initial['model_state_dict'][k]) for k,v in state.items() if k.startswith('transformer_decoder.'))
    padding=[]
    with torch.no_grad():
        for sid in prov['splits']['old']:
            x,mask,_=batches[sid];n=int(mask.sum());base_xy,_,base_out=forward_xy(model,x,mask)
            for extra in (32,128):
                longer=torch.zeros(1,5,x.shape[-1]+extra);longer[:,:,:x.shape[-1]]=x;longer[:,4,x.shape[-1]:]=1
                long_mask=torch.arange(longer.shape[-1])[None]<n
                pxy,_,pout=forward_xy(model,longer,long_mask)
                error=(pxy[:,:n]-base_xy[:,:n]).square().mean((0,1)).sqrt()
                padding.append(dict(sample_id=sid,extra_padding=extra,x_rmse_shift=float(error[0]),y_rmse_shift=float(error[1]),
                                    pen_mismatches=int((pout[:,:3,:n].argmax(1)!=base_out[:,:3,:n].argmax(1)).sum())))
    check=dict(padding_sensitivity=padding,final_step=step,final_sha256=sha,max_cpu_gpu_xy_difference=max(r['max_xy_difference'] for r in rows),
               pen_mismatches=sum(r['pen_mismatches'] for r in rows),ocr_style_unchanged=frozen,sigma_rho_unchanged=sigma,
               readout_bitwise_unchanged=readout,finite=all(torch.isfinite(v).all().item() for v in state.values()),
               source_unchanged=file_sha(root/SOURCE)==SHA,lines=rows,posterior_by_line=posterior)
    if check['max_cpu_gpu_xy_difference']>1e-4 or check['pen_mismatches'] or not all(check[k] for k in ('ocr_style_unchanged','sigma_rho_unchanged','finite','source_unchanged')):raise AssertionError(check)
    if saved['config']['initialization_mode'] in ('frozen_readout','protected_noise') and not readout:raise AssertionError('frozen readout drift')
    (directory/'cpu-reload-check.json').write_text(json.dumps(check,indent=2)+'\n')
    return dict(final_step=step,max_cpu_gpu_xy_difference=check['max_cpu_gpu_xy_difference'],pen_mismatches=check['pen_mismatches'])


def update_attribution(directory,repo,root='/data'):
    """One-update parameter-block interventions, NOT an additive attribution."""
    from .curve_study import forward_xy
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    a=torch.load(directory/'checkpoint-initial.pt',map_location='cpu',weights_only=True)['model_state_dict']
    b=torch.load(directory/'checkpoint-1.pt',map_location='cpu',weights_only=True)['model_state_dict']
    model,samples,raw,cfg,vocab,prov=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    batches=device_batches(raw,'cpu');ids=prov['splits']['old'];result={}
    selectors=dict(initial=lambda k:False,all=lambda k:True,body=lambda k:not k.startswith('transformer_decoder.'),
                   readout=lambda k:k.startswith('transformer_decoder.'),feedforward_output=lambda k:'.linear2.' in k,
                   attention_output=lambda k:'.self_attn.out_proj.' in k,projection=lambda k:k.startswith('transformer_decoder.input_proj.'),
                   layernorm=lambda k:k.startswith('transformer_decoder.') and ('.norm1.' in k or '.norm2.' in k),
                   final_fc=lambda k:k.startswith('transformer_decoder.fc.'))
    with torch.no_grad():
        for label,choose in selectors.items():
            model.load_state_dict({k:b[k] if choose(k) else v for k,v in a.items()});errors=[]
            for sid in ids:
                x,mask,_=batches[sid];xy,_,_=forward_xy(model,x,mask)
                e=(xy-model.to_model_space(x)[:,:2].transpose(1,2))[mask]
                errors.append(e.square().mean(0).sqrt().tolist())
            result[label]=dict(mean_line_x_rmse=sum(v[0] for v in errors)/len(errors),mean_line_y_rmse=sum(v[1] for v in errors)/len(errors))
    (directory/'first-update-attribution.json').write_text(json.dumps(result,indent=2)+'\n');return result
