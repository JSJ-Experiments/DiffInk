"""Matched body+posterior versus posterior-only uncertainty calibration.

Initialized polyphase40 contract, NOT a generic semantic VAE. No channel masking.
Mean/fixed-SOURCE-sigma geometry are isolation controls, not robustness gains.
The completed kl_tradeoff sources/reports remain untouched.
"""
import copy,hashlib,json,time
from pathlib import Path
import torch
from .kl_tradeoff import (SOURCE,SHA,validate_budget,evaluation_scope,load_batches,
    source_scales,evaluate,eligibility,selection)
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA,CONTRACT
from .ocr_pool_study import load_pool
from .ocr_convergence import POOL_SHA
from .writer_expansion import load,training_schedule
from .codec_kl_study import restore_optimizer,protect_pen_variance_weights,capture_parity,posterior_stats
from .fast_geometry import GeometryGraphs,fast_terms
from .initialization_study import PEN
from .latent_integration import DELTA_WEIGHT
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha

ARMS={'joint':1e-6,'posterior_only':1e-6}


def clone_state(state):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in state.items()}


def configure_arm(model,optimizer,arm):
    """Call AFTER restoring source optimizer and protecting pen weight moments.

    Keep all source groups/moments/order; Adam skips frozen parameters with
    grad=None. requires_grad=False ALONE does not clear stale gradients.
    Freeze weights, not decoder input autograd: sampled geometry still reaches
    conv_logvar through the entirely frozen decoder.
    """
    if arm not in ARMS:raise ValueError('unknown matched isolation arm')
    if arm=='posterior_only':
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith('conv_logvar.'))
    frozen={n:p for n,p in model.named_parameters() if not p.requires_grad}
    for p in frozen.values():p.grad=None
    params=dict(model.named_parameters())
    return dict(weights={n:p.detach().cpu().clone() for n,p in frozen.items()},
        moments={n:clone_state(optimizer.state.get(p,{})) for n,p in frozen.items()},
        buffers={n:v.detach().cpu().clone() for n,v in model.named_buffers()},
        groups=[dict(name=g['name'],lr=g['lr'],params=[next(n for n,p in params.items() if p is x) for x in g['params']]) for g in optimizer.param_groups])


def assert_frozen_gradients(model,snapshot):
    params=dict(model.named_parameters())
    if any(params[n].requires_grad or params[n].grad is not None for n in snapshot['weights']):
        raise AssertionError('frozen parameter gradient present; Adam could update restored moments')


def verify_frozen(model,optimizer,snapshot):
    assert_frozen_gradients(model,snapshot);params=dict(model.named_parameters())
    for n,v in snapshot['weights'].items():
        if not torch.equal(params[n].detach().cpu(),v):raise AssertionError('frozen weight changed: '+n)
        current=clone_state(optimizer.state.get(params[n],{}));original=snapshot['moments'][n]
        if current.keys()!=original.keys():raise AssertionError('frozen optimizer state keys changed: '+n)
        for k in current:
            same=torch.equal(current[k],original[k]) if torch.is_tensor(current[k]) else current[k]==original[k]
            if not same:raise AssertionError('frozen optimizer moment/step changed: '+n+':'+k)
    for n,v in model.named_buffers():
        if not torch.equal(v.detach().cpu(),snapshot['buffers'][n]):raise AssertionError('codec buffer changed: '+n)
    groups=[dict(name=g['name'],lr=g['lr'],params=[next(n for n,p in params.items() if p is x) for x in g['params']]) for g in optimizer.param_groups]
    if groups!=snapshot['groups']:raise AssertionError('restored optimizer groups changed')
    return dict(frozen_weights_unchanged=True,frozen_moments_steps_unchanged=True,buffers_unchanged=True,
        no_frozen_gradients=True,source_groups_preserved=True,frozen_parameters=len(snapshot['weights']))


def assert_output_isolation(row,reference):
    if row['fixed_source_sigma']['lines']!=reference['fixed_source_sigma']['lines']:
        raise AssertionError('posterior-only fixed SOURCE sigma output/reader changed')
    a={r['sample_id']:r['mu'] for r in row['own']['lines']}
    b={r['sample_id']:r['mu'] for r in reference['own']['lines']}
    if a!=b:raise AssertionError('posterior-only mean output/reader changed')
    return True


def gradient_diagnostics(model,batches,ids):
    """Descriptive per-parameter-group raw KL/geometry gradient norms, not a sweep."""
    params=[p for p in model.parameters() if p.requires_grad];names={id(p):n for n,p in model.named_parameters()}
    sums={k:[torch.zeros_like(p) for p in params] for k in ('geometry','kl')}
    with torch.random.fork_rng(devices=[next(model.parameters()).device.index or 0] if next(model.parameters()).is_cuda else []):
        torch.manual_seed(6042)
        for sid in ids:
            raw,pm,_=batches[sid];eps=torch.randn(1,model.config.latent_dim,raw.shape[-1]//8,device=raw.device)
            values=fast_terms(model,raw,pm,eps,include_kl=True)
            for k,loss in [('geometry',values[0]+.1*values[1]+PEN*values[2]),('kl',values[3])]:
                grads=torch.autograd.grad(loss/len(ids),params,allow_unused=True,retain_graph=k=='geometry')
                for acc,g in zip(sums[k],grads):
                    if g is not None:acc.add_(g.detach())
    result={}
    for group in ('body','posterior','all'):
        indices=[j for j,p in enumerate(params) if group=='all' or (names[id(p)].startswith('conv_logvar.'))==(group=='posterior')]
        norms={k:float(torch.stack([sums[k][j].square().sum() for j in indices]).sum().sqrt() if indices else 0.) for k in sums}
        dot=float(torch.stack([(sums['geometry'][j]*sums['kl'][j]).sum() for j in indices]).sum()) if indices else 0.
        result[group]=dict(parameter_count=len(indices),norms=norms,cosine=dot/max(norms['geometry']*norms['kl'],1e-30),weighted_kl_fraction={a:w*norms['kl']/max(norms['geometry'],1e-30) for a,w in ARMS.items()})
    model.zero_grad(set_to_none=False)
    return dict(sample_ids=ids,seed=6042,groups=result,definition='initial aggregate gradients of same mean+.1sample geometry+boundedpen versus corrected full valid-channel KL; no updates/no DEV tuning')


def run(config,repo,root='/data',steps=200,pool_sha=''):
    validate_budget(steps,pool_sha)
    if not torch.cuda.is_available():raise ValueError('T4 required')
    torch.set_num_threads(4);root=Path(root);pool,m,vocab=load_pool(root,POOL_SHA)
    base,_,_,cfg,original_vocab,provenance=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if vocab!=original_vocab:raise ValueError('alphabet drift')
    parent=torch.load(root/SOURCE,map_location='cpu',weights_only=True);initial=copy.deepcopy(base.state_dict())
    schedule=list(training_schedule(m['splits']['large_train'],steps,8,seed=4042));splits=evaluation_scope(m,m['splits']['small_train'])
    eval_ids=sorted(set(i for values in splits.values() for i in values));ids=set(i for group in schedule for i in group)|set(eval_ids)
    batches,records,origins=load_batches(root,pool,m,vocab,ids,'cuda');eval_batches={i:batches[i] for i in eval_ids}
    capture={batches[i][0].shape[-1]:batches[i] for group in schedule for i in group}
    if len(capture)>80:raise ValueError('bounded graph count exceeded')
    out=root/'checkpoints/iam_posterior_isolation'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'evaluation-records.json').write_text(json.dumps({i:dict(records[i],origin=origins[i]) for i in eval_ids},indent=2)+'\n')
    for name in ('posterior_isolation.py','kl_tradeoff.py','ocr_joint_study.py','ocr_joint_adapter.py','ocr_context_features.py','ocr_recurrent.py','fast_geometry.py','codec_kl_study.py','initialization_study.py','writer_expansion.py','latent_integration.py','trajectory_geometry.py'):
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ('vae.py','blocks.py','losses.py'):
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    results={};reference=None;scales=None;diagnostic=None
    for arm,weight in ARMS.items():
        model=copy.deepcopy(base).cuda().eval();model.ocr_model,reader_cfg=load_reader(root,cfg,'cuda')
        optimizer=restore_optimizer(model,parent);protected=protect_pen_variance_weights(model,optimizer)
        frozen_snapshot=configure_arm(model,optimizer,arm)
        reader_digest=tensor_digest(model.ocr_model.state_dict())
        if scales is None:
            scales=source_scales(model,eval_batches,eval_ids);torch.save({i:v.cpu() for i,v in scales.items()},out/'fixed-source-std.pt')

        diagnostic=gradient_diagnostics(model,batches,splits['train_probe'])
        folder=out/arm;folder.mkdir()
        (folder/'gradient-diagnostics.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
        graphs=GeometryGraphs(model,PEN,.1,kl_weight=weight);start=time.monotonic();graphs.prepare(list(capture.values()));capture_seconds=time.monotonic()-start
        parity=capture_parity(model,graphs,batches,splits['train_probe'][:8],weight)
        (folder/'capture-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        settings=dict(parent['config'],profile='posterior-isolation',arm=arm,freeze_policy='entire codec except conv_logvar' if arm=='posterior_only' else 'source body+posterior groups',frozen_parameter_names=list(frozen_snapshot['weights']),research_ocr_contract=CONTRACT,cfg=cfg,source_rel=SOURCE,source_sha256=SHA,
          reader_rel=READER_REL,reader_sha256=READER_SHA,pool_manifest_sha256=POOL_SHA,source_std_sha256=file_sha(out/'fixed-source-std.pt'),splits=splits,
          train_ids=m['splits']['large_train'],schedule_seed=4042,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),noise_seed=7042,
          source_optimizer_updates=parent['optimizer_updates'],optimizer='identical restored source AdamW groups/moments; pen variance weight moments protected',optimizer_groups=[dict(name=g['name'],lr=g['lr']) for g in optimizer.param_groups],
          max_updates=steps,max_wall_seconds=1800,eval_steps=sorted(set([0,100,steps])),physical_batch=1,gradient_accumulation=8,mean_geometry_weight=1.,sampled_geometry_weight=.1,pen_weight=PEN,
          kl_weight=weight,ctc_weight=0.,style_weight=0.,gmm_weight=0.,dropout=0.,target_delta_weight=DELTA_WEIGHT,captured_lengths=sorted(capture),capture_seconds=capture_seconds,
          checkpoint_selection='TRAIN32 corrected KL then sampledXY among BOTH own/fixed-source-sigma curve/pen gates; DEV/report never selects or stops',
          research_contract='initialized polyphase40, fixed readout; KL tradeoff not learned semantic/generative readiness',runtime=str(torch.__version__))
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        def save(name,step):
            state={k:v for k,v in model.state_dict().items() if not k.startswith('ocr_model.')};state.update({k:v for k,v in initial.items() if k.startswith('ocr_model.')})
            torch.save(dict(model_state_dict=state,config=settings,optimizer_state_dict=optimizer.state_dict(),optimizer_updates=parent['optimizer_updates']+step,continuation_updates=step,
               sample_ids=parent['sample_ids'],provenance=provenance,source_sha256=SHA,frozen_reader_sha256=READER_SHA,rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),folder/name)
        verify_frozen(model,optimizer,frozen_snapshot)
        first=evaluate(model,eval_batches,records,vocab,splits,scales,folder,0,save_trajectories=True)
        stats=posterior_stats(model,eval_batches,splits);(folder/'posterior-0.json').write_text(json.dumps(stats,indent=2)+'\n')
        if reference is None:reference=first
        else:
            if first!=reference:raise AssertionError('initial paired evaluation mismatch')
        if not eligibility(first,first,eval_ids)['passed']:raise AssertionError('initial posterior/pen gate failed')
        best=selection(stats,first);best_step=0;save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        history=[dict(step=0,groups={p:r['groups'] for p,r in first.items()},posterior=stats['groups'],gates=eligibility(first,first,eval_ids))]
        torch.manual_seed(7042);started=time.monotonic();stop='budget_completed'
        with (folder/'metrics.jsonl').open('w') as log:
            for step,group in enumerate(schedule,1):
                noise=[torch.randn(1,cfg['latent_dim'],batches[i][0].shape[-1]//8,device='cuda') for i in group]
                optimizer.zero_grad(set_to_none=False);total=torch.zeros(4 if weight else 3,device='cuda')
                for sid,eps in zip(group,noise):total+=graphs.replay(batches[sid],divisor=8,epsilon=eps)/8
                if not torch.isfinite(total).all():raise FloatingPointError('nonfinite objective before update')
                norm=float(torch.nn.utils.clip_grad_norm_(graphs.parameters,5,error_if_nonfinite=True));optimizer.step()
                assert_frozen_gradients(model,frozen_snapshot)
                values=total.cpu().tolist();log.write(json.dumps(dict(step=step,sample_ids=group,terms=values,raw_gradient_norm=norm,was_clipped=norm>5,
                    noise_sha256=hashlib.sha256(b''.join(e.cpu().numpy().tobytes() for e in noise)).hexdigest()))+'\n');log.flush()
                broken=step>=10 and values[0]>1e-4;limit=time.monotonic()-started>1800
                if step%100==0 or step==steps or broken or limit:
                    row=evaluate(model,eval_batches,records,vocab,splits,scales,folder,step,save_trajectories=True)
                    verify_frozen(model,optimizer,frozen_snapshot)
                    if arm=='posterior_only':assert_output_isolation(row,first)
                    stats=posterior_stats(model,eval_batches,splits);(folder/f'posterior-{step}.json').write_text(json.dumps(stats,indent=2)+'\n')
                    train_gate=eligibility(row,first,splits['train_probe']+splits['named']);report_gate=eligibility(row,first,eval_ids)
                    history.append(dict(step=step,groups={p:r['groups'] for p,r in row.items()},posterior=stats['groups'],gates=report_gate,train_gate=train_gate))
                    if train_gate['passed'] and selection(stats,row)<best:best=selection(stats,row);best_step=step;save('checkpoint-best.pt',step)
                    save('checkpoint-last.pt',step)
                    if not train_gate['passed']:stop='TRAIN/named own or fixedsigma geometry gate';break
                if broken:stop='shuffled TRAIN geometry gate';break
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        invariant=verify_frozen(model,optimizer,frozen_snapshot)
        if tensor_digest(model.ocr_model.state_dict())!=reader_digest or any(p.grad is not None for p in model.ocr_model.parameters()):raise AssertionError('frozen reader changed')
        unchanged=all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith(('transformer_decoder.','style_classifier.')))
        if not unchanged or not torch.equal(model.conv_logvar.weight[protected[0]],protected[1]) or file_sha(root/SOURCE)!=SHA:raise AssertionError('protected/source weights changed')
        result=dict(arm=arm,kl_weight=weight,last_step=step,best_step=best_step,stop_reason=stop,history=history,selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),
          elapsed_loop_evaluation_seconds=time.monotonic()-started,reader_weights_unchanged=True,readout_style_pen_variance_protected=True,source_unchanged=True,
          final_rng_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),final_rng_cuda_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),not_promoted=True,frozen_state_invariance=invariant,posterior_output_isolation=arm=='posterior_only')
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result
        protected[2].remove();del model,optimizer,graphs;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(last_step=r['last_step'],best_step=r['best_step'],stop=r['stop_reason']) for a,r in results.items()})
