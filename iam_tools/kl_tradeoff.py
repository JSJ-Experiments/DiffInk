"""Bounded full-TRAIN KL tradeoff with same-epsilon SOURCE-sigma controls.

Initialized polyphase40 research codec, not semantic/generative readiness.
No OCR/style/GMM training. Fixed sigma is a diagnostic, not a new posterior.
"""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .ocr_joint_study import SOURCE,SHA,LEGACY_EIGHT,gate,local_geometry,summarize
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA,CONTRACT
from .ocr_pool_study import load_pool,single_batch
from .ocr_convergence import POOL_SHA
from .writer_expansion import load,training_schedule,MANIFEST_SHA
from .codec_kl_study import restore_optimizer,protect_pen_variance_weights,capture_parity,posterior_stats
from .fast_geometry import GeometryGraphs,fast_terms
from .initialization_study import PEN
from .latent_integration import encoded,DELTA_WEIGHT
from .ocr_context_study import tensor_digest
from .inkvae import greedy_ctc,edit_distance
from .pen_ab import file_sha,boundary_metrics

# Previous1e-5 degraded sampled curves; first test below/at the source's1e-6.
ARMS={'kl0':0.,'kl1e-7':1e-7,'kl1e-6':1e-6}


def validate_budget(steps,pool_sha):
    if type(steps) is not int or not 50<=steps<=300 or pool_sha!=POOL_SHA:
        raise ValueError('50–300 updates and exact immutable TRAIN pool required')


def evaluation_scope(records,small_train):
    probe=list(small_train[::6]);named=list(LEGACY_EIGHT)
    return dict(train_probe=probe,dev=list(records['splits']['dev']),held_out=list(records['splits']['held_out']),named=named)


def load_batches(root,pool,manifest,vocab,ids,device):
    """All8 included explicitly; legacy missing pool lines NEVER enter TRAIN schedule."""
    batches={};records={};origins={};legacy=Path(root)/'diffink/iam_overfit'
    if file_sha(legacy/'manifest.json')!=MANIFEST_SHA:raise ValueError('original8 manifest drift')
    with h5py.File(pool/'lines.h5') as hf,h5py.File(legacy/'tiny_train.h5') as old:
        for sid in sorted(ids):
            if sid in manifest['records']:
                points=hf[sid]['point_seq'][:];r=manifest['records'][sid]
                if hashlib.sha256(points.tobytes()).hexdigest()!=r['points_sha256']:raise ValueError('point fingerprint drift')
                text=r['text'];writer=r['writer_id'];origin='immutable reader pool'
            elif sid in LEGACY_EIGHT:
                g=old[sid];points=g['point_seq'][:];text=g['line_text'][()].decode();writer=g['writer_id'][()].decode();origin='immutable original8; evaluation only'
            else:raise ValueError('unknown evaluation/training sample')
            if writer in manifest['test_writers']:raise ValueError('held-out test writer leakage')
            if sid in LEGACY_EIGHT:
                q=old[sid]
                if not np.array_equal(q['point_seq'][:],points) or q['line_text'][()].decode()!=text:raise ValueError('legacy/pool curve identity mismatch')
            if not np.isfinite(points).all():raise ValueError('nonfinite raw trajectory')
            batches[sid]=single_batch(points,text,vocab,device)
            records[sid]=dict(text=text,writer_id=writer,points_sha256=hashlib.sha256(points.tobytes()).hexdigest())
            origins[sid]=origin
    return batches,records,origins


def validate_scale(std,mu):
    if std.shape!=mu.shape or std.device!=mu.device or std.dtype!=mu.dtype or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('finite positive same-shape/device/dtype SOURCE std required')
    return std


def paired_latents(mu,lv,source_std,eps):
    """Same epsilon, different mean but SOURCE std held fixed across versions."""
    source_std=validate_scale(source_std,mu)
    if any(e.shape!=mu.shape or e.device!=mu.device or e.dtype!=mu.dtype or not torch.isfinite(e).all() for e in eps):
        raise ValueError('finite matching posterior epsilon required')
    return [mu]+[mu+e*(.5*lv).exp() for e in eps],[mu]+[mu+e*source_std for e in eps]


@torch.no_grad()
def source_scales(model,batches,ids):
    return {sid:(.5*encoded(model,*batches[sid][:2])[2]).exp().detach().clone() for sid in ids}


@torch.no_grad()
def evaluate(model,batches,records,vocab,splits,source_std,folder,step,draws=20,save_trajectories=False):
    """Own and fixed-source-sigma, same point data/seeds/everydraw/mean; no grad."""
    from model.losses import mixture_expectation
    ids=sorted(set(i for values in splits.values() for i in values));rows={k:[] for k in ('own','fixed_source_sigma')}
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        for j,sid in enumerate(ids):
            raw,pm,label=batches[sid];true,mu,lv,lm=encoded(model,raw,pm);n=int(pm.sum());states=raw[0,2:,:n].argmax(0).cpu().numpy()
            truth=true[0,:n].cpu().numpy();torch.manual_seed(8042+j*100)
            eps=[torch.randn_like(mu) for _ in range(draws)];own,fixed=paired_latents(mu,lv,source_std[sid],eps)
            for policy,latents in [('own',own),('fixed_source_sigma',fixed)]:
                z=torch.cat(latents);out=model.decode(z,padding_mask=(~pm).expand(len(latents),-1))
                xy=mixture_expectation(out).cpu().numpy()[:,:n];pens=out[:,:3,:n].argmax(1).cpu().numpy()
                logits=model.ocr_model(z,padding_mask=(~lm).expand(len(latents),-1),point_mask=pm.expand(len(latents),-1))
                length=(n+3)//4;old=model.ocr_model.head.ctc.reduction;model.ocr_model.head.ctc.reduction='none'
                try:losses=model.ocr_model.head.ctc(logits.clamp(-30,30).log_softmax(2),label.expand(len(latents),-1)+1,
                   torch.full((len(latents),),length,dtype=torch.long),torch.full((len(latents),),label.shape[-1],dtype=torch.long))/label.shape[-1]
                finally:model.ocr_model.head.ctc.reduction=old
                variants=[]
                for k in range(len(latents)):
                    decoded=greedy_ctc(logits[:length,k].argmax(-1).tolist(),vocab)
                    v=dict(kind='mu' if k==0 else f'z-{k-1}',geometry=local_geometry(xy[k],truth,states),pen=boundary_metrics(pens[k],states),
                       decoded=decoded,ocr_errors=edit_distance(records[sid]['text'],decoded),characters=len(records[sid]['text']),ctc_loss=float(losses[k]))
                    variants.append(v)
                    if save_trajectories:
                        q=Path(folder)/policy/f'step-{step}'/sid;q.mkdir(parents=True,exist_ok=True)
                        np.save(q/(v['kind']+'.npy'),np.column_stack((xy[k],np.eye(3,dtype=np.float32)[pens[k]])))
                rows[policy].append(dict(sample_id=sid,writer_id=records[sid]['writer_id'],text=records[sid]['text'],mu=variants[0],sampled=variants[1:]))
                del out,z,logits,losses
    result={policy:dict(step=step,draws=draws,policy=policy,lines=line_rows,groups={g:summarize(line_rows,values) for g,values in splits.items()}) for policy,line_rows in rows.items()}
    for policy,row in result.items():
        q=Path(folder)/policy;q.mkdir(parents=True,exist_ok=True);(q/f'eval-{step}.json').write_text(json.dumps(row,indent=2)+'\n')
    print(dict(step=step,dev={p:r['groups']['dev'] for p,r in result.items()}),flush=True)
    return result


def eligibility(row,reference,ids):
    # Both actual posterior and SAME perturbation must preserve mean, all draws, pens.
    checks={p:gate(row[p],reference[p],ids) for p in ('own','fixed_source_sigma')}
    return dict(passed=all(v['passed'] for v in checks.values()),policies=checks)


def selection(stats,row):
    """Same TRAIN-only prior score across arms, gated before calling."""
    g=row['own']['groups']['train_probe']['sampled']
    return stats['groups']['train_probe']['kl_per_element'],g['x_rmse']+g['y_rmse']


def gradient_diagnostics(model,batches,ids):
    """Descriptive per-parameter-group raw KL/geometry gradient norms, not a sweep."""
    params=[p for p in model.parameters() if p.requires_grad];names={id(p):n for n,p in model.named_parameters()}
    sums={k:[torch.zeros_like(p) for p in params] for k in ('geometry','kl')}
    with torch.random.fork_rng(devices=[0]):
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
        norms={k:float(torch.stack([sums[k][j].square().sum() for j in indices]).sum().sqrt()) for k in sums}
        dot=float(torch.stack([(sums['geometry'][j]*sums['kl'][j]).sum() for j in indices]).sum())
        result[group]=dict(norms=norms,cosine=dot/max(norms['geometry']*norms['kl'],1e-30),weighted_kl_fraction={a:w*norms['kl']/max(norms['geometry'],1e-30) for a,w in ARMS.items()})
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
    out=root/'checkpoints/iam_kl_tradeoff'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'evaluation-records.json').write_text(json.dumps({i:dict(records[i],origin=origins[i]) for i in eval_ids},indent=2)+'\n')
    for name in ('kl_tradeoff.py','ocr_joint_study.py','ocr_joint_adapter.py','ocr_context_features.py','ocr_recurrent.py','fast_geometry.py','codec_kl_study.py','initialization_study.py','writer_expansion.py','latent_integration.py','trajectory_geometry.py'):
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ('vae.py','blocks.py','losses.py'):
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    results={};reference=None;scales=None;diagnostic=None
    for arm,weight in ARMS.items():
        model=copy.deepcopy(base).cuda().eval();model.ocr_model,reader_cfg=load_reader(root,cfg,'cuda')
        optimizer=restore_optimizer(model,parent);protected=protect_pen_variance_weights(model,optimizer)
        reader_digest=tensor_digest(model.ocr_model.state_dict())
        if scales is None:
            scales=source_scales(model,eval_batches,eval_ids);torch.save({i:v.cpu() for i,v in scales.items()},out/'fixed-source-std.pt')
            diagnostic=gradient_diagnostics(model,batches,splits['train_probe'])
            (out/'gradient-diagnostics.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
        graphs=GeometryGraphs(model,PEN,.1,kl_weight=weight);start=time.monotonic();graphs.prepare(list(capture.values()));capture_seconds=time.monotonic()-start
        parity=capture_parity(model,graphs,batches,splits['train_probe'][:8],weight)
        folder=out/arm;folder.mkdir();(folder/'capture-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        settings=dict(parent['config'],profile='fixed-source-sigma-kl-tradeoff',research_ocr_contract=CONTRACT,cfg=cfg,source_rel=SOURCE,source_sha256=SHA,
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
                values=total.cpu().tolist();log.write(json.dumps(dict(step=step,sample_ids=group,terms=values,raw_gradient_norm=norm,was_clipped=norm>5,
                    noise_sha256=hashlib.sha256(b''.join(e.cpu().numpy().tobytes() for e in noise)).hexdigest()))+'\n');log.flush()
                broken=step>=10 and values[0]>1e-4;limit=time.monotonic()-started>1800
                if step%100==0 or step==steps or broken or limit:
                    row=evaluate(model,eval_batches,records,vocab,splits,scales,folder,step,save_trajectories=True)
                    stats=posterior_stats(model,eval_batches,splits);(folder/f'posterior-{step}.json').write_text(json.dumps(stats,indent=2)+'\n')
                    train_gate=eligibility(row,first,splits['train_probe']+splits['named']);report_gate=eligibility(row,first,eval_ids)
                    history.append(dict(step=step,groups={p:r['groups'] for p,r in row.items()},posterior=stats['groups'],gates=report_gate,train_gate=train_gate))
                    if train_gate['passed'] and selection(stats,row)<best:best=selection(stats,row);best_step=step;save('checkpoint-best.pt',step)
                    save('checkpoint-last.pt',step)
                    if not train_gate['passed']:stop='TRAIN/named own or fixedsigma geometry gate';break
                if broken:stop='shuffled TRAIN geometry gate';break
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        if tensor_digest(model.ocr_model.state_dict())!=reader_digest or any(p.grad is not None for p in model.ocr_model.parameters()):raise AssertionError('frozen reader changed')
        unchanged=all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith(('transformer_decoder.','style_classifier.')))
        if not unchanged or not torch.equal(model.conv_logvar.weight[protected[0]],protected[1]) or file_sha(root/SOURCE)!=SHA:raise AssertionError('protected/source weights changed')
        result=dict(arm=arm,kl_weight=weight,last_step=step,best_step=best_step,stop_reason=stop,history=history,selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),
          elapsed_loop_evaluation_seconds=time.monotonic()-started,reader_weights_unchanged=True,readout_style_pen_variance_protected=True,source_unchanged=True,
          final_rng_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),final_rng_cuda_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),not_promoted=True)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result
        protected[2].remove();del model,optimizer,graphs;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(last_step=r['last_step'],best_step=r['best_step'],stop=r['stop_reason']) for a,r in results.items()})
