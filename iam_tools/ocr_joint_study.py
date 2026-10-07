"""Matched geometry/pen versus geometry/pen+frozenGRU; no KL/style/GMM.

Initialized transport compatibility, NOT semantic VAE or generation readiness.
Physical encoder/decoder batches remain one minimally padded line. Only the
independent frozen reader batches padded latents (eight per effective update).
"""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA,CONTRACT
from .ocr_pool_study import load_pool,single_batch,local_metrics,SOURCE,SHA
from .ocr_convergence import POOL_SHA
from .writer_expansion import load,training_schedule
from .latent_integration import encoded
from .initialization_study import PEN
from .codec_kl_study import restore_optimizer,protect_pen_variance_weights,capture_parity
from .fast_geometry import fast_terms,GeometryGraphs
from .ocr_context_study import tensor_digest
from .inkvae import greedy_ctc,edit_distance
from .pen_ab import file_sha,boundary_metrics

ARMS=('geometry_only','geometry_ocr')
NAMED=('p08-936z-05','a07-421z-02')
LEGACY_EIGHT=('c08-434z-05','e08-429z-04','p08-936z-05','a07-421z-03','k07-640z-02','l10-072z-02','h05-195z-03','a07-421z-02')


def named_scope(records):
    """Explicitly record pool exclusions; independent report MUST cover all8.

    Never relabel or extend the immutable8192 pool just to include curve cases.
    This metadata guard was added after the first run's4-case pool scope was
    noticed; that run is preserved, with all8 independently audited on CPU.
    """
    return dict(legacy_eight=list(LEGACY_EIGHT),pool_ids=[i for i in LEGACY_EIGHT if i in records],
      outside_pool=[i for i in LEGACY_EIGHT if i not in records],
      required_supplement='immutable original iam_overfit tiny_train HDF5; all8/source/selected/final/20CPUdraws; never training/selection/stop')


def ctc_batch(model,batches,ids,noises):
    values=[];masks=[];labels=[]
    for sid,eps in zip(ids,noises):
        raw,pm,label=batches[sid];_,mu,lv,_=encoded(model,raw,pm)
        values.append(mu+eps*(.5*lv).exp());masks.append(pm);labels.append(label)
    length=max(x.shape[-1] for x in values);chars=max(x.shape[-1] for x in labels)
    z=torch.cat([F.pad(x,(0,length-x.shape[-1])) for x in values])
    pm=torch.cat([F.pad(m,(0,8*length-m.shape[-1]),value=False) for m in masks])
    labels=torch.cat([F.pad(x,(0,chars-x.shape[-1]),value=-1) for x in labels])
    return model.get_ocr_loss(z,labels,pm.reshape(len(ids),-1,8).any(-1),point_mask=pm)


def calibration(model,batches,ids,fraction=.1):
    """TRAIN-only aggregate encoder gradient norms, fixed source/noise, once."""
    params=list(model.encoder.parameters())+list(model.conv_mu.parameters())
    sums={k:[torch.zeros_like(p) for p in params] for k in ('geometry','ctc')}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(6042)
        noises={i:torch.randn(1,model.config.latent_dim,batches[i][0].shape[-1]//8,device='cuda') for i in ids}
        for sid in ids:
            terms=fast_terms(model,*batches[sid][:2],noises[sid])
            loss=(terms[0]+.1*terms[1]+PEN*terms[2])/len(ids)
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            for acc,g in zip(sums['geometry'],grads):
                if g is not None:acc.add_(g.detach())
        for start in range(0,len(ids),8):
            group=ids[start:start+8]
            loss=ctc_batch(model,batches,group,[noises[i] for i in group])*len(group)/len(ids)
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            for acc,g in zip(sums['ctc'],grads):
                if g is not None:acc.add_(g.detach())
    norms={k:float(torch.stack([v.square().sum() for v in values]).sum().sqrt()) for k,values in sums.items()}
    if min(norms.values())<=1e-12 or not all(np.isfinite(v) for v in norms.values()):raise ValueError('degenerate/nonfinite gradient calibration')
    weight=min(.1,fraction*norms['geometry']/norms['ctc'])
    dot=float(torch.stack([(a*b).sum() for a,b in zip(sums['geometry'],sums['ctc'])]).sum())
    return dict(sample_ids=ids,seed=6042,encoder_gradient_norms=norms,ctc_weight=weight,
      target_fraction=fraction,actual_fraction=weight*norms['ctc']/norms['geometry'],gradient_cosine=dot/(norms['geometry']*norms['ctc']),
      method='initial aggregate encoder+conv_mu gradients of mean geometry+.1 sampled geometry+bounded pen versus sampled CTC; cap0.1; no DEV calibration')


def local_geometry(xy,target,states):
    from .trajectory_geometry import quantiles,wrap_angle
    g=local_metrics(xy,target,states)
    dp,dt=np.diff(xy,axis=0),np.diff(target,axis=0)
    valid=(states[:-1]==0)&(np.linalg.norm(dp,axis=1)>1e-8)&(np.linalg.norm(dt,axis=1)>1e-8)
    a,b=np.arctan2(dp[:,1],dp[:,0]),np.arctan2(dt[:,1],dt[:,0])
    turns=wrap_angle(np.diff(b));err=np.abs(wrap_angle(wrap_angle(np.diff(a))-turns))*180/np.pi
    v=valid[:-1]&valid[1:]
    g['target_corner_turn_error_degrees']=quantiles(err[v&(np.abs(turns)*180/np.pi>=45)])
    g['target_shallow_turn_error_degrees']=quantiles(err[v&(np.abs(turns)*180/np.pi<20)])
    return g


def summarize(rows,ids):
    selected=[r for r in rows if r['sample_id'] in ids];out={}
    for kind in ('mu','sampled'):
        variants=[r['mu'] for r in selected] if kind=='mu' else [v for r in selected for v in r['sampled']]
        if not variants:continue
        out[kind]=dict(evaluations=len(variants),cer=sum(v['ocr_errors'] for v in variants)/sum(v['characters'] for v in variants),
          mean_ctc_loss=float(np.mean([v['ctc_loss'] for v in variants])),exact_lines=sum(v['ocr_errors']==0 for v in variants),
          x_rmse=float(np.mean([v['geometry']['x_rmse'] for v in variants])),y_rmse=float(np.mean([v['geometry']['y_rmse'] for v in variants])),
          first_difference_vector_rmse=float(np.mean([v['geometry']['first_difference']['vector_rmse'] or 0 for v in variants])),
          second_difference_vector_rmse=float(np.mean([v['geometry']['second_difference']['vector_rmse'] or 0 for v in variants])),
          mean_per_line_turn_p90=float(np.mean([v['geometry']['turn_angle_error_degrees']['p90'] or 0 for v in variants])),
          mean_per_line_tangent_p90=float(np.mean([v['geometry']['tangent_angle_error_degrees']['p90'] or 0 for v in variants])),
          mean_corner_turn_p90=float(np.mean([v['geometry']['target_corner_turn_error_degrees']['p90'] or 0 for v in variants])),
          min_pen_f1=min(v['pen']['pen_up_f1'] for v in variants),false_eoc=sum(v['pen']['non_final_false_eoc_count'] for v in variants),
          all_final_eoc_correct=all(v['pen']['final_eoc_correct'] for v in variants))
    return out


def gate(row,reference,ids):
    """Per-line mean+everydraw gate. No aggregate error can hide a kink."""
    refs={r['sample_id']:r for r in reference['lines']};fail=[]
    current_ids=[r['sample_id'] for r in row['lines']]
    if len(current_ids)!=len(set(current_ids)) or not set(ids)<=set(current_ids)&set(refs):
        raise ValueError('complete unique gated line IDs required')
    for r in row['lines']:
        if r['sample_id'] not in ids:continue
        ref=refs[r['sample_id']]
        for kind in ('mu','sampled'):
            variants=[r['mu']] if kind=='mu' else r['sampled']
            baseline=[ref['mu']] if kind=='mu' else ref['sampled']
            if len(variants)!=len(baseline):raise ValueError('same complete posterior draws required')
            for j,(v,b) in enumerate(zip(variants,baseline)):
                g,h=v['geometry'],b['geometry'];name=r['sample_id']+':'+kind+str(j)
                for axis in ('x_rmse','y_rmse'):
                    bound=max(4*h[axis],.0005) if kind=='mu' else max(1.2*h[axis],.0001)
                    if g[axis]>bound:fail.append(name+':'+axis)
                for angle in ('turn_angle_error_degrees','target_corner_turn_error_degrees','target_shallow_turn_error_degrees'):
                    bound=max((1.5 if kind=='mu' else 1.2)*(h[angle]['p90'] or 0),1.)
                    if (g[angle]['p90'] or 0)>bound:fail.append(name+':'+angle)
                p=v['pen']
                if p['pen_up_f1']!=1. or not p['final_eoc_correct'] or p['non_final_false_eoc_count']:fail.append(name+':pen')
    return dict(passed=not fail,failures=fail,
      thresholds='per-line mean axis max(4*ref,0.0005); sampled axis max(1.2*same-draw ref,0.0001); mean turn/corner/shallow p90 max(1.5*ref,1deg), sampled max(1.2*same-draw ref,1deg); exact pen/F1/EOC. Length/index differences are not physical velocity or curvature.')


@torch.no_grad()
def evaluate(model,batches,records,vocab,splits,folder,step,draws=20,save=True):
    from model.losses import mixture_expectation
    rows=[]
    with torch.random.fork_rng(devices=[0] if next(model.parameters()).is_cuda else []):
        for j,sid in enumerate(sorted(set(i for ids in splits.values() for i in ids))):
            raw,pm,label=batches[sid];_,mu,lv,lm=encoded(model,raw,pm);n=int(pm.sum())
            torch.manual_seed(8042+j*100);latents=[mu]+[mu+torch.randn_like(mu)*(.5*lv).exp() for _ in range(draws)]
            z=torch.cat(latents);point_mask=pm.expand(len(latents),-1);latent_mask=lm.expand(len(latents),-1)
            out=model.decode(z,padding_mask=~point_mask);xy=mixture_expectation(out).cpu().numpy()[:,:n]
            pens=out[:,:3,:n].argmax(1).cpu().numpy()
            logits=model.ocr_model(z,padding_mask=~latent_mask,point_mask=point_mask)
            length=(n+3)//4;labels=label.expand(len(latents),-1)
            old=model.ocr_model.head.ctc.reduction;model.ocr_model.head.ctc.reduction='none'
            try:losses=model.ocr_model.head.ctc(logits.clamp(-30,30).log_softmax(2),labels+1,
                  torch.full((len(latents),),length,dtype=torch.long),torch.full((len(latents),),label.shape[-1],dtype=torch.long))/label.shape[-1]
            finally:model.ocr_model.head.ctc.reduction=old
            true=model.to_model_space(raw)[0,:2,:n].T.cpu().numpy();states=raw[0,2:,:n].argmax(0).cpu().numpy();text=records[sid]['text']
            variants=[]
            for k in range(len(latents)):
                decoded=greedy_ctc(logits[:length,k].argmax(-1).tolist(),vocab)
                v=dict(kind='mu' if k==0 else f'z-{k-1}',geometry=local_geometry(xy[k],true,states),pen=boundary_metrics(pens[k],states),
                  decoded=decoded,ocr_errors=edit_distance(text,decoded),characters=len(text),ctc_loss=float(losses[k]))
                if save:
                    p=folder/f'step-{step}'/sid;p.mkdir(parents=True,exist_ok=True)
                    np.save(p/(v['kind']+'.npy'),np.column_stack([xy[k],np.eye(3,dtype=np.float32)[pens[k]]]))
                variants.append(v)
            rows.append(dict(sample_id=sid,writer_id=records[sid]['writer_id'],text=text,mu=variants[0],sampled=variants[1:]))
    result=dict(step=step,draws=draws,lines=rows,groups={g:summarize(rows,ids) for g,ids in splits.items()})
    if save:(folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(step=step,groups={g:s['mu'] for g,s in result['groups'].items()}),flush=True)
    return result


def selection(row):
    g=row['groups']['train_probe']['mu']
    return (g['cer'],g['mean_ctc_loss'],g['x_rmse']+g['y_rmse'])


def run(config,repo,root='/data',steps=200,pool_sha=''):
    if type(steps) is not int or not 50<=steps<=400 or pool_sha!=POOL_SHA or not torch.cuda.is_available():raise ValueError('CUDA,fixed pool,and50–400 matched updates required')
    root=Path(root);torch.set_num_threads(4);pool,m,vocab=load_pool(root,POOL_SHA)
    base,_,_,cfg,_,_=load(config,repo,root,SOURCE,SHA,writer_id=None)
    parent=torch.load(root/SOURCE,map_location='cpu',weights_only=True);initial=copy.deepcopy(base.state_dict())
    out=root/'checkpoints/iam_ocr_joint_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes())
    schedule=list(training_schedule(m['splits']['large_train'],steps,8,seed=4042));schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    probe=m['splits']['small_train'][::6]
    scope=named_scope(m['records']);named=scope['pool_ids']
    splits=dict(train_probe=probe,dev=m['splits']['dev'],held_out=m['splits']['held_out'],named=named)
    ids=set(i for batch in schedule for i in batch)|set(i for values in splits.values() for i in values)
    batches={}
    with h5py.File(pool/'lines.h5') as hf:
        for sid in sorted(ids):
            points=hf[sid]['point_seq'][:]
            if hashlib.sha256(points.tobytes()).hexdigest()!=m['records'][sid]['points_sha256']:raise ValueError('point fingerprint drift')
            batches[sid]=single_batch(points,m['records'][sid]['text'],vocab,'cuda')
    # Cache/capture only scheduled minimal lengths, never pad unrelated raw lines.
    capture={batches[i][0].shape[-1]:batches[i] for batch in schedule for i in batch}
    if len(capture)>80:raise ValueError('geometry capture count exceeds bounded budget')
    for name in ('ocr_joint_study.py','ocr_joint_adapter.py','ocr_context_features.py','ocr_recurrent.py','fast_geometry.py','codec_kl_study.py','initialization_study.py','writer_expansion.py'):
        p=out/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ('vae.py','blocks.py','losses.py'):
        p=out/'source-code/model'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model'/name).read_bytes())
    results={};cal=None;reference=None
    for arm in ARMS:
        model=copy.deepcopy(base).cuda().eval();model.ocr_model,reader_cfg=load_reader(root,cfg,'cuda')
        optimizer=restore_optimizer(model,parent);protected=protect_pen_variance_weights(model,optimizer)
        head_digest=tensor_digest(model.ocr_model.state_dict())
        folder=out/arm;folder.mkdir()
        # Real CUDA input-backward and backend parity before any optimizer step.
        raw,pm,label=batches[probe[0]]
        with torch.no_grad():_,mu,lv,lm=encoded(model,raw,pm);expected=model.ocr_model(mu,padding_mask=~lm,point_mask=pm)
        x=mu.detach().requires_grad_();actual=model.ocr_model(x,padding_mask=~lm,point_mask=pm)
        diff=float((actual-expected).abs().max());loss=model.get_ocr_loss(x,label,lm,point_mask=pm);loss.backward()
        if diff>1e-4 or not torch.isfinite(x.grad).all() or not x.grad.abs().sum()>0:raise AssertionError('frozen CUDA reader input-backward parity failed')
        backend=dict(max_logit_difference=diff,input_gradient_l2=float(x.grad.norm()),reader_gradients_absent=all(p.grad is None for p in model.ocr_model.parameters()))
        if not backend['reader_gradients_absent']:raise AssertionError('reader gradients unexpectedly enabled')
        (folder/'reader-backend-parity.json').write_text(json.dumps(backend,indent=2)+'\n')
        if cal is None:cal=calibration(model,batches,probe)
        weight=0. if arm=='geometry_only' else cal['ctc_weight']
        graphs=GeometryGraphs(model,PEN,.1)
        start=time.monotonic();graphs.prepare(list(capture.values()));capture_seconds=time.monotonic()-start
        parity=capture_parity(model,graphs,batches,probe[:8],0.)
        (folder/'capture-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        settings=dict(parent['config'],profile='matched-frozen-gru-geometry-joint',research_ocr_contract=CONTRACT,
          source_rel=SOURCE,source_sha256=SHA,reader_rel=READER_REL,reader_sha256=READER_SHA,pool_manifest_sha256=POOL_SHA,
          cfg=cfg,splits=splits,named_scope=scope,train_ids=m['splits']['large_train'],schedule_seed=4042,schedule_sha256=schedule_sha,
          optimizer='restored identical source AdamW moments; pen variance weight moments/gradients protected',optimizer_groups=[dict(name=g['name'],lr=g['lr']) for g in optimizer.param_groups],
          max_updates=steps,max_wall_seconds=1800,eval_every=50,physical_batch=1,gradient_accumulation=8,reader_batch=8,
          mean_geometry_weight=1.,sampled_geometry_weight=.1,pen_weight=PEN,ctc_weight=weight,kl_weight=0.,style_weight=0.,gmm_weight=0.,
          calibration=cal,reader_frozen=True,trajectory_dropout=0.,reader_dropout=0.,captured_lengths=sorted(capture),capture_seconds=capture_seconds,
          source_optimizer_updates=parent['optimizer_updates'],reader_contract=reader_cfg,checkpoint_selection='TRAIN32 probe CER then CTC then geometry, among strict geometry/pen gate passing evaluations; DEV/report never select',
          runtime=str(torch.__version__),not_semantic_or_generative_readiness=True)
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        def save(name,step):
            state={k:v for k,v in model.state_dict().items() if not k.startswith('ocr_model.')}
            state.update({k:v for k,v in initial.items() if k.startswith('ocr_model.')})
            torch.save(dict(model_state_dict=state,config=settings,optimizer_state_dict=optimizer.state_dict(),optimizer_updates=parent['optimizer_updates']+step,
              continuation_updates=step,sample_ids=m['splits']['small_train'],provenance=parent.get('provenance'),source_sha256=SHA,
              rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all(),frozen_reader_sha256=READER_SHA),folder/name)
        first=evaluate(model,batches,m['records'],vocab,splits,folder,0)
        if not gate(first,first,list(set(i for values in splits.values() for i in values)))['passed']:
            raise AssertionError('source fails initial all-reported mean/posterior pen gate')
        if reference is None:reference=first
        else:
            if first['lines']!=reference['lines']:raise AssertionError('identical initial paired evaluation failed')
        for r in first['lines']:
            if max(r['mu']['geometry']['x_rmse'],r['mu']['geometry']['y_rmse'])>.0025 or r['mu']['pen']['pen_up_f1']!=1.:
                raise AssertionError('initial geometry preflight failed')
        best=selection(first);best_step=0;save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        torch.manual_seed(7042);history=[dict(step=0,groups=first['groups'],gate=gate(first,first,probe+named))]
        started=time.monotonic();stop='budget_completed'
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch_ids in enumerate(schedule,1):
                noises=[torch.randn(1,cfg['latent_dim'],batches[i][0].shape[-1]//8,device='cuda') for i in batch_ids]
                optimizer.zero_grad(set_to_none=False);total=torch.zeros(3,device='cuda')
                for sid,epsilon in zip(batch_ids,noises):total+=graphs.replay(batches[sid],divisor=8,epsilon=epsilon)/8
                if weight:
                    ctc=ctc_batch(model,batches,batch_ids,noises);(weight*ctc).backward()
                else:
                    with torch.no_grad():ctc=ctc_batch(model,batches,batch_ids,noises)
                if not torch.isfinite(total).all() or not torch.isfinite(ctc):raise FloatingPointError('nonfinite objective before optimizer update')
                norm=float(torch.nn.utils.clip_grad_norm_(graphs.parameters,5,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,sample_ids=batch_ids,geometry_terms=total.cpu().tolist(),ctc=float(ctc.detach()),ctc_weight=weight,
                  raw_gradient_norm=norm,was_clipped=norm>5,noise_sha256=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in noises)).hexdigest()))+'\n');log.flush()
                limit=time.monotonic()-started>1800
                if step%50==0 or step==steps or limit:
                    row=evaluate(model,batches,m['records'],vocab,splits,folder,step);passed=gate(row,first,probe+named)
                    full_gate=gate(row,first,[i for group in splits.values() for i in group])
                    history.append(dict(step=step,groups=row['groups'],gate=passed,reporting_gate=full_gate))
                    if passed['passed'] and selection(row)<best:best=selection(row);best_step=step;save('checkpoint-best.pt',step)
                    save('checkpoint-last.pt',step)
                    if not passed['passed']:stop='TRAIN/named geometry gate';break
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        if tensor_digest(model.ocr_model.state_dict())!=head_digest or any(p.grad is not None for p in model.ocr_model.parameters()):raise AssertionError('frozen reader mutated')
        unchanged=all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith(('transformer_decoder.','style_classifier.')))
        if not unchanged or not torch.equal(model.conv_logvar.weight[protected[0]],protected[1]):raise AssertionError('protected readout/style/pen variance changed')
        result=dict(arm=arm,best_step=best_step,last_step=step,stop_reason=stop,initial=first['groups'],history=history,
          selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),
          reader_weights_buffers_bitwise_unchanged=True,readout_style_pen_variance_protected=True,
          source_unchanged=file_sha(root/SOURCE)==SHA,reader_source_unchanged=file_sha(root/READER_REL)==READER_SHA,
          elapsed_loop_evaluation_seconds=time.monotonic()-started,final_rng_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
          final_rng_cuda_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),not_promoted=True)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result
        protected[2].remove();del model,optimizer,graphs;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(best_step=r['best_step'],last_step=r['last_step'],stop=r['stop_reason']) for a,r in results.items()})
