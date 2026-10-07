"""Controlled small/large OCR supervision; faithful codec remains frozen."""
import hashlib,json,time,re
from pathlib import Path
import h5py,numpy as np,torch
from .frozen_ocr_study import SOURCE,SHA,bucket_schedule,collate_latents,summarize,batch_parity
from .writer_expansion import load
from .latent_integration import encoded
from .ocr_context_features import unpack,fit_stats,make_head
from .ocr_context_study import head_model,tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .curve_audit import curve_metrics
from .trajectory_geometry import wrap_angle,quantiles
from .inkvae import greedy_ctc,edit_distance
from .ocr_pool import assert_splits


def single_batch(points,text,vocab,device='cpu'):
    """Same minimal multiple-of8 EOC padding as released physical batch1."""
    a=torch.as_tensor(points,dtype=torch.float32);n=len(a);t=(n+7)//8*8
    padded=torch.zeros(t,5);padded[:,4]=1.;padded[:n]=a
    return padded.T[None].to(device),(torch.arange(t)[None]<n).to(device),torch.tensor([[vocab.index(c) for c in text]],device=device)


def local_metrics(xy,target,states):
    """Exact index-error and geometric angles, omitting expensive polyline search."""
    r=curve_metrics(xy,target,states);dp=np.diff(xy,axis=0);dt=np.diff(target,axis=0)
    valid=(states[:-1]==0)&(np.linalg.norm(dp,axis=1)>1e-8)&(np.linalg.norm(dt,axis=1)>1e-8)
    ap=np.arctan2(dp[:,1],dp[:,0]);at=np.arctan2(dt[:,1],dt[:,0])
    r['tangent_angle_error_degrees']=quantiles(np.abs(wrap_angle(ap-at))[valid]*180/np.pi)
    r['turn_angle_error_degrees']=quantiles(np.abs(wrap_angle(wrap_angle(np.diff(ap))-wrap_angle(np.diff(at))))[valid[:-1]&valid[1:]]*180/np.pi)
    return r


def load_pool(root,expected_sha):
    if not re.fullmatch(r'[0-9a-f]{64}',expected_sha):raise ValueError('valid lowercase manifest SHA256 required')
    archive=Path(root)/'diffink/iam_ocr_pool_versions'/expected_sha
    p=archive if archive.is_dir() else Path(root)/'diffink/iam_ocr_pool'
    if file_sha(p/'manifest.json')!=expected_sha:raise ValueError('pinned larger pool manifest required')
    m=json.loads((p/'manifest.json').read_text());vocab=list(json.loads((p/'chars.json').read_text()))
    if file_sha(p/'lines.h5')!=m['lines_h5_sha256'] or file_sha(p/'chars.json')!=m['vocab_sha256']:raise ValueError('pool data/vocab integrity failure')
    checks=assert_splits(m['records'],m['splits'],m['test_writers'],m['dev_writers'])
    if checks!=m['checks']:raise ValueError('split guard mismatch')
    return p,m,vocab


@torch.no_grad()
def cache_corpus(model,pool,m,vocab,out,device='cuda',save_geometry=True):
    """Encode/decode individually; all-point transport and mean-pen/curve gate."""
    from model.losses import mixture_expectation
    cache={};rows=[];failures=[];maximum=0.;texts={i:r['text'] for i,r in m['records'].items()}
    hf_out=h5py.File(Path(out)/'geometry-source.h5','w') if save_geometry else None
    try:
        with h5py.File(Path(pool)/'lines.h5') as hf:
            for sid in sorted(m['records']):
                points=hf[sid]['point_seq'][:];record=m['records'][sid]
                if hashlib.sha256(points.tobytes()).hexdigest()!=record['points_sha256']:raise ValueError('point fingerprint changed: '+sid)
                raw,mask,labels=single_batch(points,texts[sid],vocab,device);target,mu,lv,lm=encoded(model,raw,mask)
                n=len(points);required=len(texts[sid])+sum(a==b for a,b in zip(texts[sid],texts[sid][1:]))
                if int(lm.sum())<required:raise ValueError('CTC infeasible: '+sid)
                f,real=unpack(mu,lm);packed=f.permute(0,3,1,2).reshape(1,-1,5)[0,:n]
                difference=float((packed[:,:2]-target[0,:n]).abs().max());maximum=max(maximum,difference)
                if int(real.sum())!=n or (packed[:,2:].argmax(1)!=raw[0,2:,:n].argmax(0)).any() or difference>1e-3:
                    failures.append(sid+':transport')
                output=model.decode(mu,padding_mask=~mask);xy=mixture_expectation(output)[0,:n].cpu().numpy();pens=output[0,:3,:n].argmax(0).cpu().numpy()
                true=target[0,:n].cpu().numpy();states=points[:,2:].argmax(1);metrics=local_metrics(xy,true,states);pen=boundary_metrics(pens,states)
                if max(metrics['x_rmse'],metrics['y_rmse'])>.0025 or (metrics['turn_angle_error_degrees']['p90'] or 0)>9. or pen['pen_up_f1']!=1. or not pen['final_eoc_correct'] or pen['non_final_false_eoc_count']:
                    failures.append(sid+':mean_geometry')
                if hf_out is not None:
                    g=hf_out.create_group(sid);g.create_dataset('mean',data=np.column_stack([xy,np.eye(3,dtype=np.float32)[pens]]),compression='gzip')
                rows.append(dict(sample_id=sid,geometry=metrics,pen=pen,max_packed_xy_difference=difference))
                cache[sid]=dict(mu=mu.detach(),lv=lv.detach(),mask=lm.detach(),labels=labels.detach())
    finally:
        if hf_out is not None:hf_out.close()
    audit=dict(lines=rows,maximum_packed_xy_difference=maximum,failed_checks=failures,passed=not failures,
        definitions='point/difference metrics by original nonuniform RDP index; angles are geometric within true strokes',
        thresholds=dict(transport_max_xy_difference=.001,mean_axis_rmse=.0025,mean_turn_p90_degrees=9.,pen_f1=1.,false_internal_eoc=0,final_eoc='correct'))
    if save_geometry:(Path(out)/'codec-preflight.json').write_text(json.dumps(audit,indent=2)+'\n')
    if failures:raise AssertionError('frozen codec preflight failed; no OCR training: '+str(failures[:10]))
    return cache,texts,audit


def ctc_per_line(head,logits,labels,mask):
    """Match upstream mean CTC: none-reduction NLL divided by target length."""
    lengths=(labels!=-1).sum(1).long();inputs=mask.sum(1).long()
    repeats=((labels[:,1:]==labels[:,:-1])&(labels[:,1:]!=-1)).sum(1)
    if (lengths==0).any() or (inputs<lengths+repeats).any():raise ValueError('all evaluation lines must be exact-CTC-feasible')
    reduction=head.ctc.reduction;head.ctc.reduction='none'
    try:return head.ctc(logits.clamp(-30,30).log_softmax(2),labels+1,inputs.cpu(),lengths.cpu())/lengths
    finally:head.ctc.reduction=reduction


@torch.no_grad()
def evaluate(head,cache,texts,splits,vocab,out,step,posterior_ids=(),draws=20):
    """Batched masked means; only predeclared reporting lines get posterior draws."""
    was=head.training;head.eval();rows=[];device=next(head.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        ids=sorted(set(i for v in splits.values() for i in v),key=lambda i:(cache[i]['mu'].shape[-1],i))
        for start in range(0,len(ids),16):
            batch=ids[start:start+16];z,labels,mask=collate_latents(cache,batch);logits=head(z,padding_mask=~mask)
            losses=ctc_per_line(head,logits,labels,mask)
            if not torch.isfinite(losses).all():raise FloatingPointError('nonfinite pool evaluation CTC')
            for j,sid in enumerate(batch):
                frames=logits[:int(mask[j].sum()),j].argmax(-1).tolist();decoded=greedy_ctc(frames,vocab)
                value=dict(decoded=decoded,errors=edit_distance(texts[sid],decoded),characters=len(texts[sid]),blank_frame_fraction=frames.count(0)/len(frames))
                rows.append(dict(sample_id=sid,text=texts[sid],mu=value,sampled=[],ctc_loss=float(losses[j]),adjacent_repeats=any(a==b for a,b in zip(texts[sid],texts[sid][1:]))))
        by_id={r['sample_id']:r for r in rows}
        for j,sid in enumerate(sorted(posterior_ids)):
            c=cache[sid];torch.manual_seed(8042+j*100);z=torch.cat([c['mu']+torch.randn_like(c['mu'])*(.5*c['lv']).exp() for _ in range(draws)])
            logits=head(z,padding_mask=(~c['mask']).expand(draws,-1));n=int(c['mask'].sum())
            for frames in logits[:n].argmax(-1).T.tolist():
                decoded=greedy_ctc(frames,vocab);by_id[sid]['sampled'].append(dict(decoded=decoded,errors=edit_distance(texts[sid],decoded),characters=len(texts[sid]),blank_frame_fraction=frames.count(0)/len(frames)))
        groups={}
        for group,group_ids in splits.items():
            selected=[by_id[i] for i in group_ids]
            # summarize expects at least one posterior evaluation. For groups
            # without draws, mean records are temporary placeholders then REMOVE
            # sampled stats; never present them as actual sampled results.
            fake=[dict(r,sampled=r['sampled'] or [r['mu']]) for r in selected]
            summary=summarize(fake,group_ids)
            if all(r['sampled'] for r in selected):summary['posterior_lines']=len(selected)
            else:summary.pop('sampled');summary['posterior_lines']=sum(bool(r['sampled']) for r in selected)
            groups[group]=summary
    head.train(was);row=dict(step=step,lines=rows,groups=groups,posterior_draws=draws,posterior_sample_ids=sorted(posterior_ids))
    (Path(out)/f'ocr-{step}.json').write_text(json.dumps(row,indent=2)+'\n')
    print(dict(step=step,ocr={k:groups[k]['mu']['cer'] for k in ('train','dev','held_out')}),flush=True);return row


def run(config,repo,root='/data',pool_sha='',steps=6000):
    if not torch.cuda.is_available() or not 1000<=steps<=8000:raise ValueError('CUDA and bounded1000–8000 steps per arm required')
    torch.set_num_threads(4);root=Path(root);pool,m,vocab=load_pool(root,pool_sha)
    base,_,_,cfg,original_vocab,original_prov=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if original_vocab!=vocab:raise ValueError('codec fixed alphabet changed')
    base.cuda().eval().requires_grad_(False);source_digest=tensor_digest(base.state_dict())
    out=root/'checkpoints/iam_ocr_pool_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'original-codec-provenance.json').write_text(json.dumps(original_prov,indent=2)+'\n')
    # Preserve the implementation even if a preflight fails before any head
    # exists. Failed directories never count as completed experiments.
    for name in ('ocr_pool_study.py','ocr_pool.py','ocr_context_features.py'):
        dest=out/'source-code/iam_tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).with_name(name).read_bytes())
    try:cache,texts,audit=cache_corpus(base,pool,m,vocab,out)
    except Exception as exc:
        (out/'failed-preflight.json').write_text(json.dumps(dict(status='failed_before_any_OCR_optimizer_update',error_type=type(exc).__name__,error=str(exc),source_sha256=SHA,pool_manifest_sha256=pool_sha),indent=2)+'\n')
        raise
    # Both arms use EXACTLY the same feature moments, fitted to nested small
    # TRAIN192 only. Large-arm data amount is the only feature-independent change.
    stats=fit_stats(cache,m['splits']['small_train'],'relative_scaled');(out/'shared-feature-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    probe=list(m['splits']['small_train'][::6]);posterior_ids=probe+m['splits']['dev']+m['splits']['held_out'];results={};initial_sha=None
    for arm,train_key in [('small192','small_train'),('large2048','large_train')]:
        folder=out/arm;folder.mkdir();train_ids=m['splits'][train_key]
        splits=dict(train=train_ids,dev=m['splits']['dev'],held_out=m['splits']['held_out'],common_train_probe=probe)
        head=make_head(cfg,len(vocab)+1,'relative_scaled',stats,seed=42).cuda();digest=tensor_digest(head.state_dict())
        if initial_sha is None:initial_sha=digest
        if initial_sha!=digest:raise AssertionError('fresh head pairing failure')
        settings=dict(profile='frozen-transport-OCR-data-size-control',source_rel=SOURCE,source_sha256=SHA,pool_rel=str(pool.relative_to(root)),pool_manifest_sha256=pool_sha,
            cfg=cfg,feature_mode='relative_scaled',feature_stats=stats,feature_calibration_ids=m['splits']['small_train'],attention_radius=None,
            train_ids=train_ids,dev_ids=splits['dev'],held_out_ids=splits['held_out'],common_train_probe=probe,
            physical_batch=16,encoder_physical_batch=1,initial_head_tensor_sha256=digest,seed=42,base_lr=5e-4,final_lr=1e-4,lr_drop_step=int(.75*steps),
            betas=[.9,.99],weight_decay=1e-4,clip=5.,dropout=.1,blank_bias=0,train_latent='cached_mu',max_updates=steps,max_wall_seconds=900,
            eval_every=1000,selection='independent DEV writer/prompt-disjoint CER then CTC; original32 reporting-only',
            posterior_draws=20,posterior_evaluation_ids=posterior_ids,entire_codec_frozen=True,
            checkpoint_contract='standalone OCR head plus feature contract, not a normal VAE checkpoint',torch_version=str(torch.__version__))
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        for name in ('ocr_pool_study.py','ocr_pool.py','ocr_context_features.py','frozen_ocr_study.py'):
            p=folder/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
        p=folder/'source-code/model/ocr.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
        parity=batch_parity(head_model(head),cache,probe[:16]);(folder/'batch-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=1e-4);torch.manual_seed(42)
        def save(name,step):
            torch.save(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=step,config=settings,
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
        first=evaluate(head,cache,texts,splits,vocab,folder,0,posterior_ids)
        best=(first['groups']['dev']['mu']['cer'],first['groups']['dev']['mean_ctc_loss']);best_step=0;history=[dict(step=0,groups=first['groups'])];save('head-best.pt',0)
        started=time.monotonic();head.train();stop='budget_completed'
        with (folder/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(bucket_schedule(cache,train_ids,steps),1):
                z,labels,mask=collate_latents(cache,ids);optimizer.zero_grad(set_to_none=True);loss=head.get_ocr_loss(z,labels,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite larger-pool CTC; no update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
                if step==int(.75*steps):
                    for g in optimizer.param_groups:g['lr']=1e-4
                limit=time.monotonic()-started>900
                if step%1000==0 or step==steps or limit:
                    row=evaluate(head,cache,texts,splits,vocab,folder,step,posterior_ids);history.append(dict(step=step,groups=row['groups']))
                    score=(row['groups']['dev']['mu']['cer'],row['groups']['dev']['mean_ctc_loss'])
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                    save('head-last.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        if tensor_digest(base.state_dict())!=source_digest or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('codec frozen invariant failed')
        chosen=next(r['groups'] for r in history if r['step']==best_step)
        result=dict(arm=arm,selected=chosen,best_step=best_step,last_step=step,history=history,stop_reason=stop,
            selected_sha256=file_sha(folder/'head-best.pt'),final_sha256=file_sha(folder/'head-last.pt'),entire_codec_bitwise_unchanged=True,
            elapsed_training_evaluation_seconds=time.monotonic()-started,initial_head_tensor_sha256=digest)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result
        del head,optimizer;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={k:dict(best_step=r['best_step'],train_cer=r['selected']['train']['mu']['cer'],dev_cer=r['selected']['dev']['mu']['cer'],held_out_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()},entire_codec_bitwise_unchanged=True)
