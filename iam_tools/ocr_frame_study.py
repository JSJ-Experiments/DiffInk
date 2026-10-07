"""Fresh paired4/8-point OCR frames; initialized transport codec never changes.

Four-point frames contain the SAME chronological XY/pen fields, split from8-point
polyphase blocks. No trajectory interpolation/smoothing or semantic VAE claim.
"""
import hashlib,json,time
from pathlib import Path
import torch
from .ocr_convergence import POOL_SHA
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .frozen_ocr_study import bucket_schedule,collate_latents,batch_parity
from .ocr_context_features import unpack,fit_stats,make_head
from .ocr_context_study import tensor_digest,head_model
from .writer_expansion import load
from .pen_ab import file_sha


def split_tensor(x,frames):
    """Lossless re-indexing of first40 fields; unused fields cannot enter OCR."""
    if frames==8:return x
    if frames!=4 or x.ndim!=3 or x.shape[1]<40:raise ValueError('polyphase40 tensor and4/8 frame size required')
    b,c,t=x.shape
    points=x[:,:40].reshape(b,8,5,t).permute(0,3,1,2).reshape(b,t*8,5)
    split=points.reshape(b,t*2,4,5).reshape(b,t*2,20).transpose(1,2)
    return torch.cat((split,x.new_zeros(b,c-20,t*2)),dim=1)


def frame_cache(cache,frames):
    if frames==8:return cache
    if frames!=4:raise ValueError('4/8 frame size required')
    result={}
    for sid,c in cache.items():
        _,real=unpack(c['mu'],c['mask']);n=int(real.sum());t=(n+3)//4
        mu=split_tensor(c['mu'],4)[:,:,:t];lv=split_tensor(c['lv'],4)[:,:,:t]
        result[sid]=dict(mu=mu,lv=lv,mask=torch.ones(1,t,dtype=torch.bool,device=mu.device),labels=c['labels'])
        f,r=unpack(mu,result[sid]['mask'],4)
        if int(r.sum())!=n:raise AssertionError('four-frame real points changed')
        before=c['mu'][:,:40].reshape(1,8,5,-1).permute(0,3,1,2).reshape(1,-1,5)[:,:n]
        after=f.permute(0,3,1,2).reshape(1,-1,5)[:,:n]
        if not torch.equal(before,after):raise AssertionError('four-frame point fields changed')
    return result


def paired_posterior_sampler(original,framed,frames):
    """Draw once in ORIGINAL384×T8 space then re-index: same valid-field noise."""
    def sample(sid,draws):
        c=original[sid];values=torch.cat([c['mu']+torch.randn_like(c['mu'])*(.5*c['lv']).exp() for _ in range(draws)])
        return split_tensor(values,frames)[:,:,:framed[sid]['mu'].shape[-1]]
    return sample


def run(config,repo,root='/data',pool_sha='',steps=8000):
    if not torch.cuda.is_available() or pool_sha!=POOL_SHA or not 1000<=steps<=8000:raise ValueError('CUDA, fixed pool and bounded1000–8000 updates required')
    root=Path(root);torch.set_num_threads(4);pool,m,vocab=load_pool(root,pool_sha)
    base,_,_,cfg,alphabet,prov=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if alphabet!=vocab:raise ValueError('codec alphabet changed')
    base.cuda().eval().requires_grad_(False);digest=tensor_digest(base.state_dict())
    out=root/'checkpoints/iam_ocr_frame_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'original-codec-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    for name in ('ocr_frame_study.py','ocr_context_features.py','ocr_pool_study.py','ocr_convergence.py','frozen_ocr_study.py'):
        dest=out/'source-code/iam_tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).with_name(name).read_bytes())
    dest=out/'source-code/model/ocr.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
    try:original,texts,audit=cache_corpus(base,pool,m,vocab,out)
    except Exception as exc:
        (out/'failed-preflight.json').write_text(json.dumps(dict(error=str(exc),status='before_any_OCR_update'),indent=2));raise
    train=m['splits']['large_train'];probe=m['splits']['small_train'][::6];posterior=probe+m['splits']['dev']+m['splits']['held_out']
    splits=dict(train=train,dev=m['splits']['dev'],held_out=m['splits']['held_out'],common_train_probe=probe)
    schedule=list(bucket_schedule(original,train,steps,batch_size=16,seed=43));schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    results={};initial_sha=None
    for frames in (8,4):
        arm=f'points{frames}';folder=out/arm;folder.mkdir();cache=frame_cache(original,frames)
        stats=fit_stats(cache,m['splits']['small_train'],'relative_scaled',frames)
        head=make_head(cfg,len(vocab)+1,'relative_scaled',stats,seed=42,points_per_frame=frames).cuda()
        h=tensor_digest(head.state_dict())
        if initial_sha is None:initial_sha=h
        if h!=initial_sha:raise AssertionError('fresh weight pairing failed')
        settings=dict(profile='fresh-frozen-transport-OCR-frame-granularity',cfg=cfg,source_rel=SOURCE,source_sha256=SHA,
            pool_rel=str(pool.relative_to(root)),pool_manifest_sha256=pool_sha,points_per_frame=frames,feature_mode='relative_scaled',feature_stats=stats,
            feature_calibration_ids=m['splits']['small_train'],train_ids=train,dev_ids=splits['dev'],held_out_ids=splits['held_out'],common_train_probe=probe,
            posterior_evaluation_ids=posterior,posterior_draws=20,posterior_policy='draw same original384xT8 noise then chronological re-index; unused noise excluded',
            attention_radius=None,initial_head_tensor_sha256=h,physical_batch=16,encoder_physical_batch=1,seed=42,schedule_seed=43,schedule_sha256=schedule_sha,
            same_sample_ids=True,dropout_rng_not_paired=True,calibration_same_ids_but_refit_per_granularity=True,
            base_lr=5e-4,final_lr=1e-4,lr_drop_step=int(.75*steps),betas=[.9,.99],weight_decay=1e-4,clip=5.,dropout=.1,blank_bias=0,
            max_updates=steps,max_wall_seconds=1800,eval_every=1000,train_latent='cached_mu',entire_codec_frozen=True,
            selection='DEV mean CER then CTC; original32 report-only',
            intervention='points per OCR frame, input grouping, per-resolution TRAIN calibration and positional index granularity; NOT isolated CTC-only capacity',
            checkpoint_contract='standalone OCR feature adapter/head; not a normal VAE checkpoint',torch_version=str(torch.__version__))
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        parity=batch_parity(head_model(head),cache,probe[:16]);(folder/'batch-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=1e-4)
        sampler=paired_posterior_sampler(original,cache,frames)
        def ev(step):return evaluate(head,cache,texts,splits,vocab,folder,step,posterior,posterior_sampler=sampler)
        def save(name,step):torch.save(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=step,config=settings,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
        first=ev(0);best=(first['groups']['dev']['mu']['cer'],first['groups']['dev']['mean_ctc_loss']);best_step=0;history=[dict(step=0,groups=first['groups'])]
        torch.manual_seed(42);save('head-best.pt',0);head.train();started=time.monotonic();stop='budget_completed'
        with (folder/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(schedule,1):
                z,labels,mask=collate_latents(cache,ids);optimizer.zero_grad(set_to_none=True);loss=head.get_ocr_loss(z,labels,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite CTC before update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
                if step==int(.75*steps):
                    for g in optimizer.param_groups:g['lr']=1e-4
                limit=time.monotonic()-started>1800
                if step%1000==0 or step==steps or limit:
                    row=ev(step);history.append(dict(step=step,groups=row['groups']));score=(row['groups']['dev']['mu']['cer'],row['groups']['dev']['mean_ctc_loss'])
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                    save('head-last.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('codec mutation')
        result=dict(arm=arm,best_step=best_step,last_step=step,selected=next(r['groups'] for r in history if r['step']==best_step),history=history,
            selected_sha256=file_sha(folder/'head-best.pt'),final_sha256=file_sha(folder/'head-last.pt'),initial_head_tensor_sha256=h,
            entire_codec_bitwise_unchanged=True,sample_schedule_sha256=schedule_sha,elapsed_training_evaluation_seconds=time.monotonic()-started,stop_reason=stop)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result;del head,optimizer,cache;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={k:dict(best_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()})
