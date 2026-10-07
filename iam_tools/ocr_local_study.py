"""Matched OCR continuation: original features versus identity-initialized local context."""
import hashlib,json,time
from pathlib import Path
import torch
from .ocr_augmentation import PARENT,PARENT_SHA,PARENT_UPDATES,validate_parent
from .ocr_local_context import make_head,attach_context,SPEC
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_convergence import POOL_SHA,resumed_schedule
from .ocr_frame_study import frame_cache,paired_posterior_sampler
from .ocr_context_features import fit_stats
from .ocr_context_study import tensor_digest,head_model
from .ocr_pool_expansion import restore,state_digest
from .frozen_ocr_study import collate_latents,batch_parity
from .writer_expansion import load
from .pen_ab import file_sha
ARMS=('clean_control','local_context')


def run(config,repo,root='/data',pool_sha='',steps=6000):
    if not torch.cuda.is_available() or pool_sha!=POOL_SHA or type(steps) is not int or not 1000<=steps<=6000:raise ValueError('CUDA, pinned pool and1000–6000 additional steps required')
    root=Path(root);torch.set_num_threads(4);pool,m,vocab=load_pool(root,pool_sha)
    if file_sha(root/PARENT)!=PARENT_SHA:raise ValueError('pinned4-point parent fingerprint required')
    saved=torch.load(root/PARENT,weights_only=True,map_location='cpu');previous=validate_parent(saved,m)
    base,_,_,cfg,alphabet,prov=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if alphabet!=vocab or cfg!=previous['cfg']:raise ValueError('source/vocab/config drift')
    base.cuda().eval().requires_grad_(False);digest=tensor_digest(base.state_dict())
    out=root/'checkpoints/iam_ocr_local_context'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'parent-config.json').write_text(json.dumps(previous,indent=2)+'\n');(out/'codec-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    for name in ('ocr_local_study.py','ocr_local_context.py','ocr_augmentation.py','ocr_error_analysis.py','ocr_frame_study.py','ocr_context_features.py','ocr_pool_study.py','frozen_ocr_study.py','ocr_convergence.py','ocr_pool_expansion.py'):
        p=out/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    p=out/'source-code/model/ocr.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
    try:original,texts,audit=cache_corpus(base,pool,m,vocab,out)
    except Exception as exc:
        (out/'failed-preflight.json').write_text(json.dumps(dict(status='before_any_OCR_update',error=str(exc)),indent=2)+'\n');raise
    cache=frame_cache(original,4);stats=fit_stats(cache,previous['feature_calibration_ids'],'relative_scaled',4)
    if stats!=previous['feature_stats']:raise AssertionError('parent clean calibration changed')
    schedule=list(resumed_schedule(original,previous['train_ids'],steps,skip=PARENT_UPDATES,seed=43,batch_size=16));schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    splits=dict(train=previous['train_ids'],dev=previous['dev_ids'],held_out=previous['held_out_ids'],common_train_probe=previous['common_train_probe'])
    sampler=paired_posterior_sampler(original,cache,4);results={};shared=None;end_rng=None
    for arm in ARMS:
        folder=out/arm;folder.mkdir();head=make_head(cfg,len(vocab)+1,stats,context=arm=='local_context',seed=42,attach=False).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4);initial=restore(head,opt,saved);initial['local_branch_attachment']=attach_context(head,opt)
        if shared is None:shared=initial
        if initial!=shared:raise AssertionError('parent head/moments/RNG pairing changed')

        settings=dict(profile='paired4-point-OCR-only-local-context-continuation',cfg=cfg,parent_rel=PARENT,parent_sha256=PARENT_SHA,parent_updates=PARENT_UPDATES,source_rel=SOURCE,source_sha256=SHA,pool_rel=str(pool.relative_to(root)),pool_manifest_sha256=pool_sha,
            points_per_frame=4,feature_mode='relative_scaled',feature_stats=stats,feature_calibration_ids=previous['feature_calibration_ids'],train_ids=previous['train_ids'],dev_ids=previous['dev_ids'],held_out_ids=previous['held_out_ids'],common_train_probe=previous['common_train_probe'],posterior_evaluation_ids=previous['posterior_evaluation_ids'],posterior_draws=20,
            physical_batch=16,encoder_physical_batch=1,attention_radius=None,dropout=.1,blank_bias=0,base_lr=1e-4,betas=[.9,.99],weight_decay=1e-4,clip=5.,seed=42,schedule_seed=43,schedule_skip=PARENT_UPDATES,schedule_sha256=schedule_sha,
            local_context=arm=='local_context',local_definition=SPEC,objective='0.5clean+0.5clean;two identical forwards in BOTH arms',
            max_updates=steps,max_wall_seconds=1800,eval_every=1000,train_latent='cached_mu',initial_state=initial,entire_codec_frozen=True,selection='cleanDEV mean CER thenCTC only;32report-only',checkpoint_contract='standalone4-point transport feature reader,NOT ordinaryVAE',torch_version=str(torch.__version__))
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n');(folder/'batch-parity.json').write_text(json.dumps(batch_parity(head_model(head),cache,previous['common_train_probe'][:16]),indent=2)+'\n')
        def ev(step):return evaluate(head,cache,texts,splits,vocab,folder,step,previous['posterior_evaluation_ids'],posterior_sampler=sampler)
        def save(name,step):torch.save(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=opt.state_dict(),updates=PARENT_UPDATES+step,additional_updates=step,config=settings,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
        first=ev(0);best=(first['groups']['dev']['mu']['cer'],first['groups']['dev']['mean_ctc_loss']);best_step=0;history=[dict(step=0,groups=first['groups'])];save('head-best.pt',0)
        started=time.monotonic();stop='budget_completed';head.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(schedule,1):
                mu,labels,mask=collate_latents(cache,ids)
                opt.zero_grad(set_to_none=True);loss=.5*head.get_ocr_loss(mu,labels,mask)+.5*head.get_ocr_loss(mu,labels,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite localCTC before update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));opt.step()
                log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5,lr=opt.param_groups[0]['lr']))+'\n');log.flush()
                limit=time.monotonic()-started>1800
                if step%1000==0 or step==steps or limit:
                    row=ev(step);history.append(dict(step=step,groups=row['groups']));score=(row['groups']['dev']['mu']['cer'],row['groups']['dev']['mean_ctc_loss'])
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                    save('head-last.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('codec mutation')
        rng=state_digest([torch.get_rng_state(),torch.cuda.get_rng_state_all()])
        if end_rng is None:end_rng=rng
        elif step==steps and results['clean_control']['last_step']==steps and end_rng!=rng:raise AssertionError('dropout stream pairing drift')
        r=dict(arm=arm,last_step=step,best_step=best_step,selected=next(h['groups'] for h in history if h['step']==best_step),history=history,stop_reason=stop,entire_codec_bitwise_unchanged=True,initial_state=initial,selected_sha256=file_sha(folder/'head-best.pt'),final_sha256=file_sha(folder/'head-last.pt'),sample_schedule_sha256=schedule_sha,final_dropout_rng_sha256=rng,elapsed_training_evaluation_seconds=time.monotonic()-started)
        results[arm]=r;(folder/'result.json').write_text(json.dumps(r,indent=2)+'\n');del head,opt;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(best_additional_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for a,r in results.items()})
