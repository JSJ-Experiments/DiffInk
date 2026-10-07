"""Paired OCR-only continuations from a trained head, moments and RNG intact.

The handwriting codec stays frozen. Both arms restart bucket ordering at seed43;
this is NOT a replay of the parent data iterator. Different batch lengths imply
unpaired dropout after the shared initial RNG state. No evaluation text selects
hyperparameters or supplies training examples.
"""
import copy,hashlib,json,time
from pathlib import Path
import torch
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_pool import assert_extension
from .writer_expansion import load
from .frozen_ocr_study import bucket_schedule,collate_latents,batch_parity
from .ocr_context_features import make_head,fit_stats
from .ocr_context_study import tensor_digest,head_model
from .pen_ab import file_sha

PARENT_POOL='122a428ad0e549aeec9b9f4b67028f48ac6470123234505cd25ac191b041bd0e'
PARENT_HEAD='checkpoints/iam_ocr_pool_study/20261007-033445/large2048/head-best.pt'
PARENT_SHA='073705cc96ae29b3e991fee8820eecca291d20ae57f1b1d597f114c97de5f925'


def state_digest(value):
    """Device-independent fingerprint of nested optimizer/RNG state and scalars."""
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            x=v.detach().cpu().contiguous();h.update(str((str(x.dtype),list(x.shape))).encode());h.update(x.numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):h.update(repr(key).encode());visit(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def restore(head,optimizer,saved,restore_rng=True):
    """Preserve parent LR, moments/counters, buffers and exact CPU/CUDA RNG."""
    head.load_state_dict(saved['ocr_state_dict']);optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    if state_digest(optimizer.state_dict())!=state_digest(saved['optimizer_state_dict']):raise AssertionError('optimizer restore mismatch')
    if tensor_digest(head.state_dict())!=tensor_digest(saved['ocr_state_dict']):raise AssertionError('head restore mismatch')
    if restore_rng:
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([r.cpu() for r in saved['rng_cuda']])
    return dict(head_tensor_sha256=tensor_digest(head.state_dict()),optimizer_sha256=state_digest(optimizer.state_dict()),
        parent_rng_sha256=state_digest([saved['rng_cpu'],saved['rng_cuda']]),learning_rates=[g['lr'] for g in optimizer.param_groups])


def validate_parent(saved,parent,expanded):
    c=saved['config']
    if saved['updates']!=6000 or c['pool_manifest_sha256']!=PARENT_POOL:raise ValueError('expected selected parent step6000/pool')
    if c['source_sha256']!=SHA or c['feature_mode']!='relative_scaled':raise ValueError('source/feature contract changed')
    if c['train_ids']!=parent['splits']['large_train']:raise ValueError('parent training IDs changed')
    if c['feature_calibration_ids']!=parent['splits']['small_train']:raise ValueError('parent calibration changed')
    if c['dev_ids']!=parent['splits']['dev'] or c['held_out_ids']!=parent['splits']['held_out']:raise ValueError('parent evaluation changed')
    if expanded.get('parent_manifest_sha256')!=PARENT_POOL:raise ValueError('expanded manifest must pin parent')
    assert_extension(parent,expanded['records'],expanded['splits'],expanded['vocab_sha256'])
    if expanded['train_writers']!=parent['train_writers']:raise ValueError('writer population changed')
    if len(expanded['splits']['large_train'])<=len(parent['splits']['large_train']):raise ValueError('larger pool required')
    if any(g['lr']!=1e-4 for g in saved['optimizer_state_dict']['param_groups']):raise ValueError('inherited1e-4 LR required')
    return c


def run(config,repo,root='/data',pool_sha='',steps=6000):
    if not torch.cuda.is_available() or not 1000<=steps<=8000:raise ValueError('CUDA and bounded1000–8000 additional steps required')
    torch.set_num_threads(4);root=Path(root);pool,m,vocab=load_pool(root,pool_sha);_,parent,old_vocab=load_pool(root,PARENT_POOL)
    if file_sha(root/PARENT_HEAD)!=PARENT_SHA:raise ValueError('pinned parent head required')
    saved=torch.load(root/PARENT_HEAD,weights_only=True,map_location='cpu');previous=validate_parent(saved,parent,m)
    if len(parent['splits']['large_train'])!=2048 or len(m['splits']['large_train'])!=8192:raise ValueError('this paired study requires2048/8192 TRAIN quotas')
    if vocab!=old_vocab:raise ValueError('fixed alphabet changed')
    base,_,_,cfg,original_vocab,provenance=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if vocab!=original_vocab or cfg!=previous['cfg']:raise ValueError('codec/config drift')
    base.cuda().eval().requires_grad_(False);codec_digest=tensor_digest(base.state_dict())
    out=root/'checkpoints/iam_ocr_pool_expansion'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes())
    (out/'parent-config.json').write_text(json.dumps(previous,indent=2)+'\n')
    (out/'original-codec-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name in ('ocr_pool_expansion.py','ocr_pool_study.py','ocr_pool.py','ocr_context_features.py','frozen_ocr_study.py'):
        p=out/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    p=out/'source-code/model/ocr.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
    try:cache,texts,audit=cache_corpus(base,pool,m,vocab,out)
    except Exception as exc:
        (out/'failed-preflight.json').write_text(json.dumps(dict(status='failed_before_OCR_updates',error=str(exc)),indent=2)+'\n');raise
    stats=previous['feature_stats'];computed=fit_stats(cache,previous['feature_calibration_ids'],'relative_scaled')
    if computed!=stats:raise AssertionError('parent calibration fingerprints/moments changed')
    probe=previous['common_train_probe'];posterior_ids=previous['posterior_evaluation_ids'];results={};shared=None
    for arm,ids in [('control2048',parent['splits']['large_train']),('expanded8192',m['splits']['large_train'])]:
        folder=out/arm;folder.mkdir();splits=dict(train=ids,dev=previous['dev_ids'],held_out=previous['held_out_ids'],common_train_probe=probe)
        head=make_head(cfg,len(vocab)+1,previous['feature_mode'],stats,previous['attention_radius']).cuda()
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
        pairing=restore(head,optimizer,saved)
        if shared is None:shared=pairing
        if pairing!=shared:raise AssertionError('unpaired continuation state')
        settings=dict(previous,profile='paired-frozen-OCR-2048-to8192-continuation',pool_manifest_sha256=pool_sha,
            pool_rel=str(pool.relative_to(root)),parent_pool_manifest_sha256=PARENT_POOL,parent_head_rel=PARENT_HEAD,parent_head_sha256=PARENT_SHA,
            parent_updates=saved['updates'],train_ids=ids,feature_stats=stats,initial_state=pairing,
            initial_head_tensor_sha256=pairing['head_tensor_sha256'],parent_fresh_head_tensor_sha256=previous['initial_head_tensor_sha256'],
            seed_policy='checkpoint CPU/CUDA RNG restored; seed42 only parent/factory, bucket seed43',max_additional_updates=steps,
            max_updates=saved['updates']+steps,base_lr=1e-4,final_lr=1e-4,lr_drop_step=None,
            schedule_seed=43,schedule_policy='restart both bucket schedules with seed43; parent data iterator is NOT resumed',
            shared_initial_rng=True,samples_and_dropout_not_paired=True,max_wall_seconds=900)
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        parity=batch_parity(head_model(head),cache,probe[:16]);(folder/'batch-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        first=evaluate(head,cache,texts,splits,vocab,folder,saved['updates'],posterior_ids)
        # Reset after all setup/evaluation so both first optimizer updates start
        # from EXACT same saved RNG, not newly constructed-head or eval RNG.
        if restore(head,optimizer,saved)!=pairing:raise AssertionError('post-evaluation state mismatch')
        def save(name,step):
            torch.save(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),updates=step,config=settings,
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),folder/name)
        best=(first['groups']['dev']['mu']['cer'],first['groups']['dev']['mean_ctc_loss']);best_step=saved['updates'];history=[dict(step=best_step,groups=first['groups'])]
        save('head-best.pt',best_step);head.train();started=time.monotonic();stop='budget_completed'
        with (folder/'metrics.jsonl').open('w') as log:
            for additional,batch in enumerate(bucket_schedule(cache,ids,steps,seed=43),1):
                step=saved['updates']+additional;z,labels,mask=collate_latents(cache,batch);optimizer.zero_grad(set_to_none=True)
                loss=head.get_ocr_loss(z,labels,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite CTC; no update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,additional_step=additional,sample_ids=batch,loss=float(loss.detach()),raw_gradient_norm=norm,
                    was_clipped=norm>5,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
                limit=time.monotonic()-started>900
                if additional%1000==0 or additional==steps or limit:
                    row=evaluate(head,cache,texts,splits,vocab,folder,step,posterior_ids);history.append(dict(step=step,groups=row['groups']))
                    score=(row['groups']['dev']['mu']['cer'],row['groups']['dev']['mean_ctc_loss'])
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                    save('head-last.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        if tensor_digest(base.state_dict())!=codec_digest or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('codec mutation')
        result=dict(arm=arm,selected=next(r['groups'] for r in history if r['step']==best_step),best_step=best_step,last_step=step,
            additional_updates=additional,history=history,stop_reason=stop,selected_sha256=file_sha(folder/'head-best.pt'),final_sha256=file_sha(folder/'head-last.pt'),
            entire_codec_bitwise_unchanged=True,elapsed_training_evaluation_seconds=time.monotonic()-started,initial_state=pairing)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result;del head,optimizer;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={k:dict(best_step=r['best_step'],train_cer=r['selected']['train']['mu']['cer'],dev_cer=r['selected']['dev']['mu']['cer'],held_out_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()})
