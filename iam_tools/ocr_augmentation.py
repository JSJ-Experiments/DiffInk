"""OCR-only affine robustness on4-point initialized transport; NOT trajectory training."""
import copy,hashlib,json,time
from pathlib import Path
import torch
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_convergence import POOL_SHA,resumed_schedule
from .ocr_frame_study import frame_cache,paired_posterior_sampler
from .ocr_context_features import unpack,make_head,fit_stats
from .ocr_context_study import tensor_digest,head_model
from .ocr_pool_expansion import restore,state_digest
from .frozen_ocr_study import collate_latents,batch_parity
from .writer_expansion import load
from .pen_ab import file_sha

PARENT='checkpoints/iam_ocr_frame_study/20261007-074413/points4/head-best.pt'
PARENT_SHA='937a09830c15757c0ba9a02982497aff0f88ffbc66ac2b8fb10e72063e367a79'
PARENT_UPDATES=8000
LIMITS=dict(x_scale=[.9,1.1],y_scale=[.9,1.1],x_shear=[-.12,.12],y_shift=[-.04,.04],y_anchor=.5)
ARMS=('clean_control','affine_mix')


def validate_parent(saved,pool):
    c=saved['config']
    if saved['updates']!=PARENT_UPDATES or c['source_sha256']!=SHA or c['pool_manifest_sha256']!=POOL_SHA:raise ValueError('pinned step8000 frozen-codec pool parent required')
    if c['points_per_frame']!=4 or c['feature_mode']!='relative_scaled' or c['attention_radius'] is not None:raise ValueError('four-point relative-scaled global reader required')
    for field,key in [('train_ids','large_train'),('dev_ids','dev'),('held_out_ids','held_out'),('feature_calibration_ids','small_train')]:
        if c[field]!=pool['splits'][key]:raise ValueError('split or calibration drift: '+field)
    probe=pool['splits']['small_train'][::6]
    if c['common_train_probe']!=probe or c['posterior_evaluation_ids']!=probe+c['dev_ids']+c['held_out_ids']:raise ValueError('evaluation probe drift')
    if c['schedule_seed']!=43 or c['physical_batch']!=16 or c['seed']!=42:raise ValueError('parent RNG/data contract drift')
    for g in saved['optimizer_state_dict']['param_groups']:
        if g['lr']!=1e-4 or tuple(g['betas'])!=(.9,.99) or g['weight_decay']!=1e-4:raise ValueError('parent optimizer contract changed')
    return c


def draw_parameters(batch,generator):
    if type(batch) is not int or batch<1:raise ValueError('positive batch size required')
    if not isinstance(generator,torch.Generator) or generator.device.type!='cpu':raise ValueError('dedicated CPU generator required')
    # Dedicated CPU generator: never consume model/dropout or evaluation RNG.
    v=torch.rand(batch,4,generator=generator)
    bounds=v.new_tensor([LIMITS[k] for k in ('x_scale','y_scale','x_shear','y_shift')])
    return bounds[:,0]+v*(bounds[:,1]-bounds[:,0])


def augment_transport(x,mask,parameters):
    """x'=sx*x+shear*(y-.5); y'=.5+sy*(y-.5)+shift, only real phases.

    Positive determinant, constant affine per line, no time reindex/rotation,
    interpolation or generic smoothing; exact pen values/valid counts unchanged.
    First20fields only; unused latent noise and paddedNaNs cannot enter OCR.
    """
    fields,real=unpack(x,mask,4)
    if parameters.shape!=(x.shape[0],4) or not torch.isfinite(parameters).all() or (parameters[:,:2]<=0).any():raise ValueError('finite per-line affine coefficients/positive scales required')
    p=parameters.to(device=x.device,dtype=x.dtype);sx,sy,shear,shift=[p[:,i,None,None] for i in range(4)];anchor=LIMITS['y_anchor']
    out=fields.clone();y=fields[:,:,1]
    out[:,:,0]=torch.where(real,sx*fields[:,:,0]+shear*(y-anchor),fields[:,:,0])
    out[:,:,1]=torch.where(real,anchor+sy*(y-anchor)+shift,y)
    return torch.cat((out.reshape(x.shape[0],20,x.shape[-1]),torch.zeros_like(x[:,20:])),dim=1)


def mixed_loss(head,mu,labels,mask,augmented,use_augmentation):
    """Two identical-shaped forwards in BOTH arms: dropout stream can be paired."""
    return .5*head.get_ocr_loss(mu,labels,mask)+.5*head.get_ocr_loss(augmented if use_augmentation else mu,labels,mask)


def run(config,repo,root='/data',pool_sha='',steps=4000):
    if not torch.cuda.is_available() or pool_sha!=POOL_SHA or type(steps) is not int or not 1000<=steps<=4000:raise ValueError('CUDA, pinned pool and1000–4000 additional steps required')
    root=Path(root);torch.set_num_threads(4);pool,m,vocab=load_pool(root,pool_sha)
    if file_sha(root/PARENT)!=PARENT_SHA:raise ValueError('pinned4-point parent fingerprint required')
    saved=torch.load(root/PARENT,weights_only=True,map_location='cpu');previous=validate_parent(saved,m)
    base,_,_,cfg,alphabet,prov=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if alphabet!=vocab or cfg!=previous['cfg']:raise ValueError('source/vocab/config drift')
    base.cuda().eval().requires_grad_(False);digest=tensor_digest(base.state_dict())
    out=root/'checkpoints/iam_ocr_augmentation'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    (out/'pool-manifest.json').write_bytes((pool/'manifest.json').read_bytes());(out/'parent-config.json').write_text(json.dumps(previous,indent=2)+'\n');(out/'codec-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    for name in ('ocr_augmentation.py','ocr_error_analysis.py','ocr_frame_study.py','ocr_context_features.py','ocr_pool_study.py','frozen_ocr_study.py','ocr_convergence.py','ocr_pool_expansion.py'):
        p=out/'source-code/iam_tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).with_name(name).read_bytes())
    p=out/'source-code/model/ocr.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
    try:original,texts,audit=cache_corpus(base,pool,m,vocab,out)
    except Exception as exc:
        (out/'failed-preflight.json').write_text(json.dumps(dict(status='before_any_OCR_update',error=str(exc)),indent=2)+'\n');raise
    cache=frame_cache(original,4);stats=fit_stats(cache,previous['feature_calibration_ids'],'relative_scaled',4)
    if stats!=previous['feature_stats']:raise AssertionError('parent clean calibration changed')
    schedule=list(resumed_schedule(original,previous['train_ids'],steps,skip=PARENT_UPDATES,seed=43,batch_size=16));schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    splits=dict(train=previous['train_ids'],dev=previous['dev_ids'],held_out=previous['held_out_ids'],common_train_probe=previous['common_train_probe'])
    sampler=paired_posterior_sampler(original,cache,4);results={};shared=None;end_rng=None;end_aug=None
    for arm in ARMS:
        folder=out/arm;folder.mkdir();head=make_head(cfg,len(vocab)+1,'relative_scaled',stats,seed=42,points_per_frame=4).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4);initial=restore(head,opt,saved)
        if shared is None:shared=initial
        if initial!=shared:raise AssertionError('parent head/moments/RNG pairing changed')
        gen=torch.Generator(device='cpu').manual_seed(901)
        settings=dict(profile='paired4-point-OCR-only-affine-continuation',cfg=cfg,parent_rel=PARENT,parent_sha256=PARENT_SHA,parent_updates=PARENT_UPDATES,source_rel=SOURCE,source_sha256=SHA,pool_rel=str(pool.relative_to(root)),pool_manifest_sha256=pool_sha,
            points_per_frame=4,feature_mode='relative_scaled',feature_stats=stats,feature_calibration_ids=previous['feature_calibration_ids'],train_ids=previous['train_ids'],dev_ids=previous['dev_ids'],held_out_ids=previous['held_out_ids'],common_train_probe=previous['common_train_probe'],posterior_evaluation_ids=previous['posterior_evaluation_ids'],posterior_draws=20,
            physical_batch=16,encoder_physical_batch=1,attention_radius=None,dropout=.1,blank_bias=0,base_lr=1e-4,betas=[.9,.99],weight_decay=1e-4,clip=5.,seed=42,schedule_seed=43,schedule_skip=PARENT_UPDATES,schedule_sha256=schedule_sha,
            augmentation_seed=901,augmentation_limits=LIMITS,use_augmentation=arm=='affine_mix',objective='0.5clean+0.5augmented' if arm=='affine_mix' else '0.5clean+0.5clean;two forwards match dropout RNG',
            max_updates=steps,max_wall_seconds=1200,eval_every=1000,train_latent='cached_mu',initial_state=initial,entire_codec_frozen=True,selection='cleanDEV mean CER thenCTC only;32report-only',checkpoint_contract='standalone4-point transport feature reader,NOT ordinaryVAE',torch_version=str(torch.__version__))
        (folder/'config.json').write_text(json.dumps(settings,indent=2)+'\n');(folder/'batch-parity.json').write_text(json.dumps(batch_parity(head_model(head),cache,previous['common_train_probe'][:16]),indent=2)+'\n')
        def ev(step):return evaluate(head,cache,texts,splits,vocab,folder,step,previous['posterior_evaluation_ids'],posterior_sampler=sampler)
        def save(name,step):torch.save(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=opt.state_dict(),updates=PARENT_UPDATES+step,additional_updates=step,config=settings,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),augmentation_rng=gen.get_state()),folder/name)
        first=ev(0);best=(first['groups']['dev']['mu']['cer'],first['groups']['dev']['mean_ctc_loss']);best_step=0;history=[dict(step=0,groups=first['groups'])];save('head-best.pt',0)
        started=time.monotonic();stop='budget_completed';head.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(schedule,1):
                mu,labels,mask=collate_latents(cache,ids);parameters=draw_parameters(len(ids),gen);aug=augment_transport(mu,mask,parameters)
                opt.zero_grad(set_to_none=True);loss=mixed_loss(head,mu,labels,mask,aug,arm=='affine_mix')
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite augmentedCTC before update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));opt.step()
                log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5,lr=opt.param_groups[0]['lr'],affine_parameters_sha256=state_digest(parameters)))+'\n');log.flush()
                limit=time.monotonic()-started>1200
                if step%1000==0 or step==steps or limit:
                    row=ev(step);history.append(dict(step=step,groups=row['groups']));score=(row['groups']['dev']['mu']['cer'],row['groups']['dev']['mean_ctc_loss'])
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                    save('head-last.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('codec mutation')
        rng=state_digest([torch.get_rng_state(),torch.cuda.get_rng_state_all()]);aug_rng=state_digest(gen.get_state())
        if end_rng is None:end_rng=rng;end_aug=aug_rng
        elif step==steps and results['clean_control']['last_step']==steps and (end_rng!=rng or end_aug!=aug_rng):raise AssertionError('dropout/augmentation stream pairing drift')
        r=dict(arm=arm,last_step=step,best_step=best_step,selected=next(h['groups'] for h in history if h['step']==best_step),history=history,stop_reason=stop,entire_codec_bitwise_unchanged=True,initial_state=initial,selected_sha256=file_sha(folder/'head-best.pt'),final_sha256=file_sha(folder/'head-last.pt'),sample_schedule_sha256=schedule_sha,final_dropout_rng_sha256=rng,final_augmentation_rng_sha256=aug_rng,elapsed_training_evaluation_seconds=time.monotonic()-started)
        results[arm]=r;(folder/'result.json').write_text(json.dumps(r,indent=2)+'\n');del head,opt;torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(out),arms={a:dict(best_additional_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for a,r in results.items()})
