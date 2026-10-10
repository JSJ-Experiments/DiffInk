"""Native WSL corpus conditioning study; never allocates a Modal GPU.

Fresh AMD concat/joint controls, compact40 transport, pure-text training.
Joint attention is a HYPOTHESIS, not an established fix for composition.
"""
import argparse,copy,hashlib,json,os,shutil,sys,tarfile,time
from pathlib import Path
import numpy as np,torch,h5py
from .compact_dit import loss,X0Adapter
from .compact_dit_study import load_cache,unused_gate,denoising
from .corpus_dit_study import PosteriorPool,schedule,evaluate
from .corpus_dit import learning_rate
from .pen_ab import file_sha
from .ocr_context_study import tensor_digest
from .stroke_diagnostics import compare,aggregate
from .launch_ledger import study_path

_repo=Path(__file__).resolve().parents[1]/'third_party/DiffInk'
if not _repo.is_dir():_repo=Path(__file__).resolve().parents[1]
sys.path.append(str(_repo))

PARENT='checkpoints/iam_corpus_dit/20261008-223316'
ARMS=('concat','joint')
MODEL=dict(dim=384,depth=8,heads=6,dim_head=64,latent_dim=40,text_dim=192,num_text_embedding=81,text_mask_padding=True,conv_layers=3,ff_mult=4,dropout=.05,long_skip_connection=False)


def models(config,seed):
    """Share all analogous trajectory/text/time tensors, not identical models.

    Joint adds context projections/FFNs and has MORE parameters. A win cannot
    be attributed to cross-attention alone rather than that capacity change.
    """
    from model.dit import DiT,CrossAttentionDiT
    from utils.utils import ModelConfig
    torch.manual_seed(seed);a=DiT(ModelConfig(config));b=CrossAttentionDiT(ModelConfig(config))
    old=a.state_dict();new=b.state_dict();copied=[]
    for k,v in new.items():
        source=k.replace('.attn_norm_x.','.attn_norm.').replace('.ff_norm_x.','.ff_norm.').replace('.ff_x.','.ff.')
        if k=='input_embed.proj.weight':new[k]=old[k][:,:config['latent_dim']].clone();copied.append(k)
        elif k=='text_embed.text_embed.weight':
            new[k][:len(old[k])]=old[k];copied.append(k)
        elif source in old and old[source].shape==v.shape:new[k]=old[source].clone();copied.append(k)
    # Initialize the new text projection from the original text input columns.
    # Bias zero; latent projection retains the original input bias.
    if 'text_proj.weight' in new:
        new['text_proj.weight']=old['input_embed.proj.weight'][:,config['latent_dim']:].clone();new['text_proj.bias'].zero_();copied+=['text_proj.weight','text_proj.bias']
    b.load_state_dict(new)
    return a,b,dict(shared_or_mapped_keys=copied,concat_parameters=sum(p.numel() for p in a.parameters()),joint_parameters=sum(p.numel() for p in b.parameters()),not_same_architecture_or_parameter_count=True)


def matched_batch(pool,ids,step,seed,noise_steps):
    """Step-local DATA generator independent of either model's dropout RNG.

    Shared full384 posterior and diffusion draws BEFORE slicing. Shape/order
    identical across arms. Dropout differs architecturally; not called matched.
    """
    if not isinstance(step,int) or step<1:raise ValueError('positive actual step required')
    device=pool.mu.device;g=torch.Generator(device=device).manual_seed(seed+step)
    idx=torch.tensor([pool.index[i] for i in ids],device=device);n=max(pool.lengths[i] for i in ids);nt=max(pool.chars[i] for i in ids)
    mu=pool.mu.index_select(0,idx)[:,:n];std=pool.std.index_select(0,idx)[:,:n]
    clean=mu+std*torch.randn(mu.shape,device=device,generator=g)
    t=torch.randint(noise_steps,(len(ids),),device=device,generator=g)
    eps=torch.randn(clean.transpose(1,2).shape,device=device,generator=g).transpose(1,2)
    drop=float(torch.rand((),device=device,generator=g))<.1
    return clean,pool.text.index_select(0,idx)[:,:nt],pool.mask.index_select(0,idx)[:,:n],pool.prefix.index_select(0,idx),t,eps,drop


def validate(cfg):
    if cfg['arms']!=list(ARMS) or cfg['model']['latent_dim']!=40 or cfg['parent_relative']!=PARENT:raise ValueError('declared native two-arm corpus study required')
    if cfg['prefix_keep_probability']!=0 or cfg['text_drop_probability']!=.1 or cfg['open_confirmations'] or cfg['use_amp']:raise ValueError('pure text/no confirmations/float32 control required')
    if cfg['max_updates']!=12000 or cfg['max_train_wall_seconds']!=1800 or cfg['batch']!=32 or cfg['eval_steps']!=[0,1000,3000,6000,12000]:raise ValueError('predeclared bounded corpus/evaluation budget required')
    if cfg.get('reader_backend')!='aten_gpu_no_miopen':raise ValueError('explicit native reader kernel contract required')
    if cfg['backend']!='AMD ROCm WSL2; fresh controls, NOT T4 numerical equivalence':raise ValueError('backend provenance required')
    if (cfg['lr'],cfg['min_lr'],cfg['warmup_updates'],cfg['clip'],cfg['sampling_steps'],cfg['eval_noise_seeds'],cfg['eval_guidance'])!=(5e-5,1e-6,600,1.,50,[73142,73143],[1.,2.]):raise ValueError('matched optimizer and target-free evaluation required')
    if any(not cfg[k] for k in ['no_codec_training','no_kl','no_ctc_training','no_style_training','not_promoted']):raise ValueError('frozen nonsemantic engineering scope required')
    if cfg['quality_failures']!=['unreadable text','severe underlifting','no reliable content sensitivity']:raise ValueError('content/stroke quality gates required')


def verify(root,p,cfg):
    from .ocr_pool_study import load_pool
    from .corpus_dit_contract import validate_scope
    root=Path(root);validate(cfg)
    for relative,sha in cfg['input_hashes'].items():
        if file_sha(root/relative)!=sha:raise ValueError('missing/partial/mutated migrated input '+relative)
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256'] or file_sha(p/'dataset.json')!=cfg['dataset_sha256']:raise ValueError('native prepared source/metadata drift')
    pool,m,vocab=load_pool(root,cfg['pool_manifest_sha256']);data=json.loads((p/'dataset.json').read_text())
    if vocab!=cfg['vocab'] or str(pool.relative_to(root))!=cfg['pool_relative'] or any(m['records'].get(i)!=r for i,r in data['records'].items()):raise ValueError('immutable corpus identity drift')
    history=json.loads((root/cfg['history_relative']).read_text())['records'];validate_scope(data['records'],data['scope']['splits'],history,m['test_writers'],m['dev_writers'])
    with tarfile.open(p/'as-run-source.tar.gz') as a:
        for path in cfg['verified_source_paths']:
            f=Path(__file__).parent/path.split('/')[-1] if path.startswith('iam_tools/') else Path('third_party/DiffInk')/path
            if a.extractfile(path).read()!=f.read_bytes():raise ValueError('native runtime source drift '+path)
    return data


def prepare(root='data',repo='third_party/DiffInk'):
    root=Path(root);parent=root/PARENT;old=json.loads((parent/'config.json').read_text());audit=json.loads((parent/'baseline/codec-cache-audit.json').read_text());cfg=copy.deepcopy(old)
    cfg.update(profile='Native AMD corpus concat versus joint text conditioning; pure text/compact40/frozen initialized transport, NOT semantic VAE/reproduction',parent_relative=PARENT,arms=list(ARMS),model=dict(old['model'],latent_dim=40,use_cross_attention=False),prefix_keep_probability=0.,open_confirmations=False,use_amp=False,backend='AMD ROCm WSL2; fresh controls, NOT T4 numerical equivalence',init_seed=84010,training_draw_seed=84012,model_rng_seed=84013,quality_failures=['unreadable text','severe underlifting','no reliable content sensitivity'],source_volume_read_only=False,output_volume='native E:/autowrite-data',historical_source_archive_not_executed=True,reader_backend='aten_gpu_no_miopen')
    cfg['input_hashes']={PARENT+'/dataset.json':old['dataset_sha256'],PARENT+'/as-run-source.tar.gz':old['source_archive_sha256'],PARENT+'/baseline/posterior-cache.h5':audit['cache_sha256'],PARENT+'/baseline/whitening.pt':audit['whitening_sha256'],old['codec_relative']:old['codec_sha256'],old['reader_relative']:old['reader_sha256'],old['history_relative']:old['history_sha256']}
    for f,sha in cfg['input_hashes'].items():
        if file_sha(root/f)!=sha:raise ValueError('wait for checksum-verified migration '+f)
    cfg['verified_source_paths']=['iam_tools/wsl_conditioning.py','iam_tools/rocm_reader.py','iam_tools/compact_dit.py','iam_tools/compact_dit_study.py','iam_tools/corpus_dit.py','iam_tools/corpus_dit_study.py','iam_tools/corpus_dit_contract.py','iam_tools/stroke_diagnostics.py','model/dit.py','model/modules.py','model/diffusion.py','model/vae.py','model/blocks.py']
    p=root/'checkpoints/iam_wsl_conditioning'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());p.mkdir(parents=True,exist_ok=False);shutil.copyfile(parent/'dataset.json',p/'dataset.json')
    with tarfile.open(p/'as-run-source.tar.gz','w:gz') as a:
        for directory in ['iam_tools','model','utils']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for f in sorted(base.rglob('*.py')):
                if '__pycache__' not in f.parts:a.add(f,arcname=directory+'/'+str(f.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(p/'as-run-source.tar.gz');(p/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');verify(root,p,cfg);return str(p.relative_to(root))


def stroke_audit(root,p,cfg,data,arm,step):
    folder=p/arm;ev=json.loads((folder/f'eval-{step}.json').read_text());rows=[]
    if file_sha(folder/f'evaluation-{step}.h5')!=ev['packed_h5_sha256']:raise ValueError('saved evaluation drift')
    with h5py.File(Path(root)/cfg['pool_relative']/'lines.h5') as source,h5py.File(folder/f'evaluation-{step}.h5') as hf:
        for r in ev['rows']:
            g=hf[f'{r["split"]}/{r["seed"]}/{r["guidance"]}/{r["policy"]}/{r["sample_id"]}'];q=g['points'][:];truth=source[r['sample_id']]['point_seq'][:]
            if json.loads(g.attrs['row'])!=r or hashlib.sha256(truth.tobytes()).hexdigest()!=data['records'][r['sample_id']]['points_sha256']:raise ValueError('saved row/source geometry drift')
            rows.append(dict(split=r['split'],policy=r['policy'],guidance=r['guidance'],seed=r['seed'],sample_id=r['sample_id'],**compare(q,truth,r['characters'],8*r['estimated_blocks'],r['found_eoc'])))
    groups={}
    for r in rows:groups.setdefault(f'{r["split"]}/g{r["guidance"]}/{r["policy"]}',[]).append(r)
    out=dict(step=step,arm=arm,rows=rows,groups={k:aggregate(v) for k,v in groups.items()},offline_counts_not_generation_inputs=True)
    (folder/f'stroke-audit-{step}.json').write_text(json.dumps(out,indent=2)+'\n');return out


def preflight(device='cuda'):
    """Actual full-width optimizer/CFG/NULL on this backend; NOT handwriting evidence."""
    torch.set_num_threads(2);cfg=dict(MODEL,dropout=0.);a,b,identity=models(cfg,84010);rows=[]
    for name,m in zip(ARMS,[a,b]):
        m=m.to(device).train();opt=torch.optim.AdamW(m.parameters(),lr=1e-3);g=torch.Generator(device=device).manual_seed(3)
        x=torch.randn(2,16,40,device=device,generator=g);target=torch.randn(x.shape,device=device,generator=g);text=torch.tensor([[1,2,3,4],[4,3,2,-1]],device=device);mask=torch.ones(2,16,device=device,dtype=torch.bool);time_input=torch.tensor([.5,.99],device=device);norms=[]
        for _ in range(8):
            opt.zero_grad(set_to_none=True);out=m(x,x,text,time_input,mask=mask);value=(out-target).square().mean();value.backward();n=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True));opt.step();norms.append(float(m.text_embed.text_embed.weight.grad.norm()))
            if not torch.isfinite(value) or not np.isfinite(n):raise ValueError('nonfinite actual attention preflight')
        if max(norms[1:])<=0:raise ValueError('text never received a gradient')
        m.eval()
        with torch.no_grad():
            normal=m(x,x,text,time_input,mask=mask);null=m(x,x,text,time_input,mask=mask,drop_text=True);swapped=m(x,x,text.roll(1,1),time_input,mask=mask)
            if not torch.isfinite(null).all() or float((normal-swapped).abs().max())<=0:raise ValueError('text controls not finite/sensitive')
        rows.append(dict(arm=name,loss=float(value.detach()),text_gradient_norms=norms,correct_null_max_delta=float((normal-null).abs().max()),correct_rotated_max_delta=float((normal-swapped).abs().max())))
        del m,opt;torch.cuda.empty_cache() if str(device).startswith('cuda') else None
    return dict(torch=torch.__version__,hip=torch.version.hip,gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,identity=identity,rows=rows,synthetic_only=True,no_composition_claim=True,no_experiment_launched=True)


def run(relative,arm,root='data'):
    if not torch.cuda.is_available() or not torch.version.hip or arm not in ARMS:raise ValueError('explicit native ROCm and declared arm required')
    from model.vae import VAE
    from model.diffusion import Diffusion
    from utils.utils import ModelConfig
    from .ocr_joint_adapter import load_reader
    from .resource_monitor import ResourceMonitor
    from .rocm_reader import ATenFrozenReader
    root=Path(root);p=study_path(root,relative,'checkpoints/iam_wsl_conditioning/');cfg=json.loads((p/'config.json').read_text());data=verify(root,p,cfg);folder=p/arm;folder.mkdir(exist_ok=False);torch.set_num_threads(2)
    actual=dict(cfg,model=dict(cfg['model'],use_cross_attention=arm=='joint'));saved=torch.load(root/cfg['codec_relative'],map_location='cpu',weights_only=True);codec=VAE(ModelConfig(saved['config']));codec.load_state_dict(saved['model_state_dict']);codec.apply_checkpoint_contract(saved);codec=codec.cuda().eval().requires_grad_(False);cd=tensor_digest(codec.state_dict());sr=torch.load(root/cfg['reader_relative'],map_location='cpu',weights_only=True);reader,_=load_reader(root,sr['config']['cfg'],'cuda');reader=ATenFrozenReader(reader.requires_grad_(False));rd=tensor_digest(reader.state_dict())
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=15360,gpu=False,interval=2,sustained_seconds=45) as monitor:
        # NVIDIA sampler intentionally off: no invented AMD utilization readings.
        with monitor.in_phase('load_cache'):items,stats=load_cache(root/cfg['parent_relative'],data)
        with monitor.in_phase('source_codec_gate'):
            gate=unused_gate(codec,items,stats,cfg['eval_train_ids']+cfg['eval_dev_ids']);(folder/'unused-decoder-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        pool=PosteriorPool(items,data['records'],cfg['vocab'],data['scope']['splits']['train'],stats,'cuda');a,b,identity=models(cfg['model'],cfg['init_seed']);model=a if arm=='concat' else b;initial=tensor_digest(model.state_dict());del a,b;model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);diff=Diffusion(noise_steps=1000,schedule_type='cosine',device='cuda');adapter=X0Adapter(model,diff.alpha_hat,40,'x0',1000.).cuda();batches=list(schedule(pool.ids,pool.lengths,12000,32,cfg['schedule_seed']));order=hashlib.sha256(json.dumps(batches).encode()).hexdigest();history=[];seconds=0.;best=float('inf');best_step=0;stop='budget_completed';clipped=0
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,arm=arm,config=actual,study_config_sha256=file_sha(p/'config.json'),initial_model_digest=initial,training_order_sha256=order,source_archive_sha256=cfg['source_archive_sha256'],rng_cpu=torch.get_rng_state(),rng_gpu=torch.cuda.get_rng_state_all(),backend=cfg['backend']),folder/name)
        def assess(step):
            with monitor.in_phase('eval'):row=evaluate(adapter,codec,reader,stats,diff.alpha_hat,actual,data,folder,step,dict(train=cfg['eval_train_ids'],dev_unseen_text=cfg['eval_dev_ids']),controls=True)
            with monitor.in_phase('stroke_audit'):stroke_audit(root,p,actual,data,arm,step)
            with monitor.in_phase('target_informed_diagnostic'):denoising(adapter,items,stats,actual,data,diff.alpha_hat,folder,step)
            history.append(dict(step=step,aggregate=row['aggregate']));return row['aggregate']['dev_unseen_text']['1.0']['cer']
        (folder/'initialization.json').write_text(json.dumps(identity,indent=2)+'\n');save('checkpoint-initial.pt',0);best=assess(0);save('checkpoint-best.pt',0);torch.manual_seed(cfg['model_rng_seed']);model.train()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(batches,1):
                monitor.set_phase('train');torch.cuda.synchronize();started=time.monotonic();clean,text,mask,prefix,t,eps,drop=matched_batch(pool,ids,step,cfg['training_draw_seed'],1000);opt.zero_grad(set_to_none=True);terms=loss(model,clean,text,mask,prefix,t,diff.alpha_hat,1000.,False,drop,40,'x0',eps384=eps);terms['loss'].backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));lr=learning_rate(step,cfg)
                for group in opt.param_groups:group['lr']=lr
                opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-started;seconds+=elapsed;clipped+=norm>1;monitor.step(elapsed,len(ids));row=dict(step=step,sample_ids=ids,objective=float(terms['loss'].detach()),xy16_x0_mse=float(terms['xy16_x0_mse'].detach()),pen24_x0_mse=float(terms['pen24_x0_mse'].detach()),raw_grad_norm=norm,clipped=norm>1,lr=lr,train_seconds=seconds,text_dropped=drop,prefix_retained=False,active_tokens=int(mask.sum()),timesteps=t.cpu().tolist(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if step<=10 or step%250==0:row['data_draw_digest']=tensor_digest(dict(clean=clean,eps=eps,t=t))
                if not np.isfinite([row[k] for k in ['objective','xy16_x0_mse','pen24_x0_mse','raw_grad_norm']]).all():raise ValueError('nonfinite native update')
                log.write(json.dumps(row)+'\n')
                if step%250==0:log.flush();print(dict(arm=arm,**row),flush=True)
                limit=seconds>=1800
                if step in cfg['eval_steps'] or limit:
                    score=assess(step);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen codec/reader changed')
    result=dict(arm=arm,last_step=step,best_step=best_step,best_dev_cer=best,history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_model_digest=initial,training_order_sha256=order,backend=cfg['backend'],no_confirmations_opened=True,not_promoted=True,codec_reader_unchanged=True,quality_status='research only; readability, content sensitivity and stroke-density gates required before promotion',selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'));(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--study');p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if sum([a.prepare,a.preflight,a.train])!=1:p.error('explicit one of --prepare / --preflight / --train; no default GPU allocation')
    if a.prepare:print(prepare())
    elif a.preflight:
        result=preflight();Path('logs/wsl-conditioning-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    elif not a.study or not a.arm:p.error('explicit existing study/arm required; never create a study on re-entry')
    else:print(json.dumps(run(a.study,a.arm),indent=2))
if __name__=='__main__':main()
