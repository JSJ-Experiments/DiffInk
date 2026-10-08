"""Corpus factorial: meaningful40/full384 x direct-x0/diffusion-v.

Reuse verified frozen posterior cache, corpus splits, read-only original sources.
No new confirmations opened: prior fresh/held are now exposed, not used here.
DEV-only selection; generic handwriting, NOT learned semantic InkVAE/reproduction.
"""
import hashlib,json,math,time,tarfile
from pathlib import Path
import torch,h5py,numpy as np
from .compact_dit import ARMS,loss,X0Adapter,matched_initial_state
from .corpus_dit import learning_rate
from .corpus_dit_study import verify_sources,PosteriorPool,schedule,evaluate
from .pen_ab import file_sha
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .resource_monitor import ResourceMonitor
from .launch_ledger import study_path


def validate(cfg,parent_cfg):
    if cfg['arms']!=list(ARMS) or cfg['max_updates']!=12000 or cfg['max_train_wall_seconds']!=1800 or cfg['eval_steps']!=[0,1000,3000,6000,12000]:
        raise ValueError('bounded predeclared four-arm corpus factorial required')
    if cfg['matched_training_rng_width']!=384 or cfg['selection_split']!='dev_unseen_text' or cfg['open_confirmations']:
        raise ValueError('shared RNG and no reopened confirmation selection required')
    if cfg['parent_relative']!='checkpoints/iam_corpus_dit/20261008-223316' or cfg['cache_definition']!='reuse frozen TRAIN+DEV cache; no new encoding; no held statistics':
        raise ValueError('completed immutable baseline and cache required')
    for key in ['seed','schedule_seed','batch','lr','warmup_updates','prefix_keep_probability','text_drop_probability','eval_train_ids','eval_dev_ids','eval_noise_seeds','eval_guidance','sampling_steps','model']:
        if cfg[key]!=parent_cfg[key]:raise ValueError('factorial shared protocol drift '+key)


def verify(root,folder,cfg):
    root=Path(root);parent=study_path(root,cfg['parent_relative'],'checkpoints/iam_corpus_dit/')
    for path,sha in [(parent/'config.json',cfg['parent_config_sha256']),(parent/'dataset.json',cfg['dataset_sha256']),
                     (parent/'baseline/posterior-cache.h5',cfg['cache_sha256']),(parent/'baseline/whitening.pt',cfg['whitening_sha256']),
                     (folder/'as-run-source.tar.gz',cfg['source_archive_sha256'])]:
        if file_sha(path)!=sha:raise ValueError('pinned factorial source drift '+str(path))
    pc=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());validate(cfg,pc)
    verify_sources(root,parent,pc,data)
    with tarfile.open(folder/'as-run-source.tar.gz') as a:
        for path in cfg['verified_source_paths']:
            current=Path(__file__).parent/path.split('/')[-1] if path.startswith('iam_tools/') else Path(__file__).parent.parent/path
            if not current.exists():current=Path('third_party/DiffInk')/path
            if a.extractfile(path).read()!=current.read_bytes():raise ValueError('as-run factorial source drift '+path)
    return parent,pc,data


def load_cache(parent,data):
    items={}
    with h5py.File(parent/'baseline/posterior-cache.h5') as hf:
        expected=data['scope']['splits']['train']+data['scope']['splits']['dev']
        if set(hf)!=set(expected):raise ValueError('cache includes missing/forbidden records')
        for sid in expected:
            g=hf[sid];items[sid]=dict(mu=torch.from_numpy(g['mu'][:]),lv=torch.from_numpy(g['logvar'][:]),prefix_blocks=int(g.attrs['prefix_blocks']))
    stats=torch.load(parent/'baseline/whitening.pt',map_location='cpu',weights_only=True)
    if stats['train_ids']!=data['scope']['splits']['train']:raise ValueError('TRAIN-only whitening identity drift')
    return items,stats


@torch.no_grad()
def unused_gate(codec,items,stats,ids):
    """Gate near-unused decoder coupling, not mathematically exact independence.

    CPU audit found TRAIN-mean substitution <=1e-8; random nuisance perturbs
    XY by ~1e-4 (0.01px at report scale), with no pen changes. Gate mean at1e-6
    and random at2.5e-4 (0.025px); mean pen logits allow1e-5 for float32
    rounding (measured3.8e-6), with zero hard-pen changes; retain actual errors.
    """
    from model.losses import mixture_expectation
    rows=[];device=next(codec.parameters()).device
    for sid in ids:
        mu=items[sid]['mu'].to(device);g=torch.Generator(device=device).manual_seed(87621)
        reference=codec.decode(mu.T[None]);xy=mixture_expectation(reference);pens=reference[:,:3]
        for policy in ['TRAIN_mean','independent_gaussian']:
            changed=mu.clone();changed[:,40:]=stats['mean'][40:].to(device)
            if policy=='independent_gaussian':changed[:,40:]+=torch.randn(changed[:,40:].shape,device=device,generator=g)*stats['std'][40:].to(device)
            out=codec.decode(changed.T[None]);delta=float((mixture_expectation(out)-xy).abs().max());pd=float((out[:,:3]-pens).abs().max())
            mismatches=int((out[:,:3].argmax(1)!=pens.argmax(1)).sum())
            tolerance=1e-6 if policy=='TRAIN_mean' else 2.5e-4
            pen_tolerance=1e-5 if policy=='TRAIN_mean' else tolerance
            if not math.isfinite(delta+pd) or delta>tolerance or pd>pen_tolerance or mismatches:raise ValueError('near-unused ablation materially changes decoder '+sid)
            rows.append(dict(sample_id=sid,policy=policy,max_xy_change=delta,max_pen_logit_change=pd,pen_mismatches=mismatches,xy_tolerance=tolerance,pen_logit_tolerance=pen_tolerance))
    return dict(passed=True,rows=rows,scope='all48 source probes; near-unused coupling tolerance .025px at100px/unit for random nuisance, TRAIN-mean1e-6; NOT exact independence or proof nuisance harms diffusion')


@torch.no_grad()
def denoising(adapter,items,stats,cfg,data,alpha,folder,step):
    """Target-informed fixed48, repeated posterior/noise/text controls; not free."""
    from .corpus_dit import call
    device=next(adapter.parameters()).device;was=adapter.training;adapter.eval();rows=[]
    columns={'xy16':[j for j in range(40) if j%5<2],'pen24':[j for j in range(40) if j%5>=2],'active40':list(range(40))}
    for split,key in [('train','eval_train_ids'),('dev_unseen_text','eval_dev_ids')]:
        for sid in cfg[key]:
            mu=items[sid]['mu'].to(device)[None];lv=items[sid]['lv'].to(device)[None];g=torch.Generator(device=device).manual_seed(int(hashlib.sha256(('denoise74342:'+sid).encode()).hexdigest()[:15],16))
            clean=(mu+torch.randn(mu.shape,device=device,generator=g)*(.5*lv).exp()-stats['mean'].to(device))/stats['std'].to(device);eps=torch.randn(clean.shape,device=device,generator=g);mask=torch.ones(clean.shape[:2],device=device,dtype=torch.bool);text=torch.tensor([[cfg['vocab'].index(c) for c in data['records'][sid]['text']]],device=device)
            for t in [0,10,100,500,900,999]:
                a=alpha[t];xt=a.sqrt()*clean+(1-a).sqrt()*eps
                for policy,label in [('correct',text),('rotated_character_order',text.roll(1,1)),('null',text)]:
                    pred=call(adapter,xt,label,torch.tensor([t],device=device),mask,cfg['timestep_input_divisor'],policy=='null');err=(pred-clean).square()[mask]
                    rows.append(dict(sample_id=sid,split=split,timestep=t,policy=policy,**{k+'_mse':float(err[:,v].mean()) for k,v in columns.items()}))
    summary={}
    for split in ['train','dev_unseen_text']:
        summary[split]={str(t):{policy:{k+'_mse':float(np.mean([r[k+'_mse'] for r in rows if r['split']==split and r['timestep']==t and r['policy']==policy])) for k in columns} for policy in ['correct','rotated_character_order','null']} for t in [0,10,100,500,900,999]}
    result=dict(step=step,rows=rows,summary=summary,scope='target-informed source posterior and length, mean per-line latent MSE; NOT free generation',shared_device_noise_across_arms=True,CPU_baseline_noise_not_identical=True)
    (folder/f'denoising-{step}.json').write_text(json.dumps(result,indent=2)+'\n');adapter.train(was);return result


def run(relative,arm,root='/work',on_checkpoint=None):
    if not torch.cuda.is_available() or arm not in ARMS:raise RuntimeError('explicit predeclared bounded T4 arm required')
    from model.vae import VAE
    from model.dit import DiT
    from model.diffusion import Diffusion
    from utils.utils import ModelConfig
    root=Path(root);p=study_path(root,relative,'checkpoints/iam_compact_dit/');cfg=json.loads((p/'config.json').read_text());parent,pc,data=verify(root,p,cfg);folder=p/arm;folder.mkdir(exist_ok=False);torch.set_num_threads(2)
    saved=torch.load(root/pc['codec_relative'],map_location='cpu',weights_only=True);codec=VAE(ModelConfig(saved['config']));codec.load_state_dict(saved['model_state_dict']);codec.apply_checkpoint_contract(saved);codec=codec.cuda().eval().requires_grad_(False);cd=tensor_digest(codec.state_dict());sr=torch.load(root/pc['reader_relative'],map_location='cpu',weights_only=True);reader,_=load_reader(root,sr['config']['cfg'],'cuda');reader.requires_grad_(False);rd=tensor_digest(reader.state_dict());channels,prediction=ARMS[arm]
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        with monitor.in_phase('load_cache'):items,stats=load_cache(parent,data)
        with monitor.in_phase('codec_unused_gate'):gate=unused_gate(codec,items,stats,cfg['eval_train_ids']+cfg['eval_dev_ids']);(folder/'unused-decoder-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        train=data['scope']['splits']['train'];pool=PosteriorPool(items,data['records'],cfg['vocab'],train,stats,'cuda')
        torch.manual_seed(cfg['seed']);full=DiT(ModelConfig(cfg['model']));base=tensor_digest(full.state_dict());mc=dict(cfg['model'],latent_dim=channels)
        if channels==384:model=full
        else:model=DiT(ModelConfig(mc));model.load_state_dict(matched_initial_state(full.state_dict(),model.state_dict()))
        del full;model=model.cuda();initial=tensor_digest(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);diffusion=Diffusion(noise_steps=cfg['diffusion_steps'],schedule_type='cosine',device='cuda');adapter=X0Adapter(model,diffusion.alpha_hat,channels,prediction,cfg['timestep_input_divisor']).cuda();batches=list(schedule(train,pool.lengths,cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));orderhash=hashlib.sha256(json.dumps(batches).encode()).hexdigest();history=[];seconds=0.;clipped=0;best_step=0;stop='budget_completed'
        def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),arm=arm,channels=channels,prediction=prediction,step=step,config=cfg,initial_model_digest=initial,shared_full_initial_digest=base,training_order_sha256=orderhash,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),source_archive_sha256=cfg['source_archive_sha256']),folder/name)
        def assess(step):
            with monitor.in_phase('eval'):row=evaluate(adapter,codec,reader,stats,diffusion.alpha_hat,cfg,data,folder,step,dict(train=cfg['eval_train_ids'],dev_unseen_text=cfg['eval_dev_ids']),controls=step in [0,cfg['max_updates']])
            history.append(dict(step=step,aggregate=row['aggregate']))
            with monitor.in_phase('denoising_diagnostic'):denoising(adapter,items,stats,cfg,data,diffusion.alpha_hat,folder,step)
            return row['aggregate']['dev_unseen_text']['1.0']['cer']
        best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);model.train();torch.manual_seed(cfg['seed']+1)
        if on_checkpoint:
            with monitor.in_phase('persist'):on_checkpoint()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(batches,1):
                monitor.set_phase('train');start=time.monotonic();clean,text,mask,prefix=pool.select(batch);t=torch.randint(cfg['diffusion_steps'],(len(batch),),device='cuda');draw=torch.rand(2).tolist();drop=draw[0]<cfg['text_drop_probability'];keep=draw[1]<cfg['prefix_keep_probability'] and not drop;opt.zero_grad(set_to_none=True);terms=loss(model,clean,text,mask,prefix,t,diffusion.alpha_hat,cfg['timestep_input_divisor'],keep,drop,channels,prediction);terms['loss'].backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));lr=learning_rate(step,cfg)
                for group in opt.param_groups:group['lr']=lr
                opt.step();elapsed=time.monotonic()-start;seconds+=elapsed;clipped+=norm>cfg['clip'];monitor.step(elapsed,len(batch));row=dict(step=step,sample_ids=batch,objective=float(terms['loss'].detach()),**{k:float(terms[k].detach()) if terms[k] is not None else None for k in ['active40_objective','unused344_objective','xy16_x0_mse','pen24_x0_mse','active40_x0_mse']},raw_grad_norm=norm,clipped=norm>cfg['clip'],lr=lr,prefix_retained=keep,text_dropped=drop,active_tokens=int(terms['active_mask'].sum()),train_seconds=seconds,timesteps=t.cpu().tolist());log.write(json.dumps(row)+'\n')
                if not all(math.isfinite(row[k]) for k in ['objective','xy16_x0_mse','pen24_x0_mse']):raise FloatingPointError('nonfinite corpus factorial objective')
                if step%250==0:log.flush();print(dict(arm=arm,**row),flush=True)
                limit=seconds>=cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:
                        log.flush()
                        with monitor.in_phase('persist'):on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(v.grad is not None for v in codec.parameters()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('frozen codec/reader drift')
    result=dict(arm=arm,channels=channels,prediction=prediction,last_step=step,best_step=best_step,best_dev_cer=best,history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_model_digest=initial,shared_full_initial_digest=base,training_order_sha256=orderhash,train_lines=len(train),train_writers=len(data['training_writer_counts']),codec_reader_unchanged=True,no_confirmations_opened=True,not_promoted=True,selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),v_weighting_caveat='v-MSE changes SNR weighting as well as low-noise identity; not a skip-only intervention')
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
