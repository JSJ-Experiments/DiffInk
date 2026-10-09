"""Bounded corpus-scale actual DiT + frozen transport baseline, v2 outputs.

Novel-text inference has NO target prefix/point count/writer ID. Not semantic VAE
or paper reproduction. The old eight-line autoregressive mapper is not involved.
"""
import hashlib,json,math,time,tarfile
from pathlib import Path
import h5py,numpy as np,torch
from .corpus_dit_contract import CODEC,CODEC_SHA,POOL_SHA,READER_REL,READER_SHA,validate_scope
from .corpus_dit import training_loss,sample,requested_inputs,fit_posterior_whitening,learning_rate
from .ocr_pool_study import load_pool,single_batch
from .latent_integration import encoded
from .pen_ab import file_sha,boundary_metrics
from .generation_study import read_sequence
from .inkvae import edit_distance
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .resource_monitor import ResourceMonitor
from .launch_ledger import study_path


def validate(cfg,data):
    if cfg['codec_sha256']!=CODEC_SHA or cfg['codec_relative']!=CODEC or cfg['pool_manifest_sha256']!=POOL_SHA or cfg['reader_sha256']!=READER_SHA:raise ValueError('immutable actual-DiT corpus/codec/reader required')
    if cfg['max_updates']!=12000 or cfg['max_train_wall_seconds']!=1800 or cfg['batch']!=32 or cfg['lr']!=5e-5 or cfg['clip']!=1 or cfg['diffusion_steps']!=1000 or cfg['timestep_input_divisor']!=1000. or cfg['prediction']!='x0':raise ValueError('bounded declared corpus diffusion protocol required')
    if not all(cfg[k] for k in ['source_volume_read_only','not_promoted','no_codec_training','no_kl','no_ctc_training','no_style_training']) or cfg['output_volume']!='diffink-experiments-v2':raise ValueError('frozen engineering baseline/isolation contract required')
    train=data['scope']['splits']['train'];counts=data['training_writer_counts']
    if len(train)<7000 or len(counts)<150 or data['duration']['train_ids']!=train or data['duration']['writers']!=['corpus']:raise ValueError('broad repeated-writer TRAIN and target-free pooled duration required')
    if cfg['eval_steps']!=[0,1000,3000,6000,12000] or cfg['eval_noise_seeds']!=[73142,73143] or cfg['eval_guidance']!=[1.,2.] or cfg['sampling_steps']!=50:raise ValueError('declared repeated-noise novel-text evaluation required')


def verify_sources(root,folder,cfg,data):
    validate(cfg,data);root=Path(root)
    for path,sha in [(folder/'dataset.json',cfg['dataset_sha256']),(folder/'as-run-source.tar.gz',cfg['source_archive_sha256']),(root/cfg['codec_relative'],cfg['codec_sha256']),(root/cfg['reader_relative'],cfg['reader_sha256']),(root/cfg['history_relative'],cfg['history_sha256'])]:
        if file_sha(path)!=sha:raise ValueError('pinned source drift '+str(path))
    pool,manifest,vocab=load_pool(root,POOL_SHA)
    if str(pool.relative_to(root))!=cfg['pool_relative'] or vocab!=cfg['vocab']:raise ValueError('immutable corpus path/alphabet required')
    if any(manifest['records'].get(i)!=r for i,r in data['records'].items()):raise ValueError('prepared metadata changed from immutable corpus')
    history=json.loads((root/cfg['history_relative']).read_text())['records'];validate_scope(data['records'],data['scope']['splits'],history,manifest['test_writers'],manifest['dev_writers'])
    with tarfile.open(folder/'as-run-source.tar.gz') as a:
        for path in ['iam_tools/corpus_dit.py','iam_tools/corpus_dit_contract.py','iam_tools/corpus_dit_study.py','model/dit.py','model/modules.py','model/diffusion.py','model/vae.py','model/blocks.py']:
            f=Path(__file__).parent/path.split('/')[-1] if path.startswith('iam_tools/') else Path(__file__).parent.parent/path
            if not f.exists():f=Path('third_party/DiffInk')/path
            if a.extractfile(path).read()!=f.read_bytes():raise ValueError('as-run diffusion implementation drift '+path)
    return pool,manifest


@torch.no_grad()
def cache(codec,reader,root,pool,folder,cfg,data,monitor):
    from model.losses import mixture_expectation
    # The fresh and exposed held confirmations are not even encoded here.
    ids=data['scope']['splits']['train']+data['scope']['splits']['dev'];train=set(data['scope']['splits']['train']);probe=set(cfg['eval_train_ids']+cfg['eval_dev_ids']);rows=[];transport=[];latents={};started=time.monotonic()
    with h5py.File(pool/'lines.h5') as source,h5py.File(folder/'posterior-cache.h5','w') as out:
        for j,sid in enumerate(ids):
            points=source[sid]['point_seq'][:];r=data['records'][sid]
            if hashlib.sha256(points.tobytes()).hexdigest()!=r['points_sha256']:raise ValueError('raw source drift '+sid)
            raw,pm,_=single_batch(points,r['text'],cfg['vocab'],'cuda');truth,mu,lv,lm=encoded(codec,raw,pm);n=len(points);packed=mu[0,:40].T.reshape(-1,5)[:n]
            diff=float((packed[:,:2]-truth[0,:n]).abs().max());pens=int((packed[:,2:].argmax(1)!=raw[0,2:,:n].argmax(0)).sum())
            if diff>.001 or pens or not torch.isfinite(lv).all():raise ValueError('frozen transport preflight failure '+sid)
            # Same stroke-only prefix mask policy as released English fallback:
            # nearest exclusive true stroke end, coarse prefix blocks rounded UP.
            states=points[:,2:].argmax(1);ends=np.flatnonzero(states!=0)+1;candidates=ends[ends<n];cut=int(candidates[np.argmin(np.abs(candidates-n*cfg['prefix_target_ratio']))]) if len(candidates) else 0
            item=dict(mu=mu[0].T.cpu(),lv=lv[0].T.cpu(),prefix_blocks=(cut+7)//8);latents[sid]=item
            g=out.create_group(sid);g.create_dataset('mu',data=item['mu'].numpy(),compression='gzip');g.create_dataset('logvar',data=item['lv'].numpy(),compression='gzip');g.attrs.update(points=n,prefix_blocks=item['prefix_blocks'],split='train' if sid in train else 'dev_source_audit_only')
            transport.append(dict(sample_id=sid,max_packed_xy_difference=diff,pen_mismatches=pens))
            if sid in probe:
                output=codec.decode(mu,padding_mask=~pm);xy=mixture_expectation(output)[0,:n];hard=output[0,:3,:n].argmax(0);sequence=torch.cat((xy,torch.nn.functional.one_hot(hard,3).float()),-1);decoded=read_sequence(reader,sequence,cfg['vocab']);error=(xy-truth[0,:n]).square().mean(0).sqrt().cpu().tolist();pen=boundary_metrics(hard.cpu().numpy(),states)
                if max(error)>.0025 or pen['pen_up_f1']!=1 or not pen['final_eoc_correct'] or pen['non_final_false_eoc_count']:raise ValueError('frozen decoder source audit failure '+sid)
                rows.append(dict(sample_id=sid,x_rmse=error[0],y_rmse=error[1],pen=pen,decoded=decoded,errors=edit_distance(r['text'],decoded),characters=len(r['text'])))
                g.create_dataset('source_points_model_space',data=torch.cat((truth[0,:n],raw[0,2:,:n].T),-1).cpu().numpy(),compression='gzip')
            if (j+1)%500==0:print(dict(cache_lines=j+1,total=len(ids),seconds=time.monotonic()-started),flush=True)
            if time.monotonic()-started>1200:raise RuntimeError('bounded corpus cache wall cap exceeded; preserve partial cache, no restart')
    stats=fit_posterior_whitening((latents[i]['mu'],latents[i]['lv']) for i in data['scope']['splits']['train']);stats['train_ids']=data['scope']['splits']['train'];torch.save(stats,folder/'whitening.pt')
    audit=dict(lines=len(ids),training_lines=len(train),confirmation_not_encoded=True,cached_ids=ids,transport=transport,source_codec_and_reader_probe=rows,probe_source_cer=sum(r['errors'] for r in rows)/sum(r['characters'] for r in rows),cache_seconds=time.monotonic()-started,cache_sha256=file_sha(folder/'posterior-cache.h5'),whitening_sha256=file_sha(folder/'whitening.pt'),passed=True)
    (folder/'codec-cache-audit.json').write_text(json.dumps(audit,indent=2)+'\n');return latents,stats,audit


class PosteriorPool:
    def __init__(self,items,records,vocab,train,stats,device):
        self.ids=list(train);self.index={i:j for j,i in enumerate(train)};self.lengths={i:len(items[i]['mu']) for i in train};self.chars={i:len(records[i]['text']) for i in train};length=max(self.lengths.values());b=len(train);mean=stats['mean'].to(device);std=stats['std'].to(device)
        self.mu=torch.zeros(b,length,384,device=device);self.std=torch.zeros_like(self.mu);self.mask=torch.zeros(b,length,dtype=torch.bool,device=device);self.text=torch.full((b,max(self.chars.values())),-1,dtype=torch.long,device=device);self.prefix=torch.zeros(b,dtype=torch.long,device=device)
        for j,i in enumerate(train):
            n=self.lengths[i];mu=items[i]['mu'].to(device);lv=items[i]['lv'].to(device);self.mu[j,:n]=(mu-mean)/std;self.std[j,:n]=(.5*lv).exp()/std;self.mask[j,:n]=True;self.prefix[j]=items[i]['prefix_blocks'];self.text[j,:self.chars[i]]=torch.tensor([vocab.index(c) for c in records[i]['text']],device=device)
    def select(self,ids):
        idx=torch.tensor([self.index[i] for i in ids],device=self.mu.device);length=max(self.lengths[i] for i in ids);chars=max(self.chars[i] for i in ids)
        mu=self.mu.index_select(0,idx)[:,:length];std=self.std.index_select(0,idx)[:,:length];clean=mu+std*torch.randn_like(mu)
        return clean,self.text.index_select(0,idx)[:,:chars],self.mask.index_select(0,idx)[:,:length],self.prefix.index_select(0,idx)


def schedule(ids,lengths,steps,batch,seed):
    rng=np.random.default_rng(seed);ordered=sorted(ids,key=lambda i:(lengths[i],i));done=0
    while done<steps:
        groups=[]
        for j in range(0,len(ordered),128):
            window=list(rng.permutation(ordered[j:j+128]));groups.extend([window[k:k+batch] for k in range(0,len(window),batch)])
        for k in rng.permutation(len(groups)):
            yield [str(i) for i in groups[int(k)]];done+=1
            if done==steps:return


@torch.no_grad()
def evaluate(model,codec,reader,stats,alpha,cfg,data,folder,step,splits,controls=False):
    from model.losses import mixture_expectation
    was=model.training;model.eval();device=next(model.parameters()).device;rows=[];file=folder/f'evaluation-{step}.h5';mean=stats['mean'].to(device);scale=stats['std'].to(device)
    with h5py.File(file,'w') as hf:
        for split,ids in splits.items():
            for seed in cfg['eval_noise_seeds']:
                for guidance in cfg['eval_guidance']:
                    policies=['correct','swapped','null'] if controls and guidance==1. and seed==cfg['eval_noise_seeds'][0] else ['correct']
                    for policy in policies:
                        for start in range(0,len(ids),16):
                            batch=ids[start:start+16];requested=[data['records'][i]['text'] for i in batch];noise,text,mask,lengths=requested_inputs(requested,cfg['vocab'],data['duration'],seed,batch,device)
                            if policy=='swapped':
                                # Identical random noise and estimated lengths to correct policy.
                                other=[data['records'][ids[(ids.index(i)+1)%len(ids)]]['text'] for i in batch];text=torch.full((len(batch),max(map(len,other))),-1,dtype=torch.long,device=device)
                                for j,t in enumerate(other):text[j,:len(t)]=torch.tensor([cfg['vocab'].index(c) for c in t],device=device)
                            elif policy=='null':other=['']*len(batch)
                            else:other=requested
                            z=sample(model,noise,text,mask,alpha,cfg['sampling_steps'],guidance,cfg['timestep_input_divisor'],null=policy=='null');z=z*scale+mean
                            for j,sid in enumerate(batch):
                                n=lengths[j]*8;pm=torch.ones(1,n,dtype=torch.bool,device=device);out=codec.decode(z[j,:lengths[j]].T[None],padding_mask=~pm);xy=mixture_expectation(out)[0];pens=out[0,:3].argmax(0);points=torch.cat((xy,torch.nn.functional.one_hot(pens,3).float()),-1);hits=torch.nonzero(pens==2);stop=int(hits[0,0])+1 if len(hits) else n;decoded=read_sequence(reader,points[:stop],cfg['vocab']);r=data['records'][sid]
                                row=dict(sample_id=sid,split=split,policy=policy,seed=seed,guidance=guidance,text=r['text'],conditioning_text=other[j],decoded=decoded,errors=edit_distance(r['text'],decoded),characters=len(r['text']),errors_against_supplied_text=edit_distance(other[j],decoded) if policy!='null' else None,estimated_blocks=lengths[j],generated_points=stop,found_eoc=bool(len(hits)),actual_source_points_for_diagnostic_only=r['points'],no_source_trajectory=True,no_source_length=True,no_writer_id=True,conditioning_characters_truncated=max(0,len(other[j])-max(lengths)))
                                rows.append(row);g=hf.create_group(f'{split}/{seed}/{guidance}/{policy}/{sid}');g.create_dataset('points',data=points[:stop].cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    aggregate={}
    for split in splits:
        aggregate[split]={}
        for guidance in cfg['eval_guidance']:
            subset=[r for r in rows if r['split']==split and r['guidance']==guidance and r['policy']=='correct'];aggregate[split][str(guidance)]=dict(evaluations=len(subset),cer=sum(r['errors'] for r in subset)/sum(r['characters'] for r in subset),exact=sum(r['errors']==0 for r in subset),missing_eoc=sum(not r['found_eoc'] for r in subset),median_points=float(np.median([r['generated_points'] for r in subset])))
    result=dict(step=step,aggregate=aggregate,rows=rows,packed_h5_sha256=file_sha(file),scope='genuine diffusion from random noise; TRAIN-only requested-text duration; no reference/target/writerID; frozen corpus-familiar reader',no_paper_reproduction=True)
    (folder/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(step=step,novel_text_free=aggregate),flush=True);model.train(was);return result


def run(relative,root='/work',on_checkpoint=None):
    if not torch.cuda.is_available():raise RuntimeError('explicit bounded T4 required')
    from model.vae import VAE
    from model.dit import DiT
    from model.diffusion import Diffusion
    from utils.utils import ModelConfig
    root=Path(root);p=study_path(root,relative,'checkpoints/iam_corpus_dit/');cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());pool,manifest=verify_sources(root,p,cfg,data);folder=p/'baseline';folder.mkdir(exist_ok=False);torch.set_num_threads(2)
    saved=torch.load(root/CODEC,map_location='cpu',weights_only=True);codec=VAE(ModelConfig(saved['config']));codec.load_state_dict(saved['model_state_dict']);codec.apply_checkpoint_contract(saved);codec=codec.cuda().eval().requires_grad_(False);cd=tensor_digest(codec.state_dict());sr=torch.load(root/READER_REL,map_location='cpu',weights_only=True);reader,_=load_reader(root,sr['config']['cfg'],'cuda');reader.requires_grad_(False);rd=tensor_digest(reader.state_dict())
    with ResourceMonitor(folder,cpu_request=2,memory_request_mib=8192,gpu=True,interval=2,sustained_seconds=45) as monitor:
        with monitor.in_phase('cache'):items,stats,audit=cache(codec,reader,root,pool,folder,cfg,data,monitor)
        if on_checkpoint:
            with monitor.in_phase('persist'):on_checkpoint()
        train=data['scope']['splits']['train'];pool=PosteriorPool(items,data['records'],cfg['vocab'],train,stats,'cuda');del items
        torch.manual_seed(cfg['seed'])
        if cfg['model'].get('use_cross_attention',False):
            from model.dit import CrossAttentionDiT
            model=CrossAttentionDiT(ModelConfig(cfg['model'])).cuda()
        else:
            model=DiT(ModelConfig(cfg['model'])).cuda()
        initial=tensor_digest(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay']);diffusion=Diffusion(noise_steps=cfg['diffusion_steps'],schedule_type='cosine',device='cuda');batches=list(schedule(train,pool.lengths,cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));orderhash=hashlib.sha256(json.dumps(batches).encode()).hexdigest();history=[];best=float('inf');best_step=0;seconds=0.;clipped=0;stop='budget_completed'
        def save(name,step):torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),step=step,config=cfg,initial_model_digest=initial,training_order_sha256=orderhash,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),source_archive_sha256=cfg['source_archive_sha256']),folder/name)
        def assess(step):
            with monitor.in_phase('eval'):row=evaluate(model,codec,reader,stats,diffusion.alpha_hat,cfg,data,folder,step,dict(train=cfg['eval_train_ids'],dev_unseen_text=cfg['eval_dev_ids']),controls=step in (0,cfg['max_updates']))
            history.append(dict(step=step,aggregate=row['aggregate']));return row['aggregate']['dev_unseen_text']['1.0']['cer']
        best=assess(0);save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);model.train();torch.manual_seed(cfg['seed']+1)
        if on_checkpoint:
            with monitor.in_phase('persist'):on_checkpoint()
        with (folder/'metrics.jsonl').open('w') as log:
            for step,batch in enumerate(batches,1):
                monitor.set_phase('train');start=time.monotonic();clean,text,mask,prefix=pool.select(batch);t=torch.randint(cfg['diffusion_steps'],(len(batch),),device='cuda');draw=torch.rand(2).tolist();drop=draw[0]<cfg['text_drop_probability'];keep=draw[1]<cfg['prefix_keep_probability'] and not drop;opt.zero_grad(set_to_none=True);terms=training_loss(model,clean,text,mask,prefix,t,diffusion,cfg['timestep_input_divisor'],keep,drop);loss=terms['loss'];loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'],error_if_nonfinite=True));lr=learning_rate(step,cfg)
                for group in opt.param_groups:group['lr']=lr
                opt.step();elapsed=time.monotonic()-start;seconds+=elapsed;clipped+=norm>cfg['clip'];monitor.step(elapsed,len(batch));row=dict(step=step,sample_ids=batch,x0_loss=float(loss.detach()),active40_loss=float(terms['active40'].detach()),unused344_loss=float(terms['unused344'].detach()),raw_grad_norm=norm,clipped=norm>cfg['clip'],lr=lr,prefix_retained=keep,text_dropped=drop,active_tokens=int(terms['active_mask'].sum()),train_seconds=seconds,timesteps=t.cpu().tolist());log.write(json.dumps(row)+'\n')
                if not math.isfinite(row['x0_loss']):raise FloatingPointError('nonfinite corpus denoising objective')
                if step%250==0:log.flush();print(row,flush=True)
                limit=seconds>=cfg['max_train_wall_seconds']
                if step in cfg['eval_steps'] or limit:
                    score=assess(step);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                    if on_checkpoint:
                        log.flush()
                        with monitor.in_phase('persist'):on_checkpoint()
                if limit:stop='wall_limit';break
        save('checkpoint-last.pt',step)
        # Only AFTER completed budget and DEV-only selection: fresh/held generation.
        confirmations=dict(exposed_held=data['scope']['splits']['exposed_held'],fresh_confirmation=data['scope']['splits']['fresh_confirmation'])
        with monitor.in_phase('confirmation_final'):evaluate(model,codec,reader,stats,diffusion.alpha_hat,cfg,data,folder,'final-confirmation',confirmations,controls=True)
        selected=torch.load(folder/'checkpoint-best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['model_state_dict'])
        with monitor.in_phase('confirmation_selected'):evaluate(model,codec,reader,stats,diffusion.alpha_hat,cfg,data,folder,'selected-confirmation',confirmations,controls=True)
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd or any(p.grad is not None for p in codec.parameters()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('frozen codec/reader drift')
    result=dict(last_step=step,best_step=best_step,best_dev_cer=best,history=history,stop=stop,train_seconds=seconds,clip_fraction=clipped/step,initial_model_digest=initial,training_order_sha256=orderhash,train_lines=len(train),train_writers=len(data['training_writer_counts']),codec_reader_unchanged=True,confirmation_not_encoded_during_training=True,not_promoted=True,selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
