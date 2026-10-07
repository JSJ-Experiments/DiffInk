"""Bounded conditional generation gate with immutable faithful codec and packed HDF5.

Research prototype, not paper InkDiT. TRAIN32/same-writer unseen-form8; reader is
familiar with these corpus images, generator is not. Oracle duration is explicit.
"""
import copy,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .latent_diffusion import TextLatentDenoiser,fit_whitening,transform,masked_mse,cosine_schedule,forward_noise,ddim_sample
from .kl_tradeoff import SOURCE,SHA,load_batches
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .ocr_pool_study import load_pool
from .ocr_convergence import POOL_SHA
from .writer_expansion import load,training_schedule
from .latent_integration import encoded
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha,boundary_metrics
from .inkvae import greedy_ctc,edit_distance
from .ocr_joint_study import local_geometry

ARMS=('text','no_text');WRITER='10160';HELD_FORM='g09-301'


def validate_budget(steps,pool_sha):
    if type(steps)!=int or not 1000<=steps<=12000 or pool_sha!=POOL_SHA:raise ValueError('1000–12000 bounded updates, exact immutable pool SHA required')


def select_scope(manifest):
    candidates=sorted(i for i in manifest['splits']['large_train'] if manifest['records'][i]['writer_id']==WRITER)
    test=[i for i in candidates if manifest['records'][i]['prompt_family']==HELD_FORM]
    train=sorted((i for i in candidates if i not in test),key=lambda i:hashlib.sha256(('7314:'+i).encode()).hexdigest())[:32]
    if len(train)!=32 or len(test)!=8 or set(train)&set(test):raise ValueError('exact32/8 predeclared generator scope required')
    forms={manifest['records'][i]['prompt_family'] for i in train}
    texts={manifest['records'][i]['text'].strip() for i in train}
    if HELD_FORM in forms or any(manifest['records'][i]['text'].strip() in texts for i in test):raise ValueError('held-out form/text leakage')
    if WRITER in manifest['test_writers'] or WRITER in manifest['dev_writers']:raise ValueError('reserved writer leakage')
    return dict(train=train,unseen_prompt=test)


def collate(latents,records,vocab,ids,stats,device):
    length=max(latents[i].shape[0] for i in ids);chars=max(len(records[i]['text']) for i in ids)
    z=torch.zeros(len(ids),length,384,device=device);mask=torch.zeros(len(ids),length,dtype=torch.bool,device=device)
    labels=torch.full((len(ids),chars),-1,device=device,dtype=torch.long)
    for j,sid in enumerate(ids):
        n=latents[sid].shape[0];z[j,:n]=transform(latents[sid].to(device),stats);mask[j,:n]=True
        labels[j,:len(records[sid]['text'])]=torch.tensor([vocab.index(c) for c in records[sid]['text']],device=device)
    return z,mask,labels


def noise_for(ids,latents,seed,device):
    n=max(latents[i].shape[0] for i in ids);result=torch.zeros(len(ids),n,384,device=device)
    for j,sid in enumerate(ids):
        g=torch.Generator(device=device);g.manual_seed(int(hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()[:15],16))
        result[j,:len(latents[sid])]=torch.randn(latents[sid].shape,generator=g,device=device)
    return result


@torch.no_grad()
def read_sequence(reader,points,vocab):
    """Evaluate actually DECODED XY+hard pen states, not opaque generated z.

    Repack known polyphase fields solely for the dedicated research reader.
    Generation free-stop and oracle-window OCR both reported; no forced EOC.
    """
    n=len(points);t=(n+7)//8*8;fields=points.new_zeros(t,5);fields[:,4]=1.;fields[:n]=points
    latent=points.new_zeros(1,384,t//8);latent[0,:40]=fields.reshape(-1,40).T
    pm=torch.arange(t,device=points.device)[None]<n;lm=pm.reshape(1,-1,8).any(-1)
    logits=reader(latent,padding_mask=~lm,point_mask=pm)
    return greedy_ctc(logits[:(n+3)//4,0].argmax(-1).tolist(),vocab)


@torch.no_grad()
def decode_sample(codec,reader,z,record,vocab):
    from model.losses import mixture_expectation
    n=z.shape[0]*8;pm=torch.ones(1,n,dtype=torch.bool,device=z.device)
    out=codec.decode(z.T[None],padding_mask=~pm);xy=mixture_expectation(out)[0]
    states=out[0,:3].argmax(0);points=torch.cat((xy,torch.nn.functional.one_hot(states,3).float()),1)
    eoc=(states==2).nonzero().flatten();stop=int(eoc[0])+1 if len(eoc) else n
    free=read_sequence(reader,points[:stop],vocab);window=read_sequence(reader,points[:record['points']],vocab)
    return points.cpu().numpy(),dict(free_decoded=free,oracle_window_decoded=window,free_errors=edit_distance(record['text'],free),window_errors=edit_distance(record['text'],window),characters=len(record['text']),first_eoc_point=stop if len(eoc) else None,generated_points_at_stop=stop,oracle_points=record['points'],internal_eoc_count=int(((states==2)&(torch.arange(n,device=z.device)<record['points']-1)).sum()),state_counts=torch.bincount(states,minlength=3).cpu().tolist())


def summarize(rows,splits):
    result={}
    for split,ids in splits.items():
        result[split]={}
        policies=sorted({r['policy'] for r in rows})
        for policy in policies:
            q=[r for r in rows if r['sample_id'] in ids and r['policy']==policy]
            if q:result[split][policy]=dict(evaluations=len(q),free_cer=sum(r['free_errors'] for r in q)/sum(r['characters'] for r in q),oracle_window_cer=sum(r['window_errors'] for r in q)/sum(r['characters'] for r in q),free_exact=sum(r['free_errors']==0 for r in q),window_exact=sum(r['window_errors']==0 for r in q),mean_generated_stop=float(np.mean([r['generated_points_at_stop'] for r in q])),missing_eoc=sum(r['first_eoc_point'] is None for r in q))
    return result


@torch.no_grad()
def evaluate(denoiser,codec,reader,latents,records,vocab,stats,splits,targets,folder,step,alpha):
    denoiser.eval();ids=sorted(i for group in splits.values() for i in group);rows=[];device=next(denoiser.parameters()).device
    file=Path(folder)/f'evaluation-{step}.h5';hf=h5py.File(file,'w')
    try:
        for policy,cfg,drop in [('correct',1.,False),('null',1.,True),('swapped',1.,False),('correct_cfg3',3.,False)]:
            for seed in (9142,9143):
                for start in range(0,len(ids),8):
                    batch=ids[start:start+8];_,mask,text=collate(latents,records,vocab,batch,stats,device)
                    if policy=='swapped':
                        other=[ids[(ids.index(i)+1)%len(ids)] for i in batch];_,_,text=collate(latents,records,vocab,other,stats,device)
                    noise=noise_for(batch,latents,seed,device)
                    sample=ddim_sample(denoiser,noise,text,mask,alpha,steps=50,guidance=cfg,drop_text=drop);sample=transform(sample,stats,inverse=True)
                    for j,sid in enumerate(batch):
                        z=sample[j,:len(latents[sid])];points,metrics=decode_sample(codec,reader,z,records[sid],vocab)
                        true=targets[sid];n=len(true);geometry=local_geometry(points[:n,:2],true[:,:2],true[:,2:].argmax(1))
                        row=dict(sample_id=sid,text=records[sid]['text'],writer_id=WRITER,policy=policy,seed=seed,geometry=geometry,pen_aligned_reference=boundary_metrics(points[:n,2:].argmax(1),true[:,2:].argmax(1)),**metrics)
                        row['conditioning_text']=records[ids[(ids.index(sid)+1)%len(ids)]]['text'] if policy=='swapped' else ('' if drop else records[sid]['text'])
                        if policy=='swapped':row['free_cer_to_swapped']=edit_distance(row['conditioning_text'],row['free_decoded'])/len(row['conditioning_text'])
                        rows.append(row);group=hf.create_group(f'{policy}/{seed}/{sid}');group.create_dataset('points',data=points,compression='gzip');group.create_dataset('latent',data=z.cpu().numpy(),compression='gzip');group.create_dataset('initial_noise',data=noise[j,:len(latents[sid])].cpu().numpy(),compression='gzip');group.attrs['row']=json.dumps(row)
    finally:hf.close()
    result=dict(step=step,lines=rows,groups=summarize(rows,splits),packed_h5_sha256=file_sha(file))
    (Path(folder)/f'eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(step=step,generated=result['groups']),flush=True)
    denoiser.train();return result


@torch.no_grad()
def fixed_denoising(model,latents,records,vocab,stats,ids,alpha,arm):
    model.eval();values={}
    for t in (0,250,500,900,999):
        scores=[]
        for start in range(0,len(ids),8):
            group=ids[start:start+8];clean,mask,text=collate(latents,records,vocab,group,stats,next(model.parameters()).device);noise=noise_for(group,latents,8182,clean.device)
            level=torch.full((len(group),),t,device=clean.device,dtype=torch.long);x=forward_noise(clean,noise,level,alpha,mask)
            pred=model(x,level.float()/999,text,mask,torch.full((len(group),),arm=='no_text',device=clean.device,dtype=torch.bool))
            subsets={'all':slice(None),'xy':[i for i in range(40) if i%5<2],'pen':[i for i in range(40) if i%5>=2],'nominally_unused':slice(40,None)}
            scores.append({k:float(masked_mse(pred[:,:,v],clean[:,:,v],mask)) for k,v in subsets.items()})
        values[str(t)]={k:sum(v[k] for v in scores)/len(scores) for k in scores[0]}
    model.train();return values


def run(config,repo,root='/data',steps=8000,pool_sha=''):
    validate_budget(steps,pool_sha)
    if not torch.cuda.is_available():raise ValueError('bounded T4 required')
    torch.set_num_threads(4);root=Path(root);pool,m,vocab=load_pool(root,pool_sha);splits=select_scope(m);ids=sorted(i for v in splits.values() for i in v)
    codec,_,_,cfg,old_vocab,provenance=load(config,repo,root,SOURCE,SHA,writer_id=None)
    if old_vocab!=vocab:raise ValueError('vocabulary drift')
    codec=codec.cuda().eval().requires_grad_(False);reader,_=load_reader(root,cfg,'cuda');digest=tensor_digest(codec.state_dict());reader_digest=tensor_digest(reader.state_dict())
    batches,records,origins=load_batches(root,pool,m,vocab,ids,'cuda')
    # Metadata needs oracle real point count, not guessed from generated EOC.
    for sid in ids:records[sid]['points']=int(batches[sid][1].sum())
    out=root/'checkpoints/iam_generation_gate'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    for name in ['generation_study.py','latent_diffusion.py','ocr_joint_adapter.py','ocr_context_features.py','ocr_frame_study.py','ocr_recurrent.py','latent_integration.py']:
        q=out/'source-code/iam_tools'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['vae.py','blocks.py','losses.py']:
        q=out/'source-code/model'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((Path(repo)/'model'/name).read_bytes())
    latents={};targets={};preflight=[]
    with h5py.File(out/'source.h5','w') as hf,torch.no_grad():
        from model.losses import mixture_expectation
        for sid in ids:
            raw,pm,_=batches[sid];target,mu,lv,lm=encoded(codec,raw,pm);n=records[sid]['points'];latents[sid]=mu[0].T.detach();targets[sid]=torch.cat((target[0,:n],raw[0,2:,:n].T),1).cpu().numpy()
            pred=codec.decode(mu,padding_mask=~pm);xy=mixture_expectation(pred)[0,:n];states=pred[0,:3,:n].argmax(0);rmse=float((xy-target[0,:n]).square().mean().sqrt())
            if rmse>.0005 or not torch.equal(states,raw[0,2:,:n].argmax(0)):raise AssertionError('source codec preflight failed')
            decoded=read_sequence(reader,torch.cat((xy,torch.nn.functional.one_hot(states,3).float()),1),vocab)
            preflight.append(dict(sample_id=sid,rmse=rmse,decoded=decoded,errors=edit_distance(records[sid]['text'],decoded)))
            g=hf.create_group(sid);g.create_dataset('target',data=targets[sid],compression='gzip');g.create_dataset('latent_mean',data=mu[0].T.cpu().numpy(),compression='gzip');g.attrs['record']=json.dumps(records[sid])
    stats=fit_whitening(latents,splits['train']);torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in stats.items()},out/'whitening.pt')
    (out/'dataset.json').write_text(json.dumps(dict(splits=splits,records=records,source_preflight=preflight,pool_sha256=pool_sha,source_sha256=SHA,reader_sha256=READER_SHA),indent=2)+'\n')
    del batches
    torch.manual_seed(4142);prototype=TextLatentDenoiser(vocab_size=len(vocab)).cuda();initial=copy.deepcopy(prototype.state_dict());initial_digest=tensor_digest(initial)
    alpha=cosine_schedule(device='cuda');schedule=list(training_schedule(splits['train'],steps,8,seed=5142));results={}
    settings=dict(profile='bounded text-cross-attention latent generation diagnostic, NOT released InkDiT or English paper reproduction',model=prototype.config,source_rel=SOURCE,source_sha256=SHA,reader_rel=READER_REL,reader_sha256=READER_SHA,pool_sha256=pool_sha,codec_cfg=cfg,splits=splits,vocab=vocab,writer_id=WRITER,unseen_form=HELD_FORM,
        initial_state_sha256=initial_digest,schedule_seed=5142,noise_seed=6142,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),max_updates=steps,max_wall_seconds_per_arm=1800,lr=3e-4,betas=[.9,.99],weight_decay=.01,clip=1.,batch=8,diffusion_steps=1000,sampling_steps=50,target='deterministic mu, ALL384 channels, x0 MSE',whitening_sha256=file_sha(out/'whitening.pt'),source_h5_sha256=file_sha(out/'source.h5'),text_drop_probability=.1,eval_steps=sorted(set([0,steps//2,steps])),prefix='none; pure noise sampling',length='oracle ceil(trueN/8) used by both arms; no length predictor; not duration-independent generation',eval_reader='frozen corpus-familiar BiGRU applied to DECODED XY+predictedhard pens; no OCR loss',selection='TRAIN fixed t999 activeXY+pen score; DEV/unseen never selects/stops',scope='generator32TRAIN/8unseen wholeform samewriter; all from OCRreaderTRAIN corpus, not independent reader/writer benchmark; no writer control claim',dtype='FP32',runtime=str(torch.__version__))
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    for arm in ARMS:
        model=copy.deepcopy(prototype);model.load_state_dict(initial);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=.01)
        folder=out/arm;folder.mkdir();history=[];best=None;best_step=0;noise_rng=torch.Generator(device='cuda').manual_seed(6142);start=time.monotonic();stop='budget_completed'
        def save(name,step):
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),config=dict(settings,arm=arm),step=step,noise_rng_state=noise_rng.get_state(),initial_state_sha256=initial_digest),folder/name)
        def assessment(step):
            scores=fixed_denoising(model,latents,records,vocab,stats,splits['train'],alpha,arm);(folder/f'denoising-{step}.json').write_text(json.dumps(scores,indent=2)+'\n')
            # no_text arm cannot condition even at evaluation; correct CFG
            # branches route NULL automatically through an eval-only wrapper.
            class NoText(torch.nn.Module):
                def __init__(self,base):super().__init__();self.base=base;self.eval()
                def forward(self,x,t,text,mask,drop):return self.base(x,t,text,mask,torch.ones_like(drop))
            evaluator=NoText(model.eval()) if arm=='no_text' else model.eval()
            row=evaluate(evaluator,codec,reader,latents,records,vocab,stats,splits,targets,folder,step,alpha);model.train()
            history.append(dict(step=step,denoising=scores,groups=row['groups']));return scores['999']['xy']+scores['999']['pen']
        score=assessment(0);best=score;save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0)
        with (folder/'metrics.jsonl').open('w') as log:
            for step,group in enumerate(schedule,1):
                clean,mask,text=collate(latents,records,vocab,group,stats,'cuda');eps=torch.randn(clean.shape,device='cuda',generator=noise_rng);t=torch.randint(0,1000,(8,),device='cuda',generator=noise_rng);drop=torch.rand(8,device='cuda',generator=noise_rng)<.1
                noise_sha=hashlib.sha256(eps.cpu().numpy().tobytes()+t.cpu().numpy().tobytes()+drop.cpu().numpy().tobytes()).hexdigest()
                if arm=='no_text':drop=torch.ones_like(drop)
                x=forward_noise(clean,eps,t,alpha,mask);pred=model(x,t.float()/999,text,mask,drop);loss=masked_mse(pred,clean,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,sample_ids=group,loss=float(loss),gradient_norm=norm,was_clipped=norm>1,noise_sha256=noise_sha))+'\n')
                if step%200==0:log.flush();print(dict(arm=arm,step=step,loss=float(loss),elapsed=time.monotonic()-start),flush=True)
                limit=time.monotonic()-start>1800
                if step in settings['eval_steps'] or limit:
                    score=assessment(step);save('checkpoint-last.pt',step)
                    if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                if limit:stop='wall_limit';break
        if tensor_digest(codec.state_dict())!=digest or tensor_digest(reader.state_dict())!=reader_digest or any(p.grad is not None for p in list(codec.parameters())+list(reader.parameters())):raise AssertionError('frozen codec/reader changed')
        save('checkpoint-last.pt',step);result=dict(arm=arm,last_step=step,best_step=best_step,history=history,stop=stop,elapsed_seconds=time.monotonic()-start,codec_reader_unchanged=True,selected_sha256=file_sha(folder/'checkpoint-best.pt'),last_sha256=file_sha(folder/'checkpoint-last.pt'),final_noise_rng_sha256=hashlib.sha256(noise_rng.get_state().cpu().numpy().tobytes()).hexdigest(),not_promoted=True)
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[arm]=result;del model,optimizer;torch.cuda.empty_cache()
    if file_sha(root/SOURCE)!=SHA or file_sha(root/READER_REL)!=READER_SHA:raise AssertionError('immutable checkpoint drift')
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n');return dict(output=str(out),arms={a:dict(last=r['last_step'],best=r['best_step'],stop=r['stop']) for a,r in results.items()})
