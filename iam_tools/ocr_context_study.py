"""Paired frozen-codec OCR feature/context controls, not VAE joint training."""
import copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from .writer_expansion import load,device_batches
from .latent_integration import encoded
from .frozen_ocr_study import SOURCE,SHA,cache_latents,bucket_schedule,collate_latents,ocr_evaluate,summarize
from .ocr_context_features import unpack,transform,fit_stats,local_attention_mask,make_head
from .inkvae import greedy_ctc,edit_distance
from .pen_ab import file_sha

POLISHED='checkpoints/iam_frozen_ocr_study/20261007-015617/checkpoint-best.pt'
POLISHED_SHA='8738dc9b0b0615f60e6f1b0e740205ed31408561b456611619dbf19f56b154f6'
ARMS={'global_raw':('global_raw',None),'global_scaled':('global_scaled',None),
      'relative_scaled':('relative_scaled',None),'relative_local4':('relative_scaled',4)}


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def head_model(head):return SimpleNamespace(ocr_model=head,get_ocr_loss=head.get_ocr_loss,parameters=head.parameters)


@torch.no_grad()
def transport_preflight(cache,batches):
    rows=[]
    for sid,c in cache.items():
        raw,mask,_=batches[sid];n=int(mask.sum());f,real=unpack(c['mu'],c['mask'])
        packed=f.permute(0,3,1,2).reshape(1,-1,5)[0,:n]
        difference=float((packed[:,:2]-raw[0,:2,:n].T*.01).abs().max())
        flips=int((packed[:,2:].argmax(1)!=raw[0,2:,:n].argmax(0)).sum())
        if int(real.sum())!=n or difference>1e-4 or flips:raise AssertionError((sid,n,int(real.sum()),difference,flips))
        rows.append(dict(sample_id=sid,real_points=n,max_packed_xy_difference=difference,pen_flips=flips))
    return dict(lines=rows,max_packed_xy_difference=max(r['max_packed_xy_difference'] for r in rows),all_real_phase_masks_correct=True,all_pens_correct=True)


@torch.no_grad()
def pretrained_ablation(base,cache,texts,splits,vocab,root,out):
    """OOD perturbations measure dependence; NOT proof of the sole causal issue."""
    if file_sha(Path(root)/POLISHED)!=POLISHED_SHA:raise ValueError('pinned polished OCR source required')
    saved=torch.load(Path(root)/POLISHED,map_location='cpu',weights_only=True)
    head=copy.deepcopy(base.ocr_model).eval();head.load_state_dict({k[len('ocr_model.'):]:v for k,v in saved['model_state_dict'].items() if k.startswith('ocr_model.')})
    results={}
    for kind in ('native','no_y','no_pen','centroid_xy','width_length_only','local4'):
        rows=[]
        for sid,c in cache.items():
            z=c['mu'].clone();n=int(c['mask'].sum());f=z[:,:40].reshape(1,8,5,-1)
            if kind=='no_y':f[:,:,1]=0.
            if kind=='no_pen':f[:,:,2:]=0.
            if kind=='centroid_xy':f[:,:,:2]=f[:,:,:2].mean(1,keepdim=True)
            if kind=='width_length_only':
                width=f[:,:,0].max();f.zero_();f[:,:,0]=torch.linspace(0,1,z.shape[-1],device=z.device)[None,None,:]*width;z[:,40:]=0.
            attention=local_attention_mask(c['mask'],base.config.ocr_num_heads,4) if kind=='local4' else None
            ids=head(z,padding_mask=~c['mask'],attention_mask=attention)[:n,0].argmax(-1).tolist();decoded=greedy_ctc(ids,vocab)
            value=dict(decoded=decoded,errors=edit_distance(texts[sid],decoded),characters=len(texts[sid]),blank_frame_fraction=ids.count(0)/len(ids))
            rows.append(dict(sample_id=sid,text=texts[sid],mu=value,sampled=[value],ctc_loss=0.,adjacent_repeats=any(a==b for a,b in zip(texts[sid],texts[sid][1:]))))
        result=dict(kind=kind,lines=rows,groups={k:summarize(rows,ids) for k,ids in splits.items()})
        (out/f'pretrained-{kind}.json').write_text(json.dumps(result,indent=2)+'\n');results[kind]=result['groups']
    (out/'pretrained-ablation-summary.json').write_text(json.dumps(dict(source_rel=POLISHED,source_sha256=POLISHED_SHA,groups=results,caveat='OOD evaluation of an already-fit head: establishes dependence, not sole causality; no posterior sampling here'),indent=2)+'\n')
    return results


def run(config,repo,root='/data',steps=1000):
    if not torch.cuda.is_available() or not 1<=steps<=1500:raise ValueError('CUDA and bounded1–1500 updates per arm required')
    root=Path(root);torch.set_num_threads(4)
    base,samples,raw,cfg,vocab,prov=load(config,repo,root,SOURCE,SHA,writer_id=None)
    base.cuda().eval().requires_grad_(False);initial={k:v.cpu().clone() for k,v in base.state_dict().items()}
    batches=device_batches(raw,'cuda');texts={i:s[2] for i,s in samples.items()};cache=cache_latents(base,batches,texts)
    directory=root/'checkpoints/iam_ocr_context_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    (directory/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    preflight=transport_preflight(cache,batches);(directory/'transport-preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    pretrained_ablation(base,cache,texts,prov['splits'],vocab,root,directory)
    stats={m:fit_stats(cache,prov['splits']['train'],m) for m in ('global_scaled','relative_scaled')}
    (directory/'training-feature-statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    schedule=list(bucket_schedule(cache,prov['splits']['train'],steps));results={};initial_sha=None
    for name,(mode,radius) in ARMS.items():
        head=make_head(cfg,len(vocab)+1,mode,stats.get(mode),radius,seed=42).cuda();model=head_model(head)
        digest=tensor_digest(head.state_dict())
        if initial_sha is None:initial_sha=digest
        if digest!=initial_sha:raise AssertionError('unpaired head initialization')
        out=directory/name;out.mkdir();settings=dict(profile='frozen-transport-ocr-context-diagnostic',source_rel=SOURCE,source_sha256=SHA,
            head_initialization='fresh same seed42 weights across arms; blank bias0; NOT restored pretrained head',initial_head_tensor_sha256=digest,
            mode=mode,feature_stats=stats.get(mode),attention_radius=radius,ocr_num_layers=cfg['ocr_num_layers'],cfg=cfg,
            train_ids=prov['splits']['train'],held_out_ids=prov['splits']['held_out'],base_lr=5e-4,final_lr=1e-4,lr_drop_step=int(.75*steps),
            betas=[.9,.99],weight_decay=1e-4,grad_clip=5.,ocr_dropout=.1,physical_ocr_batch=16,raw_encoder_batch=1,
            unused_channel_policy='all344 zero ONLY at OCR input, all arms',post_eoc_phases='excluded from features and calibration',
            max_optimizer_updates=steps,max_wall_seconds=450,eval_every=250,posterior_draws=20,train_rng_seed=42,
            selection='train mean CER then CTC only; held-out reporting-only',geometry_optimization=False,
            research_only='standalone head+feature contract, NOT a model_state_dict VAE checkpoint',torch_version=str(torch.__version__))
        (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
        for f in ('ocr_context_study.py','ocr_context_features.py','frozen_ocr_study.py'):
            dest=out/'source-code/iam_tools'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).with_name(f).read_bytes())
        dest=out/'source-code/model/ocr.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path(repo)/'model/ocr.py').read_bytes())
        optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=1e-4);torch.manual_seed(42)
        def save(file,step):
            torch.save(dict(ocr_state_dict=head.state_dict(),ocr_optimizer_state_dict=optimizer.state_dict(),config=settings,
                updates=step,provenance=prov,rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all()),out/file)
        head.eval();first=ocr_evaluate(model,cache,texts,prov['splits'],vocab,out,0)
        best=(first['groups']['train']['mu']['cer'],first['groups']['train']['mean_ctc_loss']);best_step=0;save('head-best.pt',0)
        history=[];started=time.monotonic();stop='budget_completed';head.train()
        with (out/'metrics.jsonl').open('w') as log:
            for step,ids in enumerate(schedule,1):
                z,labels,mask=collate_latents(cache,ids);optimizer.zero_grad(set_to_none=True);loss=head.get_ocr_loss(z,labels,mask)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite context CTC: no update')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True));optimizer.step()
                log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5.,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
                if step==int(.75*steps):
                    for group in optimizer.param_groups:group['lr']=1e-4
                limit=time.monotonic()-started>450
                if step%250==0 or step==steps or limit:
                    row=ocr_evaluate(model,cache,texts,prov['splits'],vocab,out,step);score=(row['groups']['train']['mu']['cer'],row['groups']['train']['mean_ctc_loss'])
                    history.append(dict(step=step,groups=row['groups']))
                    if score<best:best=score;best_step=step;save('head-best.pt',step)
                if limit:stop='wall_limit';break
        save('head-last.pt',step)
        chosen=torch.load(out/'head-best.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['ocr_state_dict']);head.eval()
        selected=next(r for r in history if r['step']==best_step)['groups'] if best_step else first['groups']
        frozen=all(torch.equal(v,base.state_dict()[k].cpu()) for k,v in initial.items())
        if not frozen or file_sha(root/SOURCE)!=SHA or any(p.grad is not None for p in base.parameters()):raise AssertionError('entire codec frozen-state violation')
        result=dict(arm=name,initial=first['groups'],history=history,selected=selected,best_step=best_step,last_step=step,stop_reason=stop,
            selected_sha256=file_sha(out/'head-best.pt'),final_sha256=file_sha(out/'head-last.pt'),entire_codec_bitwise_unchanged=frozen,
            elapsed_training_evaluation_seconds=time.monotonic()-started,initial_head_tensor_sha256=digest)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[name]=result
        del head,optimizer,model;torch.cuda.empty_cache()
    (directory/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    return dict(output=str(directory),arms={k:dict(best_step=v['best_step'],train_cer=v['selected']['train']['mu']['cer'],held_out_cer=v['selected']['held_out']['mu']['cer']) for k,v in results.items()},entire_codec_bitwise_unchanged=True)
