"""Frozen faithful codec: train-only OCR refit on cached minimal-padding latents.

Raw lines are NEVER batched through temporal GroupNorm. Only the independent OCR
head sees padded batches, with attention masks. Held-out lines are reporting-only.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .writer_expansion import load, device_batches, evaluate as geometry_evaluate
from .latent_integration import encoded
from .codec_kl_study import candidate_valid
from .inkvae import greedy_ctc, edit_distance
from .pen_ab import file_sha

SOURCE='checkpoints/iam_codec_kl_study/20261007-012331/pen_bias_kl1e-6/checkpoint-best.pt'
SHA='9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625'


def bucket_schedule(cache, train_ids, steps, batch_size=16, seed=42):
    """Full epochs, shuffled 64-line length windows; no held-out IDs accepted."""
    if not train_ids or len(set(train_ids))!=len(train_ids) or not 1<=batch_size<=64 or steps<1:
        raise ValueError('unique nonempty IDs and positive bounded batch/steps required')
    ordered=sorted(train_ids,key=lambda i:cache[i]['mu'].shape[-1]);rng=np.random.default_rng(seed);done=0
    while done<steps:
        groups=[]
        for start in range(0,len(ordered),64):
            window=list(rng.permutation(ordered[start:start+64]))
            groups.extend([window[j:j+batch_size] for j in range(0,len(window),batch_size)])
        for j in rng.permutation(len(groups)):
            yield [str(i) for i in groups[int(j)]];done+=1
            if done==steps:return


def collate_latents(cache, ids, sampled=False):
    """Cache has [1,C,T] detached means/logvars; only right-pad OCR inputs."""
    if not ids or len(ids)!=len(set(ids)):raise ValueError('nonempty unique OCR batch required')
    ref=cache[ids[0]]['mu'];t=max(cache[i]['mu'].shape[-1] for i in ids);l=max(cache[i]['labels'].shape[-1] for i in ids)
    z=ref.new_zeros(len(ids),ref.shape[1],t);mask=torch.zeros(len(ids),t,dtype=torch.bool,device=ref.device)
    labels=torch.full((len(ids),l),-1,dtype=torch.long,device=ref.device)
    for j,sid in enumerate(ids):
        item=cache[sid];mu=item['mu'];value=mu+torch.randn_like(mu)*(.5*item['lv']).exp() if sampled else mu
        n=mu.shape[-1];z[j,:,:n]=value[0];mask[j,:n]=item['mask'][0];labels[j,:item['labels'].shape[-1]]=item['labels'][0]
    return z,labels,mask


@torch.no_grad()
def cache_latents(model,batches,texts):
    cache={}
    for sid,(raw,mask,labels) in batches.items():
        _,mu,lv,lm=encoded(model,raw,mask)
        required=len(texts[sid])+sum(a==b for a,b in zip(texts[sid],texts[sid][1:]))
        if int(lm.sum())<required:raise ValueError('CTC-infeasible prepared line: '+sid)
        cache[sid]=dict(mu=mu.detach(),lv=lv.detach(),mask=lm.detach(),labels=labels.detach())
    return cache


def summarize(rows,ids):
    selected=[r for r in rows if r['sample_id'] in ids]
    result={}
    for kind in ('mu','sampled'):
        values=[r['mu'] for r in selected] if kind=='mu' else [v for r in selected for v in r['sampled']]
        result[kind]=dict(cer=sum(v['errors'] for v in values)/sum(v['characters'] for v in values),
            exact_lines=sum(v['errors']==0 for v in values),evaluations=len(values),
            blank_frame_fraction=float(np.mean([v['blank_frame_fraction'] for v in values])))
        for repeated in (True,False):
            sub=[v for r in selected if r['adjacent_repeats']==repeated for v in ([r['mu']] if kind=='mu' else r['sampled'])]
            result[kind]['repeats' if repeated else 'no_repeats']=dict(evaluations=len(sub),cer=sum(v['errors'] for v in sub)/sum(v['characters'] for v in sub) if sub else None)
    result['mean_ctc_loss']=float(np.mean([r['ctc_loss'] for r in selected]));return result


@torch.no_grad()
def ocr_evaluate(model,cache,texts,splits,vocab,folder,step,draws=20):
    was=model.ocr_model.training;model.ocr_model.eval();rows=[];device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        for j,sid in enumerate(sorted(cache)):
            c=cache[sid];mu=c['mu'];torch.manual_seed(1042+j*100)
            z=torch.cat([mu]+[mu+torch.randn_like(mu)*(.5*c['lv']).exp() for _ in range(draws)])
            logits=model.ocr_model(z,padding_mask=(~c['mask']).expand(len(z),-1));n=int(c['mask'].sum());variants=[]
            for ids in logits[:n].argmax(-1).T.tolist():
                decoded=greedy_ctc(ids,vocab);variants.append(dict(decoded=decoded,errors=edit_distance(texts[sid],decoded),characters=len(texts[sid]),blank_frame_fraction=ids.count(0)/len(ids)))
            loss=model.get_ocr_loss(mu,c['labels'],c['mask'])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite evaluation CTC')
            rows.append(dict(sample_id=sid,text=texts[sid],adjacent_repeats=any(a==b for a,b in zip(texts[sid],texts[sid][1:])),mu=variants[0],sampled=variants[1:],ctc_loss=float(loss)))
    model.ocr_model.train(was)
    row=dict(step=step,lines=rows,groups={name:summarize(rows,ids) for name,ids in splits.items()})
    (Path(folder)/f'ocr-{step}.json').write_text(json.dumps(row,indent=2)+'\n')
    print(dict(step=step,ocr={k:row['groups'][k] for k in ('train','held_out')}),flush=True);return row


@torch.no_grad()
def batch_parity(model,cache,ids):
    """Head dropout OFF; cached individual lines must survive masked batching."""
    model.ocr_model.eval();z,labels,mask=collate_latents(cache,ids)
    output=model.ocr_model(z,padding_mask=~mask);maximum=0.;flips=0;single_losses=[]
    for j,sid in enumerate(ids):
        c=cache[sid];n=int(c['mask'].sum());one=model.ocr_model(c['mu'],padding_mask=~c['mask'])[:n,0]
        maximum=max(maximum,float((one-output[:n,j]).abs().max()));flips+=int((one.argmax(-1)!=output[:n,j].argmax(-1)).sum())
        single_losses.append(model.get_ocr_loss(c['mu'],c['labels'],c['mask']))
    delta=float((torch.stack(single_losses).mean()-model.get_ocr_loss(z,labels,mask)).abs())
    result=dict(max_valid_logit_difference=maximum,argmax_flips=flips,ctc_loss_difference=delta)
    if maximum>2e-4 or delta>2e-5:raise AssertionError(result)
    return result



def head_optimizer(head, parent):
    """Fresh for first refit; restore moments/counters for controlled continuation."""
    optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=1e-4)
    previous_updates=int(parent.get('ctc_head_updates',0))
    if previous_updates:
        if 'ocr_optimizer_state_dict' not in parent:raise ValueError('head continuation requires saved optimizer moments')
        optimizer.load_state_dict(copy.deepcopy(parent['ocr_optimizer_state_dict']))
        for group in optimizer.param_groups:group['lr']=1e-4
    return optimizer,previous_updates


def head_training_loss(model,cache,ids,resumed=False,sampled=False):
    if resumed:
        # Both controls consume noise and use two forwards: same dropout RNG.
        mu,labels,mask=collate_latents(cache,ids)
        z,_,_=collate_latents(cache,ids,sampled=True)
        return .5*model.get_ocr_loss(mu,labels,mask)+.5*model.get_ocr_loss(z if sampled else mu,labels,mask)
    z,labels,mask=collate_latents(cache,ids,sampled=sampled)
    return model.get_ocr_loss(z,labels,mask)

def run(config,repo,root='/data',steps=1000,metric_pool=None,source=SOURCE,sha=SHA,sampled=False):
    if not torch.cuda.is_available() or not 1<=steps<=3000:raise ValueError('CUDA and 1–3000 bounded OCR updates required')
    root=Path(root);torch.set_num_threads(4)
    model,samples,raw,cfg,vocab,prov=load(config,repo,root,source,sha,writer_id=None)
    parent=torch.load(root/source,map_location='cpu',weights_only=True)
    if len(prov['splits']['train'])!=192 or len(prov['splits']['held_out'])!=32:raise ValueError('192/32 prepared split required')
    model.cuda().eval().requires_grad_(False);frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('ocr_model.')}
    batches=device_batches(raw,'cuda');texts={i:s[2] for i,s in samples.items()};cache=cache_latents(model,batches,texts)
    out=root/'checkpoints/iam_frozen_ocr_study'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    settings=dict(parent['config'],profile='faithful-codec-frozen-ocr',source_checkpoint=str(root/source),source_sha256=sha,
        model_input_scale=.01,trans_dropout=0.,ctc_weight=1.,style_weight=0.,kl_weight=0.,gmm_weight=0.,pen_weight=0.,expected_xy_weight=0.,geometry_optimization=False,
        max_wall_seconds=900,stop_gate='finite CTC/grad;900s training/evaluation wall;frozen-state invariants;no held-out selection',
        ocr_training_latent='sampled_z' if sampled else 'cached_mu',ocr_dropout=.1,initial_blank_bias=0.,physical_ocr_batch=16,
        raw_encoder_batch=1,gradient_accumulation_steps=1,base_lr=5e-4,final_lr=1e-4,lr_drop_step=int(.75*steps),
        betas=[.9,.99],weight_decay=1e-4,grad_clip=5.,max_optimizer_updates=steps,seed=42,eval_every=250,training_schedule='seed42 bucketed complete OCR epochs, skipping source OCR updates on continuation',
        sampled_z_evaluations=20,attention_padding_mask=True,ctc_zero_infinity=False,torch_version=str(torch.__version__),
        optimizer='fresh head AdamW',checkpoint_selection='train CER, then train mean CTC only; held-out reporting-only',
        research_contract='frozen initialized transport OCR refit; not joint latent regularization or text generation')
    model.config.__dict__.update(settings)
    # Refitting existing weights, not randomly resetting the learned head. The
    # earlier 8-line blank-bias study favors zero. No additional A/B implied.
    resumed=bool(parent.get('ctc_head_updates'))
    if not resumed:
        with torch.no_grad():model.ocr_model.output_fc.bias[0]=0.
    model.ocr_model.ctc.zero_infinity=False
    model.ocr_model.requires_grad_(True)
    optimizer,previous_updates=head_optimizer(model.ocr_model,parent)
    if resumed:
        settings.update(base_lr=1e-4,final_lr=1e-4,lr_drop_step=None,initial_blank_bias='inherited fitted bias',
            source_ocr_updates=previous_updates,optimizer='restored head AdamW moments; constant LR1e-4',
            ocr_training_latent='half mean + half sampled z' if sampled else 'two mean forwards, paired noise consumed',
            paired_continuation='same source/Adam/RNG/schedule/dropout draws; target latent differs in second half')
    (out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    ids=sorted(prov['splits']['train'],key=lambda i:cache[i]['mu'].shape[-1]);probe=ids[::max(1,len(ids)//16)][:16]
    parity=batch_parity(model,cache,probe)
    (out/'batch-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n');(out/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    for f in ('frozen_ocr_study.py','writer_expansion.py','latent_integration.py'):
        dest=out/'source-code/iam_tools'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).with_name(f).read_bytes())
    for f in ('ocr.py','vae.py'):
        dest=out/'source-code/model'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((Path(repo)/'model'/f).read_bytes())
    geom=geometry_evaluate(model,samples,batches,prov['splits'],vocab,out/'geometry-source',0,metric_pool=metric_pool,draw_batch_size=21)
    if not candidate_valid(geom) or geom['groups']['held_out']['false_internal_eoc'] or not geom['groups']['held_out']['all_final_eoc_correct']:
        raise AssertionError('chosen codec source failed pen/EOC preflight')
    torch.manual_seed(42)
    if resumed:
        torch.set_rng_state(parent['rng_state_cpu']);torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
    def save(filename,step):
        state=dict(parent);state.update(model_state_dict=model.state_dict(),config=settings,provenance=prov,sample_ids=prov['splits']['train'],
            source_sha256=sha,ctc_head_updates=previous_updates+step,ocr_continuation_updates=step,ocr_optimizer_state_dict=optimizer.state_dict(),rng_state_cpu=torch.get_rng_state(),rng_state_cuda=torch.cuda.get_rng_state_all())
        state.pop('optimizer_state_dict',None);torch.save(state,out/filename)
    first=ocr_evaluate(model,cache,texts,prov['splits'],vocab,out,0);best=(first['groups']['train']['mu']['cer'],first['groups']['train']['mean_ctc_loss']);best_step=0
    save('checkpoint-initial.pt',0);save('checkpoint-best.pt',0);history=[first];started=time.monotonic();stop='budget_completed'
    model.ocr_model.train()
    with (out/'metrics.jsonl').open('w') as log:
        import itertools
        schedule=itertools.islice(bucket_schedule(cache,prov['splits']['train'],previous_updates+steps),previous_updates,None)
        for step,ids in enumerate(schedule,1):
            optimizer.zero_grad(set_to_none=True)
            loss=head_training_loss(model,cache,ids,resumed=resumed,sampled=sampled)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite CTC; no update applied')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.ocr_model.parameters(),5,error_if_nonfinite=True));optimizer.step()
            log.write(json.dumps(dict(step=step,sample_ids=ids,loss=float(loss.detach()),raw_gradient_norm=norm,was_clipped=norm>5.,lr=optimizer.param_groups[0]['lr']))+'\n');log.flush()
            if not resumed and step==int(.75*steps):
                for group in optimizer.param_groups:group['lr']=1e-4
            limit=time.monotonic()-started>900
            if step%250==0 or step==steps or limit:
                row=ocr_evaluate(model,cache,texts,prov['splits'],vocab,out,step);history.append(row)
                score=(row['groups']['train']['mu']['cer'],row['groups']['train']['mean_ctc_loss'])
                if score<best:best=score;best_step=step;save('checkpoint-best.pt',step)
                save(f'checkpoint-{step}.pt',step)
            if limit:stop='wall_limit';break
    save('checkpoint-last.pt',step)
    unchanged=all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in frozen.items())
    if not unchanged or file_sha(root/source)!=sha or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('ocr_model.')):raise AssertionError('frozen codec/source violation')
    chosen=torch.load(out/'checkpoint-best.pt',map_location='cpu',weights_only=True);model.load_state_dict(chosen['model_state_dict']);model.eval()
    final_geometry=geometry_evaluate(model,samples,batches,prov['splits'],vocab,out/'geometry-selected',best_step,metric_pool=metric_pool,draw_batch_size=21)
    # OCR metrics differ; geometry arrays must be bit-for-bit identical, all draws.
    same=True
    for sid in samples:
        for kind in ['mu']+[f'z-{i}' for i in range(20)]:
            same &= np.array_equal(np.load(out/'geometry-source'/f'step-0/{sid}/{kind}.npy'),np.load(out/'geometry-selected'/f'step-{best_step}/{sid}/{kind}.npy'))
    if not same:raise AssertionError('frozen codec trajectories changed')
    result=dict(output=str(out),source_rel=source,source_sha256=sha,best_step=best_step,last_step=step,stop_reason=stop,
        initial=first['groups'],history=[dict(step=r['step'],groups=r['groups']) for r in history[1:]],selected_sha256=file_sha(out/'checkpoint-best.pt'),
        final_sha256=file_sha(out/'checkpoint-last.pt'),all_non_ocr_state_bitwise_unchanged=unchanged,all_4704_saved_trajectories_bitwise_unchanged=same,
        source_unchanged=True,elapsed_training_evaluation_seconds=time.monotonic()-started,geometry_source=geom['groups'],geometry_selected=final_geometry['groups'],batch_parity=parity)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return {k:result[k] for k in ('output','best_step','last_step','stop_reason','selected_sha256','all_non_ocr_state_bitwise_unchanged','all_4704_saved_trajectories_bitwise_unchanged')}


def ablate_inactive_projection(model, first_inactive=40, require_exact=True):
    """Diagnostic ONLY: remove OCR's projection of proven pure-noise fields.

    Exact mode requires inactive mu rows identically zero FOR ALL inputs. Such
    mean-only OCR gradients are zero, leaving inherited noise-sensitive weights.
    Explicit require_exact=False is a RESEARCH ablation of nearly inactive rows;
    its mean/sample effects must be measured, not assumed. No automatic pruning.
    """
    if first_inactive<1 or first_inactive>=model.conv_mu.weight.shape[0]:raise ValueError('nonempty active/inactive groups required')
    w=model.conv_mu.weight[first_inactive:];b=model.conv_mu.bias[first_inactive:]
    exact=not bool(torch.count_nonzero(w) or torch.count_nonzero(b))
    if require_exact and not exact:raise ValueError('inactive means must be analytically identically zero, not just small on a dataset')
    before=model.ocr_model.input_proj.weight[:,first_inactive:].detach().clone()
    with torch.no_grad():model.ocr_model.input_proj.weight[:,first_inactive:]=0.
    return dict(first_inactive=first_inactive,inactive_channels=model.conv_mu.weight.shape[0]-first_inactive,
        mu_weight_and_bias_exactly_zero=exact,require_exact=require_exact,removed_projection_weight_l2=float(before.norm()),
        explanation='codec-specific no-training OCR projection ablation; approximate inactive means require measured output changes, NOT exact invariance or general latent pruning')
