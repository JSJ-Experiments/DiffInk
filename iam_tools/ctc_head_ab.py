"""Frozen English latents: CTC blank-bias A/B, never updates geometry/pen/style."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .objective_study import load
from .inkvae import greedy_ctc, edit_distance
from .pen_ab import file_sha


def run(config,repo,source,sha,steps=1000):
    if not 1 <= steps <= 3000:raise ValueError('CTC head chunk 1..3000')
    if file_sha(source)!=sha:raise ValueError('pinned reconstruction checkpoint required')
    from utils.mask import downsample_mask
    model,samples,batches,cfg,hashes,_=load(config,repo)
    saved=torch.load(source,map_location='cpu',weights_only=True)
    assert saved['sample_ids']==cfg['sample_ids']
    assert saved.get('pen_refit_updates') and saved.get('pen_refit_source_sha256')
    model.load_state_dict(saved['model_state_dict'],strict=True);model.apply_checkpoint_contract(saved)
    model.to('cuda').eval().requires_grad_(False)
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('ocr_model.')}
    initial={k:v.clone() for k,v in model.ocr_model.state_dict().items()}
    vocab=list(json.loads(Path(cfg['text_file']).read_text()))
    cache=[]
    with torch.no_grad():
        for sample,batch in zip(samples,batches):
            raw,mask,text,*_=batch;raw=raw.to('cuda').transpose(1,2);mask=mask.to('cuda')
            mu=model.conv_mu(model.encoder(model.to_model_space(raw))).detach()
            lm=downsample_mask(mask,8);labels=text.to('cuda')
            repeats=sum(a==b for a,b in zip(sample[2],sample[2][1:]))
            assert int(lm.sum())>=len(sample[2])+repeats
            cache.append((mu,labels,lm))
    directory=Path('/data/checkpoints/iam_ctc_head_ab')/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
    results={};started=time.monotonic()
    for name,bias in [('blank_minus5',-5.),('blank_zero',0.)]:
        model.ocr_model.load_state_dict(initial);model.ocr_model.requires_grad_(True)
        with torch.no_grad():model.ocr_model.output_fc.bias[0]=bias
        torch.manual_seed(42)
        optimizer=torch.optim.AdamW(model.ocr_model.parameters(),lr=5e-4,betas=(.9,.99),weight_decay=1e-4)
        arm_dir=directory/name;arm_dir.mkdir();fixed=[]
        def evaluate(step):
            was=model.ocr_model.training;model.ocr_model.eval();rows=[]
            with torch.no_grad():
                for j,((mu,labels,lm),sample) in enumerate(zip(cache,samples)):
                    logits=model.ocr_model(mu)[:int(lm.sum()),0];ids=logits.argmax(-1).tolist()
                    decoded=greedy_ctc(ids,vocab);error=edit_distance(sample[2],decoded)
                    rows.append(dict(id=cfg['sample_ids'][j],truth=sample[2],decoded=decoded,
                                     cer=error/len(sample[2]),errors=error,characters=len(sample[2]),
                                     adjacent_repeats=any(a==b for a,b in zip(sample[2],sample[2][1:])),
                                     blank_frame_fraction=ids.count(0)/len(ids),
                                     ctc_loss=float(model.get_ocr_loss(mu,labels,lm))))
            model.ocr_model.train(was)
            row=dict(step=step,character_weighted_cer=sum(r['errors'] for r in rows)/sum(r['characters'] for r in rows),
                     exact_lines=sum(r['decoded']==r['truth'] for r in rows),lines=rows,
                     mean_ctc_loss=float(np.mean([r['ctc_loss'] for r in rows])))
            for repeated in (True,False):
                group=[r for r in rows if r['adjacent_repeats']==repeated]
                row['repeats' if repeated else 'no_repeats']={'lines':len(group),'cer':sum(r['errors'] for r in group)/sum(r['characters'] for r in group) if group else None}
            fixed.append(row);print(dict(arm=name,**{k:v for k,v in row.items() if k!='lines'}),flush=True)
            with (arm_dir/'fixed_metrics.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
        model.ocr_model.train();evaluate(0);rng=np.random.default_rng(42)
        log=(arm_dir/'metrics.jsonl').open('w')
        try:
            for step in range(1,steps+1):
                optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for j in rng.permutation(8):
                    mu,labels,lm=cache[int(j)];loss=model.get_ocr_loss(mu,labels,lm)
                    if not torch.isfinite(loss):raise FloatingPointError('CTC nonfinite')
                    (loss/8).backward();loss_sum+=float(loss.detach())/8
                norm=torch.nn.utils.clip_grad_norm_(model.ocr_model.parameters(),5,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                if step==int(.8*steps):
                    for group in optimizer.param_groups:group['lr']=1e-4
                log.write(json.dumps(dict(step=step,effective_batch_ctc=loss_sum,gradient_norm=float(norm)))+'\n')
                if step%250==0 or step==steps:evaluate(step)
        finally:log.close()
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items())
        state=dict(saved);state['model_state_dict']=model.state_dict();state['ctc_head_updates']=steps
        state['ctc_initial_blank_bias']=bias;state['ctc_head_source_sha256']=sha
        state.pop('optimizer_state_dict',None);torch.save(state,arm_dir/'checkpoint.pt')
        results[name]=dict(initial=fixed[0],final=fixed[-1],geometry_pen_exactly_unchanged=True)
    assert file_sha(source)==sha
    result=dict(source_sha256=sha,output=str(directory),updates_per_arm=steps,training_latent='cached_mu',
                train_batch_size=1,accumulation=8,ocr_dropout=.1,ctc_path='VAE.get_ocr_loss',arms=results,
                joint_vae_training=False,kl_style_off=True,training_samples_only=True,elapsed_seconds=time.monotonic()-started)
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='arms'}
