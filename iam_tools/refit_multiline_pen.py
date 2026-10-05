"""CPU head-only pen refit on frozen multi-line geometry; no forced final EOC."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .objective_study import load
from .eightline import evaluate
from .pen_ab import boundary_metrics, file_sha


def run(checkpoint, output, steps=2000, gamma=2., cap=8.):
    torch.set_num_threads(2)
    repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
    model, samples, batches, cfg, _, _ = load(repo/'configs/engineering_english.yaml',
                                            repo, 'data/diffink/iam_overfit',
                                            'data/checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    source = Path(checkpoint); digest = file_sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    assert saved['sample_ids'] == cfg['sample_ids']
    model.load_state_dict(saved['model_state_dict'], strict=True);model.apply_checkpoint_contract(saved)
    model.requires_grad_(False);model.eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    fc = model.transformer_decoder.fc
    capture=[];hook=fc.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
    features=[];targets=[]
    try:
        with torch.no_grad():
            for sample,batch in zip(samples,batches):
                raw=batch[0].transpose(1,2);mu=model.conv_mu(model.encoder(model.to_model_space(raw)))
                model.decode(mu,padding_mask=~batch[1].bool())
                features.append(capture.pop()[0,:len(sample[1])].clone())
                targets.append(sample[1][:,2:].argmax(1))
    finally:hook.remove()
    head=torch.nn.Linear(fc.in_features,3)
    with torch.no_grad():
        head.weight.copy_(fc.weight[:3]);head.bias.copy_(fc.bias[:3])
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=0.,betas=(.9,.99))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();history=[]
    def snapshot(step):
        with torch.no_grad():
            rows=[boundary_metrics(head(f).argmax(1).numpy(),y.numpy()) for f,y in zip(features,targets)]
        row=dict(step=step,macro_f1=float(np.mean([r['pen_up_f1'] for r in rows])),
                 false_eoc=sum(r['non_final_false_eoc_count'] for r in rows),final_eoc_correct=sum(r['final_eoc_correct'] for r in rows),lines=rows)
        history.append(row);print({k:v for k,v in row.items() if k!='lines'},flush=True)
    snapshot(0)
    from .pen_refit import refit_loss
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.stack([refit_loss(head(f),y,'bounded_three_state',gamma=gamma,cap=cap) for f,y in zip(features,targets)]).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if step%500==0 or step==steps:snapshot(step)
    with torch.no_grad():
        fc.weight[:3].copy_(head.weight);fc.bias[:3].copy_(head.bias)
    weights = 'transformer_decoder.fc.weight'; bias = 'transformer_decoder.fc.bias'
    for k,value in original.items():
        actual=model.state_dict()[k]
        assert torch.equal(actual[3:] if k in (weights,bias) else actual,value[3:] if k in (weights,bias) else value), k
    assert file_sha(source)==digest
    with torch.no_grad():
        for sample,batch,f in zip(samples,batches,features):
            raw=batch[0].transpose(1,2);mu=model.conv_mu(model.encoder(model.to_model_space(raw)))
            output_now=model.decode(mu,padding_mask=~batch[1].bool())
            original_gmm=f@original[weights][3:].T+original[bias][3:]
            # Matmul shape differs from the original full 123-row readout; use
            # a tolerance for recomputation, while row/state invariants are exact.
            torch.testing.assert_close(output_now[0,3:,:len(sample[1])].T,original_gmm,rtol=1e-5,atol=1e-5)
    saved['model_state_dict']=model.state_dict();saved['pen_refit_source_sha256']=digest
    saved['pen_refit_updates']=steps;saved['pen_refit_gamma']=gamma;saved['pen_refit_cap']=cap
    # A refitted head invalidates its previous Adam moments; omit optimizer state.
    saved.pop('optimizer_state_dict',None)
    torch.save(saved,output/'checkpoint.pt')
    final=evaluate(model,samples,batches,cfg,'cpu',steps,output,sampled_count=20)
    result=dict(training=True,gpu=False,forced_final_eoc=False,source_sha256=digest,
                updates=steps,gamma=gamma,cap=cap,geometry_state_exactly_unchanged=True,
                elapsed_seconds=time.monotonic()-started,history=history,final=final)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--gamma',type=float,default=2);p.add_argument('--cap',type=float,default=8);a=p.parse_args()
    run(a.checkpoint,a.output,a.steps,a.gamma,a.cap)
