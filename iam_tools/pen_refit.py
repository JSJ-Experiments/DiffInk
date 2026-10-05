"""CPU three-state B versus binary/known-final-EOC refit on frozen MSE geometry."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .inkvae import setup
from .autopsy import selected_sample, sample_sha
from .pen_ab import file_sha, clone_pen_head, merge_head, assert_nonpen_unchanged, boundary_metrics, render_snapshot
from .reconstruction import deterministic_forward, expected_xy

MSE_SHA='f5fa77cf63298bd8eb9b79f1eff35a50b7a455d7c7ba190232518d231834644a'
ARMS=('bounded_three_state','binary_forced_final')


def refit_loss(logits,targets,arm,gamma=2,cap=8):
    if arm not in ARMS:raise ValueError('unknown arm')
    if arm==ARMS[1]:
        logits=logits[:-1,:2];targets=targets[:-1]
        if (targets>1).any():raise ValueError('binary arm cannot contain internal EOC')
    counts=targets.bincount(minlength=logits.shape[-1]).float()
    weights=(counts[0]/counts.clamp_min(1)).sqrt().clamp(max=cap)
    weights[counts==0]=0
    ce=F.cross_entropy(logits,targets,reduction='none')
    return (weights[targets]*(1-ce.neg().exp()).pow(gamma)*ce).mean()


def refit_states(logits,arm):
    if arm==ARMS[0]:return logits.argmax(-1)
    if arm!=ARMS[1]:raise ValueError('unknown arm')
    states=logits[:,:2].argmax(-1);states[-1]=2
    return states


def run(repo,checkpoint,data_root,output_base):
    torch.set_num_threads(2)
    if file_sha(checkpoint)!=MSE_SHA:raise ValueError('exact deterministic checkpoint required')
    parent=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model,train,val,cfg,_=setup(Path(repo)/'configs/vae_iam_autopsy_resume.yaml',repo,data_root)
    try:
        if cfg!=parent['config']:raise ValueError('source config mismatch')
        sample=selected_sample(train,'c08-434z-05')
        if sample_sha(sample)!=parent['sample_sha256']:raise ValueError('sample mismatch')
        batch=train.collate_fn([sample]);model.load_state_dict(parent['model_state_dict']);model.eval().requires_grad_(False)
        source=parent['model_state_dict'];del parent
        cache=[]
        hook=model.transformer_decoder.fc.register_forward_pre_hook(lambda m,args:cache.append(args[0].detach().clone()))
        with torch.no_grad():base=deterministic_forward(model,batch,.01)
        hook.remove();features=cache[0][0,:len(sample[1])];xy=expected_xy(base)[0,:len(sample[1])].numpy()
        targets=sample[1][:,2:].argmax(-1)
        directory=Path(output_base)/time.strftime('%Y%m%d-%H%M%S',time.gmtime());directory.mkdir(parents=True,exist_ok=False)
        started=time.monotonic();arms={}
        for arm in ARMS:
            path=directory/arm;path.mkdir();head=clone_pen_head(model.transformer_decoder.fc)
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.99),weight_decay=0)
            rows=[]
            for step in range(1001):
                if step:
                    optimizer.zero_grad(set_to_none=True);loss=refit_loss(head(features),targets,arm)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite pen loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10,error_if_nonfinite=True);optimizer.step()
                if step%100:continue
                with torch.no_grad():states=refit_states(head(features),arm).numpy()
                merged=merge_head(source,head);assert_nonpen_unchanged(source,merged)
                model.load_state_dict(merged)
                with torch.no_grad():output=deterministic_forward(model,batch,.01)
                if not torch.equal(base[:,3:],output[:,3:]) or not torch.equal(expected_xy(base),expected_xy(output)):
                    raise AssertionError('frozen geometry changed')
                row={'step':step,**boundary_metrics(states,targets.numpy()),'non_pen_state_unchanged':True,'gmm_output_unchanged':True,'expected_xy_unchanged':True,
                     'final_eoc_forced_using_known_length':arm==ARMS[1]}
                rows.append(row)
                with (path/'metrics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
                render_snapshot(path,f'step-{step}',xy,states,sample[1].numpy(),.01,sample[2],row)
                model.load_state_dict(source)
            merged=merge_head(source,head)
            torch.save({'model_state_dict':merged,'source_sha256':MSE_SHA,'pen_policy':arm,'pen_updates':1000,
                        'config':cfg,'sample_sha256':sample_sha(sample),'forced_final_eoc':arm==ARMS[1]},path/'checkpoint.pt')
            arms[arm]={'initial':rows[0],'final':rows[-1],'checkpoint_sha256':file_sha(path/'checkpoint.pt')}
        # Prefer the genuine three-state stop classifier on a tie.
        winner=max(ARMS,key=lambda arm:arms[arm]['final']['pen_up_f1'])
        result={'gpu':False,'training':True,'stage':'frozen-mse-pen-refit','source_sha256':MSE_SHA,
                'updates_per_arm':1000,'gamma':2,'cap':8,'lr':.001,'padding_excluded':True,
                'initial_head':'identical original MSE pen rows','arms':arms,'winner_by_pen_up_f1':winner,
                'binary_is_known_length_diagnostic_not_learned_stopping':True,'geometry_bitwise_unchanged':True,
                'output':str(directory),'elapsed_seconds':time.monotonic()-started}
        if file_sha(checkpoint)!=MSE_SHA:raise AssertionError('parent changed')
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        page=['<!doctype html><meta charset="utf-8"><h1>Frozen MSE pen refit</h1><p>Binary final EOC is forced using known target length; it does NOT demonstrate learned stopping.</p>']
        for arm in ARMS:page.append(f'<h2>{arm}</h2><img style="max-width:100%" src="{arm}/step-1000-comparison.png">')
        (directory/'index.html').write_text('\n'.join(page));return result
    finally:train.hf.close();val.hf.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='third_party/DiffInk');p.add_argument('--data-root',default='data/diffink/iam_overfit')
    p.add_argument('--checkpoint',default='data/checkpoints/iam_autopsy/reconstruction/20261005-091827/deterministic_mse/checkpoint.pt')
    p.add_argument('--output-base',default='data/checkpoints/iam_autopsy/pen_refit');a=p.parse_args()
    print(json.dumps(run(a.repo,a.checkpoint,a.data_root,a.output_base),indent=2))
if __name__=='__main__':main()
