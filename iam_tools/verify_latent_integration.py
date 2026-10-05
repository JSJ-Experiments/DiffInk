"""CPU reload/invariant checks and paired-noise fixed-reference-variance control."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .latent_integration import SOURCE_REL,SOURCE_SHA,encoded
from .objective_study import load
from .pen_ab import file_sha,boundary_metrics
from .trajectory_geometry import geometry_metrics
from .report_curve_study import aggregate
from .inkvae import greedy_ctc,edit_distance


def verify(directory,repo='third_party/DiffInk',data_root='data'):
    torch.set_num_threads(2);root=Path(data_root);directory=Path(directory)
    study=json.loads((directory/'study.json').read_text());entry=study['outputs']['ocr']
    if not entry['gate_passed']:raise ValueError('joint OCR stage not promoted')
    source=root/SOURCE_REL
    if file_sha(source)!=SOURCE_SHA:raise ValueError('reference changed')
    model,samples,raw_batches,cfg,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root/'diffink/iam_overfit',root/'checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    ref=torch.load(source,map_location='cpu',weights_only=True)
    selected=directory/'ocr/checkpoint-best.pt';saved=torch.load(selected,map_location='cpu',weights_only=True)
    expected=json.loads((directory/'ocr/result.json').read_text())['selected_sha256']
    if file_sha(selected)!=expected:raise ValueError('selected checkpoint hash mismatch')
    state=saved['model_state_dict']
    initial_joint=torch.load(directory/'ocr/checkpoint-initial.pt',map_location='cpu',weights_only=True)['model_state_dict']
    changed_ocr=sum(k.startswith('ocr_model.') and not torch.equal(v,saved['model_state_dict'][k]) for k,v in initial_joint.items())
    if not changed_ocr:raise AssertionError('joint OCR parameters never updated')
    chain=[];previous=ref['model_state_dict']
    for stage in ('sampled','kl','ocr'):
        initial=torch.load(directory/stage/'checkpoint-initial.pt',map_location='cpu',weights_only=True)['model_state_dict']
        keys=[k for k in previous if stage!='ocr' or not k.startswith('ocr_model.')]
        assert all(torch.equal(previous[k],initial[k]) for k in keys),stage
        best=torch.load(directory/stage/'checkpoint-best.pt',map_location='cpu',weights_only=True)
        chain.append(dict(stage=stage,initial_matches_selected_parent=True,ocr_transfer_excluded=stage=='ocr',selected_sha256=file_sha(directory/stage/'checkpoint-best.pt')))
        previous=best['model_state_dict']
    del previous,initial_joint,initial,best
    assert all(torch.isfinite(v).all() for v in state.values())
    for key,old in ref['model_state_dict'].items():
        if key.startswith('style_classifier.'):assert torch.equal(old,state[key]),key
        if key in ('transformer_decoder.fc.weight','transformer_decoder.fc.bias'):
            assert torch.equal(old[63:],state[key][63:]),key
    model.load_state_dict(ref['model_state_dict']);model.apply_checkpoint_contract(ref);model.eval().requires_grad_(False)
    batches=[(b[0].transpose(1,2),b[1],b[2]) for b in raw_batches]
    out=directory/'cpu-variance-control';out.mkdir(exist_ok=True)
    from model.losses import mixture_expectation
    vocab=list(json.loads((root/'diffink/iam_overfit/chars.json').read_text()))
    metrics={k:[] for k in ('reference','final_current_std','final_reference_std')};post=[];mu_diffs=[];ocr=[]
    with torch.no_grad(),torch.random.fork_rng():
        for j,(sample,batch) in enumerate(zip(samples,batches)):
            raw,mask,labels=batch;model.load_state_dict(ref['model_state_dict'])
            truth,oldmu,oldlv,lm=encoded(model,raw,mask);oldstd=(.5*oldlv).exp()
            states=raw[0,2:,mask[0]].argmax(0).numpy();target=truth[0,mask[0]].numpy()
            torch.manual_seed(1042+j*100);noises=[torch.randn_like(oldmu) for _ in range(20)]
            sid=cfg['sample_ids'][j]
            for k,epsilon in enumerate(noises):
                output=model.decode(oldmu+epsilon*oldstd,padding_mask=~mask)
                xy=mixture_expectation(output)[0,mask[0]].numpy();pens=output[0,:3,mask[0]].argmax(0).numpy()
                folder=out/'reference'/sid;folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/f'z-{k}.npy',np.column_stack([xy,np.eye(3)[pens]]))
                metrics['reference'].append(dict(sample_id=sid,draw=k,geometry=geometry_metrics(xy,target,states),pen=boundary_metrics(pens,states)))
            model.load_state_dict(state);_,mu,lv,lm=encoded(model,raw,mask);std=(.5*lv).exp()
            output=model.decode(mu,padding_mask=~mask);xy=mixture_expectation(output)[0,mask[0]].numpy()
            gpu=np.load(directory/f'ocr/step-{entry["selected_step"]}/{sid}/mu.npy')
            assert np.allclose(xy,gpu[:,:2],rtol=1e-5,atol=3e-6),sid
            assert np.array_equal(output[0,:3,mask[0]].argmax(0).numpy(),gpu[:,2:].argmax(1)),sid
            mu_diffs.append(dict(sample_id=sid,max_abs_cpu_gpu_mu_difference=float(np.abs(xy-gpu[:,:2]).max())))
            post.append(dict(sample_id=sid,reference_std_mean=float(oldstd.mean()),final_std_mean=float(std.mean())))
            for mode,sigma in [('final_current_std',std),('final_reference_std',oldstd)]:
                for k,epsilon in enumerate(noises):
                    z=mu+epsilon*sigma;output=model.decode(z,padding_mask=~mask)
                    xy=mixture_expectation(output)[0,mask[0]].numpy();pens=output[0,:3,mask[0]].argmax(0).numpy()
                    folder=out/mode/sid;folder.mkdir(parents=True,exist_ok=True)
                    np.save(folder/f'z-{k}.npy',np.column_stack([xy,np.eye(3)[pens]]))
                    metrics[mode].append(dict(sample_id=sid,draw=k,geometry=geometry_metrics(xy,target,states),pen=boundary_metrics(pens,states)))
                    decoded=greedy_ctc(model.ocr_model(z)[:int(lm.sum()),0].argmax(-1).tolist(),vocab)
                    ocr.append(dict(mode=mode,sample_id=sid,draw=k,decoded=decoded,truth=sample[2],errors=edit_distance(sample[2],decoded),characters=len(sample[2])))
    aggregates={k:aggregate(v) for k,v in metrics.items()}
    for mode in ('final_current_std','final_reference_std'):
        variants=[r for r in ocr if r['mode']==mode]
        aggregates[mode]['cer']=sum(r['errors'] for r in variants)/sum(r['characters'] for r in variants)
    result=dict(selected_sha256=expected,source_sha256=SOURCE_SHA,joint_ocr_tensors_changed=changed_ocr,stage_chain_checks=chain,finite_checkpoint=True,style_weights_unchanged=True,
                sigma_rho_output_rows_unchanged=True,posterior_head_changed=any(not torch.equal(v,state[k]) for k,v in ref['model_state_dict'].items() if k.startswith('conv_logvar.')),
                mean_cpu_gpu_checks=mu_diffs,paired_noise_seed='CPU1042+100*line,20 identical epsilons across all3 conditions',
                original_variance_control='final mean/decoder with reference per-element std; index-aligned latent coordinates; not an exact reparameterization of original latent distribution',
                cpu_samples_not_bitwise_same_as_gpu_draws=True,aggregates=aggregates,posterior=post,
                pen={k:dict(min_f1=min(r['pen']['pen_up_f1'] for r in v),all_final_eoc_correct=all(r['pen']['final_eoc_correct'] for r in v),false_eoc=sum(r['pen']['non_final_false_eoc_count'] for r in v)) for k,v in metrics.items()},
                metrics=metrics,ocr=ocr)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return aggregates


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--repo',default='third_party/DiffInk' if Path('third_party/DiffInk').is_dir() else '.');p.add_argument('--data-root',default='data');a=p.parse_args()
    print(json.dumps(verify(a.directory,a.repo,a.data_root),indent=2))
