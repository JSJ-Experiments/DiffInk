"""Frozen output-budget audit: native timing is a diagnostic, not generator input.

The generous budget is the constant256 blocks (2048 points) for EVERY prompt.
No target point count, length predictor, trajectory, or forced EOC goes into that
inference. Native/±1/estimated policies distinguish prefix deformation from
truncation and learned stopping. No training or checkpoint reselection.
"""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .generation_prefix_contract import PrefixContractWriter
from .generation_duration_eval import inputs
from .generation_coverage_study import writer_tensor
from .generation_composition import predict_duration
from .generation_capacity import DATA
from .generation_study import decode_sample
from .latent_diffusion import transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .pen_ab import file_sha

POLICIES=('native','shorter_one','longer_one','estimated','generous256')


def budget(policy, record, duration):
    if policy=='generous256':return 256
    if policy=='estimated':return predict_duration(duration,record['text'],record['writer_id'])
    n=(record['points']+7)//8
    if policy=='native':return n
    if policy=='shorter_one':return max(1,n-1)
    if policy=='longer_one':return min(256,n+1)
    raise ValueError('explicit frozen budget policy required')


def aggregate(rows):
    if not rows:raise ValueError('nonempty frozen audit required')
    return dict(lines=len(rows),cer=sum(r['free_errors'] for r in rows)/sum(r['characters'] for r in rows),
        exact=sum(r['free_errors']==0 for r in rows),missing_eoc=sum(r['first_eoc_point'] is None for r in rows),
        max_latent_prefix_difference=max(r['latent_prefix_max_abs'] for r in rows),
        max_xy_prefix_difference=max(r['xy_prefix_max_abs'] for r in rows),
        mean_x_prefix_rmse=float(np.mean([r['x_prefix_rmse'] for r in rows])),
        mean_y_prefix_rmse=float(np.mean([r['y_prefix_rmse'] for r in rows])),
        pen_changes=sum(r['pen_changes'] for r in rows),reader_changes=sum(not r['reader_equal_to_native'] for r in rows))


@torch.no_grad()
def audit(directory, repo, root='data', device='cpu'):
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):raise ValueError('available explicit CPU/CUDA evaluation device required')
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text())
    out=p/'budget-audit';out.mkdir(exist_ok=False);(out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec=codec.to(device).eval().requires_grad_(False);reader,_=load_reader(root,cc,device);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    stats={k:v.to(device) if torch.is_tensor(v) else v for k,v in stats.items()}
    ids=sorted(data['records']);results={};allrows=[]
    for arm in ['noncausal','causal']:
        result=json.loads((p/arm/'result.json').read_text());checkpoint=p/arm/'checkpoint-best.pt'
        if file_sha(checkpoint)!=result['selected_sha256']:raise ValueError('immutable TRAIN-selected checkpoint guard')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=PrefixContractWriter(**cfg['models'][arm]).to(device).eval();model.load_state_dict(saved['model_state_dict']);references={};rows=[]
        with h5py.File(out/(arm+'.h5'),'w') as f:
            for policy in POLICIES:
                for start in range(0,len(ids),8):
                    batch=ids[start:start+8];records=[data['records'][sid] for sid in batch]
                    lengths=[budget(policy,r,cfg['duration_model']) for r in records]
                    x,mask,text=inputs([r['text'] for r in records],lengths,cfg['vocab'],device)
                    wi=writer_tensor(batch,data['records'],cfg['writers'],device)
                    pred=model(x,torch.ones(len(batch),device=device),text,mask,writer_ids=wi);z=transform(pred,stats,True)
                    for j,sid in enumerate(batch):
                        latent=z[j,:lengths[j]];points,m=decode_sample(codec,reader,latent,records[j],cfg['vocab'])
                        if policy=='native':references[sid]=(latent.clone(),points.copy(),m['free_decoded'])
                        ref,xy,decoded=references[sid];blocks=min(len(ref),len(latent));n=min(records[j]['points'],len(points),len(xy))
                        diff=points[:n,:2]-xy[:n,:2]
                        row=dict(arm=arm,policy=policy,sample_id=sid,blocks=lengths[j],text=records[j]['text'],**m,
                            latent_prefix_max_abs=float((latent[:blocks]-ref[:blocks]).abs().max()),
                            xy_prefix_max_abs=float(np.abs(diff).max()),x_prefix_rmse=float(np.sqrt((diff[:,0]**2).mean())),
                            y_prefix_rmse=float(np.sqrt((diff[:,1]**2).mean())),pen_changes=int((points[:n,2:].argmax(1)!=xy[:n,2:].argmax(1)).sum()),
                            reader_equal_to_native=m['free_decoded']==decoded)
                        rows.append(row);g=f.create_group(policy+'/'+sid);g.create_dataset('points',data=points,compression='gzip');g.create_dataset('latent',data=latent.cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
                print(dict(arm=arm,budget_policy=policy,all264=aggregate([r for r in rows if r['policy']==policy])),flush=True)
        groups={split:{policy:aggregate([r for r in rows if r['policy']==policy and r['sample_id'] in scope]) for policy in POLICIES}
                for split,scope in [('train256',data['splits']['all_train256']),('exposed_dev8',data['splits']['unseen_prompt'])]}
        results[arm]=dict(selected_step=saved['step'],checkpoint_sha256=result['selected_sha256'],groups=groups,
                             packed_h5_sha256=file_sha(out/(arm+'.h5')))
        allrows.extend(rows)
    summary=dict(arms=results,lines=allrows,policy='Frozen native-TRAIN-selected weights. All264 prompts evaluated; dev is already exposed, NOT a blind test. Constant256 output budget uses only text/writer and learned firstEOC stop; no target length/forced EOC. ±1 uses oracle solely as a timing intervention; shorter outputs may truncate. Latent and decoded prefix drift reported separately.',
        not_promoted=True,evaluation_device=device,codec_sha256=cfg['source_sha256'],reader_sha256=cfg['reader_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


@torch.no_grad()
def evaluate_generous(model,codec,reader,records,vocab,stats,ids,folder,step,writers,controls=True):
    """True target-free constant256 input for paired OR unpaired prompts.

    Actual paired point counts may enter AFTER generation as reader diagnostics;
    unpaired prompts never get fabricated reference/window metrics.
    """
    was=model.training;model.eval();folder=Path(folder);device=next(model.parameters()).device;rows=[]
    policies=['generous_correct','generous_swapped','generous_null'] if controls else ['generous_correct']
    path=folder/f'generous-evaluation-{step}.h5'
    with h5py.File(path,'w') as f:
        for policy in policies:
            for start in range(0,len(ids),8):
                batch=ids[start:start+8]
                texts=[records[ids[(ids.index(sid)+1)%len(ids)]]['text'] if policy=='generous_swapped' else records[sid]['text'] for sid in batch]
                x,mask,labels=inputs(texts,[256]*len(batch),vocab,device);wi=writer_tensor(batch,records,writers,device)
                drop=torch.full((len(batch),),policy=='generous_null',device=device,dtype=torch.bool)
                z=transform(model(x,torch.ones(len(batch),device=device),labels,mask,drop_text=drop,writer_ids=wi),stats,True)
                for j,sid in enumerate(batch):
                    r=records[sid];paired='points' in r
                    score_record=r if paired else dict(r,points=2048)  # cap for internal reader call, NOT invented reference
                    points,m=decode_sample(codec,reader,z[j],score_record,vocab)
                    if not paired:
                        for key in ['oracle_window_decoded','window_errors','oracle_points','internal_eoc_count']:m.pop(key,None)
                    row=dict(sample_id=sid,policy=policy,text=r['text'],conditioning_text='' if policy=='generous_null' else texts[j],writer_id=r['writer_id'],budget_blocks=256,no_paired_target=not paired,**m)
                    rows.append(row);q=f.create_group(policy+'/'+sid);q.create_dataset('points',data=points,compression='gzip');q.create_dataset('latent',data=z[j].cpu().numpy(),compression='gzip');q.attrs['row']=json.dumps(row)
    groups={policy:dict(evaluations=len(ids),free_cer=sum(r['free_errors'] for r in rows if r['policy']==policy)/sum(r['characters'] for r in rows if r['policy']==policy),free_exact=sum(r['free_errors']==0 for r in rows if r['policy']==policy),missing_eoc=sum(r['first_eoc_point'] is None for r in rows if r['policy']==policy)) for policy in policies}
    result=dict(step=step,lines=rows,aggregate=groups,packed_h5_sha256=file_sha(path),definition='EVERY input256blocks, original requested text/writer only; swapped/NULL same fixed budget; first learnedEOC stop; no length estimator/target length/forced EOC; no fake unpaired reference metrics')
    (folder/f'generous-eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');model.train(was)
    print(dict(step=step,generous_folder=str(folder),aggregate=groups),flush=True);return result
