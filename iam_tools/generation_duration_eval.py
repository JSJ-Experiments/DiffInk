"""Target-free generator inputs and estimated-duration readout diagnostics."""
import json
from pathlib import Path
import h5py,numpy as np,torch
from .generation_composition import predict_duration
from .generation_study import decode_sample
from .generation_coverage_study import writer_tensor
from .latent_diffusion import transform
from .pen_ab import file_sha


def inputs(texts,lengths,vocab,device):
    if not texts or len(texts)!=len(lengths) or any(not t or not 1<=n<=256 for t,n in zip(texts,lengths)):raise ValueError('bounded nonempty requested text/duration required')
    x=torch.zeros(len(texts),max(lengths),384,device=device);mask=torch.arange(max(lengths),device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    labels=torch.full((len(texts),max(map(len,texts))),-1,device=device,dtype=torch.long)
    for j,t in enumerate(texts):labels[j,:len(t)]=torch.tensor([vocab.index(c) for c in t],device=device)
    return x,mask,labels


@torch.no_grad()
def evaluate_duration(model,codec,reader,records,vocab,stats,ids,folder,step,writers,duration,controls=False):
    was=model.training;model.eval();device=next(model.parameters()).device;rows=[];file=Path(folder)/f'duration-evaluation-{step}.h5'
    policies=['estimated_correct','estimated_swapped_fixed_length','estimated_null_fixed_length'] if controls else ['estimated_correct']
    with h5py.File(file,'w') as f:
        for policy in policies:
            for start in range(0,len(ids),8):
                batch=ids[start:start+8];lengths=[predict_duration(duration,records[i]['text'],records[i]['writer_id']) for i in batch]
                texts=[records[ids[(ids.index(i)+1)%len(ids)]]['text'] if policy=='estimated_swapped_fixed_length' else records[i]['text'] for i in batch]
                x,mask,labels=inputs(texts,lengths,vocab,device);wi=writer_tensor(batch,records,writers,device)
                drop=torch.full((len(batch),),policy=='estimated_null_fixed_length',device=device,dtype=torch.bool)
                pred=model(x,torch.ones(len(batch),device=device),labels,mask,drop_text=drop,writer_ids=wi);z=transform(pred,stats,True)
                for j,sid in enumerate(batch):
                    score_record=records[sid] if 'points' in records[sid] else dict(records[sid],points=8*lengths[j])
                    points,metrics=decode_sample(codec,reader,z[j,:lengths[j]],score_record,vocab)
                    actual=(records[sid]['points']+7)//8 if 'points' in records[sid] else None
                    if actual is None:
                        for key in ['oracle_window_decoded','window_errors','oracle_points','internal_eoc_count']:metrics.pop(key,None)
                    row=dict(sample_id=sid,policy=policy,text=records[sid]['text'],conditioning_text='' if bool(drop[j]) else texts[j],writer_id=records[sid]['writer_id'],predicted_blocks=lengths[j],actual_blocks_for_diagnostic_only=actual,length_relative_error=abs(lengths[j]-actual)/actual if actual else None,no_paired_target=actual is None,**metrics)
                    rows.append(row);g=f.create_group(policy+'/'+sid);g.create_dataset('points',data=points,compression='gzip');g.create_dataset('latent',data=z[j,:lengths[j]].cpu().numpy(),compression='gzip');g.attrs['row']=json.dumps(row)
    groups={}
    for policy in policies:
        q=[r for r in rows if r['policy']==policy];errors=[r['length_relative_error'] for r in q if r['length_relative_error'] is not None]
        groups[policy]=dict(evaluations=len(q),free_cer=sum(r['free_errors'] for r in q)/sum(r['characters'] for r in q),free_exact=sum(r['free_errors']==0 for r in q),mean_generated_stop=float(np.mean([r['generated_points_at_stop'] for r in q])),missing_eoc=sum(r['first_eoc_point'] is None for r in q),mean_absolute_relative_length_error=float(np.mean(errors)) if errors else None)
    result=dict(step=step,ids=ids,lines=rows,aggregate=groups,packed_h5_sha256=file_sha(file),definition='mask from TRAIN-only duration predictor on ORIGINAL requested text + writer; swapped/NULL retain SAME estimated duration to isolate conditioning; target points only enter evaluation diagnostics, NOT generation; no forced EOC; no aligned RMSE for unequal index lengths')
    (Path(folder)/f'duration-eval-{step}.json').write_text(json.dumps(result,indent=2)+'\n');model.train(was)
    print(dict(step=step,duration_folder=str(folder),estimated_cer=groups['estimated_correct']['free_cer']),flush=True);return result
