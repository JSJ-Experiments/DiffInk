"""One-shot post-training paired and unpaired composition confirmation."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np,torch
from .generation_alignment import AlignedWriterDenoiser
from .generation_cache import CachedLatentPool
from .generation_coverage_study import evaluate
from .generation_duration_eval import evaluate_duration
from .generation_study import read_sequence
from .latent_integration import encoded
from .ocr_pool_study import single_batch
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha
from .inkvae import edit_distance
from .generation_alignment_study import ARMS


@torch.no_grad()
def confirm(directory,cfg,data,results,pool_path,codec,reader,stats):
    """No training or selection here; checkpoint decisions are already frozen."""
    p=Path(directory);out=p/'confirmation'
    if out.exists():raise ValueError('refuse overwrite confirmation; do not tune/reselect on it')
    if any(r['stop']!='budget_completed' or r['last_step']!=cfg['max_updates'] for r in results.values()):raise ValueError('both budgets must finish before unsealing')
    out.mkdir();(out/'confirmation-source.py').write_bytes(Path(__file__).read_bytes())
    records=data['confirmation_seal']['records'];ids=data['confirmation_seal']['ids'];vocab=cfg['vocab'];latents={};targets={};preflight=[]
    with h5py.File(Path(pool_path)/'lines.h5') as f,h5py.File(out/'source.h5','w') as dest:
        from model.losses import mixture_expectation
        for sid in ids:
            points=f[sid]['point_seq'][:];r=records[sid]
            if hashlib.sha256(points.tobytes()).hexdigest()!=r['points_sha256'] or len(points)!=r['points']:raise ValueError('sealed source point fingerprint drift')
            raw,pm,_=single_batch(points,r['text'],vocab);target,mu,_,_=encoded(codec,raw,pm);n=len(points)
            truth=torch.cat((target[0,:n],raw[0,2:,:n].T),1).numpy();z=mu[0].T.detach()
            decoded=codec.decode(mu,padding_mask=~pm);xy=mixture_expectation(decoded)[0,:n];pens=decoded[0,:3,:n].argmax(0)
            decoded_text=read_sequence(reader,torch.cat((xy,torch.nn.functional.one_hot(pens,3).float()),1),vocab)
            err=float((xy-target[0,:n]).square().mean().sqrt());pen_errors=int((pens!=raw[0,2:,:n].argmax(0)).sum())
            if err>.0005 or pen_errors:raise ValueError('sealed codec transport failed')
            # Reader failures stay in the confirmation denominator. Never discard.
            preflight.append(dict(sample_id=sid,codec_rmse=err,pen_errors=pen_errors,reader_text=decoded_text,reader_errors=edit_distance(r['text'],decoded_text),characters=len(r['text'])))
            latents[sid]=z;targets[sid]=truth;g=dest.create_group(sid);g.create_dataset('target',data=truth,compression='gzip');g.create_dataset('latent_mean',data=z.numpy(),compression='gzip')
    (out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    cache=CachedLatentPool(latents,records,vocab,ids,stats,device='cpu',max_lines=16);synth=data['synthetic_seal']['records'];synthetic_ids=list(synth);outputs={}
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict())
    for a in ARMS:
        folder=out/a;folder.mkdir();saved=torch.load(p/a/'checkpoint-best.pt',map_location='cpu',weights_only=False)
        if saved['step']!=results[a]['best_step'] or file_sha(p/a/'checkpoint-best.pt')!=results[a]['selected_sha256']:raise ValueError('frozen TRAIN-selected checkpoint required')
        model=AlignedWriterDenoiser(**cfg['models'][a]);model.load_state_dict(saved['model_state_dict']);model.eval();step=saved['step']
        paired=evaluate(model,codec,reader,cache,latents,records,vocab,stats,dict(confirmation=ids),targets,folder,step,cfg['writers'],controls=True)
        predicted=evaluate_duration(model,codec,reader,records,vocab,stats,ids,folder,step,cfg['writers'],cfg['duration_model'],controls=True)
        tf=folder/'train-duration';tf.mkdir();train_duration=evaluate_duration(model,codec,reader,data['records'],vocab,stats,data['training_ids'][a],tf,step,cfg['writers'],cfg['duration_model'],controls=False)
        sf=folder/'synthetic';sf.mkdir();synthetic=evaluate_duration(model,codec,reader,synth,vocab,stats,synthetic_ids,sf,step,cfg['writers'],cfg['duration_model'],controls=True)
        outputs[a]=dict(selected_step=step,checkpoint_sha256=results[a]['selected_sha256'],oracle=paired['aggregate'],estimated=predicted['aggregate'],train_estimated=train_duration['aggregate'],synthetic=synthetic['aggregate'],files={str(q.relative_to(out)):file_sha(q) for q in folder.rglob('*') if q.is_file()})
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen evaluator drift')
    summary=dict(arms=outputs,preflight=dict(lines=len(ids),reader_exact=sum(r['reader_errors']==0 for r in preflight),reader_cer=sum(r['reader_errors'] for r in preflight)/sum(r['characters'] for r in preflight),no_reader_failure_exclusions=True),source_h5_sha256=file_sha(out/'source.h5'),selection='TRAIN-only checkpoint best, no confirmation tuning; steps may differ. Matched-final tables remain separate.',paired_seal=data['confirmation_seal'],synthetic_seal=data['synthetic_seal'],limitations='known writers; corpus-familiar reader; paired forms new to current fresh TRAIN/development, NOT every historical experiment; synthetic unpaired: NO oracle duration, reference geometry or aligned RMSE')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
