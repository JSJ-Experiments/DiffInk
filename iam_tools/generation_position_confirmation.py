"""New metadata-only composition seal; never reuse the opened synthetic prompts."""
import hashlib
import itertools
import math
from .ocr_pool import normalized_text


def reserve_synthetic(history_records, train_ids, writers, seed=26145):
    """Fixed grammatical recombinations; filtering ONLY duplicate text/characters."""
    known={normalized_text(r['text']) for r in history_records.values()}
    chars={c for i in train_ids for c in history_records[i]['text']}
    candidates=[f'{subject} {verb} {obj} {place}.' for subject,verb,obj,place in itertools.product(
        ['I','We','She','They'], ['left','found','kept','moved'],
        ['the blue notebook','a small letter','the old book','a cup of tea'],
        ['on the desk','by the window','near the door','in the office'])]
    safe=[t for t in candidates if normalized_text(t) not in known and set(t)<=chars]
    prompts=sorted(safe,key=lambda t:hashlib.sha256(f'{seed}:{t}'.encode()).hexdigest())[:16]
    selected=sorted(writers,key=lambda w:hashlib.sha256(f'{seed}:{w}'.encode()).hexdigest())[:8]
    if len(prompts)!=16 or len(set(prompts))!=16 or len(selected)!=8 or len(set(selected))!=8:
        raise ValueError('sixteen novel covered prompts and eight known writers required')
    return dict(records={f'contract-synthetic-{j+1:02d}':dict(text=t,writer_id=selected[j//2]) for j,t in enumerate(prompts)},seed=seed,
        selection='hash-ordered16 from fixed256 grammatical recombinations, exclude every historical/exposed normalized transcript, TRAIN character coverage only,8known writers',
        policy='Never use fake paired points/oracle duration. Open once after frozen TRAIN selection and native fit gate, no confirmation tuning.')


def fit_gate(cfg, results):
    """Do not spend blind confirmation on underfit or still-changing candidates."""
    from .generation_position_contract_study import ARMS
    if set(results)!=set(ARMS) or cfg['max_updates']!=48000:
        raise ValueError('matched bounded final positional candidates required')
    for r in results.values():
        if r['stop']!='budget_completed' or r['last_step']!=48000:
            raise ValueError('both candidates must finish before opening confirmation')
        selected=[h for h in r['history'] if h['step']==r['best_step']]
        cer=selected[0]['aggregate']['all_train256']['correct']['free_cer'] if len(selected)==1 else float('nan')
        if not math.isfinite(cer) or not 0<=cer<=.05:
            raise ValueError('keep confirmation reserved: selected native TRAIN fit gate >5% CER')


def confirm(directory, repo, root='data'):
    """CPU-only one-shot new-text gate after both TRAIN-selected models are frozen."""
    import datetime,json
    from pathlib import Path
    import h5py,numpy as np,torch
    from .generation_position_contract_study import ARMS
    from .generation_position_contract import PositionContractWriter
    from .generation_cache import CachedLatentPool
    from .generation_coverage_study import evaluate
    from .generation_duration_eval import evaluate_duration
    from .generation_study import read_sequence
    from .generation_capacity import DATA
    from .generation_timing_study import PARENT
    from .generation_composition import seal_confirmation
    from .latent_integration import encoded
    from .ocr_pool_study import load_pool,single_batch
    from .ocr_context_study import tensor_digest
    from .ocr_joint_adapter import load_reader
    from .writer_expansion import load
    from .pen_ab import file_sha
    from .inkvae import edit_distance
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text())
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS};fit_gate(cfg,results)
    out=p/'confirmation'
    if out.exists():raise ValueError('refuse overwrite/reselection/tuning on opened confirmation')
    seal_path=p/'reserved-confirmation/seal.json';seal=json.loads(seal_path.read_text())
    cutoff=datetime.datetime.strptime(p.name,'%Y%m%d-%H%M%S').replace(tzinfo=datetime.timezone.utc)
    if datetime.datetime.fromisoformat(seal['reserved_utc'])<cutoff:raise ValueError('reservation must belong to current study')
    # Reservation happened during fresh run, before final freezing; after exposure no updates allowed.
    prior=json.loads((root/DATA/'dataset.json').read_text());train=data['training_ids']['relative100']
    opened=json.loads((root/PARENT/'dataset.json').read_text())
    if file_sha(root/PARENT/'dataset.json')!=seal['previous_dataset_sha256'] or seal['previous_opened_ids']!=opened['confirmation_seal']['ids']:
        raise ValueError('previous exposure provenance drift')
    history=dict(prior['records']);history.update(opened['confirmation_seal']['records']);history.update(opened['synthetic_seal']['records'])
    blocked=data['splits']['unseen_prompt']+seal['previous_opened_ids']
    pool,manifest,vocab=load_pool(root,seal['pool_manifest_sha256'])
    if seal['paired']!=seal_confirmation(manifest,history,train,cfg['writers'],seed=26144,development_ids=blocked) or seal['synthetic']!=reserve_synthetic(history,train,cfg['writers']):
        raise ValueError('independently reproduced metadata-only fresh seal required')
    known=set(prior['records'])|set(seal['previous_opened_ids'])
    ids=seal['paired']['ids'];records=seal['paired']['records'];synth=seal['synthetic']['records']
    if len(ids)!=16 or len(synth)!=16 or set(ids)&known:raise ValueError('sixteen NEW paired and synthetic prompts required')
    if vocab!=cfg['vocab'] or any(records[sid]!=manifest['records'][sid] for sid in ids):raise ValueError('confirmation source metadata drift')
    forbidden_forms={manifest['records'][i]['prompt_family'] for i in train+data['splits']['unseen_prompt']+seal['previous_opened_ids']}
    if any(records[i]['prompt_family'] in forbidden_forms for i in ids):raise ValueError('current/exposed paired form leakage')
    torch.set_num_threads(2)
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict());latents={};targets={};preflight=[]
    out.mkdir();(out/'confirmation-source.py').write_bytes(Path(__file__).read_bytes())
    with torch.no_grad(),h5py.File(pool/'lines.h5') as f,h5py.File(out/'source.h5','w') as dest:
        from model.losses import mixture_expectation
        for sid in ids:
            points=f[sid]['point_seq'][:];r=records[sid]
            if hashlib.sha256(points.tobytes()).hexdigest()!=r['points_sha256'] or len(points)!=r['points']:raise ValueError('confirmation source fingerprint drift')
            raw,pm,_=single_batch(points,r['text'],vocab);target,mu,_,_=encoded(codec,raw,pm);n=len(points)
            truth=torch.cat((target[0,:n],raw[0,2:,:n].T),1).numpy();z=mu[0].T.detach();decoded=codec.decode(mu,padding_mask=~pm)
            xy=mixture_expectation(decoded)[0,:n];pens=decoded[0,:3,:n].argmax(0);text=read_sequence(reader,torch.cat((xy,torch.nn.functional.one_hot(pens,3).float()),1),vocab)
            err=float((xy-target[0,:n]).square().mean().sqrt());pen_errors=int((pens!=raw[0,2:,:n].argmax(0)).sum())
            if err>.0005 or pen_errors:raise ValueError('new-source codec preflight failure')
            preflight.append(dict(sample_id=sid,codec_rmse=err,pen_errors=pen_errors,reader_text=text,reader_errors=edit_distance(r['text'],text),characters=len(r['text'])))
            latents[sid]=z;targets[sid]=truth;g=dest.create_group(sid);g.create_dataset('target',data=truth,compression='gzip');g.create_dataset('latent_mean',data=z.numpy(),compression='gzip')
        cache=CachedLatentPool(latents,records,vocab,ids,stats,device='cpu',max_lines=16);outputs={}
        for a in ARMS:
            folder=out/a;folder.mkdir();saved=torch.load(p/a/'checkpoint-best.pt',map_location='cpu',weights_only=False)
            if saved['step']!=results[a]['best_step'] or file_sha(p/a/'checkpoint-best.pt')!=results[a]['selected_sha256']:raise ValueError('frozen TRAIN-selected checkpoint guard')
            model=PositionContractWriter(**cfg['models'][a]);model.load_state_dict(saved['model_state_dict']);model.eval();step=saved['step']
            paired=evaluate(model,codec,reader,cache,latents,records,vocab,stats,dict(confirmation=ids),targets,folder,step,cfg['writers'],controls=True)
            estimated=evaluate_duration(model,codec,reader,records,vocab,stats,ids,folder,step,cfg['writers'],cfg['duration_model'],controls=True)
            sf=folder/'synthetic';sf.mkdir();synthetic=evaluate_duration(model,codec,reader,synth,vocab,stats,list(synth),sf,step,cfg['writers'],cfg['duration_model'],controls=True)
            outputs[a]=dict(selected_step=step,checkpoint_sha256=results[a]['selected_sha256'],oracle=paired['aggregate'],estimated=estimated['aggregate'],synthetic=synthetic['aggregate'],files={str(q.relative_to(out)):file_sha(q) for q in folder.rglob('*') if q.is_file()})
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen evaluator drift')
    summary=dict(arms=outputs,reservation_sha256=file_sha(seal_path),reservation=seal,opened_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_h5_sha256=file_sha(out/'source.h5'),preflight=preflight,
        policy='One-shot reserved NEW prompts; TRAIN-selected checkpoint, native fit gate5%, no reader-failure exclusions, no subsequent checkpoint tuning. Synthetic only TRAIN-predicted duration, no fictional reference/oracle.',limitations='Known writers and corpus-familiar reader, new forms to current TRAIN/exposed16, not necessarily all older1024runs; only16paired/16synthetic, one seed.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
