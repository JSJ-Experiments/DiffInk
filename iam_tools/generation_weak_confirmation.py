"""Fresh weak-alignment composition gate; excludes all THREE opened sets."""
import math
import hashlib
from .generation_position_confirmation import reserve_synthetic

def verify_source_points(points, record):
    if hashlib.sha256(points.tobytes()).hexdigest()!=record['points_sha256'] or len(points)!=record['points']:
        raise ValueError('confirmation source fingerprint drift')



def reserve(root,cfg,data,reserved_utc=None):
    """Metadata ONLY, excluding1032history and all THREE opened confirmations."""
    import datetime,json
    from pathlib import Path
    from .generation_composition import seal_confirmation
    from .generation_capacity import DATA
    from .generation_timing_study import PARENT
    from .ocr_pool_study import load_pool
    from .ocr_convergence import POOL_SHA
    from .pen_ab import file_sha
    root=Path(root);prior=json.loads((root/DATA/'dataset.json').read_text())
    opened=json.loads((root/PARENT/'dataset.json').read_text())
    history=dict(prior['records']);history.update(opened['confirmation_seal']['records']);history.update({f'earliest-synthetic:{k}':v for k,v in opened['synthetic_seal']['records'].items()})
    blocked=list(data['splits']['unseen_prompt'])+opened['confirmation_seal']['ids']
    sources={PARENT+'/dataset.json':file_sha(root/PARENT/'dataset.json')}
    for relative in ['checkpoints/iam_generation_position_contract/20261008-101747/reserved-confirmation/seal.json',
                     'checkpoints/iam_generation_prefix_contract/20261008-113817/reserved-confirmation/seal.json']:
        old=json.loads((root/relative).read_text());sources[relative]=file_sha(root/relative)
        history.update(old['paired']['records']);history.update({f'{relative}:synthetic:{k}':v for k,v in old['synthetic']['records'].items()});blocked+=old['paired']['ids']
    _,manifest,vocab=load_pool(root,POOL_SHA)
    if vocab!=cfg['vocab']:raise ValueError('pinned clean alphabet required')
    train=data['training_ids']['control']
    if data['training_ids']['weak_alignment']!=train:raise ValueError('identical actual TRAIN required')
    paired=seal_confirmation(manifest,history,train,cfg['writers'],seed=46144,development_ids=blocked)
    synthetic=reserve_synthetic(history,train,cfg['writers'],seed=46145)
    return dict(reserved_utc=reserved_utc or datetime.datetime.now(datetime.timezone.utc).isoformat(),paired=paired,synthetic=synthetic,
        source_history_sha256=sources,pool_manifest_sha256=POOL_SHA,prior_generation_dataset_sha256=file_sha(root/DATA/'dataset.json'),
        exposed_paired_ids=blocked,history_record_count=len(history),
        policy='Metadata-only seeds46144/46145; exclude all1032historical generation IDs/normalized texts AND all THREE opened paired/synthetic sets; paired forms excluded from current TRAIN/exposeddev/previous48paired. Known writers/corpus-familiar reader, not all historically untouched forms. Open once ONLY after both complete/frozen TRAIN-selected models satisfy5% native CER; no tuning on confirmation.')


def resolved_seal(directory,cfg,data,root):
    """Keep original as-run reservation immutable; accept ONLY a proved repair.

    Old synthetic sets reused IDs, so dictionary updates erased historical text.
    Correcting this metadata-only leak before opening any sources/outputs does
    not change training, RNG, targets, selection or optimization. Never silently
    overwrite/reselect an opened gate.
    """
    import json
    from pathlib import Path
    from .pen_ab import file_sha
    p=Path(directory);original=p/'reserved-confirmation/seal.json'
    if file_sha(original)!=cfg['reserved_confirmation_sha256']:raise ValueError('original immutable reservation guard')
    repair=p/'reserved-confirmation-corrected/correction.json'
    if repair.exists():
        note=json.loads(repair.read_text());seal_path=repair.parent/'seal.json'
        if note['original_seal_sha256']!=file_sha(original) or note['corrected_seal_sha256']!=file_sha(seal_path) or not note['no_confirmation_sources_or_generated_outputs_opened'] or note['training_protocol_changed']:
            raise ValueError('metadata-only seal repair provenance required')
    else:seal_path=original
    seal=json.loads(seal_path.read_text())
    if seal!=reserve(root,cfg,data,reserved_utc=seal['reserved_utc']):raise ValueError('full three-set exposure union must reproduce corrected reservation')
    return seal_path,seal

def fit_gate(cfg, results):
    """Do not spend blind confirmation on underfit or still-changing candidates."""
    from .generation_weak_alignment_study import ARMS
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
    from .generation_weak_alignment_study import ARMS
    from .generation_prefix_contract import PrefixContractWriter
    from .generation_cache import CachedLatentPool
    from .generation_coverage_study import evaluate
    from .generation_duration_eval import evaluate_duration
    from .generation_prefix_budget import evaluate_generous
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
    seal_path,seal=resolved_seal(p,cfg,data,root)
    cutoff=datetime.datetime.strptime(p.name,'%Y%m%d-%H%M%S').replace(tzinfo=datetime.timezone.utc)
    if datetime.datetime.fromisoformat(seal['reserved_utc'])<cutoff:raise ValueError('reservation must belong to current study')
    # Reproduce metadata-only reservation, including ALL three opened sets.
    if seal!=reserve(root,cfg,data,reserved_utc=seal['reserved_utc']):
        raise ValueError('independently reproduced fresh pre-training seal required')
    pool,manifest,vocab=load_pool(root,seal['pool_manifest_sha256'])
    ids=seal['paired']['ids'];records=seal['paired']['records'];synth=seal['synthetic']['records']
    if len(ids)!=16 or len(synth)!=16 or vocab!=cfg['vocab'] or any(records[i]!=manifest['records'][i] for i in ids):
        raise ValueError('sixteen new paired/synthetic covered prompts, pinned source metadata required')
    torch.set_num_threads(2)
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict());latents={};targets={};preflight=[]
    out.mkdir();(out/'confirmation-source.py').write_bytes(Path(__file__).read_bytes());(out/'budget-evaluator-source.py').write_bytes(Path(__file__).with_name('generation_prefix_budget.py').read_bytes())
    with torch.no_grad(),h5py.File(pool/'lines.h5') as f,h5py.File(out/'source.h5','w') as dest:
        from model.losses import mixture_expectation
        for sid in ids:
            points=f[sid]['point_seq'][:];r=records[sid]
            verify_source_points(points,r)
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
            model=PrefixContractWriter(**cfg['models'][a]);model.load_state_dict(saved['model_state_dict']);model.eval();step=saved['step']
            paired=evaluate(model,codec,reader,cache,latents,records,vocab,stats,dict(confirmation=ids),targets,folder,step,cfg['writers'],controls=True)
            estimated=evaluate_duration(model,codec,reader,records,vocab,stats,ids,folder,step,cfg['writers'],cfg['duration_model'],controls=True)
            sf=folder/'synthetic';sf.mkdir();synthetic=evaluate_duration(model,codec,reader,synth,vocab,stats,list(synth),sf,step,cfg['writers'],cfg['duration_model'],controls=True)
            generous=evaluate_generous(model,codec,reader,records,vocab,stats,ids,folder,step,cfg['writers'])
            synthetic_generous=evaluate_generous(model,codec,reader,synth,vocab,stats,list(synth),sf,step,cfg['writers'])
            outputs[a]=dict(generous=generous['aggregate'],synthetic_generous=synthetic_generous['aggregate'],selected_step=step,checkpoint_sha256=results[a]['selected_sha256'],oracle=paired['aggregate'],estimated=estimated['aggregate'],synthetic=synthetic['aggregate'],files={str(q.relative_to(out)):file_sha(q) for q in folder.rglob('*') if q.is_file()})
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen evaluator drift')
    summary=dict(arms=outputs,reservation_sha256=file_sha(seal_path),reservation=seal,opened_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_h5_sha256=file_sha(out/'source.h5'),preflight=preflight,
        policy='One-shot reserved NEW prompts; TRAIN-selected checkpoint, native fit gate5%, no reader-failure exclusions, no subsequent checkpoint tuning. Synthetic TRAIN-predicted duration AND constant256 budget, no fictional reference/oracle.',limitations='Known writers and corpus-familiar reader, new forms to current TRAIN/ALL THREE earlier exposed paired16 sets, not necessarily all older1024runs; only16paired/16synthetic, one seed.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
