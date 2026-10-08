"""Fail-closed paired positional-contract review: fitting != timing != composition."""
import hashlib
import html
import json
import tarfile
from pathlib import Path
import h5py
import numpy as np
import torch
from .generation_position_contract import PositionContractWriter, contract_lr
from .generation_position_contract_study import ARMS, verify_duration_refit
from .generation_capacity import DATA, SOURCE_H5_SHA, DATASET_SHA, WHITENING_SHA, select_capacity
from .generation_composition import fit_duration
from .generation_coverage_study import score, writer_tensor
from .generation_study import collate, decode_sample
from .latent_diffusion import transform
from .writer_expansion import load, training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha


def verify(directory, cfg, data, results):
    p=Path(directory)
    if set(results)!=set(ARMS) or set(data['training_ids'])!=set(ARMS):
        raise ValueError('exact relative100/absolute paired results required')
    if not cfg['fresh'] or cfg['parent_step']!=0 or cfg['neural_checkpoint_initialization'] or cfg['max_updates']!=48000:
        raise ValueError('fresh empty-Adam bounded48000 protocol required')
    if data['training_ids']['relative100']!=data['training_ids']['absolute']:
        raise ValueError('identical actual TRAIN scope required')
    specs={a:{k:v for k,v in cfg['models'][a].items() if k!='position_policy'} for a in ARMS}
    if specs['relative100']!=specs['absolute'] or not specs['absolute']['alignment'] or any(cfg['models'][a]['position_policy']!=a for a in ARMS):
        raise ValueError('only declared query positional contract changes')
    initial={};rng={};schedules={}
    for a in ARMS:
        r=results[a]
        if r['last_step']!=48000 or r['stop']!='budget_completed' or not r['codec_reader_unchanged']:
            raise ValueError('matched complete finite frozen budgets required')
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()]
        if [x['step'] for x in rows]!=list(range(1,48001)):
            raise ValueError('all actual updates required')
        expected=[batch for seed in cfg['schedule_seeds'] for batch in training_schedule(data['training_ids'][a],24000,cfg['batch'],seed=seed)]
        schedules[a]=[x['sample_ids'] for x in rows]
        if schedules[a]!=expected or hashlib.sha256(json.dumps(expected).encode()).hexdigest()!=r['schedule_sha256']:
            raise ValueError('reproducible TRAIN minibatch schedule required')
        if any(x['learning_rates']!=[contract_lr(x['step'])] or not np.isfinite(x['loss']) or not np.isfinite(x['gradient_norm']) for x in rows):
            raise ValueError('actual finite matched LR/optimization required')
        saved=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        if saved['step']!=0 or saved['optimizer_state_dict']['state']:
            raise ValueError('actual fresh step0 empty Adam required')
        for g in saved['optimizer_state_dict']['param_groups']:
            if g['lr']!=1e-5 or list(g['betas'])!=cfg['betas'] or g['weight_decay']!=cfg['weight_decay']:
                raise ValueError('actual initial optimizer groups drift')
        initial[a]=tensor_digest(saved['model_state_dict'])
        if initial[a]!=r['initial_state_sha256']:
            raise ValueError('initial weight digest drift')
        rng[a]={k:saved[k] for k in ['torch_rng_state','cuda_rng_state']}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['model_seed']);fresh=PositionContractWriter(**cfg['models'][a])
        if set(fresh.state_dict())!=set(saved['model_state_dict']) or any(not torch.allclose(v,saved['model_state_dict'][k],atol=2e-6,rtol=1e-6) for k,v in fresh.state_dict().items()):
            raise ValueError('independently regenerated fresh neural initialization required')
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(p/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
        if [h['step'] for h in r['history']]!=cfg['eval_steps'] or r['best_step']!=min(r['history'],key=lambda h:h['train_score'])['step']:
            raise ValueError('all256 native TRAIN-only checkpoint selection required')
        for h in r['history']:
            ev=json.loads((p/a/f'eval-{h["step"]}.json').read_text())
            ids=[row['sample_id'] for row in ev['lines'] if row['policy']=='correct' and row['sample_id'] in data['training_ids'][a]]
            if len(ids)!=256 or set(ids)!=set(data['training_ids'][a]) or abs(score(ev,set(ids))-h['train_score'])>1e-9 or file_sha(p/a/f'evaluation-{h["step"]}.h5')!=ev['packed_h5_sha256']:
                raise ValueError('full actual TRAIN metrics/output selection drift')
            de=json.loads((p/a/f'duration-eval-{h["step"]}.json').read_text())
            if de['ids']!=data['splits']['all_train256'] or file_sha(p/a/f'duration-evaluation-{h["step"]}.h5')!=de['packed_h5_sha256']:
                raise ValueError('full TRAIN predicted-duration output drift')
            dd=json.loads((p/a/'exposed-development-duration'/f'duration-eval-{h["step"]}.json').read_text())
            if dd['ids']!=data['splits']['unseen_prompt'] or file_sha(p/a/'exposed-development-duration'/f'duration-evaluation-{h["step"]}.h5')!=dd['packed_h5_sha256']:
                raise ValueError('exposed development predicted-duration output drift')
    if len(set(initial.values()))!=1 or schedules['relative100']!=schedules['absolute'] or any(not torch.equal(rng['relative100'][k],rng['absolute'][k]) for k in rng['relative100']):
        raise ValueError('bitwise paired fresh weights/RNG/minibatches required')
    return dict(actual_complete_budgets=True,independently_regenerated_fresh=True,bitwise_paired_initial_weights_rng=True,empty_adam=True,actual_matched_order_lrs=True,all256_train_selection=True,no_confirmation_used=True,
                cross_wheel_regeneration_tolerance='atol2e-6,rtol1e-6; actual paired weights/RNG equality remains bitwise')


def report(directory, repo, root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw, split_xy
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text())
    results={a:json.loads((p/a/'result.json').read_text()) for a in ARMS};guards=verify(p,cfg,data,results)
    for name,sha in [('source.h5',SOURCE_H5_SHA),('dataset.json',DATASET_SHA),('whitening.pt',WHITENING_SHA)]:
        if file_sha(root/DATA/name)!=sha:raise ValueError('immutable data source drift')
    original=json.loads((root/DATA/'dataset.json').read_text());scope=select_capacity(original)
    if data['splits']!=scope['splits'] or data['training_ids']['absolute']!=scope['training_ids']['larger256'] or any(r!=original['records'][sid] for sid,r in data['records'].items()):
        raise ValueError('reproducible original dataset guard')
    verify_duration_refit(fit_duration(data['records'],data['training_ids']['absolute'],cfg['writers']),cfg['duration_model'],data['records'],sorted(data['records']))
    if file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:
        raise ValueError('as-run source archive guard')
    with tarfile.open(p/'as-run-source.tar.gz') as tar:
        for name in ['iam_tools/generation_position_contract.py','iam_tools/generation_position_contract_study.py','iam_tools/generation_alignment.py','iam_tools/generation_geometry.py','iam_tools/generation_cache.py','iam_tools/generation_coverage_study.py','iam_tools/generation_duration_eval.py','model/vae.py','model/losses.py']:
            if tar.extractfile(name).read()!=(Path(repo)/name).read_bytes():raise ValueError('as-run implementation drift: '+name)
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);targets={};latents={}
    with h5py.File(root/DATA/'source.h5') as f:
        for sid in data['records']:targets[sid]=f[sid]['target'][:];latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    ids=['k04-309z-09','c03-109z-03','r07-568z-04','a02-130z-03','d08-586z-01','p10-249z-01','e07-425z-02','n05-514z-02']
    clean,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');wi=writer_tensor(ids,data['records'],cfg['writers'],'cpu')
    checks={};metrics={};duration_drift={}
    for a in ARMS:
        model=PositionContractWriter(**cfg['models'][a]);saved=torch.load(p/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval()
        with torch.no_grad():z=transform(model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi),stats,True)
        checks[a]=[];metrics[a]=json.loads((p/a/'eval-48000.json').read_text())['aggregate'];drifts=[]
        with h5py.File(p/a/'evaluation-48000.h5') as native,h5py.File(p/a/'duration-evaluation-48000.h5') as estimated:
            for j,sid in enumerate(ids):
                points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=native['correct/'+sid];gpu=q['points'][:];old=json.loads(q.attrs['row'])
                checks[a].append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),reader_equal=m['free_decoded']==old['free_decoded']))
            for sid in data['splits']['all_train256']:
                n=native['correct/'+sid];e=estimated['estimated_correct/'+sid];nr=json.loads(n.attrs['row']);er=json.loads(e.attrs['row']);xy=n['points'][:];other=e['points'][:];count=min(data['records'][sid]['points'],len(xy),len(other));diff=xy[:count,:2]-other[:count,:2]
                drifts.append(dict(sample_id=sid,native_cer=nr['free_errors']/nr['characters'],estimated_cer=er['free_errors']/er['characters'],native_errors=nr['free_errors'],estimated_errors=er['free_errors'],characters=nr['characters'],delta_blocks=er['predicted_blocks']-er['actual_blocks_for_diagnostic_only'],common_points=count,common_prefix_x_drift=float(np.sqrt((diff[:,0]**2).mean())),common_prefix_y_drift=float(np.sqrt((diff[:,1]**2).mean())),common_prefix_pen_changes=int((xy[:count,2:].argmax(1)!=other[:count,2:].argmax(1)).sum()),native_missing_eoc=nr['first_eoc_point'] is None,estimated_missing_eoc=er['first_eoc_point'] is None))
        if any(c['max_xy_difference']>.0002 or c['pen_mismatches'] or not c['reader_equal'] for c in checks[a]):
            raise ValueError('independent native CPU/GPU reload drift')
        groups={}
        for group in ['unchanged','changed','shorter','longer']:
            rows=[r for r in drifts if (r['delta_blocks']==0 if group=='unchanged' else r['delta_blocks']!=0 if group=='changed' else r['delta_blocks']<0 if group=='shorter' else r['delta_blocks']>0)]
            denominator=sum(r['characters'] for r in rows)
            groups[group]=dict(lines=len(rows),native_cer=sum(r['native_errors'] for r in rows)/denominator if denominator else None,estimated_cer=sum(r['estimated_errors'] for r in rows)/denominator if denominator else None,
                mean_common_prefix_x_drift=float(np.mean([r['common_prefix_x_drift'] for r in rows])) if rows else None,mean_common_prefix_y_drift=float(np.mean([r['common_prefix_y_drift'] for r in rows])) if rows else None,pen_changes=sum(r['common_prefix_pen_changes'] for r in rows),estimated_missing_eoc=sum(r['estimated_missing_eoc'] for r in rows))
        duration_drift[a]=dict(aggregate=groups,lines=drifts)
    pages=[]
    for split in ['retained_train','added_same_writer','new_train128','expansion256','unseen_prompt']:
        for start in range(0,len(data['splits'][split]),4):
            group=data['splits'][split][start:start+4];fig,axs=plt.subplots(len(group),5,figsize=(35,3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                truth=targets[sid];draw(axs[j,0],split_xy(truth[:,:2],truth[:,2:].argmax(1)));axs[j,0].set_title(sid+' / source\n'+data['records'][sid]['text'],fontsize=8)
                for ax,(a,policy) in zip(axs[j,1:],[(a,q) for a in ARMS for q in ['native','estimated']]):
                    folder=p/a/('exposed-development-duration' if split=='unseen_prompt' and policy=='estimated' else '')
                    file=folder/('evaluation-48000.h5' if policy=='native' else 'duration-evaluation-48000.h5');mode='correct' if policy=='native' else 'estimated_correct'
                    with h5py.File(file) as f:
                        q=f[mode+'/'+sid];r=json.loads(q.attrs['row']);points=q['points'][:r['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(a+'/'+policy+'\nreader: '+r['free_decoded'],fontsize=8)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append(name)
    summary=dict(config=cfg,guards=guards,results=results,final_native=metrics,duration_drift=duration_drift,cpu_reload=checks,
        limitations='Final matched48000 outputs shown, not cherry-picked dev checkpoints. Best checkpoint uses only native TRAIN geometry. Exposed olddev8 is not blind confirmation. Native oracle, estimated duration, and true new-text composition are separate gates; no promotion inferred from TRAIN fit.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Fresh positional contract: fitting versus duration robustness</title><style>body{font:17px system-ui;max-width:2400px;margin:30px auto;padding:20px}img{width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Fresh relative100 versus absolute query positions</h1><p>Identical fresh weights/empty Adam/RNG,256TRAIN/32writers/order,48000updates, shared losses and Gaussian prior. Only relative query PE omitted. No generic smoothing, trajectory resampling, CTC/style/KL or held checkpoint selection. Separate native oracle timing, TRAIN-only estimated durations and exposed-development behavior. Absolute PE is length-independent; full self-attention outputs are not necessarily prefix-invariant.</p><p><a href="summary.json">Complete protocol/metrics/drift/reload</a> · <a href="../as-run-source.tar.gz">As-run implementation</a></p><table><tr><th>Arm/split</th><th>CER</th><th>Exact</th><th>X/Y RMSE</th><th>target segment error</th><th>min pen F1</th></tr>'
    for a in ARMS:
        for split in ['all_train256','unseen_prompt']:
            r=metrics[a][split]['correct'];body+=f'<tr><td>{a}/{split}/oracle</td><td>{100*r["free_cer"]:.3f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.6f}/{r["y_rmse"]:.6f}</td><td>{r["segment_vector_rmse"]:.6f}</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
        r=results[a]['duration_history'][-1]['train']['estimated_correct'];body+=f'<tr><td>{a}/TRAIN estimated</td><td>{100*r["free_cer"]:.3f}%</td><td>{r["free_exact"]}/256</td><td colspan="3">Different requested lengths: no fictitious target-index RMSE</td></tr>'
    body+='</table><p>All264native and target-free estimated-duration outputs shown with predicted pens/first-EOC stopping. Source / relative native / relative estimated / absolute native / absolute estimated. No filtering/markers/smoothing. Short requested windows can truncate writing; inspect termination, common-prefix drift and shorter/longer bins separately. Previously opened confirmation is not rerun. These are standalone mapper tests, NOT proof that released InkDiT or semantic InkVAE works.</p>'
    for name in pages:body+='<img loading="lazy" src="'+name+'">'
    (out/'index.html').write_text(body)
    return summary
