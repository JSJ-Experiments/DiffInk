"""Matched nuisance-jitter review; never equate duration robustness with composition."""
import collections,html,json,tarfile
from pathlib import Path
import h5py,numpy as np,torch
from .generation_position_robustness_study import ARMS
from .generation_position_robustness import jitter_lengths
from .generation_timing import TimingProbe
from .generation_timing_study import PARENT,PARENT_SHA
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA,WHITENING_SHA
from .generation_coverage_study import score,writer_tensor
from .generation_study import collate,decode_sample
from .latent_diffusion import transform
from .writer_expansion import load,training_schedule
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha


def verify(directory,cfg,results,data):
    p=Path(directory);schedule={};initial={};moments={};rng={}
    if set(results)!=set(ARMS) or set(data['training_ids'])!=set(ARMS):raise ValueError('exact control/jitter arms required')
    if cfg['models']['control']!=cfg['models']['pe_jitter'] or data['training_ids']['control']!=data['training_ids']['pe_jitter']:raise ValueError('identical model/TRAIN guard')
    if cfg['parent_step']!=48000 or cfg['max_updates']!=8000 or cfg['lr']!=1e-5 or cfg['jitter_probability']!=.5 or cfg['jitter_fraction']!=.2:raise ValueError('bounded pinned protocol guard')
    for a in ARMS:
        r=results[a];ids=data['training_ids'][a]
        if r['last_step']!=8000 or r['stop']!='budget_completed' or not r['codec_reader_unchanged']:raise ValueError('complete matched budget/frozen guard')
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()];schedule[a]=[x['sample_ids'] for x in rows]
        expected=list(training_schedule(ids,8000,8,seed=cfg['schedule_seed']))
        if schedule[a]!=expected or [x['step'] for x in rows]!=list(range(1,8001)):raise ValueError('reproducible matched TRAIN order guard')
        generator=torch.Generator().manual_seed(cfg['jitter_seed'])
        for x in rows:
            native=torch.tensor(x['native_lengths']);lengths=jitter_lengths(native,generator,probability=.5,fraction=.2);expected_lengths=lengths.tolist() if a=='pe_jitter' else native.tolist()
            if x['position_lengths']!=expected_lengths or any(n!=(data['records'][sid]['points']+7)//8 for n,sid in zip(x['native_lengths'],x['sample_ids'])) or x['learning_rates']!=[1e-5]:raise ValueError('actual nuisance-only lengths/LR drift')
            if not np.isfinite(x['loss']) or not np.isfinite(x['gradient_norm']):raise ValueError('finite optimization required')
        saved=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False);initial[a]=tensor_digest(saved['model_state_dict']);opt=saved['optimizer_state_dict']
        if saved['absolute_step']!=48000 or not opt['state']:raise ValueError('restored parent/full Adam required')
        moments[a]=tensor_digest({f'{i}:{k}':v for i,state in opt['state'].items() for k,v in state.items()});rng[a]={k:saved[k] for k in ['torch_rng_state','cuda_rng_state','jitter_rng_state']}
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(p/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
        if [h['step'] for h in r['history']]!=cfg['eval_steps'] or r['best_step']!=min(r['history'],key=lambda h:h['train_score'])['step']:raise ValueError('native TRAIN-only selection guard')
        for h in r['history']:
            ev=json.loads((p/a/f'eval-{h["step"]}.json').read_text());actual=[row['sample_id'] for row in ev['lines'] if row['policy']=='correct' and row['sample_id'] in ids]
            if len(actual)!=len(ids) or set(actual)!=set(ids) or abs(score(ev,set(ids))-h['train_score'])>1e-9 or file_sha(p/a/f'evaluation-{h["step"]}.h5')!=ev['packed_h5_sha256']:raise ValueError('all native TRAIN selection/packed outputs required')
            de=json.loads((p/a/f'duration-eval-{h["step"]}.json').read_text())
            if de['ids']!=data['splits']['all_train256'] or file_sha(p/a/f'duration-evaluation-{h["step"]}.h5')!=de['packed_h5_sha256']:raise ValueError('all256TRAIN target-free duration output required')
    if schedule['control']!=schedule['pe_jitter'] or len(set(initial.values()))!=1 or len(set(moments.values()))!=1 or any(not torch.equal(rng['control'][k],rng['pe_jitter'][k]) for k in rng['control']):raise ValueError('matched parent/Adam/RNG/minibatch guard')
    return dict(complete_budgets=True,identical_initial_model_adam_rng=True,identical_minibatches_lrs=True,actual_nuisance_schedule_verified=True,all256_native_train_selection=True,targets_and_masks_unchanged=True)


def report(directory,repo,root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());results=json.loads((p/'result.json').read_text());guards=verify(p,cfg,results,data)
    if cfg['continuation_of']!=PARENT or file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:raise ValueError('pinned source/archive required')
    with tarfile.open(p/'as-run-source.tar.gz') as tar:
        for n in ['generation_timing.py','generation_position_robustness.py','generation_position_robustness_study.py','generation_coverage_study.py','generation_duration_eval.py']:
            if tar.extractfile('iam_tools/'+n).read()!=(Path(repo)/'iam_tools'/n).read_bytes():raise ValueError('as-run source drift:'+n)
    for name,sha in [('source.h5',SOURCE_H5_SHA),('dataset.json',DATASET_SHA),('whitening.pt',WHITENING_SHA)]:
        if file_sha(root/DATA/name)!=sha:raise ValueError('immutable target/whitening drift')
    original=json.loads((root/PARENT/'dataset.json').read_text());expected=dict(original,training_ids={a:original['training_ids']['soft_gaussian'] for a in ARMS})
    if data!=expected:raise ValueError('same source dataset/seals,only arm labels differ')
    parent=torch.load(root/PARENT/'soft_gaussian/checkpoint-last.pt',map_location='cpu',weights_only=False)
    if file_sha(root/PARENT/'soft_gaussian/checkpoint-last.pt')!=PARENT_SHA['soft_gaussian']:raise ValueError('pinned parent checkpoint')
    for a in ARMS:
        initial=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        if tensor_digest(initial['model_state_dict'])!=tensor_digest(parent['model_state_dict']):raise ValueError('actual parent neural weights required')
        old_opt=parent['optimizer_state_dict'];new_opt=initial['optimizer_state_dict']
        def fingerprint(opt):return tensor_digest({f'{i}:{k}':v for i,state in opt['state'].items() for k,v in state.items()})
        if fingerprint(old_opt)!=fingerprint(new_opt):raise ValueError('actual parent Adam moments required')
        for k in ['torch_rng_state','cuda_rng_state']:
            if not torch.equal(initial[k],parent[k]):raise ValueError('actual parent RNG required')
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());targets={};latents={}
    with h5py.File(root/DATA/'source.h5') as f:
        for sid in data['records']:targets[sid]=f[sid]['target'][:];latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(root/DATA/'whitening.pt',weights_only=True)
    ids=['k04-309z-09','c03-109z-03','r07-568z-04','a02-130z-03','d08-586z-01','p10-249z-01','e07-425z-02','n05-514z-02'];clean,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');wi=writer_tensor(ids,data['records'],cfg['writers'],'cpu');checks={};final={};duration={}
    for a in ARMS:
        model=TimingProbe(**cfg['models'][a]);saved=torch.load(p/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval()
        with torch.no_grad():z=transform(model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi),stats,True)
        checks[a]=[];final[a]=json.loads((p/a/'eval-8000.json').read_text())['aggregate'];duration[a]=json.loads((p/a/'duration-eval-8000.json').read_text())['aggregate']
        with h5py.File(p/a/'evaluation-8000.h5') as f:
            for j,sid in enumerate(ids):
                points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=f['correct/'+sid];gpu=q['points'][:];old=json.loads(q.attrs['row']);checks[a].append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),reader_equal=m['free_decoded']==old['free_decoded']))
    pages=[]
    for split in ['retained_train','added_same_writer','new_train128','expansion256','unseen_prompt']:
        for start in range(0,len(data['splits'][split]),4):
            group=data['splits'][split][start:start+4];columns=[(a,policy) for a in ARMS for policy in ['native','estimated']] if split!='unseen_prompt' else [(a,'native') for a in ARMS];fig,axs=plt.subplots(len(group),len(columns)+1,figsize=(8*(len(columns)+1),3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                true=targets[sid];draw(axs[j,0],split_xy(true[:,:2],true[:,2:].argmax(1)));axs[j,0].set_title(sid+' / reference\n'+data['records'][sid]['text'],fontsize=8)
                for ax,(a,policy) in zip(axs[j,1:],columns):
                    name='evaluation-8000.h5' if policy=='native' else 'duration-evaluation-8000.h5';mode='correct' if policy=='native' else 'estimated_correct'
                    with h5py.File(p/a/name) as f:q=f[mode+'/'+sid];r=json.loads(q.attrs['row']);points=q['points'][:r['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(a+'/'+policy+'\nreader: '+r['free_decoded'],fontsize=8)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append(name)
    summary=dict(config=cfg,guards=guards,results=results,final_oracle=final,final_train_estimated=duration,cpu_reload=checks,limitations='Matched continuation from one memorized checkpoint; TRAIN relativePE-nuisance invariance, NOT new-text composition proof. Existing development8 is exposed, no new sealed evaluation or selection. Targets/strokes/RDP/index spacing unchanged, not generic smoothing.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Relative-position nuisance invariance</title><style>body{font:17px system-ui;max-width:2600px;margin:30px auto;padding:20px}img{width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Controlled relative-position nuisance-jitter test</h1><p>Both restore identical soft48000model/full Adam/RNG, train8000updates at1e-5 on same256/order, no new parameters. Jitter only: half rows get relativePE denominator drawn±20% of native block count. Actual tensor length/mask, targets, point/stroke boundaries and RDP geometry unchanged. Not smoothing/time resampling. Native inference uses actual duration denominator; no oracle PE override. Checkpoint choice uses all256native TRAIN geometry only.</p><p><a href="summary.json">Full protocol/metrics</a></p><table><tr><th>Arm/split</th><th>CER</th><th>Exact</th><th>X/Y RMSE</th><th>segment error</th><th>min pen F1</th></tr>'
    for a in ARMS:
        for split in ['all_train256','unseen_prompt']:
            r=final[a][split]['correct'];body+=f'<tr><td>{a}/{split}/oracle</td><td>{100*r["free_cer"]:.3f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.6f}/{r["y_rmse"]:.6f}</td><td>{r["segment_vector_rmse"]:.6f}</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
        r=duration[a]['estimated_correct'];body+=f'<tr><td>{a}/TRAIN estimated</td><td>{100*r["free_cer"]:.3f}%</td><td>{r["free_exact"]}/256</td><td colspan="3">Different index counts; no fictitious aligned RMSE</td></tr>'
    body+='</table><h2>Failed remedy — do not promote the jitter arm</h2><p>Jitter final native TRAIN CER41.826% versus control1.340%; predicted-duration TRAIN71.466% versus61.035%. Its TRAIN-selected checkpoint is step0 (unchanged parent), whereas the control selects8000. Native curves and local segment agreement both regress, although pen boundaries remain perfect. Losses remain finite: this is not an optimization explosion. The previously exposed development difference is not evidence of compositional improvement.</p><p>All264native marker-free lines and256target-free TRAIN duration outputs. Reference / control native / control estimated / jitter native / jitter estimated. Predicted pens/first EOC, no filtering or smoothing. No new confirmation prompt tuned or reselected. The timing autopsy remains causal evidence; this particular retrospective invariance training did not fix it.</p>'
    for n in pages:body+='<img loading="lazy" src="'+n+'">'
    (out/'index.html').write_text(body);return summary
