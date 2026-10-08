"""Matched anchor-refresh comparisons with full TRAIN and immutable CPU reload."""
import collections,html,json,tarfile
from pathlib import Path
import h5py,numpy as np,torch
from .generation_capacity import DATA,SOURCE_H5_SHA,DATASET_SHA,WHITENING_SHA,select_capacity
from .generation_anchor_refresh_study import ARMS,PARENT,PARENT_SHA
from .generation_coverage import WriterTextDenoiser
from .generation_coverage_study import writer_tensor,score
from .generation_study import collate,decode_sample
from .latent_diffusion import transform
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .pen_ab import file_sha
from .report_resources import report as resources_report


def verify(directory,cfg,results,data):
    p=Path(directory);trains=data['training_ids'];records=data['records'];held=data['splits']['unseen_prompt']
    from .ocr_pool import normalized_text
    forms={records[i]['prompt_family'] for i in held};texts={normalized_text(records[i]['text']) for i in held}
    if set(results)!=set(ARMS) or set(trains)!=set(ARMS):raise ValueError('exact three-arm guard')
    if not all(trains[a]==trains['control'] for a in ARMS):raise ValueError('shared256 TRAIN guard')
    schedules={};weights={};groups={};exposure={};optimizer_digests={}
    for a,r in results.items():
        if r['last_step']!=cfg['max_updates'] or not r['codec_reader_unchanged'] or r['stop']!='budget_completed':raise ValueError('complete matched budget/frozen guard')
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()]
        if len(rows)!=r['last_step'] or [row['step'] for row in rows]!=list(range(1,r['last_step']+1)) or any(not set(row['sample_ids'])<=set(trains[a]) for row in rows):raise ValueError('training scope/log sequence guard')
        if any(records[i]['prompt_family'] in forms or normalized_text(records[i]['text']) in texts for i in trains[a]):raise ValueError('held form/transcript leakage')
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(p/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
        schedules[a]=[row['sample_ids'] for row in rows];counts=collections.Counter(i for row in rows for i in row['sample_ids']);values=[counts[i] for i in trains[a]]
        exposure[a]=dict(min=min(values),median=float(np.median(values)),max=max(values),total=sum(values))
        saved=torch.load(p/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        if saved['step']!=16000 or not saved['optimizer_state_dict']['state']:raise ValueError('restored model/Adam guard')
        optimizer_digests[a]=tensor_digest({f'{i}:{k}':v for i,state in saved['optimizer_state_dict']['state'].items() for k,v in state.items()})
        weights[a]=tensor_digest(saved['model_state_dict']);groups[a]=[{k:g[k] for k in ('lr','betas','weight_decay','eps')} for g in saved['optimizer_state_dict']['param_groups']]
        history=r['history'];best=min(history,key=lambda h:h['train_score'])
        if r['best_step']!=best['step']:raise ValueError('TRAIN-only checkpoint selection guard')
        for h in history:
            ev=json.loads((p/a/f'eval-{h["step"]}.json').read_text())
            if abs(score(ev,set(trains[a]))-h['train_score'])>1e-9:raise ValueError('actual full TRAIN selection mismatch')
    if any(schedules[a]!=schedules['control'] for a in ARMS):raise ValueError('matched256 minibatch guard')
    if len(set(weights.values()))!=1 or len(set(optimizer_digests.values()))!=1 or any(g!=groups['control'] for g in groups.values()):raise ValueError('identical initial weights/Adam guard')
    return dict(full_budgets=True,codec_reader_frozen=True,full_actual_train_selection=True,held_forms_and_texts_excluded=True,matched256_minibatches=True,identical_model_and_adam=True,initial_optimizer_digests=optimizer_digests,actual_optimizer_groups=groups,exposure=exposure,initial_state_sha256=weights)


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);cfg=json.loads((directory/'config.json').read_text());
    if cfg['parent']!=PARENT or cfg['parent_sha256']!=PARENT_SHA or file_sha(root/PARENT)!=PARENT_SHA:raise ValueError('pinned larger256 source guard')
    data=json.loads((directory/'dataset.json').read_text());results=json.loads((directory/'result.json').read_text());source=root/DATA
    for name,sha in [('source.h5',SOURCE_H5_SHA),('dataset.json',DATASET_SHA),('whitening.pt',WHITENING_SHA)]:
        if file_sha(source/name)!=sha:raise ValueError('immutable data drift')
    if file_sha(directory/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:raise ValueError('source archive drift')
    original=json.loads((source/'dataset.json').read_text());scope=select_capacity(original);scope['training_ids']={a:scope['training_ids']['larger256'] for a in ARMS}
    if data['splits']!=scope['splits'] or data['training_ids']!=scope['training_ids']:raise ValueError('reproducible dataset selector drift')
    # Reject a report made with a different training implementation. Unrelated new files are allowed.
    with tarfile.open(directory/'as-run-source.tar.gz') as tar:
        for name in ['iam_tools/generation_anchor_refresh.py','iam_tools/generation_anchor_refresh_study.py','iam_tools/generation_coverage_study.py','iam_tools/generation_coverage.py','iam_tools/latent_diffusion.py','model/vae.py','model/losses.py']:
            if tar.extractfile(name).read()!=(Path(repo)/name).read_bytes():raise ValueError('as-run implementation drift: '+name)
    guards=verify(directory,cfg,results,data);out=directory/'report'
    if out.exists():raise ValueError('refuse overwrite completed report')
    resources_report(directory);(out/'index.html').replace(out/'resources.html');(out/'report-source.py').replace(out/'resource-report-source.py');(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    latents={};targets={};evaluations={};diagnostics={}
    with h5py.File(source/'source.h5') as f:
        for sid in data['records']:latents[sid]=torch.tensor(f[sid]['latent_mean'][:]);targets[sid]=f[sid]['target'][:]
    for a,r in results.items():
        e=json.loads((directory/a/f'eval-{r["last_step"]}.json').read_text())
        if file_sha(directory/a/f'evaluation-{r["last_step"]}.h5')!=e['packed_h5_sha256']:raise ValueError('packed evaluation drift')
        evaluations[a]=e;train=[row for row in e['lines'] if row['policy']=='correct' and row['sample_id'] in data['training_ids'][a]]
        diagnostics[a]=dict(per_line_cer_median=float(np.median([row['free_errors']/row['characters'] for row in train])),per_line_cer_p90=float(np.percentile([row['free_errors']/row['characters'] for row in train],90)),pen_f1_mean=float(np.mean([row['pen_aligned_reference']['pen_up_f1'] for row in train])))
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(source/'whitening.pt',weights_only=True)
    ids=data['splits']['retained_train'][:2]+data['splits']['new_train128'][:2]+data['splits']['expansion256'][:2]+data['splits']['unseen_prompt'][:2]
    clean,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');wi=writer_tensor(ids,data['records'],cfg['writers'],'cpu');checks={}
    for a,r in results.items():
        model=WriterTextDenoiser(**cfg['models'][a]);saved=torch.load(directory/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval()
        with torch.no_grad():z=transform(model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi),stats,True)
        checks[a]=[]
        with h5py.File(directory/a/f'evaluation-{r["last_step"]}.h5') as f:
            for j,sid in enumerate(ids):
                points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=f['correct/'+sid];old=json.loads(q.attrs['row']);gpu=q['points'][:]
                checks[a].append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),transcript_equal=m['free_decoded']==old['free_decoded']))
    initial_equal=True
    with h5py.File(directory/'control/evaluation-0.h5') as f:
        for a in ('refresh_xy','refresh_xy_segment'):
            with h5py.File(directory/a/'evaluation-0.h5') as g:
                for sid in f['correct']:initial_equal &= np.array_equal(f['correct/'+sid+'/points'][:],g['correct/'+sid+'/points'][:])
    if not initial_equal:raise ValueError('identical trained parent initial prediction guard')
    fig,axs=plt.subplots(1,3,figsize=(18,5))
    for a,r in results.items():
        split='all_train256'
        history=r['history'];xs=[h['step'] for h in history]
        axs[0].plot(xs,[100*h['aggregate'][split]['correct']['free_cer'] for h in history],label=a)
        axs[1].plot(xs,[h['aggregate'][split]['correct']['segment_vector_rmse'] for h in history],label=a)
        axs[2].plot(xs,[100*h['aggregate']['unseen_prompt']['correct']['free_cer'] for h in history],label=a)
    for ax,title in zip(axs,['Actual full TRAIN CER (%)','Full TRAIN segment error','Held form CER (%) — not used to select']):
        ax.set_title(title);ax.set_xlabel('updates');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=130);plt.close(fig)
    pages=[]
    # Non-overlapping gallery covers EVERY264 line once, no markers/smoothing.
    for split in ('retained_train','added_same_writer','new_train128','expansion256','unseen_prompt'):
        ids=data['splits'][split]
        for start in range(0,len(ids),4):
            group=ids[start:start+4];fig,axs=plt.subplots(len(group),4,figsize=(24,3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                draw(axs[j,0],split_xy(targets[sid][:,:2],targets[sid][:,2:].argmax(1)));axs[j,0].set_title(sid+' | reference\n'+data['records'][sid]['text'],fontsize=8)
                for ax,a in zip(axs[j,1:],ARMS):
                    with h5py.File(directory/a/f'evaluation-{results[a]["last_step"]}.h5') as f:
                        q=f['correct/'+sid];row=json.loads(q.attrs['row']);points=q['points'][:row['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(a+' / '+('TRAIN' if sid in data['training_ids'][a] else 'UNTRAINED')+'\nreader: '+row['free_decoded'],fontsize=8)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append(name)
    preflight=dict(lines=len(data['preflight']),max_codec_rmse=max(r['codec_rmse'] for r in data['preflight']),source_reader_exact=sum(r['reader_errors']==0 for r in data['preflight']),source_reader_errors=sum(r['reader_errors'] for r in data['preflight']))
    summary=dict(config=cfg,guards=guards,results=results,final_matrix={a:e['aggregate'] for a,e in evaluations.items()},train_distributions=diagnostics,cpu_reload=checks,initial_trajectories_bitwise_identical=initial_equal,preflight=preflight,limitations=cfg['caveats'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Target-geometry anchor refresh</title><style>body{font:16px system-ui;max-width:2200px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Matched256-line geometry anchor refresh</h1><p>Standalone deterministic text+writer prototype, NOT semantic InkVAE/original InkDiT. All arms restore identical completed larger256 step16000 model AND Adam. Same256TRAIN/order/budget/zero input/oracle duration and fixed original32 whitening. Frozen faithful codec/corpus-familiar reader; held excluded from calibration/training/selection. No KL/CTC/style, dropout or smoothing. Control keeps earlier physical anchors. refresh_xy recalibrates only isotropic XY to25% of initial shared base gradient; refresh_xy_segment also recalibrates target within-stroke index differences to10%. Norms use mean gradients across24TRAIN calibration lines; coefficients then remain fixed. These are initial gradient fractions, NOT constant later fractions. Different objectives intentionally differ in loss values; judge actual geometry/readability, not total loss.</p><p><a href="summary.json">Full metrics/provenance</a> · <a href="../config.json">As-run config and coefficients</a> · <a href="resources.html">Telemetry</a></p><h2>Equal final update budget</h2><table><tr><th>Arm</th><th>Split/condition</th><th>CER</th><th>Exact</th><th>X/YRMSE</th><th>segment/second-index-diff</th><th>mean tangent/turn p90</th><th>minpenF1</th></tr>'
    for a,e in evaluations.items():
        for split in ('all_train128','all_train256','unseen_prompt'):
            for policy,r in e['aggregate'][split].items():
                note=''
                body+=f'<tr><td>{a}</td><td>{split}{note}/{policy}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.5f}/{r["y_rmse"]:.5f}</td><td>{r["segment_vector_rmse"]:.5f}/{r["second_difference_vector_rmse"]:.5f}</td><td>{r["tangent_p90_mean"]:.1f}/{r["turn_p90_mean"]:.1f}°</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
    body+='</table><p>Selection uses EVERY actual TRAIN line, never held CER. Tables use matched final steps, not different best aliases. Swapped inputs are controls evaluated against ORIGINAL text, not claims of swapped-prompt quality. Matching a held target pointwise is not necessary for valid writing. Index differences are not physical velocity or geometric curvature; angle errors compare normalized local directions. All renders use predicted pens and actual first-EOC stopping. No markers or postprocessing.</p>'
    body+='<h2>Learning curves</h2><p>All arms use identical256lines/order and source weights/Adam. Held shown descriptively, never used for selection.</p><img src="learning-curves.png">'
    for name in pages:body+='<h2>'+name+'</h2><img loading="lazy" src="'+name+'">'
    body+='<h2>CPU reload/guards/source ceilings</h2><pre>'+html.escape(json.dumps(dict(cpu_reload=checks,guards=guards,preflight=preflight),indent=2))+'</pre>'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),preflight=preflight,cpu_reload=checks)
