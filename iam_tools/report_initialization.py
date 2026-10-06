"""All-line marker-free initialization stability report; show FINAL, not best0."""
import json
import html
from pathlib import Path
import numpy as np


def report(directory, data_root='data', include_draws=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(data_root);results=json.loads((directory/'result.json').read_text())
    arms=list(results);provenance=json.loads((directory/arms[0]/'provenance.json').read_text());splits=provenance['splits']
    target={}
    for kind,ids in [('train',splits['train']),('val',splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{kind}.h5') as hf:
            for sid in ids:
                a=hf[sid]['point_seq'][:].copy();a[:,:2]*=.01;target[sid]=a
    out=directory/'report';out.mkdir(exist_ok=True)
    stages={arm:json.loads((directory/arm/f"eval-{r['last_step']}.json").read_text()) for arm,r in results.items()}
    initial=json.loads((directory/arms[0]/'eval-0.json').read_text())
    baseline=root/'checkpoints/iam_geometry_fullset/20261006-152500'
    baseline_row=None
    if (baseline/'eval-40.json').exists():
        bp=json.loads((baseline/'provenance.json').read_text())
        if bp['samples']!=provenance['samples']:raise AssertionError('comparison baseline sample hashes differ')
        baseline_row=json.loads((baseline/'eval-40.json').read_text())
    before_columns=3 if baseline_row else 2
    def arrays_for(sid):
        old=[('previous learned192 (not matched initialization)',np.load(baseline/f'step-40/{sid}/mu.npy'))] if baseline_row else []
        return [('target',target[sid])]+old+[('initialized (0 updates)',np.load(directory/arms[0]/f'step-0/{sid}/mu.npy'))]+[(f"{a}, {results[a]['last_step']} updates",np.load(directory/a/f"step-{results[a]['last_step']}/{sid}/mu.npy")) for a in arms]
    pages=[]
    for group in ('old','new','held_out'):
        for page,ids in enumerate(paginate_ids(splits[group]),1):
            fig,axes=plt.subplots(len(ids),before_columns+len(arms),figsize=(5*(before_columns+len(arms)),2*len(ids)),squeeze=False)
            for j,sid in enumerate(ids):
                t=target[sid][:,:2];lo=t.min(0);hi=t.max(0)
                for ax,(label,a) in zip(axes[j],arrays_for(sid)):
                    draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08)
                    ax.set_title(sid+' — '+label,fontsize=8)
            fig.tight_layout();name=f'{group}-{page}.png';fig.savefig(out/name,dpi=140);plt.close(fig);pages.append((group,page,name))
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(before_columns+len(arms),2,figsize=(8,3*(before_columns+len(arms))),squeeze=False)
    for column,(sid,xlim,ylim,title) in enumerate(regions):
        for ax,(label,a) in zip(axes[:,column],arrays_for(sid)):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(title+' — '+label,fontsize=9)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=170);plt.close(fig)
    # Median/worst sampled draws for EVERY line and arm, not hand-picked draws.
    for arm,row in (stages.items() if include_draws else []):
        for line in row['lines']:
            sid=line['sample_id'];rank=sorted(line['sampled'],key=lambda v:v['geometry']['x_rmse']+v['geometry']['y_rmse'])
            arr=[('target',target[sid]),('mean',np.load(directory/arm/f"step-{results[arm]['last_step']}/{sid}/mu.npy"))]
            for label,variant in [('median posterior draw',rank[len(rank)//2]),('worst posterior draw',rank[-1])]:
                arr.append((label,np.load(directory/arm/f"step-{results[arm]['last_step']}/{sid}/{variant['kind']}.npy")))
            fig,axes=plt.subplots(4,1,figsize=(14,5));lo=target[sid][:,:2].min(0);hi=target[sid][:,:2].max(0)
            for ax,(label,a) in zip(axes,arr):
                draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(label,fontsize=8)
            fig.tight_layout();fig.savefig(out/f'{arm}-{sid}.png',dpi=130);plt.close(fig)
    summary=dict(previous_learned192=baseline_row['groups'] if baseline_row else None,initial=initial['groups'],final={a:r['groups'] for a,r in stages.items()},results=results,provenance=provenance)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    page=['<meta charset="utf-8"><title>Initialization optimizer stability</title><h1>Initialized geometry: does it survive actual updates?</h1>',
          '<p><strong>Engineering diagnostic, NOT paper reproduction or a semantic/generative VAE promotion.</strong> Geometry is explicitly reinitialized in the same architecture; dormant residual/attention weights and frozen OCR/style come from a pinned checkpoint. All arms use identical initial weights, training IDs and RNG. Active posterior std starts .002, not the .00002 CPU capacity probe. Held-out32 are evaluation-only, seen writers/forms overlap. No GMM, KL, CTC or style objectives.</p>',
          '<p>Marker-free polylines, predicted pens, identical target bounds. Severely broken predictions can be OFFSCREEN: blank panels are failure, not missing data. These are FINAL updates, even when initial0 remains the best training checkpoint. Index differences are not physical velocity/curvature; tangent and turn are geometric angles. Original target corners are preserved; no smoothing.</p>',
          '<p><a href="summary.json">Full metrics/config/provenance</a></p><table border="1"><tr><th>Arm / group</th><th>Mean X/Y RMSE</th><th>Δ / Δ² relative</th><th>turn p90</th><th>corner turn p90</th><th>pen F1</th><th>sampled X/Y</th></tr>']
    for label,row in ([('previous learned192',baseline_row)] if baseline_row else [])+[('initialized',initial)]+list(stages.items()):
        for group in ['train','held_out','old']:
            g=row['groups'][group];mu,z=g['mu'],g['sampled'];corner=mu.get('target_corner_turn_error_degrees',{}).get('p90')
            page.append(f'<tr><td>{label}/{group}</td><td>{mu["mean_per_line_x_rmse"]:.7f}/{mu["mean_per_line_y_rmse"]:.7f}</td><td>{mu["mean_per_line_first_difference_relative"]:.5f}/{mu["mean_per_line_second_difference_relative"]:.5f}</td><td>{mu["turn_angle_error_degrees"]["p90"]:.4f}°</td><td>{corner}</td><td>{g["mu_macro_pen_f1"]:.5f}</td><td>{z["mean_per_line_x_rmse"]:.6f}/{z["mean_per_line_y_rmse"]:.6f}</td></tr>')
    page.extend(['</table><h2>User-identified curves</h2><img style="max-width:100%" src="user-regions.png">'])
    for group,num,name in pages:page.append(f'<h2>{group}, page {num}</h2><img style="max-width:100%" src="{name}">')
    for sid in sorted(target):
        page.append(f'<h2>{sid}: {html.escape(provenance["samples"][sid]["text"])}</h2>')
        for arm in arms:
            if include_draws:page.append(f'<h3>{arm}</h3><img style="max-width:100%" src="{arm}-{sid}.png">')
            else:page.append(f'<a href="../{arm}/eval-{results[arm]["last_step"]}.json">{arm}: all20 posterior draw metrics</a>')
    (out/'index.html').write_text('\n'.join(page));return summary


def overview(root='/data'):
    """Small immutable-run comparison; no inference or training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw,split_xy
    root=Path(root);family=root/'checkpoints/iam_initialization_study';out=family/'research-summary';out.mkdir(exist_ok=True)
    control=family/'20261006-165824';protected=family/'20261006-170142'
    baseline=root/'checkpoints/iam_geometry_fullset/20261006-152500'
    sources=[('previous learned192',baseline,40),('initialized (untrained)',protected/'protected_noise',0),
             ('frozen readout, body5e-5',control/'frozen_readout',100),
             ('protected, update100',protected/'protected_noise',100),('protected, update200',protected/'protected_noise',200)]
    metrics={label:json.loads((path/f'eval-{step}.json').read_text())['groups'] for label,path,step in sources}
    sid_regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(1+len(sources),2,figsize=(8,3*(1+len(sources))),squeeze=False)
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5') as hf:
        for col,(sid,xlim,ylim,title) in enumerate(sid_regions):
            target=hf[sid]['point_seq'][:].copy();target[:,:2]*=.01
            arrays=[('IAM/RDP target',target)]+[(label,np.load(path/f'step-{step}/{sid}/mu.npy')) for label,path,step in sources]
            for ax,(label,a) in zip(axes[:,col],arrays):
                draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(title+'\n'+label,fontsize=9)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=160);plt.close(fig)
    info=dict(metrics=metrics,matched_comparison='frozen readout100 vs protected100: same reinitialized geometry/source/data/noise; body LR and posterior LR both differ',
              unmatched_comparison='previous learned192 is a separate optimizer/init experiment, not a causal ablation',
              caveat='Engineering initialized transport codec. Not paper reproduction, semantic latent or generation readiness. Tiny posterior noise and KL off.')
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    page=['<meta charset="utf-8"><title>English curve investigation: initialized transport</title><h1>Curve fidelity after real optimizer updates</h1>',
          '<p>Structured initialization and selective optimization preserve the real processed strokes, without smoothing or a raw-input skip. Mean curves are near-lossless; posterior reconstructions improve. <strong>This is an initialized transport codec, NOT a proven semantic/generative VAE.</strong> KL/CTC/style/GMM training remain OFF. The previously trained192 model is a separate experiment, not a matched causal baseline. Original IAM/RDP polygonality remains intentionally intact.</p>',
          '<p><a href="../20261006-170142/report/index.html">All224 lines, posterior median/worst and provenance</a> | <a href="../20261006-165824/report/index.html">Controlled optimizer failures and frozen-readout arm</a> | <a href="summary.json">Metrics</a></p>',
          '<table border="1"><tr><th>Stage / group</th><th>Mean X/Y RMSE</th><th>Δ/Δ² relative</th><th>turn p90</th><th>pen F1</th><th>sampled X/Y</th></tr>']
    for label,groups in metrics.items():
        for group in ('train','held_out','old'):
            s=groups[group];g,z=s['mu'],s['sampled']
            page.append(f'<tr><td>{label}/{group}</td><td>{g["mean_per_line_x_rmse"]:.7f}/{g["mean_per_line_y_rmse"]:.7f}</td><td>{g["mean_per_line_first_difference_relative"]:.5f}/{g["mean_per_line_second_difference_relative"]:.5f}</td><td>{g["turn_angle_error_degrees"]["p90"]:.3f}°</td><td>{s["mu_macro_pen_f1"]:.4f}</td><td>{z["mean_per_line_x_rmse"]:.6f}/{z["mean_per_line_y_rmse"]:.6f}</td></tr>')
    page+=['</table><h2>User-identified curves: marker-free, same bounds, no smoothing</h2><img style="max-width:100%" src="user-regions.png">',
           '<p>Gpoint + .204718×target first-difference matching + bounded pen + .1 sampled geometry. Physical batch1/accum8, dropout0, input scale.01, sigma/rho/OCR/style fixed. Protected: Transformer/readout fixed, encoder/decoder/mu LR1e-7, posterior LR1e-3. Train192 only; held-out32 seen-writer line-disjoint, forms overlap. Same train schedule/RNG, seed4042. Differences are nonuniform point-index differences, not physical velocity/curvature; turn angles are geometric.</p>']
    (out/'index.html').write_text('\n'.join(page));return str(out/'index.html')
