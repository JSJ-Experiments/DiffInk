"""All-line marker-free matched conditioning comparison; no smoothing."""
import html,json
from pathlib import Path
import numpy as np


def report(directory,data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(data_root);out=directory/'report';out.mkdir(exist_ok=True)
    arms={p.name:json.loads((p/'report/summary.json').read_text()) for p in directory.iterdir() if (p/'report/summary.json').exists()}
    order=[m for m in ('control','center','channel','edge_pad') if m in arms]
    if not order:raise ValueError('completed arm reports required')
    splits=arms[order[0]]['provenance']['splits'];target={}
    for name,ids in [('train',splits['train']),('val',splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{name}.h5') as hf:
            for sid in ids:
                a=hf[sid]['point_seq'][:].copy();a[:,:2]*=.01;target[sid]=a
    source=root/arms[order[0]]['provenance']['source_rel']
    # Actual source mean arrays are available beside its checkpoint.
    import torch
    source_step=torch.load(source,map_location='cpu',weights_only=True)['optimizer_updates']
    def arrays(sid):
        values={'target':target[sid]}
        source_npy=source.parent/f'step-{source_step}'/sid/'mu.npy'
        if source_npy.exists():values['expanded source']=np.load(source_npy)
        for mode in order:
            step=arms[mode]['selected_step'];values[mode]=np.load(directory/mode/f'step-{step}'/sid/'mu.npy')
        return values
    images=[]
    for group in ('old','new','held_out'):
        ids=splits[group];ncols=len(arrays(ids[0]));fig,axes=plt.subplots(len(ids),ncols,figsize=(5*ncols,2.2*len(ids)),squeeze=False)
        for row,sid in enumerate(ids):
            aa=arrays(sid);allxy=np.concatenate([a[:,:2] for a in aa.values()]);lo=allxy.min(0);hi=allxy.max(0)
            for ax,(label,a) in zip(axes[row],aa.items()):
                draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.02,hi[0]+.02);ax.set_ylim(lo[1]-.05,hi[1]+.05);ax.set_title(sid+' — '+label,fontsize=9)
        fig.tight_layout();name=group+'-comparison.png';fig.savefig(out/name,dpi=130);plt.close(fig);images.append((group,name))
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(len(arrays(regions[0][0])),2,figsize=(9,3*len(arrays(regions[0][0]))),squeeze=False)
    for col,(sid,xlim,ylim,label) in enumerate(regions):
        for ax,(heading,a) in zip(axes[:,col],arrays(sid).items()):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(label+' — '+heading,fontsize=10)
    fig.tight_layout();fig.savefig(out/'user-regions-comparison.png',dpi=150);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(arms,indent=2)+'\n')
    page=['<meta charset="utf-8"><title>Matched reconstruction conditioning</title><h1>24-line conditioning ablations</h1>',
          '<p>Same expanded source weights, train/held-out split, RNG/sample order, objective and optimizer/schedule. Arms differ only in conditioning: unchanged control, reversible real-XY bounding-box centering, or channel-only residual normalization (48 layers, copied affine weights). Physical batch1/accum8; dropout, augmentation, GMM, KL, OCR/style supervision off. XY + target first differences, not generic smoothing. Held-out inputs never enter training/calibration/selection.</p>',
          '<p>Each arm chooses its checkpoint using training-only train_score. All arms are reported; this is not held-out selection. RDP target is polygonal; compare added distortion, not smoothness alone. Means use predicted pens. All20 sampled draws per line enter metrics; arm pages include median/worst galleries and immutable eight-line reference.</p>',
          '<table border="1"><tr><th>Arm/group</th><th>X/Y RMSE</th><th>Δ / Δ² relative</th><th>Turn p90</th><th>Pen F1</th><th>Sampled X/Y</th></tr>']
    for mode in order:
        for group in ('old','new','held_out'):
            s=arms[mode]['final'][group];g=s['mu'];z=s['sampled']
            page.append(f'<tr><td>{html.escape(mode)}/{group}</td><td>{g["mean_per_line_x_rmse"]:.5f}/{g["mean_per_line_y_rmse"]:.5f}</td><td>{g["mean_per_line_first_difference_relative"]:.3f}/{g["mean_per_line_second_difference_relative"]:.3f}</td><td>{g["turn_angle_error_degrees"]["p90"]:.2f}°</td><td>{s["mu_macro_pen_f1"]:.4f}</td><td>{z["mean_per_line_x_rmse"]:.5f}/{z["mean_per_line_y_rmse"]:.5f}</td></tr>')
    page+=['</table><p><a href="summary.json">All metrics/provenance</a></p>','<h2>Highlighted curves</h2><img style="max-width:100%" src="user-regions-comparison.png">']
    for group,name in images:page.append(f'<h2>{group}, all lines</h2><img style="max-width:100%" src="{name}">')
    for mode in order:page.append(f'<p><a href="../{mode}/report/index.html">{mode}: all-line sampled gallery, exact metrics and source checks</a></p>')
    (out/'index.html').write_text('\n'.join(page));return arms
