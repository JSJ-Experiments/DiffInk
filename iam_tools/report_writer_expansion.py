"""Marker-free, all-line expansion galleries and train/held-out metrics."""
import argparse
import html
import json
from pathlib import Path
import numpy as np


def report(directory,data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(data_root)
    provenance=json.loads((directory/'provenance.json').read_text());splits=provenance['splits']
    result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else None
    selected=result['best_step'] if result else max(int(p.stem.split('-')[1]) for p in directory.glob('eval-*.json'))
    initial_step=result.get('initial_step',0) if result else min(int(p.stem.split('-')[1]) for p in directory.glob('eval-*.json'))
    stages={initial_step:json.loads((directory/f'eval-{initial_step}.json').read_text())}
    if selected!=initial_step:stages[selected]=json.loads((directory/f'eval-{selected}.json').read_text())
    out=directory/'report';out.mkdir(exist_ok=True)
    groups={i:g for g in ('old','new','held_out') for i in splits[g]}
    target={}
    for name,ids in [('train',splits['train']),('val',splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{name}.h5') as hf:
            for sid in ids:
                a=hf[sid]['point_seq'][:].copy();a[:,:2]*=.01;target[sid]=a
    images=[]
    for group in ('old','new','held_out'):
        ids=splits[group];fig,axes=plt.subplots(len(ids),1+len(stages),figsize=(18,2*len(ids)),squeeze=False)
        for j,sid in enumerate(ids):
            truth=target[sid];states=truth[:,2:].argmax(1)
            arrays={'target':truth};arrays.update({f'step {step}':np.load(directory/f'step-{step}/{sid}/mu.npy') for step in stages})
            allxy=np.concatenate([a[:,:2] for a in arrays.values()]);lo=allxy.min(0);hi=allxy.max(0)
            for ax,(label,a) in zip(axes[j],arrays.items()):
                draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)))
                ax.set_xlim(lo[0]-.02,hi[0]+.02);ax.set_ylim(lo[1]-.05,hi[1]+.05)
                ax.set_title(f'{sid} — {label}',fontsize=9)
        fig.tight_layout();name=f'{group}-means.png';fig.savefig(out/name,dpi=150);plt.close(fig);images.append((group,name))
    # Each line: input / final mean / median draw / worst draw, no cherry picking.
    final=stages[selected]
    for line in final['lines']:
        sid=line['sample_id'];draws=line['sampled'];rank=sorted(draws,key=lambda v:v['geometry']['x_rmse']+v['geometry']['y_rmse'])
        arrays={'target':target[sid],'mean':np.load(directory/f'step-{selected}/{sid}/mu.npy')}
        for label,variant in [('median sampled z',rank[len(rank)//2]),('worst sampled z',rank[-1])]:
            arrays[label]=np.load(directory/f"step-{selected}/{sid}/{variant['kind']}.npy")
        fig,axes=plt.subplots(4,1,figsize=(14,5));allxy=np.concatenate([a[:,:2] for a in arrays.values()]);lo=allxy.min(0);hi=allxy.max(0)
        for ax,(label,a) in zip(axes,arrays.items()):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_title(label,fontsize=9)
            ax.set_xlim(lo[0]-.02,hi[0]+.02);ax.set_ylim(lo[1]-.05,hi[1]+.05)
        fig.tight_layout();fig.savefig(out/f'{sid}.png',dpi=140);plt.close(fig)
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),
             ('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(3,2,figsize=(8,9),squeeze=False)
    for column,(sid,xlim,ylim,label) in enumerate(regions):
        arrays=[target[sid],np.load(directory/f'step-{initial_step}/{sid}/mu.npy'),np.load(directory/f'step-{selected}/{sid}/mu.npy')]
        for ax,a,heading in zip(axes[:,column],arrays,['target','source checkpoint','expanded checkpoint']):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim)
            ax.set_title(label+' — '+heading,fontsize=9)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=160);plt.close(fig)
    summary=dict(completed=result is not None,provenance=provenance,selected_step=selected,selection='training-only; no held-out model selection',
                 baseline=stages[initial_step]['groups'],final=final['groups'],result=result)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    page=['<meta charset="utf-8"><title>Seen-writer reconstruction expansion</title><h1>24-line seen-writer expansion</h1>',
          '<p>Eight original + sixteen new training lines; four evaluation-only lines. Forms overlap: this is NOT a writer/form-independent benchmark. No held-out gradients, calibration, OCR warmup or checkpoint selection. Marker-free polylines use predicted pens. Target is IAM/RDP, not raw pen points. Coordinate scale .01, dropout0, microbatch1/minimal padding. OCR frozen/off; CER is a transfer diagnostic, not OCR training performance.</p>',
          '<p>Differences are nonuniform point-index differences, NOT physical velocity/curvature. Angle errors are spatial tangent/turn differences. All sampled draws count in metrics; gallery shows median/worst by X+Y RMSE.</p>',
          '<p><a href="summary.json">Metrics/config/provenance</a> | <a href="provenance.json">Exact sample hashes/splits</a></p>',
          '<table border="1"><tr><th>Group / step</th><th>Mean X/Y RMSE</th><th>Δ/Δ² relative</th><th>Turn p90</th><th>Pen F1</th><th>Sampled X/Y RMSE</th><th>Frozen OCR CER</th></tr>']
    for step,row in stages.items():
        for group in ('old','new','held_out'):
            stats=row['groups'][group];g=stats['mu'];z=stats['sampled']
            page.append(f'<tr><td>{group}/{step}</td><td>{g["mean_per_line_x_rmse"]:.5f}/{g["mean_per_line_y_rmse"]:.5f}</td><td>{g["mean_per_line_first_difference_relative"]:.3f}/{g["mean_per_line_second_difference_relative"]:.3f}</td><td>{g["turn_angle_error_degrees"]["p90"]:.2f}°</td><td>{stats["mu_macro_pen_f1"]:.4f}</td><td>{z["mean_per_line_x_rmse"]:.5f}/{z["mean_per_line_y_rmse"]:.5f}</td><td>{stats["mu_cer"]:.2%}</td></tr>')
    page.append('</table>')
    if result is None:page.append('<p><strong>PARTIAL / interrupted run. These are the last saved evaluations, not a completed training result.</strong></p>')
    page.append('<h2>Original curve-fidelity retention</h2><img style="max-width:100%" src="user-regions.png">')
    for group,name in images:page.append(f'<h2>{group}: target / before / after</h2><img style="max-width:100%" src="{name}">')
    for sid in sorted(target):page.append(f'<h2>{sid} ({groups[sid]})</h2><p>{html.escape(provenance["samples"][sid]["text"])}</p><img style="max-width:100%" src="{sid}.png">')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'index.html').write_text('\n'.join(page));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--data-root',default='data');a=p.parse_args()
    print(json.dumps(report(a.directory,a.data_root)['final'],indent=2))
