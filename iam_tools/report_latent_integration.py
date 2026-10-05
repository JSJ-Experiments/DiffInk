"""CPU-only marker-free reference/staged integration and 20-draw galleries."""
import argparse
import html
import json
from pathlib import Path
import numpy as np


def report(directory,data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .preprocess import convert
    from .curve_audit import draw,split_xy
    root=Path(data_root);directory=Path(directory)
    study=json.loads((directory/'study.json').read_text());out=directory/'report';out.mkdir(exist_ok=True)
    reference=json.loads((directory/'reference/eval-0.json').read_text())
    rows={'reference':reference};paths={'reference':directory/'reference/step-0'};results={}
    for stage,entry in study['outputs'].items():
        folder=directory/stage;result=json.loads((folder/'result.json').read_text());results[stage]=result
        step=entry['selected_step'] if entry['selected_step'] is not None else 0
        label=stage if entry['gate_passed'] else stage+' [FAILED; initial shown]'
        rows[label]=json.loads((folder/f'eval-{step}.json').read_text());paths[label]=folder/f'step-{step}'
    stages=list(rows);loaded={};summaries={k:v['summary'] for k,v in rows.items()}
    for line in reference['lines']:
        sid=line['sample_id'];canonical=json.loads((root/f'canonical/iam/overfit/{sid}.json').read_text())
        target,*_=convert(canonical);target[:,:2]*=.01
        arrays={k:np.load(paths[k]/sid/'mu.npy') for k in stages}
        loaded[sid]=(target,arrays,canonical['text'])
        fig,axes=plt.subplots(1+len(stages),1,figsize=(18,1.8*(1+len(stages))),sharex=True,sharey=True)
        for ax,name,array in zip(axes,['RDP target']+stages,[target]+list(arrays.values())):
            draw(ax,list(split_xy(array[:,:2],array[:,2:].argmax(1))));ax.set_title(name+' | predicted pens' if name!='RDP target' else name)
        fig.suptitle(canonical['text']);fig.tight_layout();fig.savefig(out/f'{sid}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(loaded),3,figsize=(30,2*len(loaded)),squeeze=False)
    for row,(sid,(target,arrays,text)) in enumerate(loaded.items()):
        for ax,label,array in zip(axes[row],['target','reference',stages[-1]],[target,arrays['reference'],arrays[stages[-1]]]):
            draw(ax,list(split_xy(array[:,:2],array[:,2:].argmax(1))));ax.set_title(sid+' | '+label,fontsize=10)
    fig.tight_layout();fig.savefig(out/'all-eight-mean.png',dpi=140);plt.close(fig)
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(1+len(stages),2,figsize=(8,2.7*(1+len(stages))),sharex='col',sharey='col')
    for col,(sid,xlim,ylim,label) in enumerate(regions):
        target,arrays,_=loaded[sid]
        for ax,name,array in zip(axes[:,col],['RDP target']+stages,[target]+list(arrays.values())):
            draw(ax,list(split_xy(array[:,:2],array[:,2:].argmax(1))));ax.set_title(label+' | '+name);ax.set_xlim(*xlim);ax.set_ylim(*ylim)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=170);fig.savefig(out/'user-regions.svg');plt.close(fig)
    last=stages[-1]
    for sid,(target,arrays,text) in loaded.items():
        line=next(l for l in rows[last]['lines'] if l['sample_id']==sid)
        scores=[v['geometry']['x_rmse']**2+v['geometry']['y_rmse']**2 for v in line['sampled']];order=np.argsort(scores)
        panels=[('RDP target',target),('reference mean',arrays['reference']),('final mean',arrays[last])]
        for k,label in [(int(order[len(order)//2]),'median sampled XY error'),(int(order[-1]),'worst sampled XY error')]:
            panels.append((label,np.load(paths[last]/sid/f'z-{k}.npy')))
        fig,axes=plt.subplots(5,1,figsize=(18,9),sharex=True,sharey=True)
        for ax,(label,array) in zip(axes,panels):draw(ax,list(split_xy(array[:,:2],array[:,2:].argmax(1))));ax.set_title(label+' | predicted pens')
        fig.suptitle(last+' | '+text);fig.tight_layout();fig.savefig(out/f'{sid}-sampled.png',dpi=150);plt.close(fig)
    summary=dict(source_sha256=study['source_sha256'],stop_reason=study['stop_reason'],summaries=summaries,
                 stages={k:{n:r[n] for n in ('config','calibration','selected_step','selected_checkpoint','selected_sha256','gate_passed','elapsed_seconds','source_unchanged','style_unchanged')} for k,r in results.items()},
                 posterior={k:v['posterior'] for k,v in rows.items()},ocr={k:v.get('ocr') for k,v in rows.items()},
                 training_samples_only=True,geometry_pen_not_forced=True,metrics='macro per-line RMSE/p90; sampled all160 draws; Δ by index, angles geometric degrees')
    if (directory/'ocr_warm/result.json').exists():summary['ocr_warm']=json.loads((directory/'ocr_warm/result.json').read_text())
    control=directory/'cpu-variance-control/summary.json'
    if control.exists():
        data=json.loads(control.read_text());summary['cpu_variance_control']={k:data[k] for k in ('aggregates','pen','selected_sha256','source_sha256','mean_cpu_gpu_checks','posterior','paired_noise_seed','original_variance_control')}
    summary['gradient_clipping']={}
    for stage in results:
        log=[json.loads(l) for l in (directory/stage/'metrics.jsonl').read_text().splitlines()]
        summary['gradient_clipping'][stage]=dict(updates=len(log),fraction_clipped=float(np.mean([l['was_clipped'] for l in log])),median_raw_norm=float(np.median([l['raw_gradient_norm'] for l in log])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Bounded posterior/KL/OCR integration</title><h1>Posterior/KL/OCR integration</h1>',
          '<p>Eight memorized training lines, not held-out/generalization evidence. Original visually faithful mean reference remains immutable. Dropout/augmentation/style/GMM off, scale .01, minimal-padding microbatch1, accumulation8. Direct XY + target Δ, mean anchor, sampled training. Tiny corrected KL and later OCR are explicit separate stages.</p>',
          '<p>Marker-free renders use predicted pen states, not forced target pens. Angles are spatial, differences are by nonuniform point index, not physical velocity/curvature. Sampled metrics cover all 20 draws per line, not a lucky selected draw. Gallery samples are median/worst XY error.</p>',
          '<p><a href="summary.json">Metrics, configurations, hashes, OCR transcripts</a> | <a href="../study.json">Run provenance</a></p>',
          '<table border="1"><tr><th>Stage</th><th>Mean XY RMSE</th><th>Mean turn p90</th><th>Sampled XY RMSE</th><th>Sampled turn p90</th><th>Min pen F1 mean/z</th><th>CER mean/z</th></tr>']
    for k,row in rows.items():
        m=row['summary']['mu'];s=row['summary']['sampled'];ocr=row.get('ocr')
        cer=f'{ocr["mu"]["cer"]:.3%}/{ocr["sampled"]["cer"]:.3%}' if ocr else 'not enabled'
        page.append(f'<tr><td>{html.escape(k)}</td><td>{m["mean_per_line_x_rmse"]:.6f}/{m["mean_per_line_y_rmse"]:.6f}</td><td>{m["turn_angle_error_degrees"]["p90"]:.2f}°</td><td>{s["mean_per_line_x_rmse"]:.6f}/{s["mean_per_line_y_rmse"]:.6f}</td><td>{s["turn_angle_error_degrees"]["p90"]:.2f}°</td><td>{row["summary"]["mu_min_pen_f1"]:.3f}/{row["summary"]["sampled_min_pen_f1"]:.3f}</td><td>{cer}</td></tr>')
    page.append('</table><h2>All eight means</h2><img style="max-width:100%" src="all-eight-mean.png"><h2>Previously rough curves</h2><img style="max-width:100%" src="user-regions.png">')
    for sid in loaded:page.append(f'<h2>{sid}</h2><img style="max-width:100%" src="{sid}.png"><img style="max-width:100%" src="{sid}-sampled.png">')
    if 'cpu_variance_control' in summary:
        page.append('<h2>Paired-noise fixed-variance CPU control</h2><p>Fresh CPU draws, identical epsilons across conditions. Final encoder mean/decoder with original element-wise posterior std also improves local errors; improvement is not solely narrower variance. This is an index-aligned sensitivity proxy, not distribution equivalence. <a href="../cpu-variance-control/summary.json">All160 draws/condition, pen/OCR, reload checks</a></p><pre>'+html.escape(json.dumps(summary['cpu_variance_control']['aggregates'],indent=2))+'</pre>')
    page.append('<h2>Frequent clipping remains</h2><pre>'+html.escape(json.dumps(summary['gradient_clipping'],indent=2))+'</pre><p>Raw norms scale with the100/1000 objective coefficients. Clipping remains frequent; this is not convergence or optimizer-quality proof.</p>')
    for stage,r in results.items():
        page.append('<h2>'+stage+' diagnostics</h2><pre>'+html.escape(json.dumps(dict(config=r['config'],calibration=r['calibration'],history=[dict(step=h['step'],gate=h['gate']) for h in r['history']]),indent=2))+'</pre>')
    page.append('<p>This does not establish generalization, a usable Gaussian prior, GMM sampling quality, or InkDiT readiness. Tiny-KL compatibility is not prior matching. Old frozen OCR CER0 was revalidated/refitted on changed latents; no original Chinese reproduction claim.</p>')
    (out/'index.html').write_text('\n'.join(page))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--data-root',default='data');a=p.parse_args()
    print(json.dumps(report(a.directory,a.data_root)['summaries'],indent=2))
