"""Summarize/downloaded mechanics-run metrics and render a local HTML gallery (CPU only)."""
import argparse
import html
import json
import math
from pathlib import Path
import statistics


def report(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    fixed=[json.loads(x) for x in (directory/'fixed_metrics.jsonl').read_text().splitlines()]
    if len(metrics)!=result['steps'] or [r['step'] for r in metrics]!=list(range(1,result['steps']+1)):
        raise ValueError('incomplete or nonconsecutive training metrics')
    if not all(math.isfinite(v) for row in metrics for v in row.values()):raise ValueError('nonfinite training metrics')
    window=min(20,len(metrics))
    names=['total','gmm','pen','ctc','style','gradient_norm','ocr_gradient_norm','style_gradient_norm']
    summary={'result':result,'metric_rows':len(metrics),'all_training_metrics_finite':True,
             'window':window,'first_window_mean':{k:statistics.mean(r[k] for r in metrics[:window]) for k in names},
             'last_window_mean':{k:statistics.mean(r[k] for r in metrics[-window:]) for k in names},
             'auxiliary_gradients_nonzero_every_step':all(r['ocr_gradient_norm']>0 and r['style_gradient_norm']>0 for r in metrics),
             'fixed':{split:{'initial':next(r for r in fixed if r['split']==split),
                             'final':next(r for r in reversed(fixed) if r['split']==split)} for split in ['train','val']},
             'interpretation':'Numerical mechanics only; decreasing loss is not evidence of legible reconstruction, OCR accuracy, or paper reproduction.'}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for ax,key in zip(axes.flat,['total','gmm','pen','ctc','style','xy_rmse_model_units']):
        if key!='xy_rmse_model_units':ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],alpha=.25,label='shuffled train')
        for split in ['train','val']:
            rows=[r for r in fixed if r['split']==split]
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label='fixed '+split)
        ax.set_title(key);ax.set_xlabel('optimizer step');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(directory/'loss-curves.png',dpi=130);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>Experimental InkVAE T4 mechanics run</title>',
          '<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{max-width:100%}table{width:100%}td{width:50%}</style>',
          '<h1>Experimental InkVAE T4 mechanics run</h1>',
          '<p>Not a paper reproduction. Inspect geometry, pen states and OCR separately; losses alone do not prove success.</p>',
          '<img src="loss-curves.png"><table><tr><th>Input</th><th>Reconstruction (highest-weight GMM mean)</th></tr>']
    for row in fixed:
        name=f"{row['split']}-step-{row['step']}"
        page.append(f'<tr><td colspan="2"><h3>{html.escape(name)}: {html.escape(row["text"])}</h3>Total={row["total"]:.3f}; XY RMSE={row["xy_rmse_model_units"]:.3f}; CER={row["cer"]:.3f}; writer correct={row["style_correct"]}</td></tr>')
        page.append(f'<tr><td><img src="{name}-input.png"></td><td><img src="{name}-prediction.png"></td></tr>')
    page.append('</table>');(directory/'index.html').write_text('\n'.join(page))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    print(json.dumps(report(parser.parse_args().directory),indent=2))
if __name__=='__main__':main()
