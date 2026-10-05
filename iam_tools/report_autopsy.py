"""CPU-only mean-location vs uncertainty/clipping report for bounded autopsy runs."""
import argparse
import json
import math
from pathlib import Path
from .autopsy_resume import clipping_summary


def report(directory,parent_dir=None,visual_assessment='not-reviewed'):
    directory=Path(directory);result=json.loads((directory/'result.json').read_text())
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    fixed=[json.loads(x) for x in (directory/'fixed_metrics.jsonl').read_text().splitlines()]
    start=result.get('start_step',0);updates=result.get('additional_steps',result['steps'])
    if len(rows)!=updates or [r['step'] for r in rows]!=list(range(start+1,start+updates+1)):
        raise ValueError('incomplete/consecutive update provenance')
    if not all(math.isfinite(r[k]) for r in rows for k in ['gmm_nll','gradient_norm','kl_report_only']):raise ValueError('nonfinite training metrics')
    if not all(r['auxiliary_gradients_absent'] and r['pen_output_loss_gradient_zero'] for r in rows):raise ValueError('unexpected non-coordinate gradient')
    initial,final=fixed[0],fixed[-1]
    axes=final['diagnostics']['axes'];initial_axes=initial['diagnostics']['axes']
    summary={'result':result,'all_training_metrics_finite':True,'metric_rows':len(rows),
             'no_auxiliary_or_pen_loss_gradients_every_step':True,'initial':initial,'final':final,
             'rmse_reduction_fraction':{k:1-axes[k]['rmse_model_units']/initial_axes[k]['rmse_model_units'] for k in ['x','y']},
             'y_correlation_exceeds_0_97':axes['y']['correlation'] is not None and axes['y']['correlation']>.97,
             'clipping_first_100':clipping_summary(rows[:100],10),'clipping_last_100':clipping_summary(rows[-100:],10),
             'clipping_all':clipping_summary(rows,10),'visual_assessment':visual_assessment,
             'pen_predictions_untrained':True,
             'interpretation':'Judge mean-location errors and true-pen renders independently of NLL and uncertainty. High raw gradient norms alone do not measure Adam parameter update size.'}
    if parent_dir is not None:
        parent_dir=Path(parent_dir)
        parent=[json.loads(x) for x in (parent_dir/'metrics.jsonl').read_text().splitlines()]
        summary['parent_clipping_last_100']=clipping_summary(parent[-100:],10)
        previous=json.loads((parent_dir/'result.json').read_text())
        if previous['steps']!=start:raise ValueError('wrong parent run')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes_plot=plt.subplots(2,3,figsize=(14,7))
    axes_plot[0,0].plot([r['step'] for r in rows],[r['gmm_nll'] for r in rows],alpha=.3,label='training')
    axes_plot[0,0].plot([r['step'] for r in fixed],[r['gmm_nll'] for r in fixed],marker='o',label='fixed eval');axes_plot[0,0].set_title('GMM NLL');axes_plot[0,0].legend()
    for coord in ['x','y']:
        axes_plot[0,1].plot([r['step'] for r in fixed],[r['diagnostics']['axes'][coord]['rmse_model_units'] for r in fixed],marker='o',label=coord)
        axes_plot[0,2].plot([r['step'] for r in fixed],[r['diagnostics']['axes'][coord]['correlation'] for r in fixed],marker='o',label=coord)
        axes_plot[1,0].plot([r['step'] for r in fixed],[r['selected_sigma_'+coord+'_median'] for r in fixed],marker='o',label=coord)
    for ax,title in [(axes_plot[0,1],'RMSE of mean location'),(axes_plot[0,2],'Correlation'),(axes_plot[1,0],'Selected sigma median')]:ax.set_title(title);ax.legend()
    axes_plot[0,2].axhline(.97,color='grey',linestyle='--',linewidth=.8)
    axes_plot[1,1].plot([r['step'] for r in rows],[r['gradient_norm'] for r in rows],alpha=.5)
    axes_plot[1,1].axhline(10,color='red',label='clip cap');axes_plot[1,1].set_title('RAW gradient norm');axes_plot[1,1].legend()
    chunks=[rows[i:i+100] for i in range(0,len(rows),100)]
    axes_plot[1,2].plot([c[-1]['step'] for c in chunks],[clipping_summary(c,10)['fraction'] for c in chunks],marker='o')
    axes_plot[1,2].set_ylim(0,1.05);axes_plot[1,2].set_title('Fraction clipped / 100 updates')
    for ax in axes_plot.flat:ax.set_xlabel('global optimizer step')
    fig.tight_layout();fig.savefig(directory/'geometry-curves.png',dpi=130);plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><h1>Geometry continuation: mean error vs sigma</h1>',
          '<p>True-pen renders diagnose XY only. Pen/CTC/style/KL objectives are OFF; no MSE added.</p>',
          f'<p>Visual assessment: {visual_assessment}</p>',
          '<img style="max-width:100%" src="geometry-curves.png">']
    for row in fixed:
        step=row['step'];xy=row['diagnostics']['axes']
        page.append(f'<h2>Step {step}</h2><p>X/Y RMSE: {xy["x"]["rmse_model_units"]:.4f} / {xy["y"]["rmse_model_units"]:.4f}; Y correlation: {xy["y"]["correlation"]:.4f}; NLL: {row["gmm_nll"]:.3f}</p><img style="max-width:100%" src="train-step-{step}-comparison.png">')
    (directory/'index.html').write_text('\n'.join(page))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--parent-dir')
    parser.add_argument('--visual-assessment',default='not-reviewed',choices=['not-reviewed','partial','recognizable-with-residual-errors','clean'])
    args=parser.parse_args();print(json.dumps(report(args.directory,args.parent_dir,args.visual_assessment),indent=2))
if __name__=='__main__':main()
