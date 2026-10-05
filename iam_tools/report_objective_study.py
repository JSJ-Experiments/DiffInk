"""CPU-only comparison report for existing objective study artifacts."""
import argparse
import html
import json
from pathlib import Path
import statistics


def summarize(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    fixed=[json.loads(line) for line in (directory/'fixed_metrics.jsonl').read_text().splitlines()]
    logs=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert len(logs)==result['optimizer_updates']
    assert all(row['microbatches']==8 and row['loss_log_scope']=='effective_batch_mean' for row in logs)
    assert all(len(row['lines'])==8 and all(len(line['sampled_z'])==20 for line in row['lines']) for row in fixed)
    assert result['auxiliary_state_unchanged'] and result['source_unchanged']
    final=fixed[-1]; lines=[]
    for line in final['lines']:
        pen=line['mu']['pen']; axes=line['mu']['geometry']['axes']
        lines.append(dict(id=line['sample_id'], x=axes['x']['rmse_model_units'],y=axes['y']['rmse_model_units'],
                          f1=pen['pen_up_f1'],fp=pen['pen_up_fp'],fn=pen['pen_up_fn'],
                          false_eoc=pen['non_final_false_eoc_count'],final_eoc=pen['final_eoc_correct'],
                          sampled_xy=line['xy_sampled_summary'],sampled_f1=line['pen_f1_sampled'],
                          posterior_std_median=line['latent_std_median'],posterior_std_mean=line['latent_std_mean']))
    summary=dict(arm=result['arm'],updates=len(logs),mean_x=final['mu_mean_x_rmse'],mean_y=final['mu_mean_y_rmse'],
                 macro_f1=final['mu_macro_pen_f1'], final_eoc_correct=sum(v['final_eoc'] for v in lines),
                 false_internal_eoc=sum(v['false_eoc'] for v in lines),lines=lines,
                 clip_fraction=sum(r['gradient_norm']>result['config']['grad_clip'] for r in logs)/len(logs),
                 median_raw_grad=statistics.median(r['gradient_norm'] for r in logs),
                 loop_seconds=result['elapsed_seconds'],not_billed_duration=True,
                 source_sha256=result['source_sha256'],training_samples_only=True,
                 visual_review_required=True)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,fixed


def report(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    loaded=[summarize(path) for path in directories]
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for summary,fixed in loaded:
        for ax,key,title in zip(axes,['mu_mean_x_rmse','mu_mean_y_rmse','mu_macro_pen_f1'],['X RMSE','Y RMSE','Pen-up F1']):
            ax.plot([r['step'] for r in fixed],[r[key] for r in fixed],label=summary['arm'],marker='o');ax.set_title(title);ax.set_xlabel('Additional updates');ax.legend()
    fig.tight_layout();fig.savefig(output/'curves.png');plt.close(fig)
    summaries=[v[0] for v in loaded];(output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    page=['<!doctype html><meta charset="utf-8"><h1>Eight-line objective comparison</h1>',
          '<p>Same step-200 source, sampled latent, dropout 0, minimal padding, batch1/accum8. XY weight 100, pen weight 1; only GMM coefficient 0 versus 1. LR 5e-5, reduced to 1e-5 after 80%. Fresh identical Adam. CTC/style/KL off. Training samples only, not English reproduction.</p>',
          '<img width="100%" src="curves.png"><p>Each checkpoint includes 20 fixed-seeded z samples per line. Curves are posterior-mean readout metrics, not NLL.</p>']
    for directory,(summary,fixed) in zip(directories,loaded):
        # Reports normally sit at the common study parent on the Volume.
        import os
        rel=os.path.relpath(directory,output)
        page.append(f'<h2>{html.escape(summary["arm"])}</h2><p>X/Y {summary["mean_x"]:.5f}/{summary["mean_y"]:.5f}; F1 {summary["macro_f1"]:.4f}; final EOC {summary["final_eoc_correct"]}/8, false EOC {summary["false_internal_eoc"]}; clipping {summary["clip_fraction"]:.1%}.</p>')
        page.append(f'<p><a href="{rel}/index.html">Full checkpoint galleries</a> | <a href="{rel}/gradient_diagnostics.json">Decoder gradient diagnostics</a></p>')
        for line in summary['lines']:
            page.append(f'<p>{html.escape(line["id"])} X/Y {line["x"]:.5f}/{line["y"]:.5f}, F1 {line["f1"]:.4f}, FP/FN {line["fp"]}/{line["fn"]}, false EOC {line["false_eoc"]}, final EOC {line["final_eoc"]}</p>')
            page.append(f'<img width="100%" src="{rel}/step-{summary["updates"]}/{line["id"]}/mu-comparison.png">')
    (output/'index.html').write_text('\n'.join(page))
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(report(a.directories,a.output),indent=2))
