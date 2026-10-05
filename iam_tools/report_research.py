"""Build the completed research-session report from saved artifacts, CPU only."""
import argparse
import html
import json
import os
from pathlib import Path


def report(data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(data_root)/'checkpoints'
    output=root/'iam_objective_study/research-summary';output.mkdir(parents=True,exist_ok=True)
    refit=root/'iam_lbfgs_geometry/20261005-113159/pen_refit'
    ctc=root/'iam_ctc_head_ab/20261005-114305'
    r=json.loads((refit/'result.json').read_text());c=json.loads((ctc/'result.json').read_text())
    assert r['geometry_state_exactly_unchanged'] and not r['forced_final_eoc']
    final=r['final'];assert len(final['lines'])==8
    assert all(line['mu']['pen']['pen_up_f1']==1 and line['mu']['pen']['final_eoc_correct'] for line in final['lines'])
    summary=dict(mean_x_rmse=final['mu_mean_x_rmse'],mean_y_rmse=final['mu_mean_y_rmse'],
                 macro_pen_f1=final['mu_macro_pen_f1'],
                 sampled_median_x_mean=sum(line['xy_sampled_summary']['x']['median'] for line in final['lines'])/8,
                 sampled_median_y_mean=sum(line['xy_sampled_summary']['y']['median'] for line in final['lines'])/8,
                 all_160_sampled_pen_f1_perfect=all(v['pen']['pen_up_f1']==1 for line in final['lines'] for v in line['sampled_z']),
                 ctc={arm:dict(cer=value['final']['character_weighted_cer'],exact_lines=value['final']['exact_lines']) for arm,value in c['arms'].items()},
                 training_samples_only=True,joint_vae_ctc_training=False,kl_style_not_enabled=True,
                 visual_realism_gate_passed=False,
                 next_stage='diagnose/reduce residual curve distortion before promoting joint KL/CTC training',
                 reconstruction_checkpoint=str(refit/'checkpoint.pt'),ctc_zero_checkpoint=str(ctc/'blank_zero/checkpoint.pt'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,3))
    for arm in c['arms']:
        rows=[json.loads(line) for line in (ctc/arm/'fixed_metrics.jsonl').read_text().splitlines()]
        ax.plot([row['step'] for row in rows],[row['character_weighted_cer'] for row in rows],marker='o',label=arm)
    ax.set_xlabel('OCR-head optimizer updates');ax.set_ylabel('Training CER');ax.legend();fig.tight_layout();fig.savefig(output/'ctc-curves.png');plt.close(fig)
    page=['<!doctype html><meta charset="utf-8"><title>English DiffInk research results</title>',
          '<h1>Eight-line reconstruction and frozen CTC-head mechanics</h1>',
          '<p><strong>Later follow-up:</strong> <a href="../../iam_curve_study/research-summary/index.html">Curve investigation and improved checkpoints</a>. The geometry/status below pertains to the older checkpoint, not the improved follow-up.</p>',
          f'<p>Final posterior-mean X/Y RMSE {summary["mean_x_rmse"]:.5f}/{summary["mean_y_rmse"]:.5f}; genuine pen-up F1 1.000 on all eight lines, correct final EOC, no false internal EOCs. Twenty sampled-z pen predictions per line also all F1=1.0.</p>',
          f'<p>Mean per-line sampled median X/Y error {summary["sampled_median_x_mean"]:.5f}/{summary["sampled_median_y_mean"]:.5f}. No forced stopping or smoothing loss. Still some letter-shape error; not pixel identity.</p>',
          '<p><strong>Visual curve fidelity is NOT passed.</strong> Perfect pen/OCR metrics do not establish smooth, faithful geometry. The panels labeled input show the RDP target, not raw IAM. <a href="curve-audit/index.html">Raw IAM / RDP / marker-free reconstruction audit, including the two user-identified regions</a>.</p>',
          '<p>These are all eight memorized training lines from one writer (514–581 points), not a best-render subset, held-out/generalization or paper-English reproduction. The dataset itself was deliberately selected by writer/length. Normalization/final-only EOC remain experimental. KL/style/InkDiT/full-IAM not trained here. CTC is OCR-head-only, not a joint VAE objective.</p>',
          '<h2>Experiment sequence</h2><ol><li>Same step200 source: direct XY+pen versus GMM+strong XY+pen; 1,000 Adam updates each.</li><li>Resume both model+Adam+RNG for 2,000 more updates; direct XY wins geometry/boundaries.</li><li>Multi-knob deterministic geometry-only L-BFGS: 50 outer steps /593 eight-line closures.</li><li>CPU frozen-feature pen-head refit: 3,000 updates, no geometry change.</li><li>Frozen cached-latent OCR-head blank-bias A/B: 1,000 updates each, both CER0/all8 exact transcripts.</li></ol>',
          '<p>GPU loop times sum approximately 39 GPU-minutes including renders/save/evaluation but excluding startup/loading. This is NOT measured billing/cost. First two remote bootstrap failures had zero updates, were stopped explicitly, and are recorded separately.</p>',
          '<p><a href="../comparison-1000/index.html">1,000-update comparison</a> | <a href="../comparison-3000/index.html">3,000-update comparison</a> | <a href="../gmm_uncertainty_3000.json">GMM uncertainty</a> | <a href="../research_attempts.json">Attempt history</a> | <a href="summary.json">Summary JSON</a></p>',
          '<h2>Fixed geometry, true versus predicted pens</h2>']
    for line in final['lines']:
        rel=os.path.relpath(refit/f'step-3000/{line["sample_id"]}/mu-comparison.png',output)
        page.append(f'<h3>{html.escape(line["sample_id"])}</h3><img width="100%" src="{rel}">')
    page.append('<h2>Frozen CTC head</h2><img src="ctc-curves.png" width="650"><p>Initial blank bias is trainable, not clamped. Minus5 reaches CER0 at500, zero at250; both final CER0. Five repeat-bearing lines, three without adjacent repeats. This tiny test does not establish a universal winning bias.</p>')
    for arm,value in c['arms'].items():
        page.append(f'<h3>{html.escape(arm)}</h3><table><tr><th>Line</th><th>Transcript</th><th>Decoded</th><th>CER</th></tr>')
        for line in value['final']['lines']:
            page.append(f'<tr><td>{html.escape(line["id"])}</td><td>{html.escape(line["truth"])}</td><td>{html.escape(line["decoded"])}</td><td>{line["cer"]:.3f}</td></tr>')
        page.append('</table>')
    page.append('<p>Next: reduce residual curve distortion before joint KL/CTC training. Consider a controlled target-derivative matching diagnostic, not display smoothing. Do not confuse head-only OCR/pen success with visual realism or a validated joint training pipeline.</p>')
    (output/'index.html').write_text('\n'.join(page))
    from .curve_audit import report as curve_report
    curve_report(data_root)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='data');a=p.parse_args()
    print(json.dumps(report(a.data_root),indent=2))
