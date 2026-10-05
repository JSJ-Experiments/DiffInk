"""Rebuild the completed 2026-10-05 curve investigation overview on CPU."""
import argparse
import html
import json
from pathlib import Path
from .report_curve_study import report as stage_report


def report(data_root='data'):
    root = Path(data_root)/'checkpoints/iam_curve_study'
    first = stage_report(root/'comparison-80', data_root)
    refined = stage_report(root/'comparison-180', data_root)
    out = root/'research-summary'; out.mkdir(exist_ok=True)
    verified = json.loads((root/'cpu-verification.json').read_text())
    summary = dict(original_reference='700f84eda8b523917f0c7f337bc21f75cc6d4075a101edc979fa5432212cd4f8',
                   stage1=first, refinement=refined,
                   selected_checkpoint='/data/checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt',
                   selected_sha256=verified['delta50']['checkpoint_sha256'],
                   visual_judgment='eight memorized training-line geometry is good enough for controlled joint integration, not full-IAM readiness',
                   all_eight_mean_renders_inspected=True, sampled_median_worst_named_lines_inspected=True,
                   training_samples_only=True, architecture_unchanged=True, kl_ocr_style_not_trained=True,
                   limitations=['sampled latents remain less precise than means', 'no held-out/generalization result',
                                'fresh LBFGS diagnostics do not establish Adam production convergence',
                                'GMM likelihood/sigma output generation not trained in these arms',
                                'joint pen/OCR/KL stability and new-latent OCR compatibility remain unvalidated'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><title>English InkVAE curve fidelity investigation</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em}td,th{padding:.5em;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}</style>',
            '<h1>Curve fidelity: investigation and improved reconstruction</h1>',
            '<p><strong>Result:</strong> The old checkpoint was under-converged. Continued point supervision removes most artifacts; a small target-difference anchor adds a consistent local-fidelity benefit. No decoder/normalization/mixture-count/RDP change, smoothing, or KL/OCR/style training was required.</p>',
            '<p>The final highlighted c no longer has the added shelf, and the h arch tracks its target rather than forming an exaggerated roof. All eight final mean line renders were inspected. This is a visual engineering judgment on eight memorized lines, not pixel identity or generalization.</p>',
            '<p><a href="../comparison-80/report/index.html">Initial matched 80-step arms + all lines</a> | <a href="../comparison-180/report/index.html">Deterministic 100-step refinement + all lines + sampled-z gallery</a> | <a href="summary.json">Overview JSON</a> | <a href="../cpu-verification.json">Independent checkpoint verification</a> | <a href="../cpu-preflight.json">Spacing/phase diagnostics</a> | <a href="../readout-diagnostic.json">Readout test</a> | <a href="../attempts.json">Attempt/preemption history</a></p>',
            '<h2>Mean reconstruction metrics</h2><p>RMSE in model units (full normalized line height=1). Δ/Δ² are within-stroke index differences, NOT physical velocity/geometric curvature. Tangent/turn angles are spatial angles. Table p90 is the mean of per-line p90 values, not a pooled percentile.</p>',
            '<table><tr><th>Arm</th><th>X / Y RMSE</th><th>Δ / Δ² relative RMS</th><th>Tangent p90</th><th>Turn p90</th><th>True corner turn p90</th></tr>']
    entries = [('Original', first['before']), ('Point +80', first['point']),
               ('Point + delta20 +80', first['delta20']), ('Point + tangent20 +80', first['tangent20']),
               ('Point refinement +100', refined['point180']), ('Delta50 refinement +100 (selected)', refined['delta50-refinement'])]
    for label, m in entries:
        page.append(f'<tr><td>{html.escape(label)}</td><td>{m["mean_per_line_x_rmse"]:.6f} / {m["mean_per_line_y_rmse"]:.6f}</td>'
                    f'<td>{m["mean_per_line_first_difference_relative"]:.2%} / {m["mean_per_line_second_difference_relative"]:.2%}</td>'
                    f'<td>{m["tangent_angle_error_degrees"]["p90"]:.2f}°</td><td>{m["turn_angle_error_degrees"]["p90"]:.2f}°</td><td>{m["target_corner_turn_error_degrees"]["p90"]:.2f}°</td></tr>')
    page.extend(['</table><h2>Marker-free, same-scale named curves</h2>',
                 '<img src="../comparison-180/report/user-regions.png">',
                 '<h2>What the evidence supports</h2><ul><li>Under-convergence was the dominant contributor; point-only refinement already improves dramatically.</li><li>Local segment mismatch matters beyond point RMSE. Difference matching improves all eight lines versus its matched refinement control, including true target corners rather than rounding them away.</li><li>Short segments amplify angular error. Unit-tangent matching helped corner-angle metrics but did not beat raw difference matching overall; it is not promoted as the preferred objective.</li><li>No coherent mod-8 residual bias: observed phase-explained variance 0.087%, versus permutation-null p95 0.296%. This is not a proof excluding every possible upsampler effect.</li><li>High-mixture-entropy points did not have larger errors in the CPU check. A frozen-feature affine replacement readout was much worse (X/Y ≈0.100/0.034 versus 0.00166/0.00401). Neither justified removing the mixture head.</li><li>Raw IAM/RDP polygons and renderer markers explain some appearance, but not the added kinks. All improved plots retain actual target hooks/corners, no display smoothing.</li></ul>',
                 '<h2>Exact experiment controls</h2><p>Same eight samples, source hashes, minimal physical batch1 padding; each objective evaluates all eight lines per closure. Decoder dropout0, scale0.01, no rotation. Mean latent + mixture expectation. Geometry optimizer fresh LBFGS for each arm, LR1, history10, max_iter10, max_eval15, strong_wolfe, tolerance_grad1e-8, tolerance_change1e-12, no clipping/weight decay. Starting from model weights only, NOT continuing old LBFGS history. Heads for OCR/style/logvar and pen/sigma/rho output rows frozen during geometry. Pen features subsequently refitted with identical cached-feature AdamW2000, LR0.001, betas0.9/0.99, decay0, clip5, bounded focal gamma2 cap8; geometry remains exactly unchanged.</p>',
                 '<p>Initial80-step arms: point coefficient1; delta weight0.1077067866 (20% initial decoder-gradient norm), or tangent weight0.000034808287 (20%). Refinement: both start from the completed point80 checkpoint17a102… with fresh LBFGS; point coefficient1, delta weight0.2047183874 (50% initial norm). Refinement enables deterministic GPU operations. All coefficients fixed after initial matched aggregate-gradient calibration, not arbitrary large smoothing weights.</p>',
                 '<p>Initial attached derivative arm was canceled after60 and excluded; rerun from original source completed. One refinement attempt was preempted after20; Modal restarted from source, and the completed attempt is used. Intermediate checkpoints/partial runs and attempt history preserved. Do not count partial attempts as final results.</p>',
                 '<h2>Sampled latents and readiness</h2><p>Final all160 sampled reconstructions: mean X/Y RMSE0.002884/0.002036, mean per-draw turn p90 9.14°. All true pen predictions F1=1, all final EOCs correct, no internal false EOC; no forced stops. The median/worst gallery is linked above. Small residual errors and sampled-vs-mean gap remain.</p>',
                 '<p><strong>Ready for a bounded joint integration test on these eight lines</strong>, not a full run. Preserve geometry and sample robustness gates, restore objectives individually, and validate OCR against the changed encoder rather than assuming the old frozen-head CER0 transfers. No KL/OCR/style was enabled in this investigation. Production Adam/GMM behavior, broader IAM/generalization, variable-batch padding, and paper normalization/EOC semantics are unresolved.</p>',
                 '<h2>Selected checkpoint</h2><pre>'+html.escape(summary['selected_checkpoint']+'\nSHA256 '+summary['selected_sha256'])+'</pre>'])
    (out/'index.html').write_text('\n'.join(page))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root', default='data'); a = p.parse_args()
    print(json.dumps(report(a.data_root), indent=2))
