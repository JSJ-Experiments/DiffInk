"""All-line, marker-free comparison of explicit saved research experiments.

Reports saved evaluations, never performs checkpoint selection on validation.
Different-source followups must be labeled separately from matched ablations.
"""
import html
import json
from pathlib import Path
import numpy as np


def phase_diagnostic(evaluation, ids):
    """Descriptive index-mod8 error, not proof of an upsampler mechanism.

    Average each phase's vector RMSE equally across the specified lines. This
    does not shuffle/detrend residuals or test statistical significance.
    """
    rows = {r['sample_id']: r for r in evaluation['lines']}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= rows.keys():
        raise ValueError('unique nonempty evaluation subset required')
    values = np.array([[p['vector_rmse'] for p in rows[sid]['mu']['geometry']['point_index_mod8']] for sid in ids])
    if values.shape != (len(ids), 8) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('eight finite nonnegative phase errors per line required')
    avg = values.mean(0)
    return dict(sample_ids=ids, mean_per_line_phase_vector_rmse=avg.tolist(),
                largest_to_smallest_mean_phase_ratio=float(avg.max()/max(avg.min(), 1e-30)) if avg.max() else 1.,
                interpretation='descriptive phase imbalance only; no causal or significance claim')


def report(entries, root='data', output='checkpoints/iam_fullset_joint/research-summary'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw, split_xy
    from .reference_retention import REFERENCE
    from .pen_ab import file_sha
    root = Path(root); out = root/output; out.mkdir(parents=True, exist_ok=True)
    if not entries or len(entries) > 8 or len({e['label'] for e in entries}) != len(entries):
        raise ValueError('one to eight uniquely labeled saved experiments required')
    stages = []; splits = None
    for entry in entries:
        directory = root/entry['directory']
        provenance = json.loads((directory/'provenance.json').read_text())
        if splits is not None and provenance['splits'] != splits:
            raise ValueError('comparison requires identical sample splits')
        splits = provenance['splits']
        result_path = directory/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if result:
            step = result['best_step']
        else:
            import torch
            step = torch.load(directory/'checkpoint-best.pt', map_location='cpu', weights_only=True)['optimizer_updates']
        evaluation = json.loads((directory/f'eval-{step}.json').read_text())
        config = json.loads((directory/'config.json').read_text())
        selected = directory/'checkpoint-best.pt'
        stages.append(dict(**entry, selected_step=step, completed=result is not None, result=result,
                           config=config, provenance=provenance, evaluation=evaluation,
                           selected_sha256=file_sha(selected),
                           cpu_reload_check=json.loads((directory/'cpu-reload-check.json').read_text()) if (directory/'cpu-reload-check.json').exists() else None,
                           train_phase_diagnostic=phase_diagnostic(evaluation, splits['train'])))
    target = {}
    for name, ids in [('train', splits['train']), ('val', splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{name}.h5') as hf:
            for sid in ids:
                a = hf[sid]['point_seq'][:].copy(); a[:, :2] *= .01; target[sid] = a
    def arrays(sid, include_reference=False):
        aa = {'IAM/RDP target': target[sid]}
        ref = root/REFERENCE/sid/'mu.npy'
        if include_reference and ref.exists(): aa['faithful 8-line reference'] = np.load(ref)
        for stage in stages:
            p = root/stage['directory']/f"step-{stage['selected_step']}"/sid/'mu.npy'
            a = np.load(p)
            if a.shape != target[sid].shape or not np.isfinite(a).all(): raise ValueError('unaligned/nonfinite trajectories')
            aa[stage['label']] = a
        return aa
    regions = [('p08-936z-05', (4.94, 5.34), (.15, .57), 'c in chocolate'),
               ('a07-421z-02', (8.28, 8.75), (.40, .97), 'h in hope')]
    count = len(arrays(regions[0][0], True))
    fig, axes = plt.subplots(count, 2, figsize=(10, 3*count), squeeze=False)
    for col, (sid, xlim, ylim, label) in enumerate(regions):
        for ax, (heading, a) in zip(axes[:, col], arrays(sid, True).items()):
            draw(ax, split_xy(a[:, :2], a[:, 2:].argmax(1))); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(label+' — '+heading, fontsize=10)
    fig.tight_layout(); fig.savefig(out/'highlighted-curves.png', dpi=150); plt.close(fig)
    images = []
    for group in ('old', 'new', 'held_out'):
        ids = splits[group]; columns = len(arrays(ids[0]))
        fig, axes = plt.subplots(len(ids), columns, figsize=(6*columns, 2.3*len(ids)), squeeze=False)
        for row, sid in enumerate(ids):
            aa = arrays(sid); allxy = np.concatenate([a[:, :2] for a in aa.values()]); lo = allxy.min(0); hi = allxy.max(0)
            for ax, (label, a) in zip(axes[row], aa.items()):
                draw(ax, split_xy(a[:, :2], a[:, 2:].argmax(1))); ax.set_xlim(lo[0]-.02, hi[0]+.02); ax.set_ylim(lo[1]-.05, hi[1]+.05)
                ax.set_title(sid+' — '+label, fontsize=9)
        fig.tight_layout(); name = group+'-all-lines.png'; fig.savefig(out/name, dpi=130); plt.close(fig); images.append((group, name))
    (out/'summary.json').write_text(json.dumps(stages, indent=2)+'\n')
    page = ['<meta charset="utf-8"><title>English geometry followup</title><h1>Expanded-line geometry investigation</h1>',
            '<p>24 training lines + 4 evaluation-only lines, same writer; forms overlap. Not an IAM benchmark. No held-out optimization/calibration/selection. Target pens are ground truth; every reconstruction uses predicted pens, latent mean and mixture expectation. No markers, smoothing, splines or output postprocessing. Every selected checkpoint is chosen using training-only score; partial runs show their last saved evaluation.</p>',
            '<p>A/B share initial source/RNG and full-set optimizer; B adds a small calibrated target-relative segment loss. Channel normalization and a fresh-optimizer continuation are different-source followups, NOT matched A/B arms. Index differences are not physical velocity or curvature. Compare real target corners and short hooks, not only aggregate RMSE.</p>',
            '<p><a href="summary.json">Exact configurations, source hashes, metrics, CPU checks and phase diagnostics</a></p>',
            '<table border="1"><tr><th>Stage/group</th><th>X/Y RMSE</th><th>Δ/Δ² relative</th><th>Turn median/p90/p99</th><th>Corner/shallow p90</th><th>Pen F1</th><th>Sampled X/Y</th></tr>']
    for stage in stages:
        for group in ('old', 'new', 'held_out'):
            s = stage['evaluation']['groups'][group]; g = s['mu']; z = s['sampled']; turn = g['turn_angle_error_degrees']
            page.append(f'<tr><td>{html.escape(stage["label"])}/{group}</td><td>{g["mean_per_line_x_rmse"]:.5f}/{g["mean_per_line_y_rmse"]:.5f}</td><td>{g["mean_per_line_first_difference_relative"]:.3f}/{g["mean_per_line_second_difference_relative"]:.3f}</td><td>{turn["median"]:.2f}/{turn["p90"]:.2f}/{turn["p99"]:.2f}°</td><td>{g["target_corner_turn_error_degrees"]["p90"]:.2f}/{g["target_shallow_turn_error_degrees"]["p90"]:.2f}°</td><td>{s["mu_macro_pen_f1"]:.4f}</td><td>{z["mean_per_line_x_rmse"]:.5f}/{z["mean_per_line_y_rmse"]:.5f}</td></tr>')
    page += ['</table><h2>Highlighted smooth curves and authentic hooks</h2><img style="max-width:100%" src="highlighted-curves.png">']
    for group, name in images: page.append(f'<h2>{group}: every line</h2><img style="max-width:100%" src="{name}">')
    for stage in stages:
        link = Path(stage['directory']).relative_to(Path('checkpoints'))/'report/index.html'
        page.append(f'<p>{html.escape(stage["label"])}: selected local update {stage["selected_step"]}; {"completed bounded run" if stage["completed"] else "PARTIAL / dashboard-interrupted"}. <a href="../../{html.escape(str(link))}">All 20 sampled draws in metrics; median/worst galleries</a></p>')
    (out/'index.html').write_text('\n'.join(page)); return stages
