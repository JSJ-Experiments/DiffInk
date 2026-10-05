"""CPU-only raw/RDP/reconstruction audit; no smoothing, resampling, or training."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import h5py
import numpy as np

from .preprocess import convert


def curve_metrics(prediction, target, states):
    """Compare identical point indices, excluding pen-up derivative windows.

    Differences are not physical velocity or arc-length-normalized curvature:
    RDP leaves uneven point spacing. Relative errors compare RMS vector signals.
    """
    p, t = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    s = np.asarray(states)
    if p.shape != t.shape or t.shape != (len(s), 2) or not len(s):
        raise ValueError('matching nonempty N x 2 coordinates and states required')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isin(s, [0, 1, 2]).all():
        raise ValueError('invalid coordinates or states')
    residual = p - t
    distance = np.linalg.norm(residual, axis=1)
    result = dict(x_rmse=float(np.sqrt(np.mean(residual[:, 0]**2))),
                  y_rmse=float(np.sqrt(np.mean(residual[:, 1]**2))),
                  point_distance_p95=float(np.quantile(distance, .95)),
                  point_distance_max=float(distance.max()))
    connected = s[:-1] == 0
    for order, name, mask in [(1, 'first_difference', connected),
                              (2, 'second_difference', connected[:-1] & connected[1:])]:
        error = np.diff(residual, n=order, axis=0)[mask]
        signal = np.diff(t, n=order, axis=0)[mask]
        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1)))) if len(error) else None
        baseline = float(np.sqrt(np.mean(np.sum(signal**2, axis=1)))) if len(signal) else None
        result[name] = dict(windows=len(error), vector_rmse=rms, target_vector_rms=baseline,
                            relative_rms_error=rms/baseline if baseline and baseline > 1e-12 else None)
    return result


def split_xy(xy, states):
    start = 0
    for end in range(len(xy)):
        if states[end] != 0 or end == len(xy)-1:
            yield xy[start:end+1]
            start = end+1


def draw(ax, strokes, color='black', label=None):
    for i, stroke in enumerate(strokes):
        ax.plot(stroke[:, 0], stroke[:, 1], color=color, linewidth=1.0,
                solid_capstyle='round', solid_joinstyle='round', label=label if i == 0 else None)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def report(data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(data_root)
    source = root/'checkpoints/iam_lbfgs_geometry/20261005-113159/pen_refit'
    output = root/'checkpoints/iam_objective_study/research-summary/curve-audit'
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((source/'result.json').read_text())
    scale = .01
    lines, loaded = [], {}
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5', 'r') as dataset:
        for line in result['final']['lines']:
            sid = line['sample_id']
            canonical_path = root/f'canonical/iam/overfit/{sid}.json'
            canonical = json.loads(canonical_path.read_text())
            target, _, raw, simplified, metadata = convert(canonical)
            if not np.array_equal(target, dataset[sid]['point_seq'][:]):
                raise AssertionError('canonical conversion differs from saved HDF5')
            path = source/f'step-3000/{sid}/mu.npy'
            prediction = np.load(path)
            if prediction.shape != target.shape:
                raise AssertionError('prediction/target shape mismatch')
            states = target[:, 2:].argmax(1)
            if not np.array_equal(prediction[:, 2:].argmax(1), states):
                raise AssertionError('this audit requires exact pen boundaries')
            row = dict(sample_id=sid, text=canonical['text'], raw_points=metadata['raw_points'],
                       processed_points=len(target), rdp_epsilon_model_units=.5*scale,
                       prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       canonical_sha256=hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
                       metrics=curve_metrics(prediction[:, :2], target[:, :2]*scale, states))
            lines.append(row)
            loaded[sid] = (raw, simplified, prediction[:, :2], states, canonical['text'], target[:, :2]*scale)
    # User-identified regions, chosen for the observed failure, not for good scores.
    regions = [('p08-936z-05', 12, (4.94, 5.34), (.15, .57), 'Left: c in chocolate'),
               ('a07-421z-02', 23, (8.28, 8.75), (.40, .97), 'Right: h in hope')]
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharex='col', sharey='col')
    focuses = []
    for column, (sid, stroke_index, xlim, ylim, heading) in enumerate(regions):
        raw, simplified, xy, states, _, target = loaded[sid]
        pred_strokes = list(split_xy(xy, states))
        raw_xy = [s[:, :2]*scale for s in raw]
        rdp_xy = [s[:, :2]*scale for s in simplified]
        panels = [('Raw IAM (normalized)', raw_xy), ('RDP target (old report input)', rdp_xy),
                  ('Reconstruction, true pen', pred_strokes)]
        for row, (label, strokes) in enumerate(panels):
            ax = axes[row, column]; draw(ax, strokes)
            ax.set_title(heading+'\n'+label, fontsize=10)
        draw(axes[3, column], rdp_xy, '#2185c5', 'RDP target')
        draw(axes[3, column], pred_strokes, '#e06c2b', 'reconstruction')
        axes[3, column].legend(loc='upper right', fontsize=8)
        axes[3, column].set_title('Overlay: no markers, no smoothing', fontsize=10)
        for ax in axes[:, column]:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        focus_target = rdp_xy[stroke_index]
        focus_states = np.zeros(len(focus_target), dtype=int); focus_states[-1] = 1
        focuses.append(dict(sample_id=sid, stroke_index_zero_based=stroke_index,
                            metrics=curve_metrics(pred_strokes[stroke_index], focus_target, focus_states)))
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'user-regions.{suffix}', dpi=180)
    plt.close(fig)
    for sid, *_ in regions:
        raw, simplified, xy, states, text, _ = loaded[sid]
        fig, axes = plt.subplots(3, 1, figsize=(17, 5), sharex=True, sharey=True)
        for ax, strokes, label in zip(axes, [[s[:, :2]*scale for s in raw],
                                            [s[:, :2]*scale for s in simplified], list(split_xy(xy, states))],
                                     ['raw IAM', 'RDP target', 'reconstruction + true pen']):
            draw(ax, strokes); ax.set_title(label)
        fig.suptitle(text); fig.tight_layout(); fig.savefig(output/f'{sid}.png', dpi=160); plt.close(fig)
    audit = dict(training=False, gpu=False, new_optimizer_steps=0,
                 units='model units; full line height=1, data coordinates scaled by 0.01',
                 derivatives='point-index differences within true strokes, NOT geometric curvature',
                 rendered='latent mean + mixture expectation; no markers, smoothing, or resampling',
                 all_eight_training_lines_audited=True, pen_boundaries_exact=True,
                 visual_realism_gate_passed=False, lines=lines, user_regions=focuses)
    (output/'metrics.json').write_text(json.dumps(audit, indent=2)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><title>Curve audit</title>', '<h1>Raw IAM → RDP target → reconstruction</h1>',
            '<p>The original report input was already RDP simplified, not raw IAM. Marker-free equal-scale crops show real residual shape distortion; renderer dots accentuate it but do not cause it. No smoothing or new training.</p>',
            '<p>Pointwise MSE constrains positions, not neighboring directions. Perfect pen/OCR mechanics do not establish visual realism. Second differences below are index differences, not physical/geometric curvature; RDP spacing is uneven.</p>',
            '<img src="user-regions.png" style="max-width:100%"><p><a href="user-regions.svg">Vector image</a> | <a href="metrics.json">All eight lines and region metrics</a></p>',
            '<table><tr><th>Line</th><th>X/Y RMSE</th><th>Point error p95/max</th><th>First-difference relative RMS error</th><th>Second-difference relative RMS error</th></tr>']
    for row in lines:
        m = row['metrics']
        page.append(f'<tr><td>{html.escape(row["sample_id"])}</td><td>{m["x_rmse"]:.5f}/{m["y_rmse"]:.5f}</td>'
                    f'<td>{m["point_distance_p95"]:.5f}/{m["point_distance_max"]:.5f}</td>'
                    f'<td>{m["first_difference"]["relative_rms_error"]:.1%}</td><td>{m["second_difference"]["relative_rms_error"]:.1%}</td></tr>')
    page.append('</table><p>Next gate: reduce residual curve distortion, with a controlled target-derivative matching diagnostic if needed. Do not hide it with display smoothing or promote joint KL/CTC training on OCR/pen metrics alone.</p>')
    for sid, *_ in regions:
        page.append(f'<h2>{sid}</h2><img src="{sid}.png" style="max-width:100%">')
    (output/'index.html').write_text('\n'.join(page))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    args = parser.parse_args()
    print(json.dumps(report(args.data_root), indent=2))
