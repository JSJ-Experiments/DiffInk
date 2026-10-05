"""Marker-free all-line and user-region comparisons for completed curve studies."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from .preprocess import convert
from .curve_audit import draw, split_xy
from .trajectory_geometry import geometry_metrics


def aggregate(rows):
    result = {}
    for key in ('x_rmse', 'y_rmse', 'point_distance_p95', 'point_distance_max'):
        result['mean_per_line_'+key] = float(np.mean([r['geometry'][key] for r in rows]))
    for key in ('first_difference', 'second_difference'):
        result['mean_per_line_'+key+'_relative'] = float(np.mean([r['geometry'][key]['relative_rms_error'] for r in rows]))
    for key in ('tangent_angle_error_degrees', 'turn_angle_error_degrees', 'target_corner_turn_error_degrees', 'target_shallow_turn_error_degrees'):
        result[key] = {q: float(np.mean([r['geometry'][key][q] for r in rows if r['geometry'][key][q] is not None]))
                       for q in ('median', 'p90', 'p99')}
    if all('symmetric_vertex_to_polyline_distance' in r['geometry'] for r in rows):
        result['symmetric_vertex_to_polyline_distance'] = {q: float(np.mean([r['geometry']['symmetric_vertex_to_polyline_distance'][q] for r in rows])) for q in ('median', 'p90', 'p99', 'max')}
    return result


def report(study, data_root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, study = Path(data_root), Path(study)
    info = json.loads((study/'study.json').read_text())
    arms = list(info['outputs'])
    arm_paths = {}
    for a in arms:
        saved = Path(info['outputs'][a]['output'])
        arm_paths[a] = root/saved.relative_to('/data') if saved.is_relative_to('/data') else saved
    results = {a: json.loads((arm_paths[a]/'result.json').read_text()) for a in arms}
    source = root/'checkpoints/iam_lbfgs_geometry/20261005-113159/pen_refit/step-3000'
    out = study/'report'; out.mkdir(exist_ok=True)
    loaded, metrics = {}, {a: [] for a in ['before']+arms}
    for line in results[arms[0]]['final']['lines']:
        sid = line['sample_id']; sample = json.loads((root/f'canonical/iam/overfit/{sid}.json').read_text())
        target, _, raw, simplified, _ = convert(sample)
        target_xy = target[:, :2]*.01; states = target[:, 2:].argmax(1)
        arrays = {'before': np.load(source/sid/'mu.npy')}
        for a in arms:
            step = results[a]['final']['step']; arrays[a] = np.load(arm_paths[a]/f'step-{step}'/sid/'mu.npy')
        loaded[sid] = dict(raw=[s[:, :2]*.01 for s in raw], rdp=[s[:, :2]*.01 for s in simplified],
                           target=target_xy, states=states, arrays=arrays, text=sample['text'])
        for a, array in arrays.items():
            metrics[a].append(dict(id=sid, geometry=geometry_metrics(array[:, :2], target_xy, states)))
        fig, axes = plt.subplots(2+len(arms), 1, figsize=(18, 1.9*(2+len(arms))), sharex=True, sharey=True)
        for ax, a in zip(axes, ['rdp', 'before']+arms):
            strokes = loaded[sid]['rdp'] if a == 'rdp' else list(split_xy(arrays[a][:, :2], states))
            draw(ax, strokes); ax.set_title(a+' | true pen boundaries', fontsize=10)
        fig.suptitle(sample['text']); fig.tight_layout(); fig.savefig(out/f'{sid}.png', dpi=160); plt.close(fig)
    regions = [('p08-936z-05', 12, (4.94, 5.34), (.15, .57), 'c in chocolate'),
               ('a07-421z-02', 23, (8.28, 8.75), (.40, .97), 'h in hope')]
    panel_names = ['raw', 'rdp', 'before']+arms
    fig, axes = plt.subplots(len(panel_names), 2, figsize=(8, 2.7*len(panel_names)), sharex='col', sharey='col')
    focus = {}
    for column, (sid, stroke_index, xlim, ylim, heading) in enumerate(regions):
        entry = loaded[sid]; focus[sid] = {}
        for row, a in enumerate(panel_names):
            strokes = entry[a] if a in ('raw', 'rdp') else list(split_xy(entry['arrays'][a][:, :2], entry['states']))
            ax = axes[row, column]; draw(ax, strokes); ax.set_title(heading+' | '+a)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            if a not in ('raw', 'rdp'):
                s = np.zeros(len(strokes[stroke_index]), dtype=int); s[-1] = 1
                focus[sid][a] = geometry_metrics(strokes[stroke_index], entry['rdp'][stroke_index], s)
    fig.tight_layout(); fig.savefig(out/'user-regions.png', dpi=160); fig.savefig(out/'user-regions.svg'); plt.close(fig)
    summary = dict(source_sha256=info['source_sha256'], metrics_definition='macro averages of per-line statistics; angles are spatial degrees, differences by point index',
                   aggregates={a: aggregate(rows) for a, rows in metrics.items()}, lines=metrics, user_regions=focus,
                   pen={a: dict(macro_f1=results[a]['final']['macro_pen_f1'],
                                all_final_eoc_correct=all(r['mu']['pen']['final_eoc_correct'] for r in results[a]['final']['lines']),
                                false_internal_eoc=sum(r['mu']['pen']['non_final_false_eoc_count'] for r in results[a]['final']['lines'])) for a in arms},
                   configurations={a: results[a]['config'] for a in arms},
                   sampled={a: {l['sample_id']: [s['geometry'] for s in l['sampled']] for l in results[a]['final']['lines']} for a in arms},
                   training_samples_only=True, no_smoothing_resampling=True)
    summary['stages'] = info.get('stages')
    summary['arm_paths'] = {a: str(p) for a, p in arm_paths.items()}
    summary['sampled_aggregates'] = {}
    for a, result in results.items():
        # Stochastic metrics: all twenty samples on all eight lines, not a best draw.
        sampled_rows = [s for l in result['final']['lines'] for s in l['sampled']]
        summary['sampled_aggregates'][a] = aggregate(sampled_rows)
        summary['pen'][a]['sampled_min_f1'] = min(s['pen']['pen_up_f1'] for l in result['final']['lines'] for s in l['sampled'])
        summary['pen'][a]['all_sampled_final_eoc_correct'] = all(s['pen']['final_eoc_correct'] for l in result['final']['lines'] for s in l['sampled'])
        summary['pen'][a]['sampled_false_internal_eoc'] = sum(s['pen']['non_final_false_eoc_count'] for l in result['final']['lines'] for s in l['sampled'])
    # Keep a readable stochastic gallery for the last arm; all metrics still
    # include all draws/arms, and the report never selects a lucky reconstruction.
    a = arms[-1]
    for line in results[a]['final']['lines']:
        sid = line['sample_id']; entry = loaded[sid]
        scores = [s['geometry']['x_rmse']**2+s['geometry']['y_rmse']**2 for s in line['sampled']]
        order = np.argsort(scores); chosen = [int(order[len(order)//2]), int(order[-1])]
        panels = [('RDP target', entry['target']), ('latent mean', entry['arrays'][a][:, :2])]
        for k, label in zip(chosen, ['sampled-z median XY error', 'sampled-z worst XY error']):
            array = np.load(arm_paths[a]/f'step-{results[a]["final"]["step"]}'/sid/f'z-{k}.npy')
            panels.append((label, array[:, :2]))
        fig, axes = plt.subplots(4, 1, figsize=(18, 7.6), sharex=True, sharey=True)
        for ax, (label, xy) in zip(axes, panels): draw(ax, list(split_xy(xy, entry['states']))); ax.set_title(label)
        fig.suptitle(a+' | '+entry['text']); fig.tight_layout(); fig.savefig(out/f'{sid}-sampled.png', dpi=150); plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><title>Controlled curve fidelity study</title>',
            '<h1>Controlled eight-line trajectory reconstruction</h1>',
            '<p>Mean latent + mixture expectation. All galleries are marker-free, share coordinates, and use true pen boundaries to isolate geometry. Pen predictions are separately measured, not forced. No smoothing/resampling/KL/OCR/style/GMM or architecture changes. Eight training lines only.</p>',
            '<p>First/second differences are by point index, not physical velocity/geometric curvature. Tangent angle is segment orientation; turn is signed angle between consecutive segments, wrapped differences, excluding zero-length edges and pen-up. Table angles are mean per-line p90, not pooled p90.</p>',
            '<p><a href="summary.json">Full metrics/configurations/sampled-z metrics</a> | <a href="../study.json">Source and code hashes</a></p>',
            '<table><tr><th>Arm</th><th>X/Y RMSE</th><th>Δ/Δ² relative RMS</th><th>Tangent p90</th><th>Turn p90</th><th>Corner turn p90</th></tr>']
    for a, m in summary['aggregates'].items():
        page.append(f'<tr><td>{a}</td><td>{m["mean_per_line_x_rmse"]:.6f}/{m["mean_per_line_y_rmse"]:.6f}</td>'
                    f'<td>{m["mean_per_line_first_difference_relative"]:.1%}/{m["mean_per_line_second_difference_relative"]:.1%}</td>'
                    f'<td>{m["tangent_angle_error_degrees"]["p90"]:.2f}°</td><td>{m["turn_angle_error_degrees"]["p90"]:.2f}°</td>'
                    f'<td>{m["target_corner_turn_error_degrees"]["p90"]:.2f}°</td></tr>')
    page.append('</table><h2>User-identified curves</h2><img src="user-regions.png" style="max-width:100%">')
    for sid, entry in loaded.items():
        page.append(f'<h2>{html.escape(sid)}: {html.escape(entry["text"])}</h2><img src="{sid}.png" style="max-width:100%">')
    page.append(f'<h2>Sampled latent robustness: {html.escape(arms[-1])}</h2><p>Posterior mean plus median/worst XY-error samples of twenty fixed seeds per line; true pens for geometry isolation. Pen metrics for all 160 draws are in summary.json.</p>')
    for sid in loaded:
        page.append(f'<img src="{sid}-sampled.png" style="max-width:100%">')
    for a, result in results.items():
        page.append(f'<h2>{a} configuration</h2><pre>{html.escape(json.dumps(dict(config=result["config"], calibration=result["calibration"], outer_steps=result["outer_steps"], closure_calls=result["closure_calls"], elapsed_seconds=result["elapsed_seconds"], pen=summary["pen"][a]), indent=2))}</pre>')
    (out/'index.html').write_text('\n'.join(page))
    return summary['aggregates']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('study'); p.add_argument('--data-root', default='data')
    a = p.parse_args(); print(json.dumps(report(a.study, a.data_root), indent=2))
