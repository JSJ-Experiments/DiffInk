"""Summarize and validate an existing bounded run; never allocate a GPU."""
import argparse
import html
import json
from pathlib import Path
import statistics


def report(directory):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    fixed = [json.loads(line) for line in (directory / 'fixed_metrics.jsonl').read_text().splitlines()]
    updates = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert len(updates) == 200 and all(row['microbatches'] == 8 for row in updates)
    assert [row['step'] for row in fixed] == [0, 50, 100, 150, 200]
    ids = [line['sample_id'] for line in fixed[0]['lines']]
    for evaluation in fixed:
        assert [line['sample_id'] for line in evaluation['lines']] == ids
        for line in evaluation['lines']:
            assert line['sampled_count'] == len(line['sampled_z']) == 20
            for name in ['mu.npy', *[f'z-{i}.npy' for i in range(20)]]:
                assert (directory / f"step-{evaluation['step']}" / line['sample_id'] / name).is_file()
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['optimizer_updates'] == 200
    assert all(torch.isfinite(value).all() for value in checkpoint['model_state_dict'].values())
    assert all(int(state['step']) == 200 for state in checkpoint['optimizer_state_dict']['state'].values())
    cfg = checkpoint['config']
    assert cfg['model_input_scale'] == .01 and cfg['trans_dropout'] == 0
    assert cfg['use_decoder_padding_mask'] and all(cfg[key + '_weight'] == 0 for key in ('kl', 'ctc', 'style'))
    assert result['auxiliary_state_unchanged'] and result['source_unchanged']
    rows = []
    for line in fixed[-1]['lines']:
        pen = line['mu']['pen']
        rows.append(dict(id=line['sample_id'], points=line['real_points'],
                         x_rmse=line['mu']['geometry']['axes']['x']['rmse_model_units'],
                         y_rmse=line['mu']['geometry']['axes']['y']['rmse_model_units'],
                         pen_f1=pen['pen_up_f1'], false_breaks=pen['pen_up_fp'], missed_breaks=pen['pen_up_fn'],
                         false_internal_eoc=pen['non_final_false_eoc_count'], final_eoc=pen['final_eoc_correct'],
                         sampled_xy=line['xy_sampled_summary'], sampled_pen_f1=line['pen_f1_sampled']))
    summary = dict(gate='NOT PASSED: geometry degraded and stopping/boundaries incomplete',
                   training_samples_only=True, updates=200, physical_microbatches=1600,
                   sampled_z_per_line_per_checkpoint=20, clipping_fraction=sum(r['gradient_norm'] > cfg['grad_clip'] for r in updates) / len(updates),
                   gradient_norm_median=statistics.median(r['gradient_norm'] for r in updates),
                   loop_elapsed_seconds=result['elapsed_seconds'], final_eoc_correct_lines=sum(r['final_eoc'] for r in rows),
                   false_internal_eoc_total=sum(r['false_internal_eoc'] for r in rows), lines=rows,
                   loss_log_scope=result.get('training_loss_log_scope', 'last microbatch (v1 logger)'),
                   anchor_weight=result['anchor_calibration']['weight'],
                   anchor_note='Median matched-per-line gradient calibration, not uniform 15% on all lines or an aggregate gradient guarantee.',
                   source_sha256=result['source_sha256'], manifest_sha256=result['manifest_sha256'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, key, title in zip(axes, ['mu_mean_x_rmse', 'mu_mean_y_rmse', 'mu_macro_pen_f1'], ['Mean X RMSE', 'Mean Y RMSE', 'Macro pen-up F1']):
        ax.plot([v['step'] for v in fixed], [v[key] for v in fixed], marker='o'); ax.set_title(title); ax.set_xlabel('Optimizer updates')
    fig.tight_layout(); fig.savefig(directory / 'curves.png'); plt.close(fig)
    page = ['<!doctype html><meta charset="utf-8"><title>Eight-line T4 report</title>',
            '<h1>Eight-line T4: reconstruction/pen gate NOT passed</h1>',
            '<p>200 updates / 1,600 batch-one microbatches. Training samples only; no held-out evaluation. CTC/style/KL weights zero. No further GPU run launched.</p>',
            '<p>Twenty fixed-seeded sampled latents per line at each checkpoint. Deterministic readout is mixture expectation. GroupNorm unchanged; only minimal multiple-of-eight padding.</p>',
            '<p>All 200 updates clipped at 5. The initial memorized line regressed; other lines improve but remain visibly jagged. Final EOC correct on 3/8 lines; 51 false internal EOCs. Do not enable CTC/KL yet.</p>',
            '<p>Historical training-loss rows are last-microbatch values, NOT effective-batch means. Future logger is corrected without rerunning this experiment. KL diagnostic values were computed but have zero objective weight.</p>',
            '<img src="curves.png" width="100%"><p><a href="summary.json">Summary</a> | <a href="result.json">Full metrics/provenance</a> | <a href="calibration.json">Anchor calibration</a></p>',
            '<table><tr><th>Line</th><th>X/Y RMSE</th><th>Pen F1</th><th>FP/FN</th><th>False EOC</th><th>Final EOC</th></tr>']
    for row in rows:
        page.append(f"<tr><td>{html.escape(row['id'])}</td><td>{row['x_rmse']:.4f}/{row['y_rmse']:.4f}</td><td>{row['pen_f1']:.3f}</td><td>{row['false_breaks']}/{row['missed_breaks']}</td><td>{row['false_internal_eoc']}</td><td>{row['final_eoc']}</td></tr>")
    page.append('</table>')
    for sample_id in ids:
        page.append(f'<h2>{html.escape(sample_id)}</h2><h3>Before</h3><img width="100%" src="step-0/{sample_id}/mu-comparison.png"><h3>After</h3><img width="100%" src="step-200/{sample_id}/mu-comparison.png">')
        for image in sorted((directory / 'step-200' / sample_id).glob('*.png')):
            if image.name != 'mu-comparison.png':
                page.append(f'<p>{html.escape(image.name)}</p><img width="100%" src="{image.relative_to(directory)}">')
    (directory / 'index.html').write_text('\n'.join(page))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory')
    print(json.dumps(report(parser.parse_args().directory), indent=2))
