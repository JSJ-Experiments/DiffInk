"""Regenerate CPU pen A/B curves and independently verify saved branch models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .pen_ab import (POLICIES, SOURCE_SHA, file_sha, load_experiment,
                     assert_nonpen_unchanged, capture, output_xy, boundary_metrics)


def report(directory, repo, data_root=None, checkpoint=None):
    torch.set_num_threads(2)
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    cfg, source_path, parent, model, sample, batch, source_cfg, features, output, nll = load_experiment(
        directory / 'config.yaml', repo, data_root, checkpoint or result['source_checkpoint'])
    source = parent['model_state_dict']; del parent
    baseline_xy = output_xy(output)
    mask = batch[1]; labels = batch[0][:, :, 2:].argmax(-1)[mask].numpy()
    cached = torch.load(directory / 'frozen_features.pt', map_location='cpu', weights_only=True)
    if cached['source_sha256'] != SOURCE_SHA or not torch.equal(features, cached['features']) or not torch.equal(output, cached['source_output']):
        raise AssertionError('saved cache differs from recomputed source')
    checks = {}; histories = {}
    for policy in POLICIES:
        branch_path = directory / policy / 'checkpoint.pt'
        branch = torch.load(branch_path, map_location='cpu', weights_only=True)
        assert_nonpen_unchanged(source, branch['model_state_dict'])
        if branch['source_sha256'] != SOURCE_SHA or branch['source_geometry_step'] != 2000 or branch['pen_head_updates'] != cfg['max_steps']:
            raise AssertionError('branch provenance mismatch')
        model.load_state_dict(branch['model_state_dict'], strict=True)
        after_features, after_output, after_nll = capture(model, batch, source_cfg)
        if not torch.equal(features, after_features) or not torch.equal(output[:, 3:], after_output[:, 3:]):
            raise AssertionError('frozen features/GMM changed')
        if not torch.equal(baseline_xy, output_xy(after_output)) or nll != after_nll:
            raise AssertionError('geometry changed')
        measured = boundary_metrics(after_output[:, :3].transpose(1, 2)[mask].argmax(-1).numpy(), labels)
        final = result['arms'][policy]['final']
        if any(measured[k] != final[k] for k in measured):
            raise AssertionError('saved-model boundary metrics differ from recorded result')
        checks[policy] = {'checkpoint_sha256': file_sha(branch_path),
                          'all_non_pen_state_bitwise_unchanged': True, 'features_bitwise_unchanged': True,
                          'gmm_output_bitwise_unchanged': True, 'xy_bitwise_unchanged': True,
                          'recorded_metrics_reproduced': True}
        histories[policy] = [json.loads(line) for line in (directory / policy / 'fixed_metrics.jsonl').read_text().splitlines()]
    if file_sha(source_path) != SOURCE_SHA:
        raise AssertionError('parent checkpoint changed')
    verification = {'gpu': False, 'training': False, 'new_optimizer_steps': 0,
                    'source_sha256': SOURCE_SHA, 'source_checkpoint_unchanged': True, 'arms': checks}
    (directory / 'saved_checkpoint_check.json').write_text(json.dumps(verification, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for policy, rows in histories.items():
        steps = [r['step'] for r in rows]
        short = 'A inverse' if policy == POLICIES[0] else 'B bounded'
        for ax, metric, label in [(axes[0, 0], 'pen_up_precision', 'Pen-up precision'),
                                  (axes[0, 1], 'pen_up_recall', 'Pen-up recall'),
                                  (axes[0, 2], 'pen_up_f1', 'Pen-up F1'),
                                  (axes[1, 0], 'non_final_false_eoc_count', 'False internal EOC')]:
            ax.plot(steps, [r[metric] for r in rows], marker='.', label=short)
            ax.set_title(label); ax.set_xlabel('Head updates'); ax.grid(alpha=.2)
        axes[1, 1].plot(steps, [r['predicted_counts'][1] for r in rows], marker='.', label=short)
        axes[1, 2].plot(steps, [r['pen_focal_loss'] for r in rows], marker='.', label=short)
    axes[1, 1].axhline(36, color='black', linestyle='--', label='36 true pen-ups')
    axes[1, 1].set_title('Predicted pen-up count'); axes[1, 2].set_title('Weighted focal (scales NOT comparable)')
    for ax in axes.flat:
        ax.legend(); ax.set_xlabel('Head updates')
    fig.suptitle('Identical frozen features/XY: only pen weighting differs')
    fig.tight_layout(); fig.savefig(directory / 'pen-policy-curves.png', dpi=120); plt.close(fig)
    page = directory / 'index.html'
    current = page.read_text()
    if 'pen-policy-curves.png' not in current:
        current = current.replace('<h1>Frozen pen-head A/B</h1>', '<h1>Frozen pen-head A/B</h1><img style="max-width:100%" src="pen-policy-curves.png">')
        page.write_text(current)
    return verification


def main():
    repo = 'third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--repo', default=repo)
    parser.add_argument('--data-root'); parser.add_argument('--checkpoint')
    args = parser.parse_args()
    print(json.dumps(report(args.directory, args.repo, args.data_root, args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
