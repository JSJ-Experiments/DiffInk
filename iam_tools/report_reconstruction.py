"""CPU report and saved-state verification for independent reconstruction branches."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .reconstruction import (STAGES, experiment, initialize, evaluate, geometry_gate,
                             SOURCE_SHA, WEIGHT_KEY, BIAS_KEY, file_sha)


def draw_true_pen(ax, xy, states, label):
    start = 0
    for end in range(len(xy)):
        if states[end] != 0 or end == len(xy) - 1:
            points = xy[start:end + 1]
            ax.plot(points[:, 0], points[:, 1], color='black', linewidth=.8, marker='.', markersize=1)
            start = end + 1
    ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title(label, fontsize=10)


def report(directory, joint_directory, repo, data_root=None, source_checkpoint=None, pen_checkpoint=None):
    torch.set_num_threads(2)
    directory = Path(directory); joint_directory = Path(joint_directory)
    cfg, model, sample, batch, source_cfg, _, parent, bstate, source_path = experiment(
        directory / 'config.yaml', repo, data_root, source_checkpoint, pen_checkpoint)
    source = parent['model_state_dict']
    branches = {'joint': joint_directory / 'joint', 'deterministic_mse': directory / 'deterministic_mse'}
    results = {}; checks = {}; histories = {}
    for stage, branch_dir in branches.items():
        r = json.loads((branch_dir / 'result.json').read_text())
        history = [json.loads(line) for line in (branch_dir / 'metrics.jsonl').read_text().splitlines()]
        fixed = [json.loads(line) for line in (branch_dir / 'fixed_metrics.jsonl').read_text().splitlines()]
        if len(history) != r['steps'] or [x['step'] for x in history] != list(range(1, r['steps'] + 1)):
            raise ValueError('training step provenance incomplete')
        if not all(x['auxiliary_gradients_absent'] for x in history):
            raise ValueError('unexpected auxiliary gradients')
        if fixed[0]['step'] != 0 or fixed[-1]['step'] != r['steps']:
            raise ValueError('fixed evaluation provenance incomplete')
        if r['source_sha256'] != SOURCE_SHA:
            raise ValueError('branches must share original step-2000 source')
        checkpoint = torch.load(branch_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
        state = checkpoint['model_state_dict']
        if checkpoint['diagnostic_updates'] != r['steps'] or checkpoint['source_geometry_step'] != 2000:
            raise ValueError('checkpoint provenance mismatch')
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('nonfinite saved state')
        aux = {k: v for k, v in source.items() if k.startswith(('ocr_model.', 'style_classifier.'))}
        if not all(torch.equal(v, state[k]) for k, v in aux.items()):
            raise AssertionError('saved auxiliary weights changed')
        mse = stage == 'deterministic_mse'
        if mse:
            protected = torch.cat([torch.arange(3), torch.arange(63, 123)])
            if not all(torch.equal(source[key][protected], state[key][protected]) for key in (WEIGHT_KEY, BIAS_KEY)):
                raise AssertionError('MSE changed pen/sigma/rho output rows')
            if not all(torch.equal(v, state[k]) for k, v in source.items() if k.startswith('conv_logvar.')):
                raise AssertionError('MSE changed latent variance head')
        optimizer_steps = [int(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()]
        expected_step = r['steps'] if mse else 2000 + r['steps']
        if not optimizer_steps or set(optimizer_steps) != {expected_step}:
            raise AssertionError('optimizer step mismatch')
        initialize(model, parent, stage, cfg, bstate)
        cpu_before, _ = evaluate(model, batch, sample, cfg, source_cfg, stage, 'cpu')
        model.load_state_dict(state, strict=True)
        cpu_after, _ = evaluate(model, batch, sample, cfg, source_cfg, stage, 'cpu')
        primary = r['final']['primary']
        first = r['initial']['trajectories'][primary]; final = r['final']['trajectories'][primary]
        r['primary_rmse_reduction'] = {axis: 1 - final['axes'][axis]['rmse_model_units'] / first['axes'][axis]['rmse_model_units'] for axis in ('x', 'y')}
        r['primary_temporal_reduction'] = {name: 1 - final['temporal'][name]['within_true_strokes']['vector_rmse'] / first['temporal'][name]['within_true_strokes']['vector_rmse'] for name in ('velocity', 'second_difference')}
        checks[stage] = {'checkpoint_sha256': file_sha(branch_dir / 'checkpoint.pt'),
                         'all_saved_state_finite': True, 'auxiliary_state_unchanged': True,
                         'mse_pen_sigma_rho_rows_unchanged': True if mse else None,
                         'mse_latent_variance_head_unchanged': True if mse else None,
                         'adam_step': expected_step, 'adam_state_entries': len(optimizer_steps),
                         'cpu_primary_before': cpu_before['trajectories'][primary],
                         'cpu_primary_after': cpu_after['trajectories'][primary],
                         'cpu_vs_cuda_bitwise_equivalence_claimed': False}
        results[stage] = r; histories[stage] = fixed
    if file_sha(source_path) != SOURCE_SHA:
        raise AssertionError('original source changed')
    summary = {'gpu': False, 'new_optimizer_steps': 0, 'source_sha256': SOURCE_SHA,
               'source_checkpoint_unchanged': True, 'joint_directory': str(joint_directory),
               'mse_directory': str(directory), 'results': results, 'saved_checkpoint_checks': checks,
               'attempt_note': ('Joint and MSE come from distinct workers; the MSE-only worker did not repeat joint.' if directory.resolve() != joint_directory.resolve() else 'Both branches completed in one worker.'),
               'joint_not_repeated': True, 'ctc_not_started': True,
               'deterministic_branch_not_loss_only_ablation': True,
               'visual_gate': 'requires inspection, not inferred from NLL or RMSE'}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truth = sample[1].numpy(); states = truth[:, 2:].argmax(1); target = truth[:, :2] * cfg['model_input_scale']
    for stage, branch_dir in branches.items():
        r = results[stage]; primary = r['final']['primary']
        before = np.load(branch_dir / f'step-0-{primary}-prediction-model-units.npy')[:, :2]
        after = np.load(branch_dir / f'step-{r["steps"]}-{primary}-prediction-model-units.npy')[:, :2]
        fig, axes = plt.subplots(3, 1, figsize=(16, 5.3), sharex=True, sharey=True)
        for ax, xy, label in zip(axes, [target, before, after], ['input', 'source + TRUE pen', f'{stage} after {r["steps"]} updates + TRUE pen']):
            draw_true_pen(ax, xy, states, label)
        fig.suptitle(sample[2]); fig.tight_layout()
        fig.savefig(directory / f'{stage}-true-pen-before-after.png', dpi=120); plt.close(fig)
    # Same final checkpoint, true pen: isolate latent noise and readout choice.
    mse_dir = branches['deterministic_mse']; final_step = results['deterministic_mse']['steps']
    fig, axes = plt.subplots(4, 1, figsize=(16, 6.8), sharex=True, sharey=True)
    draw_true_pen(axes[0], target, states, 'input')
    for ax, mode in zip(axes[1:], ['latent_mean-highest_weight', 'latent_mean-mixture_expectation',
                                  'sampled_seed1042-mixture_expectation']):
        xy = np.load(mse_dir / f'step-{final_step}-{mode}-prediction-model-units.npy')[:, :2]
        draw_true_pen(ax, xy, states, mode + ' + TRUE pen')
    fig.suptitle('Same final MSE checkpoint: readout/noise controls, not extra training')
    fig.tight_layout(); fig.savefig(directory / 'mse-readout-controls.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for stage, fixed in histories.items():
        primary = fixed[0]['primary']; steps = [x['step'] for x in fixed]
        geometry = [x['trajectories'][primary] for x in fixed]
        for axis in ('x', 'y'):
            axes[0, 0].plot(steps, [g['axes'][axis]['rmse_model_units'] for g in geometry], label=stage+' '+axis)
        for ax, name, title in [(axes[0, 1], 'velocity', 'Within-true-stroke first-difference error'),
                                 (axes[0, 2], 'second_difference', 'Within-true-stroke second-difference error')]:
            ax.plot(steps, [g['temporal'][name]['within_true_strokes']['vector_rmse'] for g in geometry], label=stage)
            ax.set_title(title)
        axes[1, 0].plot(steps, [g['axes']['y']['correlation'] for g in geometry], label=stage)
    mse_rows = histories['deterministic_mse']
    axes[1, 1].plot([x['step'] for x in mse_rows], [x['fixed_losses']['expected_xy_mse'] for x in mse_rows], label='MSE primary')
    axes[1, 1].set_yscale('log'); axes[1, 1].set_title('Fixed deterministic MSE (log scale)')
    joint_rows = histories['joint']
    axes[1, 2].plot([x['step'] for x in joint_rows], [x['pen']['sampled_seed1042']['pen_up_f1'] for x in joint_rows], label='joint pen F1')
    axes[0, 0].set_title('Point RMSE (different primary readouts)'); axes[1, 0].set_title('Y correlation'); axes[1, 2].set_title('Joint pen-up F1')
    for ax in axes.flat:
        ax.legend(); ax.grid(alpha=.2); ax.set_xlabel('Branch optimizer updates')
    fig.suptitle('Independent one-line diagnostics; derivatives measured, NOT optimized')
    fig.tight_layout(); fig.savefig(directory / 'reconstruction-curves.png', dpi=120); plt.close(fig)
    page = ['<!doctype html><meta charset="utf-8"><h1>One-line reconstruction diagnostics</h1>',
            '<p>Joint: sampled latent/max-pi readout, trained Arm B pen rows, strict geometry gate. '
            'MSE: fresh optimizer, latent mean/dropout OFF/mixture expectation, TRUE pen boundaries. '
            'MSE is a capacity diagnostic, not a pure loss-only stochastic ablation. No CTC/style/KL. '
            'First/second differences use true within-stroke windows, not physical time or arc-length curvature.</p>',
            '<img style="max-width:100%" src="reconstruction-curves.png">']
    page.append('<h2>Final MSE readout/noise controls</h2><img style="max-width:100%" src="mse-readout-controls.png">')
    for stage in STAGES:
        r = results[stage]
        page.append(f'<h2>{stage}: {r["steps"]} updates; {r["stop_reason"]}</h2>')
        page.append(f'<img style="max-width:100%" src="{stage}-true-pen-before-after.png">')
        branch_dir = branches[stage]
        if stage == 'joint':
            image = branch_dir / f'step-{r["steps"]}-{r["final"]["primary"]}-comparison.png'
            page.append(f'<p>Joint predicted-pen comparison:</p><img style="max-width:100%" src="{os.path.relpath(image, directory)}">')
        else:
            for row in histories[stage]:
                name = f'step-{row["step"]}-{row["primary"]}'
                image = branch_dir / f'{name}-comparison.png'
                page.append(f'<h3>MSE update {row["step"]}</h3><p>Middle panel uses TRUE pens; bottom pens are UNTRAINED and are not the diagnostic.</p><img style="max-width:100%" src="{os.path.relpath(image, directory)}">')
    (directory / 'index.html').write_text('\n'.join(page))
    return summary


def main():
    repo = 'third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--joint-directory', required=True)
    parser.add_argument('--repo', default=repo); parser.add_argument('--data-root')
    parser.add_argument('--source-checkpoint'); parser.add_argument('--pen-checkpoint')
    args = parser.parse_args()
    result = report(args.directory, args.joint_directory, args.repo, args.data_root, args.source_checkpoint, args.pen_checkpoint)
    print(json.dumps({'source_unchanged': result['source_checkpoint_unchanged'],
                      'results': {stage: {'steps': r['steps'], 'rmse_reduction': r['primary_rmse_reduction'],
                                          'temporal_reduction': r['primary_temporal_reduction']} for stage, r in result['results'].items()}}, indent=2))


if __name__ == '__main__':
    main()
