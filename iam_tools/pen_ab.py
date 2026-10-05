"""Bounded CPU pen-head A/B on a pinned, frozen step-2000 VAE.

Cache one deterministic eval-mode representation (latent seed 1042). Train a
separate 3-row Linear, not a gradient-masked 123-row FC: AdamW must never touch
GMM rows. Merge only these rows into source copies and check full forwards.
This is a conditional linear-head diagnostic, NOT joint VAE training.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .autopsy import coordinate_forward, selected_sample, sample_sha
from .inkvae import setup, fixed_evaluation
from .trajectory_diagnostics import diagnostics

SOURCE_SHA = '23fc747d82773d6714385f1e25af650ab018c14d95f11c2db54afbe0c8a8ac33'
SAMPLE_SHA = '3adbd25fb7299c5583d22b4beea0c445daa0ce5c6f4880f72499d90b81767e1c'
MANIFEST_SHA = '6d1fafea62af6c5ddae48983bb9699d93016e4af7a11df2f8dd1743128c34c8a'
WEIGHT_KEY = 'transformer_decoder.fc.weight'
BIAS_KEY = 'transformer_decoder.fc.bias'
POLICIES = ('A_inverse_frequency', 'B_bounded_english')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_config(cfg):
    required = {'stage': 'pen-head-ab', 'source_step': 2000,
                'source_sha256': SOURCE_SHA, 'sample_id': 'c08-434z-05',
                'latent_noise_seed': 1042, 'device': 'cpu', 'base_lr': .001,
                'weight_decay': 0, 'gamma': 2, 'weight_cap': 8,
                'betas': [.9, .99], 'grad_clip': 10, 'save_every': 100}
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'unsupported A/B setting: {key}')
    if not 1 <= cfg['max_steps'] <= 1000 or not 1 <= cfg['max_wall_seconds'] <= 180:
        raise ValueError('A/B cap: 1000 updates per arm / 180 total loop seconds')


def real_targets(targets, mask):
    if targets.shape != mask.shape or mask.dtype != torch.bool or not mask.any():
        raise ValueError('nonempty boolean mask matching targets required')
    result = targets[mask]
    if result.min() < 0 or result.max() > 2:
        raise ValueError('invalid real pen classes')
    return result


def policy_weights(targets, mask, policy):
    labels = real_targets(targets, mask)
    counts = torch.bincount(labels, minlength=3).float()
    inverse = len(labels) / counts.clamp_min(1)
    inverse[counts == 0] = 0
    if policy == POLICIES[0]:
        return inverse
    if policy == POLICIES[1]:
        if counts[0] == 0:
            raise ValueError('continue normalization requires continue points')
        return (inverse / inverse[0]).sqrt().clamp(max=8)
    raise ValueError('unknown pen policy')


def masked_focal(logits, targets, mask, weights):
    """Same upstream focal: alpha * (1-exp(-UNWEIGHTED CE))**2 * CE.

    Select real points BEFORE CE so even nonfinite padded logits cannot leak.
    """
    labels = real_targets(targets, mask)
    if logits.shape != (*targets.shape, 3):
        raise ValueError('expected matching logits ending in three classes')
    ce = F.cross_entropy(logits[mask], labels, reduction='none')
    return (weights[labels] * (1 - torch.exp(-ce)).square() * ce).mean()


def boundary_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if predicted.shape != target.shape or target.ndim != 1 or not len(target):
        raise ValueError('matching nonempty unpadded state vectors required')
    if not np.isin(predicted, [0, 1, 2]).all() or not np.isin(target, [0, 1, 2]).all():
        raise ValueError('invalid pen labels')
    if target[-1] != 2:
        raise ValueError('this diagnostic expects a true final EOC')
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (target, predicted), 1)
    tp = int(confusion[1, 1]); fp = int(confusion[:, 1].sum()) - tp
    fn = int(confusion[1, :].sum()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'true_counts': confusion.sum(1).tolist(),
            'predicted_counts': confusion.sum(0).tolist(),
            'pen_up_precision': precision, 'pen_up_recall': recall,
            'pen_up_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'pen_up_tp': tp, 'pen_up_fp': fp, 'pen_up_fn': fn,
            'non_final_false_eoc_count': int(((predicted[:-1] == 2) & (target[:-1] != 2)).sum()),
            'final_eoc_correct': bool(predicted[-1] == 2),
            'confusion_true_rows_pred_columns': confusion.tolist(),
            'per_class_recall': [float(confusion[i, i] / n) if n else None
                                 for i, n in enumerate(confusion.sum(1))],
            'pen_accuracy_not_gate': float(np.trace(confusion) / len(target))}


def clone_pen_head(fc):
    head = nn.Linear(fc.in_features, 3, device=fc.weight.device, dtype=fc.weight.dtype)
    with torch.no_grad():
        head.weight.copy_(fc.weight[:3]); head.bias.copy_(fc.bias[:3])
    return head


def merge_head(source, head):
    # Detached references for unchanged tensors; clone ONLY the two changed FC tensors.
    merged = dict(source)
    for key, value in [(WEIGHT_KEY, head.weight), (BIAS_KEY, head.bias)]:
        merged[key] = source[key].clone()
        merged[key][:3].copy_(value.detach().cpu())
    return merged


def assert_nonpen_unchanged(source, candidate):
    if source.keys() != candidate.keys():
        raise AssertionError('state keys changed')
    for key, tensor in source.items():
        original, actual = (tensor[3:], candidate[key][3:]) if key in (WEIGHT_KEY, BIAS_KEY) else (tensor, candidate[key])
        if not torch.equal(original, actual):
            raise AssertionError(f'non-pen state changed: {key}')
    return True


def capture(model, batch, cfg, seed=1042):
    cached = []
    hook = model.transformer_decoder.fc.register_forward_pre_hook(
        lambda module, args: cached.append(args[0].detach().clone()))
    try:
        with fixed_evaluation(model, seed, 'cpu'):
            loss, output, _ = coordinate_forward(model, batch, cfg, 'cpu')
        if len(cached) != 1:
            raise AssertionError('expected one decoder FC call')
        return cached[0], output.detach().clone(), float(loss)
    finally:
        hook.remove()


def output_xy(output):
    from model.gmm import get_mixture_coef_max
    pi, mx, my, *_ = get_mixture_coef_max(output, 20)
    choice = pi.argmax(1, keepdim=True)
    return torch.stack([mx.gather(1, choice).squeeze(1), my.gather(1, choice).squeeze(1)], -1)


def render_snapshot(path, name, xy, states, truth, scale, text, metrics):
    """Never relabel the final state for display or truncate at a bogus EOC.

    An unfinished final stroke is drawn as-is, and final EOC correctness is
    printed. All non-continue points break the path AFTER that point.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred = np.column_stack([xy, np.eye(3, dtype=np.float32)[states]])
    np.save(path / f'{name}-prediction-model-units.npy', pred)
    fig, axes = plt.subplots(3, 1, figsize=(16, 5.3), sharex=True, sharey=True)
    panels = [(truth[:, :2] * scale, truth[:, 2:].argmax(1), 'input'),
              (xy, truth[:, 2:].argmax(1), 'fixed XY + true pen'),
              (xy, states, f'fixed XY + predicted pen | counts {metrics["predicted_counts"]} | '
               f'pen-up F1 {metrics["pen_up_f1"]:.3f} | false EOC {metrics["non_final_false_eoc_count"]} | '
               f'final EOC {metrics["final_eoc_correct"]}')]
    for ax, (points, labels, label) in zip(axes, panels):
        start = 0
        for end in range(len(points)):
            if labels[end] != 0 or end == len(points) - 1:
                stroke = points[start:end + 1]
                ax.plot(stroke[:, 0], stroke[:, 1], color='black', linewidth=.8, marker='.', markersize=1.5)
                start = end + 1
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title(label, fontsize=9)
    fig.suptitle(text); fig.tight_layout()
    fig.savefig(path / f'{name}-comparison.png', dpi=120); plt.close(fig)


def load_experiment(config_path, repo, data_root=None, checkpoint=None):
    cfg = yaml.safe_load(Path(config_path).read_text()); check_config(cfg)
    source_path = Path(checkpoint or cfg['source_checkpoint'])
    if file_sha(source_path) != SOURCE_SHA:
        raise ValueError('exact step-2000 checkpoint SHA required')
    parent = torch.load(source_path, map_location='cpu', weights_only=True)
    if parent['step'] != 2000 or parent['sample_sha256'] != SAMPLE_SHA:
        raise ValueError('source step/sample mismatch')
    model, train, val, source_cfg, root = setup(Path(repo) / cfg['source_config'], repo, data_root)
    try:
        if source_cfg != parent['config']:
            raise ValueError('geometry source config differs from checkpoint')
        if file_sha(root / 'manifest.json') != MANIFEST_SHA:
            raise ValueError('pinned dataset manifest required')
        sample = selected_sample(train, cfg['sample_id'])
        if sample_sha(sample) != SAMPLE_SHA:
            raise ValueError('pinned sample required')
        batch = train.collate_fn([sample])
        mask = batch[1]
        targets = batch[0][:, :, 2:].argmax(-1)
        if real_targets(targets, mask).bincount(minlength=3).tolist() != [544, 36, 1]:
            raise ValueError('expected 544 / 36 / 1 real points')
        model.load_state_dict(parent['model_state_dict'], strict=True)
        model.eval().requires_grad_(False)
        features, output, nll = capture(model, batch, source_cfg)
        return cfg, source_path, parent, model, sample, batch, source_cfg, features, output, nll
    finally:
        train.hf.close(); val.hf.close()


def run(config_path, repo, data_root=None, checkpoint=None, output_base=None,
        train=False, allow_experimental=False):
    if train and not allow_experimental:
        raise ValueError('explicit experimental acknowledgement required')
    torch.set_num_threads(2)
    cfg, source_path, parent, model, sample, batch, source_cfg, features, output, nll = load_experiment(
        config_path, repo, data_root, checkpoint)
    source = parent['model_state_dict']
    # Drop the large, unused geometry optimizer before head-only optimization.
    del parent
    mask = batch[1]; targets = batch[0][:, :, 2:].argmax(-1)
    weights = {policy: policy_weights(targets, mask, policy) for policy in POLICIES}
    xy = output_xy(output); real_xy = xy[mask].numpy()
    labels = targets[mask].numpy()
    initial = boundary_metrics(output[:, :3].transpose(1, 2)[mask].argmax(-1).numpy(), labels)
    report = {'stage': cfg['stage'], 'training': train, 'gpu': False, 'device': 'cpu',
              'source_checkpoint': str(source_path), 'source_sha256': SOURCE_SHA,
              'source_step': 2000, 'sample_sha256': SAMPLE_SHA, 'manifest_sha256': MANIFEST_SHA,
              'sample_id': cfg['sample_id'], 'real_points': int(mask.sum()),
              'padded_points': int(mask.numel() - mask.sum()), 'padding_excluded': True,
              'latent_noise_seed': 1042, 'features': 'cached eval-mode deterministic source; no dropout',
              'trainable_scalars_per_arm': 771, 'fresh_head_optimizer': True,
              'base_lr': cfg['base_lr'], 'weight_decay': cfg['weight_decay'], 'betas': cfg['betas'],
              'gamma': 2, 'grad_clip': cfg['grad_clip'], 'torch_version': torch.__version__,
              'weights': {p: w.tolist() for p, w in weights.items()}, 'initial_pen': initial,
              'baseline_gmm_nll_cpu': nll, 'cpu_vs_cuda_bitwise_equivalence_claimed': False,
              'objectives': {'pen': True, 'coordinates': False, 'ctc': False, 'style': False, 'kl': False}}
    if not train:
        report['optimizer_steps'] = 0
        return report
    directory = Path(output_base or cfg['output_base']) / time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.save({'features': features, 'mask': mask, 'targets': targets, 'source_output': output,
                'source_sha256': SOURCE_SHA, 'latent_noise_seed': 1042}, directory / 'frozen_features.pt')
    started = time.monotonic(); arms = {}; head_start = clone_pen_head(model.transformer_decoder.fc)
    for policy in POLICIES:
        arm_dir = directory / policy; arm_dir.mkdir()
        head = clone_pen_head(model.transformer_decoder.fc)
        if any(not torch.equal(v, head.state_dict()[k]) for k, v in head_start.state_dict().items()):
            raise AssertionError('arms must start identical')
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['base_lr'], betas=cfg['betas'], weight_decay=cfg['weight_decay'])
        rows = []; last_gradient = None; clipping = 0; step = 0

        def record():
            with torch.no_grad():
                logits = head(features)
                loss = float(masked_focal(logits, targets, mask, weights[policy]))
                predicted = logits[mask].argmax(-1).numpy()
            merged = merge_head(source, head)
            assert_nonpen_unchanged(source, merged)
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError('gradient leaked into source')
            # Full model verification, not just unchanged cached XY by construction.
            model.load_state_dict(merged, strict=True)
            _, checked, checked_nll = capture(model, batch, source_cfg)
            if not torch.equal(output[:, 3:], checked[:, 3:]):
                raise AssertionError('GMM outputs changed')
            if not torch.equal(xy, output_xy(checked)) or checked_nll != nll:
                raise AssertionError('XY or GMM NLL changed')
            actual_predicted = checked[:, :3].transpose(1, 2)[mask].argmax(-1).numpy()
            if not np.array_equal(predicted, actual_predicted):
                raise AssertionError('merged head prediction differs from cached head')
            assert_nonpen_unchanged(source, model.state_dict())
            model.load_state_dict(source, strict=True)
            metrics = boundary_metrics(predicted, labels)
            geometry = diagnostics(np.column_stack([real_xy, np.eye(3)[predicted]]), sample[1].numpy(), source_cfg['model_input_scale'])['axes']
            row = {'step': step, 'pen_focal_loss': loss, 'raw_grad_norm': last_gradient,
                   'clipping_fraction': clipping / step if step else 0., **metrics,
                   'geometry_axes_unchanged': geometry, 'non_pen_state_bitwise_unchanged': True,
                   'gmm_output_bitwise_unchanged': True, 'xy_bitwise_unchanged': True}
            rows.append(row)
            with (arm_dir / 'fixed_metrics.jsonl').open('a') as log:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            render_snapshot(arm_dir, f'step-{step}', real_xy, predicted, sample[1].numpy(),
                            source_cfg['model_input_scale'], sample[2], metrics)
            print({'arm': policy, **row}, flush=True)

        record()
        with (arm_dir / 'train_metrics.jsonl').open('w') as log:
            for step in range(1, cfg['max_steps'] + 1):
                if time.monotonic() - started > cfg['max_wall_seconds']:
                    # An unequal comparison is invalid: fail, don't present it as finished.
                    raise TimeoutError('A/B combined wall budget exhausted; comparison incomplete')
                optimizer.zero_grad(set_to_none=True)
                loss = masked_focal(head(features), targets, mask, weights[policy])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite pen loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                last_gradient = float(norm); clipping += int(last_gradient > cfg['grad_clip'])
                optimizer.step()
                log.write(json.dumps({'step': step, 'loss_before_update': float(loss.detach()),
                                      'raw_grad_norm': last_gradient, 'was_clipped': last_gradient > cfg['grad_clip']}) + '\n')
                if step % cfg['save_every'] == 0 or step == cfg['max_steps']:
                    record()
        branch = merge_head(source, head)
        torch.save({'model_state_dict': branch, 'source_geometry_step': 2000,
                    'pen_head_updates': step, 'pen_policy': policy, 'class_weights': weights[policy],
                    'head_optimizer_state_dict': optimizer.state_dict(), 'config': source_cfg,
                    'pen_ab_config': cfg, 'sample_sha256': SAMPLE_SHA, 'source_sha256': SOURCE_SHA,
                    'experimental': True, 'features_fixed': True}, arm_dir / 'checkpoint.pt')
        arms[policy] = {'updates': step, 'initial': rows[0], 'final': rows[-1],
                        'identical_initial_head': True, 'clipping_fraction': clipping / step}
    if file_sha(source_path) != SOURCE_SHA:
        raise AssertionError('parent checkpoint modified')
    assert_nonpen_unchanged(source, model.state_dict())
    report.update(arms=arms, output=str(directory), elapsed_seconds=time.monotonic() - started,
                  source_checkpoint_unchanged=True, all_non_pen_state_unchanged=True,
                  all_gmm_outputs_unchanged=True, all_fixed_xy_unchanged=True,
                  later_stages_started=False, stop_reason='equal_max_steps')
    (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    page = ['<!doctype html><meta charset="utf-8"><h1>Frozen pen-head A/B</h1>',
            '<p>CPU, same cached eval features / latent seed 1042, 771 trainable values. '
            'No geometry/CTC/style/KL training. Raw predictions are never final-EOC-corrected. '
            'An unfinished final path is drawn; final EOC correctness is separately labeled.</p>']
    for step in range(0, cfg['max_steps'] + 1, cfg['save_every']):
        page.append(f'<h2>Head update {step}</h2>')
        for policy in POLICIES:
            page.append(f'<h3>{html.escape(policy)}</h3><img style="max-width:100%" src="{policy}/step-{step}-comparison.png">')
    (directory / 'index.html').write_text('\n'.join(page))
    return report


def main():
    repo = 'third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=repo)
    parser.add_argument('--config', default=repo + '/configs/vae_iam_pen_ab.yaml')
    parser.add_argument('--data-root'); parser.add_argument('--checkpoint'); parser.add_argument('--output-base')
    parser.add_argument('--train', action='store_true'); parser.add_argument('--allow-experimental', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.repo, args.data_root, args.checkpoint, args.output_base,
                         args.train, args.allow_experimental), indent=2))


if __name__ == '__main__':
    main()
