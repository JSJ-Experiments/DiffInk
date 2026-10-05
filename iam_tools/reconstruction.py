"""Two bounded one-line diagnostics: joint GMM/pen, then deterministic expected-XY MSE.

No OCR/style/KL, no smoothing/resampling or derivative training objective.
First/second differences are measured within TRUE strokes, never across pen-up.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .autopsy import coordinate_forward, freeze_auxiliaries, selected_sample, sample_sha
from .autopsy_resume import restore_optimizer, clipping_summary
from .inkvae import setup, fixed_evaluation
from .pen_ab import (SOURCE_SHA, SAMPLE_SHA, MANIFEST_SHA, WEIGHT_KEY, BIAS_KEY,
                     POLICIES, file_sha, policy_weights, masked_focal, boundary_metrics,
                     assert_nonpen_unchanged, render_snapshot)
from .trajectory_diagnostics import diagnostics

B_SHA = 'ba20fd1598282454b1b841c00076c01975bf77f8be45f08e46e65c12e31c1749'
STAGES = ('joint', 'deterministic_mse')


def check_config(cfg):
    required = {'source_sha256': SOURCE_SHA, 'source_step': 2000,
                'sample_id': 'c08-434z-05', 'model_input_scale': .01,
                'latent_noise_seed': 1042, 'seed': 42, 'save_every': 50,
                'grad_clip': 10, 'ctc_weight': 0, 'style_weight': 0, 'kl_weight': 0,
                'joint_steps': 300, 'joint_lr': 1e-5, 'joint_pen_weight': 1,
                'mse_steps': 1000, 'mse_lr': 1e-4, 'mse_weight_decay': 0,
                'max_wall_seconds': 600, 'derivative_loss_weight': 0,
                'mse_latent': 'mean', 'mse_dropout': False}
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'unsupported reconstruction setting: {key}')
    if cfg.get('joint_pen_initialization') not in ('original_geometry', 'trained_B_rows'):
        raise ValueError('unsupported joint initialization')
    if cfg.get('pen_B_sha256') != B_SHA:
        raise ValueError('pinned Arm B required')


def expected_xy(output):
    """B x 123 x T -> B x T x 2, differentiable pi-weighted absolute means."""
    if output.ndim != 3 or output.shape[1] != 123:
        raise ValueError('expected 20-component DiffInk output')
    pi = torch.softmax(output[:, 3:23], dim=1)
    return torch.stack([(pi * output[:, 23:43]).sum(1),
                        (pi * output[:, 43:63]).sum(1)], dim=-1)


def maximum_xy(output):
    pi = output[:, 3:23].softmax(1)
    choice = pi.argmax(1, keepdim=True)
    return torch.stack([output[:, 23:43].gather(1, choice).squeeze(1),
                        output[:, 43:63].gather(1, choice).squeeze(1)], dim=-1)


def masked_xy_mse(xy, target, mask):
    if xy.shape != target.shape or xy.shape != (*mask.shape, 2) or mask.dtype != torch.bool or not mask.any():
        raise ValueError('matching XY and nonempty boolean mask required')
    # Mean over real points AND two coordinate components; pad never participates.
    return (xy[mask] - target[mask]).square().mean()


def temporal_errors(prediction, truth, true_states):
    """Index-difference errors, NOT physical time derivatives or geometric curvature.

    RDP samples have variable spacing. Comparisons use identical point indexing;
    within-stroke masks avoid discontinuous pen-up jumps dominating these errors.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    states = np.asarray(true_states)
    if prediction.shape != truth.shape or truth.shape != (len(states), 2) or not len(states):
        raise ValueError('equal unpadded N x 2 sequences and true states required')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all() or not np.isin(states, [0, 1, 2]).all():
        raise ValueError('invalid trajectory/state values')
    connected = states[:-1] == 0
    masks = {1: connected, 2: connected[:-1] & connected[1:]}
    output = {'units': 'model units per point-index difference, NOT per second',
              'curvature_definition': 'second difference, NOT arc-length-normalized geometric curvature'}
    for order, label in [(1, 'velocity'), (2, 'second_difference')]:
        a = np.diff(prediction, n=order, axis=0)
        b = np.diff(truth, n=order, axis=0)
        subsets = {}
        for name, mask in [('within_true_strokes', masks[order]),
                           ('all_points_including_penup_jumps', np.ones(len(a), dtype=bool))]:
            count = int(mask.sum())
            if count:
                error = a[mask] - b[mask]
                rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                target_rms = float(np.sqrt(np.mean(np.sum(b[mask]**2, axis=1))))
                subsets[name] = {'windows': count, 'vector_rmse': rms,
                                 'x_rmse': float(np.sqrt(np.mean(error[:, 0]**2))),
                                 'y_rmse': float(np.sqrt(np.mean(error[:, 1]**2))),
                                 'target_vector_rms': target_rms,
                                 'normalized_error': rms / target_rms if target_rms > 1e-12 else None}
            else:
                subsets[name] = {'windows': 0, 'vector_rmse': None, 'x_rmse': None,
                                 'y_rmse': None, 'target_vector_rms': None, 'normalized_error': None}
        output[label] = subsets
    return output


def geometry_gate(baseline, current):
    """Strict non-regression gate, only 1e-7 numerical tolerance; NOT a realism gate."""
    axes = {}
    for name in ('x', 'y'):
        a, b = baseline['axes'][name], current['axes'][name]
        axes[name] = {'rmse_nonregressed': b['rmse_model_units'] <= a['rmse_model_units'] + 1e-7,
                      'correlation_nonregressed': a['correlation'] is not None and b['correlation'] is not None
                      and b['correlation'] >= a['correlation'] - 1e-7}
    return {'passed': all(v for row in axes.values() for v in row.values()),
            'axes': axes, 'absolute_tolerance': 1e-7, 'human_realism_claimed': False}


def deterministic_forward(model, batch, scale):
    # eval() disables dropout but DOES NOT disable gradients. No sampled z, no KL.
    data = batch[0].clone(); data[:, :, :2] *= scale
    latent_mean = model.conv_mu(model.encoder(data.transpose(1, 2)))
    return model.decode(latent_mean)


def objectives(model, batch, cfg, source_cfg, stage, device):
    data, mask = batch[:2]
    targets = data[:, :, 2:].argmax(-1)
    target_xy = data[:, :, :2] * cfg['model_input_scale']
    if stage == 'joint':
        nll, output, _ = coordinate_forward(model, batch, source_cfg, device)
        weights = policy_weights(targets, mask, POLICIES[1])
        pen = masked_focal(output[:, :3].transpose(1, 2), targets, mask, weights)
        loss = nll + cfg['joint_pen_weight'] * pen
        terms = {'gmm_nll': nll, 'pen_focal': pen}
    elif stage == 'deterministic_mse':
        output = deterministic_forward(model, batch, cfg['model_input_scale'])
        loss = masked_xy_mse(expected_xy(output), target_xy, mask)
        terms = {'expected_xy_mse': loss}
    else:
        raise ValueError('unsupported reconstruction stage')
    if not all(torch.isfinite(v).all() for v in [output, loss, *terms.values()]):
        raise FloatingPointError('nonfinite reconstruction forward/loss')
    return loss, terms, output


def point_report(xy, states, truth, scale):
    sequence = np.column_stack([xy, np.eye(3)[states]])
    row = diagnostics(sequence, truth, scale)
    return {'axes': row['axes'], 'temporal': temporal_errors(xy, truth[:, :2] * scale, truth[:, 2:].argmax(1))}


def evaluate(model, batch, sample, cfg, source_cfg, stage, device):
    with fixed_evaluation(model, cfg['latent_noise_seed'], device):
        sampled_nll, sampled, _ = coordinate_forward(model, batch, source_cfg, device)
        mean = deterministic_forward(model, batch, cfg['model_input_scale'])
        reports = {}; arrays = {}; pens = {}
        for latent, output in [('sampled_seed1042', sampled), ('latent_mean', mean)]:
            length = len(sample[1]); states = output[:, :3].argmax(1)[0, :length].cpu().numpy()
            pens[latent] = boundary_metrics(states, sample[1][:, 2:].argmax(1).numpy())
            for readout, func in [('highest_weight', maximum_xy), ('mixture_expectation', expected_xy)]:
                name = latent + '-' + readout
                xy = func(output)[0, :length].cpu().numpy()
                reports[name] = point_report(xy, states, sample[1].numpy(), cfg['model_input_scale'])
                arrays[name] = (xy, states)
        primary = 'sampled_seed1042-highest_weight' if stage == 'joint' else 'latent_mean-mixture_expectation'
        chosen = sampled if stage == 'joint' else mean
        # Report loss for the exact fixed output, without another random draw.
        target_xy = batch[0][:, :, :2] * cfg['model_input_scale']
        if stage == 'joint':
            target = batch[0][:, :, 2:].argmax(-1); mask = batch[1]
            terms = {'gmm_nll': sampled_nll,
                     'pen_focal': masked_focal(chosen[:, :3].transpose(1, 2), target, mask,
                                              policy_weights(target, mask, POLICIES[1]))}
        else:
            terms = {'expected_xy_mse': masked_xy_mse(expected_xy(chosen), target_xy, batch[1])}
        row = {'primary': primary, 'trajectories': reports, 'pen': pens,
               'fixed_losses': {key: float(v) for key, v in terms.items()},
               'gmm_nll_sampled_report_only': float(sampled_nll),
               'all_component_sigma_x_median': float(torch.nn.functional.softplus(chosen[:, 63:83, :len(sample[1])]).median() + .001),
               'all_component_sigma_y_median': float(torch.nn.functional.softplus(chosen[:, 83:103, :len(sample[1])]).median() + .001)}
    return row, arrays


def experiment(config_path, repo, data_root=None, source_checkpoint=None, pen_checkpoint=None):
    cfg = yaml.safe_load(Path(config_path).read_text()); check_config(cfg)
    path = Path(source_checkpoint or cfg['source_checkpoint'])
    if file_sha(path) != SOURCE_SHA:
        raise ValueError('pinned step-2000 source required')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    if parent['step'] != 2000 or parent['sample_sha256'] != SAMPLE_SHA:
        raise ValueError('source step/sample mismatch')
    model, train, val, source_cfg, root = setup(Path(repo) / cfg['source_config'], repo, data_root)
    try:
        if source_cfg != parent['config'] or file_sha(root / 'manifest.json') != MANIFEST_SHA:
            raise ValueError('source config/dataset mismatch')
        sample = selected_sample(train, cfg['sample_id'])
        if sample_sha(sample) != SAMPLE_SHA:
            raise ValueError('source sample mismatch')
        batch = train.collate_fn([sample])
        bstate = None; bpath = Path(pen_checkpoint or cfg['pen_B_checkpoint'])
        if cfg['joint_pen_initialization'] == 'trained_B_rows':
            if file_sha(bpath) != B_SHA:
                raise ValueError('pinned trained Arm B checkpoint required')
            b = torch.load(bpath, map_location='cpu', weights_only=True)
            assert_nonpen_unchanged(parent['model_state_dict'], b['model_state_dict'])
            bstate = {key: b['model_state_dict'][key][:3].clone() for key in (WEIGHT_KEY, BIAS_KEY)}
        return cfg, model, sample, batch, source_cfg, root, parent, bstate, path
    finally:
        train.hf.close(); val.hf.close()


def initialize(model, parent, stage, cfg, bstate):
    # requires_grad_(False) does not clear prior .grad buffers. A branch reset
    # must clear ALL gradients, including parameters excluded from its optimizer.
    model.zero_grad(set_to_none=True)
    model.load_state_dict(parent['model_state_dict'], strict=True)
    model.requires_grad_(True); freeze_auxiliaries(model)
    if stage == 'joint' and cfg['joint_pen_initialization'] == 'trained_B_rows':
        with torch.no_grad():
            model.transformer_decoder.fc.weight[:3].copy_(bstate[WEIGHT_KEY])
            model.transformer_decoder.fc.bias[:3].copy_(bstate[BIAS_KEY])
    if stage == 'deterministic_mse':
        model.conv_logvar.requires_grad_(False)
        model.eval()  # deterministic latent-mean training; eval still permits autograd
    else:
        model.train()


def identity(config_path):
    return {'config_sha256': file_sha(config_path), 'source_sha256': SOURCE_SHA,
            'sample_sha256': SAMPLE_SHA, 'manifest_sha256': MANIFEST_SHA}


def preflight(config_path, repo, data_root=None, source_checkpoint=None, pen_checkpoint=None):
    torch.set_num_threads(2)
    cfg, model, sample, batch, source_cfg, root, parent, bstate, path = experiment(
        config_path, repo, data_root, source_checkpoint, pen_checkpoint)
    results = {}
    for stage in STAGES:
        initialize(model, parent, stage, cfg, bstate)
        results[stage], _ = evaluate(model, batch, sample, cfg, source_cfg, stage, 'cpu')
    report = {'gpu': False, 'training': False, 'optimizer_steps': 0, **identity(config_path),
              'joint_pen_initialization': cfg['joint_pen_initialization'],
              'initial': results, 'real_points': len(sample[1]),
              'branches_share_geometry_source_not_chained': True}
    (root / 'reconstruction_preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def train_pair(config_path, repo, data_root=None, allow_experimental=False,
               source_checkpoint=None, pen_checkpoint=None, mse_only=False):
    if not allow_experimental:
        raise ValueError('explicit experimental acknowledgement required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for bounded full-VAE diagnostics')
    cfg, model, sample, batch, source_cfg, root, parent, bstate, source_path = experiment(
        config_path, repo, data_root, source_checkpoint, pen_checkpoint)
    gate = json.loads((root / 'reconstruction_preflight.json').read_text())
    if any(gate.get(k) != v for k, v in identity(config_path).items()):
        raise ValueError('CPU preflight must match exact job')
    output_dir = Path(cfg['output_base']) / time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = 'cuda'; model.to(device)
    batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
    aux_before = {k: v for k, v in parent['model_state_dict'].items() if k.startswith(('ocr_model.', 'style_classifier.'))}
    started = time.monotonic(); results = {}
    for stage in (('deterministic_mse',) if mse_only else STAGES):
        initialize(model, parent, stage, cfg, bstate)
        branch = output_dir / stage; branch.mkdir()
        mse = stage == 'deterministic_mse'
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg['mse_lr'] if mse else cfg['joint_lr'],
                                       betas=(.9, .99), weight_decay=cfg['mse_weight_decay'] if mse else source_cfg['weight_decay'])
        restored = None
        if not mse:
            restored = restore_optimizer(optimizer, parent, cfg['joint_lr'])
            torch.set_rng_state(parent['rng_state_cpu']); torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
        else:
            torch.manual_seed(cfg['seed'])
        protected_rows = torch.cat([torch.arange(3), torch.arange(63, 123)])
        row_before = {key: model.state_dict()[key][protected_rows].detach().cpu().clone() for key in (WEIGHT_KEY, BIAS_KEY)}
        step = 0; history = []; fixed = []; baseline = None; stop_reason = 'max_steps'

        def record():
            nonlocal baseline
            row, arrays = evaluate(model, batch, sample, cfg, source_cfg, stage, device)
            current = row['trajectories'][row['primary']]
            if baseline is None:
                baseline = current
            row.update(step=step, stage=stage, geometry_nonregression_gate=geometry_gate(baseline, current))
            fixed.append(row)
            with (branch / 'fixed_metrics.jsonl').open('a') as log:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            for name, (xy, states) in arrays.items():
                # Main MSE views always use TRUE pen states; raw pen states remain
                # available in side-by-side renders/files and are explicitly untrained.
                metrics = row['pen']['latent_mean' if name.startswith('latent_mean') else 'sampled_seed1042']
                render_snapshot(branch, f'step-{step}-{name}', xy, states, sample[1].numpy(),
                                cfg['model_input_scale'], sample[2], metrics)
            print({'stage': stage, 'step': step, 'primary': row['primary'], 'geometry': current,
                   'pen': row['pen'], 'gate': row['geometry_nonregression_gate']}, flush=True)
            return row

        record()
        limit = cfg['mse_steps'] if mse else cfg['joint_steps']
        with (branch / 'metrics.jsonl').open('w') as log:
            for step in range(1, limit + 1):
                if time.monotonic() - started > cfg['max_wall_seconds']:
                    step -= 1; stop_reason = 'combined_wall_time'; break
                optimizer.zero_grad(set_to_none=True)
                loss, terms, _ = objectives(model, batch, cfg, source_cfg, stage, device)
                loss.backward()
                if any(p.grad is not None for module in (model.ocr_model, model.style_classifier) for p in module.parameters()):
                    raise AssertionError('auxiliary gradient leaked')
                if mse:
                    for parameter in (model.transformer_decoder.fc.weight, model.transformer_decoder.fc.bias):
                        if parameter.grad[protected_rows].count_nonzero():
                            raise AssertionError('MSE gradient reached pen/sigma/rho output rows')
                    if any(p.grad is not None for p in model.conv_logvar.parameters()):
                        raise AssertionError('latent variance gradient in deterministic branch')
                norm = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                row = {'step': step, 'loss': float(loss.detach()),
                       **{k: float(v.detach()) for k, v in terms.items()},
                       'gradient_norm': float(norm), 'was_gradient_clipped': float(norm) > cfg['grad_clip'],
                       'auxiliary_gradients_absent': True}
                history.append(row); log.write(json.dumps(row) + '\n'); log.flush()
                if step % cfg['save_every'] == 0:
                    measured = record()
                    if not mse and not measured['geometry_nonregression_gate']['passed']:
                        stop_reason = 'strict_geometry_nonregression_gate_failed'; break
        if fixed[-1]['step'] != step:
            record()
        if not all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in aux_before.items()):
            raise AssertionError('auxiliary state changed')
        if mse:
            if not all(torch.equal(v, model.state_dict()[key][protected_rows].cpu()) for key, v in row_before.items()):
                raise AssertionError('pen/sigma/rho FC rows changed in MSE branch')
            if not all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in parent['model_state_dict'].items() if k.startswith('conv_logvar.')):
                raise AssertionError('latent variance head changed')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'source_geometry_step': 2000, 'diagnostic_updates': step, 'stage': stage,
                    'config': source_cfg, 'diagnostic_config': cfg, 'source_sha256': SOURCE_SHA,
                    'sample_sha256': SAMPLE_SHA, 'joint_pen_initialization': cfg['joint_pen_initialization'] if not mse else 'original_geometry',
                    'experimental': True, 'rng_state_cpu': torch.get_rng_state(),
                    'rng_state_cuda': torch.cuda.get_rng_state_all()}, branch / 'checkpoint.pt')
        result = {'stage': stage, 'steps': step, 'requested_steps': limit, 'stop_reason': stop_reason,
                  'source_sha256': SOURCE_SHA, 'sample_id': cfg['sample_id'],
                  'objectives': {'gmm': not mse, 'pen': not mse, 'expected_xy_mse': mse,
                                 'ctc': False, 'style': False, 'kl': False, 'velocity': False, 'second_difference': False},
                  'learning_rate': cfg['mse_lr'] if mse else cfg['joint_lr'],
                  'optimizer': 'fresh AdamW' if mse else 'restored step-2000 geometry AdamW',
                  'restored_optimizer': restored, 'training_mode': 'eval-mode, latent mean, gradients enabled' if mse else 'train-mode, sampled latent',
                  'pen_initialization': cfg['joint_pen_initialization'] if not mse else 'original_geometry',
                  'primary_render': 'mixture expectation with TRUE pen' if mse else 'max-pi mean with predicted pen',
                  'initial': fixed[0], 'final': fixed[-1], 'auxiliary_state_unchanged': True,
                  'mse_pen_sigma_rho_rows_unchanged': True if mse else None,
                  'clipping': clipping_summary(history, cfg['grad_clip']), 'experimental': True,
                  'promoted_for_ctc': False, 'human_realism_gate': 'pending visual review'}
        if not mse:
            result['compatibility_gate'] = (fixed[-1]['geometry_nonregression_gate']['passed'] and
                fixed[-1]['pen']['sampled_seed1042']['pen_up_f1'] > fixed[0]['pen']['sampled_seed1042']['pen_up_f1'])
        results[stage] = result
        (branch / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if stop_reason == 'combined_wall_time':
            break
    if file_sha(source_path) != SOURCE_SHA:
        raise AssertionError('source checkpoint changed')
    summary = {'training': True, 'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
               'output': str(output_dir), 'elapsed_seconds': time.monotonic() - started,
               'source_checkpoint_unchanged': True, 'source_sha256': SOURCE_SHA,
               'branches_independent_not_chained': True, 'mse_only': mse_only, 'ctc_style_kl_off': True,
               'derivatives_diagnostic_only': True, 'results': results, **identity(config_path)}
    (output_dir / 'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    repo = 'third_party/DiffInk' if Path('third_party/DiffInk').exists() else '.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=repo)
    parser.add_argument('--config', default=repo + '/configs/vae_iam_reconstruction.yaml')
    parser.add_argument('--data-root'); parser.add_argument('--source-checkpoint'); parser.add_argument('--pen-checkpoint')
    args = parser.parse_args()
    print(json.dumps(preflight(args.config, args.repo, args.data_root, args.source_checkpoint, args.pen_checkpoint), indent=2))


if __name__ == '__main__':
    main()
