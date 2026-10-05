"""Matched deterministic geometry arms, plus independent frozen-feature pen repair.

No KL/OCR/style/GMM or augmentation. Starting weights/data/evaluation RNG and
optimizer settings are identical. Only target-difference weight changes.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .objective_study import load
from .pen_ab import file_sha, boundary_metrics
from .trajectory_geometry import geometry_metrics

SOURCE_REL = 'checkpoints/iam_lbfgs_geometry/20261005-113159/pen_refit/checkpoint.pt'
SOURCE_SHA = '700f84eda8b523917f0c7f337bc21f75cc6d4075a101edc979fa5432212cd4f8'


def forward_xy(model, raw, mask, sampled=False):
    from model.losses import mixture_expectation
    target = model.to_model_space(raw)
    encoded = model.encoder(target)
    mu = model.conv_mu(encoded)
    z = mu
    if sampled:
        z = mu + torch.randn_like(mu)*(.5*model.conv_logvar(encoded)).exp()
    output = model.decode(z, padding_mask=~mask)
    return mixture_expectation(output), target[:, :2].transpose(1, 2), output


def losses(model, raw, mask, auxiliary='delta'):
    from model.losses import target_difference_loss, target_tangent_loss
    if auxiliary not in ('delta', 'tangent'): raise ValueError('unknown geometry auxiliary')
    xy, truth, _ = forward_xy(model, raw, mask)
    states = raw[:, 2:].argmax(1)
    return {'point': (xy[mask]-truth[mask]).square().mean(),
            'delta': (target_difference_loss if auxiliary == 'delta' else target_tangent_loss)(xy, truth, states, mask)}


def calibration(model, batches, fraction=.2, auxiliary='delta'):
    """Gradient fraction on the full eight-line decoder objective, no .grad writes."""
    if not 0 < fraction <= 1: raise ValueError('initial auxiliary gradient fraction must be in (0,1]')
    params = list(model.decoder.parameters())+list(model.transformer_decoder.parameters())
    totals = {name: [torch.zeros_like(p) for p in params] for name in ('point', 'delta')}
    rows = []
    for raw, mask in batches:
        terms = losses(model, raw, mask, auxiliary)
        norms = {}
        for name in totals:
            grad = torch.autograd.grad(terms[name]/len(batches), params, retain_graph=name == 'point')
            norms[name] = float(torch.stack([g.square().sum() for g in grad]).sum().sqrt())
            for total, g in zip(totals[name], grad): total.add_(g.detach())
        rows.append(norms)
    norms = {name: float(torch.stack([g.square().sum() for g in gs]).sum().sqrt()) for name, gs in totals.items()}
    if min(norms.values()) <= 1e-12: raise ValueError('degenerate calibration')
    cosine = float(torch.stack([(a*b).sum() for a, b in zip(totals['point'], totals['delta'])]).sum()/(norms['point']*norms['delta']))
    return dict(target_fraction=fraction, delta_weight=fraction*norms['point']/norms['delta'], auxiliary=auxiliary,
                aggregate_decoder_norms=norms, cosine=cosine, per_line_decoder_norms=rows,
                method='matched aggregate eight-line decoder gradient norm, fixed initial scalar')


def snapshot(model, samples, batches, cfg, directory, step, sampled_count=0):
    rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for j, (sample, (raw, mask)) in enumerate(zip(samples, batches)):
            torch.manual_seed(1042+j*100)
            folder = directory/f'step-{step}'/cfg['sample_ids'][j]; folder.mkdir(parents=True, exist_ok=True)
            variants = []
            with torch.no_grad():
                for k in range(-1, sampled_count):
                    xy, truth, output = forward_xy(model, raw, mask, sampled=k >= 0)
                    n = int(mask.sum()); xy = xy[0, :n].cpu().numpy(); target = truth[0, :n].cpu().numpy()
                    states = raw[0, 2:, :n].argmax(0).cpu().numpy()
                    predicted = output[0, :3, :n].argmax(0).cpu().numpy()
                    name = 'mu' if k < 0 else f'z-{k}'
                    np.save(folder/f'{name}.npy', np.column_stack([xy, np.eye(3)[predicted]]))
                    variants.append(dict(kind=name, geometry=geometry_metrics(xy, target, states),
                                         pen=boundary_metrics(predicted, states)))
            rows.append(dict(sample_id=cfg['sample_ids'][j], text=sample[2], mu=variants[0], sampled=variants[1:]))
    summary = dict(step=step, lines=rows)
    for key in ('x_rmse', 'y_rmse'):
        summary[key] = float(np.mean([r['mu']['geometry'][key] for r in rows]))
    for key in ('first_difference', 'second_difference'):
        summary[key+'_relative'] = float(np.mean([r['mu']['geometry'][key]['relative_rms_error'] for r in rows]))
    summary['macro_pen_f1'] = float(np.mean([r['mu']['pen']['pen_up_f1'] for r in rows]))
    with (directory/'fixed_metrics.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
    print({k: v for k, v in summary.items() if k != 'lines'}, flush=True)
    return summary


def head_refit(model, batches, updates=2000):
    """Restore pens after geometry-only body updates; no geometry gradient/state change."""
    from .pen_refit import refit_loss
    fc = model.transformer_decoder.fc
    captured = []; hook = fc.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    features, labels = [], []
    try:
        with torch.no_grad():
            for raw, mask in batches:
                forward_xy(model, raw, mask)
                features.append(captured.pop()[0, mask[0]])
                labels.append(raw[0, 2:, mask[0]].argmax(0))
    finally: hook.remove()
    torch.manual_seed(42)
    head = torch.nn.Linear(fc.in_features, 3, device=fc.weight.device)
    with torch.no_grad(): head.weight.copy_(fc.weight[:3]); head.bias.copy_(fc.bias[:3])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, betas=(.9, .99), weight_decay=0)
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([refit_loss(head(f), y, 'bounded_three_state', gamma=2, cap=8) for f, y in zip(features, labels)]).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True); optimizer.step()
    with torch.no_grad(): fc.weight[:3].copy_(head.weight); fc.bias[:3].copy_(head.bias)


def run(config, repo, root='/data', steps=80, arms=('point', 'delta20'), source_rel=SOURCE_REL, source_sha=SOURCE_SHA,
        deterministic=False):
    if not 1 <= steps <= 100 or not arms or any(a not in ('point', 'delta20', 'delta50', 'tangent20') for a in arms):
        raise ValueError('bounded geometry study: 1..100 outer steps, known objective arms')
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise RuntimeError('T4 study needs CUDA')
    torch.set_num_threads(4); torch.manual_seed(42)
    root = Path(root); source = root/source_rel
    if file_sha(source) != source_sha: raise ValueError('pinned source SHA required')
    model, samples, raw_batches, base, hashes, _ = load(config, repo, root/'diffink/iam_overfit', root/'checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    parent = torch.load(source, map_location='cpu', weights_only=True)
    assert parent['sample_ids'] == base['sample_ids']
    initial = parent['model_state_dict']
    model.to('cuda').eval(); model.conv_logvar.requires_grad_(False)
    batches = [(b[0].to('cuda').transpose(1, 2), b[1].to('cuda')) for b in raw_batches]
    directory = root/'checkpoints/iam_curve_study'/time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    directory.mkdir(parents=True, exist_ok=False)
    # Preserve exact as-run code, including uncommitted research edits.
    for relative in ('model/losses.py', 'model/blocks.py'):
        destination = directory/'source-code'/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(repo)/relative).read_bytes())
    for name in ('curve_study.py', 'trajectory_geometry.py'):
        destination = directory/'source-code/iam_tools'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(__file__).parent/name).read_bytes())
    outputs = {}
    for arm in arms:
        torch.manual_seed(42); model.load_state_dict(initial, strict=True); model.apply_checkpoint_contract(parent)
        cfg = dict(base, profile='curve-fidelity-study', training_latent='mean', gmm_weight=0, pen_weight=0,
                   expected_xy_weight=1, target_delta_weight=0, ctc_weight=0, kl_weight=0, style_weight=0,
                   optimizer='LBFGS', base_lr=1., grad_clip=None, weight_decay=0,
                   max_optimizer_updates=steps, max_wall_seconds=900, eval_every=20,
                   source_checkpoint=str(source), source_sha256=source_sha, arm=arm,
                   lbfgs_max_iter=10, lbfgs_max_eval=15, lbfgs_history_size=10,
                   lbfgs_line_search='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
        cfg['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
        cfg['cublas_workspace_config'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        model.config.__dict__.update(cfg)
        hooks = []
        for p in (model.transformer_decoder.fc.weight, model.transformer_decoder.fc.bias):
            active = torch.zeros_like(p); active[3:63] = 1
            hooks.append(p.register_hook(lambda grad, mask=active: grad*mask))
        auxiliary = 'tangent' if arm == 'tangent20' else 'delta'
        cal = calibration(model, batches, fraction=.5 if arm == 'delta50' else .2, auxiliary=auxiliary)
        weight = 0. if arm == 'point' else cal['delta_weight']; cfg['target_delta_weight'] = weight
        cfg['geometry_auxiliary'] = auxiliary
        if auxiliary == 'tangent':
            cfg['target_delta_weight'] = 0; cfg['target_tangent_weight'] = weight
            cfg['tangent_collapsed_prediction_length_floor'] = '.01 times matching target segment length'
        print(dict(arm=arm, calibration=cal), flush=True)
        folder = directory/arm; folder.mkdir()
        (folder/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
        (folder/'calibration.json').write_text(json.dumps(cal, indent=2)+'\n')
        optimizer = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad], lr=1., max_iter=10,
                                     max_eval=15, history_size=10, line_search_fn='strong_wolfe',
                                     tolerance_grad=1e-8, tolerance_change=1e-12)
        calls = 0; last = {}; started = time.monotonic()
        first = snapshot(model, samples, batches, cfg, folder, 0)
        def closure():
            nonlocal calls, last
            optimizer.zero_grad(set_to_none=True); totals = {'point': 0., 'delta': 0.}
            for raw, mask in batches:
                terms = losses(model, raw, mask, auxiliary)
                loss = (terms['point']+weight*terms['delta'])/len(batches)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite geometry loss')
                loss.backward()
                for key in totals: totals[key] += float(terms[key].detach())/len(batches)
            calls += 1; last = totals
            return raw.new_tensor(totals['point']+weight*totals['delta'])
        def save_geometry(step):
            state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), config=cfg,
                         outer_optimizer_steps=step, closure_calls=calls, sample_ids=cfg['sample_ids'], source_sha256=source_sha)
            torch.save(state, folder/'checkpoint-geometry.pt')
        with (folder/'metrics.jsonl').open('w') as log:
            for step in range(1, steps+1):
                optimizer.step(closure); optimizer.zero_grad(set_to_none=True)
                log.write(json.dumps(dict(step=step, closure_calls=calls, **last))+'\n'); log.flush()
                if step%20 == 0:
                    save_geometry(step); snapshot(model, samples, batches, cfg, folder, step)
                if time.monotonic()-started > cfg['max_wall_seconds']: break
        for hook in hooks: hook.remove()
        state = model.state_dict()
        for key, old in initial.items():
            actual = state[key].cpu()
            if key.startswith(('ocr_model.', 'style_classifier.', 'conv_logvar.')):
                assert torch.equal(old, actual), key
            if key in ('transformer_decoder.fc.weight', 'transformer_decoder.fc.bias'):
                assert torch.equal(old[:3], actual[:3]) and torch.equal(old[63:], actual[63:]), key
        saved = dict(model_state_dict=state, optimizer_state_dict=optimizer.state_dict(), config=cfg,
                     outer_optimizer_steps=step, closure_calls=calls, sample_ids=cfg['sample_ids'], source_sha256=source_sha)
        torch.save(saved, folder/'checkpoint-geometry.pt')
        geometry = snapshot(model, samples, batches, cfg, folder, step, sampled_count=20)
        # Body changes can move fixed pen logits. Refit both arms identically,
        # report BEFORE and AFTER separately; never force boundaries for inference.
        before = {k: v.detach().cpu().clone() for k, v in state.items()}
        head_refit(model, batches)
        for key, old in before.items():
            actual = model.state_dict()[key].cpu()
            if key in ('transformer_decoder.fc.weight', 'transformer_decoder.fc.bias'):
                assert torch.equal(old[3:], actual[3:]), key
            else: assert torch.equal(old, actual), key
        final = snapshot(model, samples, batches, cfg, folder, step+2000, sampled_count=20)
        saved['model_state_dict'] = model.state_dict(); saved.pop('optimizer_state_dict')
        saved['head_refit_updates'] = 2000
        torch.save(saved, folder/'checkpoint.pt')
        result = dict(arm=arm, source_sha256=source_sha, provenance=hashes, config=cfg, calibration=cal,
                      initial=first, geometry_final=geometry, final=final, closure_calls=calls, outer_steps=step,
                      geometry_frozen_during_pen_refit=True, frozen_parameter_checks_passed=True,
                      head_refit_config=dict(updates=2000, optimizer='AdamW', lr=.001, betas=[.9, .99],
                                             weight_decay=0, grad_clip=5, focal_gamma=2, bounded_weight_cap=8,
                                             training_latent='cached mean features', geometry_frozen=True),
                      elapsed_seconds=time.monotonic()-started, gpu=torch.cuda.get_device_name(),
                      torch_version=torch.__version__, numpy_version=np.__version__, training_samples_only=True)
        (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        outputs[arm] = dict(output=str(folder), x_rmse=final['x_rmse'], y_rmse=final['y_rmse'],
                            first_difference_relative=final['first_difference_relative'],
                            second_difference_relative=final['second_difference_relative'], macro_pen_f1=final['macro_pen_f1'])
        del optimizer, saved
    code = {f: file_sha(Path(repo)/f) for f in ['model/losses.py', 'model/blocks.py']}
    code.update({f: file_sha(Path(__file__).parent/f) for f in ['curve_study.py', 'trajectory_geometry.py']})
    (directory/'study.json').write_text(json.dumps(dict(outputs=outputs, source_sha256=source_sha,
                                                     code_sha256=code, identical_initialization=True), indent=2)+'\n')
    return outputs
