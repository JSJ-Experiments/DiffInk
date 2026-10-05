"""Eight-line objective comparison; continuing research, not paper reproduction.

Both arms start from exactly the same saved 200-update checkpoint and train
sampled latents with identical LR, data order, pen weight and XY weight. The
single changed objective coefficient is GMM NLL 0 versus 1.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .eightline import load as load_base, evaluate
from .pen_ab import file_sha

SOURCE = '/data/checkpoints/iam_eightline/20261005-102304/checkpoint.pt'
SOURCE_SHA = '9d2996271cbcbfa36b1439507c9ed82c1245ed27bfd69c2bbc492473f8b596f4'
ARMS = {'xy_pen': 0., 'gmm_xy_pen': 1.}


def arm_config(base, arm, steps=1000):
    if arm not in ARMS: raise ValueError('unknown objective arm')
    if not 1 <= steps <= 5000: raise ValueError('research chunk must be 1..5000 updates')
    cfg = dict(base)
    cfg.update(profile='english-objective-study', gmm_weight=ARMS[arm], expected_xy_weight=100.,
               anchor_gradient_fraction=0., pen_weight=1., base_lr=5e-5,
               weight_decay=0., max_optimizer_updates=steps, eval_every=250,
               max_wall_seconds=2400, source_checkpoint=SOURCE, source_sha256=SOURCE_SHA,
               output_base='/data/checkpoints/iam_objective_study', objective_arm=arm,
               training_latent='sampled', initialization='identical-step-200-model-fresh-Adam')
    return cfg


def load(config, repo, data_root=None, checkpoint=None):
    source = Path(checkpoint or SOURCE)
    refit = source.parents[2] / 'iam_autopsy/pen_refit/20261005-100653/bounded_three_state/checkpoint.pt'
    model, samples, batches, cfg, root, _, hashes = load_base(config, repo, data_root, refit)
    if file_sha(source) != SOURCE_SHA: raise ValueError('wrong step-200 checkpoint')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.apply_checkpoint_contract(saved)
    assert saved['sample_ids'] == cfg['sample_ids'] and saved['optimizer_updates'] == 200
    hashes['study_source_sha256'] = SOURCE_SHA
    return model, samples, batches, cfg, hashes, source


def gradient_diagnostics(model, batches, cfg, device):
    """Same stochastic forward per line; decoder gradients, not loss magnitudes."""
    from model.losses import loss_terms
    from utils.mask import downsample_mask
    params = list(model.decoder.parameters()) + list(model.transformer_decoder.parameters())
    rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        torch.manual_seed(1042)
        for batch in batches:
            data, mask, text, _, writer = batch
            data = data.to(device).transpose(1, 2); mask = mask.to(device)
            out, ctc, kl, style = model(data, downsample_mask(mask, 8), text.to(device), writer.to(device), False, False, point_mask=mask)
            terms = loss_terms(out, model.to_model_space(data), mask, model.config, ctc, kl, style)
            gradients = {}
            for name in ('gmm', 'expected_xy', 'pen'):
                gs = torch.autograd.grad(terms[name], params, retain_graph=True, allow_unused=True)
                gradients[name] = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)]
            norm = lambda a: torch.stack([x.square().sum() for x in a]).sum().sqrt()
            norms = {k: float(norm(v)) for k, v in gradients.items()}
            def cosine(a, b):
                dot = torch.stack([(x*y).sum() for x, y in zip(gradients[a], gradients[b])]).sum()
                return float(dot / max(norms[a]*norms[b], 1e-20))
            rows.append(dict(unweighted_decoder_norms=norms,
                             weighted_decoder_norms={k: norms[k]*cfg[k+'_weight'] for k in norms},
                             gmm_xy_cosine=cosine('gmm', 'expected_xy'), pen_xy_cosine=cosine('pen', 'expected_xy'),
                             fixed_losses={k: float(v.detach()) for k, v in terms.items()}))
    assert all(p.grad is None for p in model.parameters())
    return rows


def preflight(config, repo, data_root, checkpoint):
    torch.set_num_threads(2)
    model, samples, batches, cfg, hashes, _ = load(config, repo, data_root, checkpoint)
    model.train()
    cfg = arm_config(cfg, 'xy_pen'); model.config.__dict__.update(cfg)
    return dict(gpu=False, optimizer_updates=0, source_sha256=SOURCE_SHA, provenance=hashes,
                gradient_diagnostics=gradient_diagnostics(model, batches, cfg, 'cpu'),
                initial=evaluate(model, samples, batches, cfg, 'cpu', 0, sampled_count=2))


def run(config, repo, arm, steps=1000, resume_checkpoint=None, resume_sha=None):
    if not torch.cuda.is_available(): raise RuntimeError('T4 job requires CUDA')
    from trainer.vae_trainer import train_vae_one_epoch
    model, samples, batches, cfg, hashes, source = load(config, repo)
    cfg = arm_config(cfg, arm, steps); model.config.__dict__.update(cfg)
    parent = None; source_digest = SOURCE_SHA; previous_updates = 0
    if resume_checkpoint:
        source = Path(resume_checkpoint)
        if not resume_sha or file_sha(source) != resume_sha: raise ValueError('pinned resume SHA required')
        parent = torch.load(source, map_location='cpu', weights_only=True)
        for key in ('sample_ids', 'gmm_weight', 'expected_xy_weight', 'pen_weight', 'model_input_scale', 'trans_dropout'):
            if parent['config'][key] != cfg[key]: raise ValueError('resume objective/contract mismatch: '+key)
        model.load_state_dict(parent['model_state_dict'], strict=True)
        previous_updates = parent.get('total_optimizer_updates', parent['optimizer_updates'])
        source_digest = resume_sha
        cfg.update(base_lr=1e-5, source_checkpoint=str(source), source_sha256=resume_sha,
                   previous_optimizer_updates=previous_updates, initialization='resume-model-Adam-and-training-RNG',
                   previous_objective_updates=previous_updates)
        model.config.__dict__.update(cfg)
    model.to('cuda').train()
    frozen = {k: v.cpu().clone() for k, v in model.state_dict().items() if k.startswith(('ocr_model.', 'style_classifier.'))}
    directory = Path(cfg['output_base']) / time.strftime('%Y%m%d-%H%M%S', time.gmtime()) / arm
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    torch.manual_seed(42)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['base_lr'], betas=cfg['betas'], weight_decay=0.)
    if parent is not None:
        optimizer.load_state_dict(parent['optimizer_state_dict'])
        for group in optimizer.param_groups: group['lr'] = cfg['base_lr']
    diagnostics = {'initial': gradient_diagnostics(model, batches, cfg, 'cuda')}
    if parent is None: torch.manual_seed(42)
    else:
        torch.set_rng_state(parent['rng_state_cpu']); torch.cuda.set_rng_state_all(parent['rng_state_cuda'])
    started = time.monotonic()
    fixed = [evaluate(model, samples, batches, cfg, 'cuda', 0, directory)]
    class Loader:
        def __len__(self): return steps * 8
        def __iter__(self):
            rng = np.random.default_rng(42)
            for _ in range(previous_updates): rng.permutation(8)
            for _ in range(steps):
                for j in rng.permutation(8): yield batches[int(j)]
    class TimeLimit(Exception): pass
    def save(step):
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), config=cfg,
                     optimizer_updates=step, total_optimizer_updates=previous_updates+step,
                     source_sha256=source_digest, sample_ids=cfg['sample_ids'],
                     rng_state_cpu=torch.get_rng_state(), rng_state_cuda=torch.cuda.get_rng_state_all())
        torch.save(state, directory/'checkpoint.pt')
        if step % cfg['eval_every'] == 0: torch.save(state, directory/f'checkpoint-{step}.pt')
    log = (directory/'metrics.jsonl').open('w')
    def callback(row):
        step = row['optimizer_step']; log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
        if step == int(.8*steps):
            for group in optimizer.param_groups: group['lr'] = 1e-6 if parent is not None else 1e-5
        if step % 100 == 0: print(dict(arm=arm, **row), flush=True)
        if step % cfg['eval_every'] == 0:
            save(step); fixed.append(evaluate(model, samples, batches, cfg, 'cuda', step, directory))
        if time.monotonic()-started > cfg['max_wall_seconds']: raise TimeLimit()
    reason = 'research_chunk_completed'
    try:
        history = train_vae_one_epoch(model, model.config, Loader(), optimizer, None, 0, 1, 'cuda', on_optimizer_step=callback, max_optimizer_updates=steps)
        updates = len(history)
    except TimeLimit:
        reason = 'wall_time'; updates = len((directory/'metrics.jsonl').read_text().splitlines())
    finally: log.close()
    save(updates)
    if fixed[-1]['step'] != updates: fixed.append(evaluate(model, samples, batches, cfg, 'cuda', updates, directory))
    diagnostics['final'] = gradient_diagnostics(model, batches, cfg, 'cuda')
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in frozen.items())
    assert file_sha(source) == source_digest
    (directory/'gradient_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    result = dict(arm=arm, gpu=torch.cuda.get_device_name(), optimizer_updates=updates, physical_microbatches=8*updates,
                  training_samples_only=True, auxiliary_state_unchanged=True, source_unchanged=True,
                  initial=fixed[0], final=fixed[-1], output=str(directory), elapsed_seconds=time.monotonic()-started,
                  stop_reason=reason, loss_log_scope='effective_batch_mean', config=cfg, provenance=hashes,
                  source_sha256=source_digest, total_optimizer_updates=previous_updates+updates,
                  stage='reconstruction-objective-study-no-CTC')
    (directory/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    page = ['<!doctype html><meta charset="utf-8"><h1>'+arm+'</h1><p>Eight training lines. No CTC/style/KL. Sampled-latent training, expectation rendering.</p>']
    for row in fixed:
        page.append(f'<h2>Step {row["step"]}</h2>')
        for line in row['lines']:
            for label in ('mu', 'sampled-median', 'sampled-worst'):
                page.append(f'<img width="100%" src="step-{row["step"]}/{line["sample_id"]}/{label}-comparison.png">')
    (directory/'index.html').write_text('\n'.join(page))
    return {k: v for k, v in result.items() if k not in ('initial', 'final')}


if __name__ == '__main__':
    default_repo = Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', default=str(default_repo))
    parser.add_argument('--data-root', default='data/diffink/iam_overfit')
    parser.add_argument('--config', default=str(default_repo/'configs/engineering_english.yaml'))
    parser.add_argument('--checkpoint', default='data/checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    args = parser.parse_args()
    print(json.dumps(preflight(args.config, args.repo, args.data_root, args.checkpoint), indent=2))
