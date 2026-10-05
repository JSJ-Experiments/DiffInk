"""CPU-only mixture ambiguity / frozen-feature linear-readout hypothesis test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .objective_study import load
from .curve_study import forward_xy
from .pen_ab import file_sha


def linear_readout(features, target):
    """Exact training-set affine least squares; NOT a checkpoint/architecture edit."""
    x = features.double(); y = target.double()
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)
    fit = torch.linalg.lstsq(x, y, driver='gelsd')
    return x@fit.solution, int(fit.rank)


def run(checkpoint, output, root='data', repo='third_party/DiffInk'):
    torch.set_num_threads(2); root, repo = Path(root), Path(repo)
    model, _, batches, cfg, _, _ = load(repo/'configs/engineering_english.yaml', repo,
                                       root/'diffink/iam_overfit', root/'checkpoints/iam_eightline/20261005-102304/checkpoint.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict']); model.apply_checkpoint_contract(saved); model.eval()
    features, targets, predictions, rows, captured = [], [], [], [], []
    hook = model.transformer_decoder.fc.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            for sid, batch in zip(cfg['sample_ids'], batches):
                raw, mask = batch[0].transpose(1, 2), batch[1]
                xy, target, output_now = forward_xy(model, raw, mask)
                feature = captured.pop()[0, mask[0]].double(); target = target[mask].double(); prediction = xy[mask].double()
                pi = output_now[0, 3:23, mask[0]].T.softmax(1)
                entropy = -(pi*pi.clamp_min(1e-30).log()).sum(1).numpy()
                error = torch.linalg.vector_norm(prediction-target, dim=1).numpy()
                high = entropy >= np.quantile(entropy, .9)
                rows.append(dict(id=sid, median_max_pi=float(pi.max(1).values.median()),
                                 entropy_p90=float(np.quantile(entropy, .9)),
                                 error_high_entropy_rms=float(np.sqrt(np.mean(error[high]**2))),
                                 error_rest_rms=float(np.sqrt(np.mean(error[~high]**2)))))
                features.append(feature); targets.append(target); predictions.append(prediction)
    finally: hook.remove()
    target = torch.cat(targets); estimate, rank = linear_readout(torch.cat(features), target)
    result = dict(training=False, gpu=False, source=str(checkpoint), source_sha256=file_sha(checkpoint),
                  mixture_diagnostics=rows,
                  frozen_feature_linear_readout=dict(xy_rmse=((estimate-target).square().mean(0).sqrt()).tolist(),
                                                     current_xy_rmse=((torch.cat(predictions)-target).square().mean(0).sqrt()).tolist(), rank=rank),
                  interpretation='least-squares diagnostic only, no checkpoint/model mutation')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('checkpoint'); p.add_argument('output')
    p.add_argument('--root', default='data'); p.add_argument('--repo', default='third_party/DiffInk')
    a = p.parse_args(); print(json.dumps(run(a.checkpoint, a.output, a.root, a.repo), indent=2))
