"""Bounded T4 utilization measurement. No optimizer or checkpoint writes."""
"""Bounded existing-data geometry pilot; no OCR/KL/style or full-IAM training."""
from pathlib import Path
import modal

volume=modal.Volume.from_name('diffink-data')
repo=Path('third_party/DiffInk') if Path('third_party/DiffInk').is_dir() else Path('.')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','numpy==2.5.3','h5py==3.16.0','Pillow==12.3.0','matplotlib==3.11.2','PyYAML==6.0.3')
       .workdir('/app')
       .add_local_dir(str(repo/'model'),'/app/model')
       .add_local_dir(str(repo/'dataset'),'/app/dataset')
       .add_local_dir(str(repo/'utils'),'/app/utils')
       .add_local_dir(str(repo/'trainer'),'/app/trainer')
       .add_local_dir('iam_tools','/app/iam_tools')
       .add_local_file(str(repo/'configs/engineering_english.yaml'),'/app/configs/engineering_english.yaml'))
SOURCE='checkpoints/iam_fullset_joint/20261006-140945/checkpoint-best.pt'
SHA='89ec459de6496275a3712c08629daea10d8f4f03311712c77f49e652bca915a3'

app = modal.App('diffink-english-no-update-profile')


@app.function(image=image, volumes={'/data': volume}, gpu='T4', cpu=4,
              memory=16384, timeout=300, retries=0, max_containers=1)
def profile():
    import json
    import subprocess
    import threading
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from iam_tools.writer_expansion import load, device_batches, evaluate
    from iam_tools.latent_integration import terms, encoded
    from iam_tools.trajectory_geometry import geometry_metrics
    from iam_tools.pen_ab import file_sha
    from model.losses import mixture_expectation

    torch.set_num_threads(4)
    torch.manual_seed(4042)
    model, samples, raw, cfg, vocab, provenance = load(
        '/app/configs/engineering_english.yaml', '/app', '/data', SOURCE, SHA,
        writer_id=None)
    model.to('cuda').eval()
    model.ocr_model.requires_grad_(False)
    model.style_classifier.requires_grad_(False)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    batches = device_batches(raw, 'cuda')
    ids = provenance['splits']['train'][:8]
    path = Path('/data/checkpoints/iam_performance_profile') / time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    path.mkdir(parents=True, exist_ok=False)
    gpu_samples = []
    phase = ['warmup']
    stop = threading.Event()

    def sample_gpu():
        while not stop.is_set():
            try:
                values = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
                gpu_samples.append(dict(phase=phase[0], time=time.time(),
                                        utilization=float(values[0]), memory_utilization=float(values[1]),
                                        power_watts=float(values[2])))
            except Exception as exc:
                gpu_samples.append(dict(phase=phase[0], error=str(exc)))
            stop.wait(.2)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    rows = {}

    def measure(name, fn):
        torch.cuda.synchronize()
        phase[0] = name
        wall, cpu = time.perf_counter(), time.process_time()
        value = fn()
        torch.cuda.synchronize()
        rows[name] = dict(wall_seconds=time.perf_counter()-wall,
                          process_cpu_seconds=time.process_time()-cpu, result=value)
        phase[0] = 'between_phases'

    def training(sync_scalars, updates=32):
        # Same mean + sampled-z geometry and bounded pen computation as pilot.
        # Gradients discarded; parameters and optimizer state never updated.
        last = None
        for _ in range(updates):
            model.zero_grad(set_to_none=True)
            totals = None
            for sid in ids:
                t = terms(model, batches[sid], pen_on_mean=True)
                loss = (t['mean_geometry'] + .1*t['sampled_geometry'] + .0016613753687545062*t['pen'])/8
                if sync_scalars and not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite profile loss')
                loss.backward()
                metrics = torch.stack([t[k].detach() for k in ('mean_geometry', 'sampled_geometry', 'pen')])
                if sync_scalars:
                    last = [float(v) for v in metrics]
                else:
                    totals = metrics if totals is None else totals+metrics
            if not sync_scalars:
                last = totals.cpu().tolist()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5, error_if_nonfinite=True)
            float(norm)
        model.zero_grad(set_to_none=True)
        return dict(updates=updates, microbatches=updates*8, last_metrics=last,
                    note='No optimizer steps; fixed eight representative lines, not a throughput benchmark for all lengths')

    try:
        training(True, 2)
        # Repeated in reverse order too, to expose clock/order effects.
        for name, sync in [('training_as_run', True), ('training_deferred_logging', False),
                           ('training_deferred_logging_repeat', False), ('training_as_run_repeat', True)]:
            torch.manual_seed(4042)
            measure(name, lambda sync=sync: training(sync))

        saved_arrays = []

        def gpu_evaluation():
            with torch.no_grad():
                for sid in ids:
                    raw, mask, _ = batches[sid]
                    truth, mu, lv, lm = encoded(model, raw, mask)
                    n = int(mask.sum())
                    target = truth[0, :n].cpu().numpy()
                    states = raw[0, 2:, :n].argmax(0).cpu().numpy()
                    for k in range(21):
                        z = mu if k == 0 else mu+torch.randn_like(mu)*(.5*lv).exp()
                        out = model.decode(z, padding_mask=~mask)
                        xy = mixture_expectation(out)[0, :n].cpu().numpy()
                        # OCR included, matching real evaluation's inference work.
                        model.ocr_model(z)[:int(lm.sum()), 0].argmax(-1).tolist()
                        saved_arrays.append((xy, target, states))
            return dict(lines=8, trajectories=len(saved_arrays), note='Inference + CPU transfer, no NumPy metrics or file writes')

        measure('evaluation_inference_only', gpu_evaluation)

        def cpu_metrics():
            for _ in range(2):
                for xy, target, states in saved_arrays:
                    geometry_metrics(xy, target, states)
            return dict(trajectories=len(saved_arrays)*2, note='Serial production geometry metrics, GPU idle by design')

        measure('evaluation_cpu_geometry_only', cpu_metrics)
        measure('evaluation_as_run', lambda: evaluate(
            model, {i:samples[i] for i in ids}, {i:batches[i] for i in ids},
            {'train':ids}, vocab, path, 0, draws=20)['groups']['train']['mu_macro_pen_f1'])
        if not all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items()):
            raise AssertionError('Profiling changed weights')
        if file_sha(Path('/data')/SOURCE) != SHA:
            raise AssertionError('Source changed')
    finally:
        stop.set()
        sampler.join(timeout=5)
    for name, row in rows.items():
        selected = [s for s in gpu_samples if s['phase']==name and 'utilization' in s]
        row['cpu_core_equivalents'] = row['process_cpu_seconds']/row['wall_seconds']
        row['gpu_utilization_samples'] = len(selected)
        row['mean_gpu_utilization_percent'] = float(np.mean([s['utilization'] for s in selected])) if selected else None
    result = dict(source_rel=SOURCE, source_sha256=SHA, weights_unchanged=True,
                  optimizer_updates=0, torch_threads=torch.get_num_threads(), rows=rows,
                  gpu_samples=gpu_samples,
                  caveat='nvidia-smi utilization is a coarse sampled busy-time estimate, not throughput or SM occupancy; timings include sampling overhead.')
    (path/'profile.json').write_text(json.dumps(result, indent=2)+'\n')
    (path/'as-run-launcher.py').write_bytes(Path(__file__).read_bytes()) if Path(__file__).exists() else None
    volume.commit()
    return dict(output=str(path), rows=rows, weights_unchanged=True, optimizer_updates=0)


@app.local_entrypoint()
def main(run: bool = False):
    if not run:
        print('No GPU allocated. Explicit --run required for no-update profiling.')
        return
    print(profile.remote())
