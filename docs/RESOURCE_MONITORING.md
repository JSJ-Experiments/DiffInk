# Phase-aware experiment telemetry and resource scaling

`iam_tools.resource_monitor.ResourceMonitor` is a dependency-free, best-effort
background monitor. It **does not scale resources or stop training**.
Currently wired into `generation_benchmark` and `generation_refinement`;
older research launchers do not automatically gain telemetry.

```python
with ResourceMonitor(output, cpu_request=4, memory_request_mib=16384) as monitor:
    monitor.set_phase('prepare')
    # prepare/cache inputs
    monitor.set_phase('train/main')
    # after each update: monitor.step(elapsed_seconds, examples=batch_size)
    with monitor.in_phase('eval/main'):
        # evaluation; no low-GPU alerts here
        ...
```

## Meaning and limitations

- `cpu_cores`: process-tree CPU seconds / elapsed wall seconds. Per-thread
  hottest usage also logged; newly born/exited workers can be undercounted.
  NOT host-wide CPU percentages. Four requested CPUs are a Modal reservation,
  not necessarily the container's actual CPU limit. PyTorch thread count is
  separately configurable; it doesn't parallelize arbitrary Python dispatch.
- RSS sums processes and may count shared pages multiple times. Compare with
  a trustworthy container memory counter before scaling from RSS alone.
- Cgroup membership/raw usage/throttling/quota/memory counters collected where
  available. Scope is **unverified**; no alert assumes those are your container.
  This Modal run exposed no usable cgroup-v2 membership; no throttling conclusion.
- NVIDIA GPU busy%, memory activity%, VRAMMiB and powerW are distinct metrics.
  Low VRAM occupancy is not GPU underutilization. Reject ambiguous multi-GPU
  queries unless an explicit GPU UUID is provided; do not attribute host GPUs.
- Step latency measures host update wall time including the runner's scalar
  synchronization. It is not CUDA kernel time or cold-start-inclusive runtime.
- Alerts require a homogeneous TRAIN phase with a sustained30-second window;
  phase changes reset it. Current implementation rejects sparse/gapped windows.
  Startup, evaluation, sampling/report and CPU-only phases don't emit idle-GPU
  alerts. Memory pressure is also monitored during TRAIN.
- Console alerts are immediate and persisted to `alerts.jsonl`. `resources.jsonl`
  contains raw evidence; `resource-summary.json` aggregates each phase and step
  throughput/p50/p90. No webhook/email destination has been configured.
- Missing nvidia-smi/cgroups/telemetry storage must not abort training.

## Measured T4 benchmark, 2026-10-07

Artifacts: `checkpoints/iam_resource_benchmark/20261007-155649/`.
Five warmed45-second cases, one run each. Same parent text8000 prototype/FP32,
32source lines, restored Adam state/LR1e-4. No benchmark weights promoted.

| Case | CPU threads | Batch | examples/s | GPU busy mean | process CPU cores |
|---|---:|---:|---:|---:|---:|
| live input + per-step noise digest | 4 | 8 | 280.2 | 32.8% | 0.985 |
| cached input + noise digest | 4 | 8 | 293.5 | 33.5% | 0.982 |
| cached input, no per-step digest copy | 4 | 8 | 321.5 | 36.4% | 0.984 |
| same, 1 PyTorch CPU thread | 1 | 8 | 315.8 | 35.9% | 0.989 |
| same, larger physical batch | 1 | 32 | 1218.3 | 38.8% | 0.982 |

Cache alone +4.7%; removing the digest copies adds another9.5%; together+14.7%.
One vs four CPU threads differs1.8% in this single trial—not evidence that the
three idle cores need more work. No single thread was pegged: hottest means
0.56–0.62cores. This is consistent with small-kernel/dispatch/synchronization
limits, **not yet proven by a CPU/CUDA profiler**. Batch32 processes3.79times as
many examples/s versus optimized batch8,4.35times versus original live batch8.
GPU remains underused; the monitor correctly warned on all5sustained phases.

Memory: CUDA allocator peak149.3MiB at batch32 (device VRAM~320MiB); process RSS
~4.6GiB, mostly non-trajectory memory/runtime. Larger GPU memory or more CPU
reservation isn't indicated by this test. CUDA graphs/compilation/fused dispatch
and a larger batch sweep are reasonable follow-ups. **Do not change quality
experiments' batch size without a matched optimization comparison**: examples/s
is not convergence/s, and repeating32lines to fill a larger batch adds no data.
Don't jump to L4 or additional GPU containers just to fill idle resources.
Use per-phase evidence and time-to-quality to choose CPU/GPU/batch/parallelism.

As-run monitor snapshots are immutable. The reusable monitor subsequently adds
stricter gap/coverage and mean-VRAM guards; benchmark/continuation snapshots
retain their exact initial implementation.
