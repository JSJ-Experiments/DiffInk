# Native AMD English conditioning investigation

Development is now authoritative at `/home/timothy/code/autowrite`, reached with
`ssh -p 2222 timothy@100.101.148.8`. Do not sync older source code over this tree.
See workspace `docs/TIMOTHY_WSL.md` for ROCm activation and migration caveats.
No new Modal training job is launched by this study.

## Why this is a different question

The unresolved goal is readable handwriting for genuinely new text, not polishing
an eight-line reconstruction. The completed T4 representation/objective factorial
used 8,144 TRAIN lines and 186 writers, but all four arms remained unreadable.
Final DEV32, both noise seeds, guidance1:

| Arm | CER | Median generated lifts | Matched IAM lifts | Severe-underlifting rows |
| --- | ---: | ---: | ---: | ---: |
| full384 x0 | 82.21% | 9.5 | 28 | 52/64 |
| compact40 x0 | 90.39% | 24 | 28 | 1/64 |
| full384 diffusion-v | 86.86% | 24 | 28 | 2/64 |
| compact40 diffusion-v | 88.29% | 23.5 | 28 | 2/64 |

Counts improving did NOT establish text composition. None produced an exact
sentence in these fixed evaluations. Target-informed low-noise denoising also
improved without solving target-free generation; diffusion-v has a low-noise
identity path by construction. Compact changes loss allocation AND nuisance
channels, not just one independent mechanism. No monocausal conclusion follows.
Historical artifacts: `checkpoints/iam_compact_dit/20261008-232330/`.
Full final/selected marker-free galleries and stroke audits exist on the source;
full archive migration and exhaustive secondary/control visual review are not
claimed complete here.

Another task mismatch remains plausible: only 3,211 of the old 12,000 updates
had correct text and no same-target clean prefix. A clean prefix can supply a
form/geometry shortcut. This is evidence for testing task alignment, NOT proof
that prefix conditioning caused failure. Original self-attention already accesses
text globally; joint attention is not a magical guarantee of character composition.
The codec remains initialized polyphase40 transport, NOT learned semantic InkVAE.

## Fresh same-backend comparison

Current native study: `checkpoints/iam_wsl_conditioning/20261010-112127`.
The earlier `20261010-110622` attempt is preserved as failed-before-training.
`iam_tools/wsl_conditioning.py` compares:

- `concat`: original DiT, compact40 latent, concatenated text features.
- `joint`: current `CrossAttentionDiT`, compact40 latent, separate natural-length
  text stream with corrected masks/CFG and per-head rotary positioning.

Both remove clean-prefix support. This directly trains the text-only task used
at inference. Both use the same 8,144 TRAIN lines, frozen posterior cache and
TRAIN whitening, order, per-step full384 posterior/diffusion noise, timesteps,
and text-drop draws. Data RNG is explicitly separate from model/dropout RNG.
Analogous text/time/trajectory parameters are copied bitwise; initialization
mapping is saved. Joint adds context parameters (roughly42.33M vs22.97M), so a
win would not isolate attention from additional capacity. Dropout draws are NOT
claimed identical across different architectures.

Exact shared settings:

```text
latent40; hidden384; depth8; heads6; text192; conv text layers3
float32, dropout .05; batch32; bucket128
AdamW(.9,.99), weight decay .0001
LR5e-5, warmup600, cosine to1e-6, clip1
12,000 updates OR1,800 actual training seconds per arm
prefix retention0, text dropout .1
initial seed84010, data draw seed84012+step, model RNG84013
order seed72144
checkpoints/eval0/1000/3000/6000/12000 (plus actual wall-cap step)
DDIM50, seeds73142/73143, guidance1/2
```

Generation uses requested text, TRAIN-only pooled duration and random noise.
No target trajectory, target length, writer ID, forced EOC or smoothing.
Every checkpoint includes correct/swapped/NULL controls, pen-lift/stroke audits,
and explicitly target-informed denoising diagnostics. DEV guidance1/both-seeds
selects; no previous fresh/held confirmation is reopened. DEV is repeatedly
exposed research data, not a newly independent confirmation set.
No KL/CTC/style/codec updates. A lower loss, better count, or low-noise identity
is not a readability/composition win. Severe under-lifting is a quality failure
(<half paired-source lift density both per character and per point). Counts per
point are not physical rates because IAM/RDP spacing is nonuniform. Generic
writer styles need not reproduce one source writer's exact count.

All required historical inputs are SHA-checked before preparation/each arm.
The historical source archive is hashed as provenance, not executed or changed.
Current source is separately archived and checked. Do NOT demand T4/AMD bitwise
training equivalence or overwrite historical configs/checkpoints.

## Backend checks and performance

Actual AMD forward/backward/AdamW and CFG/NULL preflight passes for both designs;
text embedding gradients become nonzero after the zero-initialized gates open.
This is synthetic execution evidence, NOT handwriting quality.

A synthetic worst-case batch32,104 latent blocks,80 text tokens benchmark gave
warm update means .106s concat / .174s joint and peak allocation1.56GB /2.51GB,
without the corpus pool/reader/codec allocations. These benchmark token counts
use one fewer embedding row than the final vocabulary; actual as-run parameter
counts/config are saved separately. No AMP or experimental attention flag enabled.
No guarantee of identical full-run throughput or VRAM peaks.

Evidence on the workstation:
`logs/wsl-conditioning-preflight.json`, `logs/wsl-conditioning-benchmark.json`,
`logs/wsl-conditioning-full-tests.log` (788 root tests /752 fork tests +11 subtests each PASS).
NVIDIA utilization monitoring is intentionally disabled. `rocm-smi` cannot
initialize the Linux amdgpu driver under this ROCDXG setup, so do not invent AMD
GPU-busy percentages. Log synchronized actual step times, CPU phases and PyTorch
VRAM allocations instead. Official tagged PyTorch RNG documentation is preserved
in `logs/wsl-torch-rng-docs.json` (Context7 CLI unavailable on this workstation;
rendered docs returned403, tagged official source was retrieved successfully).

## Run safely

```sh
source scripts/activate_rocm.sh
python -m iam_tools.wsl_conditioning --preflight
python -m iam_tools.wsl_conditioning --prepare   # CPU preparation; needs verified inputs
scripts/run_wsl_conditioning.sh checkpoints/iam_wsl_conditioning/<prepared-study>
```

Coordinator runs arms serially on ONE RX7800XT. A persistent flock plus exclusive
per-arm start status prevents speculative restarts. A started/noncomplete status
requires investigation, not resubmission. Training lives in tmux session
`iam-wsl-conditioning`; log `logs/wsl-conditioning-run.log`. Observe that SAME
session/study. Importing the module or invoking it without explicit mode starts
no training. Status/PID files and exclusive arm directories are durable; do not
restart merely because SSH polling timed out.

At launch, no handwriting improvement is claimed yet. Inspect all48 fixed prompts,
marker-free renders, content controls and stroke-density failures before deciding
whether joint conditioning or text-only training helped. Semantic representation
and monotonic alignment remain open hypotheses if this comparison also fails.


### ROCm packed-GRU integration fix

The first native attempt failed at step0 evaluation, **before any corpus training
updates**, with `miopenStatusUnknownError` from the frozen packed-GRU reader.
Its config/source/initial checkpoint/partial outputs/log remain preserved, with
explicit `failed_before_training` status. No ambiguous retry or overwritten study.

The new study explicitly uses `reader_backend=aten_gpu_no_miopen` via
`ATenFrozenReader`: vendor RNN dispatch disabled ONLY around the frozen reader
forward, flags restored on exit/exception. Exact weights/features/masks remain;
no CPU fallback and no reader/OCR training. Other model kernels are unaffected.
Official tagged PyTorch flags implementation is preserved in
`logs/torch-cudnn-flags-source.py`.

Actual GPU-vs-CPU checks: synthetic packed two-length batch max logit delta8.58e-6;
all48 cached source probes max3.24e-5, **all decoded strings identical**, TRAIN16
CER0%, DEV32 CER8.6912%. Source posterior inputs, not free-generation evidence
or a new independent benchmark. See `logs/rocm-reader-source-parity.json` and
`logs/rocm-reader-fallback-synthetic.json`. Five regression tests cover frozen
parameters, mask forwarding, disabled training dropout, flags restored on both
success/exception, and no hidden retries. Fresh corrected study was separately
prepared/archived rather than mutating the failed source archive.
