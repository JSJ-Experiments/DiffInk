# English IAM-OnDB → DiffInk

This workspace implements IAM parsing, experimental preprocessing, HDF5 export,
and bounded T4 reconstruction/OCR mechanics experiments. **Full IAM/InkDiT
training has not run.** The later [curve fidelity investigation](CURVE_FIDELITY.md)
passes the eight-memorized-line geometry gate. The subsequent
[bounded sampled/KL/OCR integration](LATENT_INTEGRATION.md) also passes, with
faithful geometry, perfect pen boundaries and CER0 on means/all160 sampled
latents. Neither is a generalization/full-training gate; earlier audit status
remains historical.

Fork: https://github.com/JSJ-Experiments/DiffInk/tree/english-iam

This branch is based on upstream commit
`97bc6a3c39a5bdaa9728daaab6d3707480006343`. Our English compatibility
patch is saved in `patches/diffink-english.patch`.

## Read this caveat first

The released repo does **not** include its raw-data converter or coordinate
normalization formula. Our aspect-preserving line-height=100 normalization is
an explicitly experimental baseline, not verified paper-equivalent. RDP uses
0.5 in those units. IAM line XML has no usable character indices in the samples
we inspected, so the dataset stores empty `char_points_idx`, explicit stroke
ends, and requires our stroke-prefix fallback for DiT. The third pen-state class
is used only at the final line point; exact English end-state equivalence is
also unverified. See [the audit](DIFFINK_AUDIT.md).

## What works

- IAM CSR transcript parsing, exact 1-based line pairing, form→writer lookup.
- Canonical raw JSON keeps x,y,time and original stroke/point ordering.
- Absolute N×5 coordinates, aspect-preserving normalization, y flipped up,
  per-stroke RDP, retained acquisition times, no pen-up bridges.
- 250 training + 50 validation lines. Five validation writers and 25 untouched
  future-test writers are excluded from training; all splits are writer-disjoint.
- Vocabulary comes only from safely paired training transcripts, excluding both
  validation and future-test writers, never the Chinese `All_zi.json`.
- Released **TrainDataset is unchanged** and successfully loads/collates a batch.
- English validation token, stroke-prefix masking, inference preview selection,
  and exact CTC feasibility have small explicit upstream patches.

## View the outputs

- `data/canonical/iam/preview/index.html`: the original 10 raw line previews.
- `data/diffink/iam/previews/index.html`: raw **above** / encoded N×5 **below**.
- `data/diffink/iam/tiny_train.h5`, `tiny_val.h5`, `chars.json`, `writers.json`.
- `data/diffink/iam/manifest.json`: split IDs, preprocessing policy, rejected data.
- `data/diffink/iam/batch_check.json`: CPU loader results and padding/mask checks.
- `data/canonical/iam/tiny/`: unmodified raw canonical samples.

Open a gallery file in a browser, or use the PNG files directly. Display scaling
never changes canonical coordinates. Some source CSR labels contain typos or
omit punctuation visible in strokes; we preserve annotations, not silently
replace them with the differently wrapped OCR prompt.

## Local commands (existing venv)

```sh
source venv/bin/activate
pip install -r requirements-iam.txt
pip install -r requirements-cpu.txt  # CPU-only PyTorch; loader checks only

python -m iam_tools.preview --limit 6
python -m iam_tools.build
python -m iam_tools.check_batch --repo .
python publish_iam.py  # persist validated data on Modal, never GitHub
python -m unittest discover -s tests -v
```

The full raw archives have been downloaded and extracted locally. All persistent
Volume originals are retained. The build is deterministic with seed 42. Samples
outside the loader length limits, with validation OOV, ambiguous transcripts,
or infeasible CTC targets are explicitly reported rather than padded/truncated
to force acceptance. A rerun overwrites generated HDF5/metadata, not raw IAM.

## Modal: no SSH needed

The `diffink-data` Volume persists independently of a shell. The same root is
mounted as `/mnt/diffink-data` by `modal shell`, and `/data` by our functions.
Neither mount automatically appears on this machine. Use the CLI locally:

```sh
modal volume ls diffink-data diffink/iam
modal shell --volume diffink-data
# Inside a NEW shell:
ls /mnt/diffink-data/diffink/iam
```

Repeatable CPU-only remote jobs (no torch/model/GPU imports):

```sh
modal run modal_iam.py --limit 10       # copy/extract raw + raw parser gate
modal run modal_prepare.py             # experimental tiny conversion
```

Small-file access through the Volume can be slow. The first conversion was
completed locally on CPU after downloading the archives; generated outputs are
uploaded back to the Volume. No paid GPU is needed for this stage.

To download persistent outputs:

```sh
mkdir -p data/diffink
modal volume get diffink-data diffink/iam data/diffink --force
```

Create the destination directory first. Modal includes the remote basename
under it on directory downloads. Files are committed/uploaded to:

```text
Volume root
├── iam/                          # your original files, untouched
├── raw/iam/                      # persistent archives + extracted raw data
├── canonical/iam/preview/        # original raw visual gate
├── canonical/iam/tiny/           # raw JSON for tiny conversion
├── diffink/iam/                  # HDF5 + metadata + comparisons + check report
└── checkpoints/                 # experimental InkVAE mechanics runs
```

Modal docs: https://modal.com/docs/guide/volumes

## Next gate

Review before/after shape preservation, and confirm normalization/end-state
choices against the authors' prepared English data or forthcoming multilingual
code. Only then consider a tiny InkVAE overfit job. We have **not** proved model
reconstruction, English generation, Markdown layout, or LaTeX handling.

## Follow-up fixes and separate overfit preparation

The earlier CTC test missed the duplicate `VAE.get_ocr_loss` used in the actual
forward/training path. That is fixed by delegating to the shared helper, with
regression tests through `VAE.forward`. The previous claim of a complete CTC
training fix was premature; see the audit above for the correction.

Generated-output rebuilds now replace completed directories rather than
accumulating stale JSONs/previews. Failed conversion preserves the prior build;
raw IAM is never cleaned. Publishing also removes stale generated remote files.

A separate same-writer, **line-disjoint** mechanics dataset has eight writers
×24 train lines (192) and ×4 val lines (32). This is not the original 250/50
writer-disjoint dataset and is not a held-out-writer benchmark. Test writers
remain reserved. Prepare/check it without training:

```sh
python -m iam_tools.overfit
python -m iam_tools.check_batch --out data/diffink/iam_overfit
python -m iam_tools.inkvae --data-root data/diffink/iam_overfit  # CPU forward only
python publish_iam.py --overfit
```

The CPU smoke report is `data/diffink/iam_overfit/vae_smoke.json`. All actual
model/loss terms are finite with the explicitly documented config. No optimizer
step, backward pass on the model, or GPU allocation occurs in that smoke check.

**Additional caveat:** direct height-100 input caused NaNs at initialization.
The prepared mechanics config therefore multiplies model x,y by 0.01, leaving
the stored RDP data unchanged; checkpoint and preview code preserve/undo the
adapter. The new runner also uses a log-space GMM density, because upstream's
PDF floor saturates. These are numerical mechanics choices, not paper-equivalent
English preprocessing. The exact author normalization and internal character
endings remain unresolved.

A bounded single-T4 job is implemented in `modal_inkvae.py` and
`configs/vae_iam_overfit.yaml`, and was launched with user approval on 2026-10-05. Its limits are 200
steps / 600 loop seconds, and the container times out after 900 seconds. Both
explicit flags `--train` and `--allow-experimental` are required; the default
entrypoint never invokes the remote function. The user approved a first T4 mechanics run; results are recorded separately below.

### Fixed-sample instrumentation for the first T4 run

The runner evaluates the same train/validation line at step 0, 50, 100, 150,
200 (or the budget-limited final step). `fixed_metrics.jsonl` records all loss
terms, greedy OCR/CER, writer correctness, coordinate RMSE in model units and
pen accuracy. Evaluation disables dropout and reuses a fixed latent-noise seed,
restoring the training RNG and model mode afterward. Input and reconstruction
renders use that same forward. Training metrics record nonzero/finite OCR and
style gradient norms as a supervision diagnostic.

Only one T4 container is permitted, with no automatic retries or GPU fallback.
The run remains experimental: line normalization, internal character endings,
the reversible input adapter, log-space GMM and masked reconstruction losses
are not the authors' released English training pipeline.

### First T4 result — 2026-10-05

Run `20261005-072316` completed 200 updates on a Tesla T4 with PyTorch
2.14.1+cu130; reported loop/evaluation/checkpoint time was 17.0 seconds
(not total container lifetime or billed duration). One earlier attempt stopped
at step 0 because the newly added OCR hook was bypassed by a direct `.forward`
call; that instrumentation was fixed and regression-tested before relaunch.
No L4, full-dataset training, or InkDiT job was launched.

All 200 logged losses/gradient norms are finite, and OCR/style parameter
gradients are nonzero every step. First/last 20-step mean total loss fell
5.96 → 3.91; CTC 4.16 → 3.23; style CE 2.13 → 1.86. Fixed train total
fell 10.13 → 3.64, fixed val 9.50 → 3.85.

**Reconstruction gate NOT passed.** Visual inspection of both final examples
shows mostly flattened, noisy traces rather than legible words. Their greedy
OCR outputs are empty (CER 1.0), and both fixed writer predictions are wrong
at step 200. Coordinate RMSE improved substantially, but global x progression
is not letter-shape reconstruction. Lower auxiliary losses/nonzero gradients
show that supervision is wired, not that OCR/style objectives are solved.

At batch size 2, 200 updates cover only about two passes over 192 train lines.
Next recommendation: stay on T4, use a truly minimal 1–8-line memorization
experiment with a separately approved bounded budget, and inspect letter shape,
pen-state accuracy and blank-heavy CTC behavior before expanding or using L4.
The normalization/character-ending policies and runner numerical changes
remain experimental; no author English result has been reproduced.

Persistent artifacts: `/data/checkpoints/iam_overfit/20261005-072316/`
(or `/mnt/diffink-data/checkpoints/iam_overfit/20261005-072316/` in your shell).
This contains checkpoint, shuffled/fixed metrics, prediction arrays, stepwise
input/reconstruction PNG/SVG, `result.json`, `summary.json`, `loss-curves.png`,
and `index.html`. Downloaded local copy: `data/checkpoints/iam_overfit/20261005-072316/`.
To regenerate the CPU-only report:

```sh
python -m iam_tools.report_inkvae data/checkpoints/iam_overfit/20261005-072316
```

### Staged single-line autopsy

The next approved experiment is stage 1 only: exactly `c08-434z-05` on every
update, fresh initialization (seed 42), coordinate GMM NLL only. Pen, CTC, style
and KL objectives are off; OCR/style forward calls are suppressed and their
parameters frozen/excluded from the optimizer. Tests verify no auxiliary
gradients and zero loss gradient on the three pen-output rows. No extra MSE,
normalization, RDP, architecture, dropout or latent-sampling change is introduced.
The coordinate NLL has weight 1 because it is the sole objective. Existing
192/32 files are sources; no other line can enter the repeated pre-collated batch.

`configs/vae_iam_autopsy_geometry.yaml` and `modal_autopsy.py` specify one T4,
maximum 1,000 updates / 600 loop seconds / 900 container seconds, no retries,
no GPU fallback. The previous 200-step runner's cap remains unchanged. Its CPU
preflight checks config, manifest and selected sample hashes before training.

```sh
python -m iam_tools.trajectory_diagnostics data/checkpoints/iam_overfit/20261005-072316
python -m iam_tools.autopsy --data-root data/diffink/iam_overfit  # CPU only
# Upload autopsy_preflight.json to /diffink/iam_overfit/ before the approved run.
modal run modal_autopsy.py --train --allow-experimental
```

CPU old-run diagnostics reproduce the final train/val pen histograms exactly.
Train Y correlation is only 0.100 (X 0.986); val Y correlation 0.446 (X 0.994).
Using true pen states reconnects the traces but does not recover letter shapes.
Inverse weights from real points alone are 1.068/16.139/581 on the train line.
However upstream computes weights *before padding is masked*: the fixed single
sample is padded to 584, making actual class weights 1.074/16.222/146. Padding
therefore also changes EOC weighting with batch composition. This is a plausible
failure mechanism, not a proven causal result without a controlled pen A/B.

The new diagnostics exclude padding, save separate X/Y RMSE/correlation,
true/predicted state counts, confusion matrix and per-class recall. Geometry
snapshots show input, predicted XY with true pen states, and predicted XY with
predicted states in the same coordinate frame; histograms and unmodified NPYs
are saved too. Predicted pen accuracy is diagnostic only while its loss is off.
Low NLL or high global X correlation alone does not pass the letter-shape gate.

Later stages are conditional, not automatic: geometry first; then compare
original inverse-frequency focal with a bounded English pen policy; then CTC
with style still off; then one writer's eight lines; then multiple writers and
style. No InkDiT, full IAM training or normalization changes in this autopsy.

### One-line geometry result — 2026-10-05

Run `20261005-075059` completed exactly 1,000 updates on a Tesla T4 in 55.9
reported loop/evaluation/checkpoint seconds (not total billed time). The same
581-point line was repeated at every update. All logged losses and gradient
norms stayed finite. OCR/style state was unchanged; auxiliary gradients were
absent and pen-output loss gradients zero on all 1,000 updates. The old
200-step cap was not loosened, and no later stage was launched.

An initial Modal launch failed before the training function could start because
a neighboring entrypoint module was not included remotely. That app was
explicitly stopped, and `modal_autopsy.py` was made self-contained. Note that
`retries=0` disables function-input retries, not all platform attempts to start
a failing container. A regression test now checks the entrypoint's top-level
imports and opt-in guards. The successful run had no training retries.

| Fixed geometry diagnostic | Step 0 | Step 1,000 |
|---|---:|---:|
| GMM NLL | 19.051 | -2.399 |
| X RMSE (model units) | 7.077 | 0.102 |
| Y RMSE (model units) | 0.530 | 0.064 |
| X correlation | 0.090 | 0.9998 |
| Y correlation | -0.004 | 0.898 |
| Selected sigma X median | 0.732 | 0.100 |
| Selected sigma Y median | 0.721 | 0.064 |

First/last 50-update mean training GMM NLL: 3.270 → -2.178. Negative NLL is
possible for a continuous density in these model units; it is not a success
criterion by itself.

**Geometry gate: partial, not perfect memorization.** True-pen renders now show
letter shapes and much better Y structure, but remain jagged/distorted and are
not reliable handwriting reconstruction. This is substantial progress over
flattened traces, not a reason to switch on the other objectives yet. Predicted
pen states are deliberately untrained in this run and are not scored as a
pen-learning failure.

CPU-only checkpoint controls (`iam_tools.autopsy_checkpoint_diagnostics`) verify
strict loading and finite state. Four latent-noise seeds plus a latent-mean
inference control show that sampling noise is not the whole residual error.
At seed 1042 on CPU, highest-weight selection gives Y RMSE 0.064/correlation
0.897; using latent mean gives 0.061/0.909. A target-posterior oracle improves to
about 0.053/0.932 but still leaves distorted shapes and is **not** a usable
inference/generation method. No updates occur during these controls.

Next proposed experiment remains **geometry only**: a separately bounded,
lower-learning-rate continuation to tighten the per-point fit before pen A/B.
That continuation would need explicit resume/config support; the current
command intentionally starts fresh and stops at 1,000. No continuation, pen,
CTC, style, larger dataset or normalization experiment has been launched.

Artifacts (checkpoint, raw predictions, same-frame comparisons, true/predicted
pen renders/histograms, all metrics, CPU controls, curves and HTML gallery):

```text
Volume: /data/checkpoints/iam_autopsy/geometry/20261005-075059/
Shell:  /mnt/diffink-data/checkpoints/iam_autopsy/geometry/20261005-075059/
Local:  data/checkpoints/iam_autopsy/geometry/20261005-075059/
```

`index.html` is the consolidated gallery. Old-run CPU diagnostics are in its
original run directory as `trajectory_diagnostics.json` and `autopsy-index.html`.
The IAM normalization and internal EOC policy remain experimental; this is not
an author-result reproduction.

### Approved low-LR continuation

The next run resumes the exact step-1,000 model **and** AdamW state from
`20261005-075059`, pinned by SHA256
`383bbdeb21cbda0b1713150f079a7843daf77ed7ac35824bd9abed3dcbfbcf7e`.
Its only changed training setting is LR 1.5e-4 → 1e-5. The continuation config
is compared against the checkpoint and rejects other setting/sample changes.
All 200 Adam state entries restore at step 1,000; the LR override occurs after
`load_state_dict`. `max_steps: 1000` means **additional** updates, ending at
global step 2,000. Every 100 updates record per-axis RMSE/correlation, selected
sigma medians, NLL, most recent raw gradient norm/clipping flag, and last-100/
cumulative clipping fractions. The old clip of 10 stays unchanged.

```sh
python -m iam_tools.autopsy --config third_party/DiffInk/configs/vae_iam_autopsy_resume.yaml --data-root data/diffink/iam_overfit --resume-checkpoint data/checkpoints/iam_autopsy/geometry/20261005-075059/checkpoint.pt
# Persist autopsy_resume_preflight.json, then the approved launch:
modal run modal_autopsy.py --train --allow-experimental --resume
```

No auxiliary losses, MSE, normalization, sampler, dropout, architecture, GMM
math, betas, weight decay or clipping changes. One T4, at most 1,000 additional
updates / 600 loop seconds / 900 container seconds; no automatic function-input
retries or GPU fallback. The new run writes a separate directory and never
replaces the source checkpoint.

**Reproducibility limitation:** the old checkpoint did not store RNG states.
Model and Adam moments are fully restored, but the stochastic stream must
restart at seed 42 rather than continue bitwise. This is recorded in provenance;
new checkpoints save both CPU and CUDA RNG states for future resumes. Fixed
comparisons retain the same evaluation seed 1042.

The strict geometry gate remains mean-error/visual, not just NLL: ideally
Y correlation >0.97, materially lower X/Y RMSE, recognizable smooth letters
using true pen boundaries, and no apparent variance tightening without location
improvement. Pen A/B remains conditional, and MSE is not included in this run.

### Low-LR continuation result — 2026-10-05

Run `20261005-081356` restored the pinned model and all 200 AdamW state entries
at step 1,000, then executed exactly 1,000 additional updates at LR 1e-5 on
Tesla T4. Reported loop/evaluation/checkpoint time was 54.6 seconds (not total
billed duration). Fixed evaluations at global steps 1,000/1,100/…/2,000 use the
same latent seed 1042; the resumed step-1,000 GPU evaluation matches the source
run's final fixed evaluation exactly. Other training settings stayed identical.

| Fixed diagnostic | Step 1,000 | Step 2,000 |
|---|---:|---:|
| X RMSE (model units) | 0.10186 | 0.05902 |
| Y RMSE (model units) | 0.06368 | 0.03120 |
| Y correlation | 0.89788 | 0.97583 |
| GMM NLL | -2.399 | -3.144 |
| Selected sigma X median | 0.10035 | 0.03652 |
| Selected sigma Y median | 0.06391 | 0.03153 |

X/Y RMSE improved 42%/51%; Y correlation exceeded the suggested 0.97 numeric
criterion. True-pen renders are much more recognizable and follow the sentence,
but retain angular/noisy details, especially the initial R and final word.
This is numerical-gate progress, not perfect memorization or an end-to-end
handwriting success. The visual gate should be reviewed before pen-policy A/B.

**Clipping persists:** 1,000/1,000 continuation updates clipped at the unchanged
cap 10. First/last 100 mean raw norms were 30.9/82.4 (last median 81.3, max149.1).
Sigma keeps shrinking, and final X RMSE/NLL are not monotonic over snapshots.
Thus this is not fully settled optimization. However location errors improve
substantially alongside uncertainty reduction: it is not *merely* tightening
density around unchanged means. Raw gradient norm/clipping alone is not a
measurement of Adam parameter update size.

CPU verification confirms all final Adam steps are 2,000, LR1e-5, finite model
state, saved CPU/CUDA RNG, and the original local source checkpoint unchanged.
The successful job saved its new checkpoint in a separate directory; the old
source was not overwritten. The old missing-RNG limitation remains explicitly
recorded, so this is model/optimizer continuation rather than bitwise stochastic
continuation. No MSE, pen, CTC, style, normalization or larger-data experiment
was launched after this run.

```text
Volume: /data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/
Shell:  /mnt/diffink-data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/
Local:  data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/
```

`index.html` shows the same-frame comparisons and six-panel curves separating
mean errors, sigma, NLL and clipping. Raw metrics, fixed metrics, model/Adam/RNG
checkpoint, summary and CPU verification are persisted alongside it. Regenerate
the report without training/GPU:

```sh
python -m iam_tools.report_autopsy data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356 --parent-dir data/checkpoints/iam_autopsy/geometry/20261005-075059 --visual-assessment recognizable-with-residual-errors
```

The IAM normalization and EOC encoding remain experimental. No authors' English
result, OCR accuracy, writer-style learning or English text generation has been
reproduced by this one-line test.

### Geometry accepted; controlled pen-head A/B — 2026-10-05

The user reviewed the step-2,000 continuation and **passed the geometry gate
for the autopsy**. Geometry polishing stops: no further LR continuation and
no MSE. This does not establish the authors' English result or generalization.

`iam_tools.pen_ab` branches from that exact step-2,000 checkpoint, SHA256
`23fc747d82773d6714385f1e25af650ab018c14d95f11c2db54afbe0c8a8ac33`.
Only the first three decoder FC output rows and their biases can change.
The frozen VAE is in eval mode; one deterministic representation (latent seed
1042) is cached. Each arm gets an independently initialized **identical 771-value
linear head**, fresh AdamW at LR1e-3, betas 0.9/0.99, weight decay 0, clip 10,
and exactly 1,000 updates. This is a deliberately head-only optimization setup,
not a continuation of the geometry optimizer. No dropout or latent-noise drift
is introduced between arms. All other VAE weights and auxiliary objectives
remain frozen/off. Neither arm hit clipping.

This very small optimization ran on **CPU**, with no GPU allocation/credit.
There are 581 real points and 3 padding points; padding is excluded before class
counting, focal loss and metrics. Both arms use upstream's *same* focal formula,
`alpha[target] * (1 - exp(-unweighted_CE))**2 * unweighted_CE`; only alpha differs:

- A inverse frequency: `1.068 / 16.139 / 581`.
- B sqrt inverse frequency normalized to continue, capped 8: `1 / 3.887 / 8`.

A separate linear head, rather than masking gradients on the full FC, prevents
optimizer weight decay/moments from moving GMM rows. At every 100-update snapshot,
the head is merged into the **full VAE** and a deterministic forward verifies
bitwise-identical GMM outputs and XY, identical NLL, and unchanged non-pen state.
A separate CPU checker reloads both saved branch models and reproduces these
invariants and boundary metrics. The parent checkpoint's SHA stays unchanged.

**Same-device comparison:** CPU is not claimed bitwise equal to the prior T4
forward. The source CPU X/Y RMSE is 0.05890/0.03098 and Y correlation 0.97615
(vs 0.05902/0.03120/0.97583 on T4). Both arms' XY is exactly equal to that same
CPU source throughout. Source CPU pen counts before either arm are 299/35/247;
the untrained geometry-stage pen head is not a learned pen baseline.

| After 1,000 head updates | A inverse | B bounded |
|---|---:|---:|
| Pen-up precision | 0.590 | **0.791** |
| Pen-up recall | **1.000** | 0.944 |
| Pen-up F1 | 0.742 | **0.861** |
| True-positive boundaries | 36 | 34 |
| False pen-up breaks | 25 | **9** |
| Missed pen-up boundaries | **0** | 2 |
| Predicted continue / pen-up / EOC | 519 / 61 / 1 | 537 / 43 / 1 |
| Non-final false EOC | 0 | 0 |
| Final EOC correct | yes | yes |

Target counts are 544/36/1. Raw overall accuracy is logged but **not a gate**;
always-continue already gets 93.6%. B wins the requested boundary-F1 comparison
and markedly reduces false pen breaks. Its predicted-pen render is recognizably
close to the fixed-XY/true-pen reference, but still has 9 extra breaks and 2
missing boundaries. **The perfect-boundary gate has not passed.** Both policies
learned final EOC and eliminated false internal EOCs; this run does NOT show
that A inevitably fails, that it cannot improve with more updates, or that
weighting alone caused the old jointly trained 192-line failure. It supports
bounded weighting as the better policy at this fixed representation/budget,
not an English/multilingual generalization claim. Weighted losses have different
scales and should not be compared as objective-quality scores.

Run `20261005-084901` took 21.4 seconds for both head loops, evaluations, renders
and saves (excluding source loading). No CTC/style/KL, joint-VAE or larger-data
stage was started. The next proposed experiment is joint geometry+pen using B
on this same line, checking geometry retention and boundary improvement;
**CTC remains off until the single-line boundary reconstruction is acceptable**.

```sh
# Default is CPU preflight only, zero optimizer steps:
python -m iam_tools.pen_ab --data-root data/diffink/iam_overfit --checkpoint data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/checkpoint.pt
# Re-run the bounded A/B (CPU only), into a new output directory:
python -m iam_tools.pen_ab --data-root data/diffink/iam_overfit --checkpoint data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/checkpoint.pt --output-base data/checkpoints/iam_autopsy/pen_ab --train --allow-experimental
# Independent saved-model verification and curves, no training:
python -m iam_tools.report_pen_ab data/checkpoints/iam_autopsy/pen_ab/20261005-084901 --data-root data/diffink/iam_overfit
```

```text
Local:  data/checkpoints/iam_autopsy/pen_ab/20261005-084901/
Volume: /data/checkpoints/iam_autopsy/pen_ab/20261005-084901/
Shell:  /mnt/diffink-data/checkpoints/iam_autopsy/pen_ab/20261005-084901/
```

`index.html` contains the A/B snapshots and six-panel boundary curves. Raw NPY
predictions are never rewritten to force a final EOC; renders draw any unfinished
last path without relabeling it and explicitly display final-EOC correctness.
Each arm has fixed metrics, every-update loss/gradient logs, and its own merged
model checkpoint with head optimizer state. `frozen_features.pt` and
`saved_checkpoint_check.json` preserve the cache and independent verification.
IAM/checkpoint artifacts stay in local/Volume storage, never GitHub. All 43
unit tests pass in both root and fork, including padding exclusion, exact
upstream focal parity, majority-class failure, AdamW row isolation and an
actual full-VAE merge/forward invariance regression test.

### Joint compatibility and deterministic coordinate diagnostic — 2026-10-05

The user correctly noted that even true-pen XY remained angular. The earlier
numerical geometry gate was a mechanics gate, **not a human-realism gate**.
We implemented two independent branches from the exact step-2,000 geometry
checkpoint, retaining IAM/RDP/scale 0.01 and the same architecture. No CTC,
style or KL loss, no temporal smoothing/resampling, and no derivative loss.

**Joint branch:** the user chose trained Arm B pen rows; all other model state
comes from the original step-2,000 geometry checkpoint. The source geometry
AdamW moments restore at step 2,000 (pen rows' old moments were zero), LR1e-5,
GMM NLL 1 + bounded focal 1. Train mode/sampled latent/dropout follow the previous
geometry runner. The cap is 300 updates, evaluating every 50. Geometry must not
regress in either per-axis RMSE or correlation against the exact same-device
baseline; only absolute numerical tolerance 1e-7 is allowed. Failed snapshots
are saved for diagnosis but not promoted.

The joint branch **stopped at 50** under that guard:

| Fixed T4 primary readout | Before | After 50 |
|---|---:|---:|
| X RMSE | 0.05902 | 0.05011 |
| Y RMSE | 0.03120 | 0.03252 |
| Y correlation | 0.97583 | 0.97416 |
| Pen-up F1 | 0.87179 | 0.86842 |
| Within-stroke first-difference error | 0.05079 | 0.05326 |
| Within-stroke second-difference error | 0.08625 | 0.09082 |

Y error increased 4.2%; jitter metrics also worsened. All 50 steps clipped at 10.
This did **not** prove geometry/pen coexistence under these optimizer/objective
settings. It does not prove the objectives fundamentally cannot coexist.
The T4 pen baseline differs slightly from the CPU A/B (0.861), so only the
same-device 0.87179→0.86842 comparison is used to judge this branch.

**Deterministic branch:** resets to the *untouched* step-2,000 geometry model,
not the joint model or trained B head. It optimizes
`MSE(sum_k softmax(pi_logits)_k * mu_k, true_XY)` over 581 real points and both
coordinate axes, excluding padding. One line, 1,000 updates, fresh AdamW at
LR1e-4, betas 0.9/0.99, decay 0, clip 10. It decodes **latent mean** in **eval mode**
(dropout off, gradients enabled). Only the pi/mu output rows receive objective
gradients; pen/sigma/rho FC rows and the latent-variance head remain unchanged.
The encoder/decoder/transformer and latent-mean head still optimize normally.

This deliberately deterministic capacity test differs from stochastic GMM
training in loss, latent/dropout behavior and optimizer initialization. Success
is **not a pure loss-only causal ablation**. Each evaluation therefore records
all four controls: sampled latent seed 1042 vs latent mean, and highest-weight
component mean vs mixture expectation. No target-aware component selection.

| Deterministic latent-mean / expectation readout | Before | After 1,000 |
|---|---:|---:|
| X RMSE | 0.05497 | **0.00228** |
| Y RMSE | 0.02740 | **0.00170** |
| Y correlation | 0.98139 | **0.99993** |
| Within-stroke first-difference vector RMSE | 0.04482 | **0.00297** |
| Within-stroke second-difference vector RMSE | 0.07689 | **0.00551** |
| Expected-XY MSE | 0.0018864 | **0.00000406** |

X/Y RMSE fell 95.8%/93.8%; first-/second-difference errors fell 93.4%/92.8%.
The final true-pen reconstruction visually follows the processed input very
closely, without the earlier local zig-zags. **The deterministic one-line
capacity diagnostic passes our visual/numerical inspection.** It is not literal
pixel equality, generalization to other lines, or reproduction of unprocessed
IAM/source images. The same 8× architecture and provisional representation can
represent this line closely; they are not a hard capacity barrier here.
No derivative objective was required to obtain this local-shape improvement.

No clipping occurred in the MSE branch (median/mean/max raw norms
0.094/0.182/5.98). Nevertheless convergence is not monotonic: step 950 had
X/Y RMSE 0.00985/0.01078 before the strong step 1,000 result. This is a good
saved reconstruction, not proof training has reached a stable asymptote.
The loop/evaluation/render/save time was 91.8 seconds on T4, excluding startup
and source loading; this is not total billed time.

**Readout/noise still matter.** At the same final MSE checkpoint:

| Final XY readout | X RMSE | Y RMSE |
|---|---:|---:|
| Latent mean + mixture expectation | 0.00228 | 0.00170 |
| Latent mean + highest-weight component | 0.02565 | 0.03549 |
| Fixed sampled latent + mixture expectation | 0.01768 | 0.00990 |
| Fixed sampled latent + highest-weight component | 0.03195 | 0.03575 |

Only the expectation was explicitly supervised. A beautiful expectation does
not imply beautiful individual components or sampled trajectories. The fixed
sampled-latent expectation also improves substantially from its own baseline,
but is noticeably worse than the deterministic mean. The next production
loss decision must account for stochastic behavior and the actual rendered
readout; do not blindly replace GMM or claim likelihood alone caused everything.

**Derivative definitions:** first differences use 544 connected true-stroke
edges; second differences use 508 three-point windows fully inside true strokes.
Both report vector and per-axis RMSE plus target-relative error. All-point
versions including pen-up jumps are recorded as secondary controls. These
are index-difference errors on nonuniformly RDP-sampled points, **not physical
velocity or arc-length-normalized curvature**.

**A runner handoff bug was caught:** the first worker completed the guarded
joint branch, then stopped before its first MSE optimizer update because a
newly frozen latent-variance head retained old `.grad` buffers. Changing
`requires_grad` does not clear existing gradients. Branch initialization now
clears *all* gradients before changing trainability. A regression test covers
this transition. Only MSE was relaunched; the joint run was not repeated.
`attempt.json` preserves this history rather than hiding the failed worker.

Independent CPU saved-state checks confirm unchanged OCR/style state, all
MSE pen/sigma/rho FC rows and latent-variance parameters unchanged, finite
weights, source checkpoint SHA unchanged, joint Adam step 2,050 (200 entries)
and fresh MSE Adam step 1,000 (198 entries). CPU/CUDA bitwise equivalence is
not claimed. All 51 unit tests pass in both root and fork.

```sh
# CPU-only preflight, zero updates; upload reconstruction_preflight.json:
python -m iam_tools.reconstruction --data-root data/diffink/iam_overfit --source-checkpoint data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/checkpoint.pt --pen-checkpoint data/checkpoints/iam_autopsy/pen_ab/20261005-084901/B_bounded_english/checkpoint.pt
# Guarded bounded pair, NOT a production training command:
modal run modal_reconstruction.py --train --allow-experimental
# To run ONLY the independent MSE branch, not redo joint:
modal run modal_reconstruction.py --train --allow-experimental --mse-only
# Report/verify existing results, no training/GPU:
python -m iam_tools.report_reconstruction data/checkpoints/iam_autopsy/reconstruction/20261005-091827 --joint-directory data/checkpoints/iam_autopsy/reconstruction/20261005-091556 --data-root data/diffink/iam_overfit --source-checkpoint data/checkpoints/iam_autopsy/geometry_lr1e5/20261005-081356/checkpoint.pt --pen-checkpoint data/checkpoints/iam_autopsy/pen_ab/20261005-084901/B_bounded_english/checkpoint.pt
```

```text
Joint: /data/checkpoints/iam_autopsy/reconstruction/20261005-091556/joint/
MSE:   /data/checkpoints/iam_autopsy/reconstruction/20261005-091827/deterministic_mse/
Report: /data/checkpoints/iam_autopsy/reconstruction/20261005-091827/index.html
```

Replace `/data` with `/mnt/diffink-data` in a Modal shell, or `data` locally.
The report links both runs and contains true-pen before/after panels, derivative
curves, readout/noise controls, raw snapshots and CPU checkpoint checks. All
IAM/checkpoint artifacts remain on local disk/Volume, never GitHub.

**No CTC enabled.** Pen loss was intentionally off in MSE and its raw predicted
pen states are meaningless. Its hidden representation changed, so simply
transplanting the old B head is not an end-to-end solution. Proposed next step:
fit bounded pen rows against the new frozen geometry, then test the rendered
coordinate objective alongside GMM/pen with careful stochastic controls.
Only promote once predicted-pen reconstruction retains the input geometry;
CTC, eight-line/writer and full-IAM experiments are still deferred.

## Engineering contract patches and bounded eight-line T4 result (2026-10-05)

**Current status: eight-line reconstruction/pen gate NOT passed. No CTC, KL,
style, InkDiT, or full-IAM training promoted or launched.** This section
supersedes the earlier proposed next steps, not the historical experiment data.

### Frozen geometry pen refit

On the deterministic MSE checkpoint, 1,000 CPU head-only updates per arm yield
identical pen-up F1 **0.9474**, precision 0.9 / recall 1.0, four false breaks,
zero missed breaks, no internal EOC and correct final EOC. Bounded three-state
and binary-with-forced-final-EOC tie. We retain **bounded three-state B**:
binary rendering uses known target length to force EOC, so does not prove learned
stopping. Both arms start from identical pen rows; non-pen state, GMM output
and expected XY are checked bitwise unchanged. Source and outputs remain on
the Volume, not GitHub.

### Implemented contract/stability changes

- `trans_dropout` explicit (legacy default 0.1, engineering English 0).
- Deterministic mixture `expectation`, genuinely greedy `argmax`, separate
  stochastic `sample`; same softplus+epsilon sigma in train and inference.
  Explicit `sigma_parameterization='exp'` retains a legacy control.
- KL averages valid latent **elements**, independent of channel/time lengths;
  padded values are excluded before exponentiation.
- VAE public `encode`/`forward` own `model_input_scale`; raw HDF5 XY is scaled
  automatically, and `to_data_space` reverses it for rendering. Explicit
  `input_is_model_space=True` is reserved for already-scaled helpers.
  VAE checkpoint metadata propagates this contract to DiT loading/encoding.
- Optional full-resolution Transformer decoder padding mask is wired; temporal
  **GroupNorm is unchanged and still padding-sensitive**. Batch one with gradient
  accumulation mitigates this; it does not establish padding invariance.
- Rotation/scaling independently configurable; engineering rotation/augmentation
  are off. No new smoothing or normalization layers.
- Actual `trainer/vae_trainer.py` implements log-space GMM, bounded real-only pen,
  expected-XY anchor, physical gradient accumulation, configured clipping, and
  independently gated auxiliaries. Production launcher supports calibration and
  config selection. The bounded test uses this real trainer; multi-rank DDP and
  DiT training have not been GPU-tested. Checkpoints save the model contract.

`configs/engineering_english.yaml` is the guarded engineering experiment.
`configs/paper_english.yaml` is a **paper-informed hyperparameter control**, not
an exact reproduction: paper LR 5e-5, batch 128, clip 5, weights GMM/pen/OCR/style/KL
1/2/1/0.5/1e-6 and 5% warmup. English coordinate normalization, final-only EOC,
scale, and augmentation remain provisional. Decoder dropout is not specified in
the paper; its 0.1 control value comes from the released code default.

### Authorized T4 test: completed, not promoted

Persistent report: `checkpoints/iam_eightline/20261005-102304/index.html`.
Eight **training** lines from writer 10174, deliberately narrow lengths
514–581; no held-out evaluation or full-length-range/generalization claim.
All 200 optimizer updates / 1,600 physical microbatches completed. Physical
batch 1 + accumulation 8, LR 1e-5, dropout 0, rotation off, model scale 0.01,
GMM + bounded pen + expected-XY anchor. CTC/style/KL **weights zero**.

| Fixed eight-line mean metric | Step 0 | Step 200 |
|---|---:|---:|
| Latent-mean X RMSE (model units) | 0.944 | 0.388 |
| Latent-mean Y RMSE (model units) | 0.115 | 0.059 |
| Macro pen-up F1 | 0.287 | 0.698 |

Twenty fixed-seeded sampled latents **per line per checkpoint**, at steps
0/50/100/150/200, are saved with raw arrays, per-axis errors, within-stroke
index-difference diagnostics, pen histograms/F1, posterior uncertainty and
mean/median/worst galleries. Final sampled reconstructions are close to the
latent-mean errors; geometry is still visibly degraded, not simply hidden by
latent sampling. The formerly memorized line regresses from approximately
0.0033/0.0018 to 0.0765/0.0416 X/Y RMSE. Two unfamiliar lines remain especially
poor in X (1.377 and 0.843). Final EOC is correct on **3/8**, with **51** false
internal EOCs across the eight latent-mean reconstructions.

All **200/200** updates clip at norm 5; median raw norm 29.58. No numeric crash.
Loop/evaluation/render/save time 103.3 s excludes startup/loading, and is **not
billed duration/cost**. Checkpoints/optimizer state are finite; all Adam states
record step 200. Source and frozen OCR/style state were verified unchanged.

Anchor coefficient **6.6665** was calibrated from median matched per-line decoder
gradient ratios targeting 15%; actual initial per-line ratios range **0.0038 to
0.3735**. This is not a uniform or aggregate 15% guarantee. That heterogeneity
and the geometry regression need diagnosis before another experiment.

Historical `metrics.jsonl` losses are **last-microbatch values**, not effective-
batch averages, although gradients were accumulated correctly. Therefore don't
interpret their first/last NLL as convergence. Future trainer logging is corrected
to effective-batch means with a regression test; this correction did not rerun
or alter the completed test. Nonzero logged KL is a diagnostic with zero weight.

All **61 tests pass** in root and fork. GroupNorm, compression, latent width,
RDP=0.5 and absolute XY representation remain unchanged. Defer KL/CTC until
multi-line geometry and boundaries are clean; do not infer failure of model
capacity from this short combined-objective experiment.

```sh
# CPU-only refit (from root); use --repo . when running inside fork:
python -m iam_tools.pen_refit --help
# Verify/report existing artifacts: no training/GPU:
python -m iam_tools.report_eightline data/checkpoints/iam_eightline/20261005-102304
# Import/default invocation does not allocate a GPU:
modal run modal_eightline.py
# Already executed with explicit authorization; repeating incurs another job:
# modal run modal_eightline.py --train --allow-experimental
```

In your Modal shell use `/mnt/diffink-data/checkpoints/iam_eightline/20261005-102304/`;
inside the job use `/data/checkpoints/...`; locally use `data/checkpoints/...`.
No SSH server is needed. CPU refit artifacts live at
`checkpoints/iam_autopsy/pen_refit/20261005-100653/` on the same Volume.

## Continuing objective research (user lifted the 200-update limit)

`modal_objective_study.py` runs explicitly authorized research chunks rather
than changing the original bounded-test guard. `--train` is still required;
import/default invocation allocates no GPU. CPU preflight and unit tests run
before the new experiments. All source/sample/manifest pins are retained.

The first pair branches from the **same original eight-line step-200 model**,
with fresh identical Adam, sampled latents, dropout0, batch1/accum8, no rotation,
no KL/CTC/style. Both use expected-XY MSE100 + bounded pen1, LR5e-5 reduced to
1e-5 after 800 updates, weight decay0. Only GMM coefficient changes (0 vs1).
This pair isolates the extra GMM objective; comparison against the older
200-update experiment is NOT a single-variable ablation (budget/LR/anchor change).
The 100 coefficient is an intentionally strong diagnostic, not a paper weight
or claimed best production hyperparameter.

At 1,000 additional updates:

| Arm | Mean X/Y RMSE | Macro pen F1 | Final EOC | False internal EOC |
|---|---|---:|---:|---:|
| expected XY + pen | 0.04163 / 0.03564 | 0.99476 | 8/8 | 1 |
| GMM + expected XY + pen | 0.05055 / 0.02851 | 0.93155 | 7/8 | 4 |

Both improve greatly; pure direct-XY learns almost-perfect boundaries, while
GMM improves Y somewhat more but has worse X/boundaries. Neither dominates
all geometry metrics, and the rendered trajectories still have shape distortion.
All updates clip at5 (raw-norm median138.2 vs377.9 respectively). Loop times
297.7/313.5 seconds are NOT billed time. Twenty fixed-seeded sampled-z outputs
per line/checkpoint remain close to the mean reconstruction; posterior std has
shrunk further without KL, so this is not a regularized-latent validation.
Saved-state CPU checks confirm finite weights, Adam step1000 and frozen OCR/style.
Pure-XY sigma/rho FC rows remain exactly unchanged (no weight decay/no GMM).

CPU head-only refit on frozen **GMM-arm step1000** improves F1 .93155→.96576,
removes all false EOCs and learns final EOC8/8. No forced-final rule is used.
Every non-pen parameter and GMM FC row stays bitwise unchanged. Recomputed GMM
outputs agree within float tolerance; CPU/CUDA bitwise output equality is not
claimed. This independently isolates residual boundary readout from geometry.
The head-refitted checkpoint is a separate diagnostic, not silently substituted
into either continuing arm; its obsolete Adam state is deliberately omitted.

Decoder-gradient diagnostics use matched stochastic forwards, norms and cosines
per line. Some poorly reconstructed lines have near-orthogonal/negative GMM-XY
cosines. That is evidence of objective interaction, not proof GMM fundamentally
cannot work, nor proof normalization/generalization is solved.

Two first launchers failed during remote bootstrap (missing local launcher
import), before any optimizer updates. Those failed apps were stopped explicitly;
the launcher is now self-contained. `research_attempts.json` records app IDs and
successful replacements. Files remain on Volume/local disk, never GitHub.

```sh
# CPU-only gradient/data/model check:
python -m iam_tools.objective_study
# Fresh controlled pair (already executed):
modal run modal_objective_study.py --train --arm xy_pen --steps 1000
modal run modal_objective_study.py --train --arm gmm_xy_pen --steps 1000
# Resume a saved arm with model + Adam + Torch RNG, pinned SHA, same objective:
# modal run modal_objective_study.py --train --arm xy_pen --steps 2000 \
#   --resume-checkpoint /data/checkpoints/iam_objective_study/<run>/xy_pen/checkpoint.pt \
#   --resume-sha <SHA256>
```

Report: `checkpoints/iam_objective_study/comparison-1000/index.html`.
Arms: `20261005-110842/xy_pen`, `20261005-110843/gmm_xy_pen` under that study root.
Frozen CPU head refit: `pen_refit-gmm-1000/`.
Both arms are being continued from their own saved states at1e-5, reduced to1e-6
for the final20%. This is polishing actual reconstructions, not adding auxiliaries.

### Continuations completed: direct XY wins this eight-line control

Both arms completed **3,000 objective-study Adam updates each** beyond the old
200-update run. Continuations restore model, Adam, Torch RNG and advance the
same seeded per-update line permutations; LR1e-5→1e-6 for the last20%.

| Arm at 3,000 | Mean X/Y RMSE | Pen F1 | Final EOC | False internal EOC |
|---|---|---:|---:|---:|
| expected XY + pen | **0.01191 / 0.01822** | **1.000** | **8/8** | **0** |
| GMM + expected XY + pen | 0.03150 / 0.02060 | 0.93783 | 8/8 | 3 |

Direct XY now wins both axes and boundaries in this particular controlled
training-set test. It is NOT a universal claim that likelihood cannot work.
Continuation clipping rates:81.15% direct vs100% GMM; raw norm medians29.27 vs641.21.
Direct finishes below the clip limit at the lower LR. GMM highest-pi component
sigmas are typically ~0.013 X/~0.011 Y versus ~0.065/~0.049 for direct-only;
active correlations are not near singularity (no |rho|>0.99). Density tightening
and gradient scale, rather than correlation saturation, remain relevant.
Full gradient/uncertainty diagnostics are retained, not inferred from NLL alone.

Continuation report: `checkpoints/iam_objective_study/comparison-3000/index.html`.
Runs: `20261005-111735/xy_pen`, `20261005-111714/gmm_xy_pen` under that study root.
Adam states independently verified at step3000, finite model states, frozen
OCR/style unchanged. No full-IAM or InkDiT training.

### Deterministic L-BFGS diagnostic and frozen pen refit

To test residual optimization rather than repeat the same Adam loop, branch
from direct-XY step3000, train **mean-latent expected-XY MSE only** with L-BFGS,
50 outer steps /593 full-eight-line closure evaluations. No GMM/pen/CTC/style/KL
objective. LR1, strong-Wolfe search, history10, max10 inner iterations per outer
step; **no clipping or weight decay**. This changes optimizer, latent readout
and pen objective together, so is explicitly a **multi-knob capacity diagnostic**,
not a causal optimizer-only ablation. API verified against
[PyTorch L-BFGS documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html).

Mean X/Y improves **0.01191/0.01822→0.00525/0.01003**. Predicted pen F1 falls to
0.90474 because shared hidden features change even though the pen rows stay
fixed. Therefore refit ONLY the pen rows on cached new features: bounded3state,
gamma2/cap8, 3,000 CPU updates, genuine learned final EOC (not forced). F1 reaches
**1.000 on all8 lines**, no false breaks, missed breaks or false internal EOCs;
final EOC8/8. All non-pen state/GMM rows remain exactly unchanged. Expected XY
recomputations differ only at float tolerance; CPU/CUDA bitwise outputs are not
claimed. Obsolete optimizer state is removed from the head-refitted checkpoint.

Mean per-line **20-sampled-z median** X/Y errors after refit are **0.00602/0.01011**,
and every one of the20 sampled pen predictions on every line has F1=1.0 in
this saved CPU evaluation. These are the eight training lines only, narrow
length range, unregularized posterior; not generalization/paper reproduction or
pixel-perfect input identity. The rendered lines now track processed input
closely enough to proceed to an isolated OCR-head test, while some letter-shape
error remains. No velocity/curvature objective or architecture change was added.

L-BFGS loop/eval/save225.9seconds excludes startup, not billed duration. The first
runner's config carried inactive baseline LR/clip/budget fields; actual optimizer
settings are annotated in result.json. Future runner metadata is corrected, with
no optimizer change or rerun. Do not count outer steps as Adam-equivalent updates.

Volume: `checkpoints/iam_lbfgs_geometry/20261005-113159/index.html`;
refit galleries/checkpoint: same run's `pen_refit/` directory.
Source SHA: `c59a3d7d70405d10d0de6f284a8866bd0dd13c669948a7cc36d141132db806b2`.
Head-refitted SHA: `700f84eda8b523917f0c7f337bc21f75cc6d4075a101edc979fa5432212cd4f8`.

### Frozen English CTC head A/B

After inspecting close geometry and perfect genuine pen boundaries, freeze the
**entire** handwriting/pen/variance/style model and cache its eight latent means.
Train only the original OCR Transformer head, through **VAE.get_ocr_loss()**.
Two identical source heads/data/RNG, changing only initial blank bias−5 versus0.
Batch1/accum8, OCR dropout0.1 (unchanged), AdamW LR5e-4→1e-4 after800 updates,
clip5, 1,000 updates per arm. Correct target+adjacent-repeat CTC lengths are
checked before allocation; CPU backward is finite with no encoder gradients.

Five lines contain adjacent repeated labels, three do not. Compare CER in both
groups and overall; this is cached-latent training-set memorization, **not joint
VAE/CTC training**, useful OCR generalization, or full English generation.
Geometry/pen/style state is checked bitwise unchanged for both arms. Blank bias
is trainable after initialization, not permanently clamped. The −5 arm already
reaches CER0 /8 exact transcripts at500 updates; zero bias reaches that at250.
This disproves an absolute “−5 cannot learn English repeats” claim on this test,
while supporting zero as a reasonable next engineering control. Final results
and decoded transcripts are on the Volume, not inferred from CTC loss alone.

CTC run: `checkpoints/iam_ctc_head_ab/20261005-114305/`.
**KL, style and joint encoder/decoder CTC training remain off/unvalidated.**
Originally proposed next: joint sampled-latent reconstruction/pen + tiny corrected
KL + warmed OCR. The curve audit below supersedes that promotion: residual shape
distortion needs its own visual gate before joint auxiliary training/scaling.
All65 unit tests pass in root/fork. No restricted IAM/checkpoint files are pushed.

Combined report: `checkpoints/iam_objective_study/research-summary/index.html`
(on your shell: `/mnt/diffink-data/checkpoints/iam_objective_study/research-summary/index.html`).
Final CTC CER is0 for both arms at1,000 updates, with all8 transcripts exact;
independent CPU checks confirm ONLY OCR state changed and every saved weight is
finite. Joint encoder/decoder auxiliary training remains a separate next test.

```sh
# Rebuild the existing session report on CPU; no GPU/training:
python -m iam_tools.report_research
# Executed deterministic geometry diagnostic (not production training):
# modal run modal_objective_study.py --train --arm lbfgs --steps 50 \
#   --resume-checkpoint /data/checkpoints/iam_objective_study/20261005-111735/xy_pen/checkpoint.pt \
#   --resume-sha c59a3d7d70405d10d0de6f284a8866bd0dd13c669948a7cc36d141132db806b2
# Executed frozen OCR-head A/B (runs BOTH arms, not joint VAE):
# modal run modal_objective_study.py --train --arm ctc --steps 1000 \
#   --resume-checkpoint /data/checkpoints/iam_lbfgs_geometry/20261005-113159/pen_refit/checkpoint.pt \
#   --resume-sha 700f84eda8b523917f0c7f337bc21f75cc6d4075a101edc979fa5432212cd4f8
```

### Initial user-identified curve distortion (before the follow-up)

CPU-only audit of all eight saved posterior-mean/mixture-expectation trajectories,
with focused marker-free common-scale crops of `p08-936z-05` (c in chocolate)
and `a07-421z-02` (h in hope). The report's old **input was the RDP target**, not
raw IAM. Re-parsing canonical JSON exactly reproduces the HDF5 targets. Raw IAM
and RDP both contain polygonality; the model adds real kinks that persist without
point markers, with true pen boundaries and no sampling/dropout/display smoothing.

Within-stroke first-difference relative RMS errors for these full lines are
17.0% /20.2%; second-difference errors are32.7% /37.6%. In the indicated strokes,
second-difference errors are72.4% /54.3%. These are differences by **point index**,
not physical velocity or arc-length-normalized geometric curvature: RDP spacing
is uneven. Max coordinate-vector errors across the full lines are0.06483 /0.07950
model units (full normalized line height1); global X/Y averages hide local spikes.

Pointwise position loss does not directly match neighboring segment directions.
Residual displacement can turn a shallow arc into a corner while preserving low
RMSE, perfect pen state, and exact OCR transcript. This identifies the immediate
geometric error, **not** a proven architectural/optimization root cause. One-line
capacity does not prove all eight lines have been optimized to equivalent quality.
Do not mask the issue with spline rendering or claim visual fidelity has passed.
Next diagnostic if training: matched point-MSE control versus a target first/second
difference anchor, within real strokes only (not generic smoothing to straightness).

Report: `checkpoints/iam_objective_study/research-summary/curve-audit/index.html`.
Run `python -m iam_tools.curve_audit` or rebuild `iam_tools.report_research`;
both are CPU-only and perform zero optimizer steps. No new GPU credit spent.

### Completed curve fidelity follow-up

See [the detailed investigation](CURVE_FIDELITY.md). Continued position-MSE was
the largest improvement; calibrated target first-difference matching improves
the remaining local errors without rounding away real target corners. Best
mean X/Y RMSE0.000538/0.001437, tangent/turn p90 3.22°/5.17°, genuine pen F1=1
on all eight means and all160 sampled-z reconstructions. No architecture change,
RDP change, smoothing, or OCR/KL/style objective was needed.

Overview: `checkpoints/iam_curve_study/research-summary/index.html`.
Final raw/target/before/after/all-eight/sample gallery:
`checkpoints/iam_curve_study/comparison-180/report/index.html`.
Selected checkpoint: `checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt`.
These are memorized training lines, not full English generation/paper reproduction.
Joint auxiliary/regularized latent behavior remains a separate bounded test.

### Completed bounded posterior/KL/OCR integration

Three sequential400-update T4 stages preserve the reference and improve local
geometry. Mean X/Y RMSE0.000270/0.001162, turn p90 3.63° (reference5.17°);
all8 means and all160 draws retain genuine pen F1=1 and OCR CER0. The transferred
OCR head already works on the changed encoder, so warmup needed zero updates.
A paired-noise CPU control with **original fixed std** improves too, ruling out
variance narrowing as the only source of the sampled improvement.

See [LATENT_INTEGRATION.md](LATENT_INTEGRATION.md) for exact objective, failed
attempts, gates, clipping, tiny-KL/CTC limitations, hashes and commands. This is
custom mean-anchored engineering training, not the paper/production GMM baseline.
Tiny KL does not establish prior matching; held-out/multi-writer behavior remains
unvalidated. The standard trainer now rejects unsupported mean-anchor configs.

Volume report: `checkpoints/iam_latent_integration/latest/index.html`.
Final: `checkpoints/iam_latent_integration/20261005-133620/ocr/checkpoint-best.pt`.
89 root/fork tests pass. All T4 jobs completed/stopped.

### Same-writer expansion and held-out reconstruction

The faithful eight-line source is preserved. Expanding to24 training lines with
four untouched, line-disjoint samples exposes a real generalization gap. A1200
update Adam run and80-outer-step full-set L-BFGS diagnostic improve positions but
**do not pass visual/local-curve fidelity**. Geometry-only optimization also
weakens the tested pen-head compatibility; head-only repair is insufficient on this expanded set.
Frozen-feature linear XY probes are worse, so a simple readout swap is not a fix.

See [WRITER_EXPANSION.md](WRITER_EXPANSION.md) for exact splits, gradients, metrics,
configs, hashes, marker-free galleries and the bounded balanced-joint follow-up.
The latter was dashboard-interrupted after saved step200, then explicitly resumed
from model+Adam+RNG with the same sample-order suffix and remaining1000 updates.
Validation never enters any gradients, caches, calibration or checkpoint choice.
This is seen-writer/line-disjoint (forms overlap), not an IAM benchmark.

Volume reports: `checkpoints/iam_writer_expansion/latest/index.html` and
`checkpoints/iam_writer_polish/latest/index.html`. No original checkpoint is
replaced; no production/full-IAM/InkDiT readiness claim is made.

The balanced continuation finished at1200: train mean X/Y RMSE .009704/.013157,
pen F1 .9883; held-out .149929/.060841 and F1 .7372. Training renders are mostly
readable, but local curves regress relative to the immutable eight-line reference
and held-out lines remain visibly jagged. No expanded fidelity pass. The latest
report includes marker-free target/reference/expanded comparisons.103 tests pass;
all bounded jobs finished.

### Expanded-curve conditioning and optimizer investigation (2026-10-06)

See [CONDITIONING_STUDY.md](CONDITIONING_STUDY.md) for the new controlled T4/CPU
investigation, source hashes, exact objectives and failure accounting. The
higher-LR Adam control barely improves local curves. Channel-only normalization
removes most measured global padding drift but initially damages learned
reconstruction; it is opt-in research, not a default architecture change.
Centering the pretrained inputs produces extreme posterior variance and the
sampled Adam arm aborts before an update; it is not silently retried.

Full-set joint mean geometry **and pen** L-BFGS makes substantially more progress
than the expanded Adam endpoint without adding a smoother, changing RDP, or
turning on OCR/KL/style. A matched target-relative segment ablation weights short
true-stroke errors moderately (20% initial decoder gradient), with a target q25
length floor. It improves angle error modestly, not held-out fidelity. All4
held-out lines stay strictly evaluation-only. Compare shared outer steps; a
fresh-optimizer weight continuation is explicitly separate from matched A/B.

Volume overview: `checkpoints/iam_fullset_joint/research-summary/index.html`.
Reports include every old/new/held-out line, the c/h crops, immutable faithful
8-line reference, mean and20-draw metrics, corner/turn/Δ errors, and independent
CPU reload checks. Full configurations/as-run code are preserved per experiment.
Default architecture/data contract remains unchanged; standard paths fail closed
on unsupported research normalization/objectives. Distinct Modal app names and
atomic pointers to immutable dated reports avoid ambiguous concurrent jobs or
mixed latest galleries.127 root/fork tests pass. No paper-reproduction or full-IAM/
InkDiT readiness claim.

The completed geometry polish reaches train X/Y .000760/.001425 and turn p90
6.00°, with pen F1=1 for all24 means/all480 sampled draws; unseen4 still fail.
The next bounded geometry-only pilot uses the existing192/32,8-writer data and
keeps every prior validation line excluded. No OCR/KL/style enablement. See
CONDITIONING_STUDY.md for the explicit source, objective, calibration and limits.


CUDA graph training acceleration and CPU metric workers are now validated and
wired into the guarded English geometry research runner; no larger physical
batches or changed precision.136 tests pass. Exact performance/fidelity evidence
is in CONDITIONING_STUDY.md. The192-line follow-up remains rough; the separate
near-identity control is explicitly untrained and not a production promotion.

### Initialized transport follow-up (engineering, not generative promotion)

A controlled T4 study found that the near-lossless initializer is extremely
sensitive to ordinary Adam updates. Freezing its high-gain Transformer/readout,
using body LR1e-7 and posterior LR1e-3 preserves faithful geometry after200 actual
updates: train192 mean X/Y5.40e-6/4.39e-6, held-out32 5.14e-6/4.33e-6. Posterior
errors improve from~.002 to~.00061; all224 means+4480 draws have perfect pen states.
Named c/h crops and all original8 are visually faithful to the processed target.
This is an explicitly reinitialized transport codec, NOT paper reproduction or
proof of semantic/generative latents; frozen OCR still has~78–79% CER, KL is OFF.
No production model promotion, no InkDiT/full-IAM launch. See [CONDITIONING_STUDY.md](CONDITIONING_STUDY.md).

Volume overview: `checkpoints/iam_initialization_study/research-summary/index.html`.
All-line report: `checkpoints/iam_initialization_study/20261006-170142/report/index.html`.
Selected checkpoint: `.../protected_noise/checkpoint-best.pt` (SHA5ba90c38…).
Guarded launcher `modal_initialization_study.py`;139 regression tests pass.

## Initialized-codec KL compatibility study (2026-10-07)

Matched200-update T4 continuations restore the same protected source/Adam/RNG.
All arms preserve near-lossless mean curves; only KL0 has perfect final posterior
pens. KL1e-6 final200 has3 internal EOCs (selected100 is clean); KL1e-5 has noisier
posterior curves and6 faulty draws, including5 internal EOCs.
Small KL1e-6 lowers prior loss but is NOT a reconstruction improvement over KL0:
most reduction is increased pen-field uncertainty, not a more Gaussian coordinate
mean distribution. OCR/style/GMM stay off; generation remains untested.

Volume report: `checkpoints/iam_codec_kl_study/latest/index.html`.
Dated artifacts: `checkpoints/iam_codec_kl_study/20261007-010118/`.
See `CONDITIONING_STUDY.md` for exact configurations, metrics, caveats and hashes.
Opt-in `modal_codec_kl_study.py --train --steps 200`; CPU-only `--report-rel ...`
/ `--tails-rel ...`.147 tests pass; no core architecture/production-default change.

Follow-up `20261007-012331`: freeze only feature-dependent pen-logvar weight rows,
retain bias learning and KL1e-6. All4480 sampled pens/stops are now correct, with
near-lossless mean curves and unchanged sampled XY fidelity. Strict checkpoint
eligibility now catches internal EOCs even when pen-up F1=1. Reproduce with
`--train --steps 200 --protected-pen`; see the stable latest report above.

### Frozen English OCR refit (2026-10-07)

`checkpoints/iam_frozen_ocr_study/research-summary/index.html` contains the new
combined report, complete transcripts and all224 marker-free comparisons.
1000-update mean-head fitting + matched500-update continuations preserved every
non-OCR state tensor and all4704 mean/posterior trajectories bitwise. The
mean-only head now memorizes192/192 training transcripts (CER0); held-out CER is
still≈83–84%, so this is NOT English OCR/generalization or generation readiness.

A distinct posterior-OCR issue was isolated:344 near-unused latent channels have
means≈1e-7 but std≈1; mean-only training doesn't reject their inherited projection
noise. Sample-aware fitting cuts train sampled CER61.24% →3.75% versus a matched
mean-only control. A separate paired CPU diagnostic zeros ONLY these OCR input
projection columns and yields3840/3840 exact training posterior transcripts,
without any mean-transcript or codec change. Diagnostic weights remain separate;
not automatic pruning of learned latents. Full detail/configs/hashes and caveats
are in CONDITIONING_STUDY.md.

Reusable patch: OCR now masks padded attention and removes invalid features
before projection; `get_ocr_loss` accepts Boolean/binary-float valid masks.
Raw lines remain minimally padded microbatches; only cached OCR inputs batch16.
`modal_frozen_ocr_study.py` is explicit opt-in, bounded T4, no retries/crash loop.
160 root/fork tests pass. Geometry remains faithful; next investigate unseen-text
recognition while protecting it, rather than more curve polishing or blindly
unfreezing the VAE/adding style.

### Frozen OCR context controls — 2026-10-07

`modal_ocr_context_study.py` completed four sequential1000-update T4 head-only
controls from the unchanged9c53f68 codec. Fresh identical seeds/weights,
train-only feature calibration, identical sample/LR schedules, batch16 cached
means, AdamW5e-4→1e-4/dropout.1; no geometry/KL/style/readout updates. Selected
TRAIN-only checkpoints. Held-out mean CER: absolute raw83.0645%, standardized
absolute80.2419%, standardized relative-X**71.8750%**, relative-X local-radius4
72.9839%. Three arms train CER0/192 exact/3840 exact posterior draws, but **none
reads any of the32 unseen lines exactly**. Curve fidelity remains protected;
no head promoted and no joint training/InkDiT launch.

Report: `checkpoints/iam_ocr_context_study/20261007-031059/report/index.html`
on `diffink-data`; local mirror under `data/`. All224 transcript comparisons,
all32 marker-free held-out target panels, learning curves, pinned checkpoints,
CPU reload/config/schedule/source audits. Every mean transcript matches CPU/GPU.
170 tests pass. See `docs/CONDITIONING_STUDY.md` for exact results/hashes and
research-only polyphase feature contract. The optional core OCR attention mask
preserves default behavior. A larger TRAIN-only pool is the next generalization
test; missing labels explain only1/992 held-out characters. Global context
ablation demonstrates dependence, not a sole-cause diagnosis. No paper-English,
writer/form-independent benchmark or semantic-generative readiness claim.

### Prompt-guarded OCR supervision expansion — 2026-10-07

Completed two sequential T4 controls,6000 updates each: nested192 versus2048
TRAIN lines with the same186 writers, identical fresh OCR weights/feature
moments/LR budgets. DEV128 from five reserved writers selects checkpoints;
original32 reports only. TRAIN excludes both evaluation prompt families
(including IAM writer-version variants) and normalized transcripts;25 test
writers never enter training. Original overfit data and all codec tensors stay
unchanged. New `iam_tools/ocr_pool.py` prepares a separately pinned pool safely.

DEV CER **75.64%→28.29%**, original32 **74.50%→25.10%**; larger TRAIN2036/2048
exact/.01925% CER. All2208 observed pool trajectories pass mean curve/pen gates
(X/Y RMSE7.24e-6/5.70e-6, turn p90.04094°, perfect boundaries/EOCs). No added
spikes, no geometry optimization. All selected mean predictions agree CPU/GPU.
Report `checkpoints/iam_ocr_pool_study/20261007-033445/report/index.html` (local
`data/` mirror) includes all160 evaluation target/shared-reconstruction panels,
transcript comparisons, learning curves and audits.182 tests pass. See
`docs/CONDITIONING_STUDY.md` for configs, hashes and strict provenance caveats.

Data starvation/memorization is a major OCR contributor;25–28% unseen CER still
isn't dependable. This is not a paper/independent-pretraining IAM benchmark or
a generative latent proof. The frozen codec's earlier geometry training had
prompt overlap with original32. Research heads remain standalone with explicit
polyphase feature contracts; no joint VAE/DiT launch. Next useful test is more
TRAIN-only supervision, not further tiny-set memorization or releasing geometry.

### Paired OCR continuation:2048→8192 — 2026-10-07

Same selected2048 head/Adam moments/counters/RNG/calibration,6000 additional
T4 updates per arm. Continuing2048 yields DEV27.71%/report32 23.39% CER;
expanding TRAIN to8192 yields19.22%/16.23%. Parent baseline28.29%/25.10%.
DEV-only selection: control step8000, expanded12000; final expanded step is
still learning (TRAIN7.22%), so this is not convergence or a dependable teacher.
Posterior20-draw CER19.27%/16.30% closely matches means. Same186 writers,
fixed81 characters, pinned calibration192, DEV128/report32 IDs and every parent
payload; train/form/transcript/test-writer guards retained. Whole codec stays
bitwise unchanged. All8352 observed means pass fidelity gates (X/Y RMSE
7.25e-6/5.70e-6; turn p90.041°; perfect pen/EOC). No new spikes or smoothing.

Report: `checkpoints/iam_ocr_pool_expansion/20261007-041949/report/index.html`
(local `data/` mirror), all160 eval panels with before/control/expanded OCR
captions; trajectory is shared and intentionally identical.189 tests pass.
See `docs/CONDITIONING_STUDY.md` for exact configs/hashes, CPU reload scope,
FP32 baseline-loss tolerance/regression and historical metadata clarification.
Next: bounded reader convergence on the fixed larger pool, keeping geometry
locked. No unrestricted joint VAE/OCR/KL/style/InkDiT; no paper reproduction or
independent-pretraining benchmark claim.


### Fixed8192 OCR convergence/readout diagnostic — 2026-10-07

Two matched8000-update T4 continuations from step12000, LR1e-4 versus2e-4;
same batches, restored optimizer/RNG, resumed iterator. TRAIN CER7.22→2.56/3.13%,
but unseen-writer DEV19.22→19.71/19.84%. BOTH select unchanged step12000.
Do not promote higher-LR final based on seen-writer report32 14.21%.
CPU beam10 (no LM) only moves DEV19.22→19.01%; large backward pen jumps occur
in just2/128 DEV lines, so neither explains the broad reader error. Geometry
remains frozen and faithful, all8352 source gates pass; no new handwriting or
joint codec training. Next justified hypothesis: controlled OCR-only frame
resolution, not blind longer optimization. Not yet launched.

Report `checkpoints/iam_ocr_convergence/20261007-044341/report/index.html`,
interpretation `report/conclusion.html`, local `data/` mirror on Volume
`diffink-data`. Exact configs/hashes/caveats and negative results in
`docs/CONDITIONING_STUDY.md`.203 root/fork tests pass; all Modal apps stopped.


### OCR-only4 versus8-point frames — 2026-10-07

Controlled fresh readers: same weights/parameter count, TRAIN8192/calibration192
IDs,8000 updates/batch IDs/LR schedule; four-point grouping preserves every
chronological XY/pen field without resampling. DEV CER15.30→11.43%, report32
12.40→10.18%.20-draw posterior DEV15.27→11.51%. CPU reload192 lines gives zero
mean transcript differences. DEV84 improved/22 tied/22 worse; report exact2→1
although total errors fall. No claim every line improves or CTC length alone
causes gain: input grouping/calibration/position indices differ; dropout unpaired.

All8352 observed mean trajectories/pen gates pass and entire codec stays bitwise
frozen. Gallery shows SHARED handwriting, compare OCR captions only. This is an
initialized polyphase transport research reader, not semantic-VAE reproduction
or reliable joint-training oracle. Keep geometry locked; next replicate finer
frame benefit or test2-vs4 OCR frames, not blindly enable joint VAE/CTC/KL/DiT.
Report `checkpoints/iam_ocr_frame_study/20261007-052514/report/index.html` and
`report/conclusion.html` on `diffink-data`, local `data/` mirror. Exact config/
hashes/limitations in `docs/CONDITIONING_STUDY.md`.213 tests pass; all jobs finished.
