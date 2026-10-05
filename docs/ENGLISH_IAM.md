# English IAM-OnDB → DiffInk

**No training or GPU job has run.** This workspace now implements a raw parser,
experimental preprocessing, a tiny HDF5 exporter, and CPU loader validation.

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
