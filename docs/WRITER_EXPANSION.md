# English InkVAE: same-writer expansion and held-out reconstruction

This follows the eight-line posterior/KL/OCR compatibility study. That study
passed its training-sample fidelity gates; it did **not** prove generalization.
The original reference remains immutable.

## Protocol

- Writer `10174`: all 24 existing training lines (original eight + sixteen new).
- Four existing validation lines are strictly evaluation-only: `a07-421z-01`,
  `a07-421z-04`, `h05-195z-02`, `k07-640z-03`.
- This is **seen-writer, line-disjoint**. Forms overlap with training, so this is
  neither a form-independent nor writer-independent benchmark. Official held-out
  writers are not involved.
- Pinned manifest SHA: `6d1fafea62af6c5ddae48983bb9699d93016e4af7a11df2f8dd1743128c34c8a`.
- Source: `checkpoints/iam_latent_integration/20261005-133620/ocr/checkpoint-best.pt`.
  SHA: `fffe1405db10f8f6c2ce6ec1e030706b7947c93d83fa4eaeffec3b6a8c5b08f9`.
- Held-out lines never enter optimizers, gradient calibration, pen/OCR caches or
  checkpoint selection. Training-only checkpoint score combines mean XY error,
  target first-difference relative error, pen F1 and sampled XY error. It is an
  engineering score, not a calibrated perceptual metric.
- English normalization/RDP/line-final EOC, architecture and scale .01 unchanged.
  Dropout/augmentation/rotation/GMM/style/CTC/KL off for this geometry expansion.
  Frozen OCR CER is reported only as a latent/head transfer diagnostic.
- Physical batch1, minimal multiple-of-eight padding. No dynamic cross-line
  padding. All evaluations use mean and20 fixed sampled-z draws per line; no
  sampled draw is dropped from metrics. Before/after means and median/worst
  sampled galleries use predicted pen states and marker-free unsmoothed polylines.
- Differences are by **nonuniform point index**, not physical velocity/curvature.
  Tangent/turn metrics use spatial segment directions.

## Baseline

CPU no-training evaluation agrees with GPU mean reconstruction within normal
floating-point tolerance. The eight originals remain faithful; new and held-out
lines do not. This is direct evidence against interpreting the eight-line result
as a general-purpose English reconstruction model.

| Group | Mean X RMSE | Mean Y RMSE | Mean per-line turn p90 | Pen F1 (macro) |
|---|---:|---:|---:|---:|
| Original8 | .000270 | .001162 | 3.63° | 1.000 |
| New16 | .340718 | .094479 | 156.63° | .7010 |
| Held-out4 | .451014 | .124806 | 157.91° | .6777 |

CPU baseline uses three noise draws for a cheap preflight. Main GPU baseline and
subsequent evaluations use20. Aggregates are macro averages of per-line metrics;
turn p90 is an average of per-line p90s, not a pooled percentile.

## Experiment1: sampled/mean-anchored Adam expansion

Executed1200 optimizer updates, eight microbatches/update, full-training-set
shuffled epochs (each line occurs once per24-microbatch epoch), Torchseed4042,
sampler NumPyseed42, fresh AdamW betas(.9,.99), decay0, clip5. LR1e-5, multiplied
by.3 for the final quarter. Exactly9600 training microbatches =400 passes/line.
Objective:100 × sampled geometry +1000 × mean geometry + bounded pen focal loss,
where geometry is real-point MSE +.20471838744633777 × target ΔMSE. Pen gamma2,
sqrt-inverse-frequency normalized to continue, cap8, real points only.

All1200 updates clipped; raw gradient norm median first/last200 was988.2/200.0.
This experiment expanded training coverage but **failed the visual fidelity
standard**: the original named curves became noticeably distorted again. It is
retained as evidence, not promoted as a replacement for the faithful eight-line
reference. Frozen OCR/style weights are bitwise unchanged. The original source
SHA is unchanged. Best training score was step1200, not selected on validation.

| Group | X RMSE | Y RMSE | Turn p90 | Macro pen F1 |
|---|---:|---:|---:|---:|
| Original8 | .021056 | .018201 | 74.99° | .9895 |
| New16 | .033878 | .035227 | 113.35° | .9746 |
| All train24 | .029604 | .029552 | 100.56° | .9796 |
| Held-out4 | .166628 | .076964 | 141.41° | .8025 |

Held-out position error improved markedly from baseline, but local curves and
pen states still fail. It would be misleading to call this reconstruction
"lossless" or ready for InkDiT. Frozen-head CER on old8/new16/held-out4 is
.606%/77.905%/76.389%; the head was trained on the original eight only, so this is
not evidence that English CTC itself fails.

- Run: `checkpoints/iam_writer_expansion/20261005-142611/`.
- Selected Adam checkpoint SHA: `86b5adba337000b42d581f4e59eac2ce2948dd5c47257467abb4b0fa21d83981`.
- Actual loop/evaluation time656.72s on T4 (startup excluded).
- Report: `checkpoints/iam_writer_expansion/latest/index.html`.

A bounded full-set deterministic L-BFGS diagnostic follows. It changes optimizer,
effective accumulation and sampled-vs-mean objective together: **not a clean
single-factor causal ablation**, and not proposed as the production trainer.
Its purpose is to distinguish failure to optimize24 observed trajectories from
failure to generalize to the four untouched trajectories.

## Experiment2: deterministic full-set optimizer diagnostic

From the step1200 Adam checkpoint above, fresh L-BFGS (LR1, max_iter10,
max_eval15, history10, strong-Wolfe, tolerance_grad1e-8/tolerance_change1e-12).
Mean latent only, mean geometry objective averaged across all24 train lines via
physical microbatch1. Delta coefficient stays.20471838744633777. No gradient
clipping. OCR/style/logvar-head parameters and pen/sigma/rho FC rows frozen during
geometry. Then2000 frozen-feature pen-head updates atLR.001, bounded gamma2/cap8,
cycling mean + four fixed sampled features for each **training** line. This pen
refit changes only the first three FC rows; coordinate output is bitwise unchanged.

Completed80 outer L-BFGS updates with928 closure calls, followed by the2000
pen-head updates. Actual loop/evaluation/refit703.45s (pen refit39.10s). Saved
snapshot label2080 is80 geometry +2000 pen updates, **not2080 L-BFGS updates**.
The reusable runner now records these counters separately. The as-run snapshots
and original files remain preserved.

| Group | Mean X/Y RMSE | Δ / Δ² relative error | Turn p90 | Mean / sampled macro pen F1 |
|---|---|---|---:|---|
| Original8 | .011314 / .012854 | .287 / .519 | 61.00° | .8928 / .8943 |
| New16 | .012430 / .015744 | .323 / .557 | 66.89° | .8472 / .8452 |
| Train24 | .012058 / .014781 | .311 / .544 | 64.92° | .8624 / .8616 |
| Held-out4 | .148978 / .060664 | 1.056 / 1.762 | 138.12° | .6798 / .6791 |

Position errors improve, but this is **still not visually lossless**. Named
curves remain distorted and geometry-only body optimization made features less usable by the
tested linear pen head. Head-only refitting cannot fully restore all boundaries.
Thus this checkpoint is a useful research branch, not a promotion over the
original eight-line reference. All20 sampled draws/line are measured; sampled
train X/Y RMSE.012464/.014892, held-out.149027/.060694.

- Run/report: `checkpoints/iam_writer_polish/20261005-143958/report/index.html`.
- Stable report: `checkpoints/iam_writer_polish/latest/index.html`.
- Selected checkpoint SHA: `a7945b5ef01d3a3423d8dba4a3d5fe2b8ae997be2815b12da0a82a46dca21167`.
- Independent CPU reload: finite, input source unchanged, OCR/style and sigma/rho
  rows unchanged; all28 means agree with saved GPU arrays (max XY difference
  8.58e-6), pen argmax identical. Pen refit verifies body/XY parameters unchanged.

## Additional CPU diagnostics

**Target derivative gradient is not vanishing.** At the Adam step1200 checkpoint,
full24-training-set decoder gradients have norms point.069821, delta.209750,
cosine.78994. The retained coefficient gives delta norm **61.5%** of point norm;
a50% calibration would choose.166439. This does not justify blindly raising the
coefficient or adding generic smoothing. Saved `decoder-gradient-calibration.json`.

**A linear readout does not rescue these frozen decoder features.** Fit a
float64 least-squares XY probe on the training features immediately before FC,
using point + target Δ equations with equal per-line weight; held-out equations
excluded. Model tensors unchanged. Includes intercept and a fixed SVD cutoff
1e-6 to drop near-null LayerNorm directions. A1e-10 variant is retained as a
conditioning diagnostic, not cherry-picked. This is not a replacement checkpoint
or a GMM sampling result. Both probes degrade geometry substantially; a simpler
linear readout on this frozen representation is not an immediate fix. It does
not rule out training a different readout end-to-end.

Probe and calibration artifacts live beside the Adam run; exact metrics and
coefficients are in `linear-readout-probe-rcond1e-6/` and `...-rcond1e-10/`.

## Experiment3: balanced joint reconstruction (interrupted, then resumed)

From the polished checkpoint, objective:

```
G(mu) + .1 G(sampled_z) + calibrated_pen_weight * .5[pen(mu)+pen(sampled_z)]
G = real-point MSE + .20471838744633777 * within-true-stroke target ΔMSE
```

Initial full-training-set encoder gradient calibration targets20% geometry norm
for pen, capped at scalar1, excluding validation. It chooses pen scalar
.0016613753687545062. Fresh AdamW, betas(.9,.99), decay0, geometryLR5e-6,
pen-rowLR5e-5, clip5; final quarterLR×.3. The shared FC tensor uses per-row proposed
Adam displacement scaling, not gradient scaling (which Adam would cancel).
Regression tests compare this against separate Adam parameter groups. Same
shuffled epochs, accumulation8, dropout/augmentation/KL/CTC/style/GMM off.

The original1200-update job was stopped from the Modal dashboard after the saved
step200 evaluation. **Not a numerical failure.** At200, train penF1 improved
.8624→.9261, mean X/Y .012058/.014781→.011333/.014391, turn p9064.92°→61.65°.
The source and partial artifacts remain at
`checkpoints/iam_writer_expansion/20261005-145727/`.

On2026-10-06 continuation restores the saved model, **Adam state and Torch CPU/
CUDA RNG**, reconstructs the same NumPy sample-order suffix, retains the original
calibration and completes exactly the remaining1000 of1200 planned updates.
It does not restart from the geometry-only source or change the experiment.
Resume checkpoint200 SHA:
`9b39fa4b331286af068c850f9f0df8b76363f149d252a4c14adda86450e10911`.
Any unsaved updates after200 are discarded rather than falsely called resumed.

Commands (GPU only with explicit `--train`):

```sh
venv/bin/modal run --detach modal_writer_expansion.py --train --steps 1200 --lr 1e-5
# CLI spelling uses spaces:
venv/bin/modal run --detach modal_writer_expansion.py --train --mode balanced \
  --resume --steps 1000 --lr 5e-6 \
  --source-rel checkpoints/iam_writer_expansion/20261005-145727/checkpoint-200.pt \
  --source-sha 9b39fa4b331286af068c850f9f0df8b76363f149d252a4c14adda86450e10911
venv/bin/modal run modal_writer_expansion.py --report-rel checkpoints/iam_writer_polish/20261005-143958
```

**Next gate is faithful reconstruction across the expanded set and untouched
lines, not joint OCR/KL, multi-writer/style or InkDiT yet.** Eight-line capacity
is still established, but extending that result is an unresolved optimization/
generalization problem, not something an aggregate RMSE improvement settles.

**A fixed eight-phase offset is not the main repair.** Estimate residual biases
for point-index modulo8 from train lines only, center to preserve the global
mean, then subtract those biases from all predictions. This does not smooth or
modify model weights, and pen states are unchanged. At Adam step1200, train Δ
relative error changes.5693→.5638 and turn p90100.56°→100.34°; held-out turn
141.41°→141.69° (worse). Tiny effect, not a practical fix. This rules against a
single shared phase-bias correction, **not** context-dependent deconvolution/
aliasing effects. Saved `phase-bias-probe/summary.json` and corrected arrays.
