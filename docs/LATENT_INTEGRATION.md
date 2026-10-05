# Bounded English InkVAE posterior / KL / OCR integration

This follow-up starts from the visually faithful curve checkpoint, not from the
old rough-curve or 192-line runs. Its purpose is **compatibility**, not more
one-glyph polishing or a claim of English generalization.

## Immutable reference

- Branch base: `f37e421`.
- Source: `checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt`.
- SHA256: `52f2417e218b21736d46f1502156b4d31f4cd273c1ef669ec8ede51ee759255d`.
- Same eight training lines, writer10174, unchanged manifest/targets/RDP0.5.
- Posterior mean + mixture expectation: X/Y RMSE .000538/.001437,
  mean per-line turn p90 5.17°. All8 means and all160 sampled-z pen states exact.
- All original reference/checkpoint files remain unchanged. Each run stores a
  fresh fixed-seed reference evaluation and exact as-run source copies.

## Experiment controls

No GMM likelihood, style objective, augmentation/rotation, rendering smoothing,
architecture change, normalization change, or RDP change. Model scale .01;
decoder dropout0 and padding mask on. Physical batch1, every update averages
all eight separately/minimally padded lines. OCR dropout is explicitly disabled
with eval mode for this controlled test; gradients remain enabled.

Define `G(z) = XY_MSE(z,target) + .20471838744633777 * target_Δ_MSE(z,target)`.
The difference term matches the target, within true strokes only; it is not
physical velocity or generic smoothing. Final retry objective:

```
100 * G(sampled_z) + 1000 * G(mu)
+ bounded_pen_weight * focal_pen(sampled_z)
+ stage_KL_weight * corrected_KL(mu,logvar)
+ stage_CTC_weight * CTC(sampled_z)
```

These are explicit research-runner coefficients, not the released/paper objective
and not changes to default production loss weights. `target_delta_weight` in the
saved config is the **inside-G** coefficient; actual sampled/mean Δ coefficients
are20.4718387446 /204.7183874463 respectively.

- Fresh AdamW each stage; betas(.9,.99), weight decay0, global clip5.
- Geometry/variance LR5e-7; last25% uses0.3× LR. Joint OCR head LR1e-5.
- 400 updates/stage, 900s training-loop cap/stage, 3600s Modal timeout.
- Training samples new latent noise each update. Eval20 fixed seeds per line;
  all160 draws included in metrics, median/worst XY-error samples shown.
- Pen focal gamma2, square-root inverse-frequency normalized to continue, cap8;
  real points only, no forced final EOC. Pen encoder-gradient contribution is
  capped at10% of initial geometry norm, coefficient capped at1.
- Style parameters frozen. Sigma/rho FC rows frozen (their outputs can still
  change as shared features change). OCR frozen until its deliberate stage.
- No bitwise CUDA reproducibility claim; RNG/provenance/configuration preserved.

### Sequential stages

1. **sampled**: train encoder/mean/logvar/decoder/pen jointly; KL/CTC off.
2. **kl**: same objective, corrected per-valid-latent-element KL coefficient1e-6.
   This is a tiny-KL compatibility test, **not** evidence of Gaussian-prior matching.
3. **ocr**: transfer only old blank-zero OCR head (SHA
   `b4c78fe5277bb8884e7e0203aa0c655cb1cfc08d51f0a07de2bac48cabbac0bd`),
   first measure it against NEW latents, warm/refit on cached new mean+sampled
   latents if needed, then joint encoder/decoder/variance/pen/OCR training.
   Warmup cap500, AdamW5e-4, no decay, clip5, evaluate every50; all non-OCR
   weights bitwise invariant. It must reach CER0 on means AND all160 draws.
   Joint CTC weight targets10% of full-eight-line initial encoder geometry
   gradient, capped at0.1, then remains fixed. Old CER0 is never assumed.

## Promotion gates

Checked separately for **every line**, not just aggregate metrics:

- Mean X/Y RMSE ≤max(2× reference value, .0025 model units).
- Mean turn p90 ≤max(1.5× reference value,9°).
- All160 sampled mean RMSE/turn p90 aggregates ≤1.2× reference.
- Mean pen F1≥.99; every sampled pen F1≥.95; every final EOC correct;
  no nonfinal false EOC.
- OCR-stage mean AND sampled CER0.

Evaluate every100 updates. Gate failures at/after200 stop the stage; no later
stage starts. Select minimum sampled target-Δ error among passing evaluations,
not the last checkpoint by default. A numerical pass still needs marker-free
visual review. These gates tolerate small positional tradeoffs; they do not
claim exact identity to the reference or independently calibrated human perception.

## Attempts retained

- `20261005-132930`, app `ap-EUCTh6Hsg8yZGmh7ilFYKV`: zero optimizer updates;
  posterior-stat diagnostic used the upstream float downsampled mask as an index.
  Fixed by converting it to bool; a real finite-backward regression test covers it.
- `20261005-133203`, app `ap-eAL2sVICOYJNpdoGjcHHCr`: sampled-only, LR1e-6,
  equal sampled/mean weights100, stopped at200 on X-position gate. Curves/pen
  remained good but X RMSE rose to.002303. No KL/OCR trained, not promoted.
  Mean-vs-centered residual diagnostics show both positional bias and shape error,
  not exclusively rigid translation. `attempt-result.json` preserves the findings.
  A post-stop checkpoint reload rejected TorchVersion metadata; storing a plain
  version string fixes weights-only loading, regression-tested. Saved original
  attempt checkpoints are retained, not silently rewritten.
- Retry changes BOTH mean-anchor strength and LR. It is an engineering correction,
  **not** a one-factor causal ablation of either individual change.

## Reproduce / inspect

Use the root `venv`, or create a venv in the fork. Import/default CLI allocates no
GPU; explicit training required:

```sh
venv/bin/modal run --detach modal_latent_integration.py --train --steps 400
venv/bin/python -m iam_tools.report_latent_integration \
  data/checkpoints/iam_latent_integration/<completed-run>
venv/bin/python -m unittest discover -s tests
```

Modal Volume `diffink-data` mounts at `/data` inside jobs and
`/mnt/diffink-data` in the user's shell. Reports/checkpoints are under
`checkpoints/iam_latent_integration/`. Datasets and generated images/checkpoints
are not committed. No SSH or GPU training on import is required.

## Scope and remaining questions

A pass only establishes eight-line integration mechanics. Held-out reconstruction,
multiple writers, style supervision, length variability/GroupNorm sensitivity,
production GMM+XY training, posterior/prior behavior and InkDiT remain separate.
Increasing tiny KL substantially needs an explicit tradeoff experiment, not an
incidental roadmap toggle. Preserve the geometry reference throughout scaling.

## Completed result (2026-10-05)

Run `20261005-133620`, app `ap-b4kfPW6Ebli7qGEhMm17yY`: all three stages
completed400 updates each (1200 effective optimizer updates /9600 physical
microbatches). Every stage selected its passing step400 checkpoint. The final
marker-free means and median/worst-of20 stochastic galleries were visually
reviewed on **all eight lines**, including the two named curves. No geometric
regression visible at ordinary viewing scale; real RDP corners remain intact.

| Stage | Mean X/Y RMSE | Δ / Δ² relative RMS | Tangent p90 | Turn p90 | True-corner turn p90 | Sampled X/Y RMSE | Sampled turn p90 |
|---|---|---|---|---|---|---|---|
| reference | 0.000538 / 0.001437 | 2.57% / 4.70% | 3.22° | 5.17° | 7.92° | 0.002884 / 0.002036 | 9.14° |
| sampled | 0.000341 / 0.001298 | 2.23% / 4.03% | 2.55° | 4.25° | 6.45° | 0.002763 / 0.001913 | 8.50° |
| kl | 0.000291 / 0.001223 | 2.08% / 3.75% | 2.35° | 3.91° | 5.96° | 0.002692 / 0.001839 | 8.17° |
| ocr | 0.000270 / 0.001162 | 1.97% / 3.54% | 2.19° | 3.63° | 5.81° | 0.002627 / 0.001776 | 7.89° |

All stages retain genuine pen F1=1 on means AND all160 sampled reconstructions;
all final EOCs correct, zero false internal EOCs. The transferred trained OCR
head already has CER0 on the new8 means/160 draws, so warmup performs **zero**
updates. Joint OCR retains CER0 at every evaluation; final sampled CTC loss
7.37e-6. Its capped weight0.1 corresponds to only0.0023% of the initial geometry
encoder-gradient norm because the head is already memorized. This is compatibility,
not a strong regularization stress test or proof that OCR improves geometry.

Tiny KL1e-6 similarly does **not** establish prior matching: mean per-element KL
rises from7.130 to7.231, and average posterior std falls from0.005578 to0.005138.
Reconstructions improve partly because uncertainty narrows. A CPU paired-noise
control deliberately holds each original latent std element fixed while using
the final encoder mean/decoder:

| Paired CPU condition (160 draws) | X/Y RMSE | Δ relative error | Turn p90 | True-corner turn p90 |
|---|---|---|---|---|
| reference | 0.002954 / 0.002040 | 4.67% | 9.05° | 13.64° |
| final_current_std | 0.002699 / 0.001781 | 4.11% | 7.81° | 11.91° |
| final_reference_std | 0.002908 / 0.001854 | 4.35% | 8.25° | 12.56° |

The fixed-original-std control improves too: this is **not solely variance
collapse**, though smaller variance adds improvement. CPU draws share identical
epsilons across conditions but are not bitwise the saved GPU draws. Latent
channels remain index-aligned; this is a sensitivity proxy, not equality of
original/final latent distributions. Both final CPU conditions retain pen F1=1
and CER0 on all160 draws.

Clipping remains frequent (sampled78.5%, KL80.75%, OCR89.25%; median raw norms
15.55/18.09/18.52). These norms include100/1000 loss scaling. No convergence or
optimizer-quality claim follows from a passing fidelity gate. The sequential
results include more optimization; improvements cannot be attributed causally
to KL or OCR without matched-off controls.

### Saved artifacts and independent checks

- Volume report: `checkpoints/iam_latent_integration/20261005-133620/report/index.html`.
- Stable report link: `checkpoints/iam_latent_integration/latest/index.html`.
- Final checkpoint: `checkpoints/iam_latent_integration/20261005-133620/ocr/checkpoint-best.pt`.
- SHA256: `fffe1405db10f8f6c2ce6ec1e030706b7947c93d83fa4eaeffec3b6a8c5b08f9`.
- All-stage configs, calibrations, logs, initial/best/last/every100 checkpoints,
  all20-draw NPYs, immutable reference, exact as-run source/hashes retained.
- `cpu-variance-control/summary.json`: independently safe-loaded model is finite,
  style weights and sigma/rho FC rows unchanged, posterior head genuinely changed;
  mean CPU/GPU render agrees within tolerance (max absolute difference1.34e-5),
  pen argmax identical. No CPU/GPU bitwise equivalence claim.
- Original reference SHA verified unchanged. `attempts.json` includes both failed
  attempts; none silently promoted. Later committed runner additionally saves
  the warm-head checkpoint, including failed warmups. In this as-run zero-update
  warmup the transferred head is preserved in `ocr/checkpoint-initial.pt`.
- 89 root/fork tests pass. Float-mask indexing, real joint finite backward, safe
  version-metadata loading, per-line/all-draw gates and a standard-trainer guard
  are regression-tested. The standard trainer rejects a mean-anchor config it
  cannot implement instead of silently dropping the loss.

**Bounded eight-line sampled/KL/OCR integration gate: PASS.** Next is more lines
with explicitly held-out reconstruction, then multiple writers/style; not full
IAM or InkDiT. This custom mean-anchored objective is not yet implemented as a
production multilingual training baseline. Preserve the reference and galleries.
