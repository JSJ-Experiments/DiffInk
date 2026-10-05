# English InkVAE curve fidelity investigation (2026-10-05)

## Conclusion

The added jaggedness in the `73bf724` galleries was mainly **under-converged
geometry**, with position-only supervision providing a relatively weak local
direction constraint during finite optimization. It was not an inability to
represent the curves. Continued direct point-MSE already removed most artifacts;
a calibrated target first-difference anchor improved the remaining local fidelity
consistently on all eight lines. No architecture, RDP, coordinate representation,
normalization layer, dropout, mixture count, or display smoothing change was made.

The final `c` in `p08-936z-05` no longer has the added flat shelf; the `h` in
`a07-421z-02` follows the target arch rather than the earlier exaggerated roof.
All eight final mean renders were inspected, not just those two regions.

This passes the **eight-memorized-line geometry gate for a bounded joint integration
experiment**, not a full-IAM readiness/generalization/paper-reproduction gate.
No OCR/KL/style objective was enabled, and production Adam/GMM convergence is not
established by these deterministic L-BFGS diagnostics.

## Evidence and alternative hypotheses

- The saved canonical conversion exactly matches the HDF5 targets on all eight
  lines. Raw IAM and RDP already have some polygonality, but the old model adds
  distortions beyond that. Marker-free, equal-scale raw/RDP/prediction crops
  distinguish them. Nothing was spline-smoothed or resampled for display/training.
- Average RMSE hides directions: a small alternating point displacement can
  produce a large segment/turn error. Point-only optimization asymptotically
  constrains these too; the anchor is a conditioning/fidelity aid, not proof that
  MSE has an irreducible bad optimum.
- Short segments strongly amplify angular sensitivity: initial shortest-quartile
  tangent p90 ranges about37–73°, longest-quartile about6–12° across the eight lines.
  Unit-tangent matching helped true corner angles, but did not beat raw difference
  matching overall. No inverse-length weighting or generic curvature penalty
  was promoted. The sparse named `c` is not solved simply by favoring short edges.
- Global mod8 phase means explain0.087% of residual energy, below the500-permutation
  null p95=0.296% (seed42, within-line phase-label permutation). Some per-phase
  RMS variation exists; this single test does **not** exclude all possible
  upsampling effects. There was no evidence warranting a ConvTranspose replacement.
  Existing kernel4/stride2 overlap does not itself prove artifact-free decoding.
- Fixed latent mean, dropout0, mixture expectation, and true pens still exhibit
  the old kinks: dropout, random component selection, and wrong pen breaks are not
  the cause of those displayed errors. Minimal per-line padding remains fixed
  within the comparison. This does not solve broader GroupNorm/padding sensitivity.
- On the completed delta20 checkpoint, the top10% mixture-entropy points have
  no higher RMS error than the rest in any of the eight lines. Mixture ambiguity
  is not supported as the main remaining-error location. A float64 affine
  least-squares readout on frozen decoder features gives X/Y RMSE0.10025/0.03371,
  much worse than current0.00166/0.00401. It does not justify replacing the mixture
  head with a two-row affine head on those frozen features; joint retraining of a
  different head remains an untested hypothesis, not ruled out universally.

## Metric definitions

Coordinates are model units, normalized full line height1 (`model_input_scale=0.01`).
Point RMSE is reported separately for X/Y. Δ and Δ² errors are vector RMS errors
divided by the corresponding target vector RMS, **within true strokes** only.
They are point-index differences, not physical velocity or geometric curvature:
RDP spacing is uneven and timestamps do not enter this model.

Tangent error is the wrapped angle between corresponding segment directions.
Turn error is the wrapped disagreement between the signed angles of consecutive
segments. Both are spatial angles, in degrees. Zero-length edges and pen-up
windows are excluded. Target corners have absolute target turn≥45°; shallow
turns are<20°. Table percentiles are averages of each line's percentile, not
pooled percentiles. A supplemental symmetric vertex-to-corresponding-stroke
polyline distance avoids treating along-curve point displacement as perpendicular
shape error; it is **not** exact continuous Hausdorff distance.

## Exact controls

Eight IDs, writer10174:
`c08-434z-05`, `e08-429z-04`, `p08-936z-05`, `a07-421z-03`,
`k07-640z-02`, `l10-072z-02`, `h05-195z-03`, `a07-421z-02`.
Each physically collated separately with its minimal multiple-of8 padding.
Every closure sees all eight lines once, averaging line losses equally.
Mean latent + mixture expectation, dropout0, rotation/augmentation off.
GMM/pen/KL/OCR/style geometry-objective coefficients0; point-MSE coefficient1.
Encoder/conv_mu/decoder/readout geometry trainable. OCR/style/logvar weights and
pen/sigma/rho FC rows frozen during geometry. Fixed logvar weights do not imply
posterior std is fixed, since shared encoder features still change.

Fresh L-BFGS **per arm**, weights-only initialization (not restored optimizer
history): LR1, history10, max_iter10, max_eval15, strong_wolfe,
tolerance_grad1e-8, tolerance_change1e-12, no clipping or weight decay.
Outer steps are NOT SGD updates; one closure includes eight backwards. Per-arm
loop wall cap900seconds, function timeout3000seconds, T4/cpu4/memory16GB,
no user-code retries. Modal may still restart preempted containers.

### First matched arms:80 outer steps each

All source:
`checkpoints/iam_lbfgs_geometry/20261005-113159/pen_refit/checkpoint.pt`
SHA256 `700f84eda8b523917f0c7f337bc21f75cc6d4075a101edc979fa5432212cd4f8`.

- A point only: delta/tangent coefficient0.
- B point + target Δ matching, weight**0.10770678657952544**, calibrated to20%
  of the aggregate eight-line initial decoder point-loss gradient norm.
- C point + target unit-tangent matching, weight**0.000034808286903642766**,
  same20% calibration. Loss=half squared unit-vector disagreement (1−cos(angle)
  for nondegenerate segments); exclude zero-length target edges, clamp collapsed
  predicted edge length to1% of matching target length for finite gradients.

Same initial source/data/evaluation RNG, fresh optimizer, same budgets. These
first arms did not enforce deterministic CUDA algorithms: do not overinterpret
very small differences or assume a seed alone guarantees bitwise GPU replay.

### Matched refinement:100 outer steps each

Both start from completed A80 weights:
`checkpoints/iam_curve_study/20261005-123232/point/checkpoint.pt`
SHA256 `17a102009fa373dead243c922862859e6d36a9990eebb05f001fe2a9f3008204`.
Fresh L-BFGS in both, same settings above. Deterministic GPU algorithms and
cuDNN deterministic mode enabled, benchmarking off, CUBLAS_WORKSPACE_CONFIG
`:4096:8`. This is a matched refinement, not an optimizer-state continuation.

- A refinement: point only.
- B refinement: point + target Δ weight**0.20471838744633777**, calibrated to50%
  of the *new source's* aggregate decoder gradient norm. Fixed after calibration;
  it is not a large arbitrary smoothing coefficient.

After each completed arm, cache detached mean-latent decoder features and refit
only the three pen rows for2000 AdamW updates: LR0.001, betas0.9/0.99, decay0,
clip5, bounded inverse-sqrt focal gamma2 cap8, real-only supervision. No forced
final EOC. All other state and GMM rows stay bitwise unchanged. Separate pre/post
pen-refit metrics/checkpoints are saved; obsolete geometry optimizer is removed
from head-refitted checkpoints.

## Results

| Mean reconstruction | X RMSE | Y RMSE | Δ relative | Δ² relative | Tangent p90 | Turn p90 | Corner turn p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | .005249 | .010030 |19.59%|36.22%|27.02°|43.62°|73.92°|
| A: point +80 | .001841 | .004050 |8.27%|15.91%|11.43°|19.04°|30.59°|
| B: delta20 +80 | .001651 | .003966 |7.42%|13.81%|10.02°|16.51°|26.01°|
| C: tangent20 +80 | .001851 | .004250 |8.51%|16.28%|10.38°|17.17°|23.14°|
| A: point refinement +100 | .000633 | .001564 |3.16%|6.07%|4.06°|6.50°|10.03°|
| B: delta50 refinement +100 | **.000538** | **.001437** |**2.57%**|**4.70%**|**3.22°**|**5.17°**|**7.92°**|

Final B beats final A on X/Y, Δ/Δ², tangent p90, turn p90, and true-corner turn
p90 **on all eight lines**. Against A, final B reduces mean Δ error18.5%, Δ²
error22.5%, tangent p90≈20.6%, and turn p90≈20.5%. No rounded-away target-corner
tradeoff is evident; true corner p90 also improves≈21.0%.

Named `c`: Δ² relative error72.4%→12.0%, turn p90 58.9°→5.19°.
Named `h`: Δ²54.3%→4.46%, turn p90 30.9°→5.87°.
Small differences remain under magnification, including sparse-point polygonality
in the actual target. Do not promise pixel identity or smoother-than-target curves.

All means and all160 fixed sampled-z predictions have pen F1=1, correct final
EOC, and no false internal EOC. Final sampled-z average X/Y RMSE0.002884/0.002036,
mean per-draw tangent p90 5.63°, turn p90 9.14°. Median/worst-by-XY gallery is
saved; sampled latents are noticeably less precise than means but retain readable
geometry. Geometry is not being judged solely on the best latent sample.

| Completed arm | Closures (eight lines each) | Loop/eval/head/save seconds |
|---|---:|---:|
| point80 |941|246.5|
| delta20 |939|259.1|
| tangent20 |947|247.4|
| point refinement |1191|369.3|
| delta50 refinement |1181|360.3|

These times exclude startup/loading and incomplete/preempted attempts, and are
NOT measured billing. Initial attached delta run canceled after60 and excluded,
then rerun from original source. One refinement attempt preempted after20 and
restarted from source. Completed deterministic prefix matches its first attempt.
Partial artifacts/attempt history retained; no GPU job remains running.

## Artifacts, checks, and next stage

Volume paths (prepend `/mnt/diffink-data/` in your shell):

- Overview: `checkpoints/iam_curve_study/research-summary/index.html`.
- Initial matched gallery: `checkpoints/iam_curve_study/comparison-80/report/index.html`.
- Final gallery/raw crops/all-eight lines/sample gallery:
  `checkpoints/iam_curve_study/comparison-180/report/index.html`.
- Selected: `checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt`,
  SHA256 `52f2417e218b21736d46f1502156b4d31f4cd273c1ef669ec8ede51ee759255d`.
- Matched point control same run's `point/checkpoint.pt`, SHA256
  `00e47c8177475795091d658ef715a735ac51f5d7e69677e22ade3f89642fa5b6`.
- Logs/config/calibration/μ+20z arrays under each arm; full L-BFGS state in
  `checkpoint-geometry.pt`. Later runs preserve exact as-run source snapshots
  and code hashes. CPU preflight/readout/verification/attempt JSONs under study root.

Independent CPU reload verifies finite weights, unchanged OCR/style/logvar and
GMM variance rows, geometry unchanged by pen refit, exact saved pen predictions,
and CPU/GPU coordinate agreement within floating-point tolerance (max1.34e-5
model units, not bitwise equivalence). No restricted IAM/checkpoint images pushed.

`model/losses.py::target_difference_loss` is reusable and opt-in through actual
trainer `target_delta_weight` (default absent/0). Tests cover target corners,
translations, stroke breaks, NaN padding/backward, degenerate segments, wrapped
angle definitions, affine-readout diagnostic, matched-gradient calibration, and
the real training loss path. Existing checkpoint shapes/state keys unchanged.
Final suite: **79 tests pass** in workspace and fork; the refreshed English patch
also passes `git apply --check` against upstream97bc6a3. Default Modal launch was
checked and allocates no GPU without `--train`.

Proceed only with a bounded geometry-preserving **joint integration** on these
eight lines. Restore sampled-latent/pen optimization, corrected tiny KL, and OCR
deliberately, watching all these geometry metrics/galleries. The encoder has
changed, so do not assume the older frozen OCR head's CER0 automatically transfers.
No useful regularized latent distribution, multi-writer style learning, held-out
generalization, full IAM training, or InkDiT result has been established here.

CPU report rebuild: `python -m iam_tools.report_curve_research`.

```sh
# First geometry arms (no GPU allocated without --train):
modal run --detach modal_curve_study.py --train --steps 80 --arms point,delta20
# Additional tangent diagnostic:
modal run --detach modal_curve_study.py --train --steps 80 --arms tangent20
# Matched refinement actually executed:
modal run --detach modal_curve_study.py --train --steps 100 --arms point,delta50 \
  --deterministic \
  --source-path /data/checkpoints/iam_curve_study/20261005-123232/point/checkpoint.pt \
  --source-sha 17a102009fa373dead243c922862859e6d36a9990eebb05f001fe2a9f3008204
```

## Subsequent integration

[Bounded sampled-latent / tiny KL / joint OCR integration](LATENT_INTEGRATION.md)
now also passes on the same eight memorized lines, with the original checkpoint
unchanged. The final marker-free geometry remains faithful, pen/OCR stay exact
on means/all160 draws, and a fixed-reference-std CPU control improves too.
This does not retroactively change the controls or claims of the geometry study
above, and does not establish held-out/multi-writer or production GMM behavior.
