# English reconstruction conditioning investigation (2026-10-06)

## Question

The eight-line reference is faithful but expanded24-line reconstructions add
unnatural corners. The four line-disjoint, same-writer validation lines are poor.
This study does not redefine those artifacts as acceptable or smooth them away.
It tests optimization and conditioning before larger architecture changes.

## Evidence before training

Source: `checkpoints/iam_writer_expansion/20261006-022925/checkpoint-best.pt`,
SHA256 `8d591cfe41b5fa73f16877c34d7b3e62bcf3349d109b114aece0e9c681c4ebee`.
Split/manifest and original eight-line reference stay pinned as documented in
[WRITER_EXPANSION.md](WRITER_EXPANSION.md).24 train lines,4 evaluation-only lines;
writer10174, forms overlap, not an IAM benchmark. No validation gradients,
calibration, caches, optimizer moments or checkpoint choice.

A no-training CPU perturbation appends64 masked EOC positions without changing a
single real coordinate. On the two named training lines, the **early** valid XY
RMS drift (excluding the final100 points) is .128/.138 with GroupNorm, compared
with .00055/.00089 after changing only residual normalization to per-point
channel LayerNorm with copied affine parameters. Held-out examples .160/.203
versus .00184/.00243. This demonstrates a real global padding sensitivity, NOT
proof that normalization caused all current curves or that a layer swap alone
repairs learned weights. Changing normalization initially worsens reconstruction.

GroupNorm(1,C) aggregates channels AND time for B,C,T. Channel-only normalization
aggregates C at each point. It does not make the full encoder/decoder invariant
to convolution boundaries, attention or padding; no such claim is made.

Centering reduces the padding drift somewhat but creates a large initial
coordinate distribution shift. Replicating the final valid XY in padding does
not consistently fix drift. Neither is silently adopted as production policy.

CPU artifact: `checkpoints/iam_conditioning_study/cpu-preflight/padding-sensitivity.json`.

## Controlled T4 experiment

Three arms start from identical expanded source weights, data, seed4042 and
complete-epoch shuffled train order seed42. Physical batch1, accumulation8:

- `control`: original XY + original residual GroupNorm.
- `center`: subtract valid input bounding-box center from real XY, preserve
  aspect/spacing, leave padded XY sentinel-zero, inverse-transform outputs.
- `channel`: original XY; replace48 residual GroupNorm layers with per-point
  channel LayerNorm, preserving affine state-dict keys/values.

All arms use fresh AdamW betas(.9,.99), decay0, clip5,1000 optimizer updates.
LR5e-5 cosine-decays to3e-6. Pen FC rows receive10x proposed Adam displacement
(not gradient scaling), matching the previously tested split-parameter policy.
Same objective:

```
G(mu) + .1 G(sampled_z) + .0016613753687545062 * .5[pen(mu)+pen(z)]
G = valid-point XY MSE + .20471838744633777 * target within-stroke first-difference MSE
```

Pen bounded square-root inverse frequency / cap8 / focal gamma2 / real points.
Dropout, rotation, augmentation, GMM NLL, KL, CTC/style supervision OFF. Frozen OCR
CER is diagnostic only. Coordinates remain absolute XY with .01 scale; center arm
adds only an explicit reversible translation. No spline/resampling/curve smoothing.

This compares conditioning arms **at a shared higher-LR schedule**. Comparing the
control against the previous run also changes optimizer freshness, schedule and
additional update count; that is not a pure LR-only causal ablation.

Evaluate all28 lines at0/250/500/750/1000 with latent mean +20 fixed sampled z per
line; save every checkpoint/evaluation and as-run source. Select only by original
train_score, never validation. Independently reload selected checkpoints on CPU
with their recorded research conditioning and inverse transform, compare saved
GPU means/pens, check protected parameters and source hashes. Research checkpoint
must use the conditioning loader, not the standard production trainer/inference.

Reports include all lines, named c/h crops, true target versus predicted pens,
mean/median/worst sampled draws and immutable eight-line reference. Index Δ/Δ² are
not physical velocity/curvature because RDP point spacing is nonuniform. Tangent
and signed-turn errors are spatial angle differences; authentic corners count.

## Implementation/tests

`iam_tools/conditioning.py`: reversible microbatch translations, edge-padding
probe, channel normalization/replacement. `conditioning_study.py`: bounded matched
training and independent CPU reload. `report_conditioning.py`: all-arm/all-line
marker-free comparison. `modal_conditioning_study.py`: explicit opt-in T4 runner
and separate CPU rendering, no automatic GPU on import.

127 tests pass in root and fork (full-suite rerun after the final patch). Tests cover normalization's independence from
extra points, LayerNorm equivalence, unchanged affine state keys, gradients,
center roundtrip/differences/pens/padding, invalid inputs and edge-padding scope.
The original model architecture defaults and source checkpoint are untouched.

```
venv/bin/modal run modal_conditioning_study.py --train --steps 1000 --modes control,channel
# CPU only, after completion:
venv/bin/modal run modal_conditioning_study.py --report-rel checkpoints/iam_conditioning_study/<run>
```

The completed control/channel arms and failed centered arm are documented below. No generalization/paper-reproduction promotion.

## Additional optimizer diagnostic: full-set joint polish

CPU measurement at the common source, using mean geometry + the same bounded pen
objective, gives expected eight-of24 train-batch gradient noise RMS **.574x** the
full-set gradient RMS (exact finite-population correction, fixed source, no latent
noise). This is nontrivial, not evidence that noise is the sole cause. Saved
`cpu-preflight/gradient-noise.json`; only train lines enter the measurement.

A separate matched-source T4 job tests a deterministic full24 mean-geometry +
mean-pen objective with fresh L-BFGS: LR1,240 outer updates,max_iter10,max_eval15,
history20,strong-Wolfe,tolerance_grad1e-9,tolerance_change1e-13,no clipping,2400s
wall cap. Same delta/pen weights, physical batch1/full-set accumulation. Freeze
OCR/style/logvar head and sigma/rho rows; pen/pi/XY/body train jointly. This is
not the previous geometry-only+head-refit experiment. Evaluate every40 outer
updates, all28 means+20 sampled z; select train-only. Samples are evaluated, not
optimized in this deterministic diagnostic. Optimizer and latent objective differ
from the Adam arms, so this is not an optimizer-only causal ablation.

Volume family `checkpoints/iam_fullset_joint/`, launcher `modal_fullset_joint.py`.
Intermediate/best snapshots keep model weights; final snapshot also keeps the
large L-BFGS optimizer state for provenance/possible continuation. All source and
protected rows remain immutable. Standard trainer fails closed on explicit
research contracts, rather than silently sampling a mean-only research objective.


## Completed conditioning evidence and failure accounting

Control: `checkpoints/iam_conditioning_study/20261006-132430/control/`.
Channel: `checkpoints/iam_conditioning_study/20261006-133613/channel/`.
Channel ran separately after the centered arm aborted the first multi-arm app;
its source/RNG/data/order/settings are matched to control, not an accidental retry.

| Adam1000, mean reconstruction | Train X/Y RMSE | Train turn p90 | Train pen F1 | Held-out X/Y RMSE | Held-out turn p90 |
|---|---|---|---|---|---|
| Expanded source | .009704/.013157 | 54.98° | .9883 | .149929/.060841 | 138.12° |
| Control | .009267/.012505 | 51.64° | .9975 | .147020/.060918 | 135.95° |
| Channel-only norm | .028662/.026813 | 96.19° | .9764 | .056388/.043711 | 123.90° |

Statistics are **macro averages of per-line statistics/quantiles**, not pooled
point/angle quantiles. New normalization improves held-out positions but does
not pass train/held-out curve fidelity; changing normalization invalidates much
of the learned source behavior. It is not adopted in the default architecture.

The centered arm saved its baseline but **failed before its first optimizer
update** with a nonfinite aggregate FP32 gradient norm. Do not interpret that
message as proof of elementwise nonfinite gradients (the as-run code did not
record them). CPU replay measures posterior logvar up to47.87 / std2.485e10 after
centering: a major input/feature distribution shift. We stopped it rather than
turning off error checks. Current reusable code saves `failure.json`, including
elementwise gradient finiteness, before aborting. No centered reconstruction
improvement is claimed. Its baseline/initial weights remain for provenance.
The first launcher also failed on a missing remote Python import before research;
the self-contained launcher fixes that packaging error.

CPU frozen pre-Transformer linear readout is worse: train X/Y .32658/.08890,
turn165.98°, held-out .34852/.10102 and166.64°. All weights bitwise unchanged;
only train equations enter the float64 least-squares fit. This rejects that
specific linear shortcut, not all learned readouts/upsampler architectures.

Artifacts, including as-run CPU scripts:
`checkpoints/iam_conditioning_study/cpu-preflight/{padding-sensitivity.json,gradient-noise.json,posterior-distribution-shift.json,pre-transformer-linear-probe/}`.

## Target-relative segment ablation and operational provenance

Short target segments amplify angular errors. At the expanded source, mean
per-line tangent p90 is80.39° in the shortest within-line length quartile versus
12.23° in the longest. B adds the following **target matching**, not smoothing:

```
R = mean connected-real-edge [ (Δpred - Δtarget) / max(|Δtarget|, target nonzero-edge q25) ]²
joint objective = point MSE + .20471838744633777 * raw target-Δ MSE
                + .0016613753687545062 * bounded pen
                + .00016742507143101805 * R   # B only
```

Nonzero true-stroke edges only; post-stroke jumps/padding excluded before arithmetic.
The floor is per-line q25, so tiny segments cannot get unbounded weights. Both
length and direction match the target, including real corners/hooks. No penalty
on target curvature, no spline or point redistribution. Fixed λ is calibrated
on aggregate **train-only decoder gradients**, R at20% of existing geometry:
geometry norm .01452058, R norm17.345764, gradient cosine .93450. No held-out
search or gradients.

A `20261006-133107` and B `20261006-134641` share source/RNG/data/optimizer;
only B has R. Compare shared saved outer counts, not unmatched final budgets.
A was manually dashboard-stopped after saved200. Its selected weight SHA256 is
`dfa285b5ee8c412efaa2cc7f0b12d135128f14e4ddb8e169a9745e12deaadef1`.
An explicit bounded120-update **fresh-LBFGS weight continuation** starts there;
it is not an exact optimizer resume and not the same-source A/B endpoint.

C `20261006-135755` is a160-update full-set continuation from channel Adam1000,
SHA256 `15b9d46b3cb4db6eb4cc337d282eea990436241d6ec257bf911ab4bc7216214b`.
It inherits/reinstalls the channel norm. This tests further convergence, not a
same-source normalization A/B. All3 initial T4 jobs were intentionally parallel;
no crash-loop retries (`retries=0`). The failed/orphan app was stopped, not restarted.

At shared200: A train X/Y .001607/.002722, turn12.67°; B .001643/.002917,
turn11.79°. Both mean pen F1=1. The auxiliary has a modest angular benefit with
slightly worse point error; optimizer progress is a much larger effect. Held-out
geometry remains bad (X≈.15, Y≈.061); do not promote on training memorization.

A200 mean phase vector RMSE ranges .002854–.003334 (max/min1.168).
This descriptive point-index-mod8 check shows no dramatic common phase error;
it is **not** a statistical/causal exclusion of convolutional upsampling artifacts.

Distinct names for future runs are supported without changing research semantics:

```sh
DIFFINK_EXPERIMENT_NAME=diffink-english-A-polish-from200 venv/bin/modal run modal_fullset_joint.py \
  --train --steps 120 \
  --source-rel checkpoints/iam_fullset_joint/20261006-133107/checkpoint-best.pt \
  --source-sha dfa285b5ee8c412efaa2cc7f0b12d135128f14e4ddb8e169a9745e12deaadef1
```

All selected checkpoints get independent CPU mean/pen reload checks, original
source/protected OCR/style/sigma row comparisons, and frozen logvar-head checks
for L-BFGS. Mean-only training does not train sampled-z robustness;20 fixed draws
are evaluated separately. No KL/CTC/style enablement here. Latest report links
now atomically point to a dated directory rather than concurrently replacing
whole galleries. Latest publication is not a best-model designation.


## Completed joint geometry results

A weight continuation: `checkpoints/iam_fullset_joint/20261006-140945/`, selected120
(fresh optimizer, after the interrupted A200 source). All24 means and all480
sampled draws have pen F1=1, correct final EOC and zero internal false EOC.
No hand-set pens in reconstruction. All24 training-line renders were inspected,
including the named c/h crops; the added shelf/spiky bend is substantially
repaired. Tiny local differences remain at magnified scale; no raw-IAM/pixel-exact
or held-out fidelity claim. RDP's authentic polygonality/corners are retained.

| Stage | Train X/Y RMSE | Mean Δ / Δ² relative | Train turn p90 | Held-out X/Y | Held-out turn p90 |
|---|---|---|---|---|---|
| Expanded source | .009704/.013157 | see source eval | 54.98° | .149929/.060841 | 138.12° |
| A saved200, interrupted | .001607/.002722 | .05219/.09685 | 12.67° | .150619/.061168 | 143.57° |
| B completed240 | .001177/.002246 | .03962/.07223 | 8.29° | .150444/.061179 | 142.62° |
| A200 + fresh120 | **.000760/.001425** | **.02562/.04707** | **6.00°** | .150853/.061445 | 142.99° |
| Channel1000 + full-set160 | .008159/.010356 | .24212/.45688 | 50.17° | .038972/.031951 | 117.56° |

Final A mean turn median/p90/p99 =1.52/6.00/15.88°; true-corner p90=10.50°,
shallow-turn p90=3.71°. Source→polish position RMSE improves92.2% X /89.2% Y.
All-statistics macros are means of per-line statistics, not pooled quantiles.
Sampled-z X/Y train .002970/.002359 (mean-only optimization, not a robustness
training test). Original8 sampled Y .002256 and turn p90 9.93° still exceed the
original integration reference .001776/7.89°, so **the strict original sampled
retention gate is not claimed passed**. Mean geometry and pen fidelity improved;
latent distribution/prior quality remains a separate question.

Budget accounting: B240/2784 closures/2186s; channel160/1854/1668s;
A continuation120/1407/1031s. All3 completed their bounded budgets. Original A
was user-stopped from the dashboard; selected200 and partial status are preserved.
A/B's causal comparison is the shared200 table above, not these unmatched ends.

Selected model SHA256:
- A200: `dfa285b5ee8c412efaa2cc7f0b12d135128f14e4ddb8e169a9745e12deaadef1`.
- B240: `ec15f7ffcfe34bcd7f3abaa461c9313ed7aa9e4276e424857ba2654217ad8b57`.
- Channel160: `5e621ac7b41f0364dd8f10090a7d60e515aa0522c0b2c4d03faaea0719b15d12`.
- A continuation120: `89ec459de6496275a3712c08629daea10d8f4f03311712c77f49e652bca915a3`.

Independent CPU reload maxima: A/B/continuation8.58e-6 XY; channel1.14e-5;
zero pen argmax mismatches. Protected OCR/style/logvar heads, sigma/rho rows and
original source files remain bitwise unchanged for each L-BFGS run. These checks
compare against the actual run input, not against the selected checkpoint itself.

Overview: `checkpoints/iam_fullset_joint/research-summary/index.html`, with all6
stages/all28 lines, immutable8-line reference and named crop comparisons.
Dated `report/index.html` pages include every line's median/worst sampled draw;
all20 draws enter numerical metrics. Reports/checkpoints/as-run code stay on the
Volume, never GitHub. Useful tests and reusable research code are in the fork.

### What this establishes / does not establish

- Current architecture can reproduce24 processed lines faithfully enough that
  the conspicuous expanded c/h artifacts are largely optimization artifacts,
  not a demonstrated capacity limit. Noise measurement and full-set joint
  improvement support optimization as a contributor; multiple knobs differ
  from Adam, so do not assign all improvement causally to batch noise alone.
- Low point RMSE and raw-Δ MSE under-emphasize short-edge directions. Relative
  target matching gives a modest controlled angular benefit, not a magic cure.
- Pen errors, random GMM component choice, latent draws and rendering smoothing
  are not required for the remaining mean-curve distortions. No smoother was used.
- Global padding sensitivity is real; channel normalization reduces it and
  improves held-out positions. But a transplanted norm does not immediately
  restore fidelity, and its separate continuation still looks rough.
- Descriptive mod8 data do not expose a dramatic common phase error. They do not
  statistically rule out local ConvTranspose/phase artifacts.
- Held-out4 lines are **visibly bad**, despite near-faithful train24. No geometry
  generalization pass, joint OCR/KL advancement, full IAM/InkDiT or paper reproduction.

## Next bounded geometry-diversity pilot

The next experiment deliberately broadens reconstruction examples, rather than
adding smoothing or incidentally enabling auxiliary objectives. It uses the
**existing** pinned IAM-overfit HDF5:192 train lines /8 writers,32 line-disjoint
held-out lines /same8 writers. All prior4 held-out lines stay held out. No test
writers/raw-data rebuild/full-IAM training. The original source24 and added168
are reported separately; also original8 and same-writer4 versus other28 validation.

Source is the A continuation120 checkpoint/hash above. Default GroupNorm remains.
Fresh AdamW,1000 updates maximum (1200s wall cap after initial evaluation), LR5e-5
cosine→3e-6, clip5, microbatch1/accum8, seed4042/order42, pen-row displacement10x.
`G(mu)+.1G(z)+lambda_pen*.5[pen(mu)+pen(z)]`, same point/target-Δ objective,
GMM/KL/CTC/style/augmentation/dropout OFF. Pen lambda is freshly calibrated from
aggregate192-train encoder gradients to20% geometry norm, capped1; validation never
enters it. Every250 updates evaluate all224 means +20 fixed z draws. Sampled
reconstruction training is explicit; tiny KL is not enabled. Variance narrowing
alone must not be confused with robust latent reconstruction.

This is an **exploratory pilot**, not a pure data-size causal ablation: source,
optimizer and mean/sample objective differ from the completed L-BFGS runs. The
question is whether additional observed stroke diversity materially improves
held-out geometry while retaining the source24. If it does not, inspect the
encoder/normalization/upsampling more deeply rather than declare training RMSE
sufficient. Family `checkpoints/iam_geometry_diversity/`, guarded launcher:

```sh
venv/bin/modal run modal_geometry_diversity.py --train --steps 1000
```

Research scope is serialized and auto-restored only by research data loaders;
old checkpoints still select writer10174. Retained reference groups survive
checkpoint reloads rather than silently becoming all192. Galleries paginate16
lines and include every line/draw metric, not a hand-picked selection. GPU job
runs once with no retries; the launcher then performs CPU verification/rendering
and publishes an atomic dated report pointer. No additional GPU is needed to render.
### Completed192-line pilot (not promoted)

Run `checkpoints/iam_geometry_diversity/20261006-144049/control/`, selected1000;
checkpoint SHA256 `ce4512fd4d17c643d29ac903a092452f36810db3bea852d0eeff2a658ea7b00b`.
Fresh calibrated pen coefficient0.02099049935353879 (encoder geometry norm
0.911551714, pen norm8.685374260; target fraction20%). All1000 updates completed.

| Group | Mean X/Y RMSE | Turn-error p90 | Mean pen F1 |
|---|---:|---:|---:|
| train192 | .047796 / .033723 |112.66°|.99152|
| retained24 | .037432 / .024426 |91.74°|.99487|
| added168 | .049277 / .035051 |115.65°|.99105|
| held-out32 | .049607 / .036604 |111.67°|.97434|

The larger observed dataset improves held-out position error versus initialization
(.154776/.060373), but does **not** establish faithful curve reconstruction.
Retained24 geometry regresses severely from .000760/.001425 and6.00° turn p90.
Marker-free c/h crops show new sharp distortions; these were visually inspected,
not accepted on aggregate RMSE. No promotion or OCR/KL advancement. This bounded
Adam schedule is not evidence that all192-line training will fail, only that this
pilot does not meet the fidelity/retention gate. Default normalization, optimization
and dataset/latent adaptation remain unresolved contributors.

Independent CPU reload: max XY difference1.1444e-5, zero pen argmax mismatches;
OCR/style heads and sigma/rho rows unchanged, source unchanged. Logvar was explicitly
trainable in this sampled-z pilot. Variance robustness was not inferred from mean
metrics. Report `.../control/report/index.html` includes all224 lines, sampled draws,
retained groups and provenance. The earlier faithful24 checkpoint remains protected.


## Utilization check (2026-10-06, no weight updates)

The original pilot had already finished when utilization was queried. A separate
bounded T4 benchmark used its exact immutable input checkpoint, four allocated
CPU cores, one-line physical batches, eight-microbatch gradient accumulation,
and the same mean/sample/pen computation. No optimizer was constructed or stepped;
all parameters/source hashes were verified unchanged. It uses the earlier pen
scalar .001661375 rather than the192-line pilot calibration .020990499; scalar
weighting differs, though the forward/backward operations are the same. It profiles
eight fixed training lines, not every length or an exact replay of the live run.

| Phase | Wall time | CPU core-equivalents | Sampled GPU busy % |
|---|---:|---:|---:|
| training32 effective batches |13.03s|.985|37.83|
| repeat training32 |12.87s|.987|37.16|
| defer scalar reads to effective-batch end |12.84s|.994|38.22|
| repeat deferred reads |12.76s|.984|37.72|
| inference168 trajectories, no geometry/file I/O |1.21s|1.013|40.20|
| serial geometry metrics336 trajectories |2.58s|.981|4.40|
| full production evaluation168 trajectories |3.32s|.971|29.93|

Thus GPU underutilization with approximately one busy CPU core is measurable,
not merely inferred from a dashboard. Deferred logging yields only roughly1%
training improvement in this check; scalar logging is **not** the main proven
bottleneck. Small physical batches and serial host dispatch/work are plausible
contributors, but nvidia-smi alone cannot distinguish launch overhead from kernel
inefficiency or measure SM occupancy. More assigned cores do not automatically
parallelize Python. Serial NumPy geometry evaluation intentionally leaves the GPU
mostly idle. No data-loader I/O occurs in the measured training loop: batches are
already resident on the GPU.

Next performance work should separately benchmark CPU metric offload/parallelism
and faster kernel dispatch, keeping physical padding and fidelity unchanged. Do
not simply increase physical batch size: this changes GroupNorm statistics and
is not a correctness-neutral throughput optimization. Faster hardware alone also
does not solve serial CPU work. The current benchmark has not yet demonstrated a
safe speedup beyond deferred logging; no throughput fix is represented as proven.

Artifact `checkpoints/iam_performance_profile/20261006-150908/profile.json`, with
all timestamped nvidia-smi samples and stage timings. Coarse sampling (.2s poll,
with driver sampling windows) and sampler overhead limit precision; evaluation
inference has only5 samples, so its percentage is especially approximate. CPU
core-equivalents are process CPU seconds / wall seconds, not allocation percent.
The guarded `modal_inkvae_profile.py --run` requests one T4 for no-update profiling;
without `--run` it allocates no GPU. A first packaging attempt failed on an omitted
launcher import before profiling; the final launcher is self-contained and that
failed app was stopped rather than left crash-looping.

## Implemented throughput fixes (validated, not a fidelity claim)

`fast_geometry.py` evaluates the SAME mean/sample expected-XY, true-stroke raw-Δ,
and bounded focal-pen losses using rectangular masked reductions. Masks/weights
are dynamic graph inputs; no extra padding, mixed precision, batch-size change,
augmentation or loss coefficients are introduced. Unused KL is not evaluated.
A redundant Transformer nested-inference mask check synchronized CUDA BEFORE
PyTorch checked that gradients disable that inference path. The research helper
skips only that inapplicable optimization during gradient-bearing decoder calls,
restoring the original inference setting afterward. Padding masks remain active.

CUDA replay captures forward/backward by EXACT minimal padded length, independent
memory pools, persistent parameter-gradient buffers and external fresh random
noise. No set_to_none=True is allowed after capture. Effective-batch scaling is
copied from immutable base weights (never cumulatively divided/multiplied). Warmup
and capture do not change parameters or consume the caller's RNG stream. Three
spawned CPU workers compute the unchanged spatial metrics while GPU inference
continues; worker exceptions propagate before saving/scoring the evaluation.

Two bounded, no-update T4 validations:
`checkpoints/iam_throughput_validation/20261006-152125/benchmark.json` and
`checkpoints/iam_throughput_validation/20261006-152759/benchmark.json`.
Each uses the pinned faithful24 input checkpoint, eight fixed lines/32 effective
batches, the192-line pilot's pen scalar .02099049935353879, and .1 sampled-geometry
weight. No optimizer updates. Checks cover accumulated gradients, changed input
coordinates and reading live parameters after a temporary restored mutation.
Maximum loss difference7.45e-9, gradient relative L2≤1.36e-7; all model/source
weights unchanged at completion.

| Validation | Eager training | CUDA replay | Sampled GPU busy |
|---|---:|---:|---:|
| first |14.82 /14.72s|4.12 /4.10s|35.7–37.3% →94.9–96.7%|
| repeat |12.10 /12.27s|4.11 /4.10s|37.3–38.8% →94.2–96.8%|

Thus training is3.0–3.6x faster in these measurements, with the same operation's
loss/gradients validated. Capture takes4.8–6.2s for7 lengths; second benchmark
peak reserved memory1.18GiB. GPU busy percentages are coarse sampled estimates,
NOT SM occupancy; eight-line speed is not a universal all-IAM throughput promise.
An initial capture attempt exposed the redundant nested-mask synchronization and
failed before updates; it was fixed, not left retrying.

Evaluation24 lines ×(mean+20 draws): first serial12.64s →parallel10.07/9.87s.
Repeat serial9.96s →parallel5.62/5.82s →same-line draw batching2.21s. Parallel
metrics alone are EXACTLY equal. Same-line posterior batches preserve every
sample's physical length/GroupNorm statistics and use repeated randn_like(mu) to
preserve RNG draw order. They change FP32 kernel roundoff: max XY difference
1.24e-5 across504 trajectories, zero pen argmax changes and identical OCR strings.
This faster optional mode is NOT the default pending broader checkpoint coverage;
CPU workers/CUDA training replay are wired into the guarded control/diversity
runner. Do not batch differently sized lines for this optimization. KL/OCR/style
are still not incidentally enabled. `modal_inkvae_throughput.py --run` reproduces
the benchmark; `modal_geometry_diversity.py --train` uses captured control training.

## Return to geometry: completed192-line full-gradient polish

`checkpoints/iam_geometry_fullset/20261006-152500/`, selected40,468 L-BFGS closures,
991.47s loop/evaluation time, budget completed. Input is the unpromoted192-line
Adam pilot at step1000, SHA ce4512fd…; not the faithful24 geometry checkpoint.
Same point/target-Δ/bounded-pen mean objective, pen scalar .02099049935353879.
Fresh L-BFGS LR1/max_iter10/history20/strong_wolfe, no clipping, physical batch1,
full192-line gradient accumulation, dropout/augmentation/GMM/KL/CTC/style OFF.
Sampled latents are evaluated, not optimized. Held-out32 never enter calibration,
gradients or checkpoint selection. CUDA training replay captures42 lengths in
10.53s/2.97GiB; a live training check observed95–100% GPU busy and ~5GiB total GPU
memory. All224 means and20 draws per line are evaluated at0/20/40; CPU workers3.
This is a geometry experiment with a changed optimizer/full-gradient/mean-only
objective, NOT a performance-only causal ablation against the previous Adam run.

| Group | Input X/Y RMSE | Final X/Y RMSE | Input→final turn p90 | Final mean pen F1 |
|---|---:|---:|---:|---:|
| train192 |.047796/.033723|.028448/.027623|112.66°→99.30°|.99872|
| held-out32 |.049607/.036604|.033393/.030021|111.67°→102.17°|.98329|
| original8 |.035836/.023101|.025817/.023701|85.34°→87.39°|1.00000|

Despite aggregate improvement, the named c/h crops and original-eight gallery
still show added shape distortion. Visually inspected; **NOT promoted** and no
OCR/KL readiness. Better utilization did not magically solve the geometry issue.
Selected SHA bfb1378e5f39a06dd3ac5b361f61f39d66596569fc356c521e435e1eb1cd2b04.
CPU reload max XY difference1.1444e-5, zero pen argmax mismatches; OCR/style/logvar
heads, sigma/rho rows and source checkpoint unchanged. Report
`.../report/index.html`; all224 lines/draw metrics and dated provenance retained.

## CPU initialization/representation control (UNTRAINED)

Rather than conclude the whole architecture cannot generalize geometry from the
polish failure, an independent CPU control configures a near-identity route in the
SAME VAE classes/widths: stride2 convolutions pack8×5 real fields into the first40
latent channels; transposed convolutions unpack them; zero residual-output/
attention/feedforward-output projections leave that route intact. A balanced
constant-feature carrier makes the existing Transformer LayerNorm/readout nearly
linear. All20 GMM means use the same XY readout; pen logits come from the encoded
pen fields. There is no target oracle, sample-ID rule, raw-input skip or smoothing.
This is handcrafted initialization, not evidence of learned geometry or semantic
latents. Preset active posterior std2e-5/unused std1 is NOT learned uncertainty;
GMM uncertainty is unfitted. Subsequent optimization/DiT usefulness are untested.

CPU control `checkpoints/iam_identity_preflight/20261006-cpu/`,224 lines, zero
updates. Mean X/Y RMSE train192 =9.54e-6/5.09e-7; held-out32 =8.95e-6/4.82e-7.
Worst per-line X/Y RMSE across224 =5.11e-5/1.58e-6. Worst per-line turn p90 .00534°;
zero real-point pen mismatches. Named c/h marker-free crops and all224 paired
means are in `report/index.html`; visually indistinguishable at the report scale.
Invalid masked padding is not an EOC-supervision claim. Unit tests verify all8
packing phases, real-state reconstruction, unused-latent-noise independence and
extra-padding invariance. Source script/config/manifest SHA and all sample IDs
are preserved with the report. Reusable opt-in `identity_geometry_probe.py` is
never automatically installed into a trained model or the normal training path.

This establishes that the architecture can carry all these observed trajectories
(and synthetic curves) near-losslessly, not merely memorize one line. In scalar
count it is not an information bottleneck:384 channels/8 positions =48 scalars
per original point versus5 input fields. It rules out an unavoidable8x/ConvTranspose
information-loss explanation, NOT learned upsampling artifacts or poor training
conditioning. It motivates a carefully controlled initialization/protected-geometry
experiment next; it does NOT justify claiming paper reproduction, a trained
English semantic VAE, stochastic GMM generation or automatically enabling OCR/KL.

## Initialized transport: controlled optimizer stability (2026-10-06)

The prior CPU control was untrained. It is NOT sufficient to promote a generative
VAE. This follow-up explicitly tests whether its fidelity survives updates, and
whether meaningful posterior reconstruction can improve. Reinitialize geometry
of the pinned faithful24 checkpoint89ec459… with the SAME analytic polyphase
route, scale512 and active posterior std **.002** (100× the CPU probe's preset).
Dormant residual/attention weights and frozen OCR/style are reused from that source;
this is **not** continuation of its learned geometry. Initial means copy processed
trajectories; therefore excellent held-out copying is not evidence of statistically
learned style/text generalization. No new target points, smoothing or input skip.

All arms:192 train/32 held-out, manifest6d1fafea…, dropout/augmentation0, scale.01,
physical1/accum8, seed4042, same shuffled schedule and fresh posterior noise,
fresh AdamW betas.9/.99/decay0/clip5, Gpoint+.204718×target raw first-difference
matching + .1 sampled geometry + .02099049935353879 bounded focal pen. GMM,
KL, CTC/style OFF. Pen scalar is fixed from the preceding192-line experiment,
NOT claimed recalibrated for near-zero initialized pen loss. Sigma/rho readout
rows and OCR/style weights are protected. No held-out gradients or selection.
CUDA replay/minimal padding and three exact CPU metric workers remain enabled.

A CPU first-update parameter-block intervention corroborates the high-gain issue:
readout-only updated weights produce X/Y~32.33/31.96; body-only~.0663/.0651.
Feedforward-output-only updates produce~20.49/20.27, attention-output-only~11.00/
10.88, final-FC-only~.0123/.00935. These are interventions against identical
initialized weights, NOT an additive attribution (nonlinear blocks interact).
`uniform/first-update-attribution.json` preserves every measured intervention.

Controlled arms `checkpoints/iam_initialization_study/20261006-165824/`:

| Arm | Body LR | Transformer/readout | Posterior LR | Final updates / outcome |
|---|---:|---|---:|---|
| uniform |5e-5|train, LR5e-5|5e-5|10, training-failure gate|
| scaled_readout |5e-5|train, LR5e-5/512|5e-5|10, training-failure gate|
| frozen_readout |5e-5|bitwise fixed|5e-5|100, completed|

The failure gate uses TRAINING losses only: after update10, shuffled mean-geometry
loss >1e-3 stops the arm. Both failures are retained, not silently replaced by
best=0. Ordinary Adam turns near-zero initialized mean error into ~32/32 model
units after ONE update; scaling just the readout LR by512 still yields~.112/.098.
Frozen readout first-update error is~.063/.061, recovering by100. This exposes a
severe optimizer/parameterization sensitivity of THIS handcrafted high-gain
initialization. It is not evidence that Adam is universally broken, nor a complete
causal diagnosis of the older trained model's jaggedness. Scalar gradient scaling
is not an Adam LR fix; the experiment uses real per-group learning rates.

Follow-up `checkpoints/iam_initialization_study/20261006-170142/protected_noise/`:
Transformer/readout BITWISE fixed, encoder/decoder/mu LR **1e-7**, posterior head
LR **1e-3**, same initial weights/objective/data/RNG,200 updates. Two knobs differ
from frozen_readout, so later benefits cannot be assigned solely to either one.
First-update mean error is ~.000117/.000122 rather than .063/.061: posterior-head
updates do not affect mean decoding at that update, supporting the body-LR
explanation for this early stability improvement. Optional same-line21 posterior
batching was enabled only AFTER serial/batched parity on8×21 trajectories:
max XY difference0, zero pen/OCR changes. Evaluation preserves training RNG.

| Final arm / group | Mean X/Y RMSE | Mean per-line turn p90 | Mean pen F1 | Sampled X/Y RMSE |
|---|---:|---:|---:|---:|
| uniform10 / train192 |9.048628/2.029811|158.89°|.14003|failed|
| scaled10 / train192 |.073366/.050367|165.22°|1|failed|
| frozen100 / train192 |.000310/.000336|1.95°|1|see saved evaluation|
| protected200 / train192 |.000005405/.000004392|.0304°|1|.000608/.000609|
| protected200 / held-out32 |.000005142/.000004325|.0302°|1|.000612/.000614|
| protected200 / original8 |.000005913/.000004487|.0290°|1|.000521/.000523|

Protected initial sampled errors are ~.002/.002; final errors improve~69.5%,
not merely mean decoding with preset negligible noise. No updates hit clipping
(0/200); raw gradient norms range .000139–.005817. Worst per-line final mean
X/Y RMSE across224 =1.53e-5/6.09e-6; worst per-line turn p90 .0455°. Final mean
first-/second-difference relative errors train192 =.000113/.000212; genuine target
corner turn p90 .0468°. These are nonuniform INDEX differences, not physical
velocity/curvature. Final ALL224 means +4480 sampled trajectories have perfect
pen boundaries/final EOC and zero false internal EOC. No generic smoothing loss.

Visually inspected named c/h crops, all original8, both32-line held-out pages,
and named median/worst-posterior renders: added spikes/distortions are absent at
report scale; processed target polygonality is intentionally retained. This is
near-lossless reconstruction of IAM/RDP, NOT the original raw pen trace. The old
learned192 model is included as an UNMATCHED engineering comparison, not a causal
ablation: its train mean .028448/.027623 and turn99.30° remain unpromoted.

Selected protected200 file SHA
`5ba90c388946e7a19693b8cb578c0218686ddb0637eebcecb36c5705d68ae948`;
last file SHA `4ea12aed19f48f23e5be98ebb6c0c13608ad56ef52f84ed63f06cd1b0844dca4`.
Independent CPU reload: max XY difference9.54e-6, zero pen changes. On all original8,
extra32/128 EOC padding shifts means by at most X2.10e-6/Y7.16e-7, zero pen changes:
padding sensitivity is tiny in THIS protected transport path, not a general fix
for GroupNorm in the original model. Across224 lines, mean per-line median XY
posterior std .000348; pen-field std stays .002, unused std~.99959/unused mu RMS
~1.69e-7, mean corrected KL diagnostic1.121 (not trained). The scalar bias median
alone stays .002 and would misleadingly hide learned XY uncertainty. Fixed readout/
sigma/rho/OCR/style/source checks pass. Initial dated configs inherit nonoperative
YAML source/output labels: guarded loader constants and provenance.source_rel/SHA
are authoritative; `resolved-source-contract.json` clarifies this. Future runs
explicitly override those stale labels. As-run helper/model source is preserved.

**Readiness:** faithful transport after learning is now demonstrated across the
prepared dataset, including held-out copying. Semantic/generative readiness is
NOT established: frozen OCR CER remains~78–79%, latent means deliberately encode
raw polyphase fields, active noise is tiny and KL is OFF; GMM uncertainty remains
unfitted. Do not claim paper reproduction or launch InkDiT/full IAM on this basis.
The next justified test is a deliberate latent-prior/semantic-supervision trade-off
with geometry retention gates, not another smoothing penalty. The initialized
codec must be compared separately from the original learned-VAE baseline.

Reports: `checkpoints/iam_initialization_study/research-summary/index.html` and
`.../20261006-170142/report/index.html`; failed/control arms
`.../20261006-165824/report/index.html`. All224 paired means and each line's median/
worst posterior render, all20-draw metrics/trajectories, configs/checkpoints and
source provenance are retained. `modal_initialization_study.py --train --steps 100`
runs the three original arms; `--train --steps 200 --arm protected_noise` runs the
follow-up. Import/no flags allocates no GPU. CLI flags have spaces, e.g.
`--steps 200`. CPU-only `--report-rel ...` and `--overview` do not train.

Opt-in helpers never alter the standard production trainer or automatically
install this initializer. Tests cover complete/non-overlapping/frozen optimizer
parameter groups, actual per-group Adam displacement (not gradient scaling), and
invalid-input rejection. Root/fork **139 tests pass**.

The first CPU control-gallery client ended with SIGTERM before publishing its
HTML; GPU training was already complete and saved. Cause is not established.
Recovered with cached checkpoint-SHA-matched CPU checks and all-line MEAN
galleries; failed/control arms retain all20 posterior metrics/trajectory files,
without needlessly rendering hundreds of broken posterior images. The protected
report includes every line's median/worst posterior images. No GPU retraining.

## Initialized codec: matched KL continuations (2026-10-07)

Continued the pinned protected step200 checkpoint, SHA
`5ba90c388946e7a19693b8cb578c0218686ddb0637eebcecb36c5705d68ae948`.
**No geometry reset and no fresh optimizer.** Three sequential T4 arms, each200
further updates (absolute step400), restore the same AdamW moments/group order,
CPU/CUDA RNG and seed42 training schedule after skipping its first200 batches.
Same192 train/32 held-out lines, physical1/accum8, body LR1e-7, posterior LR1e-3,
readout/OCR/style frozen, betas.9/.99, wd0, clip5, dropout/rotation0, scale.01.
Objective unchanged except KL: mean expected-XY point + .2047183874 target index
first-difference matching + .1 sampled geometry + .0209904994 bounded focal pen;
KL0/1e-6/1e-5. GMM/CTC/style OFF. No generic smoothing. Held-out data never changes
training, stopping or checkpoint selection; original8 retention remains separate.
These are initialized engineering transport latents, NOT paper reproduction.

Capture-safe optional KL averages valid latent channel×time ELEMENTS, matching
`VAE.kl_divergence_new`. Invalid padding is selected BEFORE square/exp so NaNs
cannot affect value or gradient. The default KL0 three-term path is preserved.
GPU preflight compares eager core-KL gradients to captured gradients with fixed
noise on all original8: relative gradient L2 ≤1.52e-7, max term difference0,
max core-KL difference1.20e-7. Capturing performs no updates; initial states are
bitwise source-equal. Actual logged training IDs are identical across arms.
Fixed all224-line evaluation at0/100/200 uses mean +20 paired posterior draws;
as-run code, checkpoints, metrics and trajectories are retained.

Final endpoints, **not** whichever best checkpoint hides a regression:

| KL / group | Mean X/Y RMSE | Mean turn p90 | Sampled X/Y RMSE | Sampled turn p90 |
|---|---|---|---|---|
| source200 / train192 | .000005405 / .000004392 | .03036° | .0006082 / .0006085 | 3.459° |
| 0 / train192 | .000006579 / .000005449 | .03769° | .0004447 / .0004452 | 2.349° |
| 1e-6 / train192 | .000010223 / .000005886 | .03470° | .0005810 / .0005801 | 3.388° |
| 1e-5 / train192 | .000012212 / .000016893 | .07333° | .0012752 / .0012763 | 8.347° |
| 0 / held32 | .000006459 / .000005417 | .03766° | .0004482 / .0004495 | 2.319° |
| 1e-6 / held32 | .000010070 / .000005820 | .03464° | .0005859 / .0005857 | 3.353° |
| 1e-5 / held32 | .000011771 / .000016828 | .07059° | .0012811 / .0012821 | 8.271° |

All224 MEANS have perfect pen boundaries in every arm. KL0 also has perfect
boundaries/final EOC on all4480 posterior draws. KL1e-6 final200 has3 false internal
EOCs despite perfect pen-up F1; its selected100 checkpoint has ZERO training EOC
errors and perfect train sampled pens. KL1e-5 has6 failing draws:
one false pen-up plus FIVE false internal EOCs across `e10-546z-02`, `k08-779z-03`,
`n04-290z-03`; all final EOCs remain correct. Pen-up F1 alone would miss five of
these failures. Training minimum sampled F1 .98734; all held-out pens are correct.
No updates hit clipping (0/600). Per-arm loop/evaluation times~98 seconds exclude
capture/preparation. Train-only selected steps: KL0=200, KL1e-6=100, KL1e-5=0.
No automatic model promotion; report always shows final200.

The KL trade-off is real, not a free geometric improvement. Train mean per-line
KL/element source1.12347 → KL0 1.15425 / KL1e-6 1.03952 / KL1e-5 .97661.
At1e-6, ~94.5% of the KL reduction relative to source comes from pen-field uncertainty:
pen contribution .36764→.28835, XY contribution .75582→.75116. XY latent mean RMS
barely changes4.41275→4.41202. Pen std median .002→.00775; XY std median
.000347→.000378; unused median std stays near1. At1e-5, XY std median .001266,
pen .009311; lower prior loss costs noisier local curves and rare false stops.
Median std alone can hide tails; the CPU-only posterior-tail audit records each
failed point's routed pen std and every line's p90/p99/max. This route diagnostic
is NOT an exact propagated decoder variance or a causal intervention.

**Decision:**full-affine1e-6 is clean at100 but fails the strict EOC gate at200,
and is not a geometry winner:
its sampled XY errors are ~31% higher than the matched KL0 continuation, although
slightly lower than source and tiny at normal rendering scale. That does NOT
excuse false internal stops. Do not choose1e-5
just for its lower KL. The unchanged coordinate mean distribution is NOT N(0,I),
and semantic/generation quality is not established. DiT can learn a nonstandard
latent distribution—N(0,I) is not a mandatory readiness condition here. Frozen
OCR transfer CER remains~78–79%; no semantic-supervision success is claimed.
After a clean posterior endpoint is established, refit OCR on the frozen codec first,
then a separately controlled joint update, rather than another smoothing loss.

Artifact root `checkpoints/iam_codec_kl_study/20261007-010118/`:
`report/index.html`, `report/user-regions.png`, all224 marker-free mean comparisons,
original8 median/worst posterior galleries, each arm's `eval-{0,100,200}.json`,
`posterior-{0,100,200}.json`, checkpoints/config/source-code/metrics/provenance,
CPU reload checks, and `posterior-tail-audit.json` plus failed-draw galleries.
Stable link: `checkpoints/iam_codec_kl_study/latest/index.html`.
Final checkpoint hashes:
- KL0: `038e8ffd2b6763bb062a8e063c690665a4ce108ee3f5fcc3b0642b441d06c0fa`
- KL1e-6: `f367ab2bbe8158427c4fe0a0e84124a2e9f3a2fb41c6d10063e3d0b7663b31d2`
- KL1e-5: `3deddfbf55ecf59a567fc729c947aa2993d305305fc1584f35e4b2e64fac77e0`

`modal_codec_kl_study.py --train --steps 200` explicitly allocates one T4 for all
three sequential arms. No flags/import allocates no GPU. `--report-rel ...` and
`--tails-rel ...` are CPU-only. An initial launcher dependency-import failure
occurred before any training; that app was stopped, the launcher made standalone,
and the successful bounded run completed. No failed/crash-looping app is left.
Tests cover KL values/gradients/NaN padding/sequence normalization, unchanged
optional/default losses, exact restored Adam moments and next displacement,
reordered-group rejection, continued schedule, posterior statistic accounting/RNG,
polyphase point-field indexing, strict EOC eligibility independent of F1, and
frozen posterior rows under restored Adam momentum. Root/fork147 tests pass.

### Follow-up: freeze feature-dependent pen variance, not pen variance learning

The tail audit made the failure local and interpretable: all9 failed points in
the two full-affine KL arms occur near the end of very wide lines, target X
18.31–19.36 and92.6–98.7% of point indices. Full KL1e-6 pen std median .00775
hides a MAXIMUM .37870; full1e-5 maximum .42545. This is a posterior-noise/stop
problem, NOT an explanation of the earlier mean-trajectory underfitting/jaggedness.
Routed pen std at failed points is .08–.41, despite nearly exact coordinate means.
This supports a high-leverage absolute-X logvar hypothesis; it is not alone proof
of a propagated-output variance formula. No generic smoothing or sigma cap added.

Controlled T4 follow-up `20261007-012331/pen_bias_kl1e-6`,200 further updates from
the SAME protected200 source/Adam/data/RNG/schedule as full-affine KL1e-6. The only
policy change is freezing the24 pen-field FEATURE-WEIGHT rows of `conv_logvar`
(first40 channels whose channel%5≥2). Their biases, XY/unused posterior weights,
and original body optimizer remain trainable. Gradient masks alone would not
freeze restored Adam momentum: those rows' moments are explicitly zeroed as well.
Weight decay0; final protected rows must be bitwise source-equal. This is a
checkpoint-compatible optimization policy, NOT a decoder architecture change,
blanket smoothing, forced EOC rule or removal of KL. As-run settings record this
intentional subset moment reset; all other optimizer state stays restored.

Final200 training192 mean X/Y RMSE .000009131/.000005911, turn p90 .06111°;
held32 .000009130/.000005930, turn .06013°. Sampled train X/Y .0005792/.0005800,
turn3.3885°; held .0005842/.0005856, turn3.3525°. These are essentially unchanged
from full-affine KL1e-6 sampling geometry, but **all224 means +4480 posterior draws
have perfect pen boundaries/final EOC and ZERO internal EOCs**. Thus feature-
dependent pen variance contributes to the newly observed rare false stops.
Pen std median now .00202274 (vs.00775242 full-affine); corrected KL1.11813
(vs1.03952 full-affine) demonstrates why the apparently better prior loss was
misleading. XY mean RMS4.41269, no meaningful coordinate prior whitening.
Train-only selected continuation100. Selected SHA
`9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`;
final SHA `f85d7b97144eba1e0041d157666d6ba4c23cfbca6b0c66ab57c162f111d00dfd`.

Checkpoint eligibility now explicitly requires training mean/sampled pen F1=1,
all final EOCs correct and zero internal EOCs, independently of pen-up F1 and
before train-score comparison. Held-out metrics cannot select a checkpoint.
The earlier as-run selection used train_score alone; its full1e-6 selected100
happens to be clean, but final200 is not. That distinction remains visible.

Run `modal_codec_kl_study.py --train --steps 200 --protected-pen` to reproduce the
single follow-up; original command still runs the three full-affine arms. These
policies are not silently installed in production training. Report includes all
224 means, named c/h, original8 posterior median/worst, and ALL former full1e-6
failed draws replayed with IDENTICAL evaluation noise. Numerical paired data and
source configs remain in both immutable dated roots. Refit OCR on the frozen
codec next; do not infer semantic/text-generation readiness from this success.

Independent final CPU reload checks all224 lines: maximum CPU/GPU XY difference
KL0=1.34e-5, full1e-6=1.91e-5, full1e-5=1.53e-5, protected-pen=1.15e-5; zero pen
changes. Readout/OCR/style/source and protected pen variance-weight rows are
bitwise unchanged where required. Protected max routed pen std .00202275, versus
.37870 in the matched full-affine1e-6 control. All three formerly failed1e-6 draws
are repaired under the identical evaluation noise; no new failed draws observed.
Visually inspected original8 all-mean panel, named c/h crops, both held-out32
pages, named worst-posterior renders and paired false-EOC crops. At normal report
scale the protected output is visually faithful; extreme-zoom IAM/RDP polygonality
is retained, not artificially smoothed. Mean local index first-/second-difference
and spatial-angle metrics remain in every evaluation/report, not RMSE alone.
Reconstruction/pen gate passes for this protected engineering codec. Ready for a
FROZEN-codec OCR refit, not unconstrained joint/readout training or a DiT claim.

The protected report is `checkpoints/iam_codec_kl_study/20261007-012331/report/index.html`,
with the original full-affine comparison linked separately at
`.../20261007-010118/report/index.html`; latest points to the protected report.
Both are downloaded under local `data/checkpoints/iam_codec_kl_study/` as well as
on the persistent Volume. Follow-up captured gradient relative L2 9.55e-8 and
core-KL difference1.20e-7. Moment reset maxima are recorded (~2.01e-15/7.07e-29);
freezing restored moments is still required for EXACT row preservation. All T4
runs completed and no GPU/research container is left running.

Additional CPU/log check: this initializer sets correct pen logits near20.
Float32 cross-entropy/focal loss at `[20,0,0]` is numerically0, with zero gradient.
The KL0 continuation logs pen loss0 on all200 updates; both full-affine KL arms
also have pen loss0 for their first20 updates, then nonzero losses on77/144 later
updates. Thus a saturated pen head initially supplies no variance counter-pressure
while KL raises pen uncertainty. This is specific evidence about the engineered
readout, not a general claim that focal loss is broken. A lower pen-logit strength
or calibrated probabilistic pen objective is a plausible ALTERNATIVE, not tested
here. The feature-weight freeze is a successful controlled safety policy, not
claimed to be the only correct production fix.

## Frozen-codec OCR refit and posterior-noise audit — 2026-10-07

**Geometry remains passed; OCR generalization is NOT passed.** Work continues on
`english-iam`; geometry source is protected codec selected100
`checkpoints/iam_codec_kl_study/20261007-012331/pen_bias_kl1e-6/checkpoint-best.pt`,
SHA `9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`.
All192 train/32 seen-writer held-out lines use manifest
`6d1fafea62af6c5ddae48983bb9699d93016e4af7a11df2f8dd1743128c34c8a`.
Forms overlap; this is not a writer/form-independent IAM benchmark.

Production correctness patch: `ChineseHandwritingOCR.forward` now optionally
accepts Boolean `[B,T]` **padding** masks (True=ignored), removes padded features
before projection (including NaNs), and masks Transformer attention.
`get_ocr_loss` accepts legacy binary float or Boolean **valid** masks, checks
right-padding, and passes the inverse to attention. Exact repeated-label CTC
feasibility is unchanged, including the actual VAE training delegation. Optional
`mask=None` now means all-real inputs. No checkpoint tensor/architecture/default
blank-bias change. API behavior checked against official PyTorch sources/docs:
[attention masks](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/activation.py),
[CTCLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html).

### Controlled T4 work

Only OCR parameters train. Encoder/decoder/readout/posterior/style remain frozen.
Raw lines are encoded individually at their original minimal padding; ONLY cached
OCR latent inputs are length-bucketed into physical batches16 with attention
masks. This avoids introducing temporal-GroupNorm raw-padding effects. All224
samples are exact-CTC-feasible; these runs use `zero_infinity=False` to expose
rather than silently suppress unexpected invalid losses. OCR dropout.1 retained;
trajectory dropout/rotation0, model scale.01. Mean+20 fixed posterior draws/line
are evaluated at0/250/500 (also750/1000 for the first refit). Paired evaluation
preserves training RNG. No KL/GMM/style/geometry optimization, no generic smoothing.

- First: `iam_frozen_ocr_study/20261007-014901`:1000 updates, inherited head weights,
  blank bias0, fresh AdamW5e-4 →1e-4 at750, betas.9/.99, weight decay1e-4, clip5,
  cached means. Train-only CER/CTC checkpoint selection picks1000.
  Selected SHA `39a10dbb266ce9476e01f41c807e9d46d997964ad664de33d033c4a0b11cd02e`.
- Mean-only control: `.../20261007-015617`:500 continuation updates (total1500)
  from that exact source, restored head Adam/RNG, LR1e-4 throughout.
  Selected SHA `8738dc9b0b0615f60e6f1b0e740205ed31408561b456611619dbf19f56b154f6`.
- Sample-aware: `.../20261007-015548`:same source/Adam/RNG/schedule/LR,500 updates;
  CTC=.5 mean+.5 sampled-z. The control consumes the same noise RNG and performs
  two mean forwards, ensuring matching dropout draws. Selected SHA
  `4405923eec8bb29c918b36051ae44d98f1f35c9b9b1e0969b9b4649710c301b4`.

Pairing audit confirms initial OCR evaluation, all500 batch-ID sequences and LRs
are identical. Held-out data never enters gradients/calibration/selection.

| endpoint | train mean CER / exact lines | train sampled CER | held-out mean / sampled CER |
|---|---:|---:|---:|
| first source, bias0 |78.3323% /0|81.7783%|79.1331% /83.9919%|
| first mean fit1000 |0.2685% /180|59.9084%|82.1573% /88.8810%|
| paired mean control+500 |0% /192|61.2445%|84.1734% /90.8216%|
| paired mean+sample+500 |0.0948% /187|3.7547%|82.6613% /82.7571%|

Repeat/nonrepeat breakdown is in every report (93 repeated-character training
lines). Source all-non-OCR state tensors are bitwise preserved in EVERY run;
all4704 saved before/after mean+posterior trajectories (224×21) are bitwise equal.
Pen boundaries/EOCs remain perfect. Source mean train X/Y RMSE
7.26895e-6/5.63885e-6 and turn p90 .0394448° therefore do not change. Named c/h,
original8, all192 training and32 held-out mean marker-free pages are preserved.
We visually checked named regions/original8/held-out gallery: no added spikes,
no geometrical degradation. Target IAM/RDP polygonality is deliberately retained.

### A different noise failure — not jaggedness

The initialized transport uses40 routed fields and344 nearly inactive channels.
Their training mean maxabs1.38e-6/RMS1.74e-7 is tiny, while posterior std≈.9999125.
Their means are **not analytically zero** after body learning. Inherited OCR
projection weights see little mean-training signal to reject this noise.

A CPU no-training paired ablation zeros ONLY OCR `input_proj.weight[:,40:]`.
The codec, latent distribution, all geometry/pen parameters and active OCR columns
stay unchanged. Every prepared mean transcript stayed unchanged (measured, NOT an
assumption about arbitrary inputs). Mean-fit1000 training sampled CER falls
59.8958% →0.2582%. The polished mean-only control falls61.1039% →**0%**, with
**3840/3840** training posterior draws exact, alongside192/192 exact means.
Sample-aware head falls3.6497% →0.0963%. Held-out remains poor (≈82–84%).
This isolates nuisance latent projection, NOT a failure of sampled geometry and
NOT a recommendation to prune learned semantic channels generally. Diagnostic
weights saved separately as `checkpoint-inactive-projection.pt`; original heads
and optimizers remain unchanged. No automatic promotion or generation claim.

Independent CPU reload: source/frozen state checks pass for all3. First and
mean-control mean transcripts agree with GPU for all224; sample-aware has one
held-out `k04-265z-01` near-tied mean argmax discrepancy, recorded with its CPU
logit margin. All training mean transcripts/CER agree. CPU/CUDA posterior RNGs
are different; only **within-device** before/after ablations are paired.

### Reports, reproducibility and next gate

Volume paths (local equivalents under `data/checkpoints/`):
- `checkpoints/iam_frozen_ocr_study/research-summary/index.html` — combined evidence,
  exact hashes/configs/pairing checks and complete transcript/report links.
- Each dated study's `report/index.html`, `result.json`, `metrics.jsonl`,
  `ocr-*.json`, `cpu-reload-check.json`, `inactive-projection-ablation.json`,
  checkpoints, original source-code snapshots and geometry arrays.
- `latest/index.html` points to the combined summary, not a claim of best model.

First as-run configs inherited prior-codec textual schedule/optimizer metadata.
Original configs/checkpoints are preserved; `metadata-clarification.json` records
actual bucket scheduling and fresh/restored OCR optimizer behavior, established
from source snapshots and per-update sample logs. Current runner explicitly sets
those fields. A startup runner restart occurred before control training; partial
initialization output is not used in the completed-study pairing or summary.

Use `venv/bin/modal run modal_frozen_ocr_study.py --train --steps 1000`; no `--train`
allocates no GPU. Continuations require pinned `--source` and `--sha`; `--sampled`
selects the paired half-mean/half-sampled objective on a fitted source. CPU report
`--report-rel checkpoints/iam_frozen_ocr_study/research-summary` allocates no GPU.
Bounded3000 updates, wall900s, T4/cpu4, retries0/maxcontainers1 per app.

160 root/fork tests pass (13 new tests): padded batch logits/CTC parity, padded NaN
value/gradient safety, legacy binary float masks, prefix/type checks, cache/data
split isolation, restored head moments, identical paired RNG consumption, and
explicit exact-versus-approximate inactive-channel ablations. GPU source parity
max logit difference5.72e-6, no argmax flips, CTC loss difference0.

Do not keep spending updates merely memorizing192 lines. Head fitting is proven,
and its posterior noise mechanism is isolated, but **unseen-text reading is the
next unresolved gate**. Test controlled local-context/translation invariance or
larger TRAIN-only OCR supervision with frozen faithful geometry; report held-out
reading without selecting on it. Do not infer semantic/generative readiness,
paper reproduction, or successful joint OCR/KL training from a transport codec
plus a memorizing OCR head. Keep readout/geometry gates protected.

## Frozen OCR context/coordinate controls (2026-10-07)

Completed `iam_ocr_context_study/20261007-031059`: **four sequential T4 arms**,
1000 updates each, ~36–38s actual training/evaluation per arm. One GPU app;
no retries/crash loop. Frozen codec source9c53f68 (same pinned SHA as above),
192 training/32 held-out lines, unchanged manifest. Every original codec tensor,
**including its old OCR**, remains bitwise source-identical. New heads are
standalone research artifacts; no geometry, pen, posterior, KL/style/readout
updates, and no new reconstruction-array comparison is claimed in this study.
The previously established marker-free geometry fidelity remains unchanged.

Fresh identical OCR weights/blankbias0, same seed42 bucketed batch16 schedule,
AdamW5e-4→1e-4 at750, betas.9/.99, decay1e-4, clip5, dropout.1. Cached latents
are encoded physically one minimally padded line at a time. TRAIN-only XY
moments, no held-out calibration. Checkpoints selected only by TRAIN mean CER,
then CTC. Same random initialization tensors and sample/LR logs verified;
different attention kernels need not generate identical dropout masks.

| OCR input/context | Train mean CER (exact/192) | Train sampled CER | Held-out mean/sample CER |
|---|---:|---:|---:|
| Absolute XY, global attention | .1421% (185) | .1437% | 83.0645% /83.0494% |
| Train-axis standardized XY, global | 0% (192) | 0% | 80.2419% /80.2167% |
| Relative-X standardized, global | 0% (192) | 0% | **71.8750% /71.8196%** |
| Relative-X standardized, radius4 | 0% (192) | 0% | 72.9839% /72.9940% |

All selected checkpoints are update1000. Sampled metrics use20 fixed GPU draws
per line (3840 train/640 held-out). Every arm reads **0/32** held-out lines exactly.
Translation-invariant conditioning helps11.19 percentage points against the
matched raw baseline, but does **not solve generalization**. Restricting attention
does not improve further. One seed/small exploratory pool, forms overlap and
writers seen; not an IAM writer/form-independent benchmark. This does not prove
absolute X is the sole cause: representation/conditioning also changes.

Adapter contract is specific to the initialized polyphase40 codec: verify all224
packed payloads against raw XY/pen states first; preserve the first line-final
EOC and exclude synthetic tail phases; zero344 unused channels ONLY at OCR input
in every arm. Relative-X retains within-block phase offsets and between-block
first-X displacement (invertible up to horizontal origin), not physical velocity.
Y remains absolute before standardization. Three radius4 layers see±12 feature
blocks; X features additionally reference the preceding block. Padded queries
have a finite dummy key; valid queries never get a global key0 escape. Minimal
core patch exposes optional `attention_mask`; default behavior/weights unchanged.

No-training OOD ablations of the prior memorized head8738dc9:
train CER native0%, no-Y18.1459%, no-pen33.9861%, block-centroid XY37.6184%,
synthetic width/length-only77.3531%, local-radius4 attention78.0322%.
These establish dependence, **not sole causality**. Width alone does not preserve
reading, but whole-line context dependence is large. Held-out remains poor in all.
Missing training labels are not the explanation: only one held-out character
(`F`) out of992 is unseen; no held-out transcript exactly matches training.
There are only6332 train character occurrences. Next strong test: a larger
TRAIN-only OCR pool with protected geometry and explicit split/form provenance,
not another memorization continuation or unrestricted joint VAE optimization.

CPU reload of actual saved heads/features: all224 mean transcripts agree with GPU
in every arm; posterior CPU/CUDA RNGs are not paired. Random initializer tensors
match when saved nonrandom buffers are restored. Host/build-generated sinusoidal
positional buffers differ by max3.05171e-5 on the remote CPU reload
(local CPU build max6.10352e-5, over1000 positions), so regenerate-and-
hash assertions are not portable. Reload uses saved tables exactly; discrepancies
are recorded, not silently ignored. All four state/source checks pass.

Volume report: `checkpoints/iam_ocr_context_study/20261007-031059/report/index.html`
(local `data/` mirror). Report includes learning curves, all224 selected
transcripts, all32 marker-free held-out handwriting/prediction panels, CPU reload,
training text coverage, exact configs/source snapshots, initialization/schedule
pairing and checkpoint hashes. Standalone `head-best.pt` / `head-last.pt` in each
arm; never load these into VAE without their research feature contract.
Selected SHA256:
- global_raw: `e9f0cf036ba765d8acb2fa717073790942f37168133ce24230337915db661351`
- global_scaled: `9aafd35d79c59065353fe977cdb8464d44e9977bbf3946c3846f91dc0f16cadd`
- relative_scaled: `32892410a78a3f7d417dcb945d7308ade4f9dd78686d63ac32c31b8a321e0da7`
- relative_local4: `2655078211bbba6c4b5e9d8019839ee2e007ee9f8707243b586c1bc489aef8ba`

`modal run modal_ocr_context_study.py --train --steps 1000` explicitly opts into
T4; noflag allocates no GPU. Four bounded sequential arms,450s maximum per arm,
2400s app timeout, maxcontainers1/retries0. `--report-rel <dated-directory>` is
CPU-only, `--annotate-only` updates interpretation without reloading/evaluating.
170 root/fork unit tests pass:10 new checks cover transport mapping, translation
invariance, tail/unused NaN isolation, train-only statistics, finite local masks
and backward/padding parity, restricted receptive field, seed/RNG preservation,
legacy default attention parity and text coverage. Geometry gate stays protected;
no research head promoted and no InkDiT or joint OCR/KL/style launched.

## Larger prompt-guarded OCR pool (2026-10-07)

Completed `iam_ocr_pool_study/20261007-033445`. This tests data amount while
**every original codec tensor remains bitwise unchanged**, including its old
OCR head. Fresh standalone research heads only; no shared encoder/decoder,
readout, pen, posterior, geometry, KL/style optimization or InkDiT job.

### Split and experiment controls

New independent generated directory `diffink/iam_ocr_pool` (original overfit,
raw IAM and canonical directories untouched):
- large TRAIN2048 /62,334 target characters /186 writers;
- nested small TRAIN192 /5911 characters /the SAME186 writers;
- DEV128 /3850 characters /five previously reserved writers,24–27 lines each;
- original report-only32 /992 characters /eight seen writers, four each.

TRAIN excludes all DEV/report **prompt families**, removing IAM writer-version
suffixes (`a01-000u/w/z` →`a01-000`), as well as normalized exact transcripts.
DEV/report prompt families and normalized transcripts are disjoint too. DEV
writers are wholly excluded from TRAIN;25 test writers excluded entirely. Fixed
81-character vocabulary inherited from the broad prior TRAIN inventory excluding
test/DEV writers; no evaluation labels expand it. Selection/conversion seeded42,
round-robin balanced by writer. Height100/RDP.5/input scale.01 and200–2000 point
loader/exact-CTC filters unchanged. Raw XML/transcript and point fingerprints,
rejections, IDs, writer/prompt/text checks, HDF5 and vocabulary hashes preserved.
Atomic replacement safeguards preserve old complete output on build failure.

Dataset manifest SHA256:
`122a428ad0e549aeec9b9f4b67028f48ac6470123234505cd25ac191b041bd0e`
HDF5 SHA256:
`69e937e22567f94b04b5d954b1d536e3f19b2ccde9b08a7e21ef0dea64c94f1d`

Both arms: same fresh seed42 OCR tensor initialization, blankbias0, dropout.1,
relative-X/global attention, **identical moments calibrated ONLY on nested small
TRAIN192**, cached means, masked head batch16; raw codec batch1/minimal padding.
AdamW5e-4→1e-4 at4500, betas.9/.99, wd1e-4, clip5.6000 updates each (~96k
presentations), eval every1000. Same initial tensors/LR/update budgets verified;
training sample/dropout realizations differ because datasets differ. GPU
posterior evaluation noise is paired across arms for each common line; CPU and
GPU noise streams are NOT paired. DEV-only CER then CTC selects checkpoints;
original32 never selects/calibrates anything. Both select step6000.

This stricter small control is **not** the earlier8-writer192-line corpus. Compare
these two matched arms, not raw old percentages. Frozen geometry pretraining
still used the original192 with prompt overlap against original32, so this is
OCR supervision isolation, **not** a fully independent pretrained-representation
IAM benchmark, paper reproduction or proof of semantic/generative latents.

### Results: data amount matters strongly

| Selected arm | TRAIN mean CER /exact | DEV mean/posterior CER /exact means | report32 mean/posterior CER /exact means |
|---|---:|---:|---:|
| small192 | 0% /192/192 |75.6364% /75.6247% /0/128 |74.4960% /74.4304% /0/32 |
| large2048 | **.019251% /2036/2048** |**28.2857% /28.2545% /2/128** |**25.1008% /25.1563% /0/32** |

Common32 TRAIN probes: mean AND all640 posterior draws CER0 in both. Posterior
OCR evaluation covers128 DEV +32 report +32 common TRAIN =192 lines,20 draws
per line; **does not pretend every2048 TRAIN line has20 posterior draws**.
All2208 pool lines pass codec mean checks; the large head evaluates all2208
means and the small head evaluates its352 TRAIN/DEV/report means. All CPU/GPU
mean predictions agree:
352 unique evaluated lines for small and2208 for large. Source buffers restored
from checkpoints rather than regenerated; max host positional-table difference
3.05171e-5, random weights match once saved nonrandom buffers restored.

Actual T4 training/evaluation128.760s (small) /131.528s (large), sequential in
one app, no parallel GPU jobs. Large report32 happened to score24.2944% at5000
but **we did NOT choose it**: independent DEV preferred6000. Remaining25–28%
CER is still far too high for a dependable recognizer; nearly perfect TRAIN fit
is not generalization. Data starvation/line memorization is a major contributor,
not a lack of faithful curve capacity or missing labels. Single seed remains a
limitation; no assertion that more data will remove every residual error.

### Geometry remained protected on a much wider corpus

Before ANY OCR update, all2208 real sequences passed all-point packed-coordinate,
real-phase, actual decoded-mean curve and exact pen/EOC checks:
- mean per-line X RMSE **7.24164e-6**, Y RMSE **5.69862e-6**;
- mean per-line geometric turn-angle p90 **.0409361°**;
- maximum packed-coordinate difference7.91550e-5;
- every mean pen boundary/final EOC exact; no internal false EOC.

CPU reload repeats the whole corpus codec gate and selected-head checks. Source
codec9c53f68 is unchanged after each arm; a single shared pre-training
`geometry-source.h5` stores all decoded means. We do not claim newly saved
before/after arrays are bitwise compared in this study. Index differences are
not physical velocity/curvature; angle definitions match the common geometry
metrics. No smoothing or target-corner removal. Visually inspected all32 original
report target/reconstruction panels and representative DEV pages1/6/11/16;
curve/pen fidelity retained, larger OCR much more interpretable but has deletions,
letter/case/spacing errors. Gallery includes ALL128 DEV lines, not just examples.

One initial GPU preflight failed **before any optimizer update** because its
new diagnostic supplied the compressed latent mask to the point-resolution
decoder. Fixed to raw point mask and added an identity-codec regression that
would catch it. Failed directory033152 is preserved with `failed-preflight.json`
and excluded from reports. The completed experiment starts fresh; no hidden
optimizer restart. Future runner snapshots/records even preflight failures.

### Artifacts and next gate

`checkpoints/iam_ocr_pool_study/20261007-033445/report/index.html` on Volume,
local equivalent under `data/`: all DEV/report marker-free comparisons,
transcript tables, learning curves, exact configs and hashes, mean/20-draw
metrics, CPU reload, pairing/split/geometry audits. Full TRAIN mean transcripts
are in each arm's `ocr-*.json`. `latest/index.html` points to this report.

Selected standalone `head-best.pt` SHA256:
- small192 `9c7e0b121ad34a949c4551afa05c50a610297d66aeefbd79e2a54803e77c671f`
- large2048 `073705cc96ae29b3e991fee8820eecca291d20ae57f1b1d597f114c97de5f925`

Build CPU-only: `venv/bin/python -m iam_tools.ocr_pool`. Upload the generated
pool using Modal Volume `batch_upload().put_directory()` (see
[official Volume reference](https://modal.com/docs/reference/modal.Volume)),
without modifying raw/overfit data. Explicit launch:
`venv/bin/modal run modal_ocr_pool_study.py --train --pool-sha <manifest SHA> --steps 6000`.
No flag/hash allocates no GPU. Bounded1000–8000 updates, wall900s per arm,
T4/cpu4, whole function timeout2400s, retries0/maxcontainers1. CPU-only
`--report-rel checkpoints/iam_ocr_pool_study/20261007-033445`.

182 root/fork tests pass:12 new tests cover prompt variants/reserved writers/
transcript guards, balanced nesting, unchanged released collation, per-line CTC
normalization parity, local geometric-angle definitions, partial-posterior
report honesty, point-versus-latent decoder-mask resolution and safe output
paths/quotas and immutable dataset-version fallback. No research head is installed into standard VAE training without
its polyphase feature contract. Next: enlarge TRAIN-only supervision further
with pinned calibration/selection, before exposing faithful geometry to a still
25%-CER OCR teacher. Do not spend more updates merely memorizing2048 or enable
unrestricted joint OCR/KL/style/InkDiT on the strength of TRAIN accuracy alone.

The exact19MB prepared pool is archived under `diffink/iam_ocr_pool_versions/`
plus the manifest SHA above (local equivalent under `data/`). Future hash-pinned
reloads prefer this immutable copy, so enlarging the current alias cannot break
old-study reproducibility. It was archived AFTER the completed experiment; as-run
configs/source snapshots correctly retain the original unversioned pool path.
CPU report `--annotate-only` refreshes interpretation without another model reload.

## Paired trained-head continuation: 2048→8192 supervision (2026-10-07)

The next control branches from selected `large2048/head-best.pt` at step6000,
not freshly initialized heads. Source head:
`checkpoints/iam_ocr_pool_study/20261007-033445/large2048/head-best.pt`,
SHA `073705cc96ae29b3e991fee8820eecca291d20ae57f1b1d597f114c97de5f925`.
The codec source/SHA and relative-scaled OCR feature contract stay unchanged.

Expanded pool manifest SHA
`d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9`;
HDF5 SHA `7c33d8e56865e28947c10dc086f531903ff7c234ac27c5312f8d68e0d9c0fe86`.
Immutable local/Volume archive:
`diffink/iam_ocr_pool_versions/<manifest SHA>` (local prefix `data/`).
8192 TRAIN lines from the same186 writers,128 unchanged DEV and32 unchanged
report lines (8352 unique total). The parent2048 is an exact ordered TRAIN
prefix. Every original parent record/source-point fingerprint, calibration192
ID/order, DEV/report ID/order and81-char vocabulary is retained.698 candidate
rejections include reserved DEV selection failures; filters remain200–2000
points/exact CTC feasibility/strict timestamps/RDP0.5. No new normalization,
pen semantics or representation change.

Both arms restore exact head parameters/buffers, Adam moments/counters, CPU and
CUDA RNG from the same checkpoint. Inherited LR1e-4 is held constant; AdamW
betas.9/.99, decay1e-4, clip5, OCR dropout.1, blankbias0, global attention,
cached means/masked batch16; encoder physical batch1. Feature moments are
inherited verbatim and independently checked from the pinned calibration192,
not refit on8192. Both bucket iterators restart seed43: this is NOT restoration
of the parent's data-iterator position. Different samples/lengths imply
unpaired dropout after the shared initial RNG. Both get6000 additional updates
(max total12000), evaluate every1000; DEV CER then CTC selects checkpoints,
original32 reporting-only. Per-arm900s wall guard, T4/cpu4, retries0,
maxcontainers1, function2400s. Both arms run sequentially in one T4 container.

Whole codec, including original OCR, is frozen. All8352 means are individually
encoded/decoded and gated for transport/XY/geometric turns/pen boundaries before
any OCR update.20 paired GPU posterior draws cover DEV128/report32/common32,
not all8192 training lines. CPU reporter independently reloads all192 evaluation/
probe lines; full TRAIN metrics are explicitly as-run GPU, not8192 CPU reloads.
Gallery changes OCR captions only; target and frozen trajectory are unchanged.
This remains a transport-codec OCR study, not a paper reproduction, novel-text
handwriting generator, fully independent IAM benchmark or permission to deform
faithful geometry for a partially generalized reader.

CPU build (keeps original pool immutable):
```python
from iam_tools.ocr_pool import build
build(out='data/diffink/iam_ocr_pool8192',train_size=8192,
      parent_pool='data/diffink/iam_ocr_pool_versions/122a428ad0e549aeec9b9f4b67028f48ac6470123234505cd25ac191b041bd0e')
```
Launch after immutable Volume upload:
`venv/bin/modal run modal_ocr_pool_expansion.py --train --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9 --steps 6000`. No train/hash allocates no GPU.
A first launcher import failed before research/preflight because it imported a
sibling Modal entrypoint that was not bundled into the container; that app was
explicitly stopped and launcher made self-contained. It is not a training arm.

Six new regression tests verify unchanged expansion prefix/calibration/eval/
record/vocabulary contracts, restored moments/LR/RNG with matching next updates,
parent checkpoint/calibration/LR guards, state fingerprints, and protected
parent-output paths and self-contained Modal entrypoint imports. Existing evaluation regression now also verifies that
posterior evaluation preserves training RNG.188 tests pass in root and fork.

Completed paired continuation:
`checkpoints/iam_ocr_pool_expansion/20261007-041949/report/index.html`;
family latest `checkpoints/iam_ocr_pool_expansion/latest/index.html`.
Local mirrors under `data/`. Each arm received6000 updates; total steps6000→12000.
DEV selected control2048 at8000, expanded8192 at12000 (the final expanded step,
not a convergence claim). Selected-head hashes:
- control2048: `6ca0ccf2acd99b86bee4b47bd26508f62337dd8cf2b5a02fd475179cd41f817e`.
- expanded8192: `5de8792405583c7651f83de12ac7300f398fa5b088c5d0570ad00bb9f22b75ed`.

| DEV-selected arm | TRAIN mean CER/exact | DEV mean/posterior CER | report32 mean/posterior CER |
|---|---|---|---|
| parent2048/6000 |0.01925% /2036of2048|28.2857% /28.2545%|25.1008% /25.1563%|
| control2048/8000 |0.00321% /2046of2048|27.7143% /27.7584%|23.3871% /23.5232%|
| expanded8192/12000 |7.22379% /2606of8192|19.2208% /19.2740%|16.2298% /16.2954%|

Expanded DEV exact4/128, report exact1/32; control1/128 and0/32. CTC DEV
1.88058→0.90038, report1.58443→0.73870 (control→expanded). Common32 TRAIN
probe changes0→2.4414% mean /2.4609% posterior CER: expanding the distribution
also gives up some tiny-pool memorization. Posterior recognition closely tracks
means; posterior noise is not the primary reader-generalization failure here.
Train/evaluation loops took132.55s/169.63s, respectively; excludes all-line
preflight/container startup/CPU gallery, and is not total billed time.

All **8352** unique pool lines pass frozen mean geometry/pen gates (8192+128+32;
an earlier informal8368 count was an arithmetic typo): mean per-line X/Y RMSE
7.25312e-6/5.70495e-6, mean-per-line turn p90.04101°, maximum packed XY
roundoff8.29697e-5; every pen boundary/final EOC correct, no internal EOC.
Entire source codec tensors and source file hash stay unchanged after both arms.
There is no before/after geometry intervention: one protected reconstruction is
shared by both heads. CPU reload covers192 eval/probe lines, not8192 TRAIN.

The first CPU report stopped at an over-strict equality assertion comparing
complete baseline metric dictionaries. Both baseline branches have IDENTICAL
mean/posterior decoded records on all2208 parent-pool lines. Different masked
mean evaluation batch shapes produce only FP32 CTC roundoff (max per-line
4.76837e-7). The reporter now permits1e-5 absolute CTC roundoff, but still
requires exact mean/posterior decode records and full parent ID retention;
a regression verifies that changed transcripts/draws cannot pass this guard.
No GPU training was rerun for this report-only fix. Failed launcher/CPU report
logs are preserved outside completed arms. As-run config/checkpoints/snapshots
are immutable: their inherited legacy `initial_head_tensor_sha256` still refers
to the parent's fresh initialization. The authoritative continuation digest is
`initial_state.head_tensor_sha256`, paired across arms together with optimizer
and RNG digests. Future runner explicitly overwrites the legacy field and adds
parent-fresh digest/seed-policy metadata rather than rewriting historical files.

Interpretation: more TRAIN supervision helps beyond another6000 steps of
memorizing2048 (8.49 percentage points better DEV,7.16 better report32 against
matched continuation control). It does NOT solve recognition:19% unseen-writer
CER and7% TRAIN CER remain, and the final expanded step was still improving.
Do not release faithful geometry to this reader. Next sensible experiment is
bounded convergence/reader optimization on the fixed8192 pool before increasing
corpus again or claiming a semantic/generative latent. No joint codec/CTC/KL/
style/InkDiT training is promoted. The report preserves all128 DEV/32 report
marker-free panels/transcripts, mean/posterior metrics, learning curves, paired
line-error changes and predeclared CTC frame-slack diagnostics (index frames,
not physical duration).189 root/fork tests pass.

CPU reload: all192 eval/probe mean decoded records agree with the selected GPU
checkpoints in both arms (zero differences). Against control, expanded improves
108/128 DEV lines, ties13, worsens7; report32 improves25, ties5, worsens2.
Against the old parent, it improves112/128 DEV (two worse) and28/32 report
(one worse). All32 report panels plus DEV pages1/6/11/16 (32 varied DEV lines)
were visually inspected: shared source curves remain indistinguishable from
IAM/RDP at gallery scale; authentic sharp hooks/polygonality are not smoothed.
OCR improves visibly but still misreads names, case, spaces and individual
letters, even on visually straightforward lines. All160 panels are preserved.

Exploratory compression-slack result: for expanded DEV,18 lines with
`(ceil(points/8)-CTC_required)/characters≤.25` have29.15% CER;66 with margin>.5
have14.81%. These are nonuniform index-frame margins, not physical time.
This is a correlation confounded by text/length/writer difficulty, NOT proof
that the8× bottleneck destroys geometry (transport geometry is faithful).
If longer reader training plateaus, a controlled OCR-only temporal-resolution
ablation is justified; don't change the protected codec based on this alone.
All Modal GPU/CPU apps are stopped after artifact publication.

## Fixed8192 OCR convergence: paired LR1e-4 versus2e-4 (2026-10-07)

Next experiment keeps the supervision pool fixed, NOT another corpus expansion.
Pin `expanded8192/head-best.pt` at total step12000:
`checkpoints/iam_ocr_pool_expansion/20261007-041949/expanded8192/head-best.pt`,
SHA `5de8792405583c7651f83de12ac7300f398fa5b088c5d0570ad00bb9f22b75ed`.
Same immutable pool SHA
`d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9`,
same8192 TRAIN/128 DEV/32 report,186 TRAIN writers, fixed81 chars, calibration192,
relative-scaled features, global attention, blankbias0/dropout.1.

Why test a modest higher LR rather than immediately lower it: parent clipping
occurred on only1/6000 steps (none in last1000). First1000 median/p95 gradient
2.369/3.414; last1000 2.102/2.665, under clip5. Median shuffled training CTC
.978→.527; final mean TRAIN CER7.224% was still falling. This does not prove
higher LR is better; it makes2× LR a bounded optimization diagnostic, not a
response to exploding gradients or a geometry-polishing change.

Both heads restore exact parent parameters/buffers, Adam moments/counters and
CPU/CUDA RNG. Control keeps inherited LR1e-4; other changes ONLY group LR to2e-4,
with full optimizer-except-LR fingerprints proving moments/counters/other
settings are untouched. Both run8000 additional updates (total12000→20000),
constant LR, AdamW betas.9/.99, decay1e-4, clip5, cached mu/masked batch16;
encoder physical batch1. Unlike the preceding expansion, data iterator is
actually resumed: regenerate seed43 bucket stream, skip the6000 consumed parent
batches, continue within its partially consumed epoch. Both arms have identical
batches/shapes, so dropout draws can be paired; every batch ID, sample-schedule
hash, RNG at each1000-step evaluation and final RNG are checked. No LR scheduler,
augmentation, KL/style, head architecture or codec/readout change is incidental.

Whole codec (including its old OCR) stays frozen. All8352 observed means are
encoded/decoded and gated before any head update; source SHA/state checked after
each arm. Feature moments inherited verbatim, independently checked.20 paired
GPU posterior draws on DEV128/report32/common TRAIN32; full8192 TRAIN metrics
are means only. DEV CER then CTC selects independently, original32 report-only.
Same cumulative DEV reuse/five-writer/pretrained-codec caveats still apply: not
paper reproduction, fully independent IAM benchmark or generation readiness.

Explicit bounded T4 launch:
`venv/bin/modal run modal_ocr_convergence.py --train --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9 --steps 8000`.
No flag or wrong/missing fixed-pool hash allocates no GPU. T4/cpu4, sequential
arms, per-arm900s/whole-function2400s, retries0/maxcontainers1. Shared runner /
reporter retain expansion mode as default; convergence adds a pinned preset,
not duplicated model/training implementations. CPU report reloads192 eval/probe
lines independently, full TRAIN/all8352 source gates explicitly as-run GPU.

A separate zero-training CPU readout diagnostic is predeclared: CTC prefix beam
width10 over ALL alphabet columns, no language model/lexicon/token pruning,
on the selected heads' mean logits for DEV/report only. It sums CTC paths rather
than simply collapsing per-frame argmax. Checkpoint selection remains GREEDY
DEV, never beam or report scores. Beam inference does not change geometry or
training; no posterior beam evaluation is claimed. Raw stable log_softmax logits
are used, not the CTC loss-only [-30,30] clamp. Beam pruning is approximate;
small exhaustive-path tests verify algorithm correctness and repeated-character
blank semantics.

Eleven new tests: exact iterator suffix across partial epochs without disturbing
Torch RNG, LR-only optimizer mutation, pinned split/feature/iterator guards,
paired dropout RNG despite different LRs, self-contained Modal launcher,
CTC path-sum advantage over greedy, repeated characters, exhaustive path parity,
empty/all-blank behavior, invalid inputs, and same-logit mean-only head audit /
training-mode/RNG restoration.200 root/fork tests pass.


### Completed: more optimization did NOT improve unseen-writer reading

Both arms completed all8000 updates, total step20000. Parent baseline records
match exactly on all8352 lines (maximum CTC difference0); per-update sample IDs,
iterator digest, evaluation/final RNG and optimizer-except-LR fingerprints match.
Both independently select the ORIGINAL step12000 on greedy DEV; selected model
tensors are unchanged from the source. The new `head-best.pt` file hashes differ
because their continuation settings/LR/provenance differ, not better weights.

| Mean CER (%) | Parent / selected both | Final LR1e-4 | Final LR2e-4 |
|---|---:|---:|---:|
| TRAIN8192 |7.2238|2.5593|3.1268|
| DEV128 unseen writers |19.2208|19.7143|19.8442|
| report32 seen writers |16.2298|15.7258|14.2137|
| common TRAIN32 |2.4414|1.8555|2.7344|

Final TRAIN exact lines2606→5301/4614 of8192; DEV4→2/1 of128.
Final DEV20-draw posterior CER19.6714/19.7935%, report15.7762/14.3196%.
Final mean DEV CTC .98885/1.02935 versus parent .90038. LR1e-4 loop/evaluation
221.51s and LR2e-4 221.99s; these exclude startup, corpus cache/preflight and CPU
reporting and are NOT total billed GPU times. Do NOT promote LR2e-4 based on
14.21% report32: report IDs NEVER select, DEV worsened. This falsifies a simple
"just another8000 updates /2× LR will improve DEV" hypothesis, not every possible
optimization schedule. The TRAIN/DEV gap now argues for representation/reader
or generalization diagnostics instead of blind longer training.

CPU prefix-beam width10, no LM/lexicon, all alphabet columns, SAME mean logits:
DEV19.2208→19.0130% (18 improved/99 tied/11 worse;740→732 errors/3850 chars),
report16.2298→16.1290% (3/27/2;161→160 errors/992 chars). Exact counts stay4/128
and1/32. All selected arms identical, hence all beam results identical. Tiny net
changes do NOT establish beam as a production improvement; greedy readout is
not the main19% CER bottleneck. No posterior beam scores or beam-selected head.

Additional CPU acquisition proxy counts ONLY true pen-boundary jumps with
`nextX-previousX < -0.5` model units (half normalized lineheight). Continuous-stroke
backtracking/loops do not count. This is chronological index geometry, not
physical velocity, a character alignment, a label-error claim or proof of
late-dot causation. Every HDF5 point fingerprint/evaluation record was checked.

| Group |No large backward pen jump: lines / CER|At least one: lines / CER|
|---|---:|---:|
| TRAIN |7423 /6.9769%|769 /9.6885%|
| DEV |126 /18.9833%|2 /31.5068%|
| report |31 /15.9375%|1 /25.0000%|

TRAIN association is descriptive/confounded;2 DEV positives are insufficient
for conclusions. Most DEV errors remain without large backward jumps, so this
specific phenomenon cannot explain the broad failure. Smaller deferred strokes
and local acquisition order remain unresolved. This does not rule out an
OCR-only finer-frame adapter; earlier CTC frame-slack association is a distinct,
also confounded, hypothesis. A4-versus8-point/frame readout is a sensible NEXT
controlled test, NOT implemented or launched in this study. Do not change the
8× transport codec or enable joint VAE/CTC/KL/style/InkDiT from these results.

Codec source SHA `9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`
and all parameters/buffers remain bitwise unchanged. All8352 observed means:
X/Y RMSE7.2531e-6/5.7049e-6, mean per-line turn p90 .041012°, packed XY maximum
8.29697e-5; perfect pen boundaries/final EOC and no internal false EOC. This
remains an initialized40-field chronological polyphase transport research
codec, NOT a normal learned semantic VAE or a paper reproduction. CPU independently
reloads192 DEV/report/probe lines, not all8192 TRAIN. All selected decoded records
match GPU.20 sampled-z metrics are the GPU as-run records. No geometry updates.

Report: `checkpoints/iam_ocr_convergence/20261007-044341/report/index.html`;
read `report/conclusion.html` first for the negative-result interpretation and
acquisition proxy. Volume `diffink-data`, local ignored `data/` mirror. All128
DEV/32 report marker-free panels are generated; this turn visually inspected
learning plot, report page2 and DEV page6, not every160 panel anew. Shared codec
unchanged, all selected OCR captions intentionally identical. Prior expansion
review covers all32 report panels and32 diverse DEV panels with this same codec.

Saved heads under the dated study:
- `lr1e-4/head-best.pt`: SHA `c41749c10877727b9f01623aa6f0b3da69b060dec647310956ef037737d182f7`
- `lr1e-4/head-last.pt`: SHA `71d16b4693fc6e0fb5060ec30569554bed6cc1f9f0c7705a636a0c5be1dfe7c1`
- `lr2e-4/head-best.pt`: SHA `3273b38c34e50c830ca0484869a10eb8f8765a47698ce5998da545d528416157`
- `lr2e-4/head-last.pt`: SHA `98abe5ee472d4d6683fecd74942384d745785b6aae9f4540a8f74e8de7727379`

As-run report summary inherited the expansion helper's generic "different masked
batch shapes" baseline note. In THIS fixed-pool study batch shapes are identical
and CTC difference exactly0; reusable reporter now writes the correct note.
Immutable as-run source/checkpoints/summary are not rewritten; post-run review
preserves this clarification and CPU analysis source separately. Three extra
backward-jump tests bring this turn to14 new tests /203 root and fork tests passing.
All GPU/CPU Modal apps finished; no runaway continuation remains.

## OCR-only frame granularity: fresh4/8-point readers (2026-10-07)

Predeclared next test, following the negative fixed8192 LR continuations.
Same immutable pool SHA
`d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9`;
8192 TRAIN /128 DEV /32 report,186 TRAIN writers,25 test writers excluded.
Same research initialized polyphase40 codec/source SHA
`9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`,
all codec params/buffers frozen, physical encoder batch1/minimal8-pad.
All8352 means/point fingerprints/geometry/pen gated BEFORE any OCR update.

Control8-point frames retains existing feature path bitwise. Alternative splits
chronological8×[XY,pen3] first40 fields into two4×[XY,pen3] first20 fields,
then zeros unused input channels. No interpolation, smoothing, RDP/data change,
new trajectories or additional labels. Valid length is ceil(real points/4),
including real final EOC but excluding an entirely synthetic final half-block.
Exact packed chronology/pen states/logvar mapping checked by tests. It is a
research transport adapter, NOT a generic learned-latent upsampler.

Both heads FRESH seed42, same384 input/hidden widths,3 layers/4 heads/parameter
count/initial weight digest. Original step12000 head is NOT transplanted into a
new input convention. Shared batch IDs are precomputed from original8-frame
cache, seed43 buckets, batch16,8000 updates. AdamW5e-4 for6000 updates then1e-4,
betas.9/.99,decay1e-4,clip5,dropout.1,blankbias0,global attention, mean training.
Same calibration TRAIN192 IDs/real points, but moments fitted separately for4/8
because local anchor-difference distributions change. Sequence position indices
also change. This tests frame/input granularity as a package, NOT isolated CTC
length alone. Dropout streams unpaired because shapes differ; no such claim.
Same labels/exposure/optimizer budgets/LR;4-frame attention costs more FLOPs.

Evaluate every1000 updates: means on all8352;20 posterior draws on DEV128,
report32,common TRAIN32. Noise is drawn in ORIGINAL384×T8 space then split into
4-point grouping, so valid-field noise is paired on SAME device without doubling
unused-channel sampling. CPU/CUDA draws differ; CPU report independently reloads
192 eval/probe lines only, not8192 TRAIN. DEV mean CER then CTC selects; report32
NEVER selects. Compare both readers within SAME reference8-frame slack bins,
not different populations. CTC frame slack is nonuniform index margin, not time.
Same cumulative five-DEV-writer/reuse and codec prior192/report prompt-overlap
caveats: not a fully independent IAM benchmark, semantic VAE or paper reproduction.

Launch: `venv/bin/modal run modal_ocr_frames.py --train --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9 --steps 8000`.
Guarded/import-only allocates no GPU. One T4,cpu4,16GB, sequential arms;
per-arm1800s and whole function4800s, retries0/maxcontainers1. No architecture,
geometry loss, joint CTC/KL/style or InkDiT changes. If a wall cap prevents equal
updates, reporter must reject paired-completion interpretation, not hide it.
Eight new tests cover remainder/end masking, chronology/gradient mapping,
unused/padded NaN exclusion, backwards-compatible8 path, translation/calibration,
paired original-space noise, identical head weights/finite backward/mask parity,
posterior callback/RNG and launcher isolation. Results appended AFTER completion.

CPU pool-only diagnostic before results: fixed TRAIN8192 processed points range
200–813;8-point max102 frames versus4-point204, well below OCR's1000-position
encoding buffer. Median normalized TRAIN slack .480→1.958; tight≤.25 cases
1299→0. DEV128 max points579, median slack .515→2.016, tight18→0. This is NOT
an accuracy result. Every item was already8-frame CTC feasible; this test does
NOT evaluate excluded short/infeasible IAM lines. Feature granularity, not data
selection, changes. Additional report tests check same-reference bin assignment
and ID-based paired errors; total10 new tests /213 root/fork tests expected.
CTC shape/length semantics verified against official PyTorch2.14 docs
(https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html); Context7
mask docs available, specific CTC query absent, hence official-document fallback.

Guard smoke verified: invoking `modal_ocr_frames.py` without flags creates no
GPU call. Root/fork full suites now BOTH pass213 tests (not merely expected).
A faulty unit fixture initially overwrote its own final EOC with pen-up at
N=4; corrected fixture BEFORE GPU launch. No failed GPU run or model/data patch
was needed for that test failure. Two later report-only tests added after launch;
training helper/model source unchanged. CPU report snapshots its own as-run code.


### Completed4/8-point study: finer OCR frames help, geometry stays locked

Both8000-update arms finished, identical fresh weights (as-run tensor SHA
`9e63e4fdbaf4d6f9922eaf494e5f4e7e3b1792577907c6cc8a991c74c88c0238`),
per-update batch IDs, schedule hash, LR exposure and source codec. Eight-point
selects step7000; four-point step8000 using DEV only. Comparing selected readers:

| CER (%) |8-point fresh control|4-point fresh alternative|
|---|---:|---:|
| TRAIN8192 mean |7.3601|4.8016|
| DEV128 mean |15.2987|11.4286|
| DEV20-draw posterior |15.2662|11.5078|
| report32 mean |12.3992|10.1815|
| report20-draw posterior |12.4093|10.1058|
| common TRAIN32 mean |10.2539|5.7617|

DEV589→440 errors/3850 characters,25.30% relative fewer errors; exact7→12/128.
Report123→101 errors/992 characters, but exact lines2→1/32; do not hide this.
CPU independent selected192-line reload has ZERO mean transcript differences
from GPU in both heads. Paired DEV84 lines improve,22 tie,22 worsen; report17
improve,6 tie,9 worsen. Finer frames are NOT uniformly better on every line.
4-point step7000 also beats8-point step7000 (11.7143 vs15.2987% DEV), so the
result is not solely comparing selected7000 versus8000. At equal final8000:
8-point TRAIN6.4668/DEV15.3247/report12.8024%;4-point4.8016/11.4286/10.1815%.
8-point loop/eval221.69s;4-point293.35s (1.32×). These exclude startup/cache/
preflight/CPU reporting; NOT total billed runtime. No nonfinite failure/wall cap.

Same reference8-frame slack bins, DEV:
- tight≤.25:18 lines,24.9216→18.4953% CER (159→118 errors/638 chars);
- middle .25–.5:44 lines,15.8510→12.1987% (217→167/1369);
- loose>.5:66 lines,11.5572→8.4102% (213→155/1843).

Gain spans all bins, so tight CTC margins are not the ONLY contribution. Input
phase grouping, relative anchor features/calibration and positional indices
also changed. This establishes benefit of this complete4-point OCR adapter on
this single-seed pool, NOT proof of pure CTC-length causation. Need replication
before making robust generalization claims. All tests still excluded as before;
five DEV writers/cumulative reuse and codec prior report prompt overlap persist.

An additional useful observation: even the fresh8-point control improves the
previous step12000 reader's19.2208% DEV to15.2987%, despite fewer absolute updates.
That combines fresh full8192 exposure/reset and higher-initial-LR/drop schedule;
it is NOT isolated evidence for reset alone, schedule alone or frame resolution.
Don't attribute the entire19.22→11.43 gain to splitting frames.

Frame-dependent calibration, same69204 TRAIN192 real points:
8-point XY mean [.1182713,.5076760],std [.2144859,.1962347];
4-point mean [.0657100,.5076760],std [.1689183,.1962347]. No DEV moments fit.
DEV mean blank-frame fraction20.378→56.232%, while CER falls: with longer CTC
sequences, more blanks is expected/useful and NOT evidence of blank collapse.

All8352 frozen observed means remain unchanged: X/Y RMSE7.2531e-6/5.7049e-6;
mean per-line geometric turnp90 .041012°; packed XY maximum8.29697e-5; every
pen boundary/final EOC perfect and no internal falseEOC. No new stroke geometry,
no smoothing/resampling, encoder/decoder/style/KL/old OCR updates. Posterior
metrics here evaluate READING, not newly certify all8352 sampled-z geometries.

Marker-free visual review this turn: learning plot, report page2 and DEV page6
(16 varied lines), all160 panels generated/preserved. Target/reconstruction
curves remain indistinguishable at gallery scale, authentic IAM/RDP hooks and
corners unchanged. Captions show concrete corrections such as "narrow across
the knuckles" and "reproduce the same form", but names/case/punctuation remain
wrong on many lines; one a07-family line is worse in4-point. No claim of perfect
reading or that every rendered handwriting line was newly visually reviewed.

Extra local initialization integrity check: CPU-only versus CUDA Torch2.14.1
wheels regenerate sinusoidal PE with maximum6.1035e-5 differences (buffer, not
learned weights). Direct whole-state local-init hash therefore differs. Copy
ONLY the saved NONTRAINABLE PE buffer into a fresh local head: entire initial
hash matches as-run in BOTH arms, proving learnable initialization identical.
Selected checkpoints already load their full saved buffers; CPU reload has
zero decode differences. This is not a model/data bug or a new architecture
change; preserve `initialization-reload-diagnostic.json` rather than claiming
raw cross-wheel fresh-state hashes match.

Report: `checkpoints/iam_ocr_frame_study/20261007-052514/report/index.html`;
interpretation `report/conclusion.html`, local ignored `data/` mirror, Volume
`diffink-data`. Selected heads downloaded locally and SHA verified; final heads
remain saved in Volume. Exact hashes:
- `points8/head-best.pt`: `cfda1aac570f582bca58024e7b14a45a66ad34e1ecd8c8723a6be08ba42d524d`
- `points8/head-last.pt`: `712ab09574bd9cb8ca761d554a2fd566792d65d0f828db7bce79a04cd5a7c355`
- `points4/head-best.pt`: `6b94ac8ab6897ff384b7a9b9c78fb0e67d0c60d925797cfe8eebb4cc25a75fee`
- `points4/head-last.pt`: `9957719ae8e400e636be46e0a71bde9ab712791cdd906f418db702ad7b26f2fa`

213 root/fork tests pass,10 new frame/report tests. GPU app
`ap-DH9XEht7yWV3IJ0ELLRLka`, CPU report `ap-EpTFPIrJ3vchbU7w0iCaGP` completed;
no-flag guard app `ap-LE4jaMyFZ1QmtllDQsnkb3` allocated no GPU. All jobs finished.

Next: replicate the4-vs8 gain, or a predeclared OCR-only2-vs4 resolution test;
not launched here.11.43% unseen-writer CER is progress, NOT a dependable oracle
for releasing faithful geometry. Keep codec/geometry protected; no unconditional
joint VAE/CTC/KL/style/InkDiT promotion. This initialized transport adapter is
not a generic semantic-VAE latent upsampler or authors' English reproduction.

## Predeclared replication:4/8-point OCR, initialization/dropout seed137

Repeat the exact fixed8192 paired8000-update experiment with a second head
initialization AND training dropout seed137. Data-order seed43 remains fixed,
so every sample/update exposure matches the seed42 pair; posterior draw seeds
8042+j*100 also remain fixed in original384×T8 space. Seed42 is NOT retrained
or overwritten. All filters/splits/source/calibration, features/positions,
head architecture/batch16, AdamW5e-4→1e-4 after6000, clip5, mean training,
eval1000, DEV-only selection,20 posterior draws and frozen-codec gates unchanged.
The only across-run intervention is fresh head initialization/dropout RNG seed;
within each run it is4 versus8-point OCR grouping. Same source/pool SHA as above.
No data-order robustness, extra writers, test-set performance, multiple new
seeds, language model or joint codec training is claimed.

Runner/CLI expose validated `--seed` (default42 retains historical behavior);
negative/non-integer/out-of-range seeds fail before GPU allocation. CPU report
records/uses saved seed; selected state/buffers always fully reloaded. T4/cpu4,
sequential arms,8000 updates each,1800s per-arm/4800s whole guard, no retries.
Launch: `venv/bin/modal run modal_ocr_frames.py --train --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9 --steps 8000 --seed 137`.

Primary question: does the4-point advantage reappear on the SAME five DEV
writers with another initialization/dropout stream? Report both seeds separately,
not just whichever seed has the lowest DEV score, and preserve all regressions.
Inspect per-writer changes to distinguish broad gains from a single writer's
contribution. Two-seed averages repeat the SAME128 lines, not256 independent
samples, and are NOT confidence intervals or a formal IAM benchmark. Results
and checkpoint/report hashes appended after completion. No geometry changes.

Three new seed tests validate bounds/default42/within-seed weight identity,
across-seed weight differences and RNG preservation. Five cross-seed reporter
tests guard fixed hyperparameters/data/schedule, predeclared seeds and actual
completion, CPU reload agreement, writer/character counts, malformed/duplicate
errors, and immutable seed42 summary hash.221 root/fork tests pass. CPU-only
cross-seed reporter added while GPU runs; training helper remains unchanged.
PyTorch current docs confirm manual_seed affects CPU/CUDA and fork_rng restores
states (Context7 official source); paired posterior evaluation still uses its
old fixed original-space draw stream. Raw Torch seed range is wider than this
runner's deliberately restricted nonnegative32-bit experimental seed contract.


### Completed seed137 replication:4-point benefit repeats across all5 DEV writers

Both fresh seed137 heads completed8000 updates, selected8000 by DEV alone.
Initial weight SHA `84ccf4049a53af1e7ffe69154ac793ac8c644b5c09b0ca33706def460a063913`
is identical within the137 pair and differs from42. Every batch/update schedule,
LR exposure, source/pool/config/calibration field and frozen-codec guard matches
the seed42 experiment. Only initialization/dropout RNG changed across runs.
CPU cross-seed publisher checked all fixed fields and pinned immutable first
summary SHA `7e6f8385567840df0863d060a7a5b587b1b7f54d2585414574e2226d2f6dca58`.

| Mean CER (%) |seed42 8-point|seed42 4-point|seed137 8-point|seed137 4-point|
|---|---:|---:|---:|---:|
| TRAIN8192 |7.3601|4.8016|6.5658|4.8716|
| DEV128 |15.2987|11.4286|14.9351|11.5844|
| report32 |12.3992|10.1815|12.5000|10.4839|
| DEV20-draw posterior |15.2662|11.5078|14.9169|11.5766|
| report20-draw posterior |12.4093|10.1058|12.5857|10.3881|

Seed137 DEV575→446 errors/3850 chars (22.43% fewer); exact11→13/128.
Report124→104/992 (16.13% fewer); exact3→2/32.80 DEV lines improve/24 tie/24
worsen; report18/7/7. Seed42 has84/22/22 DEV and17/6/9 report; no regression is
hidden. Seed1374-point report7000 was lower9.7782% than selected8000 10.4839%,
but NEVER selects on report. Its DEV8000 is better11.5844 than7000 11.6883%.
Mean across2 seeds: DEV15.1169→11.5065% (23.88% fewer errors), report12.4496→
10.3327%. These reuse SAME128 DEV/32 report lines, NOT256/64 independent lines,
not an ensemble or confidence interval. No cherry-picking the better seed.

Every DEV writer improves in BOTH seeds. Seed137 per-writer micro CER:

| writer |lines/chars|8-point CER|4-point CER|
|---|---:|---:|---:|
|10066|24/989|24.0647%|19.0091%|
|10163|25/882|5.6689%|4.0816%|
|10192|26/662|12.0846%|8.0060%|
|10207|27/680|11.3235%|10.0000%|
|10211|26/637|20.4082%|15.8556%|

Thus gain is not solely one lucky writer. Writer10066 still~19%,10211~16%:
substantial style-dependent weakness remains. Five writers/repeated DEV reuse
and prior codec report-prompt overlap still limit generalization claims.
Same reference8-frame slack bins also improve in137: tight18 lines25.2351→
19.9060%, middle44 15.7049→10.6647%, loose66 10.7976→9.3869%. As before,
frame grouping/calibration/positions changed together; no pure CTC-length claim.

CPU selected192-line reload has ZERO mean transcript differences in both137
heads. GPU/CPU posterior RNG differs; GPU20-draw metrics above are as-run.
Local fresh137 learnable-weight hash independently matches as-run after reusing
ONLY saved immutable sinusoidal PE buffer (cross-wheel regenerated buffer differs
by6.1035e-5). Selected checkpoints always load full saved buffers. Seed42 results,
checkpoints and source snapshots not rewritten. Calibration moments match exactly
across seeds for each resolution, including same69204 real TRAIN192 points.

All8352 source mean gates repeat exactly: X/Y RMSE7.2531e-6/5.7049e-6;
mean per-line geometric turnp90 .041012°, maximum packedXY8.29697e-5, perfect
pen boundaries/final EOC/no internal falseEOC. Entire codec params/buffers/old
OCR bitwise frozen after BOTH runs. No geometry/GMM/pen/KL/style or InkDiT update.
Reader training is mean-only; posterior reading tests do not newly certify every
8352 sampled-z trajectory. Four-point loop/eval294.35s versus8-point225.55s;
exclude startup/preflight/cache/CPU reporting, not total billed time.

Visual review this turn: learning plot, report page2 and DEV page6 (16 varied
lines), all160 marker-free panels generated/preserved. Shared target/reconstruction
curves remain indistinguishable at gallery scale; only OCR captions change.
Corrections repeat on "narrow across the knuckles"/"reproduce the same form",
while names, case, punctuation and "survival after death" remain wrong in places.
Authentic target corners/polygonality are not smoothed or modified. No claim all
160 images or old eight exemplars were newly visually re-audited this turn.

New run: `checkpoints/iam_ocr_frame_study/20261007-055354/report/index.html`.
Paired BOTH-seed comparison/per-writer table:
`checkpoints/iam_ocr_frame_study/20261007-055354/replication/index.html`.
Volume `diffink-data`, local ignored `data/` mirrors, latest report pointer is
not necessarily the best model. Selected137 heads downloaded and SHA verified;
final heads remain saved in Volume. Hashes:
- `points8/head-best.pt`: `f3fd770773f6a25cacf7feae323b270cd9d23f4d1307c18514659aeba925cc4c`
- `points8/head-last.pt`: `46c605e5f6d65bf7f739e4e3b4d048d8002019f02bd69540299ab9b8491709ac`
- `points4/head-best.pt`: `235811c3db0b4fe091b1a60cb0190c1a4d33733ef7e00e06144a04cb754bd872`
- `points4/head-last.pt`: `825008f04bc800eea2443221bd6f6546f9960ff0bdaee70a66df1b544f8e2812`

221 root/fork tests pass. T4 app `ap-sNylCTSEYfCdSjBE1iE4eG` and CPU app
`ap-Qn0fRmeRYQstk2D3u3D85z` completed, no failed/crash-looping jobs or runaway
continuations. Preferred experimental reader granularity is now4-point for this
initialized transport. Default factory remains8 to avoid silently breaking old
checkpoints;4-point heads require explicit frame_cache/feature contract and are
NOT drop-in generic learned-latent OCR heads or ordinary VAE checkpoints.

Next justified question: bounded2-versus4-point OCR-only comparison, or isolate
another representation feature, keeping source/samples/geometry locked. Not
implemented/launched in this replication.11–12% DEV CER is not a dependable oracle
for unrestricted joint geometry training. No semantic-VAE/paper-reproduction/
independent-IAM-test or generation-readiness claim.


## Finer2-point OCR trial (2026-10-07, completed; rejected under this recipe)

A fresh matched4-versus2-point pair tested whether the replicated4-point benefit
continues monotonically. It does NOT under the current recipe. The completed
interrupted-session job/report was recovered and verified, not redundantly
relaunched. Saved as-run GPU/CPU code and existing checkpoints were NOT rewritten.

Runner now accepts ONLY predeclared `(8,4)` or `(4,2)` pairs; old8/4 default and
factory8-point default are unchanged. `--finer-frames` explicitly selects4/2.
Original8-point/40-field tensors are split chronologically into20 or10 active
fields padded to384channels, with valid lengthceil(realN/frameSize); final partial
frame retained, fully synthetic post-end frames excluded. Whole-line finalEOC and
all real XY/pen fields checked exactly in BOTH caches. Unused channels cannot enter
features/noise/calibration. No interpolation, resampling, smoothing or VAE change.

Configuration: SAME immutable8352 pool (8192TRAIN/128DEV/32report), same186TRAIN
writers/81chars and excluded25testwriters; pool SHA
`d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9`;
same frozen source SHA
`9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`.
Initialized polyphase40 transport research codec, NOT ordinary semanticVAE or
paper reproduction. Same fresh seed42 weights/dropout seed, data-order43 and
per-update batches/LR/exposure; input/hidden384,3layers/4heads,dropout.1,
blankbias0, globalattention, physicalOCRbatch16, encoderphysical1, noaugmentation.
AdamW5e-4 forupdates1–6000 then1e-4 through8000, betas.9/.99,wd1e-4,clip5.
SameTRAIN192 calibration IDs/69204realpoints, moments refit perresolution:
4-point XYmean[.0657099915,.5076760165],std[.1689183411,.1962346701];
2-point mean[.0396493722,.5076760165],std[.1300428191,.1962346701].
Frame grouping/calibration/localXanchor/position indices change as a PACKAGE, not
pureCTC-length isolation. Dropout unpaired across unequal shapes. Same original
384×T8 posterior noise stream drawn before re-indexing,20draws on192eval/probe
lines, not all8192TRAIN. DEV CER thenCTC alone selects;32report-only neverselects.
Both completed8000 and selected8000,1800s per-arm cap/4800s GPU job/retries0.

Launch (already completed; do not launch again merely to inspect):
```sh
venv/bin/modal run modal_ocr_frames.py --train --finer-frames --seed 42 \
  --steps 8000 \
  --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
venv/bin/modal run modal_ocr_frames.py \
  --report-rel checkpoints/iam_ocr_frame_study/20261007-074413
```

| selected CER |4-point control|2-point trial|
|---|---:|---:|
|TRAIN8192 mean|4.8105%|74.2023%|
|DEV128 mean|11.8961%|74.9091%|
|report32 mean|10.0806%|74.7984%|
|DEV20-draw posterior|11.8974%|74.9104%|
|report20-draw posterior|10.0151%|74.8034%|

DEV errors458→2884/3850chars; all128lines worsen, no ties/improvements.
Report32 errors100→742/992chars; all32worsen. Four-point exact3154TRAIN,
12DEV/3report;2-point ZERO exact lines anywhere. All5DEVwriters regress with2.
This is not a visual trajectory regression: curves are SHARED/frozen. Mean DEV
output length29.51 versus12.01chars for30.08targetchars. By1000updates2-point
DEV blank fraction99.7905%,82/128empty outputs; at8000 still92.7872%blank,
frequent short common-letter strings rather than faithful transcripts. Four-point
DEV blank56.9453%, meaningful captions. Increased blank fraction alone would NOT
establish collapse in longerCTC sequences; the huge CER, early empty strings and
inadequate output lengths establish that reading failed here.

Two-point runs are initially harder to optimize: first100updates clipping53%
versus16%, maxnorm56.63 versus23.60. Final1000 BOTH0%clipping, median norms
1.10 versus1.41. Persistent clipping is NOT the endpoint failure explanation.
CPU no-update initial-gradient probe used SAME32TRAINprobe, exact TARGET fields
packed into transport (NOT actual saved encoder mu), dropoutoff, fresh42 and
bias0/-2/-5. Bias0 rawnorm2-point59.8–63.3 versus4-point20.8–21.0, but negative
blank-bias gradients SIMILAR(-.32–.34 versus-.34–.37), not a3x initialblankpush.
Negative blank bias reduces blank gradient while increasing loss/totalnorm;
this does NOT demonstrate a working rescue. No DEV tuning, parameter updates or
production bias/LR change. See diagnostic source/metrics. Likely optimization and
framing-package interactions; which mechanism dominates remains unresolved.

Integrity: maximum4/2readerlength204/407, within1000positionalbuffer; no overflow.
CPU selected192-line reload ZERO mean decode differences in botharms; selected
fileSHAs verified locally. All8352geometry means repeat X/YRMSE7.2531e-6/
5.7049e-6, mean per-line geometric turnp90 .041012°, maxpackedXY8.29697e-5,
perfect pens/finalEOC/no internalEOC. Entire codec parameters/buffers/oldOCR
bitwise unchanged afterboth. Mean/posterior failure agrees, so samplednoise not
responsible.4/2 loops+evaluation353.86/554.82s; not billedduration and exclude
startup/cache/preflight/report. CPU8GB/GPU16GB,4CPUcores, no retries/crashloops.

New4control versus old42/4-control:11.8961 versus11.4286%DEV. Same headinitial,
source/data/features/calibration/schedule/hyperparameters; newconfig differences
only metadata. First159gradient and166loss differences, first158gradient and165
loss values identical. CUDA bitwise determinism NOT enabled; precise firstnumeric
divergence notisolated. Do not claim cross-run numerical identity. Primary
comparison is this new WITHIN-run matched4/2pair, not a cherry-picked prior4head.
Earlier TWOseed8/4replication remains unchanged and supports4-point preference.

Visual review: learning plot/reportpage1/DEVpage6,16varied lines; all160markerfree
panels saved, NOT all manually re-reviewed. Target/sharedreconstruction remains
visually matched, correct corners/microstructure not smoothed.2-point captions
collapse into nonsense;4-point captions broadly readable but names/case/
punctuation and some words still wrong. No claim new drawing-quality improvement.

Artifacts on `diffink-data` (local ignored `data/` mirror):
- `checkpoints/iam_ocr_frame_study/20261007-074413/report/index.html`
- `.../points4/head-best.pt`, `.../points2/head-best.pt`; final heads also onVolume.
- `.../diagnostics/engineering-review.json`, `initial-ctc-gradients.json`,
  `initial_ctc_gradients.py`, `paired_reader_review.py`, code-review provenance.
As-run summary/source/checkpoints not overwritten; diagnostics separate post-run.
GPU app `ap-LXPseOqedknkz4cqf7rRvC`, CPU `ap-4sqMb5yEyMThQUqk6hQ5aZ` completed.
Sevennewtests cover2-point allremainders/chronology/finalEOC, gradient/unusedNaN,
relativefeatures/calibration/padding, pairedoriginalnoise, weight/finiteCTC/
batchparity, predeclaredpair/defaultcompatibility and generalizedreportlabels.
228root/forktests pass. Two-point production promotion REJECTED under this recipe;
prefer explicit4-point experimentalreader. Does NOT prove2-pointcapacityfailure,
and no new independent-IAM benchmark (sameDEVwriters/prompts/repeated evaluations,
priorcodec reportpromptoverlap). Do not release handwritinggeometry to imperfect
OCR. No jointVAE/CTC/KL/style/InkDiT or generation-readiness promotion.


## Predeclared4-point OCR affine robustness experiment (2026-10-07)

The remaining11.8961%DEV CER is primarily genuine recognition, not case alone:
lowercasing only11.6623% (9fewer/458errors), lowercasing plus removingASCII
punctuation11.3612% on a DIFFERENT denominator. Never change verbatimlabels/CER.
One descriptive minimum edit path gives279substitutions/126deletions/53insertions,
with u→n,t→l,a→o,f→t common. Edit tie attribution is NOT character alignmentGT.
Writer10066 has196/989errors19.818%,10163 33/882 3.7415%,10211107/63716.7975%.
This suggests writer/shape robustness is a useful hypothesis, not proven cause.

Resume pinned4-point step8000 parent
`checkpoints/iam_ocr_frame_study/20261007-074413/points4/head-best.pt`, SHA
`937a09830c15757c0ba9a02982497aff0f88ffbc66ac2b8fb10e72063e367a79`.
Same8352pool/source/81chars, full8352geometry gate, immutablecodec and rawphysical1;
same4pointframe grouping, cleanTRAIN192calibrationmoments, parameterarchitecture,
parent head/buffers/AdamWmoments/counters/CPU+CUDA RNG, LR1e-4/betas.9/.99/wd1e-4/
clip5/dropout.1/blankbias0. Schedule regenerates original8frame length buckets and
continues EXACTLY after8000 parentupdates,seed43,batch16;4000additionalupdates,
evaluatecleanmeans every1000,DEVonly selection includingparentstep0. Same20draw
posteriorcheck192IDs with original384×T8noise before4frame reindex. NoKL/style/
trajectorytraining/augmentation or changes to rawIAM/RDP/scaling/contracts.

A=`clean_control`:0.5CTC(clean)+0.5CTC(clean), TWO forwards.
B=`affine_mix`:0.5CTC(clean)+0.5CTC(affine), TWO forwards.
Both draw identical per-line parameters with isolatedCPUgeneratorseed901, never
consume dropout RNG; same shapes/two forwards preserve matcheddropout streams.
Compare finalRNG hashes AND per-stepbatches/LR/affineparameter fingerprints.
Perturb ONLY OCRinput realXYphases, in rawmodelcoordinates BEFORE existingreader
transform: sx,sy independentlyUniform[.9,1.1],shearUniform[-.12,.12],yshift
Uniform[-.04,.04], x'=sx*x+shear*(y-.5), y'=.5+sy*(y-.5)+shift. Positive scales;
slant is HORIZONTALshear ofy, NOT rotation producing large sentence-endYdrift.
Sameaffineperline; all penvalues/strokeorder/finalEOC/syntheticphases/validpoint
counts preserved. Noresampling/interpolation/genericsmoothing; cachedmu immutable,
unusedchannels/noise excluded. Labels unchanged. Evaluation NEVER augmented.

T4,cpu4,16GB;1200sperarm/4800sfunction/retries0/maxcontainers1,sequentialarms.
Ifaugmentationfails, retainfailure and clean/parentreader; do NOT retune onreport.
No universalexpectation thataugmentationmusthelp or release geometry to11%OCR.
9newtests:affineformula/identity/pen/EOC/padding/unusedNaN/gradients, independent
boundedRNG, pairedtwo-forwarddropout/finiteCTC, parent/split/optimizerguards,
self-containedguardedlauncher, correctminimumedittraces/characterweighting.
Root/fork237tests passed preflight (lastdedicatedgeneratorguard also rechecked).

```sh
venv/bin/modal run modal_ocr_augmentation.py --train --steps 4000 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

CPU report will also repeat prefixbeamwidth10/noLM/noLexicon on the SAME clean
mean logits forparent and selectedarms, verbatim labels, DEV/reportonly. Never
select a head or tune width/LM usingbeam/report; posteriorbeam NOT evaluated.
Reportcontrols additionally reject unpairedRNG, affine draws, updateexposure,
featurecalibration or hyperparameter drift; three additionalreport tests.

AdditionalCPU robustness probe predeclared duringGPUrun:8fixed endpoints, one
dimension at a time (xscale.9/1.1,yscale.9/1.1,slant±.12,yshift±.04), applied
ONLY to192eval/probe OCRmeans forparent andbothselectedreaders. No draws/codec
renderchanges; always label verbatim originaltext. Measures sensitivity toKNOWN
syntheticaffine perturbations, NOT independentwriter generalization. No tuning
coefficients/checkpoint choice onprobeperformance, cleanDEV alone stillselects.


## Predeclared4-point OCR spatialX cue (2026-10-07)

Affine4000-per-arm run completed: parent11.8961%DEV, selectedclean2000additional
11.1688%, selectedaffine2000 11.2208% (2more/3850errors); report8.8710 versus
8.7702% (1fewer/992errors). No meaningful evidence augmentationbeats matchedclean.
Existingcoordinate signal is invertible up toXtranslation, but Transformermust
integrate anchor displacements to recover spatiallocation and pointspacing is
nonuniform. Next hypothesis: give explicit spatialX IN ADDITION tolocalfeatures,
not re-order points or addlabels/characteralignment. No causalclaim yet.

BOTH arms resume SAME pinned4-point8000 parent/AdamWmoments/counters/RNG,
LR1e-4 andoriginal8frame schedule suffix8000→12000, physical16, two0.5cleanCTC
forwards (matcheddropout),4000additionalupdates/cleanDEVselectionincludingparent0.
No augmentation/CTCresampling/newparameters/labelchanges. Input remains384;
20current localXY/pen channels unchanged. Treatment `spatial_x` appends4phase
values `(realX-min_realX)/max(realXspan,.01)` in previously unusedcolumns20:24;
control `local_control` keeps thoseinputs0. Syntheticphases excluded/zero;
chronologicalbackwardstrokes remainbackward. Featuresuse onlyobservedcoordinates,
NOT transcripts/writers or targetcharacterboundaries. Spatialposition is NOTtime,
velocity, curvature or trueCTCalignment. Original344unusedlatentnoise stillignored.

Common BOTH-arm intervention: zero INPUTprojection weights20:24 and assert their
parent exp_avg/exp_avg_sq EXACTLY0; parentfunction unchanged because oldfeature
columns always0. Noreset ofcounter/moments orunrelatedweights; sameheadparamcount,
head/effectiveinitialstate hashes, cleanstep0 means/draws mustagree exactly. Same
restoredbuffers/masks/geometry8352gate/CPU192reload/20draw noisecontract. Standalone
researchreader, not genericlearned-latentVAE/paperreproduction. Fixedsource/data,
DEV reused/fivewriters, report32only andpriorcodec reportpromptoverlap remain.
T4,cpu4,16GB,1200sperarm/4800sfunction,retries0,sequentialarms. If fails, preserve
failure and retainclean-only; no incidentaljointVAE/KL/style/InkDiT updates.
8tests cover realphase/spatialminmax/order/masking/unusedNaN, positiveaffine
invariance/backwardstrokes, constantXfloor/gradients, no local/pen featurechange,
exactbaselineparity aftercommonzeroing withmoments/counterspreserved, previously
usedcolumnrefusal, newgradient/dropoutpairing, guardedself-containedlauncher.

```sh
venv/bin/modal run modal_ocr_spatial.py --train --steps 4000 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Completed reader robustness/local-position controls (2026-10-07)

Both studies are finished, each with two sequential T4 arms of 4,000 additional
updates. Same pinned step-8,000 parent, 8,192 TRAIN / 128 DEV / 32 report lines,
optimizer moments/counters, data schedule, clean calibration and frozen codec.
Every arm selects +2,000 (total 10,000) using clean DEV alone, not the final step.
The historical blank bias was initialized at 0; these continuations restore the
learned parent bias, not reset it. Two half-weighted CTC forwards also average
training dropout relative to the parent's one-forward training: improvement over
the parent cannot be attributed to extra updates alone.

| Study / reader | TRAIN mean CER | DEV mean CER | DEV 20-draw CER | report mean CER |
|---|---:|---:|---:|---:|
| pinned parent 8,000 | 4.8105% | 11.8961% | see step-0 run records | 10.0806% |
| affine study / clean control +2,000 | 3.1779% | **11.1688%** | 11.2351% | 8.8710% |
| affine study / affine mix +2,000 | 3.1940% | 11.2208% | 11.2545% | 8.7702% |
| spatial study / local control +2,000 | 3.1763% | 11.3506% | 11.3571% | 8.8710% |
| spatial study / spatial X +2,000 | 3.1554% | 11.2727% | 11.2987% | 8.7702% |

Affine is 2 errors worse than its matched control (432 versus 430 / 3,850 DEV
characters). Spatial X is only 3 errors better than its own control (434 versus
437). Each treatment is only 1 error better on the 992-character report split,
which never selects. These small single-run differences do NOT establish a
reliable natural-writer robustness or spatial-feature benefit. Keep the clean
continuation as the best observed experimental reader, not a new production
contract. Independent controls differ by 7 DEV errors; matched RNG/exposure is
not bitwise deterministic CUDA optimization, and cross-study rankings are not
replication. No feature/augmentation/default architecture is promoted.

Late training is not uniformly helpful: affine-study final-step TRAIN CER falls
to 2.0980% / 2.1587%, while DEV worsens to 11.6883% / 11.5325%. The writer10066
bottleneck persists (clean 189/989, affine 188/989 errors); writer10192 worsens
versus the parent in both arms. No claim that simply training longer solves OCR.

CPU diagnostics, never checkpoint selection:
- Affine study DEV beam-10/no-LM mean-only CER: parent 11.5325%, clean 10.9351%,
  affine 10.8831%; only modest gains versus greedy. No posterior beam or lexicon.
- Eight fixed single-axis affine endpoint probes: average DEV CER parent
  11.9578%, clean 11.4513%, affine 11.3604%. This small synthetic-perturbation
  benefit is not demonstrated natural writer generalization.
- Original raw CSR and forms writer map match all 8,352 pool records across
  1,517 transcription forms, with ZERO text/writer mismatches. A percent-like
  handwritten mark in f10-558z-03 is absent from the ORIGINAL CSR ('these items
  were 22 ,'); this is a possible annotation disagreement, not a dropped parser
  character or a confirmed relabeling. No labels/alphabet/data filters changed.

Exact configuration, source snapshots, per-step IDs/LR/loss/clip records,
selected/final heads, full pool geometry preflight, CPU reload and galleries:
- `checkpoints/iam_ocr_augmentation/20261007-085841/report/index.html`
- `checkpoints/iam_ocr_spatial/20261007-091828/report/index.html`
All on Modal Volume `diffink-data`; selected heads and reports mirrored under
local `data/`. Combined reader review is linked from the separate
`checkpoints/iam_ocr_reader_studies/20261007-093500/index.html`.
Never overwrite as-run outputs to incorporate later review; diagnostics and
post-commit provenance live in separate dated review files.

Whole 8,352-line mean geometry/pen gate remains unchanged: mean per-line
X/Y RMSE 7.2531e-6 / 5.7049e-6, mean turn-error p90 0.0410 degrees, all pen
boundaries/final EOC perfect. No KL/style/trajectory decoder/InkDiT updates;
posterior OCR checks cover the same 192 evaluation/probe lines, not a new
all-pool sampled-trajectory certification. These are initialized polyphase
transport research readers, NOT the paper's ordinary semantic VAE experiment.
Repeated use of five DEV writers, only 32 report lines and prior codec report
prompt overlap remain limitations. Reconstruction pictures do not improve in
these OCR-only runs: look for reading-caption changes, not smoother curves.

23 new regression tests cover bounded affine parameters, isolated RNG, masking,
pen/EOC preservation, edit accounting, paired controls, spatial phases, exact
initial function parity/common unused-column zeroing, optimizer-state guards,
independent report guards and no-GPU-by-default launchers. Root and fork suites
both pass all 251 tests. Full reports include independent CPU selected-head
reload checks and the precise source/checkpoint SHA-256 values.

Next evidence-based question is broader writer/shape generalization, not another
uncontrolled LR or geometry continuation. Before unlocking geometry to OCR,
consider a controlled data-coverage or spatially appropriate OCR representation
study, with writer-stratified DEV metrics and a fresh locked validation set for
confirmation. Neither small feature change here resolves the ~11% OCR floor.

Independent spatial-study CPU reload has zero mean transcript disagreements for
parent/control/treatment, and every pairing/complete-budget/frozen-codec guard
passes. Spatial versus local DEV: 10 lines improve, 110 tie, 8 worsen; report:
2 improve, 29 tie, 1 worsen. Spatial improves writer10066 by 5 errors but worsens
10207 by 1 and 10211 by 2; no uniform writer benefit. CPU mean-only beam-10 DEV
local 11.0909%, spatial 11.0390%; report 8.7702% / 8.6694%, never selects heads.
Manually inspected spatial learning curve, dev pages7/12 and report page1;
affine learning curve, dev page7 and report page1. All160 panels generated per
study, NOT all manually inspected. f05-342z-06 remains misread in its tight
connected words; drawings are visually unchanged and target corners retained.
Separate `visual-review.json` records the exact inspected scope.

Training+evaluation loop times (not billing or complete wall time): affine
clean 242.20s / mix 241.32s; spatial local 270.02s / X 288.48s, excludes startup,
full-pool preflight and CPU gallery generation. GPU apps:
`ap-HvBsH5UwnB7FiAW8z1K2TN`, `ap-IESBiDCaliKZgbt5VRMnjY`; CPU reports:
`ap-M65Jkxm7HMllBIvgqSJCaA`, `ap-IZQhchwiDkBOyxpz3lVIFv`.
Selected clean SHA256 `f64d79e1ccfbfedaaafdaa365ba0458d9b0b28d2f7b99bca43d56cd86fe14998`;
affine `6495d2e0bf8e57d6bcfdd8cbadf136809b4e60ad0a9e977795eb84790c44e7f2`;
local `8b587a92473d08b23e68a19ed5715a3792d651da6b2ee72518c9b3e5f37564b0`;
spatial `d8e1396859eb2b7c4aee379142a7d17304d62257411daf5931d6969bc14a406e`.

## Predeclared local OCR context residual (2026-10-07)

Neither affine robustness nor explicit horizontal location solved writer-sensitive
reading. Parent frame input has only a linear projection before global attention;
connected stroke shape spans multiple irregularly spaced acquisition blocks.
Hypothesis: a small shared local nonlinear stencil can help recognize those
patterns. This is NOT a claim that temporal CNN context is the only missing cause.

Matched pair resumes the SAME pinned four-point step8000 parent used above, not a
cherry-picked stronger arm. All8352 observed mean codec/pen gates run again before
any training. Same TRAIN8192 / DEV128 / report32, calibration192, AdamW parent
moments/counters/CPU+CUDA RNG, original8-frame bucket schedule suffix8000→14000,
physicalOCR16/rawencoder1, LR1e-4/betas.9/.99/wd1e-4/clip5/dropout.1 and two
half-weighted clean CTC forwards. Every1000 clean evaluation and DEV-only selection
includingstep0; 20 fixed original384×T8 posterior draws then4pointreindex on192IDs.
Learned parent blank bias restored, no reset. No affine/spatialX/newlabels/rotation,
geometry/VAE/KL/style/DiT or data changes. T4/cpu4/16GB, 6000additionalupdates per
arm,1800sperarm/4800sfunction,retries0/maxcontainers1,sequential.

BOTH arms add identical7744parameter residual AFTER strict parent restoration:
Conv1d20→64/kernel5/pad2, GELU, Conv1d64→20/kernel1/nooutputbias, finalprojection
ZERO. Applied to existing20 relative-scaled fields before existing384inputlinear.
Same parent function and extra initial weights in both arms; control gate0,
treatment gate1. Parent optimizer state/indices/counters remain intact, branch
gets a new group at inherited hyperparameters. No residual dropout, resampling,
normalization or token length change; hence same dropout RNG/exposure is testable.
Zero frames before/afterconv and zero synthetic phases afterresidual. This spans
five acquisition frames (~20points), NOT arc-length/physical-time neighborhoods.
Pens remain observed context; this is OCR filtering, never target smoothing.

Nine tests cover exact initial parity, optimizer/counter/RNG preservation,
trainable branch versusdisabledcontrol, paddedNaN immunity and length invariance,
actual neighbor sensitivity/inputimmutability, strictselectedreload, duplicate
attachment rejection, guardedlauncher, pairedconfig/state/budget/RNG/exposure
reportguards. Initial seven-test full suites pass258; nine-test focused suite
passes, full260 suite will be rechecked. Explicit research checkpoint contract,
not ordinary semanticVAE/paper reproduction. ReusedDEV and reportoverlap caveats
continue; don't promote tiny per-run differences.

```sh
venv/bin/modal run modal_ocr_local.py --train --steps 6000 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Predeclared packed recurrent OCR comparison (2026-10-07)

A read-only source-geometry audit on128DEV lines (parent8000 predictions) gives
Pearson(lineCER, points/character)=-0.360; writer10066 averages10.67points/char
versus12.47 for10163. X versusacquisition-index correlation is >.99 for all five
writers and has only-.066 correlation withlineCER. Large (>one model-height)
stroke-start backtracks are rare. These descriptive/reusedDEV correlations do
NOT prove point density causes errors, but don't support delayedstrokes as the
main explanation. Preserve diagnostic/source, no reordering or label changes.

Test a stronger serial-context inductive bias, separate from localresidual:
fresh4-point globalTransformer384/3layers/4heads/sinusoidalpositions versus
fresh3-layer packed bidirectionalGRU320/direction, inputlinear384 and output640.
Same20relative-scaled inputfields/calibration192/4-point clocks/masks/vocab;
GRU has no sinusoidalpositions (order suppliedbyrecurrence). 5,502,802 versus
5,250,002 parameters, GRU4.59%smaller. This is a WHOLE reader architecture test,
not isolated recurrence with identicalparameter/dropoutinitialization. No norms,
newdata/augmentation/derivativefeatures/label/trajectory/codec changes. Packed
right-padded sequences REQUIRED: backward recurrence must never process padding.
All dropout=.1; each freshseed42, not paired randomoperators acrossarchitectures.

Same8,192TRAIN/128DEV/32report and original8-frame seed43bucket schedule, physical16,
8,000updates each, oneCTCforward/update (same asoriginalfreshframe controls),
AdamW LR5e-4 untilupdate6,000 then1e-4, betas.9/.99/wd1e-4/clip5. Every1,000 clean
means/20fixedposterior draws, DEV-onlyselection includingstep0. No oldhead or
optimizer transfer (different architectures). Codec/source/data are identical to
otherstudies. Threshold for a convincing first improvement: at least~1percentage
point DEVgain againstmatchedTransformer, spread acrosswriters rather than one
fortunateglyph; report32neverselects. Confirm a substantial win with another
trainingseed before recommending the architecture. Small differences stayweak.

To avoid paying for identicaldecode/curve-scoring work again, bind andreuse
previousfull8352mean geometry/pen evidence from affine20261007-085841 ONLYafter
exactsourceSHA/config/poolmanifestbytes and complete-ID/pass validation. Refresh
ALL8352 encoder/packedrealXY/pen/mask/pointfingerprint/CTC checks from actualsource
weights. Copy originalgeometry/audit and save geometry-gate-reuse.json withboth
SHA256s. This is NOT a newly decodedfullpool or newstochasticgeometry test. CPU
report independently re-decodes192eval/probe lines and strictlyreloadsselected
heads. Immutablecodec hash/buffers/gradientabsence checked aftertraining.

T4/cpu4/16GB;1800sperarm/5400sfunction,retries0/maxcontainers1,sequentialarms.
This DIFFERENT study may run alongside the local-residual study, at mosttwoT4s,
not duplicatejobs. Ten tests cover packedpadding/NaN immunity, backwardpadding
invariance, independentseededinitialization, finiteCTCgradients, unusednoise
exclusion, invalidprefix masks, strict source/config/pool/ID reuseguards and
noGPU-by-default launcher. Standalone initializedtransport research, NOT ordinary
semanticVAE/paperreproduction. No KL/style/geometry/InkDiT promoted.

```sh
venv/bin/modal run modal_ocr_recurrent.py --train --steps 8000 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Predeclared OCR dropout regularization continuation (2026-10-07)

Local-context result is weak: matchedclean11.2987%DEV versuslocal11.1688% (only
5fewererrors), report8.8710% versus9.0726%worse. It merely ties the earlierbestclean
score and isn't a convincing win. Preserve rather than promote it. A fresh
Transformer/BiGRU architectural comparison is still running independently.

Strong TRAIN/DEVgap and lateDEVregression justify one orthogonalregularization
control: OCRdropout.1 versus.3, BOTH from selectedclean total10000 head
`iam_ocr_augmentation/20261007-085841/clean_control/head-best.pt`, SHA
`f64d79e1ccfbfedaaafdaa365ba0458d9b0b28d2f7b99bca43d56cd86fe14998`.
Keep actualparentweights/learnedblankbias/AdamWmoments/counters/CPU+CUDA RNG,
LR1e-4,betas.9/.99/wd1e-4/clip5 and twohalfweightedcleanCTCforwards. Resume exact
original8-frame seed43schedule AFTER10000, same8192TRAIN/calibration192/128DEV/
32report/4-point feature/input/outputarchitecture.6000additionalupdates/1800sperarm,
DEVonlyselection includingparent0, evalevery1000/20fixedposteriorOCRdraws192IDs.
Change only9Dropout.p and3MultiheadAttention.dropout attributes, no tensorchange
or extra parameters. Require initialevalparity and compare finaldropoutRNG plus
all per-updateIDs/LR. Trainingmasks/activations differintentionally. This is OCR
DROPOUT ONLY: trajectorydecoder remainsfrozenwithdropout0; do NOT reintroduce
its earlier jaggedness-causingdropout. Bind/reuse previous8352mean geometrygate,
refreshall8352encoders/realfieldchecks, independentlyre-decode192CPUreport asabove.

At mosttwoT4s concurrently; localGPUfinished. No label/newdata/augmentation/
trajectory/KL/style/DiTchanges. A >=~1ppDEVgain with broadwriter benefit would
warrant confirmation; another fewerrors doesnot resolvegeneralization. Fourunit
tests establish unchangedtensors/optimizer/RNG/evalfunction, completeOCRdropout
sitecoverage, trainingeffect withmatched RNGcount, invalidrates and guardedlaunch.

```sh
venv/bin/modal run modal_ocr_dropout.py --train --steps 6000 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Predeclared recurrent seed confirmation (2026-10-07)

First fresh architecture run completed: Transformerbest7000 DEV11.3506%,
BiGRUbest8000 DEV8.6753% (103fewer/3850errors,2.6753percentagepoints), report
9.5766% versus8.5685%. This exceeds the predeclared~1ppgate, but is one run on
reusedDEV. Do NOT recommend switch until savedhead reload andwriter audit.
Repeat BOTH fresharchitectures with seed137, samecode/data/calibration/schedule43/
8000updates/LRs/budgets/DEVselection/posteriordrawcontract. Only initialization
andtrainingdropout seed differs from42; still NOT pairedrandomoperators across
architectures. No hyperparameterchanges from the promisingfirstrun. Expose guarded
--seed for this replication; preserve oldseed42as-run launcher separately, core
traininghelper unchanged. At mosttwoT4s: firstrecurrentGPU finished, dropoutstudy
stillrunning; secondarchitecturaljob runsalongside it. Two-seed evidence stillnot
independent128×2samples, ensemble, freshwriters or papersemanticVAE reproduction.

```sh
venv/bin/modal run modal_ocr_recurrent.py --train --steps 8000 --seed 137 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Completed OCR reader improvement and confirmation (2026-10-07)

A meaningful reader improvement was repeated **without changing handwriting**:

| Fresh seed / reader | selected step | TRAIN mean CER | DEV mean CER | DEV 20-draw CER | report32 mean CER |
|---|---:|---:|---:|---:|---:|
| 42 / Transformer | 7000 | 5.3507% | 11.3506% | 11.4195% | 9.5766% |
| 42 / packed BiGRU | 8000 | 0.0205% | **8.6753%** | 8.7026% | 8.5685% |
| 137 / Transformer | 8000 | 4.6480% | 10.5974% | 10.5753% | 9.6774% |
| 137 / packed BiGRU | 8000 | 0.0149% | **8.7273%** | 8.7506% | 7.5605% |

Each pair uses the same 8,192 TRAIN lines, 128 DEV lines / 3,850 characters,
32 report-only lines / 992 characters, 8,000 updates per arm, four-point
inputs/output clock and original eight-frame sample schedule (seed43).
Actual per-update sample IDs and LRs match within both pairs; all budgets
completed and frozen-source guards pass. GRU reduces DEV errors 437→334 in
seed42 (103 fewer, 2.6753pp) and 408→336 in seed137 (72 fewer, 1.8701pp).
Two-seed average: 10.9740→8.7013%, 2.2727pp / approximately20.71% relative.
These are **not 256 independent DEV lines, an ensemble or fresh-writer
confirmation**. Seed42 is the canonical engineering candidate; seed137 is
confirmation, not selected using its lower report CER. Previous best clean
11.1688% is also higher than both GRU CERs, but came from an additional
continuation with different training exposure; do not conflate the controls.

All five DEV writers improve in seed42. Seed137 improves four; writer10163
is one error worse (21→22 out of882 characters):

| writer / characters | seed42 Transformer→GRU errors | seed137 Transformer→GRU errors |
|---|---:|---:|
| 10207 / 680 | 75→45 | 52→45 |
| 10163 / 882 | 35→17 | 21→22 |
| 10192 / 662 | 48→29 | 47→34 |
| 10211 / 637 | 93→88 | 93→85 |
| 10066 / 989 | 186→155 | 195→150 |

Seed42 DEV lines:72 improve /33 tie /23 worsen; exact transcripts13→29.
Seed137:60 /41 /27, exact20→24. Report line comparisons:seed42 16 /6 /10,
seed13716 /7 /9. Not every line improves. Hardest writer still has15–16%
CER and10211 approximately13–14%; punctuation, capitalization and tight
connected glyphs remain imperfect. `f05-342z-06` still has substantial errors.
Mean CTC loss need not improve with CER: seed42 GRU DEV0.4454 versus
Transformer0.4153 despite better greedy reading. Overconfidence/calibration
remains relevant before using this reader to supply geometry gradients.
**No automatic joint-training unlock.**

CPU reports independently reloaded all four selected heads: zero mean
transcript differences on192 eval/probe IDs versus GPU. The same20-draw
posterior contract excludes unused latent noise; no ensemble or posterior
beam was evaluated. Beam10 / no language model is not the win: seed42 GRU
8.6753→8.7532% (slightly worse), seed1378.7273→8.6753% (tiny improvement).
Keep greedy decoding canonical rather than selecting beam using report scores.

The source codec remained bitwise unchanged. For architecture/dropout studies,
reuse the exact, bound prior8,352-line mean decoder/pen gate; newly repeat all
8,352 encoder/real-phase/XY/pen/CTC checks. Independently re-decode and re-gate
192 CPU evaluation/probe lines. This is not a newly decoded full pool or a
new full-pool sampled-trajectory certification. Mean drawings, target corners
and pen boundaries are unchanged.

**What is established:** the recurrent reader package learns this online
sequence better at the same data/update budget, with4.59% fewer parameters.
This is not proof that recurrence alone caused the gain: it also removes
sinusoidal positions, changes parameter layout/initialization and training
dropout operators. Four-point grouping, CTC clock, labels and curve geometry
did not change, so those changes cannot explain this particular gain.
Positional-encoding scale or Transformer optimization remain possible
contributors. No decoder modification or generic smoothing was needed.

Negative experiments retained:
- Local residual,6,000 updates per arm: clean best+2,000 DEV11.2987%, local
  best+5,000 11.1688%, only five fewer errors. DEV34 better /59 tie /35 worse;
  report7 /17 /8 and8.8710→9.0726% (worse). Merely ties old best clean;
  not promoted.
- OCR dropout,6,000 updates per arm from selected total10,000 clean: both
  select step0 /11.1688% DEV. Last dropout0.1 TRAIN0.9782% /DEV11.5584%;
  dropout0.3 TRAIN5.5747% /DEV11.9740%. No benefit in this bounded continuation,
  not proof that all regularization is useless. Parent moments, batches,
  LRs and final dropout RNG match. Trajectory decoder dropout stays0/frozen.

Exact reports, source snapshots, checkpoints/configs/per-update logs:
- `checkpoints/iam_ocr_local_context/20261007-094937/report/index.html`
- `checkpoints/iam_ocr_recurrent_study/20261007-100011/report/index.html` (seed42)
- `checkpoints/iam_ocr_recurrent_study/20261007-101618/report/index.html` (seed137)
- `checkpoints/iam_ocr_dropout_study/20261007-100919/report/index.html`

Combined review:
`checkpoints/iam_ocr_reader_improvements/20261007-104500/index.html`, including
explicit selected-reader contract, negative results, manual visual-review
scope, descriptive source/order audit and post-commit provenance. Artifacts
are on Modal Volume `diffink-data` and the ignored local `data/` mirror.
Original as-run artifacts are not overwritten by later engineering reviews.
Canonical standalone head: seed42 `bigru/head-best.pt`, SHA
`2ebac71d897920233c8399a38e1f16fce24075cfa722c7d8e588ab14e564100e`.
Confirmation SHA:
`01781cf158372d5d884f4b02bf7200a267ad1b39a585d459f5a2ecaffbdb345f`.

278 tests pass in root and fork (27 new). Legacy eight-frame defaults,
official VAE/DiT training paths and paper-English configs remain unchanged.
These are standalone four-point initialized-polyphase40 research readers,
**not drop-in ordinary VAE heads or paper reproduction**. Reused five DEV
writers, only32 report lines and prior codec report-prompt overlap limit
claims. Before joint training: implement a deliberate differentiable adapter
and actual VAE-loss-path test, calibrate geometry gradients, and impose strict
geometry/pen gates. Reader CER alone does not authorize changing the current
faithful trajectories.

One CPU dropout report was canceled by local client disconnect (remote log
confirms cancellation); retry only the CPU report, not GPU training. The
canceled log is preserved separately. Installed CLI help/current Modal docs
confirm `modal run --detach` for long jobs that must survive disconnects;
retain bounded function timeouts and monitor completion rather than leaving
duplicate inputs running.

## Predeclared frozen-GRU joint geometry compatibility test (2026-10-07)

Source codec SHA `9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`;
frozen canonical seed42 GRU SHA
`2ebac71d897920233c8399a38e1f16fce24075cfa722c7d8e588ab14e564100e`.
Same immutable pool8192TRAIN/128DEV/32report. Planned old8 scope was corrected below:4 in pool, all8 audited separately on CPU.
Research initialized-polyphase40 contract, NOT generic learned semantic latents.

A: mean geometry +0.1 sampled geometry + bounded pen.
B: same plus sampled CTC with frozen GRU. Geometry matches target point positions
and within-true-stroke first differences (inside-G coefficient0.20471838744633777);
not generic smoothing, physical velocity or curvature. No KL/style/GMM, dropout,
augmentation, labels, RDP or architecture changes beyond explicit reader adapter.

Actual VAE.forward→get_ocr_loss supports an opt-in point-mask-aware adapter.
Explicit original real-point lengths—not predicted EOC—determine four-point OCR
frames. A fake early EOC cannot hide later points. Frozen GRU uses cuDNN training
reserves with dropout0 only when input backward is needed; GPU preflight checks
eval-logit parity and finite input gradients before any update. Reader weights
remain frozen and reader eval/dropout flags are restored. Dedicated research
checkpoint loader required; standard loader rejects the new contract.

200 updates per arm initially (CLI50–400), restored source AdamW moments, body
LR1e-7/posterior1e-3, readout and style frozen, existing pen-variance weight/moment
protection retained. Physical raw batch1 /accumulation8; only OCR latents batch8.
Seed4042 full TRAIN schedule; seed7042 paired original-shape posterior noise.
Geometry uses verified eager-equivalent CUDA graphs, same exact minimal lengths.
A also evaluates sampled CTC without optimizing it. CTC weight initially targets
10% of TRAIN32-probe aggregate encoder+conv_mu geometry gradient, capped0.1;
calibration seed6042, fixed across arms, never DEV.

Evaluate every50 updates on TRAIN32probe+DEV128+report32+old8 union, all20 fixed
posterior draws. Per-line gates: mean axis max(4×reference,0.0005modelunits); each
sampled axis max(1.2×same-draw reference,0.0001); mean turn/corner/shallow p90
max(1.5×reference,1degree), sampled max(1.2×same-draw reference,1degree); exact
pen F1/final EOC, no false internal EOC. TRAIN/old8 gate can stop safely; DEV/report
only report, never choose or stop. Common selection uses TRAIN32 CER then CTC then
geometry among gate-passing checkpoints, including step0. Save selected AND final
metrics/checkpoints, so a baseline-selected result cannot hide later regression.

One T4, sequential arms,1800s loop/eval perarm,5400s function timeout,retries0.
No new full8352 post-update or generative-readiness claim from this bounded test.
CPU reports independently reload selected codecs/readers and compare mean XY/pen/
transcripts; CPU/CUDA posterior noise streams are not bitwise paired.

```sh
venv/bin/modal run --detach modal_ocr_joint.py --train --steps 200 \
 --pool-sha d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9
```

## Completed matched frozen-GRU joint test (2026-10-07)

**Compatibility passed; OCR benefit failed. No joint checkpoint promotion.**
Both T4 arms completed200updates. Same source codec, restored AdamW moments,
TRAIN schedule(seed4042), explicit posterior noise(seed7042) and final RNG hashes.
Physical raw batch1/accum8; only frozen OCR uses latent batch8. BodyLR1e-7,
posteriorLR1e-3; readout/style frozen, pen variance weights/moments protected.
No KL/style/GMM/dropout/augmentation. Objective:
`G(mu)+0.1*G(z)+0.02099049935353879*boundedPen`, where
`G=pointMSE+0.20471838744633777*within-true-stroke target first-differenceMSE`.
B additionally uses sampled frozen-GRU CTC. These index differences are neither
physical velocity nor geometric curvature, and no generic smoothing was added.

Initial TRAIN32probe aggregate encoder+conv_mu gradient norms: geometry
0.00018073410319630057, CTC0.9888355731964111. Fixed B coefficient
1.8277467770711117e-5 gives the predeclared10% gradient fraction; cosine0.1178598.
Reader CUDA logits match eval exactly, input gradient finite(L2~0.043104),
reader weights/gradients remain frozen. Geometry eager/CUDA gradient relative
error<1e-12, term difference0. No update hits clip5. Loop+evaluation138.830s(A),
147.304s(B), excluding startup/capture/CPU reporting/billing.

TRAIN-only selector chooses A200 and B0 (source). B200 is preserved and shown,
not hidden behind unchanged selected B0. DEV/report never choose or stop.
DEV128 mean results (X/Y in modelunits, turn = average per-line p90 error degrees):

| Codec | CER | X RMSE | Y RMSE | Turn | Pen F1 |
| --- | --- | --- | --- | --- | --- |
| Source | 8.6753247% | 0.000007121 | 0.000005656 | 0.03891 | 1 |
| A200 geometry | 8.6753247% | 0.000005887 | 0.000006499 | 0.05788 | 1 |
| B200 + frozen OCR | 8.7012987% | 0.000007077 | 0.000003147 | 0.02533 | 1 |

B adds one DEV character error; these tiny geometry changes are not visibly
meaningful. All reported mean/every20draw per-line XY/turn/corner/shallow/pen
compatibility gates pass. Scope correction to the predeclaration above: only
4original8 curves occur in the immutable reader pool (c08,e08,p08,l10). GPU
union196lines = TRAINprobe32+DEV128+report32+4curves. Original `a07-421z-02`
("h in hope"), a07-03,k07,h05 are outside the pool, not silently added to it.
Supplemental CPU audit reloads all original8 from immutable originaltinyTRAINh5,
source/Aselected/Bselected/finalB,20paired CPUdraws; never training/selection/stop.
Independent main CPU reload covers196SELECTEDmeans, not20CPUdraws for all196.
Mean CPU/GPU XY differences<=9.54e-6, zero pen/transcript differences.

### Sampled improvement is largely posterior contraction, not robustness

DEV20draws: source X/Y0.000582251/0.000584779, turn3.33354degrees;
A2000.000409145/0.000410972, turn2.14006;
B2000.000409027/0.000410870, turn2.13892. Approximately30% less point error.
But KL was OFF. CPU TRAIN32+DEV128 posterior audit finds DEV mean XYsigma
0.000476659→0.000291562(A)/0.000291467(B), ~39% lower; valid-element KL
1.092986→1.127987/1.127998 (~3.2% higher).

Controlled legacy8×20 paired CPUnoise, keeping the SAME source sigma in every
version: source X/Y0.000500623/0.000506794, turn2.53676degrees;
A2000.000500618/0.000506726, turn2.53704;
B2000.000500646/0.000506704, turn2.53690. Effectively unchanged. With each
learned sigma, A/B X/Y~0.0003457/0.0003505, turn~1.559degrees. Thus most
sampled gain is shrinking posterior uncertainty, NOT a demonstrated stronger
decoder at the same perturbation or improved prior matching. All pen states exact.
This fixed-sigma diagnostic is descriptive only, not checkpoint selection.

Selected source/B0/A200 latent-versus-rendered frozen-reader audit on160probe+DEV
means finds unchanged CER, nearly equal CTC, gradient energy~97.2%XY/2.8%pen/
unused0. No evidence these SELECTED checkpoints exploit hidden pen amplitudes;
finalB200 was NOT included, and this is not a general proof against hidden cues.

Manual review: original8 selected mean full-line galleries, finalA200/B200 full
mean and source-worst20draw galleries, named c/h closeups (means and same-draw
samples), DEV page7, report page2, learning curves. Target corners/hooks survive;
no new spikes seen in inspected views. The c/h targets themselves remain polygonal
from IAM/RDP. All196 are quantified/gallery-generated, not all manually inspected.
Prior source/codec is already near-lossless; this experiment does not make drawings
visibly better. Residual reader errors do not establish a bad dataset.

### Implementation, artifacts and limits

19 new tests (297 total, root and fork). Safe opt-in differentiable adapter tested
through actual VAE.forward→get_ocr_loss→encoder backward; explicit real-point
lengths mandatory, predicted EOC cannot hide the transcript. Research OCR
checkpoints rejected by ordinary loaders. Legacy standalone reader behavior,
ordinary Chinese/English VAE paths and default configs unchanged. Dedicated
initialized-polyphase40 transport contract, NOT opaque semantic VAE latents.

Study root on Modal Volume `diffink-data` / local ignored `data/` mirror:
`checkpoints/iam_ocr_joint_study/20261007-114143/`.
- immutable completed report: `report/index.html`;
- explicit finalB, sampled galleries, fixed-sigma diagnosis, provenance:
  `diagnostics/engineering-review/index.html`;
- sources/configs/per-update logs/selected AND final checkpoints under study root;
- `diagnostics/posterior-stats-local.json`, `fixed-source-sigma.json`,
  `latent-vs-rendered-reader.json`, `legacy-eight/summary.json`;
- original GPU sources preserved. Two reviewed post-GPU changes only: explicit
  excluded-curve scope metadata, and earlier missing-pointmask rejection. Actual
  run supplied explicit valid masks; objective/sample set unchanged.

One CPU render was preempted/restarted then deliberately stopped to avoid repeated
expensive work; a second completed evaluation but failed in diagnostic writing on
missing `greedy_ctc` import. Final CPU retry succeeded. Source/error logs preserved,
import fixed with real-reader reporter regression, checkpoint/source/data-bound
completed CPU-stage cache added; no GPU retraining or overwritten as-run artifacts.
All four study Modal apps stopped/tasks0 at final audit; do not stop unrelated apps.

Keep original codec and canonical standalone GRU, not jointB. Next useful study:
bounded deliberate KL/prior-versus-reconstruction tradeoff with fixed-sigma control,
not blindly increasing OCR weight or unlocking InkDiT. Reused five DEV writers,
32report lines, prior codec prompt overlap, experimental normalization/EOC and
initialized transport still prevent paper-reproduction/general generation claims.

## Predeclared KL tradeoff with fixed-source uncertainty (2026-10-07)

Follow-up to frozen-GRU joint test; source remains original protected codec SHA
9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625, NOT a
posterior-contracted joint checkpoint. Same canonical GRU is evaluation-only.
Three matched200update arms: KL0,1e-7,1e-6. Previous1e-5 already degraded sampled
curves, so do not repeat the obviously poor stronger setting. Original1e-6 source
was trained on192lines; this continuation uses pinned8192TRAIN schedule seed4042.

Same restored AdamW moments/bodyLR1e-7/posteriorLR1e-3; frozen readout/style/reader;
protected pen logvar weights/moments, trainable pen variance bias. Physicalbatch1,
accum8, explicit same noise(seed7042) across arms. Geometry and bounded pen objective
unchanged: G(mu)+0.1G(z)+PEN*boundedPen, G=pointMSE+DELTA_WEIGHT*target first
index-differenceMSE. KL uses corrected valid-channel×latent-time normalization.
No GMM/OCR/style/dropout/augmentation objective or architectural change. Initial
TRAIN32 raw geometry-versus-KL gradient norms/cosines recorded per body/posterior
group; coefficients are a bounded prior tradeoff, not tuned on DEV.

Evaluate0/100/200: TRAINprobe32+DEV128+report32+all original8 =200unique lines.
Four old8 outside reader pool loaded from immutable originaltinyTRAINh5 ONLY for
evaluation; never inserted into TRAIN schedule. For every line,20 paired draws
under BOTH its current posterior sigma AND fixed source sigma, with identical
original-shape epsilon across versions/policies. Mean-location changes permitted,
source uncertainty held fixed for the diagnostic. Save source std tensor/SHA,
configs, as-run sources, selected AND final checkpoints, metrics and trajectories.

Gates are the joint study's per-line mean/sampled XY/turn/corner/shallow/pen gates
for BOTH noise policies; TRAINprobe/original8 can stop, DEV/report cannot. Same
TRAIN-only selector minimizes corrected TRAINprobe KL then own sampledXY among
gate-passing checkpoints, including0. KL0 may legitimately select0 even if actual
sampled geometry improves; always show final separately. Fixed-sigma diagnostic
cannot be mistaken for samples of the current learned posterior. No full8352 or
semantic/generative-readiness claim from this bounded test. Prior closeness alone
is not semantic regularization or proof diffusion generation will work.

One detached T4 job, sequential3arms,1800s loop/eval limit perarm/7200s function,
retries0/maxcontainers1. CPU independent reload/marker-free visual review follows;
all8 named curves and final arms shown, no selected0 hiding later failures.

## Completed KL / fixed-source-sigma tradeoff (2026-10-07)

All three matched T4 arms reached200updates. At the final evaluation each hit a
predeclared TRAIN/named corner gate, so the stop reason is a geometry gate even
though the full update budget completed. No new checkpoint is promoted; retain
original codec/canonical standaloneGRU. This is a useful negative result plus a
causal diagnosis, not a visually improved production VAE or paper reproduction.

Same source/restored optimizer/full8192TRAIN schedule/noise; all batch/noise/RNG
pairing and frozen-reader/protected-weights checks pass. Initial gradient norms
reveal why a numerically tiny KL matters: body geometry0.000257483 versus raw
KL2.354502 (weighted1e-6 fraction0.00914, cosine−0.184); posterior geometry
7.60817e-8 versus rawKL0.07201185 (weighted1e-6 fraction0.9465, cosine−0.9418).
1e-7 contributes~9.5% of the variance-head gradient,1e-6~95%. A total-gradient
fraction alone would hide this important parameter-group imbalance. No clipping
on any of600updates. Loop/evaluation191.269/192.739/191.837s, excluding startup,
capture, report and billing. Saved default model/trainer/dataset behavior unchanged.

DEV128 final own-posterior20draws (mean per-line/per-draw errors):

| State | KL/valid element | Mean XYsigma | X RMSE | Y RMSE | Mean turnp90° |
| --- | --- | --- | --- | --- | --- |
| Source | 1.0929865 | 0.000476659 | 0.000583650 | 0.000584415 | 3.32585 |
| KL0/200 | 1.1279872 | 0.000291562 | 0.000410284 | 0.000410495 | 2.13166 |
| KL1e-7/200 | 1.1236696 | 0.000305396 | 0.000421607 | 0.000421846 | 2.21860 |
| KL1e-6/200 | 1.0900645 | 0.000465288 | 0.000547097 | 0.000547085 | 3.18790 |

The two lower weights again mostly contract uncertainty and WORSEN KL.1e-6
slightly lowers KL(~0.27%) and sampled point error(~6.3%), without broad variance
collapse: XYsigma mean~2.4% lower, median0.000355220→0.000370239. XY KL
contribution0.725733→0.723623; pen0.367254→0.366442; unused~0. Small gains in
both active subsets, not a large prior/semantic-latent breakthrough. Mean XY
RMSE stays~6e-6; mean pen states exact. DEV meanCER unchanged8.6753% for0/1e-7,
1e-6final8.7013% (one more error); no OCR objective trained. Own sampled CER is
also reported but tiny changes are not recognized as meaningful reader improvements.

Holding the SAME source sigma/epsilon in every version leaves sampled X/Y around
0.000584/0.000584 and turn3.33degrees; no robustness gain. Final gates fail7/7/8
fixed-sigma corner/shallow checks for0/1e-7/1e-6;1e-6 additionally fails25 own
noise angle checks. These are metric failures (not necessarily25distinct lines),
not pen/meanXY failures. TRAIN-only selector retains0 forKL0/1e-7 and100for1e-6.
The selected1e-6/100 also has DEV/report angle failures; no promotion on the basis
of TRAIN eligibility. Preserve/show ALLfinal checkpoints, not just selected ones.
Do not loosen the preregistered gates after seeing these results.

Examples of actual failures, not only source-worst/handpicked previews:
- `k06-499z-02`, GPUfixedσ draw14: cornerp90 3.56664→4.46024degrees, bound4.27997;
  paired source/final trajectory RMS shift3.12e-5. Source/final XRMSE~0.000578/
  0.000574. Thus low aggregate point error can hide a local direction change.
- `p03-331z-03`, fixedσ draw1: cornerp90 3.31024→4.12494, bound3.97229.
- `b07-568z-04`, fixedσ draw14: shallowp90 1.40761→1.69450, bound1.68913
  (barely beyond gate). Do not exaggerate this as an obvious visual deformation.
- Supplemental CPU all8×20:0/1e-7 pass both noise-policy gates.1e-6 final fails
  one own-noise corner check, `a07-421z-02` draw13; fixed-sigma gate passes.
  Source-worst gallery draw17 would not show that failure, so actual draw13 is
  included separately. These CPU epsilons are NOT GPU-paired.

Independent CPU reload verifies1000means: source plus all3final and1e-6selected100,
200lines each. Max CPU/GPU XYdiff9.54e-6, zero pen/transcript differences. Main
report separately renders all200 means and original8 mean/source-worst20draw c/h
closeups. Manual review: all8 full means, c/h mean/own/fixed-noise closeups,
actual failed draw full lines and worst-increase target-corner closeups. Drawings
remain visually near-lossless in these views; no return of the old obvious spikes.
Those remaining target hooks/polygonality are deliberately preserved, not smoothed.
Not all200gallery lines or all4000draws were manually reviewed.

### CPU causal probe: nominally unused latent noise leaks through conv decoder

"Frozen readout" in optimizer_groups freezes `transformer_decoder`, NOT
`model.decoder` (the convolutional upsampling stack), which was still in the
body group atLR1e-7. Even tiny body updates can change its noise response.

Posthoc CPU module swaps on15known curves/failure examples,20paired CPUeps each,
with identical fixedSOURCE sigma; no training/selection/promotion:
- FullKL0final versus source: mean displacementRMS8.37e-6; sampled3.035e-5.
- SOURCEencoder/mu plus UPDATEDconvdecoder: mean8.44e-6; sampled3.049e-5.
- UPDATEDencoder/mu plus SOURCEconvdecoder: mean9.06e-6; sampled9.06e-6.
Thus swapping only convdecoder reproduces nearly all the additional stochastic
source-to-final drift on this diagnostic set; encoder/mu changes are chiefly a
small fixed location shift. This is causal module-swap evidence for this scope,
not a universal statement about every dataset/model.

Zeroing ONLY noise in channels>=40, retaining the same active-field noise and
all means, reduces full source-to-final sampled drift3.035e-5→8.377e-6, essentially
mean-shift level. SOURCE itself already has unused-noise leakageRMS2.843e-5;
final2.923e-5. The *pattern* changes more than its aggregate amplitude. These
344channels are nominally unused only in this explicit initialized-polyphase40
codec, NOT in a normal384-d semantic VAE. Do not mask generic learned latents.
This is a contributor to the small fixed-noise corner failures, not evidence
that unused noise explains all active-coordinate sampling error or the historical
large mean-geometry artifacts. Module-swap outputs are diagnostics, NOT promoted
surgical checkpoints. Probe completed/serialized all15×8policy/model×21means/draws;
wrapper later reportedexit143, full artifact structure/SHAs were checked. No rerun
or hidden partial result; preserve the abnormal wrapper status in provenance.

Next justified comparison: freeze the COMPLETE geometry decoder (conv+Transformer)
when calibrating posterior noise, possibly encoder/mu too for a fully isolated
uncertainty refit. Also compare explicit unused-channel masking for this transport
contract only. Do NOT blindly increase KL, generic smoothing, change RDP, turn on
style/CTC or declare InkDiT readiness. These simpler causal hypotheses now have
more support than architecture-capacity or "bad IAM dataset" explanations.

Study root on Volume `diffink-data` and local ignored `data/`:
`checkpoints/iam_kl_tradeoff/20261007-132418/`.
- immutable CPU report: `report/index.html` (metrics-table/configs/SHAs/gates);
- actual failed draws, causal explanation, provenance: `diagnostics/engineering-review/index.html`;
- causal module swaps: `diagnostics/component-swap/summary.json` + exact probe source;
- selected/final checkpoints, as-run sources, sourceSTDtensor/SHA, every-update log.
310 tests root/fork (13new). Same experimental normalization/EOC, reusedDEV5writers,
report32prior codec overlap and initialized transport limitations remain explicit.
All200original/heldout means quantified; no new full8352 stochastic certification.

## Preregistered posterior-isolation follow-up (2026-10-07)

Source remains `iam_codec_kl_study/20261007-012331/pen_bias_kl1e-6` SHA
`9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`.
Compare two sequential bounded T4 arms,200 updates each: **joint** restores
previous bodyLR1e-7/posteriorLR1e-3 behavior; **posterior_only** restores those
same groups/moments first, then freezes EVERYTHING except `conv_logvar`.
Clear frozen `.grad=None`, preserve/check frozen Adam moments/steps, all codec
buffers and weight tensors. Decoder input autograd stays active for posterior
learning. Pen variance feature-weight rows retain previous moment+gradient
protection. No latent-channel masking, new architecture or generic smoothing.

Both use betaKL1e-6, mean geometry1 + sampled geometry0.1 + bounded pen
0.02099049935353879; geometry is XY MSE +0.20471838744633777 within-true-stroke
first-index-difference matching. Same immutable8192TRAIN schedule(seed4042),
effective8/physical1, sampled-noise seed7042, restored source AdamW .9/.99,wd0,
clip5; no GMM/OCR/style/dropout/augmentation. Fixed frozen GRU evaluates only.
Evaluate0/100/200 on same200unique lines and20paired own/fixedSOURCE-sigma draws.
TRAIN-only prior-score selector and unchanged per-line/per-draw curve/pen gates;
DEV/report cannot select or stop. Original8including4outsidepool are evaluation
only. No automatic production promotion.

Expected isolation control: posterior-only latent-mean AND fixedSOURCEsigma
reconstructions/reader outputs should be **exactly unchanged**, alongside frozen
weights/optimizer states. This is NOT an improvement in fixed-perturbation
robustness. Own-sigma changes may still fail local angle gates; retain failures,
not just selected checkpoints. Independent CPU reload and all8 marker-free
comparisons follow. This initialized polyphase40 experiment does NOT establish
learned semantic latents, English paper reproduction or InkDiT readiness.


### Completed posterior isolation: decoder drift removed, two rare variance checks remain

Study `checkpoints/iam_posterior_isolation/20261007-142410/` on `diffink-data`
and local ignored `data/`. Same preregistered source/reader/pool SHA,200updates,
TRAIN schedule/noise/restored optimizer/objective as above. Two matched arms:
joint body+posterior versus full-codec freeze except `conv_logvar`. Restoring
before freezing is essential; stale gradients are cleared; body Adam moments
AND per-parameter steps stay unchanged. Frozen decoder still propagates input
gradients to posterior. Pen variance feature-weight rows retain protection.
No architecture, raw data, RDP, latent-channel mask, smoothing or OCR/style/GMM
change. KL is deliberately tested at1e-6, not enabled incidentally.

| DEV, GPU20paired own-sigma draws | Source | Joint200 | Posterior-only200 |
|---|---:|---:|---:|
| Corrected KL / valid element |1.0929865|1.0900645|1.0901890|
| Sampled X RMSE |0.000583650|0.000547099|0.000546790|
| Sampled Y RMSE |0.000584415|0.000547085|0.000546737|
| Mean per-line turnp90° |3.325850|3.187912|3.184338|
| FixedSOURCEsigma turnp90° |3.325850|3.331469|3.325850|
| All200 own/fixed angle-check failures |0/0|23/8|2/0|

Posterior-only X/Y error improves6.32%/6.45%, turnp90 improves4.25%, corrected
KL improves0.26%. These are modest uncertainty calibration gains, NOT a new
mean-geometry visual breakthrough. Frozen-source-sigma outputs are unchanged,
NOT improved fixed-perturbation robustness. All means, pens and frozen-reader
mean outputs unchanged. DEV mean CER8.6753%; exact pen F1/EOC under all evaluated
draws. Neither arm clips any update. Joint own failures23across17lines versus
posterior-only2across2lines; counts refer to metric checks, not visible defects.

Independent serializer check: all non-logvar checkpoint state tensors and body
optimizer moments/steps equal. GPU saved-array check compares
8800 trajectory pairs (all200means+all20fixedσdraws at100/200):
EXACT equality including pen states. Runtime checks also protect codec buffers,
frozen gradient=None, reader, source checkpoint and restored group structure.
800 independent CPU means: source, both finals and joint selected100. Maximum
CPU/GPU XYdifference 9.53674316e-06; no pen/transcript differences. All8original
curves independently evaluated20CPU own/fixedσ draws in source/bothfinal; these
CPU epsilons are NOT GPU-paired. CPU legacy gate results:
`{'joint-final': False, 'posterior_only-final': True}`.

Actual remaining GPU own-sigma failures (both DEV, original gates unchanged):
- `j10-499z-04`,draw17: cornerp90 2.044916→2.500154°, bound2.453899°;
  worst-increase local corner2.116833→2.651301°. Adjacent target lengths
  0.007143/0.047445model units. Source/final point RMSshift4.92e-5.
- `p03-331z-04`,draw0: cornerp90 2.912100→3.550630°, bound3.494520°;
  local6.013837→7.047689°, target lengths0.006249/0.088001;
  source/final point RMSshift3.98e-5.
These exceed p90bounds by~0.05°, not large synthetic spikes. Mean geometry is
unchanged; fixed-sigma is identical, so changed posterior variance is the cause
of these remaining paired-output differences. Short segments make angular
metrics sensitive; nonuniform index differences are not physical velocity or
curvature. Preserve target hooks/corners; no generic blur/RDP change justified.
Manual marker-free review: both actual failed draws full+worst-increase corner
closeups, all8full mean comparisons and named c/h mean/source-worst own/fixed
closeups. Near-lossless mean appearance; no return of obvious old jaggedness.
All200quantified and gallery generated; not all manually reviewed.

TRAIN selector retains joint100 and posterior200. Posterior200 TRAIN/named
passes and budget completes, but all200report audit catches2DEV failures.
**No production promotion**; do not secretly switch to posterior100 using DEV,
even though its audited gates pass. This establishes full-codec freezing as a
safer uncertainty-calibration method, not that a384-d semantic/generative prior
has been learned. Tiny KL improvement does not establish InkDiT readiness.
Same initialized polyphase40 transport, reused DEV writers/report overlap,
experimental normalization/line-final EOC and no full8352stochastic certification.

Next justified follow-up is predeclared TRAIN-only short/variance-trust-region
posterior calibration or a gradient-calibrated relative-segment/tangent
geometry objective for sampled noise, still freezing full geometry. Avoid
blindly increasing KL, generic smoothing or incidental CTC/style. Storage is
near500kVolume inode limit: pack large per-draw artifacts before another study.

Artifacts:
- `report/index.html`: independent reload, paired guards, all200gallery,
  all8mean/own/fixed-source-sigma marker-free c/h closeups;
- `diagnostics/engineering-review/index.html`: causal explanation, actual
  two failures, full configurations/metrics/provenance links;
- `joint/` and `posterior_only/`: source/selected/final checkpoints, configs,
  every-update batches/noise SHAs, graph parity, gradients, per-draw metrics;
- immutable as-run `source-code/`, source STD tensor/SHA, manifest identities.
322tests root/fork (12new): actual decoder-input autograd, stale gradients,
frozen Adam moments/steps/weights, exact mean/fixed-noise arrays, serialized
checkpoint guards, paired study/report contracts, empty body gradient group.

## Preregistered bounded generation gate (2026-10-07)

Move from curve reconstruction to text control. Keep original faithful source
`iam_codec_kl_study/20261007-012331/pen_bias_kl1e-6` SHA9c53f68e… frozen;
no posterior200 promotion, no source weights/optimizer changes. Frozen canonical
BiGRU remains evaluation-only. Prototype is **NOT released InkDiT/paper config**:
4×128-wide full-text cross-attention blocks,4heads, FF4×, dropout0, all384latent
channels, x0 cosine diffusion1000steps, deterministic DDIM50 sampling.
Full transcript memory (not aligned/truncated to latentT); an always-valid NULL
key makes text dropout independent of transcript/padding length. Native PyTorch
attention, no unneeded audio/x-transformers dependencies. Existing official
DiT/Diffusion/modules/trainers remain unchanged. CPU audit found zero current
pool transcripts actually truncated by upstream latentT: do not claim this is
a demonstrated cause of prior curve trouble. Upstream no-prefix CFG retains
text even in its "unconditional" branch; this prototype explicitly drops TEXT.

Generator scope fixed BEFORE training: writer10160,32SHA-sorted TRAIN lines
(seed7314) from immutable8192TRAIN reader pool, entire form g09-301 held out
(8lines/unseen transcripts). TRAIN/held-out texts and forms disjoint; reserved
25test and5DEV writers excluded. These images ARE in the previously trained
reader corpus. Thus generator unseen prompts, NOT an independent OCR/English
benchmark. One writer only; no writer-control claim. TRAIN-only per-channel
mean/populationstd floor0.1 whitening, inverse before frozen decoder; no channel
masking, no prefix/target trajectory input to sampler. Use latent MEAN targets
intentionally for the first mechanics test; NOT sampled-z paper reproduction.

Two8000update sequential T4 arms: text vs no_text, identical initial weights,
TRAIN32 minibatch8 schedule(seed5142), Gaussian noise/t/drop draws(seed6142).
Text arm null conditioning10%; no_text arm consumes same RNG draws, alwaysNULL.
AdamW3e-4,betas .9/.99,wd.01,clip1,FP32,no augmentation. No VAE KL/CTC/style/GMM
training. Max1800s per arm, finite gradients/objective mandatory. Evaluate
0/4000/8000; TRAIN fixed-terminal-noise activeXY+pen denoising error selects,
held-out prompts never select/stop. Full-noise generation with matched per-ID
initial epsilon, two seeds9142/9143; correct/null/swapped text and true textCFG3.
Save ALL results, including baseline. No automatic production promotion.

Important limitation: both models receive oracle ceil(trueN/8) duration.
Evaluate actual predicted first-EOC termination AND oracle-window OCR separately;
never force final EOC or true pen states. Frozen reader scores actually DECODED
XY+predictedhard pen fields, repacked only for its explicit research contract,
not arbitrary semantic z. TRAIN trajectory errors measure memorization; unseen
trajectory need not match one reference point-for-point. Reader CER is a proxy;
marker-free visual legibility/text response is the primary meaningful gate.

Artifacts packed into source/evaluation HDF5 files (including actual initial GPU
noise), avoiding another50kVolume inodes. Independent CPU sampler reload MUST
use saved GPUepsilon, not a CPU RNG with the same integer seed. Runtime digest
checks freeze codec/reader. Preserve configs/data/noise/weights/sources/metrics,
report all40lines, same-noise text ablations and both seeds for held-out prompts.
Do not confuse lower denoising loss with valid generated handwriting. Outcomes
may establish tiny-set text memorization only, not unseen text generation.

## Resource monitoring and preregistered generation refinement (2026-10-07)

Idle CPU cores alone are not a defect. Add dependency-free process-tree/thread
CPU-seconds per wall-second, summedRSS, contextual cgroup usage/throttling/limits,
NVIDIA device busy/memoryactivity/VRAM/power and phase-tagged step latency.
Cgroup scope is unverified: never infer allocated-core saturation from host or
shared cgroup counters. CPU request4 is a reservation, not a hard4-core limit.
Thirty-second homogeneous TRAIN windows trigger console+JSONL recommendations;
eval/startup/CPU-only phases don't trigger idle-GPU warnings. No automatic
resource scaling or invented external notification destination. Persist
`resources.jsonl`, `alerts.jsonl`, `resource-summary.json` beside every next run.

First bounded throughput study:5disposable45s warmed cases from the identical
text8000checkpoint: live4threads/batch8; cached4/batch8; cached/no-hash4/batch8;
cached/no-hash1/batch8; cached/no-hash1/batch32. Same model/FP32/dataset/T4,
restored parent Adam,LR1e-4. Last case changes effective batch and is throughput
ONLY, not a matched quality result. Separate cache and per-step noise-digest
CPU synchronization interventions. GPU boundary synchronize; update wall timing
ends after the ordinary loss/gradient scalar reads. Benchmark does not promote
any disposable weights. Existing source32 mean-latents fit in GPU memory;
corpus-sized caching is not assumed safe.

Generation report images inspected: text controls familiar TRAIN prompts, but
TRAIN is still jagged/inexact and unseen whole-form text is mostly illegible.
Hypothesis: per-channel whitening gives X~14times Y physical error tolerance,
so uniform whitened MSE is poorly aligned with local model-space handwriting.
Preregister3matched2000update continuations from text8000
SHA`ddf0720e29c8692eea2bd31a2f0f28444dfa230fee271ceaab0a48080f2bfceb`:
A uniform384-channel whitened MSE; B add physicalXY anchor; C add sameXY anchor
plus within-true-stroke target segment-difference matching. No generic smoothing;
index differences are not physical velocity/curvature. Only valid for this
explicit polyphase40 transport; forbidden for generic semantic VAE channels.
Frozen faithful codec/reader, same source32/held-form8/whitening/oracle length,
restored Adam/RNG, exact schedule offset8000,LR1e-4,batch8,dropout0,no other
objectives. Fixed auxiliary coefficients calibrated on first4TRAIN minibatches
(seed11242): full-model gradient norms,XY25% and segment10% of base gradient.
Per-arm600s safety limit; terminal TRAIN denoising selects, never unseen CER.
Evaluate0/2000 with original correct/null/swapped/CFG3 policies/two savedGPUnoise
seeds and marker-free renders. No automatic promotion. Resource telemetry on.

## Completed initial generation gate

`checkpoints/iam_generation_gate/20261007-153404/report/index.html`.
Both matched arms reached8000updates; TRAIN-only selector chose8000. Frozen
codec preflight40/40 passed (source reader0errors). Sampling uses saved realGPU
noise for independentCPUreload (no pen/transcript discrepancy; maxXY drift
6.20e-5 for text,1.62e-5 no_text). All as-run source/reporter bytes preserved.

Free-stop CER: text TRAIN35.58%, NULL84.99%, swapped86.11%; matched no_text
TRAIN72.44%. Held-form8correct86.21%; no_text85.80%. No exact generated lines.
CFG3 hurts TRAIN72.01%, held81.69% still fails. At terminal-noise TRAIN,
text standardizedXYMSE0.00704 versus no_text0.24781. **Text dependency exists,
but readable generation/unseen composition does NOT pass.** Representative
4TRAIN/4held gallery rows visibly remain jagged/illegible. Not the old codec
failure: exact frozen source geometry and source reader pass.

Oracle length caveat:16distinctTRAINlatentlengths,6singleton lines outof32;
length can leak line identity. No unseen characters, but only32lines aren't
adequate evidence of compositional generation. Reader is corpus-familiar;
CER is not an independent benchmark. TRAIN correct meanX/YRMSE0.12895/0.01304,
penF1perfect; per-channel meanX/Ywhiteningstd2.89585/0.20491 (ratio14.13).
Scale/objective mismatch is a plausible contributor, not yet a unique cause.

Resource study `iam_resource_benchmark/20261007-155649/report/index.html`
established caching/no per-step noiseGPUcopy +14.7% examples/s; batch32 is
4.35times original batch8 throughput. CPUprocessusage~0.98core, GPUbusy33–39%;
4→1PyTorchCPUthreads did not improve speed. See `RESOURCE_MONITORING.md` for
raw metrics, alert scope, profiling/scaling recommendations and limitations.
No automaticscaling; never increase cores just because three are idle.

Continuation as-run sources are preserved separately from subsequent hygiene
updates. Initial continuation config inherited parent schedule/initial-state
SHA and1800s wall-cap fields; actual runner uses the offset8000 schedule and
600s TRAIN safety cap. Actual pairing verified from per-update sampleIDs and
periodic/final RNGSHAs, parent checkpoint SHA and as-run source. Future runner
now serializes the effective600s cap, actual continuation schedule/initial-state
SHA and inherited RNGSHA. These metadata-only fixes do not alter completed
models or retroactively rewrite completed config files. Reusable monitor also
adds storage-failure guards and keeps I/O outside training timing locks.

## Completed monitored generation refinement

Artifacts: `checkpoints/iam_generation_refinement/20261007-160458/`;
`report/index.html` all32TRAIN/8held marker-free comparisons; both held noise
seeds, CPUreload; `report/resources.html` measured utilization/alerts. All3arms
2000updates; same parent step8000,Adam moments,minibatch schedule/inheritedGPU
noise state. Checks pass: unchanged frozen codec/reader, identical periodic/final
RNGSHA and every-update sampleIDs. Auxiliary calibratedXYweight0.001813513526,
segmentweight0.004050897769 (full-model gradient fractions25%/10%,not arbitrary
smoothing). PhysicalXYlookup verified against source targets(maxerror<0.0005).
IndependentCPUfull-noise sampler2TRAIN/2held per arm: maxXYdifference1.72e-5,
0pen discrepancies, all free-stop reader transcripts identical.

Correct-text TRAIN,means over32lines×2seeds:

| Model | FreeCER | XRMSE | YRMSE | segmentdiff vectorRMSE | seconddiff vectorRMSE | tangentp90mean° | turnp90mean° |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent8000 |35.58%|0.128951|0.013042|0.068509|0.098548|106.28|131.83|
| A uniform continuation |25.69%|0.082056|0.008036|0.057668|0.081414|96.94|124.75|
| B +physicalXY |23.99%|0.079985|0.008076|0.056068|0.079620|96.12|122.94|
| C +sameXY+targetsegments |24.68%|0.079459|0.008069|0.049041|0.066949|89.85|116.73|

AllTRAINpenF1=1.0,0non-finalfalseEOC,no exact generated lines in any arm.
Derivative matching improves segment error15.0% and seconddifference17.8%
relative to matched A,with XRMSE3.2%better,YRMSE0.4%worse. It is an objective
contributor,not a complete cure. B has slightly better readerCER than C; don't
pretend all metrics agree. Angles are means of per-line p90 distributions,
not pooledp90; index differences are not physical velocity/curvature.
Held correct freeCER A85.60%,B84.36%,C85.60%: still fundamentally fails.

Visually reviewed all32TRAIN and all8held seed9142: continuation makes more
words recognizable, and targetmatching reduces some zigzags, but all3 still
have pointed/artificial curves and malformed letters. Held text remains largely
scribbles. Neither production generation nor paper reproduction passes.
No architecture/codec change, no source-weight promotion. The old codec's
lossless reconstruction does not prove this small denoiser can generalize text.

Monitored actual refinement phases:334–337examples/s,batch8; GPUbusy36–38%,
CPUtree0.95–0.96cores. Sustained lowGPU alerts were correctly emitted in all3
TRAIN phases and not in eval. No measured memory pressure. This is useful
iteration speed despite idle GPU capacity. Next optimization candidates:
CPU/CUDA profiler,CUDAgraphs/compilation/fused dispatch,matched larger-batch
learning study; increasing CPUreservation or moving toL4 is not yet supported.
Next quality candidates: stronger-but-controlled physical/tangent alignment,
terminal-noise weighting versus uniform timestep,broader text coverage and
explicit duration/style conditioning. Do not blindly keep tweaking the codec
or enable OCR/KL as a cure for unproven generator text composition.

363tests pass root/fork;41new versus322 previous. Tests cover text/null masking,
whitening/DDIM,CPUclock/resource attribution,phase/sustained/gap guards,noGPU,
ambiguous hostGPUs,memoryspikes,nonfatal sampler/storage failure,shutdown,
exact vectorized raggedcollation,polyphase contract guards,true-stroke/padding
exclusion and matching rather than indiscriminate smoothing.

## Preregistered conditioning/denoising/sampling isolation (2026-10-07)

Pause resource tuning; keep telemetry only. Initial CPU no-training check on
first8fixed TRAIN,actual saved GPUepsilon9142,step10000C model: direct zero-input
X/Y/segmentRMSE0.1060/0.01261/0.05709; terminalone-pass0.11285/0.01575/0.06526;
DDIM2≈0.07988/0.00808/0.04918; DDIM10/50/100all≈0.0800/0.00815/0.04924.
This preliminary latent-field geometry proxy suggests sampling-step count is
NOT the main jaggedness cause; iterative cleanup improves the terminal estimate.
Full frozen-codec/reader probe on all40/twoGPUnoise seeds follows.

Three matched4000update continuations from Cstep10000 checkpoint
`iam_generation_refinement/20261007-160458/physical_xy_segments/checkpoint-last.pt`
SHA`011adfc6050ee22677db31023183d913b9632c53d392674967054a1ac3959db4`:
A uniform original cosine denoising; B pureGaussian input,t999,one-pass regression;
C zeros,t999,deterministic text+oracle-duration regression. Same128×4architecture,
same384-channel targets/source32TRAIN/wholeform8held,full frozen codec/reader,
whitening,Adam moments/LR1e-4,exact schedule offset10000,identical inheritedGPU
RNG (epsilon/t/drop draws consumed by all arms),batch8,textdrop10%,noaugmentation.
Only modelinput/time distribution differs. B has NO tinyalpha999target signal;
C sees neither target nor noise. Native readouts A=DDIM50,B=terminal,C=zero.
Same inherited physicalXY/targetsegment auxiliary coefficients,not recalibrated.
No decoder/codec/reader training; no KL/CTC/style/incidental architecture change.

Per-arm900s TRAIN safety cap,4000updates; evaluate0/2000/4000 all40,zero/terminal/
DDIM50 plus nativeNULL/swapped text,matched seeds9142/9143. Source-only probe
addsDDIM2/10/100. Preserve genuine predictedEOC stop AND oracle-window scores,
predictedpenstates,localgeometry,alltrajectories/GPUepsilon packedHDF5. TRAIN/native
XRMSE+YRMSE+.25segmentRMSE+.1(1-minpenF1)selects; never heldCER. Directzero's two
seed labels are repeatedevaluations,NOTindependentdraws. Oracle duration remains
identity-leakage caveat; heldprompts are generator-unseen but reader-familiar.
No production/paper-promotion claim. If B/C get clean TRAIN while A doesn't,
conditioning/representation works and trainingdistribution is implicated. If
allremainpoor,look at conditional mapping/optimization,not more sampling steps.

## Evidence-driven follow-up: NULL-loss interference

First three4000update arms ended with nativeTRAINCER19.23%(uniform),26.55%
(terminal),17.84%(direct),versus sourceDDIM24.68%. Direct alone is not a cure;
heldnativeCER85.60/82.51/83.13% still fails. AllTRAINpenF1perfect. No sampler-
count optimization,architecture change or production promotion follows.

Exact CUDA RNG replay from the parent matches62periodic snapshots,allbatchIDs;
first provisional replay captured3999directrows while finalrowstillbuffered.
It is preserved,not overwritten. A complete final replay will verify4000rows.
With10%per-exampletextdrop,~56.6%batch8updates containNULLexamples. Direct
NULL-containing batches had meanwhitenedloss0.00606 versus0.00006045(~100×),
gradnorm0.01943 versus0.00344(~5.65×). Uniform and terminal show similar stronger
NULL-associated gradients. This is correlation,NOTcausalproof of interference.

Preregistertwo matched4000update direct-zero continuations from direct14000
SHA`561a96b6e0646fd25500b92a6f71925c82e961ab2e3885da9ea4af5c09f6774e`:
textdrop10%control versus0%intervention. Identical model/Adam/RNG/dataset/oracle
length/sourcecodec/reader/whitening/LR1e-4/batch8/XY+segmentweights,zeros/t999.
Consume identicalGaussian/t/randomdropdraws even when applieddropmask iszero;
logdrawn andappliedNULLcounts. Exactscheduleoffset14000,TRAINonlyselector,
900sTRAINcap,eval0/2000/4000correct/NULL/swappedtext onall40. drop0NULLreadout
is untrained/OOD,NOTa valid unconditionalgenerator metric. This is a capacity/
optimization diagnostic,not abandoningCFG in a future generative model.

## Completed conditioning-path and text-dropout studies (2026-10-07)

**Decision:** keep resource telemetry, but prioritize generation fidelity rather
than more CPU/GPU-utilization tuning. Two bounded T4 studies completed. The
largest measured improvement came from conditional-training behavior, not
additional sampling steps. This is a tiny-set mechanics result, not usable
unseen-text generation or reproduction of the paper.

### Contracts and provenance

- Path study: `checkpoints/iam_generation_path/20261007-163247/`.
- Dropout study: `checkpoints/iam_generation_null/20261007-164542/`.
- Each has `report/index.html`, marker-free galleries for all 32 TRAIN and eight
  whole-form-held lines, `report/summary.json`, packed evaluation HDF5, initial/
  best/last checkpoints, optimizer/RNG states, logs and immutable source snapshots.
- Frozen codec: `iam_codec_kl_study/20261007-012331/pen_bias_kl1e-6/checkpoint-best.pt`,
  SHA `9c53f68e0f3797f837223f60e87de132293fe5b3389fdfd0fe6f2bebccea7625`.
  This is initialized **polyphase40 transport**, not a learned semantic InkVAE:
  the first 40 of 384 channels pack eight XY/pen points. Do not describe these
  experiments as fixing the original InkVAE or released InkDiT architecture.
- Frozen evaluation-only recurrent reader:
  `iam_ocr_recurrent_study/20261007-100011/bigru/head-best.pt`,
  SHA `2ebac71d897920233c8399a38e1f16fce24075cfa722c7d8e588ab14e564100e`.
  Source trajectories read perfectly; reader is corpus-familiar, not an
  independent unseen-corpus OCR benchmark.
- Writer 10160; 32 TRAIN lines, all eight lines of form g09-301 held out.
  Transcript/form splits disjoint; no unseen characters. Oracle duration
  `ceil(N/8)` is supplied, not predicted: 16 distinct TRAIN durations, six
  singleton lines can leak identity. No target/reference trajectory at inference.
- Standalone native-PyTorch prototype: four blocks, width 128, four heads,
  FF expansion 4, dropout zero; all 384 whitened latent-mean channels modeled.
  Whitening uses TRAIN only with std floor 0.1. No codec/reader training,
  KL, CTC, style, rotation, architecture or preprocessing changes.
- Genuine predicted pen states and first-EOC stopping used in renders/CER;
  oracle-window CER is also saved. Geometry compares full aligned valid points,
  separately from free-stop CER. Held-trajectory mismatch is not itself a
  definitive generation failure: valid writing is nonunique; reader and visual
  failure are the important evidence here.

### Study 1: conditioning versus terminal denoising versus sampling

Three matched 4,000-update continuations from step 10,000
`iam_generation_refinement/20261007-160458/physical_xy_segments/checkpoint-last.pt`,
SHA `011adfc6050ee22677db31023183d913b9632c53d392674967054a1ac3959db4`:

- **uniform:** original random-t cosine denoising, native DDIM50.
- **terminal:** exactly pure Gaussian input at t=999, native one-pass prediction;
  no residual `sqrt(alpha999)*target` contamination.
- **direct:** zero input at t=999, native deterministic text/duration regression.

Identical parent weights/Adam/RNG, batch 8, LR 1e-4, AdamW betas 0.9/0.99,
weight decay 0.01, clip 1, text dropout 10%, schedule seed 5142 offset 10000.
All consume identical epsilon/t/drop random draws, including unused draws.
Inherited auxiliary coefficients unchanged: physical XY 0.0018135135257448293,
within-target-stroke first-difference 0.004050897768969159. These were calibrated
previously by model-gradient magnitudes, not a generic smoothing penalty.
900-second TRAIN cap per arm; all reached 4,000 updates in 91.5–94.5 seconds of
training-loop time. Evaluate 0/2000/4000; native TRAIN-only checkpoint score:
`XRMSE + YRMSE + .25*segment_vector_RMSE + .1*(1-min_pen_F1)`.

Full no-training source readout sweep, TRAIN32 × two saved GPU-noise seeds:

| Readout | CER | X RMSE | Y RMSE | segment-vector RMSE |
|---|---:|---:|---:|---:|
| DDIM2 | 24.145% | .079514 | .007996 | .048977 |
| DDIM10 | 24.679% | .079445 | .008070 | .049049 |
| DDIM50 | 24.679% | .079459 | .008069 | .049041 |
| DDIM100 | 24.679% | .079458 | .008066 | .049041 |
| terminal one-pass | 36.325% | .111516 | .016031 | .064898 |
| zero one-pass | 29.487% | .099045 | .012124 | .056528 |

More DDIM steps are **not the dominant source-model limitation**. Iterative
cleanup helps compared with the terminal estimate, but 2–100 steps are nearly
identical in geometric error. This does not prove every future sampler behaves
this way.

Final, matched-budget native results (not a mixture of best-selected steps):

| Arm | TRAIN CER | X RMSE | Y RMSE | segment RMSE | second-diff RMSE | tangent p90 mean | held CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 19.231% | .066387 | .007878 | .043697 | .058174 | 81.77° | 85.597% |
| terminal | 26.549% | .090801 | .011861 | .054198 | .074962 | 96.25° | 82.510% |
| direct | 17.842% | .069871 | .006671 | .043414 | .058516 | 85.50° | 83.128% |

All TRAIN pen F1=1, no false EOC. Direct-only regression is not a cure. Native
TRAIN selection picked step 4000 for uniform/terminal and 2000 for direct. The
next study deliberately branches from **direct last step 14000**, not its
best-selected step 12000, keeping a fixed-budget parent explicit.
Independent CPU native reload (two TRAIN/two held per arm) max XY difference
2.6703e-5, zero pen/transcript discrepancies. Exact saved sampler epsilon,
paired minibatches/RNG and unchanged codec/reader checks pass.

### Diagnostic: NULL-associated losses

Exact CUDA dropout-draw replay matches all 4,000 minibatches and 63 periodic
RNG snapshots across the three arms. With per-example text dropout 10%, 2,262
of 4,000 batch-8 updates (56.55%) contain at least one NULL example. Direct arm:

| Batch group | whitened MSE | raw gradient norm |
|---|---:|---:|
| no NULL | .000060455 | .0034377 |
| contains NULL | .00606026 | .0194292 |

NULL-containing batches have ~100× loss and ~5.65× gradient norm. This is
**association**, not direct measurement of conflicting gradient directions.
Provisional `diagnostics/dropout-replay.json` captured 3999 direct rows while
its final log row was buffered. It is preserved. Complete replay is a new
`diagnostics/dropout-replay-final.json`, also copied to the second study's
`dropout-replay.json`; no completed diagnostic was silently replaced.

### Study 2: matched text-dropout ablation

Two 4,000-update continuations from identical direct step-14000 checkpoint:
`iam_generation_path/20261007-163247/direct/checkpoint-last.pt`,
SHA `561a96b6e0646fd25500b92a6f71925c82e961ab2e3885da9ea4af5c09f6774e`.

**drop10:** 10% applied text dropout. **drop0:** no applied text dropout. All
other settings above fixed, schedule offset 14000. Both consume the same
Gaussian/t/random-drop draws; applied mask is the intervention. Drawn/applied
NULL counts logged at every update. Both reached budget in ~100 seconds of
training-loop time; evaluation at 0/2000/4000. TRAIN-only selector picks 2000
for drop10, 4000 for drop0. Below compares **final step 4000 for both**:

| Metric | drop10 | drop0 |
|---|---:|---:|
| TRAIN CER | 14.103% | **0.962%** |
| exact transcripts, unique lines | 0/32 | **25/32** |
| X RMSE | .066220 | **.025967** |
| Y RMSE | .006617 | **.001492** |
| target segment-vector RMSE | .039003 | **.024845** |
| target second-difference RMSE | .051446 | **.033799** |
| tangent-angle p90 mean | 74.19° | **45.76°** |
| turn-angle p90 mean | 99.45° | **67.15°** |
| true-corner turn-angle p90 mean | 131.98° | **107.21°** |
| minimum TRAIN pen F1 | 1 | 1 |
| non-final false EOC, TRAIN | 0 | 0 |
| held-form CER | 81.893% | **81.481%** |

Relative to control, X error improves 60.8%, Y 77.4%, segment 36.3%, second
index difference 34.3%; CER improves 13.14 percentage points (93.2% relative).
Angles are **means of per-line p90s**, not pooled p90s. Index differences of
nonuniform RDP points are not physical velocity or geometric curvature.
Direct zero-input seed9142/9143 evaluations are duplicates, not independent
stochastic trials: offline validation confirms bit-identical XY/pen/latents
for **both** arms, all 40 lines, correct/NULL/swapped text conditions.
Independent CPU reload max XY drift 2.7180e-5, no pen/transcript differences.

All 32 TRAIN and eight held seed9142 marker-free images manually reviewed.
TRAIN letters/words are visibly more coherent and rounded, with fewer zigzags.
However, local `f`/`r` bends, tiny hooks/corners and punctuation/end strokes
still distort. **Not visually lossless.** Held text still largely scribbles.
CER near zero is not evidence of exact geometric reconstruction.

**Causal scope:** removing dropout changes BOTH NULL supervision and conditioned
exposure (expected 11.1% more conditioned examples), so pure harmful-gradient
interference is not conclusively isolated. An equal-conditioned-exposure or
masked-NULL-loss arm would separate mechanisms. No claim that CFG/text dropout
must universally be removed; drop0's NULL readout is untrained/OOD, not a useful
unconditional generator. Zero-input deterministic prediction also averages
away noise; it is a capacity control, not a stochastic diffusion replacement.

### Engineering closure and next decision

381 tests pass in both root and fork (18 new). New coverage: target/noise leakage
exclusion, exact pure-terminal input, eval guards, TRAIN-only native selection,
bounded budgets, paired minibatch/RNG/frozen/checkpoint checks, deterministic
zero generation remains text-sensitive, two-arm reporting, NULL diagnostic
grouping and unknown/missing group handling. All four study GPU/CPU apps stopped
with zero tasks. No ongoing GPU spend or further utilization changes.

As-run runners/helpers/model bytes match reviewed code. The first report used
reporter v1; second report used generalized two/three-arm v2. Each has its own
immutable `report/report-source.py`; do not rerender the first merely to match
the newer reporter. Engineering-review provenance records these hashes.

**Next priority:** preserve no-drop deterministic baseline as a mechanics control;
investigate broader text coverage/compositional conditioning and unseen text,
then return deliberately to stochastic diffusion/balanced NULL supervision.
Do not spend another run just polishing the same 32 TRAIN lines, enable codec
KL/OCR/style incidentally, or declare generation ready based on memorization.

Launcher-only closure hygiene: corrected copied module descriptions and the
NULL launcher's no-GPU help text (two dropout arms, not three denoising arms).
As-run launchers archived in engineering-review; no training semantics changed.

## Preregistered broader text-coverage diagnostic (2026-10-08)

Nested scope from immutable pool SHA
`d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9`:
A original32 writer10160; B all41 usable same-writer TRAIN lines; C1024 lines
across32 explicitly conditioned writers. Eight g09-301 lines held throughout;
ALL matching form families/normalized transcripts excluded globally, reserved
TEST/DEV writers excluded. Fixed reporting32retained+9added samewriter+24added
otherwriters+8held; checkpoint score uses ONLY reporting lines actually TRAIN
for each arm. Broad arm changes both textcoverage and writerdiversity;32→41
isolates samewriter addition. This is not a claim data is the sole bottleneck.

Identical no-drop step18000 parent model/Adam moments, SHA
`a63c20b70f7167680d391a6d07650879eb965a072b5d3f11e76b99aa12205439`.
One minimal style adaptation: zero-initialized learned writer bias before shared
blocks, ALLarms same architecture; zero initialization tested exactly equivalent
to old mapping. Base model unchanged, subclass explicit writer ID; new optimizer
parameter group, old moments retained. All384targets, originalTRAIN32whitening
fixed acrossarms, zeros/t999, textdrop0, batch8, LR1e-4, betas.9/.99, wd.01,
clip1, inherited XY/within-stroke targetsegment coefficients. Frozen polyphase40
codec and recurrentreader. No KL/CTC/styleclassifier/augmentation enabled.

Bounded8000updates/arm,900sTRAINcap; evaluate0/2000/4000/8000; final swappedtext/
swappedwriter controls. Different datasets imply different minibatches; don't
claim matched per-example gradients/RNG exposure. Same schedule algorithm/seed9142
and budget; no stochastic input (stored parent RNG not consumed). Oracle duration
still supplied; no reference/target at inference. Corpus-familiar reader plus
marker-free visual audit; unseen writing is nonunique. Save packedHDF5, source/
config/initial/best/last/optimizer state/logs/resource telemetry. No promotion.

New user archive `IAMonDo-db-1.0.tar.gz` inspected without extracting into Volume:
SHA `a71d647ffcbb76b5a1e77c65dac375be6ffd61ca1ba8d18a7f6a596d7cb9102f`;
941InkMLdocuments,193authorIDs,356189traces,5setfiles. It has hierarchical
Word/Textline/Structure annotations with transcriptions, affine canvas transforms
and compressed first/second-difference InkML trace notation—not existing IAM-OnDB
lineXML. Potentially useful, but needs its own canonical parser/coordinate and
split audit; do not silently mix datasets or assume writer IDs are shared.
No archive contents extracted into inode-constrained Volume. Existing clean pool
is sufficient for this first coverage experiment; preserve archive unchanged.

## Preregistered follow-up: soft local text hint (2026-10-08)

Coverage broad1024 failed both TRAIN and held readability at8000updates. Cross-
attention inspection of2retained/2held lines shows largely vertical token bands,
no obvious left-to-right path, even when original32memorization is excellent.
Attention is descriptive, NOT causal proof or true alignment measurement.

Two matched4000update continuations from broad1024step8000 (absolute26000), SHA
`f527a73d15576fba26a3c1af5fe787382a12b25d761f25d4834327019c3714b1`:
control0 versus small localcharacterembedding hint0.1. Hint linearly interpolates
ALLrequestedcharacterembeddings along normalized latent-duration fraction;
NOreference/target/characterboundaries provided, no char truncation, no smoothing.
Global cross-attention remains unrestricted (important for late dots/crossbars).
Same1024TRAIN/32writers/splits/whitening/modelweights/Adamgroups/moments/minibatches
(schedule9142offset8000), zero input/t999, textdrop0, batch8/LR1e-4/clip1, sameXY/
within-stroke targetsegment losses. No newparameters; zero hint tests exactly
recover source forward mapping. Hint excluded when NULL text is requested.
Report initial hint RMS vs hidden RMS; not an arbitrary large loss coefficient.
This approximate position prior may be wrong for nonuniform character durations;
retain a matched control and inspect genuine corners/strokes and held readability.
900sTRAINcap/arm, evaluate0/2000/4000, sameTRAIN65probe selector, final swappedtext/
writer controls; frozen codec/reader, packed outputs and provenance. No promotion.

## Completed text-coverage diagnostic (2026-10-08)

Artifacts `checkpoints/iam_generation_coverage/20261008-045818/`; all73marker-free
comparisons `report/index.html`, source1032linepackedHDF5, 3arms initial/best/last,
per-update logs, independentCPUreload, telemetry and text-attention maps.
All3arms8000updates, loop188.5/194.2/197.2seconds,900sTRAINcap not reached.
Zero-initialized writer adapter starts ALLarms with bit-identical trajectories;
all source1032codec RMSE<=2.626e-5, all73reporting source-reader transcripts exact.
Thus corrupt data/codec or inadequate source-reader recognition do NOT explain
this study's failure. This doesn't establish allrawIAM or newInkML is clean.

Actual optimizer groups: restored base beta(.9,.99), new writer embedding beta
(.9,.999) from AdamWdefaults, bothLR1e-4/wd.01/eps1e-8. Serialized top-level
`betas` describes restored base only; report records both actual groups. Policy
identical acrossarms. Preserve as-run config/code; no retroactive modification.

Final matched-update correct-text results (unique deterministic observations):

| Arm | retained32 CER | added9samewriter CER | added24otherwriter CER | held8 CER |
|---|---:|---:|---:|---:|
| small32 | **0.000%** | 81.356% (untrained) | 84.783% (untrained) | 83.128% |
| writer_all41 | 1.068% | **2.373%** (TRAIN) | 84.783% (untrained) | 83.539% |
| broad1024/32writers | 72.222% | 82.034% (TRAIN) | 82.754% (TRAIN) | 83.128% |

Do not average untrained added lines into small-armTRAIN or claim biggerdataset
comparison is a pure text-only intervention: broad adds writerdiversity too.
Original32sourcebeforecontinuation CER0.962%,X/YRMSE.025967/.001492; small32
final0CER/32of32exact,X/Y.008551/.000877,segment.010791,secondindexdifference
.017608,tangentp90mean22.14°,turnp90mean34.33°,cornerturnp90mean60.29°.
Perfectpen on small32 and all41 writer_all lines. LowCER still doesn't establish
visuallylossless localcurves. Broaddataset training doesn't pass basic fidelity:
retained32X/Y.338425/.070849, minimumpenF1.88; added24X/Y.346378/.117459,
minimumpenF1zero. All broad TRAINprobe lines stop, no nonfinalfalseEOC, and
free/oraclewindowCER are identical: prematurestopping is NOT the main TRAIN
failure. Heldfree/oraclewindow83.128/82.305% also remains bad.

Reader/codec frozen; independentCPUreload2retained/2heldperarm maxXY3.100e-5,
zero pen/transcript discrepancies. Initialmodelweights/moments match pinned
no-drop parent; initialwriterparameters zero. Coverage TRAIN minbatch schedules
differ by design; source/EOC controls preserved. Finalreadout budgets match;
bestcheckpoint selection TRAIN-only, neverheld. No failedbroadweight promotion.

Marker-free representative4retained/4added samewriter/4added otherwriter/4held
comparisons inspected: small32 improved natural readable shapes; writer_all
learned addedsentences, some hooks/ends stilldistort; broad even TRAIN becomes
flattened/zigzagged scribbles. Held scribbles in ALLarms. Other galleries saved,
not all73manually inspected. This result rules out “just add this1024-linepool
atthisbudget/policy” as a sufficient cure, NOT broaderdata generally. Could be
optimization, capacity, mean-regression ambiguity, writerheterogeneity or text
composition/alignment. Don't call the source dataset bad or corecodec broken.

Descriptive attention audit2retained/2held,4layers/4heads/arm: vertical token
bands rather than obvious monotonic writing/text mapping. Even perfectsmall32
memorization shows this. Head-mean time/tokenbarycenter correlations roughly
-.52to+.32 across inspected layers, not ground-truth alignmentaccuracy. Observer
requiresneed_weights=True, so kernel changes only in debug; maxstandardized
outputdrift<=1.824e-5, original sampling/CPUreload remain optimized path. Tests
confirm hooks removed, weights unchanged, NULL/padding preserved. No causal
claim from attention alone; motivates explicitly controlled localtext-hint test.

Archive furtherinventory:24,637Textline annotations (33blank),67,504Word,
493Formula groups, plus documents/drawings/tables/diagrams. These are annotation
counts, NOT acceptedline/formulatraining counts or verifiedLaTeX targets.
InkMLcompressedtraces, canvasrotation, hierarchy/trace references, splits, writer
namespace and possible duplicates/OOV need audit before import. Existing archive
unchanged; not merged into this study; no per-stroke files created on97.9%inode
Volume. Potentially useful later for word-level alignment and document/math work.

## Completed soft local text-hint comparison (2026-10-08)

Artifacts `checkpoints/iam_generation_text_anchor/20261008-051702/`; both arms
4000updates from identical broad8000 modelweights/Adam moments. Same1024TRAIN,
all4000minibatchIDs/order match, all65TRAINprobe/8held outputs saved, zero-input
same targetloss, no training randomness. Hint coefficient.1 gives initialRMS
**8.07%** of sharedhiddenRMS on first8retainedTRAIN; no newparameters. Uniform
interpolation is NOT actual character alignment and could be a poor durationprior.
This intentionally small intervention isn't an exhaustive alignment investigation.

Final matchedbudget correct-text results:

| Arm | retained32CER | added9samewriterCER | added24otherwriterCER | held8CER |
|---|---:|---:|---:|---:|
| control |75.214%|80.000%|80.000%|**78.189%**|
| local_hint.1 |75.962%|79.322%|78.696%|**83.539%**|

Control retainedX/Y.363133/.074348,segment.179990,secondindexdifference.255417;
hint.375585/.074888,.179353/.255045. OnotherwriterTRAIN, hintX/Y.405842/.103545
vscontrol.385546/.101424; segment.192186vs.191477. No coherent global win;
heldCER worsens5.35percentagepoints. Control's78%stillfundamentally fails; no
claim of readiness from a few fewer reader errors. **Reject hint as a promoted
fix.** Tiny gains on someTRAINreader cases do not justify worse geometry/heldtext.
IndependentCPUreload maxXY3.577e-5, zero pen/transcript discrepancies. Source
model/optimizergroups match; only appliedlocalhint changes at step0. Attention
observer/report kernel is debug-only, never used for training or real evaluation.
As-run config inherits some coverage-only caveat fields; explicitintervention/
sameTRAINschedule above and reportinglimitations describe this matched study.

### Additional zero-training objective autopsy

At coverage broad8000 checkpoint, frozen no-updateCPUshared-gradient check on
first8retained and first8otherwriterTRAIN: pen/geometry gradientnormratios
**1.30** and **0.50**; cosine between them **+.224** and **+.083**. PenMSE can
be larger without its shared gradient dominating or conflicting. These two
batches do NOT support assuming a replay of the earlier Chinesepenweighting
failure. Small sample, not a global proof losses never compete.
`iam_generation_coverage/20261008-045818/diagnostics/loss-gradient-audit.json`
and exactsource preserve this diagnostic; weights/gradbuffers asserted unchanged.

344nominallyunused channels are nearzero (maxabs1.34e-6 in65sourceTRAINprobe);
final contribution to full384channelMSE~0.7–1.5e-7. ActiveXY/pen contribute
~.003–.058. Thus wasting loss on unusedchannels isn't the main observed failure.
PrematureEOC isn't either: correctTRAINfree/windowCERidentical inbroad/hintarms.
Not a randommixture or latent readout bug: exactpolyphase transport, zero input,
deterministicmean output and predictedpens used consistently.

### What this establishes, and what remains

Useful retained32fidelity gain:0CER/32exact and materiallybetter localgeometry.
The nine added singlewriter lines can also be learned. **However, the lightweight
128×4 textmapper demonstrably underfits this1024line/32writer setting at tested
budgets; no unseencomposition gate passes.** Capacity proof on32lines was never
proof this standaloneprototype could scale, nor that initializedpolyphase40
transport is equivalent to a learned semanticInkVAE. The releasedDiT config is
much larger (896wide×16blocks, textdim512/3convblocks), with materiallydifferent
conditioning/training. Don't interpret this as DiffInk fundamentally failing.

Nextqualitypriority: establish a controlled intermediate128/256line capacity/
optimization gate and a properlybudgeted strongertextmodel comparison, preserving
clean sourcecodec and reader as controls. Investigate learnedsemantic/glyph-aware
conditioning deliberately rather than indefinitely polishing a point-packed tiny
regressor. Newarchive may offer shortword/wordgroup supervision after its own
parseraudit, but moredata is not a substitute for a model that fits its current
TRAIN scope. Do not enable codecKL/CTC/style incidentally or promote failedbroad/
hint checkpoints. Keep workingpaper-baseline versus researchcodec/prototypes
clearly separated. No infrastructuretuning done; telemetry remains active.

Checkpoint-selection closure: small32 selected8000; writer_all selected4000;
broad selected8000. Both hint/control TRAIN-only selectors retained their
step0 source, despite smaller later latentloss and a few better heldreader errors.
Tables intentionally compare equal finalbudgets, not selectedbest aliases. Do not
promote continuation by heldCER. Marker-free4retained/4held finalhint/control
comparisons also inspected: both still flattened/illegible, no visualrescue.
403tests pass root/fork (22new vs381); no production codec/model changed.

## Fresh intermediate capacity gate and target-geometry refresh (2026-10-08)

Completed artifacts (Volume prefix `/mnt/diffink-data/`, container `/data/`):

- `checkpoints/iam_generation_capacity/20261008-054853/report/index.html`
- `checkpoints/iam_generation_anchor_refresh/20261008-061315/report/index.html`

Each report contains **all264 marker-free comparisons**, full actual TRAIN
metrics, held8, swapped-text/writer controls, learning curves, independent CPU
reloads, immutable configs, initial/best/last checkpoints, packed evaluations,
RNG/Adam states and a packed complete as-run Python source archive. No old reports
or checkpoints were rewritten. No original production VAE/DiT code changed.

This remains a **standalone direct deterministic text+writer mapper** over the
faithful initialized polyphase40 research codec, not a learned semantic VAE,
original InkDiT, stochastic generator, paper reproduction or independent OCR
benchmark. Codec `9c53f68e...` and corpus-familiar reader `2ebac71d...` stayed
frozen, with state and absent-gradient checks. Zero input, normalized t999,
oracle ceil(N/8) duration, full384 predicted channels; no target trajectory or
prefix at inference, no text/trajectory dropout, no KL/CTC/style loss. The supplied
duration can leak identity. The eight held prompts are repeatedly inspected
research controls, not an untouched final benchmark.

### Dataset and matched capacity comparison

Use the first128/256 entries of the existing safely round-robin packed1024
source, retaining all41 safe lines from original writer10160. Both sizes cover
32 writers; no new archive or normalization/RDP changes. Same global held form
and normalized-text exclusions, reserved writers rechecked against clean pool.
All original32 whitening statistics held fixed, no held statistics. Source HDF5
SHA `d513ec4e81330685055d006b5527043bac7e07d16d688251d41621af43648c71`;
source manifest/whitening SHA pinning in `generation_capacity.py`.

**Fresh** initialization and empty Adam for all three arms, avoiding a warm small
versus fresh big comparison. Small128/small256 weights identical; larger necessarily
has different weights, same initialization seed18142 and identical initial zero
readout predictions. Physical batch8,16000 updates, LR1e-4, betas(.9,.99),wd.01,
clip1, schedule seed19142. Identical256 minibatches/order for small256/larger256.
128 receives1000 exposures/line,256 receives500; not equal exposure or FLOPs.
Larger changes both width AND depth, one initialization, not independent repeats.
TRAIN-only selection evaluates **every** actual TRAIN line, not the prior65probe.

| Arm | Model/parameters | Actual TRAIN CER/exact | X/Y RMSE | segment/second-index-difference error | held8 CER |
|---|---|---|---|---|---|
| small128 |128×4 /1,303,808|43.815% /0/128|.121684/.019597|.075073/.104368|79.424%|
| small256 |128×4 /1,303,808|63.766% /0/256|.203526/.045384|.113596/.161252|81.070%|
| larger256 |256×6 /7,073,408|49.330% /0/256|.151252/.026768|.077987/.104942|86.008%|

Training loop352.7/352.5/502.5seconds respectively; all16000 budgets complete,
zero clipped steps. Bigger fits better at this budget, **not enough** to clear
TRAIN fidelity or composition. We have NOT proven a fundamental capacity limit:
curves are still improving with training, budgets differ greatly in per-line
exposure from the old32 memorization. All264 source trajectories round-trip with
maximum RMSE1.0733e-5, pens exact, reader **264/264 exact**. This does not suggest
bad labels/codec as the explanation for these failed generated trajectories.
Small256 TRAIN CER rises63.77→82.00% with swapped text; larger49.33→85.08%.
Thus requested text matters on TRAIN; this is not proof of held composition or
freedom from duration leakage. Bigger writer swap only49.33→51.41% CER; not proof
of good writer/style control.

### A newly measured packing-boundary artifact

CPU pooled within-true-stroke segment audit, real link start index modulo8:
for larger256 at16000, phase7 (crossing packed latent blocks) segment error
**.15656**, versus **.05805–.06697** for phases0–6. Actual target segment RMS is
.08613–.08727 across phases: unequal RDP spacing alone cannot explain this strong
phase pattern. Similar effect for small128 (.14751 vs.05761–.06539) and small256
(.24528 vs.07867–.09508). This is empirical periodic error, **not yet a causal
identification** of which positional features/readout/optimization mechanism makes
it. Do not call it a proven ConvTranspose bug.

The actual *generated* unwhitened first40 fields versus decoded XY differ at most
**7.248e-5** over every TRAIN trajectory, including padding; pooled RMSE~2.18e-5,
no direct-field/decoded pen mismatches. This checks off-source generated latents,
not just valid source reconstruction. Frozen decoder/GMM readout cannot explain
the .15656 jump error; it originates mainly in the text mapper's predictions.
`diagnostics/{phase-audit,decoder-audit}.json` and exact CPU scripts preserve it.

### Loss anisotropy and controlled refresh

Original TRAIN32 whitening has sigmaX~2.89, sigmaY~.20. For equal raw XY position
errors the **base** standardized latent MSE penalizes X about200× less than Y.
This is a declared weighting policy, not a coding/normalization correctness bug;
Adam and shared gradients mean it is NOT by itself proof of the cause.
At small128 step16000, three TRAIN8 batches have X/Y RMSE~.113–.130/.019–.020,
shared X gradient norms.00267–.00362 versus Y.00917–.01086. Other evidence and
exact scripts in `diagnostics/axis-loss-audit.json`; no parameter/grad-buffer edits.

Branch all three8000-update arms from **identical larger256 step16000 weights and
Adam**, SHA `f87f99bb9c945cf241c4cbfcde2976073874ecc9c6a2a720984cff51d1991787`.
Same256 lines/minibatches/schedule seed20142, all prior hyperparameters and frozen
codec/reader unchanged. No geometry postprocessing or architecture edit.

Recalibrate mean shared-backbone gradients (excluding final readout) on24 fixed
TRAIN lines only; base norm.00690111, isotropic XY norm.35727552, target-segment
norm.03636326. Old anchor shares at this parent are9.39%/2.13%, not the earlier
intended25%/10%. Fixed new coefficients give those initial25%/10% shares; they do
NOT adapt thereafter. No huge arbitrary curvature/smoothing weight.

| Arm | active XY/segment coefficients | TRAIN CER/exact | X/Y RMSE | segment / second-index-difference | mean per-line tangent/turn p90 |
|---|---|---|---|---|---|
| control |.001813513526/.004050897769|23.566% /3/256|.118174/.015521|.055584/.072303|89.35°/113.57°|
| refresh_xy |.004828983342/.004050897769|31.044% /1/256|.098585/.017606|.063814/.085455|100.34°/123.71°|
| refresh_xy_segment |.004828983342/.018978251569|19.586% /3/256|.074082/.013152|.045525/.060970|80.58°/106.92°|

Index differences are **not physical velocity or geometric curvature**. Segment
matching minimizes Δpred−Δtarget only inside true strokes, preserving target
corners/hooks rather than making trajectories universally smooth. Tangent angles
compare normalized segment vectors; turn angles compare consecutive valid
within-stroke directions. Tables average each line's p90, not a pooled p90.
True-corner turn p90 remains poor:141.56°/145.80°/138.36° respectively.

Against matched control, XY+segment reduces X37.3%,Y15.3%,segment18.1%,second
index-difference15.7%,TRAIN CER3.98percentagepoints. Phase7 error .09354→**.07177**
(23.3%); other phases also improve (.04582–.05484→.03924–.04442). Seams remain
worse than within-block links. XY-only decreases X16.6% yet **worsens** local
geometry/readability; reject it as a promoted fix. This demonstrates why point
RMSE alone was misleading. Derivative refresh gives a genuine but **partial**
geometry improvement, not a solved one. Baseline's extra training itself accounts
for a large49.33→23.57%CER gain; don't credit all progress to loss changes.

Pens: control mean/minF1=1/1; XY-only .998535/.933333; XY+segment
.998936/.926829. Zero false non-final EOC in all arms; free/window CER equal,
so premature stopping isn't the main TRAIN failure. The rare boundary regressions
are real, despite excellent average F1. All gradient clipping fractions remain0.
Held CER83.951/86.831/81.070%: **all fail**, no new-prompt readability gate.

Both studies saved all264 galleries. Manually inspected4retained/4new128/
4expansion/4held per study (not claiming every gallery was manually inspected).
Bigger/later models restore more readable TRAIN letter structure; segment refresh
reduces roof-like discontinuities and is generally closer in these reviewed lines,
but angular loops/odd joins persist. Held remains illegible scribbles. This is
**not visually lossless** across this larger setting and not ready for full IAM,
production English InkDiT or an incidental OCR/KL/style training launch.
Independent CPU reloads,8stratified lines/arm: maxXY3.839e-5 for capacity,
1.526e-5 for refresh, zero pen/transcript differences. All selectors picked final
steps; report guards verify full TRAIN selection, equal schedules, immutable
source/checkpoint SHA, identical restored Adam and initial predictions.

**Next:** keep the larger+target-segment arm as a TRAIN-fidelity research candidate,
not a promoted generator. Test lower-LR convergence and, if seams persist,
relative-target-segment/explicit boundary diagnostics before further increasing
data. Contextual/learned text alignment and genuinely semantic/paper latent
conditioning remain separate likely needs for composition: a weak uniform hint's
failure did not rule them out. Do not indefinitely polish32lines or claim this
polyphase prototype is the authors' architecture. Existing clean pool is ample;
IAMonDo archive remains immutable/not merged.424tests pass root/fork (21new).

## Matched learning-rate convergence audit (2026-10-08)

Continue from the completed target-XY+segment parent, **absolute step24000**:
`checkpoints/iam_generation_anchor_refresh/20261008-061315/refresh_xy_segment/checkpoint-last.pt`,
SHA `eff1181fcf6edeb953b8bd5c3501363b74bb66470fa5ccba7a480ba31a432081`.
This remains the standalone deterministic zero-input **text+writer mapper**, with
frozen initialized polyphase40 codec and corpus-familiar reader; **NOT** a learned
semantic InkVAE, released InkDiT, stochastic generator or paper reproduction.

New immutable study: `checkpoints/iam_generation_lr_polish/20261008-070802/`.
Three matched8000-update continuations: `control`1e-4, `midpoint`5e-5,
`polish`1e-5. Same256TRAIN,32writers,8held form prompts; width256/depth6,
7,073,408parameters; batch8, AdamW(.9,.99),wd.01,eps1e-8,clip1,dropout0.
Restore **all model weights, Adam moments and CPU/CUDA RNG**, then explicitly
change LR in **every actual optimizer group**. No reset/recalibration. Fixed
XY coefficient .0048289833417749835 and target-index-difference coefficient
.018978251569199526, same original32 whitening. Schedule seed21142 gives
identical minibatch order across arms; each line receives250 extra exposures.
AdamW decay magnitude also follows LR; this is not an isolated decay experiment.
No KL/CTC/style, new data, positional edits, smoothing or inference filtering.
Evaluate0/2000/4000/8000, plus matched swapped-input controls only at final;
selector uses EVERY256 actual TRAIN line, never held prompts. Final comparisons
are equal-budget checkpoints, not independently cherry-picked best aliases.

New reusable CPU diagnostic: `packed_error_audit.block_error_statistics`.
Decompose coordinate residual into a per-eight-point-block mean and zero-mean
shape residual. This uses **target positions**, so is diagnostic only, NOT a
legal generation correction or postprocessing trick. Squared vector energies
must retain the signed cross term; do not claim separate seam-energy fractions
sum to100%. Partial real blocks count, padding does not. Stroke-ending jumps
are excluded from derivative windows; unequal RDP spacing is retained, and
index differences are NOT physical velocity or geometric curvature.
At the step24000 XY+segment parent,54.61% of pooled point-error energy is in
the block means; seam error energy .005150 versus within-block .001690.
Seam block-mean/shape-difference energies .003902/.004154, cross term−.002906.
Uncentered adjacent-error correlation .600 at seams versus .844 within blocks.
Thus both block-location drift and local shape residuals contribute; this does
not causally identify a layer bug or make generic smoothing appropriate.
Segment errors persist across target-length quartiles, not just sparse long
RDP segments. Exact source and parent audit scripts/JSON are preserved with
the review. All264 source preflights still require faithful codec/pens and
reader transcripts; the extra IAMonDo archive remains immutable/not merged.

### Completed results: substantial convergence improvement, not composition

All three8000-update budgets completed (absolute step32000), zero clipped steps.
Loop266.8/275.6/272.3seconds respectively; full evaluations/readbacks are separate.
Equal-final-budget results, mean per-line geometry except explicitly pooled seams:

| Arm/LR | TRAIN CER/exact | X/Y RMSE | segment / second-index-difference | mean tangent/turn p90 | mean corner/shallow-turn p90 | mean/min penF1 |
|---|---|---|---|---|---|---|
| control1e-4 |14.514% /16/256|.084691/.009375|.040300/.054615|71.93°/98.85°|133.64°/58.16°|.999630/.967742|
| midpoint5e-5 |3.095% /120/256|.087416/.007268|.027500/.039185|49.28°/73.67°|114.38°/41.84°|1/1|
| polish1e-5 |**2.367% /145/256**|**.029026/.001479**|**.025095/.034325**|**41.70°/61.61°**|107.64°/35.17°|**1/1**|

Against matched1e-4 control,1e-5 reduces X65.7%,Y84.2%,segment37.7%,second
index-difference37.2%; TRAIN CER drops12.15percentagepoints. Against the
step24000 parent, CER19.586→2.367%, X.074082→.029026,Y.013152→.001479.
Midpoint shows why CER alone is insufficient: large X drift remains despite
much more readable output. Full-TRAIN geometry selector chooses control0,
midpoint4000,polish8000; tables/galleries still compare matched FINAL8000, not
mixed best aliases. Polish is the new TRAIN-fidelity candidate, **not promoted**
to a production generator. Last checkpoint SHA
`c4ffb8fb93c0008ef88645eeb3ba41ad45c13b728e342c0a39b7f7a141065890`.

**Important hypothesis update:** the every-eighth-link excess is NOT persistent
in the polished model. Pooled TRAIN phase7 segment error is control.059206,
midpoint.039260,polish**.016840**. Polish phases0–6 are.022959–.032580:
the seam is now **better**, not worse, than within-block links. Against control,
seam error drops71.6%; versus parent.071766 it drops76.5%. Target segment RMS
at phase7 is.086951 and remains identical across arms. This rules out an
irreducible packed-block/8× capacity barrier for this TRAIN scope and weakens
the case for a dedicated seam upweight or architectural upsampling replacement.
It does NOT identify a universally bad layer or prove all future scales solved.
The prior off-source codec audit already excluded decoder/GMM as the primary
source of the large seam errors. In this matched setup the LR/AdamW update
policy has a strong causal effect; the exact optimization/readout interaction
behind its earlier periodicity remains unknown. No clipping/explosion explanation.

Polished pooled point-error energy now20.22% block means, versus54.61% at parent.
Remaining error is predominantly within-block **shape** residual, not simply a
common location drift. Source/RDP nonuniform spacing is preserved; no resampling,
blur, curve fitting or generic smoothing was used to obtain these improvements.

Held8 CER83.951/84.774/82.305%, still visually unreadable and no composition
improvement. Text swaps against original TRAIN targets remain controls, not
new-prompt quality. Corpus-familiar reader, one parent, oracle duration,
repeatedly inspected held form and TRAIN memorization all limit interpretation.
No new data/held statistics were used, and no full IAM/production InkDiT launch.

### Visual review and remaining artifacts

Full report: `.../20261008-070802/report/index.html`, all264 marker-free lines,
predicted pens and actual first-EOC stopping; all final pens correct for both
lower-LR arms, no non-final false EOC or stopping differences. Independently
reviewed4retained+4new128+4expansion+4held whole lines, not every264 gallery.
Independent CPU reload8stratified lines/arm: maxXY1.1444e-5, zero pen or reader
transcript mismatches. Report verifies source SHA, exact initial model/Adam/RNG,
all actual param-group LRs, full TRAIN selectors,250 exposures/line, packed
output SHA and bitwise identical initial decoded trajectories.

Additional `diagnostics/curve-focus/index.html`:12 difficult TRAIN closeups,
selected **only from immutable step24000 parent errors**, not whichever final arm
looks good. Same physical axes, target/parent/control/midpoint/polish, no markers
or filtering. Each crop is8 sampled points within a true stroke, every target
turn≤45°, total absolute sampled turn≥25°. True pens deliberately isolate geometry
**in these crops only**; full gallery uses predicted pens. This is a targeted
failure diagnostic, NOT an unbiased benchmark. Source script/metrics preserved.
All12 crops manually inspected. Rounded/roof-shaped distortions improve clearly
in `k04-309z-09`, `k04-309z-04`, `a02-130z-03`, `e07-425z-02`, `l08-922z-06`.
But visible pointed/flattened bends remain in `d08-586z-01`[77:85],
`p10-249z-01`[156:164], `n05-514z-02`[309:317]; even a correctly read transcript
can have an unnaturally changed curve. Original `p08-936z-05`/`a07-421z-02` are
NOT in this264-line mapper study: do not claim these experiments retested them.
Their faithful source-codec reconstruction remains a separate earlier result.

**No visually-lossless gate claimed.** Smooth-curve TRAIN fidelity is much better,
but not uniformly faithful; novel text still fails completely. Do not incidentally
enable KL/CTC/style or increase model/data scale on the strength of low CER alone.
Next localized fidelity diagnostic should target the remaining within-block shape
errors (e.g. modest gradient-calibrated target-relative segment/tangent matching,
with a finite short-segment floor, preserving target corners), NOT an obsolete
phase7 penalty or generic curvature smoothing. A controlled contextual/learned
text-alignment study is separately needed for composition; more tiny-set polishing
alone will not establish that. Existing data are sufficient for these diagnostics.
440 tests pass root/fork (16new); sources, moments/RNG, logs, metrics and galleries
preserved. Own GPU/CPU apps completed; user Modal shell was not touched.

## Fresh text-composition alignment control (2026-10-08, completed)

This investigation pivots from polishing known trajectories to testing character
composition. The earlier TRAIN2.367% versus unseen82.305% CER, with unseen swapped
text81.481%, is **not an acceptable or inevitable small-data outcome**. It is
consistent with memorized conditioning and insufficient reusable alignment;
attention observations are suggestive, not ground-truth character alignment.
The former weak1024-line experiment was underfit and does not rule out more data.

Fresh paired study: `checkpoints/iam_generation_alignment/20261008-081416`.
Both arms start from identical **new** neural weights/empty Adam/RNG, not the
polished memorized model. Same256TRAIN/32writers, physical batch8, width256,
6blocks/4heads, no dropout/CTC/style/KL, original TRAIN32 whitening, frozen
initialized polyphase40 transport codec and frozen corpus-familiar reader.
Fixed inherited XY.004828983342 and target-index-difference.018978251569 anchors
are shared; not recalibrated on the fresh network. Updates24000, warmup1000
1e-5→5e-5, hold through16000, cosine→1e-5. Actual schedules/LRs/seeds/checkpoint
hashes and every TRAIN line used in selection are verified from saved artifacts.

- `global`: original unrestricted cross-attention.
- `soft_gaussian`: finite log Gaussian, center based on TRAIN-average packed
  blocks/character, sigma2.5characters, floor−12;3heads biased/1global for delayed
  dots/crossbars, BOS bias−2, plus trainable text/ink PE **amplitudes**. This is a
  bundle, not an isolation of PE versus masking. Distant attention remains possible.

Reference: TrInk EMNLP2025 sections2.2–2.3,
<https://aclanthology.org/2025.emnlp-main.244/>. The actual paper scales PE
amplitudes, not positional indices, and uses TRAIN-average points/character for
its center. Our intervention is only inspired by those components: no contextual
text encoder/autoregression/offset MDN or600k-example training. This remains a
standalone deterministic zero-input mapper, **not** released InkDiT, semantic
InkVAE or a TrInk reproduction.

Duration inference also gets a TRAIN-only ridge baseline on requested text
features and shrunk writer intercepts. No sample ID or target points are accepted
by its prediction API. TRAIN fit absolute relative duration error mean6.64%,
median5.54%,p90 14.58%; TRAIN-average1.53778packed blocks/character. These are
in-sample calibration statistics, not unseen performance. Oracle and estimated
lengths are separated, with fixed-length swapped/NULL controls. Models still
train on oracle lengths; estimating lengths at inference introduces distribution
shift as well as removing an identity shortcut.

**TRAIN-only convergence extension, predeclared before soft arm completion:** if
either arm's matched-final TRAIN CER exceeds10%, continue BOTH24000updates at
1e-5 with their own full model/Adam/RNG, same new order. Global final TRAIN22.93%
triggered this; do not conflate its underfit fresh result with the earlier long-
trained2.37% prototype. Sealed confirmation stays unopened until both extended
budgets/selection finish. Original immutable study and continuation are separate.

Confirmation is metadata-sealed before training:16IAM lines/8known writers,
new IDs/normalized transcripts outside every1032 prior packed generator example,
forms excluded from CURRENT fresh TRAIN/dev. Older1024 runs touched other lines
from every available form for these writers: historical form exposure is NOT
hidden. A second seal contains16 genuinely new grammatical recombinations, with
no paired trajectory. Use TRAIN-only selected checkpoints after both budgets;
no confirmation tuning or reselection. Source-reader failures stay in denominators.
Synthetic prompts get predicted durations ONLY; no invented oracle geometry.
Reports preserve full marker-free galleries, true first-EOC stopping, oracle vs
estimated controls, CPU reload and kernel-drift-qualified attention observations.
Completed results and conclusions follow below.

### Completed results and interpretation (2026-10-08)

Both fresh budgets completed, then both TRAIN-triggered continuations completed.
Final source: `checkpoints/iam_generation_alignment_continuation/20261008-084328`.
Total48000updates/arm,1500exposures/every TRAIN line, same order/LRs and full
own-arm Adam/RNG restoration. No clipping, no frozen codec/reader drift. Both
TRAIN-only best selections are the matched final step48000. No subsequent
training or hyperparameter selection used confirmation outputs.

| Evaluation (CER) | Global | Soft Gaussian + PE amplitudes |
|---|---:|---:|
| Fresh24000 TRAIN256 / oracle duration |22.929%|17.336%|
| Final48000 TRAIN256 / oracle duration |5.020%|2.406%|
| Final TRAIN / swapped text, fixed oracle duration |84.120%|84.406%|
| Final TRAIN / estimated duration |63.584%|61.217%|
| Previously inspected development8 / oracle |83.128%|85.597%|
| Development8 / estimated duration |80.658%|83.951%|
| Sealed paired16 / oracle |85.714%|84.009%|
| Sealed paired16 / estimated duration |82.729%|84.648%|
| Sealed synthetic16 / estimated duration ONLY |81.208%|78.523%|

No confirmation prompt is exact in either arm. Sealed paired source reader is
**16/16exact,CER0**, and all264 existing sources are reader-exact with faithful
codec round trips. Failure is visible scribbling, not simply a reader falsely
rejecting plausible alternative writing. Synthetic prompts have no reference
trajectory, oracle duration or aligned RMSE. Correct-vs-swapped paired estimated
CER is global82.729/82.729%,soft84.648/83.795%; synthetic81.208/80.872% and
78.523/80.034%. The slight differences are not a composition gate or replicated
statistical evidence of generalization.

The intervention **does improve familiar-text fitting** at matched budget:
TRAIN CER reduces52.1%, exact transcripts82→152/256. X/Y RMSE .042636/.004360
→.032951/.003198; target-index-segment error .029275→.024456; second-index-
difference .038677→.033892. All TRAIN pen boundaries/termination correct in both
arms. This is not an across-regime comparison with the older pretrained polish
model, and not proof of visually-lossless local curves. Marker-free TRAIN views
still show angular/flattened bends; authentic source microstructure is not all
preserved. No smoothing/resampling or decoder change explains this comparison.

**New decisive diagnostic: duration fragility, even on familiar text.**
Only36/256 predicted durations equal the oracle block count; on those36, every
reader string is unchanged (global CER4.920% both modes;soft2.554% both). On the
other220, every reader string changes: global5.036→72.934%,soft2.382→70.567%.
The predictor's overall TRAIN duration error is only6.64%. This is strong
sensitivity to exact timing/grid conditions, not just lack of known characters.
It supports the earlier unique writer+oracle-length shortcut concern but does
**not** separately prove lookup versus out-of-distribution timing/index inputs.

Code contains both absolute position and `PE(relative_progress * 100)`, where
relative progress uses `mask.sum()-1`. Changing total duration changes common-
prefix features throughout a line: index49 with50→51blocks changes the PE
argument100→98, a2-radian highest-frequency phase shift. The Gaussian center on
common queries is unchanged, but normalized PE and self-attention context change.
This is a plausible contributing mechanism, **not yet an isolated causal ablation**.
Next experiments should explicitly address this duration/grid dependence rather
than merely improve a duration regressor or further polish memorized outputs.

**A diagonal attention plot is not sufficient.** On4fixed stratified lines,
last-layer soft biased-head time/token-barycenter correlations average
.989/.966/.984, versus global.513/.170/.391; the un-biased fourth head remains
negative, −.234 versus−.221. The intervention genuinely induces approximate
left-to-right attention, yet new text remains unreadable. These are descriptive
soft-attention observations, NOT true character-boundary accuracy, and an
approximate linear center is not necessarily the correct alignment. Global heads
and ink self-attention can still carry whole-prompt identity information. Observing
weights switches kernel: output max drift1.16e-5global/7.39e-6soft, recorded.

What this rules out / does not rule out:
- Basic source decoding, alphabet OOV, gross CPU/GPU reload mismatch, broken reader
  on these paired sources, and simple absence of a Gaussian prior are not adequate
  explanations or fixes for this failure.
- Bad TRAIN fit alone is no longer adequate: soft reaches2.4%TRAIN CER but still
  fails the fresh prompts. TRAIN and new-text quality must stay separate gates.
- We have **not** ruled out better learned/local alignment, contextual text
  representations, an autoregressive/content-semantic objective, more data under
  a strong converged regime, or duration-aware training. The older1024failure
  remains underfit evidence, not proof that data cannot help.
- No diagnosis of an irreducible InkVAE architecture, dataset corruption, or a
  broken released InkDiT is justified: this is the separate deterministic mapper.

**Recommended next controlled stage:** remove/replace the target-length-dependent
high-frequency relative-position identity cue (retain the control), and make
training tolerate target-free timing; separately test contextual character
representations/stronger compositional alignment. Any temporal augmentation must
preserve true stroke geometry/corners, not smooth it or invent target alignment.
Reserve NEW confirmation prompts before these experiments; current confirmation
is now exposed, not reusable as a fresh blind gate. Revisit a larger corpus only
with the stable training protocol and separate oracle/predicted-duration reports.
Do not incidentally enable KL/style or declare original semantic InkVAE/InkDiT
ready because familiar-text curves/CER improved.

### Artifacts, verification and visual review

- Full report: `.../20261008-084328/report/index.html`, metrics `summary.json`,
  all264marker-free lines +16paired +16synthetic, predicted pens/first-EOC stopping.
- Sealed outputs/source preflight: `.../confirmation/summary.json`, packed
  `source.h5`, both selected models' oracle/predicted/control HDF5/JSON.
- Same-text timing comparisons: `.../diagnostics/duration-fragility/index.html`,
 8fixed TRAIN examples, all256 duration-matched/error-bin metrics and audit source.
- Attention raw HDF5/JSON and fixed4-line,6-layer/4head descriptions in `report/`.
- Independent CPU reload8stratified lines/arm: maxXY2.81e-5,zero pen mismatches,
  all reader transcripts match GPU. Every TRAIN line enters checkpoint selection;
  actual seeds/state/data/source hashes, budgets/minibatches/LRs are verified.
- Fresh initialization regenerated independently. CPU-only/CUDA wheel normal-
  initialization math differs at most1.61e-6 in text embeddings; numerical2e-6
  tolerance applies ONLY to this cross-wheel regeneration. Actual arm-to-arm
  initial weights/RNG and own-parent restored weights/Adam/RNG remain bitwise equal.
- Manually inspected ALL16paired and ALL16synthetic outputs,12TRAIN lines,
 4development lines, all8same-text duration comparisons, and the held attention
  heatmaps. All264galleries saved, not all manually reviewed. Original named
 `p08-936z-05`/`a07-421z-02` are outside this264 mapper scope, not retested here.
- 455tests pass root/fork (15new vs440). Original fresh source archive, continuation
  archive, full moments/RNG/logs/checkpoints and reporter/confirmation sources kept.
- Own2T4apps and CPUreport app finished/stopped; no user app stopped or modified.

**Verdict:** useful controlled fitting improvement and a much clearer timing
failure, but composition still FAIL. No production/full-IAM/InkDiT promotion.

## Isolated relative-PE timing autopsy (2026-10-08)

Completed CPU-only frozen-checkpoint diagnostic:
`checkpoints/iam_generation_timing/20261008-093817/report/index.html`.
Every256TRAIN request tested on BOTH48000-update arms. `TimingProbe` changes ONLY
relative-progress PE denominator when requested. Its native path delegates exactly
to the original model; override at native counts is separately parity-tested.
Actual context/output length and PE denominator are manipulated independently;
no model/codec/reader weights, target point spacing or trajectories are modified.

| Frozen TRAIN CER | Global | Soft Gaussian |
|---|---:|---:|
| Native |5.020%|2.406%|
| PE denominator +1block, native context |60.450%|57.758%|
| PE denominator −1block, native context |59.683%|55.898%|
| Context/output +1block, oracle PE held fixed |9.754%|4.708%|
| Context/output −1block, oracle PE held fixed |18.156%|11.120%|
| Both +1block |58.382%|56.236%|
| Both −1block |58.395%|55.443%|

PE-only damage is not just termination or CER accounting: for soft+1, common-
prefix mean XY drift .200253/.121404 and4588pen changes; mask-only+1 has
.035141/.010331 and **zero** common-prefix pen changes. Negative PE shifts can
cause large terminal XY extrapolation, so positive-shift/visual evidence is
particularly useful. PE-only+1 puts EOC beyond the original output window, while
mask-only−1 truncates terminal points (all256missingEOC); report both stop and
prefix geometry. Oracle PE overrides isolate mechanism but are NOT a deployable
text-only generation workaround.

**Causal conclusion limited to these frozen checkpoints:** changing the
length-dependent relative-position feature alone is sufficient to reproduce the
large within-line distortion; the self-attention-context change alone is much
less damaging, especially +1 where no truncation occurs. The feature is a major
contributor to familiar-text duration fragility. This does not prove it is the
only cause of new-text composition failure or that removing it from scratch
will solve generation.

Independent native CPU reload of ALL512arm/line combinations: maxXY2.90e-5,
zero pen mismatches, all512reader strings equal GPU. Source/archive/packed output
hashes and frozen parameters verified. Report has eight fixed TRAIN examples
including d08/p10/n05 across±1separate interventions, no markers/smoothing or
held-example selection. Manually inspected soft±1on k04/c03 and soft+1on d08/p10;
all256outputs saved, not every line manually reviewed. No GPU allocated.

### Target-preserving nuisance-jitter continuation — completed, failed remedy

`checkpoints/iam_generation_position_robustness/20261008-095300`.
Matched `control`/`pe_jitter` branch from the **identical** completed soft48000
model/full Adam/RNG.8000updates each at1e-5, same256TRAIN/minibatches/shared
physical-loss anchors, no new architecture parameters/CTC/style/KL.
On half TRAIN rows, jitter ONLY relativePE denominator by a rounded±20% length
factor; tensor length, self-attention mask, target XY/pen sequence, all corners,
RDP spacing and input samples remain untouched. This is nuisance-feature
invariance, **not generic smoothing or trajectory resampling**. Jitter RNG is
separate/reproducible, its actual length schedule logged; control consumes same
nuisance draws but uses native denominators. Native inference uses actual
requested duration, NO oracle override. Compare native fidelity versus target-free
TRAIN-duration robustness; native all256TRAIN geometry selects checkpoints.
Existing development8 may be reported descriptively but is exposed. Previously
opened16paired/16synthetic confirmation is NOT rerun/tuned on or called fresh.
Do not infer composition readiness even if the timing test improves.


Completed result, matched8000updates per arm:

| Final evaluation | Native control | Relative-PE jitter |
|---|---:|---:|
| TRAIN/oracle CER |1.340%|41.826%|
| TRAIN/predicted duration CER |61.035%|71.466%|
| Native X/Y RMSE |.026838/.003455|.121495/.035301|
| Native target-segment error |.021158|.071887|
| Native pen min F1 |1.000|1.000|
| Previously exposed development8 CER |85.185%|80.247%|

The jitter arm's TRAIN-selected checkpoint is **step0**, not its final model.
Its final native curves are substantially damaged; the development difference
is NOT composition progress. The control's TRAIN-selected checkpoint is8000.
Losses stayed finite; clipping fractions0/.000375, no numerical explosion.
The same target geometry/corners/pens were preserved: corrupting a memorized
positional feature was not sufficient to teach invariance retrospectively.
Do not promote this arm, resume it blindly, replace the good frozen checkpoint,
or claim that this failure rules out a fresh positional contract.

Report: `checkpoints/iam_generation_position_robustness/20261008-095300/report/index.html`.
All264native and256predicted-duration TRAIN outputs are packed and rendered,
with separate first-EOC/pen and source comparisons; no smoothing. The frozen
relative-PE intervention remains useful causal evidence even though its first
proposed training remedy failed. Root/fork suites465tests pass. Both own timing
and jitter Modal apps finished/stopped; other apps were not modified.
Next: a fresh paired test retaining the soft Gaussian prior and absolute index
PE but removing target-length-dependent relative PE, with identical fresh
weights/order/budgets. Report duration robustness separately from TRAIN fitting
and genuine unseen composition. Do not reuse the opened confirmation as blind.

## Fresh positional-contract test (started2026-10-08; not a result yet)

`PositionContractWriter` keeps the soft Gaussian prior, absolute query PE and
trainable PE amplitudes, with identical parameter/state keys. `relative100`
delegates exactly to the original soft model; `absolute` omits ONLY the relative
progress PE. No learned normalization/encoder/decoder changes, target resampling,
smoothing, neural checkpoint warm start, CTC/style/KL, or new held-example tuning.
An absolute query's positional features are independent of requested length;
its self-attention context is **not** invariant and oracle training masks still
leak timing. Removing this feature may improve duration robustness without
solving composition or may worsen fit; both are legitimate outcomes.

Two bounded T4 arms run in parallel, each2CPU/8GiB, torch CPU threads2,
48000updates, same256TRAIN/32writers/batch8/order/RNG/empty Adam. Same fresh
seed28142 and initial weights, first24000 use the original1000warmup1e-5→5e-5,
hold16000/cosine→1e-5 schedule; second24000 retain full Adam and use1e-5.
Schedule seeds29142/39142. Same fixed physical XY/segment anchors
.0048289833417749835/.018978251569199526 and TRAIN32 whitening. Selection uses
all256native TRAIN geometry. Native and TRAIN-estimated durations evaluated at
0/8000/16000/24000/36000/48000, exposed development8 descriptive only. No previous
confirmation reused. Separate TRAIN curve quality, timing stability and unseen
composition gates; oracle native fit alone cannot pass the latter two.
The prepared Volume manifest/source archive is written once by CPU; each GPU
writes only its own directory, no shared mutable checkpoint or retries.

Regression tests cover exact control parity, identical fresh parameter states,
length-independent absolute query features (NOT full output invariance), padding/
NULL/finite gradients, config round trip and exact bounded schedule. Root/fork
473tests pass (465existing +8new). Guarded launcher:
`modal run --detach modal_generation_position_contract.py --train`.

Fresh run path: `checkpoints/iam_generation_position_contract/20261008-101747`,
Modal app `ap-Wa944EDRtMGIt3rOFje6Ft`. An initial CPU-only preparation guard
rejected the independently refitted duration coefficients at~1e-14 BLAS roundoff
before allocating a GPU; refit now uses1e-10 numerical tolerance while requiring
EXACT metadata/scope and rounded prediction counts on all264records. The pinned
original duration coefficients are retained after that independent check.
No data/train-policy change was made to get past the guard.
The first8000update comparison is underfit (native TRAIN64.3%/69.5%CER for
relative/absolute), so no remedy/generalization verdict is inferred yet.

`report_generation_position_contract.py` verifies actual48000updates, empty
initial Adam, regenerated fresh initialization, bitwise paired initial neural
states/RNG, exact minibatches/LRs, frozen evaluators, all256native TRAIN selection,
as-run bytes/hashes, and all estimated-duration packed outputs before rendering.
It reports shorter/longer/unchanged-duration bins, common-prefix XY/pen drift,
termination and CPU/GPU reload; no invented aligned RMSE on unequal trajectories.
Root/fork479tests pass (six additional fail-closed paired-report guard tests).
Completed timing/jitter reviews are separately preserved under each experiment's
`diagnostics/review/index.html`; immutable original report sources retained.

New16paired/16synthetic confirmation reserved with metadata-only seeds26144/26145
before final training completes. Excludes all1032 historical generator IDs/texts,
opened16paired/16synthetic; paired forms also exclude earlier opened paired forms,
current256TRAIN and exposed development8. Archive:
`.../20261008-101747/reserved-confirmation/seal.json` plus exact reservation sources.
The old `seal_synthetic` seed changes writer assignment, NOT prompt text; a trial
reuse correctly failed the duplicate-text guard. New helper selects16hash-ordered
unseen grammatical recombinations from a fixed256candidate inventory, filtering
only transcript duplication/TRAIN character coverage; no model output selection.
This is a small known-writer/corpus-familiar-reader gate, not a broad arbitrary-
text or independent-writer benchmark. No fake oracle lengths/references for
unpaired synthetic prompts. `generation_position_confirmation.confirm` opens
these ONLY after both frozen TRAIN-selected candidates reach native TRAIN CER<=5%,
verifies reserved metadata/exposure/source/checkpoint fingerprints, and never
excludes a reader/source failure from the denominator. No post-confirmation
candidate tuning/reselection. If either remains underfit, keep the seal unopened.
Four additional reservation/fit-gate tests; root/fork483tests pass.
The paired confirmation gallery includes source plus each arm's oracle/estimated
output; the synthetic gallery has only generated outputs and requested text,
not an invented reference. All32prompts are rendered, with correct/swapped/null
controls tabulated separately and source-reader ceiling retained.

## Fresh positional-contract results — completed2026-10-08

`checkpoints/iam_generation_position_contract/20261008-101747`.
Both48000budgets completed, both TRAIN-selected best=final48000, clipping0%.
Control reproduces the earlier soft Gaussian48000TRAIN CER **exactly**.

| Matched-final evaluation | Relative100 control | Absolute-only queries |
|---|---:|---:|
| TRAIN native/oracle CER |2.406%|3.199%|
| TRAIN native exact |152/256|119/256|
| TRAIN estimated-duration CER |61.217%|37.976%|
| Native X/Y RMSE |.032951/.003198|.035523/.003722|
| Target-segment RMSE |.024456|.026913|
| Native pen min F1 |1.000|1.000|
| Exposed development8 native CER |83.128%|81.070%|
| NEW16paired/oracle CER |82.998%|82.998%|
| NEW16paired/estimated CER |81.879%|83.445%|
| NEW16synthetic/estimated CER |81.470%|81.470%|

**Real but limited improvement:** familiar-text estimated-duration CER decreases
23.241percentage points (38.0%relative reduction), without collapse of native
fit or pens. Native curve/segment errors slightly worsen; do not call this a
free improvement or claim expected-duration writing is now usable.
On220changed-duration TRAIN lines, common-prefix mean X/Y drift
.590288/.165055→.500932/.071849; pen changes5937→912 (84.64%reduction), changed-
duration CER70.567→43.637%.36exact-duration lines preserve every pen state and
reader string; XY differences~1e-6 from batching. Longer-duration requests still
fail: CER70.198→44.910%, not just early truncation. Shorter requests include both
prefix distortion and truncation, so do not interpret CER alone.

**Composition still FAIL:** ALL16paired and ALL16synthetic produce no exact
transcripts and visually scribble-like trajectories. The identical aggregate
CERs above are coincidences, NOT duplicated inference: all16paired and all16
synthetic reader strings differ between arms, with different per-line edit
counts/trajectories. Source reader15/16exact,2errors/447characters=0.447%CER;
that line is retained, not excluded to inflate the ceiling. Known writers,
corpus-familiar reader, one seed and small confirmation remain limitations.
These prompts are now exposed and must not be reused as fresh blind tests.
No candidate update, checkpoint reselection or prompt selection after opening.

Independent native CPU reload of8fixed TRAIN samples/arm: maxXY1.91e-5,
zero pen mismatches, all16reader strings equal GPU. Actual weights/empty initial
Adam/RNG/order/LRs/48000rows, fresh regeneration, all256selection and source/
output hashes checked by reporter. T4 loop times1572/1690seconds (32minute
parallel app including evaluation), own app `ap-Wa944EDRtMGIt3rOFje6Ft` stopped,
zero tasks. No user/other app stopped. Telemetry reports low GPU activity without
CPU saturation: not evidence that adding CPU cores would help this tiny batch.

Full marker-free TRAIN/development report: `.../report/index.html`.
New one-shot composition report: `.../confirmation/report/index.html`.
Manually inspected12fixed TRAIN lines (including k04/c03/d08/p10/n05) across
oracle/estimated outputs, and ALL16new paired +ALL16new synthetic outputs.
All264native+256TRAIN estimated and8development estimated saved; not every
TRAIN gallery manually inspected. Original p08-936z-05/a07-421z-02 remain outside
this264scope; do not claim they were retested.

An initial CPU confirmation watcher was interrupted with exit143(SIGTERM,
source unknown) partway through the final synthetic arm. No GPU/training job or
checkpoint lost. Partial directory preserved as
`confirmation-interrupted-sigterm-20261008-105336`; diagnostics rerun with EXACT
same frozen candidates/prompts, no tuning or reselection. Five completed partial
HDF5files (all policy arrays/reader rows) repeat bitwise identically, checked in
`confirmation/repeated-cpu-verification.json`. Original partial outputs/log kept.

**What is established:** the high-frequency, target-length-dependent relative
PE is a major causal familiar-text timing vulnerability; omitting it from fresh
initialization improves robustness, unlike the failed retrospective jitter fix.
It is NOT the sole generation/composition problem. Absolute query features do
not change with length, but unrestricted self-attention context still changes;
its remaining prefix/context dependence is the next timing hypothesis to test,
not something this ablation isolated independently. Better contextual character
representations/learned compositional alignment, a genuinely semantic codec/
generative objective, and more data under a converged regime remain untested
alternatives. Do not enable KL/style/CTC incidentally, blame IAM, call this TrInk/
released InkDiT reproduction, or spend another long run polishing known-line
RMSE alone. Separate a prefix-invariant timing test from a content-composition
intervention; neither gate can substitute for the other. No production promotion.

## Causal-query timing isolation + weak alignment probe —2026-10-08 IN PROGRESS

Fresh paired study `checkpoints/iam_generation_prefix_contract/20261008-113817`.
Initial attempt `.../20261008-112323` interrupted by launcherSIGTERM; preserved
metrics/checkpoints/resources plus `interruption.json`, last logged19735/19543,
last saved16000, NOT a matched completed result. No partial candidate promotion.
Restart is fresh same weights/order/settings, NOT a checkpoint continuation.
Both arms use absolute query PE/soft Gaussian prior/full text cross-attention,
identical fresh weights/empty Adam/RNG,256TRAIN/32writers/batch8/order/shared
losses and the same48000-update schedule. ONLY hidden-query self-attention becomes
causal. This is parallel zero-input prediction, NOT teacher-forced autoregressive
stroke generation. Detached remote coordinator owns both bounded T4 calls,
2CPU/8GiB each, threads2; no other app stopped or performance helper overwritten.
Causal module adds no parameter keys; noncausal arm delegates exactly the prior
absolute model. CPU tests use nonzero readouts (avoid vacuous zero-init passes):
extend/truncate budget, add padding/batch peers/NULL text, disturb future inputs,
and assert gradients from prefix cannot reach future input. No claim about a
trained decoded prefix until frozen-budget audit completes. Codec-only CPU test
on8fixed source latents, three extensions/zero OR random future latents: maxXY
prefix drift1.13e-4, zero pen changes. So decoder independence is approximate,
not bitwise; record latent and decoded drift separately.

NEW16paired/16synthetic metadata reserved before final freezing with seeds
36144/36145, excludes BOTH previous opened sets plus all1032historical generator
IDs/texts, currentTRAIN/dev/previous opened paired forms. Published/readback
verified. Keep sealed unless BOTH nativeTRAIN-selected candidates finish48000
and achieve<=5%nativeTRAIN CER. No new confirmation tuning/reselection; do not
reuse previous confirmations as blind gates. Completed source archives untouched.

Separately, TRAIN-only CPU weak-alignment probe:
`checkpoints/iam_generation_alignment_probe/20261008-112944/report/index.html`.
Frozen prior absolute48000candidate, all256source lines, frozen corpus-familiar
BiGRU. Source reader256/256exact,CER0. Added forcedCTC Viterbi implementation
checked against exhaustive small-path enumeration including repeated-label blank
constraints. Forced timing is NOT IAM segmentation: bidirectional reader lookahead,
delayed marks, blank frames and nonuniform processed spacing all matter.
On forced nonblank emissions (four processed indices/frame), compare character
TOKEN index to static Gaussian center at floor(frame/2) packed8queries:

| Absolute token-index mismatch | median |p90|p99|
|---|---:|---:|---:|
| Static prior |1.338|4.093|7.541|
| Final3local attention conditional-character mean |1.545|4.004|7.485|

26.886%of forced nonblank frames lie outside the prior's2.5-character sigma.
All256line paths/attention arrays/statistics and all32writer summaries saved.
Eight fixed declared TRAIN source/timing/attention galleries generated; k04
manually inspected. Means/heatmaps do NOT establish causal character routing or
rule out useful per-head alignment. This supports investigating learned/weakly
supervised alignment separately from timing, not treating attention means as
true character boundaries. No model updated/selected, no held source opened,
no KL/CTC/style training enabled. GPU outcomes/budget galleries pending; no
production promotion or claim of unseen-text improvement yet.

Infrastructure correction: initial local launcher exited143(SIGTERM, source
unconfirmed); Modal logs explicitly stopped the ephemeral app after client
heartbeat/disconnect loss. Hardened launcher uses `--detach` with ONE durable
remote coordinator which spawns AND waits for BOTH GPU calls. This avoids the
detached-local-entrypoint "retain only last triggered call" hazard. Deliberately
terminated only our own new local CLI to test disconnection; cloud parent and
both GPU children must remain active and complete. Never stopped unrelated apps.
Added two AST launcher regression tests without submitting jobs. Neither initial
nor restarted confirmation reservation has been opened; same still-unexposed
metadata safely re-reserved for the restart before final freeze. Study/model
source from initial as-run archive kept unchanged; infrastructure/report helpers
are separately versioned. Initial partial file upload/readback preserved.

The deliberate disconnect test revealed a second SDK distinction: local
`coordinate.remote()` cancellation canceled the coordinator input, even on a
detached app. BOTH spawned GPU children nevertheless continued (same loss/weight
trajectory, tracked input IDs), so no training restart. Future launcher now uses
`coordinate.spawn().get()` and prints its durable call ID before waiting, matching
Modal's long-training example. The current run's canceled coordinator cannot
write its final orchestration.json: separately retrieve BOTH saved FunctionCall
results and then stop only our detached app once both are complete. Do not claim
the current remote parent remained active; only children survived the test.

Before final candidate freeze/opening, expanded the new composition gate to also
report a constant256-block budget for every paired and synthetic prompt, with
correct/swapped/NULL text and first learnedEOC stopping. This tests the causal
contract without a too-short estimated window confounding composition. Paired
source length only enters post-generation reader diagnostics. Synthetic
reference/window metrics explicitly omitted; no fake oracle/reference. New
unit test asserts fixed mask/window and omitted unpaired target metrics.
Protocol amendment recorded in `reserved-confirmation/budget-protocol.json`,
source snapshots accompany the CPU opening. No new prompt scored/loaded and no
model/loss/data/schedule/checkpoint-selection change due to this amendment.

## Causal-query results — completed2026-10-08

`checkpoints/iam_generation_prefix_contract/20261008-113817`.
Both48000updates complete; both best=final48000 selected on all256nativeTRAIN
geometry, clipping0%, frozen codec/reader unchanged. Noncausal control reproduces
prior absolute48000TRAIN CER **exactly**. Fresh restart's step16000 model,
FULL Adam state and CPU/CUDA RNG match the interrupted run bitwise for BOTH arms;
serialization SHAs differ because metadata/archive provenance differ, not weights.
Restart verification saved under `diagnostics/restart-verification`.

| Matched evaluation | Absolute noncausal | Absolute causal queries |
|---|---:|---:|
| TRAIN/native oracle CER |3.199%|4.318%|
| TRAIN/estimated-duration CER |37.976%|7.140%|
| TRAIN/constant256budget CER |107.543%|4.318%|
| TRAIN/native exact |119/256|89/256|
| TRAIN/generous exact |0/256|89/256|
| Native X/Y RMSE |.035523/.003722|.041753/.004554|
| Target-segment RMSE |.026913|.029846|
| Second-index-difference RMSE |.035694|.038096|
| Mean line tangent p90(degrees) |45.946|50.350|
| Mean line turn p90(degrees) |66.766|71.650|
| Native pen minimumF1 |1.000|1.000|

**Causal timing gate PASS:** every TRAIN line uses the SAME constant256-block
(2048point) window, then stops at the first learnedEOC. No requested source length,
length predictor, target trajectory, sample-ID neural input, or forced finalEOC.
All256causal TRAIN reader strings AND pen states remain unchanged versus native,
all stop successfully. Max common-prefix latent difference6.68e-6, decodedXY
4.20e-5 (FP32/kernel/codec roundoff), zero pen changes. Noncausal extension deforms
all256reader strings and6134pen states, meanX/Y prefix drift.756/.177.
±1block audit further isolates deformation: noncausal TRAIN CER18.975%shorter /
18.247%longer, versus causal6.451%shorter /4.318%longer. Causal shorter has no
prefix pen changes but256missingEOCs: these windows genuinely truncate the learned
line, not a broken stop classifier. Estimated windows similarly truncate107TRAIN
lines; CER7.140% is mostly a remaining duration-budget problem, not prefix warping.
All264(including8exposeddev) policies/latents/decoded outputs saved. In dev,
roundoff/changed post-window content can alter reader strings, so do NOT claim
bitwise readout equality or identical readers for every unseen line.

**Not a free geometry improvement:** native point/segment/direction errors mildly
worsen, and native exact count falls119→89. This change fixes an inference contract,
not the residual native local-curve error. No generic smoothing, target resampling,
new KL/CTC/style training, or semantic-codec claim. Index differences are not
physical velocities/curvature; turn/tangent metrics use geometric directions.

**NEW composition remains FAIL**, despite eliminating length identity from latent
prefixes and providing a genuinely target-free generous budget:

| NEW frozen one-shot gate | Noncausal | Causal |
|---|---:|---:|
|16paired/oracle CER |81.059%|80.652%|
|16paired/estimated CER |80.652%|81.263%|
|16paired/constant256 CER |110.794%|81.670%|
|16synthetic/estimated CER |79.522%|79.522%|
|16synthetic/constant256 CER |90.273%|79.352%|

Zero exact prompts in EVERY arm/policy. Source reader16/16exact,0errors/491chars,
zero source exclusions. Manually inspected ALL16paired and ALL16synthetic outputs
across native/estimated/generous arms: scribbles/false joins/flattened angular
geometry, not readable requested handwriting. All16source trajectories are plainly
handwritten. Timing is now robust on familiar texts; it does NOT solve composition.
There is no reason to call80%CER inevitable or blame IAM. Nor does this establish
causal attention is the right architecture for released diffusion InkDiT.
All32new prompts are now exposed: exclude them from future fresh gates, never tune
or reselect these candidates on them. One seed,32known writers,256TRAIN, small
confirmation and corpus-familiar reader remain limitations.

Reports:
- `.../report/index.html`: full264matched-final native/estimated gallery, guards,
  all256TRAIN selection, CPU reload8fixed lines/arm (maxXY1.53e-5, pens/readers equal).
- `.../budget-audit/report/index.html`: full264TRAIN/dev native versus constant
  target-free budget, all5budget policies/latent versus decoder drift.
- `.../confirmation/report/index.html`: ALL16new paired and16synthetic outputs,
  correct/swapped/NULL controls, estimated AND fixed budgets, no fake references.
- Weak TRAIN alignment: `checkpoints/iam_generation_alignment_probe/20261008-112944/report/index.html`.

Manually reviewed12fixed TRAIN examples (including k04/c03/r07/a02/d08/p10/e07/n05)
in3generous-budget pages, plus ALL32new prompts. All264galleries saved, not every
TRAIN line manually inspected. Original p08-936z-05/a07-421z-02 outside this264scope,
NOT retested. Native causal renders remain close but not pixel-identical to source;
this test is not a new smooth-curve/lossless reconstruction promotion.

Infrastructure: initial ephemeral app stopped after unknown localSIGTERM;
partial19735/19543logs,16000checkpoint, source archive/resources preserved.
Detached restart survives local disconnect for BOTH spawned GPU children; its
remote() coordinator input was canceled by the deliberate disconnect test, but
both child calls independently collected after completion. Future launcher now
uses durable `spawn().get()` parent call (AST tests; not a second training rerun).
Own restart app stopped automatically12:14:14UTC; additional frozen-budget T4
app stopped12:19:34UTC, zero tasks. No unrelated app stopped. Train loops
1900/1839seconds, each2CPU/8GiB; budget-only GPU~2.5minutes including load/audit.
Telemetry: low GPU activity WITHOUT CPU saturation, not a reason to add more cores.

First CPU confirmation attempt failed immediately on a missing `hashlib` import,
before ANY generated output; empty source/attempt/source-code saved as
`confirmation-failed-missing-import-20261008-121814`. Fixed import + unit test on
the actual source fingerprint guard, then ran EXACT SAME frozen candidates/prompts,
with no tuning/selection/exclusions. This was not a GPU or dataset failure.
510tests pass root/fork; guarded GPU-budget evaluator and synthetic no-fake-target
contracts tested. Reusable code pushed to existing `english-iam` branch.

**Next recommendation:** retain this prefix-invariant mapper timing contract as a
controlled research baseline, but stop spending long runs polishing its familiar
TRAIN RMSE. Investigate actual character composition: test TRAIN-only weak forced-
CTC alignment supervision (explicitly not exact IAM segmentation) versus the same
causal control, with a small gradient-calibrated auxiliary and identical data/
order/budgets. Contextual text encoding, an autoregressive offset/semantic-codec
objective, and more data under a converged schedule remain alternatives, not ruled
out. Do not assume attention means establish causality; test the intervention.
A NEW composition seal must exclude all THREE opened confirmation sets. No fullIAM,
KL/style, releasedInkDiT launch, or production handwriting promotion from this run.

### 2026-10-08 — controlled TRAIN weak-alignment intervention (completed; not promoted)

New paired T4 run: `checkpoints/iam_generation_weak_alignment/20261008-130852`
(app `ap-Xgdja664202IegAcVTjAZY`, durable coordinator
`fc-01M4DT25ZDKAA4P33A9WJKD7ZQ`). Fresh causal absolute-query control vs the
SAME architecture with weak forced-reader character-attention supervision. Same
256TRAIN/32writers, seeds28142/29142/39142, batch8,48000updates, schedule/base
losses/codec/reader as completed causal-query experiment. No incidental KL,
OCR reconstruction CTC or style objectives. Still the standalone deterministic
mapper, NOT released InkDiT or semantic InkVAE.

Teacher is immutable TRAIN probe `.../iam_generation_alignment_probe/20261008-112944`
(`alignment.h5` SHA2cc1fc2c9c9c1856f238fe18fbb746ab2c73aa96242091298ce2aa6e625f1205).
Its source reader reads256/256TRAIN exactly. Each packed8point query receives its
two4index forced-CTC token emissions: omit blanks, include spaces, average the two
labels if both occur; no invented interpolation across blanks or exact glyph
boundaries. Median65.6% queries supervised;1010queries contain two distinct tokens.
Use CE on final block's3local heads only; fourth/global head remains unconstrained
for delayed marks. Bidirectional lookahead/irregular spacing/delayed marks remain
limitations; this timing is weak, NOT exact IAM segmentation or an independent
reader benchmark.

A separate differentiable QK/log-softmax readout retains the original
`need_weights=False` SDPA forward. Regression tests compare against MHA weights,
prove nonvacuous routing gradients, exact unchanged main outputs/state/RNG, no
supervision on held/padded queries, no global-head/V-row gradients, and unchanged
causal prefix/checkpoint contracts. Actual CPU3update preflight also has ZERO
capture-induced output change; it is NOT initialization/selection for GPU runs.

Calibration at fresh step0 would be WRONG: zero-initialized final readout makes
base body gradients zero. Both arms instead receive identical1000base-only updates;
on the shared state measure first8TRAIN minibatches without updates or RNG/state
changes. Fixed coefficient `0.0015943569375687838` gives auxiliary10% of median
base-objective body gradient norm, excluding final readout. Control computes the
same diagnostic but uses weight0; intervention activates at1001. Both calibrations
and warmup state digests are identical in actual cloud logs. No claims about new
text yet: decreasing teacher CE is not evidence of composition.

**Metadata-only reservation correction:** the third exposed synthetic set reused
`contract-synthetic-*` keys from the second. Plain `history.update()` erased16old
transcripts; initial new synthetic reservation accidentally repeated2exposed texts.
Detected BEFORE any confirmation sources/output scoring or final candidates.
Namespaced synthetic history now preserves all48exposed synthetic texts. Same
seeds46144/46145; paired IDs unchanged,2synthetic texts replaced. Original
config/archive/reservation are byte-immutable; repair/provenance in
`reserved-confirmation-corrected/`. Actual training/RNG/targets/order/selection
UNCHANGED. Corrected evaluator reproduces full three-set metadata union. This is
not performance-based reselection. Original as-run preparation helper is preserved;
training sources remain unchanged. Regression test explicitly reuses synthetic IDs
across all three sets. No mass extraction/new inode pressure beyond packed files.

Final evaluation must first freeze all256TRAIN-selected candidates and pass native
TRAIN CER<=5% in BOTH arms. Otherwise do NOT consume the corrected fresh gate.
Report native geometry/pen/CER and TRAIN weak timing separately; generous256blocks
(first learnedEOC, no oracle/predictor) must be the target-free composition control.
Open the new16paired/16synthetic prompts once only, with swapped/NULL text controls
and source-reader preflight; render ALL prompts marker-free, no exclusions. Do not
infer English composition from familiar reconstruction or attention means alone.

#### Completed weak-alignment result — training routing learned, composition FAIL

Fresh48000updates completed with nativeTRAIN CER4.318%control /6.438%supervised.
The latter failed predeclared5% fit gate; no confirmation source was opened.
Matched12000update continuation (BOTH arms, own48000weights/FULLAdam/CPU+CUDA RNG,
constant1e-5, new shared schedule seed49142, original fixed coefficients unchanged):
`checkpoints/iam_generation_weak_continuation/20261008-134416`.
Both selected final60000 by ALL256TRAIN geometry; nativeCER2.289% /3.811% passes
fit gate. Original48k run remains immutable; continuation records and verifies its
parent configs/data/results/checkpoints and full moment/RNG restoration. No held
performance triggered the extension or selection. Initial weights/emptyAdam/RNG,
1000warmup weights/fullAdam/RNG, actual schedules/LRs/losses/source archive/data
and frozen codec/reader independently verified. Shadow-readout control48k model
is BITWISE equal to previous causal48k model: the separate attention readout did
not secretly change forward kernels, optimization, RNG or results.

|60000 native TRAIN metric|Control|Weak alignment|
|---|---:|---:|
|CER|2.289%|3.811%|
|Exact reader strings|140/256|117/256|
|X RMSE(model space)|0.037070|0.037435|
|Y RMSE(model space)|0.004105|0.004396|
|target first-index-difference vector RMSE|0.025613|0.027973|
|target second-index-difference vector RMSE|0.034792|0.037294|
|mean per-line tangent p90(deg)|44.406|47.936|
|mean per-line turn p90(deg)|66.023|70.372|
|min pen-up F1|1.000|1.000|
|non-final false EOC / missing EOC|0 /0|0 /0|
|TRAIN forced-timing CE|3.391034|0.095536|
|TRAIN target-weighted label probability|0.124335|0.931956|
|mean line attention-index p90|3.358333|0.420982|

**Important metric definition:** stored `target_mass` is target-distribution-
weighted label probability E_target[p(label)], NOT probability summed over the
set of all correct labels.1010/7669supervised queries have two distinct emissions,
each weight.5. Therefore irreducible teacher CE is0.091287 and weighted-probability
ceiling0.934150. SupervisedCE0.095536 is very near that floor, not an inadequately
trained head with7%remaining ordinary accuracy error. Blanks are not interpolated;
all timing is processed INDEX timing, not physical time/velocity or true borders.
This correction makes increasing the existing auxiliary's weight a weak next bet.

**One-shot NEW16paired/16synthetic, constant256blocks/2048points, first learnedEOC:**
no source/oracle length, predictor, target/prefix trajectory or forced finalEOC at
inference. Corrected pre-training metadata seal excludes all THREE earlier opened
sets. Source reader16/16exact,0/509character errors; codec preflight passed every
line; no source/reader exclusions. Both frozen60000candidates opened only once.

|Fresh confirmation CER|Control|Weak alignment|
|---|---:|---:|
|Paired/native oracle budget|80.354%|78.193%|
|Paired/TRAIN-estimated budget|79.568%|80.550%|
|Paired/constant256 correct text|81.729%|80.943%|
|Paired/constant256 swapped text|84.872%|83.104%|
|Paired/constant256 NULL text|112.181%|99.804%|
|Synthetic/TRAIN-estimated budget|78.224%|80.570%|
|Synthetic/constant256 correct text|78.894%|80.067%|
|Synthetic/constant256 swapped text|79.899%|81.072%|
|Synthetic/constant256 NULL text|93.970%|100.000%|

ZERO exact prompts in every arm/policy. Constant-budget correct/swapped all32
successfully stop; NULL can missEOC. Native paired15control /16supervised windows
missEOC: oracle source length can genuinely truncate a different generated line,
not an excuse for composition failure. Paired point-index geometry/pen alignment
is only a reference-index diagnostic: a legitimate new generated trajectory need
not match the source's point spacing, boundaries or duration. Primary metrics are
free-stop CER and actual marker-free visuals, not unseen pointRMSE.
Manually inspected ALL16paired and ALL16synthetic constant-budget outputs for BOTH
arms, plus12fixedTRAIN lines across3galleries. Familiar writing looks close with
some local shape changes/truncation under estimated budgets. New outputs remain
angular scribbles/false joins/flattened shapes, NOT requested readable handwriting.
The tiny paired0.79percentage-point change is not a useful quality win; synthetic
worsens1.17points. One seed/small known-writer gate, not a statistical reproduction.
Not all256TRAIN inspected manually. Original p08-936z-05/a07-421z-02 NOT retested.

**Frozen mechanism controls, not another training run:**
- `attention-usage/`: all256TRAIN latent-field XY/pen drift, actual frozen-codec
  decode/reader on8fixedTRAIN. Rotate ONLY character-embedding VALUES in last3local
  heads, leaving BOS/positional values/keys/queries/all earlier layers/correct text
  unchanged. Fixed8CER2.165%→59.307%control /46.753%supervised. Zero local contexts:
  77.489% /84.416%. Global value rotation:32.468% /40.693%. Manual same-weight
  readout maxlatent drift3.81e-6/4.29e-6, no pen/reader changes, state/RNG unchanged.
  This rules out the simplistic claim that those heads or character values are
  merely ignored/decorative. It does NOT establish per-character composition, nor
  remove text from earlier layers. Local token rotation causes2030control versus
  only2supervised raw pen changes across256; learned pen timing can remain memorized
  while glyph geometry changes. This is an ablation, not valid new-text output.
- `confirmation-timing-probe/`: AFTER opening, same frozen models/full256query
  budgets versus source forced-reader timing on ALL16paired, no updates/selection.
  ControlCE3.2953 /supervised7.3882; mean line indexp90=3.3259 /3.8152, p99=4.0023 /
  6.4349. SupervisedTRAINp90=.421 does NOT carry to new source timing. Attention is
  sharper but often confidently wrong relative to the weak reference; means alone
  obscure this. Source-reader directlatent16/16exact. Different legitimate output
  timing could differ from source, and BiGRU/delayed marks confound true borders:
  do NOT call this exact generated-glyph alignment ground truth or claim alignment
  has been ruled out. Current actual outputs are nevertheless plainly unreadable.

**What this establishes:** data/reader/codec checks remain healthy; learned text
conditioning and familiar capacity are real. Budget deformation was fixed earlier,
and the same causal architecture is retained. Supervision can fit TRAIN routing
near-perfectly, but has not produced transferable timing/letter-to-stroke behavior.
The generalization bottleneck remains; no evidence82%CER is inevitable or IAM bad.
This is NOT evidence released semanticInkVAE/InkDiT fails. No KL/style/new OCR loss,
fullIAM or production handwriting promotion. Both48k and12k apps stopped normally,
zero tasks; no unrelated apps stopped. Loops1609.65/1662.17sec plus507.15/570.48sec,
0%clipping; each T4/2CPU/8GiB, threads2. Continuation GPUbusy~35–36%, treeCPU~.95core,
hottest~.55core: lowGPU alerts WITHOUT CPU saturation, no evidence more cores help.
Volume98%inodes: packed targets/checkpoints, no mass extraction.

Review: continuation `diagnostics/engineering-review/index.html`; full264matched
native/estimated `report/index.html`; fresh ALL32 `confirmation/report/index.html`;
mechanism `attention-usage/index.html`; post-opening weak timing
`confirmation-timing-probe/index.html`.48k parent report also retained.

**Next recommendation:** do not keep sharpening this near-floor TRAIN attention
loss or polishing familiarRMSE. Test explicit text-derived monotonic progress /
within-character phase (and/or genuine autoregressive offset prediction) so timing
and stroke primitives must transfer instead of identifying a familiar whole line.
Start with TRAIN-form-held CPU timing diagnostics before a fresh controlled GPU
comparison. A deliberate generated-trajectory semantic reader objective and more
data under a CONVERGED regimen remain alternatives, not ruled out; contextual
encoding alone may increase memorization. No incidental KL/style milestone.
A NEW blind gate must exclude all FOUR now-opened confirmation sets, including the
CURRENT corrected seal; namespace synthetic history to avoid ID collisions.

**Supplemental frozen objective decomposition (2026-10-08):** checked whether the
344 unused polyphase transport channels dominate the latent loss. Across all256
TRAIN samples /11824valid blocks at each selected60000 checkpoint, their fraction
of total whitened latent squared error is only0.008619%control /0.009258%supervised.
Active40 per-element MSE=.000250155 /.000297012; unused344=2.50734e-9 /3.19757e-9.
Source unused channels are numerical residue (RMS1.6506e-7), not semantic features.
Unused-channel loss dominance is therefore not the problem at these checkpoints;
this is a descriptive decomposition, NOT a causal masked-loss experiment. No model
update, new held-source access, or selection. Supplemental artifact:
`diagnostics/engineering-review/unused-channel-diagnostic.json` under the same
continuation run. Published original review/index/manifest remain immutable.

## Text-only monotonic progress: TRAIN-form-held CPU diagnostic (2026-10-08)

Run `checkpoints/iam_character_progress/20261008-145649` on `diffink-data` and
local `data/`; `index.html` has plots, `summary.json` all256 rows, `predictions.h5`
all256 packed clock outputs, `fold-*-models.json` five fitted sets, and
`full-train-model.json` the reusable but NOT promoted text-only clock.

**Question:** can an explicit compositional text/writer clock predict transferable
left-to-right progress, unlike the earlier almost-perfect TRAIN attention that
failed on new prompts? Use precisely established256TRAIN forced paths, no new
reader run, trajectory extraction, neural checkpoint update or GPU. Eight exposed
DEV and all FOUR opened confirmations untouched. Reader corpus-familiar; this is
NOT independent unseen-text recognition or generated-writing evidence.

Five fixed form-held folds (seed57142), with normalized-transcript duplicates
connected across forms before splitting. Every256line evaluated exactly once on
a model never fitted to its form or normalized text; all32 writers represented in
each fit. Unknown fold characters (e.g. digits/capitals/punctuation) use category
fallbacks; no examples excluded. Frozen source reader256/256exact. Complete
form/text TRAIN/evaluation disjointness guarded. Exploratory CPU pilot on these
same TRAIN sources preceded the fixed probe; do NOT call it preregistered/blind.

Teacher center = mean((four-index emission-frame+.5)/2) for each transcript token;
positive intervals between origin/centers/ceil(realpoints/8). These are processed
index-block coordinates, NOT physical time, actual character borders or pen phase.
Blanks are omitted from labels, not converted to invented glyph boundaries.
Positive log-gap ridge regression, fixedL2=10, equal line weight, one TRAIN-only
arithmetic smearing factor. Features: writer intercept, simple position/length,
optional character/category and optional left/right neighbor identity. No sample
ID, source length, held path, whole-text neural embedding or trajectory input at
primary inference. Monotonic interpolation between centers is an explicit weak
phase scaffold; prefixes are invariant to the caller's output-budget length.

|Out-of-form clock|Pooled median index error|p90|p99|Mean linep90|Nearest token index accuracy|
|---|---:|---:|---:|---:|---:|
|Static TRAIN-average blocks/character|1.404|4.097|7.315|3.224|21.84%|
|Existing text/writer duration + uniform progress|1.193|3.127|6.454|2.529|23.05%|
|Same gap model, no glyph identity|1.094|3.331|6.382|2.569|25.62%|
|Shared character-gap coefficients|1.000|3.109|5.975|2.442|30.46%|
|Left/right neighbor coefficients|0.985|3.057|5.710|2.425|31.47%|
|Uniform spacing, same predicted first/last centers|1.111|3.049|5.600|2.481|25.21%|
|**Oracle** first/last centers + uniform|0.669|1.911|3.355|1.594|40.02%|
|**Oracle** first/last centers + neighbor spacing|0.412|1.359|2.663|1.188|56.74%|

Metrics compare weak progress token INDEX to forced nonblank token index; nearest
index accuracy is NOT character recognition accuracy or CER. Half-frame convention
differs from prior attention probe's floor(frame/2), so don't directly compare the
old attention numbers. Neighbor durationMAE=3.586blocks vs existing duration3.574;
full256 in-sample neighbor p90=2.780, not tiny memorized timing error. All256 stored;
manually inspected12plots: six worst neighbor linep90 and six around its median.

Descriptive2000resample form-cluster bootstrap, seed57143, mean linep90 differences
neighbor-minus-baseline: static -0.799[-1.043,-0.576]; duration-uniform
-0.104[-0.252,+0.053]; no-glyph -0.144[-0.309,+0.015]; same-endpoints-uniform
-0.056[-0.137,+0.029]. Models/folds are NOT refitted per bootstrap; intervals omit
reader/fold/seed uncertainty and multiplicity adjustment. Current256 has no
cross-form identical normalized texts, explicitly checked for bootstrap grouping.

**Established:** cheap local identity information transfers somewhat, particularly
median/index assignment, but fixed predicted pacing still accumulates substantial
error against source emission clocks. Worst plots show estimated progress reaching
the last token while source emissions are several tokens behind, or vice versa.
Oracle-endpoint controls expose much stronger reusable local spacing information;
global delay/span uncertainty dominates much of its metric advantage. A fixed
clock alone is not a demonstrated generation fix, and evidence over the simpler
text/writer duration model is weak. Different legitimately generated handwriting
can have a different clock from this one source: these numbers do NOT prove that
source timing mismatch caused the ~81%CER failure or that text-only generation is
impossible. The reader's delayed emissions/lookahead remain confounds. No generated
trajectory, pen policy, curve-quality or new CER improvement claimed.

**Next research:** do not silently replace production attention by a hard frozen
schedule or spend another48k run merely sharpening it. Test an adaptive monotonic
progress mechanism tied to generated stroke history (a genuine autoregressive
index-displacement/pen baseline), with teacher-forced vs free-running evaluation
separated, explicit learned stopping, and marker-free novel-text visual gates.
A TRAIN-derived phase scaffold/target reparameterization or deliberate generated
semantic-reader objective remains viable; the diagnostic alone does not rule them
out. Continue to exclude all FOUR opened gates plus all future training prompts
when reserving a fresh confirmation. Original semanticInkVAE/releasedInkDiT are
still NOT tested by this standalone mapper. No incidental KL/style/fullIAM.

## Genuine autoregressive offset pilot — prepared (2026-10-08)

The fixed clock diagnostic motivates testing feedback-based progress, not claiming
it already solves composition. Added a separate small recurrent research model;
original mapper/InkVAE/checkpoints remain unchanged. This is NOT semanticInkVAE or
releasedInkDiT. Two same-weight arms: fixed TRAIN-average positive text advance
versus a learned positive advance/sigma driven by previous stroke history. Single
Gaussian text window; character embeddings + explicit textEOS, no whole-text neural
summary or absolute query identity. Both two192-wide GRUCells,64-wide text vectors,
16-wide writer embeddings. Chronological8-point outputs are normalized XY index
DISPLACEMENTS (including pen jumps and origin-to-firstpoint) and8three-state logits.
They are not physical velocities. Normalization exclusively established256TRAIN.

Training receives only PREVIOUS true blocks, rigorously shifted: target blockj
cannot enter predictionj. Free generation starts zero/BOS and feeds back ONLY its
own predicted offsets and hard pen states. It accumulates offsets, stops at first
learnedEOC, and uses the same2048point safety cap for every prompt. Source lengths,
timing teachers, reference strokes and duration predictors never enter generation.
Teacher-forced cumulative output can still use true history; it is NOT generation.

Fixed seed58142, order58143, batch16,8000updates/arm,3600loop-second wall cap,T4,
2CPU/8GiB,2torchthreads. LR2e-4,200update warmup,5e-5 from6000; AdamW(.9,.99),
weightdecay1e-4,clip5. Normalized offsetMSE plus bounded English focalpen(gamma2,
continue-normalized square-root frequency cap8), with common frozen pen coefficient
calibrated to25%initial full-model coordinate gradient on first8TRAIN minibatches.
Calibrate adaptive surrogate for fixed control to retain identical objective scale.
All weights fresh; no sampler randomness/dropout/noise/scheduled sampling. This is
an initial deterministic AR comparison, not a claim that MSE captures all possible
handwriting continuations. Exposure bias and offset accumulation are explicit risks.

Evaluate all256teacher-forced geometry/pen and fixed8TRAIN + already-exposedDEV8
free outputs at0/1000/2000/4000/8000; swapped text at8000. TRAIN-only selection:
fixed8TRAIN freeCER then all256 teacheroffsetMSE. Reader frozen/corpus-familiar;
no new blind gate opened, no KL/style/OCR training loss, no production promotion.
Durable Modal coordinator owns both GPUchildren. Checkpoints include fullAdam/RNG,
metrics/outputs packedHDF5, as-run source archive and resource telemetry retained.

Launch (guarded; without `--train` no GPU):
`venv/bin/modal run --detach modal_autoregressive_study.py --train`.
Regression checks include no current/future teacher leakage, identical initial
fixed/adaptive forwards, monotonic center, partialblock/jump/backtrack roundtrip,
padding exclusion, target-free budget-invariant prefixes, learned-vs-capped stopping,
TRAIN-only calibration and durable boundedlaunch. Results NOT YET established.

Packaging correction before training: first remote attempt
`checkpoints/iam_autoregressive_study/20261008-152128`, app
`ap-zubLQkIJrHewQSYgDcY8TU`, failed BEFORE initialization/calibration/updates because
`model.__init__` eagerly importsVAE, which imports`utils`. BothGPUcalls failed;
app authoritatively stopped/zero tasks. No research result or cost comparison from
that attempt. Include`utils` in image/archive and probe the eager import in CPU
prepare BEFORE spawning GPUs; added launcher regression. Preserve failed attempt,
start fresh directory rather than overwriting/retrying partially trained state.

**Pilot actually launched:** corrected app`ap-dkblx956dzT6Vk7wD4S33F`, durable
coordinator`fc-01M4E1WGH61XSZHZFV6PQ0XNDE`, output
`checkpoints/iam_autoregressive_study/20261008-152547`. Same initial weights and
step0 outputs, minibatch order and gradient coefficient botharms verified. Common
penweight=.037631938059517804. Source-reader8fixedTRAIN exact in localpreflight;
sourceSHA/readerSHA pinned.570tests passed bothtrees before launch/report changes.
Progress is committed at fixed evaluations, without changing training/RNG.

Completed1000update interim diagnostic (NOT final outcome): fixed/adaptive
all256 teacheroffsetMSE=.816244/.815346, cumulative XRMSE=.9556/.9793,
YRMSE=.6805/.6791; minpenF1=0, finalEOC10/256vs14/256. FreeTRAIN8CER97.95%/134.02%,
exposedDEV104.12%/349.38%; both miss4TRAIN EOCs, all8DEV EOCs and can run to cap.
Every16freecorrect output manually inspected: flat drifts/periodic small waves,
NOT readable handwriting. Regression/coordinates/teacher-vs-free separation hold;
poor outputs must not be presented as improvement. Teacher forcing itself is still
underfit, so this is NOT yet a clean isolation of exposure bias or text composition.
Jobs continue on their original bounded schedule; no interim tuning or newgate.
`progress-1000/index.html` is a clearly labeled completed-step interim report.

A useful objective diagnostic: real target pen-up jumps are5829/93687=6.22%points,
but account for59.97%normalized displacement variance (zero-normalized prediction
baseline). Origin256points5.21%; drawn within-stroke87602points34.82%. Atfixed1000,
actual all256teacher displacement squared error is62.92%pen-up jumps,36.08%within
stroke,.996%origin. Jumps per-axisMSE8.2548vs.3150within strokes. This is descriptive
error accounting, NOT evidence downweighting them causally helps; resume locations
matter and must not be erased. `offset-error-decomposition.json` coversbotharms in
report; new tested reusable helper uses PREVIOUS truepen state, correctly assigning
the move after a lift rather than the stroke's final inked segment. Possible next
objective hypotheses include robust/stratified penalties or a multimodal jump head,
AFTER checking the completed matched pilot, not an unannounced mid-run change.

Interim telemetry: fixed32.96%GPUbusy,~.97CPUtreecores/~.53hottest on2CPU, alert
`gpu_low_no_cpu_saturation`. No evidence3more cores solves this. Small sequential
GRUCell kernels/recurrent dispatch are plausible throughput limit, not proved yet.
Measured loop at1000: fixed115.4s/adaptive194.6s. Do not restart or alter optimization
solely because a GPUbusy alert fires. Volume98.3%inodes; all outputs/checkpoints packed.

New SVG report initially used down-positive display for up-positive modelY; caught
by comparing source letters to existing matplotlib/preprocess convention BEFORE
publication, corrected with explicit Y flip and renderer regression. This was a
new report-only bug, NOT training/data mirroring or an explanation of generation
failure. Unpublished local diagnostic drafts retained under clearly named paths;
only correctly oriented progress report is published. No training code changed.

Progress1000 publication:102artifacts byte-readback verified,82relativeHTML links,
zero overwrites, `progress-1000/publication-verification.json`. Primary SVGs use
fixed100px/model-unit and correct up-positiveY conversion; four temporary fit-to-
panel contact sheets are explicitly diagnostic-only.574tests pass both mirrors.
Upstream rechecked2026-10-08: officialmain remains97bc6a3, no public multilingual
replacement to adopt.

Renderer follow-up for FINAL/future reports: actual one-point strokes (36/6085
trueTRAIN strokes) must not disappear just because vertex markers are disabled.
Represent them with a round-capped zero-length SVG ink segment, not a marker at
every sampled point; regression added. Published progress1000 stays immutable and
omits those isolated taps. This is visualization-only, no loss/model/evaluation
metric change and no explanation of the current flat free-running failures.
