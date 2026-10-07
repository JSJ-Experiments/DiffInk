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
