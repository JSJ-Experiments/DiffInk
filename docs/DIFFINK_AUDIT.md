# English IAM compatibility audit

Upstream: https://github.com/awei669/DiffInk

Inspected commit: `97bc6a3c39a5bdaa9728daaab6d3707480006343`.
Branch: `english-iam`. Initial audit was CPU-only; the approved first T4 run and its failed reconstruction gate are documented in `ENGLISH_IAM.md`.

## Confirmed from released code

- `utils/visual.py` reads **absolute** x,y and connects point i to i+1 only
  when column 2 (zero-based) is 1. Its OpenCV renderer flips positive-up model
  y for the display. IAM raw y is down-positive.
- The last three columns are one-hot continuation, stroke end, character end.
  Character-end semantics come from `plot_line_cv2_new`; the paper's prose is
  inconsistent in one appendix paragraph, so we retain the implementation's
  interpretation rather than treating all stroke ends as character ends.
- `TrainDataset` accepts 200–2000 points and returns writer string, trajectory,
  text, and character index array. The English patch leaves this class unchanged.
- Padding is `[0,0,0,0,1]`, to a multiple of 8. Our x offset of +1 prevents
  real endpoints from colliding with that exact sentinel.
- Metadata is `chars.json` object key order and `writers.json` with a `train`
  list. Character IDs begin at 0; OCR adds 1 before CTC(blank=0), so space is
  a real character and does not collide with the CTC blank. No extra blank
  character should be inserted in `chars.json`.
- The loader's augmentation is random geometric scaling/rotation, NOT a
  normalization or an IAM raw-data converter. No released raw preprocessor or
  normalization formula was found in this pinned tree.

## Small English patch (`patches/diffink-english.patch`)

1. Validation appends Chinese `、` only if it exists in the vocabulary. Pure
   English validation returns exactly the supplied line text.
2. Empty `char_points_idx` explicitly means unaligned IAM. The prefix function
   then requires the actual `point_seq` and chooses an **exclusive actual
   stroke end** nearest 30% of real points. It refuses to invent char indices.
   All four DiT callers pass the trajectory. Original aligned Chinese behavior
   remains intact. The third output is a temporal suffix mask, as in upstream;
   it is NOT a newly invented character-label mask.
3. Stroke-only inference uses the non-character-truncating plotter, since the
   released plotter otherwise returns without drawing if there are fewer than
   two character ends.
4. OCR uses the exact CTC minimum: target length + adjacent repeated labels.
   Upstream's `2 * target_length - 1` is an unnecessarily conservative bound
   that silently rejects valid English lines after 8× compression. A CPU
   evaluation-only test checks that a 3-step, 3-distinct-character case has
   a finite nonzero loss. No backward/optimizer/training is involved.

## Experimental, NOT verified paper-equivalent

The line-height normalization is an explicit baseline: preserve aspect ratio,
set raw y range to 100 units, x begins at +1, and y is flipped to positive-up.
Apply independent per-stroke point-to-segment RDP with epsilon 0.5 **after**
normalization. Coordinates/timestamps stay intact in raw canonical JSON. Only
retained points are passed to N×5. Each stroke's final point is pen-up; the
line-final point alone uses the third state. No precise internal character
boundaries are fabricated.

This proves **format/loader compatibility**, not reproduction of the authors'
English experiment. The actual English coordinate scaling and end-state policy
still need confirmation from authors or their prepared English data. The
Chinese checkpoint must not be advertised as English-ready merely because
its architecture accepts five columns.

## Tiny data checks

- Fixed seed 42; reserve 25 writers for untouched future testing, and
  five different writers for validation, before vocabulary/sample selection.
- Training vocabulary comes from all safely paired training transcripts,
  excluding BOTH test and validation writers; validation OOV is rejected/reported.
- Tiny selection is random within those writer-disjoint pools: 250 train,
  50 val. Length-filter and exact-CTC-infeasible candidates are reported rather
  than artificially inflated/truncated to satisfy the loader.
- JSON manifest records excluded forms, unknown marker cases, and accepted IDs.
  Additional `stroke_points_idx` stores EXCLUSIVE stroke ends; the legacy
  `char_points_idx` field is present but empty. A patched DiT mask is mandatory
  for those empty alignment arrays.
- Canonical JSON is the clean source; HDF5 contains derived arrays with a
  normalization manifest and retained acquisition timestamps.

Remaining gates before any paid GPU training: review before/after previews,
confirm the experimental normalization/end-state choices are acceptable, and
plan an InkVAE overfit experiment. LaTeX/Markdown layout is not implemented.

## Observed local CPU results

- Safely paired inventory: 12,112 lines from 216 writers; 11 rejected forms.
- Actual training-only vocabulary: 81 characters.
- Tiny HDF5: 250 train lines from 144 writers; 50 val lines from five writers;
  25 separate writers reserved for future testing. 30 candidate lines rejected
  during tiny selection. Processed lengths 201–739.
- Train batch: `[4, 744, 5]`; validation batch: `[4, 576, 5]`. Both padding,
  text round-trip, state/sentinel checks, and stroke-prefix masks passed on CPU.
- All accepted targets meet exact CTC feasibility after 8× compression.
- 16 unit/integration tests pass, including export through the real released
  loader and unchanged legacy character-prefix behavior.
- Three actual before/after pairs were visually inspected in a contact sheet:
  orientation, letter shapes, and stroke separation were preserved. This is a
  spot check, not a review of all 300 labels or a mathematical quality claim.

## Follow-up correction: real VAE CTC path and mechanics preparation

The initial CTC patch/test covered `ChineseHandwritingOCR.get_ocr_loss`, but
**not** the separate duplicate implementation invoked by `VAE.forward`.
That omission is corrected: `VAE.get_ocr_loss` now delegates to the shared OCR
implementation. Four regression tests invoke the actual `VAE.forward` path,
including distinct targets at the exact minimum length, padding, repeated
labels, and equality with the shared helper. CTC lengths are copied to CPU
before CTCLoss for backend portability. All-invalid targets return a
model-connected zero instead of a detached leaf constant.

The user's reported eligibility counts were reproduced: the old VAE filter
would wrongly reject 241/250 training and 47/50 validation lines. The correction
changes no IAM coordinates, RDP points, transcripts, or writer splits.

Builds now stage both generated directories and replace them only after a
complete successful conversion. Successful rebuilds remove stale files;
failed conversion preserves previous artifacts. The raw directories are
protected from output overlap. Local tiny artifacts now have exactly 300 raw
JSONs and 10 comparison previews. Publishing prunes stale generated remote
JSONs/previews after uploading a successful build, and never modifies raw IAM.

The dedicated `iam_overfit` dataset uses eight non-test, non-held-out-val
writers: 24 training lines each (192 total) and four different validation lines
each (32 total). This intentionally **shares writer identities**, not line IDs,
for a mechanics/style-classifier check; it is not a held-out-writer benchmark.
The original 250/50 writer-disjoint loader dataset remains separate. The 25
future-test writers remain excluded from both datasets.

### Additional numerical findings from the full CPU model

A real 13,633,109-parameter VAE forward check, not just loader instantiation,
found NaNs on direct height-100 input: the first line's learned log-variance at
random initialization reached approximately 168. The new single-GPU mechanics
config uses a **reversible model-input multiplier 0.01** on x,y. Stored HDF5,
canonical JSON, and RDP remain unchanged; predictions are converted back for
visualization, and the multiplier is saved in checkpoints. Thus the model sees
height 1 and effective RDP tolerance 0.005. This is an experimental numerical
adapter, NOT the authors' verified English normalization.

The upstream probability-space GMM loss also saturates at its `+1e-8` density
floor for ~25% of this first batch even after that adapter. The new English
mechanics runner evaluates the same correlated Gaussian mixture in log space
using `logsumexp`, and masks both coordinate and pen losses at padded points.
It does not silently change the original trainer/GMM functions. A static
loss-gradient test checks nonzero finite mean gradients at far coordinates;
no optimizer or model update is involved.

With the explicit adapter and log-space loss, the full CPU forward check
passes: input `[1,584,5]`, output `[1,123,584]`, eight writer classes, 82 OCR
classes (81 vocabulary characters + CTC blank), finite losses, CTC ≈6.19 and
style ≈2.26. The report records zero optimizer steps, config/manifest SHA256s,
and both experimental caveats. This does not validate CUDA execution or prove
that a model can reconstruct training trajectories yet.

Prepared, not executed: `configs/vae_iam_overfit.yaml`, `modal_inkvae.py`, and
`iam_tools.inkvae`. The Modal job is one L4, at most 200 optimizer steps and
600 training-loop seconds, with a 900-second container timeout. Both `--train`
and `--allow-experimental` are required to invoke the remote GPU function.
The CPU smoke report must match the exact config and dataset before optimizer
creation. Metrics, checkpoints, and train/val reconstruction previews are saved
on the Volume. No GPU has been allocated or training run during these fixes.

Follow-up suite: 25 tests pass in both the workspace and fork checkout. No GPU
training has run; the real VAE CTC path and staged-rebuild rollback are tested.

## Approved first T4 mechanics run (2026-10-05)

The previously prepared run was changed from L4 to T4 and instrumented with
fixed train/val checks at 0/50/100/150/200, preserving latent-noise seed and
training RNG. A zero-update logging failure was corrected before a successful
200-update run. The complete run is persisted as `20261005-072316`.

Finite/decreasing losses and nonzero OCR/style gradients passed the numerical
wiring gate. Visual reconstruction did not pass: flattened/noisy traces, empty
greedy OCR (CER 1.0) and wrong fixed writer predictions at the final step.
These results warrant a smaller memorization test, not full-scale training
or a claim of reproducing the authors' English experiment. No further GPU job
was started after this result. See `ENGLISH_IAM.md` for metrics and paths.

## Single-line geometry isolation (2026-10-05)

A 1,000-update T4 coordinate-only autopsy of `c08-434z-05` completed, with all
other objectives off and verified absent auxiliary/pen loss gradients. It
improved fixed Y correlation to 0.898, but true-pen render inspection still
shows jagged/distorted shapes. The geometry gate remains partial, not perfectly
memorized, and no pen/CTC/style or larger-data stage was launched. CPU-only
latent-mean and component-selection controls do not eliminate residual error.
See `ENGLISH_IAM.md` for metrics, startup packaging correction and artifacts.

The pen-loss audit also found that inverse class weights are computed before
padding is masked. For the fixed 581-point line padded to 584, this reduces the
real-point-only EOC factor 581 to an effective fixed-batch 146 while retaining
approximately 16x pen-up weighting. Training-batch composition can change the
EOC factor. A controlled pen-policy A/B is still needed to establish causality.

## Pinned low-LR model/optimizer continuation (2026-10-05)

`20261005-081356` resumes the exact step-1,000 model and Adam state at 1e-5 LR,
with no other setting/objective changes. It reaches global step2,000 after
1,000 additional T4 updates. Fixed X/Y RMSE decreases to0.059/0.031 and Y
correlation reaches0.976. True-pen handwriting is more recognizable, with
residual roughness; the visual gate remains subject to review. Clipping persists
on every update, and sigma continues decreasing despite real mean-location
improvement. No later-stage GPU run or MSE objective was launched.

The original checkpoint omitted RNG state; provenance records the seed42
restart rather than claiming bitwise continuation. New checkpoints save RNG
states and final Adam steps are verified at2,000. `ENGLISH_IAM.md` records the
metrics, strict resume/config checks and artifact paths.

## Geometry gate accepted and frozen pen-policy A/B

The user accepted the step-2,000 geometry autopsy gate. No additional geometry
optimization/MSE was performed. Run `20261005-084901` is CPU-only, two identical
fresh 771-scalar heads, 1,000 updates each, cached eval features/latent seed 1042,
LR1e-3 AdamW (0.9/0.99), no decay, gamma 2 focal and clip 10. Source checkpoint
SHA256 `23fc747d82773d6714385f1e25af650ab018c14d95f11c2db54afbe0c8a8ac33`
remains unchanged. Padding is excluded from weighting/loss/metrics. This runner
is separate from upstream `get_loss`: it does not silently change the upstream
Chinese pen policy or existing smoke runner.

Arm A real inverse weights 1.068/16.139/581: pen-up precision/recall/F1
0.590/1.000/0.742 ; 25 false breaks, 0 missed, counts 519/61/1.
Arm B sqrt inverse normalized/capped 8, weights 1/3.887/8:
0.791/0.944/0.861; 9 false breaks, 2 missed, counts 537/43/1.
Both eliminate false internal EOC and learn correct final EOC. Target 544/36/1.
Every snapshot merges into the full model and checks all non-pen state,
GMM outputs and fixed XY **bitwise unchanged**; reloading saved branch models
independently confirms this. CPU and prior CUDA output equality is NOT claimed.
B is the boundary-F1 winner at this budget, not perfect boundary reconstruction
or proof A cannot converge. Generalization, earlier joint-run causation and
paper-English reproduction remain unproven. No joint training or CTC followed.
See ENGLISH_IAM.md for commands, experiment controls and persistent paths.

## Joint compatibility failed; deterministic one-line capacity succeeded

Both branches use the pinned step-2,000 geometry source. Joint inserts ONLY
trained B pen rows (user selected), restores original geometry Adam state,
LR1e-5, GMM 1 + bounded-focal 1, stochastic train mode. Its strict XY RMSE and
correlation guard stops at 50: Y RMSE 0.03120→0.03252, Y corr 0.97583→0.97416,
pen F1 (T4 same-device) 0.87179→0.86842. It is not promoted.

Independent deterministic MSE resets ORIGINAL geometry (not joint/B head),
uses latent mean/eval dropout off, fresh AdamW LR1e-4/decay 0, 1,000 updates.
Only expected-XY MSE participates; true pens for the geometry assessment,
CTC/style/KL/derivative objectives off. X/Y RMSE 0.05497/0.02740→0.00228/0.00170;
Y corr 0.99993. Within-stroke first/second-difference vector errors
0.04482/0.07689→0.00297/0.00551. The rendered observed line is very close to
processed input: architecture/8× bottleneck/encoding can represent this line.
This is NOT a pure loss-only ablation or English reproduction/generalization.
Fixed-noise expected readout remains worse (0.01768/0.00990), and even latent
mean max-pi readout remains rough (0.02565/0.03549). Production sampling/readout
and objective integration remain unresolved.

First worker completed joint then caught stale conv_logvar.grad on switching
branches BEFORE first MSE optimizer step. Initialization now clears every .grad;
transition regression added. Only MSE relaunched, no repeated joint spend.
All 51 root/fork tests pass. Independent saved-state CPU checks confirm MSE
pen/sigma/rho output rows + conv_logvar + OCR/style unchanged; original source
SHA unchanged; optimizer steps 2050 joint and 1000 MSE. No CTC followed.

Joint `reconstruction/20261005-091556/joint`; MSE/report
`reconstruction/20261005-091827`. Full settings, invariants, nonmonotonic
convergence/readout caveats, failed-attempt history and commands in ENGLISH_IAM.

## Contract patches and eight-line integration (2026-10-05)

Frozen deterministic-geometry pen refit: B and known-length binary tie at F1
0.9474, FP4/FN0. Keep B; binary forced EOC does not prove learned termination.
Exposed dropout, deterministic/greedy/stochastic readout, shared softplus sigma,
valid-element KL, model-owned XY scaling/checkpoint metadata, decoder mask,
independent rotation control and real-trainer accumulated masked losses are now
implemented. GroupNorm and fundamental architecture are unchanged.

User-authorized T4 test uses actual VAE trainer, 8 writer-10174 training lines
(514–581 points), batch1/accum8, sampled latent, dropout0, LR1e-5, GMM+boundedpen
+gradient-calibrated expectedXY. CTC/style/KL off. Completes 200 updates/1600
microbatches; 20 sampled-z evaluations per line at five checkpoints. Mean mu X/Y
RMSE .944/.115→.388/.059, macro F1 .287→.698. **Gate NOT passed:** original-line
geometry regresses, unfamiliar lines remain jagged, 51 false internal EOC,
correct final EOC only3/8. All200 updates clipped at5. No extra run/KL/CTC follows.

Anchor6.6665 is median per-line gradient calibration, not uniform15%: initial
ratios .0038–.3735. Historical training rows log last microbatch, not effective
batch mean; corrected future logging has regression coverage, with no rerun.
Nonzero logged KL diagnostic has zero objective weight. Source/frozen auxiliaries
unchanged; finite final state and Adam step200 independently CPU checked.
61 tests pass in root/fork. Full DDP/DiT are not GPU validated.

Engineering and paper-informed control configs are separate; latter is not exact
English reproduction because normalization/EOC/data-contract remain unresolved.
Details, commands, refit provenance and limitations are in ENGLISH_IAM.md.
Volume report: `checkpoints/iam_eightline/20261005-102304/index.html`.

## Authorized continuation beyond the old 200-update cap

User explicitly authorized exploratory T4 spend. Controlled pair: same old
step200 model, sampled latents, MSE100/pen1, GMM0 vs1 only, fresh identical Adam
LR5e-5→1e-5, 1,000 updates each; then pinned model+Adam+RNG continuations2,000
at1e-5→1e-6. Final direct X/Y .01191/.01822, penF1=1/allEOS8/no falseEOS;
GMM X/Y .03150/.02060, F1 .93783/allEOS8/falseEOS3. Gradient/variance diagnostics
preserved; GMM sigmas tighter, active rho not singular. Training samples only.

Mean-latent geometry-only L-BFGS from direct3k: 50 outer/593 closures, no clipping,
no GMM/pen/auxiliary objective, improves X/Y .00525/.01003. This is a multi-knob
non-paper capacity diagnostic. Frozen pen rows lose accuracy as features move;
CPU refit only those rows restores F1=1/allEOS8/no falseEOS on all8 means AND
all20 sampled-z pen predictions per line. Sampled median X/Y mean .00602/.01011.
All non-pen state/GMM rows unchanged, no forced ending. No exact pixel identity
or generalization claim. Legacy inactive config fields explicitly annotated.

Then CTC blank-bias−5/0 A/B trains ONLY the OCR head on cached frozen means,
through VAE.get_ocr_loss, after finite CPU backward. Both reach CER0/8 exact
training transcripts (−5 by500, zero by250); 5 repeat-bearing/3 no-repeat lines.
Full geometry/pen/style invariant checked. Not joint VAE/CTC or English OCR
recognition/generalization; trainable bias makes initial−5 nonfatal here.
65 root/fork tests pass. No KL/style/InkDiT/full-IAM training promoted.

Details/commands/limitations/provenance in ENGLISH_IAM. Volume reports:
`iam_objective_study/comparison-3000`, `iam_lbfgs_geometry/20261005-113159`,
`iam_ctc_head_ab/20261005-114305` under `checkpoints/`. Two initial remote-bootstrap
import failures (zero updates) were explicitly stopped and fixed; history retained.

## Curve fidelity follow-up (2026-10-05)

The raw/RDP/marker-free audit exposed genuine local curve errors despite low
aggregate RMSE. This was chiefly under-converged geometry: continued direct
point-MSE improves substantially without any architecture change. A matched
target first-difference anchor adds a consistent benefit on all eight lines;
unit-tangent matching was tested as a short-edge alternative but was not the
best overall choice. No generic smoothing, RDP change, KL/OCR/style objective,
normalization replacement or upsampler change was applied.

Final mean X/Y RMSE0.000538/0.001437, tangent p90 3.22°, turn p90 5.17° (mean of
per-line percentiles), with genuine pen F1=1 on all8 means/all160 sampled-z outputs.
Target corner turn p90 improves too. Geometry is good enough for bounded joint
integration on these memorized lines, not full-IAM/generalization readiness.
Old OCR-head CER0 must be revalidated against the changed encoder, not assumed.

Global mod8 residual bias was below the permutation null; this is not universal
proof against all decoder artifacts. High-mixture-entropy points had no larger
errors, and a frozen-feature affine alternative was much worse. Sampled latents
retain a precision gap; production Adam/GMM and regularized latent behavior
remain unvalidated. Configurations, canceled/preempted attempts, source hashes,
CPU reload checks and marker-free galleries: [CURVE_FIDELITY.md](CURVE_FIDELITY.md).

Volume overview: `checkpoints/iam_curve_study/research-summary/index.html`.
Selected: `checkpoints/iam_curve_study/20261005-124900/delta50/checkpoint.pt`.

## Bounded posterior/KL/OCR follow-up (2026-10-05)

The immutable visually faithful reference survives three400-update stages:
sampled+mean-anchored XY/targetΔ/pen, then KL1e-6, then joint OCR. Final mean
X/Y .000270/.001162 and turn p90 3.63° versus reference .000538/.001437 /5.17°.
Pen F1=1 and CER0 on all8 means/all160 draws. Old OCR weights were revalidated
on the changed encoder (zero warmup needed), not assumed valid. Full all-line
marker-free mean/median/worst galleries were reviewed.

First equal-anchor LR1e-6 attempt stopped at200 on position regression; retry
uses mean anchor1000 versus sampled100 and LR5e-7. This changes two knobs and is
not a one-factor ablation. Tiny KL/CTC have very weak gradients on memorized
lines: compatibility only, not prior matching or strong regularization proof.
Posterior std narrows, but CPU paired-noise reconstruction with original std
also improves. Clipping remains frequent. See [LATENT_INTEGRATION.md](LATENT_INTEGRATION.md)
for all controls, source hashes, attempts and independently reloaded checkpoints.

Volume report `checkpoints/iam_latent_integration/latest/index.html`; final
`checkpoints/iam_latent_integration/20261005-133620/ocr/checkpoint-best.pt`.
SHA256 `fffe1405db10f8f6c2ce6ec1e030706b7947c93d83fa4eaeffec3b6a8c5b08f9`.
89 tests pass; standard trainer fails closed on mean-anchor configs it cannot
implement. Next gate is more/held-out lines, then multi-writer/style, not full
IAM/InkDiT or paper reproduction. No source architecture/preprocessing change.

## Same-writer expansion follow-up (2026-10-05/06)

24 train lines (eight old + sixteen new), four held-out lines, writer10174. Forms
are shared across splits; this is not form/writer-independent evaluation. All
validation lines remain outside optimizer/calibration/head caches/selection.
The original faithful eight-line checkpoint is immutable.

The original source reconstructs new/held-out lines poorly. A1200-update sampled/
mean Adam expansion improves held-out XY error but degrades original local
curves; full-set deterministic L-BFGS further improves positions while weakening
pen-head compatibility. Neither meets the visual fidelity gate. Direct linear probes
on frozen decoder features are worse. Target Δ gradient is already substantial
(61.5% point-gradient norm at Adam endpoint), so blindly increasing smoothing is
not supported. A calibrated joint geometry/pen follow-up was dashboard-stopped
after saved200, then resumed with model+optimizer+RNG and sample order intact.

Details: [WRITER_EXPANSION.md](WRITER_EXPANSION.md). Keep this investigation
separate from the passed eight-line capacity/posterior/OCR gates. Do not advance
to full IAM or InkDiT on the basis of these aggregate position improvements.

Completed balanced result: train mean X/Y .009704/.013157, pen F1 .9883; held-out
.149929/.060841, F1 .7372. Original-reference curve retention and held-out visual
fidelity still fail.103 root/fork tests pass; all jobs completed/stopped. Latest
report includes immutable-reference comparisons with source-array hashes.

## Conditioning/optimizer follow-up (2026-10-06)

CPU perturbations demonstrate global padding sensitivity from temporal GroupNorm;
channel-only norm strongly reduces it but changing a pretrained layer is not a
free geometry fix. The centered sampled arm fails before its first update with
an extreme posterior distribution shift. Higher-LR Adam only marginally improves
control curves. Source and failed artifacts remain immutable; no broken retries.

Joint full24 geometry+pen L-BFGS substantially improves the actual c/h renders.
A shared-source target-relative vector loss has a modest angular benefit, with
slightly worse point error at shared200. It matches genuine target derivatives,
not zero curvature/smoothness. A fresh-optimizer weight continuation and a
channel-source continuation are separate followups, not conflated with A/B.
Held-out reconstruction remains the limiting generalization gate, not solved by
training memorization. No incidental KL/CTC/style. Normalization changes remain
research-only and dedicated reload/inverse transforms are mandatory.

Details: [CONDITIONING_STUDY.md](CONDITIONING_STUDY.md). Volume overview
`checkpoints/iam_fullset_joint/research-summary/index.html`; all mean/20-draw
metrics and marker-free galleries are preserved, with source/protected-parameter
checks.127 root/fork tests pass. New report publication uses atomic dated links.

The completed geometry polish reaches train X/Y .000760/.001425 and turn p90
6.00°, with pen F1=1 for all24 means/all480 sampled draws; unseen4 still fail.
The next bounded geometry-only pilot uses the existing192/32,8-writer data and
keeps every prior validation line excluded. No OCR/KL/style enablement. See
CONDITIONING_STUDY.md for the explicit source, objective, calibration and limits.

### Validated GPU dispatch/evaluation acceleration

CUDA replay now provides3.0–3.6× training throughput in paired no-update T4 checks,
with GPU busy time rising from~37% to~95–97% and gradient-relative error<1.4e-7.
Spawned CPU metric workers preserve exact results; optional same-line posterior
batching further speeds evaluation without different-length padding. Guards,
RNG checks and136 tests cover reusable changes. See CONDITIONING_STUDY.md for
exact timings/provenance and the completed192-line full-gradient follow-up. That
trained follow-up is still visually rough and not promoted. An explicitly
UNTRAINED, opt-in polyphase initialization control reconstructs all224 observed
lines nearly exactly; it is a capacity/conditioning diagnostic, not a learned
semantic/generative model or automatic OCR/KL readiness.

## Initialized transport optimizer audit, 2026-10-06

139 tests pass. Controlled initializer arms expose extreme high-gain update
sensitivity; uniform/scaled-readout arms stop on a TRAIN-only failure gate.
Freezing the readout and using body LR1e-7/posterior LR1e-3 preserves near-lossless
224-line geometry after200 real updates, with improved sampled reconstruction and
perfect pen boundaries. CPU reload/source/frozen-head checks pass. Explicitly
reinitialized engineering codec, NOT original learned-VAE/paper reproduction or
semantic/generative readiness. Original input RDP corners are retained, no smoothing.
Full configs, limitations, SHA and report links: [CONDITIONING_STUDY.md](CONDITIONING_STUDY.md).

## Matched initialized-codec KL continuations (2026-10-07)

Same protected200 source/Adam moments/RNG/training schedule; three200-update T4
continuations differ only in KL0/1e-6/1e-5. Optional captured KL matches corrected
core per-valid-element KL values and gradients; default geometry path unchanged.
All224 mean curves/pens remain faithful. Full-affine
KL1e-6 final200 has3 false internal EOCs despite perfect pen-up F1 (selected100
is clean), and costs~31% more sampled XY error than matched KL0. Most prior
loss reduction comes from pen uncertainty, not coordinate latent mean whitening.
KL1e-5 adds more posterior jitter and6 failing draws (including5 false internal
EOCs), despite perfect MEAN pens and lower KL. Not promoted. No OCR/style/GMM,
no claim of paper reproduction, semantic latent quality or tested generation.
A diffusion model does not inherently require an N(0,I) latent distribution.
See CONDITIONING_STUDY.md and `checkpoints/iam_codec_kl_study/latest/index.html`.
Root/fork147 tests cover optional KL, padding gradients, restored continuation,
explicit EOC eligibility and momentum-safe posterior row freezing.

The matched `20261007-012331` KL1e-6 follow-up freezes only24 pen-logvar feature
weight rows (and zeroes their restored Adam moments), retaining bias learning.
All4480 draws now have zero false internal EOCs and perfect pen boundaries, with
essentially unchanged sampling geometry. Full-affine pen std had long-line tails
up to .38–.43, hidden by tiny medians; prior loss improved largely by increasing
pen uncertainty. Mandatory train-only EOC eligibility now prevents such endpoints
from winning on pen-up F1 alone. This is a new posterior stop-error mechanism,
not a retroactive explanation of the old latent-MEAN jaggedness.

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
