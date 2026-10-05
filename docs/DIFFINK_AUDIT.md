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
