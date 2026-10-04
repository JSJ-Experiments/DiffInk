# English IAM compatibility audit

Upstream: https://github.com/awei669/DiffInk

Inspected commit: `97bc6a3c39a5bdaa9728daaab6d3707480006343`.
Branch: `english-iam`. No weights downloaded; no training or GPU jobs run.

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
