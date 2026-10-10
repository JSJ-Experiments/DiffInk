# Source-only topology counterfactual audit — 2026-10-10

## Why this check is needed

The architectural audit identifies a disconnected pen/content gradient in
motion56. The pre-existing `topology_motion56.py` opens a biased straight-through
gradient, while retaining hard forward geometry/pens. Its first real-reader
contract proved a gradient exists, **not that it suggests correct hard changes**.
This audit tests finite changes before another expensive corpus experiment.
It does not revisit attention-only controls, CTC scalar sweeps or eight-line
pen-head fitting. No training recipe/model weights were changed.

## Exact experiment

Workstation CPU, two torch threads,14.91 actual diagnostic seconds.
Sixteen fixed **TRAIN** IDs from native control config; original point bytes
checked against the pinned corpus manifest. No DEV or confirmation selection.
Local model space uses the existing0.01 input scale.

Per line choose two continue and two pen-up indices, deterministically spanning
each class's available index range (not uniform time/arc length). Anchor and real
final EOC are excluded. Each source contributes four indices, giving64 per arm:

1. **Binary corruption:** flip a correct continue/pen-up label to the other class.
2. **Binary repair:** undo that exact single corrupted label.
3. **Early EOC:** replace that internal label with EOC.

All motion fields stay fixed, including the inactive draw/jump stream. The
historical previous-pen hard gate therefore still switches which motion is used.
Both exact CTC forwards retain actual TRAIN point length, deliberately preventing
predicted EOC from hiding the suffix in supervision. Generation would instead
stop at its first EOC. Frozen reader weights and gradient buffers remain unchanged.

Measure actual CTC delta and the proposed biased surrogate directional derivative
toward the new one-hot logit vector. This is a finite one-unit score replacement,
**not an infinitesimal derivative of argmax**. Identity motion statistics here
only express the diagnostic in raw coordinates; they are NOT production TRAIN
whitening or calibration. Valid source XY round-trip tolerates float32 accumulation.

## Findings

| Probe (64 each) | Exact CTC favors flip | Surrogate favors flip | Sign agreement | Median suffix displacement |
| --- | ---: | ---: | ---: | ---: |
| Corrupt a correct binary boundary |6 |34 |53.1% |0.1643 |
| Repair a corrupted binary boundary |58 |55 |82.8% |0.1643 |
| Insert internal EOC |18 |33 |60.9% |0.00428 |

Source median CTC.000961; binary-corrupted median.012602. Thus the actual reader
usually penalizes these boundary corruptions and the surrogate often helps repair
them. It is **not uniformly reliable near already-correct topology**. A single
hard boundary flip also moves the entire suffix; this is geometry/topology
coupling, not merely an extra drawn connector. Displacement units are model XY,
with line-height approximately1, not physical velocity.

Interior EOC probes retain median48.8% of the source under actual first-EOC free
stopping, yet fixed-length CTC improves in18/64. The surrogate favors33/64.
This flags the already-described TRAIN-prefix/free-stop mismatch and shows why
a nonzero pen derivative alone cannot certify a safe stopping policy.
Some deltas are small near the already-low source CTC floor; these results do not
quantify the contribution to the corpus CER plateau.

**Do not infer** that supervised categorical pen training necessarily fails, that
the surrogate can never help, or that this source-only probe solves generation.
No net training objective, denoising trajectory, free text or independent
recognition/visual legibility gate was tested here.

## Artifacts and implementation

`data/checkpoints/iam_topology_source_audit/20261010-122515/` on Timothy WSL:

- `result.json`: all192 probes, source IDs/counts/hashes and scope.
- `counterfactuals.h5`: original and before/after hard trajectories.
- `index.html`: all16 sources and64 corruption/EOC comparisons; fixed-scale,
  marker-free, no smoothing, including actual first-EOC truncated versions.
- `as-run-source.py`, `source/`, `artifact-manifest.json`: provenance.

Windows report:
`E:\autowrite-data\checkpoints\iam_topology_source_audit\20261010-122515\index.html`.

Reusable new helper: `iam_tools/topology_counterfactual.py`; nine new regression
tests cover real finite flips, stop-vs-TRAIN prefix semantics, source immutability,
padding invariance and invalid/no-op requests. The original untrained topology
prototype and its five workstation tests are preserved unchanged. The new helper
takes explicit implementation callbacks, avoiding historical-runner dependencies
or duplicated topology code in the published fork. This is a
CPU input-gradient diagnostic, not a new diffusion sampler or a quality result.

## Next non-redundant conceptual intervention

Prioritize a **learned source-faithful stroke/segment codec with structured ink
actions**, not another renamed polyphase packing. Proposed falsifiable order:

1. Extract actual strokes and split long strokes into bounded contiguous pieces,
   retaining original nonuniform IAM/RDP points, singletons, true hooks/corners
   and exact boundary flags. Piece boundaries must NOT introduce pen lifts.
   Stroke ≠ character: many cursive strokes span several characters.
2. Train a genuine bottleneck/codebook on local segment geometry, with explicit
   reversible anchors/layout and lengths. No text/CTC reward in this source
   reconstruction/prior gate. Do not silently resample/smooth the source or claim
   every decoder latent is ink-like because TRAIN reconstruction is excellent.
3. Verify source fidelity, held-out-stroke/writer transfer, within-stroke direction
   fidelity, topology, sampled/perturbed-code behavior and real ink renders.
   TRAIN-only calibration; existing exposed DEV is development evidence.
4. Only then compare a typed structured generator (draw-piece / move / stop)
   against a matched control. Text gradients should operate through real decoded
   ink; STOP gets explicit source supervision and a deliberate free-stop contract,
   not assumed-valid reader gradients at fixed oracle length.
5. Require target-free composition, source-like geometry, stroke counts/density
   (severe under-lifting remains a failure), independent visual evidence and reader
   transfer together. Leave confirmations closed until combined gates pass.

This is a proposed experiment, not an implemented/trained success. Codebook size,
piece duration, reconstruction-vs-prior tradeoffs and generation alignment must
be chosen using source evidence, not optimism. A learned stroke codec can fail,
including by erasing microstructure, losing long joins or memorizing its sources.

Historical metadata/HDF5 for the latest valid-motion pair has migrated, but final
training checkpoints and later absolute40 content artifacts are still incomplete.
Verify their immutable hashes before any warm start; never sync old code over WSL.


## Validation and reproduction note

Workspace808 tests and31 subtests; fork772 tests and31 subtests PASS.
Logs: `logs/topology-source-{full,fork}-tests-final.log`. Initial fork collection
failed because copying the prototype would require unpublished historical motion
runners. Rather than copy a large ancestral module chain, the diagnostic was
refactored to accept explicit content/forward callbacks; the original prototype
remains unchanged on WSL. Test doubles verify the generic querying contract.

The bounded192-probe source audit was repeated with the final callback interface.
All source records, probe values and saved before/after points exactly match the
first audit. New canonical repeat report:
`data/checkpoints/iam_topology_source_audit/20261010-123143/index.html`.
`callback-refactor-parity.json` records the comparison; both immutable artifact
directories remain available. No new training or confirmation evaluation occurred.

Near-floor effects are not all rounding noise:17 of64 early-EOC probes improve
CTC by >1e-6; six by >1e-4. For the32 pen-up→EOC replacements, XY is literally
unchanged;12 improve by >1e-6 (three by >1e-4), yet free stopping truncates the
suffix. This specifically separates the reader's pen-channel preference from
geometry changes and illustrates the stopping-contract mismatch.

Available historical terminal evaluation HDF5/source/statistical hashes for
both valid-motion arms were checked on WSL. Terminal DEV g1 CER78.7832/80.2147%
and equal initializer/order records agree with the ledger. Neither final training
checkpoint is present, and the later absolute40 content directory is absent.
Do not claim checkpoint replay or complete migration. Inventory:
`logs/reconciled-history-artifact-integrity.json`.
