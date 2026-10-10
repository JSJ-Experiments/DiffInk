# Reconciled research handoff — 2026-10-10

The native-run handoff omitted important later experiments recorded by another
agent in workspace `TEXT_CONDITIONING.md` and `ARCHITECTURE_AUDIT.md`. The native
concat/joint comparison remains an immutable bounded **AMD backend/control
comparison**, not the newest conceptual intervention or evidence that content
supervision has never been tried. Do not use its failure to erase the later T4
content result. Source records below were read; not every historical entry in the
383KB CONDITIONING_STUDY ledger was reread. The selected relevant sections cover
weak/local alignment, positional timing, progress clocks, AR closure and corpus
controls. Some later cloud-only artifacts are not yet fully migrated; historical
numbers here are documentation-reported until independently reloaded.

## Established findings that must carry forward

| Intervention / source record | Outcome / implication |
| --- | --- |
| Corrected joint compact40, pure text, no CTC | Already tested on T4; selected87.17%, final90.03% DEV CER. Attention access alone did not solve composition. Native fresh controls do not make this a new hypothesis. |
| Same corrected joint + frozen-reader CTC | DEV8.64%, arbitrary-text8.98% primary-reader CER. Second-reader DEV37.93%, arbitrary36.37%. Reusable content signal exists, but generated paths are dense zig-zag codes, not usable handwriting. |
| Post-CTC geometry repair | CTC-off repairs segment lengths while rapidly losing text; CTC-on keeps content and pathological geometry. Neither passes the combined gate. |
| Explicit CTC-gradient caps0.3/1/3x | No joint content/geometry pass. Do NOT propose another scalar/norm-ratio sweep as untested. |
| Separate draw/jump motion56 | Local geometry improves substantially; text remains poor. Additional1/3/10x caps and unchanged12k continuation already failed the combined gate. |
| Real-point masking correction | Padding contaminated TRAIN motion moments, noise and loss. Fix is real, but matched terminal valid-points80.21% versus historical78.78% showed no quality win. Preserve correctness without calling it a plateau break. |
| Local Gaussian hints / weak reader alignment / fixed progress clocks / small AR pilots | Already investigated in CONDITIONING_STUDY.md. Strong familiar fit or nicer routing did not establish unseen composition. A new alignment proposal must be meaningfully different. |

Recognition gains are not human-legibility evidence. The second reader is a
same-family/corpus check, not independent human validation. Severe under-lifting
remains a quality failure; phase masking and all-head RoPE fixes must survive any
new implementation. Existing DEV/confirmation sets are exposed; do not call them
fresh or repeatedly select on a reserved gate.

## Actual next priority, not another attention-only experiment

Follow ARCHITECTURE_AUDIT.md: jointly coherent **pen topology + geometry + content**,
and a reusable learned stroke/segment representation or structured decoder.
The initialized polyphase codec proves faithful transport/reconstruction, not a
semantic handwriting manifold or an ink prior. KL alone is not evidence of either.

Candidate source already exists in the workstation:
`iam_tools/topology_motion56.py` with `tests/test_topology_motion56.py`.
It provides categorical pen supervision and an optional biased straight-through
hard-forward topology/content derivative. It is NOT a trained result or discrete
diffusion sampler. Do not overwrite this pre-existing work or implement it twice.
Its two arms retain identical hard forward values and motion gradients, but the
surrogate itself needs deliberate testing. Equal-class inverse-frequency weighting
and reader-directed pen gradients are hypotheses, not automatically safe English
policies. Check per-class gradients/counts, source fidelity, actual hard pen/EOC
changes and whether reader improvement survives marker-free visual review.

Before another corpus generation launch:
1. Reconcile and verify the latest motion/content artifacts (pull missing files
   from Modal if needed; no Modal training; no source-code sync over WSL).
2. Establish the proposed structured/learned representation's source reconstruction,
   local draw geometry, stroke topology and perturbation/prior behavior. Use valid
   real-point phases and split-safe TRAIN calibration; no generic curve smoothing.
3. Choose one falsifiable intervention, version its representation/model/objective/
   sampler contract, and compare against a matched control on AMD. No accidental
   attention, CTC-weight, padding, optimizer and representation bundle.
4. Evaluate source-faithful local geometry AND target-free text sensitivity/readability,
   lifts/strokes, duration/EOC and independent visual evidence; neither CER nor
   geometry alone is a pass. Keep confirmations closed until these gates pass.

## Current run handling

`iam_wsl_conditioning/20261010-112127`: concat complete, joint still bounded and
running at reconciliation. Preserve its source/config/provenance; do not alter the
runtime or submit a duplicate. Let it terminate and collect the combined verified
report. A quiet SSH process-exit monitor and CPU-only report waiter are active.
Further plain-attention/no-CTC repeats are NOT the research recommendation.

## No-training check of the existing topology prototype

Read-only CPU probe on genuine TRAIN `l02-166z-07` (270points), using the frozen
reader and existing prototype: hard-forward values and motion-field gradients
match bitwise across detached/joint arms. Detached pen gradient is zero; joint
surrogate pen-gradient norm.00228595. Source CTC.000888729; reader weights and
parameter grad buffers unchanged. This confirms the proposed gradient path is
actually available to the real reader, NOT that it produces better generation.
Identity motion moments were used only for this gradient-contract probe, not as
production normalization or a fitted TRAIN statistic. One source line only.

Evidence/scripts on WSL: `logs/topology-real-reader-source-contract.{py,json}`.
The first diagnostic caller failed before computation because the nested model
module was absent from PYTHONPATH; preserved separately, then fixed explicitly in
the standalone diagnostic script. No training retry or model change occurred.
