# IAM OCR segmentation audit (isolated research worktree)

This is a CPU-only diagnostic. It does **not** change the protected codec,
handwriting geometry, OCR checkpoints, pool, or IAM source data.

External source: the public Character Queries IAM-OnDB segmentation annotations
(Jungo et al., ICDAR 2023). They are not vendored here. Run
iam_tools.ocr_segmentation_audit with an explicit annotation directory.

## Why this matters

The frozen engineering codec is intentionally close to a lossless transport:
one latent frame carries eight chronological [x,y,pen3] points in its first
40 channels. The recent four-point OCR adapter improved unseen-writer CER, but
that experiment changed several tightly coupled things at once (frame count,
phase grouping, calibration, and positional indices).

The Character Queries ground truth gives an independent way to ask whether a
reader frame mixes multiple characters before spending another GPU run.

## Pinned-pool audit

Pool manifest:
d9546704f5debd83b79ab45d6218f76c18b39e29f7e5e907c3d7b3d6a21778f9.

The downloaded annotations contain 7,330 IAM line IDs. They overlap 4,885 of
8,192 TRAIN lines; 4,873 have an exact matching transcript. DEV has 71/128
exact annotated lines and the report split 17/32.

Mapping annotation CTC spikes through the **current** height-100 + per-stroke
RDP-0.5 preprocessing is stable: on the 71 annotated DEV lines, nearest retained
point displacement has p50=0, p90=0, p99=1 and max=2 raw-point indices.

### Character-spike collisions inside OCR frames

These counts use one annotation spike per transcript character and the processed
RDP point index. A collision means two or more annotated character spikes land
in the same reader frame. CTC is still free to choose a nearby alignment, so
this is a diagnostic, not a proof of causation.

| Split | frame | lines with collision | excess character collisions |
|---|---:|---:|---:|
| TRAIN exact annotated (4,873) | 8 pt | 4,847 (99.5%) | 21,059 |
| | 4 pt | 4,730 (97.1%) | 14,015 |
| | 2 pt | 4,159 (85.3%) | 8,289 |
| | 1 pt | 4 (0.1%) | 4 |
| DEV exact annotated (71) | 8 pt | 69 (97.2%) | 274 |
| | 4 pt | 62 (87.3%) | 171 |
| | 2 pt | 52 (73.2%) | 104 |
| | 1 pt | 0 | 0 |
| report exact annotated (17) | 8 pt | 17 | 66 |
| | 4 pt | 17 | 51 |
| | 2 pt | 17 | 31 |
| | 1 pt | 0 | 0 |

Using the denser per-point character ink ranges on the 71 DEV lines, the mean
fraction of frames containing points assigned to more than one character is
44.81% at 8 points/frame, 19.72% at 4, 6.49% at 2, and 0 at 1. The observed
8->4 reduction in mixed-frame fraction has a modest positive association with
the observed 8->4 error-count improvement. Current four-point mixed-frame
fraction also has a positive association with current line error. Treat both as
exploratory because this covered subset is only 71 DEV lines.

No backwards character-index transition was observed among retained labelled
points in those 71 lines. Delayed strokes therefore do not explain the broad
reader error on this covered DEV subset, although this does not rule them out
elsewhere.

## Implications / next controlled tests

1. Replicate the already-observed 4-vs-8 improvement before changing the codec.
2. A **2-point OCR-only** reader is now justified by more than CTC slack: it
   sharply reduces mixed-character reader frames while keeping the protected
   geometry unchanged. A 1-point reader is the cleanest alignment extreme but
   makes global Transformer attention substantially more expensive.
3. The public character ink ranges can provide a TRAIN-only auxiliary local
   glyph supervision signal for the frozen reader (or later for a carefully
   gated joint encoder update). DEV/report annotations must never enter training
   or checkpoint selection.
4. A point-level/BiLSTM or local-convolution reader using delta-XY + pen fields
   is worth comparing with the global Transformer if 2-point Transformer gains
   plateau; classic online handwriting recognition benefits from local motion
   features and recurrence.
5. Character annotations may also let us revisit experimental English
   char_points_idx / internal character semantics, but directly converting
   character boundaries into pen breaks would be incorrect for connected
   English cursive and must be studied separately.

The audit intentionally does not claim that finer framing alone caused the
four-point result, nor that segmentation supervision is required by DiffInk.
