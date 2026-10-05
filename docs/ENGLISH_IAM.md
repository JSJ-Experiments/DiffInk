# English IAM-OnDB → DiffInk

**No training or GPU job has run.** This workspace now implements a raw parser,
experimental preprocessing, a tiny HDF5 exporter, and CPU loader validation.

Fork: https://github.com/JSJ-Experiments/DiffInk/tree/english-iam

This branch is based on upstream commit
`97bc6a3c39a5bdaa9728daaab6d3707480006343`. Our English compatibility
patch is saved in `patches/diffink-english.patch`.

## Read this caveat first

The released repo does **not** include its raw-data converter or coordinate
normalization formula. Our aspect-preserving line-height=100 normalization is
an explicitly experimental baseline, not verified paper-equivalent. RDP uses
0.5 in those units. IAM line XML has no usable character indices in the samples
we inspected, so the dataset stores empty `char_points_idx`, explicit stroke
ends, and requires our stroke-prefix fallback for DiT. The third pen-state class
is used only at the final line point; exact English end-state equivalence is
also unverified. See [the audit](DIFFINK_AUDIT.md).

## What works

- IAM CSR transcript parsing, exact 1-based line pairing, form→writer lookup.
- Canonical raw JSON keeps x,y,time and original stroke/point ordering.
- Absolute N×5 coordinates, aspect-preserving normalization, y flipped up,
  per-stroke RDP, retained acquisition times, no pen-up bridges.
- 250 training + 50 validation lines. Five validation writers and 25 untouched
  future-test writers are excluded from training; all splits are writer-disjoint.
- Vocabulary comes only from safely paired training transcripts, excluding both
  validation and future-test writers, never the Chinese `All_zi.json`.
- Released **TrainDataset is unchanged** and successfully loads/collates a batch.
- English validation token, stroke-prefix masking, inference preview selection,
  and exact CTC feasibility have small explicit upstream patches.

## View the outputs

- `data/canonical/iam/preview/index.html`: the original 10 raw line previews.
- `data/diffink/iam/previews/index.html`: raw **above** / encoded N×5 **below**.
- `data/diffink/iam/tiny_train.h5`, `tiny_val.h5`, `chars.json`, `writers.json`.
- `data/diffink/iam/manifest.json`: split IDs, preprocessing policy, rejected data.
- `data/diffink/iam/batch_check.json`: CPU loader results and padding/mask checks.
- `data/canonical/iam/tiny/`: unmodified raw canonical samples.

Open a gallery file in a browser, or use the PNG files directly. Display scaling
never changes canonical coordinates. Some source CSR labels contain typos or
omit punctuation visible in strokes; we preserve annotations, not silently
replace them with the differently wrapped OCR prompt.

## Local commands (existing venv)

```sh
source venv/bin/activate
pip install -r requirements-iam.txt
pip install -r requirements-cpu.txt  # CPU-only PyTorch; loader checks only

python -m iam_tools.preview --limit 6
python -m iam_tools.build
python -m iam_tools.check_batch --repo .
python publish_iam.py  # persist validated data on Modal, never GitHub
python -m unittest discover -s tests -v
```

The full raw archives have been downloaded and extracted locally. All persistent
Volume originals are retained. The build is deterministic with seed 42. Samples
outside the loader length limits, with validation OOV, ambiguous transcripts,
or infeasible CTC targets are explicitly reported rather than padded/truncated
to force acceptance. A rerun overwrites generated HDF5/metadata, not raw IAM.

## Modal: no SSH needed

The `diffink-data` Volume persists independently of a shell. The same root is
mounted as `/mnt/diffink-data` by `modal shell`, and `/data` by our functions.
Neither mount automatically appears on this machine. Use the CLI locally:

```sh
modal volume ls diffink-data diffink/iam
modal shell --volume diffink-data
# Inside a NEW shell:
ls /mnt/diffink-data/diffink/iam
```

Repeatable CPU-only remote jobs (no torch/model/GPU imports):

```sh
modal run modal_iam.py --limit 10       # copy/extract raw + raw parser gate
modal run modal_prepare.py             # experimental tiny conversion
```

Small-file access through the Volume can be slow. The first conversion was
completed locally on CPU after downloading the archives; generated outputs are
uploaded back to the Volume. No paid GPU is needed for this stage.

To download persistent outputs:

```sh
mkdir -p data/diffink
modal volume get diffink-data diffink/iam data/diffink --force
```

Create the destination directory first. Modal includes the remote basename
under it on directory downloads. Files are committed/uploaded to:

```text
Volume root
├── iam/                          # your original files, untouched
├── raw/iam/                      # persistent archives + extracted raw data
├── canonical/iam/preview/        # original raw visual gate
├── canonical/iam/tiny/           # raw JSON for tiny conversion
├── diffink/iam/                  # HDF5 + metadata + comparisons + check report
└── checkpoints/                 # experimental InkVAE mechanics runs
```

Modal docs: https://modal.com/docs/guide/volumes

## Next gate

Review before/after shape preservation, and confirm normalization/end-state
choices against the authors' prepared English data or forthcoming multilingual
code. Only then consider a tiny InkVAE overfit job. We have **not** proved model
reconstruction, English generation, Markdown layout, or LaTeX handling.

## Follow-up fixes and separate overfit preparation

The earlier CTC test missed the duplicate `VAE.get_ocr_loss` used in the actual
forward/training path. That is fixed by delegating to the shared helper, with
regression tests through `VAE.forward`. The previous claim of a complete CTC
training fix was premature; see the audit above for the correction.

Generated-output rebuilds now replace completed directories rather than
accumulating stale JSONs/previews. Failed conversion preserves the prior build;
raw IAM is never cleaned. Publishing also removes stale generated remote files.

A separate same-writer, **line-disjoint** mechanics dataset has eight writers
×24 train lines (192) and ×4 val lines (32). This is not the original 250/50
writer-disjoint dataset and is not a held-out-writer benchmark. Test writers
remain reserved. Prepare/check it without training:

```sh
python -m iam_tools.overfit
python -m iam_tools.check_batch --out data/diffink/iam_overfit
python -m iam_tools.inkvae --data-root data/diffink/iam_overfit  # CPU forward only
python publish_iam.py --overfit
```

The CPU smoke report is `data/diffink/iam_overfit/vae_smoke.json`. All actual
model/loss terms are finite with the explicitly documented config. No optimizer
step, backward pass on the model, or GPU allocation occurs in that smoke check.

**Additional caveat:** direct height-100 input caused NaNs at initialization.
The prepared mechanics config therefore multiplies model x,y by 0.01, leaving
the stored RDP data unchanged; checkpoint and preview code preserve/undo the
adapter. The new runner also uses a log-space GMM density, because upstream's
PDF floor saturates. These are numerical mechanics choices, not paper-equivalent
English preprocessing. The exact author normalization and internal character
endings remain unresolved.

A bounded single-T4 job is implemented in `modal_inkvae.py` and
`configs/vae_iam_overfit.yaml`, and was launched with user approval on 2026-10-05. Its limits are 200
steps / 600 loop seconds, and the container times out after 900 seconds. Both
explicit flags `--train` and `--allow-experimental` are required; the default
entrypoint never invokes the remote function. The user approved a first T4 mechanics run; results are recorded separately below.

### Fixed-sample instrumentation for the first T4 run

The runner evaluates the same train/validation line at step 0, 50, 100, 150,
200 (or the budget-limited final step). `fixed_metrics.jsonl` records all loss
terms, greedy OCR/CER, writer correctness, coordinate RMSE in model units and
pen accuracy. Evaluation disables dropout and reuses a fixed latent-noise seed,
restoring the training RNG and model mode afterward. Input and reconstruction
renders use that same forward. Training metrics record nonzero/finite OCR and
style gradient norms as a supervision diagnostic.

Only one T4 container is permitted, with no automatic retries or GPU fallback.
The run remains experimental: line normalization, internal character endings,
the reversible input adapter, log-space GMM and masked reconstruction losses
are not the authors' released English training pipeline.

### First T4 result — 2026-10-05

Run `20261005-072316` completed 200 updates on a Tesla T4 with PyTorch
2.14.1+cu130; reported loop/evaluation/checkpoint time was 17.0 seconds
(not total container lifetime or billed duration). One earlier attempt stopped
at step 0 because the newly added OCR hook was bypassed by a direct `.forward`
call; that instrumentation was fixed and regression-tested before relaunch.
No L4, full-dataset training, or InkDiT job was launched.

All 200 logged losses/gradient norms are finite, and OCR/style parameter
gradients are nonzero every step. First/last 20-step mean total loss fell
5.96 → 3.91; CTC 4.16 → 3.23; style CE 2.13 → 1.86. Fixed train total
fell 10.13 → 3.64, fixed val 9.50 → 3.85.

**Reconstruction gate NOT passed.** Visual inspection of both final examples
shows mostly flattened, noisy traces rather than legible words. Their greedy
OCR outputs are empty (CER 1.0), and both fixed writer predictions are wrong
at step 200. Coordinate RMSE improved substantially, but global x progression
is not letter-shape reconstruction. Lower auxiliary losses/nonzero gradients
show that supervision is wired, not that OCR/style objectives are solved.

At batch size 2, 200 updates cover only about two passes over 192 train lines.
Next recommendation: stay on T4, use a truly minimal 1–8-line memorization
experiment with a separately approved bounded budget, and inspect letter shape,
pen-state accuracy and blank-heavy CTC behavior before expanding or using L4.
The normalization/character-ending policies and runner numerical changes
remain experimental; no author English result has been reproduced.

Persistent artifacts: `/data/checkpoints/iam_overfit/20261005-072316/`
(or `/mnt/diffink-data/checkpoints/iam_overfit/20261005-072316/` in your shell).
This contains checkpoint, shuffled/fixed metrics, prediction arrays, stepwise
input/reconstruction PNG/SVG, `result.json`, `summary.json`, `loss-curves.png`,
and `index.html`. Downloaded local copy: `data/checkpoints/iam_overfit/20261005-072316/`.
To regenerate the CPU-only report:

```sh
python -m iam_tools.report_inkvae data/checkpoints/iam_overfit/20261005-072316
```
