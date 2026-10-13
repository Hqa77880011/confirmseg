# ConfirmSeg

Implementation of **ConfirmSeg: Approval-Conditioned Edit Footprints for Interactive Medical Segmentation**.

ConfirmSeg keeps foreground and background approvals separate from unreviewed predictions. A candidate decoder proposes a correction, a learned footprint controls its amplitude, and projection preserves active approvals. Corrections conflicting with an approval produce a conflict overlay; the reviewer must revoke that region before changing it.

## Installation

Use Python 3.11 and install PyTorch for your CPU or CUDA environment, then install the package:

```bash
python -m venv .venv
python -m pip install -e .
```

Activate `.venv` before installation (`source .venv/bin/activate` on Linux or `.venv\Scripts\Activate.ps1` in PowerShell). Commands below run from the repository root. CUDA is selected when available; add `--device cpu` to use the CPU.

## Data preparation

Obtain the labeled training cohort from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and extract its patient folders. All phases, slices and structures from a patient stay in one partition. The converter uses a 60/20/20 split with seed 2027 within diagnostic groups and emits one binary task per structure and nonempty slice.

```bash
confirmseg prepare --source acdc --input data/raw/ACDC/training --output data/acdc
```

For [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/), obtain the DICOMs and XML reader annotations, install `pylidc`, and configure its DICOM directory according to the [pylidc documentation](https://pylidc.github.io/install.html). Inspect ambiguous nodule associations before including them.

```bash
python -m pip install -e ".[lidc]"
confirmseg prepare --source lidc --output data/lidc
```

The LIDC export retains individual contour volumes. A task selects one contour, fixes its largest-area slice and a 96 mm square crop around an interior foreground localization point, and resizes to 256 × 256. Localization counts as action 1. MRI normalization is volume-wise; CT is clipped to −1000 to 400 HU. Spacing is adjusted after resizing and used by dilation and ASSD.

You can also import curated NPZ cases with `prepare --source manifest --input data/cases.json --output data/manifest.json`. Each manifest entry needs `id`, `patient_id`, `dataset`, `path`, and optionally `modality`. The NPZ contains `image` (H × W or H × W × Z), binary `masks` (readers × image shape) and `spacing` in millimetres. Paths are relative to the input manifest. This import assigns patient splits and checks image/mask alignment.

## Training

First train the initial segmenter, then freeze its encoder and train the correction decoder and gate:

```bash
confirmseg train --stage initial --manifest data/acdc/manifest.json --output checkpoints/initial --seed 17
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --output checkpoints/confirmseg --method confirmseg --seed 17
```

`configs/default.json` specifies 40 initial epochs, 80 correction epochs, gradient accumulation over eight images, eight charged training actions, AdamW and cosine learning-rate decay. The four loss weights control gate supervision, editable gate area, regression on previously correct unapproved pixels, and physical boundary gradients. Defaults are implementation choices; adjust them using validation data. `--config` selects another JSON configuration. Each training directory contains `best.pt` selected by validation Dice and a numerical `history.json`.

## Interaction and evaluation

```bash
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --checkpoint checkpoints/confirmseg/best.pt --method confirmseg --output runs/confirmseg
confirmseg evaluate --inputs runs/confirmseg/results.json --output runs/comparison.json
confirmseg plot --input runs/comparison.json --output runs/figures
confirmseg plot --input runs/confirmseg/curves.json --output runs/curves
```

The simulator alternates largest-error clicks and short boundary strokes, with bounded review patches every third charged action. It uses reference labels to simulate reviewer feedback; the editor receives only image features, prompts and review state. An episode stops at 12 charged actions. Review patches cover at most 1% of the image, and repeated inspected area counts toward the 12% cumulative cap.

`results.json` stores actions, per-action confusion counts, conflict/revocation events, approval denominators, regression, rework, physical boundary distances and measured inference stages. `curves.json` stores micro-overlap at each charged budget. Example overlays show the reference, prediction and errors. `evaluate` recomputes aggregate metrics from raw snapshots and accepts several runs for comparison.

Dice and IoU share TP/FP/FN. Actions90 uses the first charged action reaching Dice 0.90; failures receive 13. Rework tracks an earlier repaired error component that reopened, using component IoU above 0.5. Approval violation measures disagreement with supplied labels across snapshots. Regression uses previously correct unapproved pixels as its denominator, including distance bands 0–3, 3–6 and over 6 mm. ASSD is in millimetres; one empty mask receives the ROI diagonal penalty and is counted separately. Area-weighted action cost is charged actions plus ten times cumulative reviewed fraction. Timing separates candidate decoding, the footprint head, merge, projection and dispatch with image features cached. The summary also retains trainable parameters and peak CPU/GPU memory.

## Baselines and ablations

Train a separate correction checkpoint for each method, using the same initial checkpoint, partitions and seed. Available baselines are `global`, `hard_lock`, `fixed_roi`, `progressive_merge`, `entropy_roi` and `dmp`; component removals are `no_approval`, `no_area`, `no_regression`, `no_boundary` and `no_revocation`.

```bash
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --output checkpoints/hard_lock --method hard_lock
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --checkpoint checkpoints/hard_lock/best.pt --method hard_lock --output runs/hard_lock
confirmseg evaluate --inputs runs/confirmseg/results.json runs/hard_lock/results.json --output runs/comparison.json
```

Use `--replay runs/global/results.json` to apply the neutral global trajectory to every method. Without replay, methods receive the same policy and RNG seed but feedback adapts to their predictions. `--wrong-approval 0.05` or `0.10` introduces controlled label flips; use `no_revocation` for the corresponding removal. `--stratum low|middle|high` selects initial Dice ranges [0.65,0.75), [0.75,0.85), [0.85,0.95). Repeat training with seeds 17, 29 and 43 for the paper's seed protocol.

The initial decoder receives the charged foreground localization disk so the same image can support distinct target structures. LIDC scan IDs remain separate while their patient split is shared.

The local and Progressive Merge baselines use the shared candidate architecture with spatial merge rules. DMP uses local correction and distant probability-preservation losses. These choices, largest-contour CT localization and the explicit simulated review schedule define the implemented comparison. They are exposed in the configuration and action records so results can be interpreted consistently.

## Experiment coverage

Run each data domain separately and repeat model fitting with `--seed 17`, `29` and `43`. A separate correction checkpoint is required for every trained row; `fixed_roi` is the fixed-gate removal. Evaluation keeps dataset and seed identifiers.

| Analysis | Execution setting | Computed outputs |
| --- | --- | --- |
| Main comparison | Every baseline and `confirmseg` | Micro-Dice/IoU, Actions90 and failures, rework, regression, approval violations, ASSD |
| Components | `no_approval`, `fixed_roi`, `no_area`, `no_regression`, `no_boundary` | The same accuracy and rework metrics |
| Common interactions | Add `--replay runs/global/results.json` to `run` | Matched action replay, changed/corrected pixels, beneficial/harmful remote changes |
| Initialization | Add `--stratum low`, `middle` or `high` | Metrics in the selected initial-Dice interval |
| Incorrect approvals | Add `--wrong-approval 0.05` or `0.10`; compare `no_revocation` | Conflicts, charged revocations, violations and final overlap |
| Efficiency | Included in each run | Candidate/gate/merge/projection/dispatch/total milliseconds, parameters, memory |

`evaluate` accepts all corresponding `results.json` paths. `plot` writes PNG and editable SVG figures from computed summaries. Keep replay and adaptive runs in separate output directories.
