# ConfirmSeg

Implementation of **ConfirmSeg: Approval-Conditioned Edit Footprints for Interactive Medical Segmentation**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Obtain and extract the labeled [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) training cohort. Preparation creates binary structure tasks and keeps each patient in one split (60/20/20, seed 2027).

```bash
confirmseg prepare --source acdc --input data/raw/ACDC/training --output data/acdc
```

For [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/), obtain DICOMs and individual XML contours, install `python -m pip install -e ".[lidc]"`, and configure the [pylidc DICOM path](https://pylidc.github.io/install.html). Then run `confirmseg prepare --source lidc --output data/lidc`. Curated NPZ cases can use `prepare --source manifest --input cases.json --output data/custom/manifest.json`; records need `id,patient_id,dataset,path`, and NPZ arrays `image,masks,spacing`.

## Training

Train the initial segmenter first, then the correction model. `configs/default.json` records fitting and interaction settings; `--config` selects another file. Keep seed 17 fixed through a paired run.

```bash
confirmseg train --stage initial --manifest data/acdc/manifest.json --output checkpoints/initial --seed 17
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --output checkpoints/confirmseg --method confirmseg --seed 17
```

## Inference and evaluation

`run` simulates reviewer feedback on the test split. Its fixed policy seed is explicit; use `--replay` with an existing results file for matched action replay.

```bash
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --checkpoint checkpoints/confirmseg/best.pt --method confirmseg --seed 17 --policy-seed 31415 --output runs/confirmseg
confirmseg evaluate --inputs runs/confirmseg/results.json --output runs/comparison.json
confirmseg plot --input runs/comparison.json --output runs/figures
```

## Comparisons and ablations

Train a separate correction checkpoint for every method. Component removals are `no_approval`, `no_area`, `no_regression`, `no_boundary` and `no_revocation`; spatial controls include `hard_lock` and `fixed_roi`.

```bash
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --method no_area --seed 17 --output checkpoints/no_area
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/initial/best.pt --checkpoint checkpoints/no_area/best.pt --method no_area --seed 17 --policy-seed 31415 --output runs/no_area
confirmseg evaluate --inputs runs/confirmseg/results.json runs/no_area/results.json --output runs/comparison.json
```

## Outputs

Each fitting directory contains validation-selected `best.pt` and `history.json`. Interaction runs contain `results.json`, `curves.json` and example overlays. Evaluation writes the requested aggregate JSON; plotting creates PNG/SVG figures. Results retain config, model seed, policy seed and per-action records.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
