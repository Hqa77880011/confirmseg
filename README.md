# ConfirmSeg

**Approval-Conditioned Edit Footprints for Interactive Medical Segmentation**

ConfirmSeg maintains separate foreground/background approvals and revocations, predicts a continuous edit footprint, and projects updates onto active approved labels. Image features are cached across corrections.

## Installation

Use Python 3.10 or newer. Run commands from the repository root in a virtual environment:

~~~bash
python -m venv .venv
~~~

Activate with source .venv/bin/activate on Linux/macOS or .venv\Scripts\Activate.ps1 in PowerShell, then install:

~~~bash
python -m pip install -e .
confirmseg --help
~~~

Every model command accepts --device cpu or --device cuda. CUDA is selected when available. The optional raw LIDC adapter uses Python 3.11 and the lidc extra:

~~~bash
python -m pip install -e ".[lidc]"
~~~

Curated LIDC NPZ data use the core environment.

## Data

### ACDC

Download the labeled training cohort from the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), including each patient's Info.cfg, ED/ES images and ground-truth NIfTI files.

~~~bash
confirmseg prepare --source acdc --input data/raw/ACDC/training --output data/acdc --split-seed 2027
~~~

Preparation standardizes each MRI volume using its nonzero tissue voxels before deriving binary RV/myocardium/LV slice tasks. Patients are split 60/20/20 within diagnostic groups; all phases, structures and slices stay together. For the 100-patient labeled cohort this gives 60/20/20 patients. This is a study split of the labeled cohort.

Images and nearest-neighbor masks use a 256×256 grid, with spacing adjusted to preserve physical distances. The default evaluation protocol selects five distinct nonempty phase/structure/slice tasks per test patient, shuffling tasks and cycling phase/structure strata with policy seed 31415. Training and validation retain all nonempty structure tasks. The --protocol all option instead evaluates all tasks/readers.

### LIDC-IDRI

Obtain the CT DICOM collection from [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/). The raw adapter reads the contour database supplied by [pylidc](https://pylidc.github.io/); configure its [DICOM path](https://pylidc.github.io/install.html) before preparation.

~~~bash
confirmseg prepare --source lidc --output data/lidc --cohort 500 --split-seed 2027
~~~

Eligible nodules have median annotated diameter ≥3 mm. The adapter deterministically selects 500 eligible patients, selects the largest eligible nodule per patient, and splits patients 300/100/100. Nodule ties use scan/cluster order; contour order uses pylidc annotation IDs, retained as reader_ids. Training can sample retained contours; patient-protocol evaluation uses the designated first contour. Setting --cohort 0 retains all eligible patients and changes the cohort size.

For each selected contour, its maximal-area slice and distance-transform center define a 96×96 mm crop. CT is clipped to [-1000,400] HU and scaled to [0,1], then resized to 256×256 with updated spacing. This is annotation-assisted localized correction; its localization prompt is charged as action 1. All nodules, readers and slices from a patient remain in the same split.

### Curated arrays

A JSON manifest is a list of records:

~~~json
[
  {
    "id": "case_001",
    "patient_id": "patient_001",
    "dataset": "LIDC",
    "modality": "CT",
    "path": "arrays/case_001.npz",
    "split": "train"
  }
]
~~~

Each NPZ contains image (H×W or H×W×Z), binary masks (readers×image dimensions), and positive physical spacing (one millimetre value per spatial axis). Paths are relative to the source manifest. CT arrays contain HU; MRI normalization uses the loaded volume/case. Set normalized to true for already normalized images. Prepared ACDC arrays already carry this flag.

~~~bash
confirmseg prepare --source manifest --input data/raw/cases.json --output data/custom/manifest.json --split-seed 2027
~~~

Explicit splits are preserved and checked for patient leakage. Otherwise all records must omit split, and a seeded patient split is created. Curated ACDC patient-protocol records also need phase (ED/ES) and structure (1/2/3); curated LIDC records can include diameter_mm and ordered reader_ids. There is no population transform fitted on evaluation patients or target-domain adaptation.

## Training

Train an initial segmenter, then one correction checkpoint per method and seed:

~~~bash
confirmseg train --stage initial --manifest data/acdc/manifest.json --config configs/default.json --seed 17 --output checkpoints/acdc/s17/initial
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/acdc/s17/initial/best.pt --config configs/default.json --method confirmseg --seed 17 --output checkpoints/acdc/s17/confirmseg
~~~

Repeat with seeds 29 and 43, keeping the same initial checkpoint within each paired seed. Replace the manifest/output domain for LIDC.

The encoder widths are (32,64,128,256), with two 3×3 convolutions, GroupNorm(8), ReLU and average pooling per scale. The candidate decoder uses four blocks (128,64,32,32). The footprint head receives 135 quarter-resolution channels (128 image + 7 state). State channels are previous probability, positive/negative prompts, foreground/background approvals, revocations, and a 3 mm approval-boundary band. Click radius is 2 mm; stroke half-width is 1 mm.

The [default configuration](configs/default.json) specifies 40 initial epochs at lr 1e-3 and 80 correction epochs at lr 1e-4, AdamW weight decay 1e-4, cosine floor 1e-6, eight accumulated episodes per update, and at most eight charged training actions. Training augmentation is horizontal flip (p=0.5) and rotation uniform in [-10°,10°]. Correction training freezes the initial segmenter, caches encoder features per episode, averages losses over decoded corrections, and detaches state between actions.

The objective is full-grid BCE + soft Dice, plus gate BCE, gate area, regression and boundary-gradient losses with weights (1,0.01,0.5,0.1). The gate target dilates touched 8-connected error components by 3 mm and excludes approvals. Masked means of empty sets contribute zero; confirmations/revocations receive no gate targets.

Checkpoint selection uses patient-macro validation Dice; ties retain the earlier epoch. Default validation uses localization plus three subsequent actions. The [E02 configuration](configs/e02.json) selects at the full twelve-action validation budget for the four head-conditioning variants. Test labels never select checkpoints, seeds or hyperparameters.

## Inference and offline interaction

### Explicit user actions, without reference labels

Prepare a normalized 256×256 image, a two-element spacing, and a binary positive localization prompt in an NPZ. No reference mask is read. The localization prompt is action 1. Action coordinates are integer [row,column] pairs on this grid:

~~~json
[
  {"type":"confirm","coordinates":[[100,100]],"labels":[1]},
  {"type":"correct","coordinates":[[150,150]],"labels":[0],"prompt":"click"},
  {"type":"revoke","coordinates":[[100,100]]}
]
~~~

For a stroke, supply its sampled coordinates with corresponding binary labels and prompt set to stroke.

~~~bash
confirmseg predict --input data/session_input.npz --actions data/actions.json --initial checkpoints/acdc/s17/initial/best.pt --checkpoint checkpoints/acdc/s17/confirmseg/best.pt --output runs/session
~~~

Conflicting labeled actions are charged and return a conflict overlay without changing probabilities or prompts. A subsequent explicit revocation is required before replacement. A review patch covers at most 1% of the grid; accepted cumulative review area, including repeated inspection, is capped at 12%. Confirmation/revocation do not decode the image. Active supplied labels remain binding until revoked, including incorrect approvals.

The output final.npz contains the image, probability, mask, change overlay and complete interaction state. It can be passed as the next --input; trace.json records each verdict and overlay coordinates.

### Simulated feedback and scoring

~~~bash
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/acdc/s17/initial/best.pt --checkpoint checkpoints/acdc/s17/confirmseg/best.pt --method confirmseg --seed 17 --policy-seed 31415 --output runs/acdc/s17/confirmseg
~~~

The run command uses reference labels only to simulate reviewer actions and score outputs. It alternates largest-error center clicks/local boundary samples and supplies random square review patches on actions divisible by three. Corrections, review, revocation, conflicts and localization all consume the twelve-action budget. The adaptive policy reads each method's current mask. Configuration is loaded from the correction checkpoint unless --config explicitly overrides it.

## Evaluation

~~~bash
confirmseg evaluate --inputs runs/acdc/s17/confirmseg/results.json --output runs/acdc/evaluation.json
confirmseg plot --input runs/acdc/evaluation.json --output runs/acdc/figures
confirmseg plot --input runs/acdc/s17/confirmseg/curves.json --output runs/acdc/figures/s17
~~~

Pass all method/seed result files to the same evaluate --inputs invocation. Output separates summary (per-seed pooled counts), pooled_summary (counts pooled over supplied seeds), and seed_summary (mean/sample SD of per-seed estimands). Patient-macro measures are nested separately.

- **Dice / IoU:** final TP/FP/FN are pooled before division; D=2J/(1+J) holds for these shared counts, not separately averaged patient metrics.
- **Actions90:** first charged action attaining Dice≥0.90; failure receives 13 under the default budget.
- **Rework:** individual previously repaired/reopened 8-connected components are matched at IoU>0.5; at most one repeated correction is counted per action.
- **Approval violation:** disagreements divided by active approved-pixel evaluations across all action snapshots.
- **Regression:** newly wrong / previously correct editable pixels, pooling counts across decoded corrections.
- **ASSD:** symmetric boundary distance in millimetres; both empty gives zero, one empty gives the physical ROI-diagonal penalty and an empty-mask failure.
- **E02 repair / changed:** repaired old errors in prompt-touched components / their old errors; binary changes outside approvals / editable pixels. Empty denominators are reported as null.
- **E03 cost:** A90+α·reviewed-area fraction at the first hit, α=0,10,50. Failures use action-12 area and A90=13. Patient means precede seed averaging; repeated accepted review is charged again.
- **E04 bands:** [0,3), [3,6), [6,∞) mm from the active approval boundary; only decoded corrections with nonempty approvals contribute. Eligible counts are pooled within each seed before seed mean/sample SD.
- **RA01:** trapezoid mean Dice on recorded budget knots [1,3,5,7,9,11,12], normalized over actions 1–5 or 1–12; success-only Actions90 retains failures separately.

For E01, include the same patient/task/reader/budget records for both methods at seeds 17,29,43:

~~~bash
confirmseg evaluate --inputs runs/acdc/s17/confirmseg/results.json runs/acdc/s17/dmp/results.json runs/acdc/s29/confirmseg/results.json runs/acdc/s29/dmp/results.json runs/acdc/s43/confirmseg/results.json runs/acdc/s43/dmp/results.json --paired confirmseg dmp --bootstrap 10000 --bootstrap-seed 271828 --output runs/acdc/paired.json
~~~

Episode values are averaged within patients, across patients, then across seeds. Paired patient clusters are resampled jointly for both methods and all seeds; intervals use 2.5/97.5 percentiles conditional on the fitted seeds. Dice differences are in percentage points. Sample SD needs multiple seeds; a single-seed SD remains null.

## Baselines and ablations

All variants use the same encoder, initialization and candidate architecture. Train a separate correction checkpoint with the chosen --method, then evaluate that matching method.

| Method | Update / supervision |
| --- | --- |
| global | Candidate everywhere, without hard projection |
| hard_lock | Candidate outside active approvals, with projection |
| fixed_roi | Current prompt support dilated by 16 mm, with projection |
| progressive_merge | Candidate/old-mask disagreement components touched by the current prompt, with projection |
| entropy_roi | Smoothed candidate entropy outside approvals centers a 16 mm ROI, with projection |
| dmp | Segmentation + local absolute error + remote preservation, with projection |
| fixed_gate | Constant amplitude 0.5 outside approvals, set by fixed_gate |
| no_area, no_regression, no_boundary | Remove the corresponding loss term |
| no_revocation | Learned footprint with revocation disabled |

The four E02 conditions retain the same full architecture, losses and true approvals for exclusion/projection:

| Candidate / footprint approval inputs | Method |
| --- | --- |
| 0 / 0 | no_approval |
| 1 / 0 | candidate_only |
| 0 / 1 | gate_only |
| 1 / 1 | confirmseg |

Train each with configs/e02.json and the same per-seed initial checkpoint:

~~~bash
confirmseg train --stage correction --manifest data/acdc/manifest.json --initial checkpoints/acdc/s17/initial/best.pt --config configs/e02.json --method candidate_only --seed 17 --output checkpoints/acdc/s17/candidate_only
~~~

Create the fixed neutral trajectory using the hard-lock seed-17 checkpoint, then reuse that file for each E02 condition and all three seeds:

~~~bash
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/acdc/s17/initial/best.pt --checkpoint checkpoints/acdc/s17/hard_lock/best.pt --method hard_lock --seed 17 --output runs/acdc/neutral
confirmseg run --manifest data/acdc/manifest.json --initial checkpoints/acdc/s17/initial/best.pt --checkpoint checkpoints/acdc/s17/candidate_only/best.pt --method candidate_only --seed 17 --replay runs/acdc/neutral/results.json --output runs/acdc/s17/candidate_only_replay
~~~

Checkpoint configuration carries the full inference budget; replay requires the same grid, budget and complete task set. Fixed replay measures responses to identical supplied actions; adaptive trajectories supply rework/cost comparisons.

For wrong-approval interventions, add --wrong-approval 0.05 or 0.10. Default --wrong-mode event reverses an entire selected review patch's labels with that probability; --wrong-mode pixel flips individual reviewed pixels. The --disable-revocation option applies the no-revocation intervention to any baseline, including progressive_merge. Initial-quality filtering uses --stratum low [0.65,0.75), middle [0.75,0.85), or high [0.85,0.95), with natural held-out initial predictions.

The historical wrong-approval table used an event-wise sub-patch protocol whose exact sub-patch construction is not recorded locally. The current event option treats the complete supplied review patch as one event; the pixel option is a separate intervention. Neither option establishes reproduction of that historical table. E01–E04 use correct approvals and are unaffected by this distinction.

## Cached-feature timing

~~~bash
confirmseg benchmark --input runs/session/final.npz --initial checkpoints/acdc/s17/initial/best.pt --checkpoint checkpoints/acdc/s17/confirmseg/best.pt --warmup 20 --repeats 200 --output runs/session/timing.json
~~~

Batch-one timing excludes image encoding, synchronizes CUDA, and reports candidate decoding, gate/merge, hard projection, dispatch, total latency, active parameter count, process peak memory and CUDA peak allocation. It reuses the same supplied state each repetition. Smaller changed regions do not skip candidate decoding.

## Paper modules and code

| Module / metric | Entry point | Configuration / command |
| --- | --- | --- |
| Patient grouping, normalization, physical ROI | data.py | prepare; --split-seed, --cohort, --protocol |
| Encoder / candidate / footprint | network.py, method.Editor | train; configs/default.json |
| Approval, conflict, revocation, projection | method.State, method.infer | predict, run |
| Five-term objective and footprint target | method.loss_terms, method.footprint_target | loss, band_mm |
| Spatial controls / component removals / E02 heads | method.Editor, cli.train | --method; configs/e02.json |
| Dice, IoU, Actions90, rework, ASSD, violation | metrics.summarize | evaluate |
| E01 paired patient uncertainty | metrics.paired_intervals | evaluate --paired ... |
| E02 repair/change, E03 first-hit pricing, E04 bands | metrics.transition, metrics.episode_statistics | run, evaluate |
| RA01 budget summaries | metrics.budget_integral | evaluate |
| Overlays, curves, comparisons, paired intervals | utils.plot_results | plot |
| Cached-feature stages, parameters, memory | cli.benchmark | benchmark |

## Outputs and verification

Training produces validation-selected best.pt and history.json. Checkpoints retain model seed, method, epoch, configuration and validation budget. Runs produce per-action records, complete action lists, results.json, curves.json and selected overlays. Evaluation produces per-seed, pooled and patient-macro summaries; plotting writes PNG/SVG.

~~~bash
python -m unittest discover -s tests -v
python -m confirmseg.verify --output runs/smoke --device cpu
~~~

The verify command creates a synthetic 32×32 fixture, exercises NIfTI/NPZ preparation, one initial fit and five one-epoch correction fits (two training cases each), all method forward/loss/backward paths, checkpoint reload, neutral replay, aggregation, plots, explicit label-free actions and a short timing check. It retains inspectable outputs under the chosen directory. These data and outputs are functional checks, not paper experiments.

Validated on Windows, Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 2.1.3 (OpenBLAS), SciPy 1.15.3, Matplotlib 3.10.0 and nibabel 5.4.2. Numerical checks cover hand-computed losses/metrics, patient pairing, first-hit area, both learned heads' gradients/parameter updates and output-preserving checkpoint reload. CUDA interaction/timing was also exercised on an RTX 4060 Laptop GPU. Full medical-data training, real LIDC DICOM preparation and the paper's reported scores were not rerun; trained checkpoints and medical data are not bundled.

The existing MIT license applies to the source code. Dataset access and use follow the respective providers' terms.
