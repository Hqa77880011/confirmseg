"""Small synthetic end-to-end checks; outputs are functional checks, not experiments."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .cli import argument_parser, load_models, main, tensor, train
from .data import fixed_field, records
from .method import METHODS, Editor, footprint_target, loss_terms
from .utils import load, save, seed_all


def verify(output, device="cpu"):
    """Retain inspectable data/checkpoints under output; never run the full schedule."""

    def fit(argv):
        return train(argument_parser().parse_args(argv))

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed_all(17)
    rng = np.random.default_rng(17)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    yy, xx = np.indices((32, 32))
    rows = []
    for index, split in enumerate(("train", "train", "validation", "test", "test")):
        target = (yy - 15) ** 2 + (xx - 16) ** 2 <= (5 + index % 2) ** 2
        image = (target + rng.normal(0, 0.1, target.shape)).astype(np.float32)
        path = raw / f"case_{index}.npz"
        np.savez_compressed(
            path,
            image=image,
            masks=target[None].astype(np.uint8),
            spacing=np.array([1.0, 1.0]),
        )
        rows.append(
            dict(
                id=f"case_{index}",
                patient_id=f"p{index}",
                dataset="synthetic_check",
                split=split,
                path=path.name,
                normalized=True,
                purpose="functional_check",
            )
        )
    save(raw / "cases.json", rows)
    manifest = output / "data" / "manifest.json"
    main(
        [
            "prepare",
            "--source",
            "manifest",
            "--input",
            str(raw / "cases.json"),
            "--output",
            str(manifest),
        ]
    )

    # Exercise the actual NIfTI adapter and patient stratification without real scans.
    acdc = output / "nifti_fixture"
    for index in range(1, 6):
        patient = acdc / f"patient{index:03d}"
        patient.mkdir(parents=True, exist_ok=True)
        (patient / "Info.cfg").write_text(
            "Group: NOR\nED: 1\nES: 2\n", encoding="utf-8"
        )
        for frame in (1, 2):
            image = rng.normal(size=(12, 10, 2)).astype(np.float32)
            labels = np.zeros_like(image, dtype=np.uint8)
            labels[1:4, 1:4] = 1
            labels[5:8, 1:4] = 2
            labels[3:7, 6:9] = 3
            stem = patient / f"patient{index:03d}_frame{frame:02d}"
            nib.save(
                nib.Nifti1Image(image, np.diag([1.5, 2.0, 5.0, 1.0])),
                str(stem) + ".nii.gz",
            )
            nib.save(
                nib.Nifti1Image(labels, np.diag([1.5, 2.0, 5.0, 1.0])),
                str(stem) + "_gt.nii.gz",
            )
    main(
        [
            "prepare",
            "--source",
            "acdc",
            "--input",
            str(acdc),
            "--output",
            str(output / "acdc"),
        ]
    )
    inventory = load(output / "acdc" / "manifest.json")
    assert {r["split"] for r in inventory} == {"train", "validation", "test"}
    case = records(output / "acdc" / "manifest.json", "test")[0]
    _, _, spacing = fixed_field(case, size=32)
    np.testing.assert_allclose(spacing, [12 * 1.5 / 32, 10 * 2 / 32])

    cfg = dict(
        size=32,
        initial_epochs=1,
        initial_lr=0.001,
        epochs=1,
        lr=0.0001,
        weight_decay=0.0001,
        batch_size=1,
        unroll=5,
        budget=6,
        validation_actions=6,
        band_mm=3,
        roi_mm=16,
        fixed_gate=0.5,
        loss=dict(gate=1.0, area=0.01, regression=0.5, boundary=0.1),
    )
    config = output / "config.json"
    save(config, cfg)
    initial_path = output / "initial" / "best.pt"
    initial = fit(
        [
            "train",
            "--stage",
            "initial",
            "--manifest",
            str(manifest),
            "--output",
            str(initial_path.parent),
            "--config",
            str(config),
            "--device",
            device,
        ]
    )
    editors = {}
    for method in (
        "hard_lock",
        "confirmseg",
        "no_approval",
        "candidate_only",
        "gate_only",
    ):
        editors[method] = fit(
            [
                "train",
                "--stage",
                "correction",
                "--manifest",
                str(manifest),
                "--initial",
                str(initial_path),
                "--output",
                str(output / method),
                "--method",
                method,
                "--config",
                str(config),
                "--device",
                device,
            ]
        )
    checkpoint = output / "confirmseg" / "best.pt"
    restored_initial, restored_editor = load_models(initial_path, checkpoint, device)
    sample = tensor(rng.normal(size=(32, 32)).astype(np.float32), device)
    context = torch.zeros(1, 7, 32, 32, device=device)
    context[:, 0], context[:, 3, 5, 5] = 0.3, 1
    initial.eval()
    editors["confirmseg"].eval()
    with torch.inference_mode():
        expected_initial = initial(sample, torch.zeros_like(sample))
        actual_initial = restored_initial(sample, torch.zeros_like(sample))
        features = initial.encoder(sample)
        expected = editors["confirmseg"](features, context)[0]
        actual = restored_editor(restored_initial.encoder(sample), context)[0]
    torch.testing.assert_close(actual_initial, expected_initial, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    # All spatial controls must update through the same rule used for evaluation.
    target = np.zeros((32, 32), np.float32)
    target[8:24, 8:24] = 1
    support = np.zeros_like(target, bool)
    support[16, 16] = True
    for method in METHODS:
        editor = Editor().to(device)
        initial_features = [feature.detach().clone() for feature in features]
        state = context.clone()
        gate_target = footprint_target(
            state[0, 0].cpu().numpy(),
            target,
            state[0, 3].cpu().numpy() > 0,
            support,
            np.array([1.0, 1.0]),
            3,
        )
        updated, candidate, gate = editor(
            initial_features,
            state,
            method,
            spacing=np.array([1.0, 1.0]),
            support=support,
        )
        loss = loss_terms(
            updated,
            candidate,
            gate,
            state,
            tensor(target, device),
            tensor(gate_target, device),
            (1, 1),
            cfg["loss"],
            method,
        )
        assert torch.isfinite(loss), method
        loss.backward()
        assert editor.candidate.output.weight.grad is not None, method

    for method in ("hard_lock", "confirmseg"):
        main(
            [
                "run",
                "--manifest",
                str(manifest),
                "--initial",
                str(initial_path),
                "--checkpoint",
                str(output / method / "best.pt"),
                "--method",
                method,
                "--output",
                str(output / (method + "_run")),
                "--overlays",
                "1",
                "--device",
                device,
            ]
        )
    replay = output / "hard_lock_run" / "results.json"
    for method in ("confirmseg", "no_approval", "candidate_only", "gate_only"):
        main(
            [
                "run",
                "--manifest",
                str(manifest),
                "--initial",
                str(initial_path),
                "--checkpoint",
                str(output / method / "best.pt"),
                "--method",
                method,
                "--replay",
                str(replay),
                "--output",
                str(output / (method + "_replay")),
                "--overlays",
                "0",
                "--device",
                device,
            ]
        )
    summary = output / "comparison.json"
    main(
        [
            "evaluate",
            "--inputs",
            str(output / "confirmseg_run" / "results.json"),
            str(replay),
            "--paired",
            "confirmseg",
            "hard_lock",
            "--bootstrap",
            "100",
            "--output",
            str(summary),
        ]
    )
    main(["plot", "--input", str(summary), "--output", str(output / "figures")])
    main(
        [
            "plot",
            "--input",
            str(output / "confirmseg_run" / "curves.json"),
            "--output",
            str(output / "figures"),
        ]
    )
    image = rows[0]
    with np.load(raw / image["path"], allow_pickle=False) as case:
        localization = np.zeros_like(case["image"])
        localization[15, 16] = 1
        np.savez_compressed(
            output / "input.npz",
            image=case["image"],
            spacing=case["spacing"],
            localization=localization,
        )
    actions = [
        dict(type="confirm", coordinates=[[5, 5]], labels=[1]),
        dict(type="correct", coordinates=[[5, 5]], labels=[0]),
        dict(type="revoke", coordinates=[[5, 5]]),
        dict(type="correct", coordinates=[[5, 5]], labels=[0]),
    ]
    save(output / "actions.json", actions)
    main(
        [
            "predict",
            "--input",
            str(output / "input.npz"),
            "--actions",
            str(output / "actions.json"),
            "--initial",
            str(initial_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output / "session"),
            "--device",
            device,
        ]
    )
    assert [row["verdict"] for row in load(output / "session" / "trace.json")] == [
        "confirm",
        "conflict",
        "revoke",
        "correct",
    ]
    main(
        [
            "benchmark",
            "--input",
            str(output / "session" / "final.npz"),
            "--initial",
            str(initial_path),
            "--checkpoint",
            str(checkpoint),
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--output",
            str(output / "timing.json"),
            "--device",
            device,
        ]
    )
    losses = {
        method: load(output / method / "history.json")[0]["loss"] for method in editors
    }
    assert all(np.isfinite(value) for value in losses.values())
    verification = dict(
        purpose="functional_check",
        size=32,
        training_episodes=2,
        epochs_per_fit=1,
        device=device,
        losses_finite=True,
        checkpoint_max_error=float((actual - expected).abs().max()),
        initial_checkpoint_max_error=float(
            (actual_initial - expected_initial).abs().max()
        ),
        checked_methods=list(METHODS),
        paths=[
            "curated NPZ",
            "ACDC NIfTI",
            "initial training",
            "correction training",
            "neutral replay",
            "evaluation",
            "paired bootstrap",
            "plots",
            "label-free actions",
            "cached-feature timing",
        ],
    )
    save(output / "verification.json", verification)
    print(f"Functional checks passed; retained outputs: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    verify(arguments.output, arguments.device)
