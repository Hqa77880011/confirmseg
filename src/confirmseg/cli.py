import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import fixed_field, prepare_acdc, prepare_lidc, prepare_manifest, records
from .method import (
    Editor,
    InitialModel,
    METHODS,
    State,
    footprint_target,
    infer,
    loss_terms,
    physical_dilation,
    simulate_action,
)
from .metrics import assd, counts, overlap, summarize
from .network import segmentation_loss
from .utils import load, peak_memory_mib, plot_results, save, seed_all


def tensor(array, device):
    return torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None].to(device)


def localization_prompt(target, spacing):
    if not target.any():
        raise ValueError("The initial localization reader must contain foreground")
    distance = ndi.distance_transform_edt(target.astype(bool), sampling=spacing)
    point = np.zeros(target.shape, dtype=bool)
    point[np.unravel_index(distance.argmax(), target.shape)] = True
    return physical_dilation(point, 2, spacing).astype(np.float32)


def load_models(initial_path, editor_path, device):
    initial = InitialModel().to(device)
    initial.load_state_dict(
        torch.load(initial_path, map_location=device, weights_only=True)["model"]
    )
    initial.eval()
    editor = Editor().to(device)
    if editor_path:
        checkpoint = torch.load(editor_path, map_location=device, weights_only=True)
        editor.load_state_dict(checkpoint["model"])
    editor.eval()
    return initial, editor


def train(args):
    cfg = load(args.config)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    training, validation = (
        records(args.manifest, "train"),
        records(args.manifest, "validation"),
    )
    if args.stage == "initial":
        model = InitialModel().to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["initial_lr"], weight_decay=cfg["weight_decay"]
        )
        epochs = cfg["initial_epochs"]
        initial = None
    else:
        if not args.initial:
            raise ValueError("--initial is required for correction training")
        initial, model = load_models(args.initial, None, args.device)
        for parameter in initial.parameters():
            parameter.requires_grad_(False)
        if args.method in (
            "global",
            "hard_lock",
            "fixed_roi",
            "progressive_merge",
            "entropy_roi",
            "dmp",
        ):
            for parameter in model.gate.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        epochs = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-6
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        accumulated = 0
        loss_total = 0.0
        optimizer.zero_grad()
        for index in rng.permutation(len(training)):
            row = training[index]
            with np.load(row["path"], allow_pickle=False) as obj:
                reader = int(rng.integers(len(obj["masks"])))
            image, masks, spacing = fixed_field(row, initial=reader, size=cfg["size"])
            target = masks[reader]
            if rng.random() < 0.5:
                image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
            angle = rng.uniform(-10, 10)
            image = ndi.rotate(image, angle, reshape=False, order=1).astype(np.float32)
            target = ndi.rotate(target, angle, reshape=False, order=0).astype(
                np.float32
            )
            x, y = tensor(image, args.device), tensor(target, args.device)
            localization = localization_prompt(target, spacing)
            prompt = tensor(localization, args.device)
            if args.stage == "initial":
                loss = segmentation_loss(model(x, prompt), y)
            else:
                with torch.no_grad():
                    features = initial.encoder(x)
                    p = initial.decoder(features, prompt).sigmoid()[0, 0].cpu().numpy()
                state = State.initial(p, localization)
                losses = []
                while state.actions < cfg["unroll"]:
                    action = simulate_action(state, target, spacing, rng)
                    verdict, support = state.act(action, spacing, args.method)
                    if verdict == "conflict":
                        revoke = {
                            "type": "revoke",
                            "coordinates": np.argwhere(support).tolist(),
                        }
                        if state.actions < cfg["unroll"]:
                            state.act(revoke, spacing, args.method)
                        continue
                    if verdict != "correct":
                        continue
                    channels = torch.from_numpy(
                        state.channels(spacing, cfg["band_mm"])
                    )[None].to(args.device)
                    target_gate = footprint_target(
                        state.probability,
                        target,
                        state.labels >= 0,
                        support,
                        spacing,
                        cfg["band_mm"],
                    )
                    edited, candidate, gate = model(features, channels, args.method)
                    losses.append(
                        loss_terms(
                            edited,
                            candidate,
                            gate,
                            channels,
                            y,
                            tensor(target_gate, args.device),
                            spacing,
                            cfg["loss"],
                            args.method,
                        )
                    )
                    if args.method in (
                        "global",
                        "hard_lock",
                        "dmp",
                        "fixed_roi",
                        "progressive_merge",
                        "entropy_roi",
                    ):
                        with torch.no_grad():
                            infer(
                                initial,
                                model,
                                image,
                                state,
                                spacing,
                                args.method,
                                args.device,
                                cfg["roi_mm"],
                                features,
                                support,
                                cfg["band_mm"],
                            )
                    else:
                        state.probability = edited.detach()[0, 0].cpu().numpy()
                        state.revoked.fill(0)
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
            (loss / cfg["batch_size"]).backward()
            loss_total += float(loss.detach())
            accumulated += 1
            if accumulated % cfg["batch_size"] == 0:
                optimizer.step()
                optimizer.zero_grad()
        if accumulated % cfg["batch_size"]:
            scale = cfg["batch_size"] / (accumulated % cfg["batch_size"])
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            optimizer.step()
        scheduler.step()
        model.eval()
        values = []
        with torch.inference_mode():
            for row in validation:
                image, masks, spacing = fixed_field(row, size=cfg["size"])
                localization = localization_prompt(masks[0], spacing)
                prompt = tensor(localization, args.device)
                if args.stage == "initial":
                    prediction = (
                        model(tensor(image, args.device), prompt)[0, 0].cpu().numpy()
                    )
                else:
                    prediction = (
                        initial(tensor(image, args.device), prompt)[0, 0].cpu().numpy()
                    )
                    state = State.initial(prediction, localization)
                    local_rng = np.random.default_rng(31415)
                    for _ in range(3):
                        action = simulate_action(state, masks[0], spacing, local_rng)
                        verdict, support = state.act(action, spacing, args.method)
                        if verdict == "correct":
                            infer(
                                initial,
                                model,
                                image,
                                state,
                                spacing,
                                args.method,
                                args.device,
                                cfg["roi_mm"],
                                prompt_support=support,
                                band_mm=cfg["band_mm"],
                            )
                    prediction = state.probability
                values.append(overlap(counts(prediction, masks[0]))[0])
        score = float(np.mean(values))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_total / max(accumulated, 1),
                "validation_dice": score,
            }
        )
        if score > best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "seed": args.seed,
                    "method": args.method,
                    "epoch": epoch + 1,
                },
                output / "best.pt",
            )
        save(output / "history.json", history)
        print(
            f"Epoch {epoch + 1}: loss={history[-1]['loss']:.6f} validation Dice={score:.6f}",
            flush=True,
        )


def episode(row, reader, initial, editor, cfg, args, replay=None):
    image, masks, spacing = fixed_field(row, initial=reader, size=cfg["size"])
    truth = masks[reader]
    localization = localization_prompt(truth, spacing)
    prompt = tensor(localization, args.device)
    with torch.inference_mode():
        x = tensor(image, args.device)
        features = initial.encoder(x)
        probability = initial.decoder(features, prompt).sigmoid()[0, 0].cpu().numpy()
    state = State.initial(probability, localization)
    initial_dice = overlap(counts(probability, truth))[0]
    rng = np.random.default_rng(args.policy_seed)
    snapshots, actions, repaired, rework = [], [], [], 0
    last_update = {}
    pending = None
    actions90 = 1 if initial_dice >= 0.9 else cfg["budget"] + 1
    while True:
        metrics = counts(state.probability, truth)
        distance, empty_failure = assd(state.probability, truth, spacing)
        approved = state.labels >= 0
        metrics.update(
            {
                "action": state.actions,
                "assd_mm": distance,
                "empty_failure": empty_failure,
                "violation_numerator": int(
                    ((state.probability >= 0.5) != state.labels)[approved].sum()
                ),
                "confirmation_denominator": int(approved.sum()),
                "regression_numerator": 0,
                "prior_correct_unconfirmed": 0,
                "band_regression": {k: [0, 0] for k in ("0_3mm", "3_6mm", "over_6mm")},
            }
        )
        if snapshots:
            metrics.update(last_update)
        snapshots.append(metrics)
        if actions90 > cfg["budget"] and overlap(metrics)[0] >= 0.9:
            actions90 = state.actions
        if state.actions >= cfg["budget"]:
            break
        old = state.probability.copy()
        old_error = (old >= 0.5) != truth.astype(bool)
        if replay is not None:
            if len(actions) >= len(replay):
                break
            action = replay[len(actions)]
        elif pending is not None:
            action, pending = pending, None
        else:
            action = simulate_action(state, truth, spacing, rng, args.wrong_approval)
        actions.append(action)
        verdict, support = state.act(action, spacing, args.method)
        last_update = {
            "verdict": verdict,
            "regression_numerator": 0,
            "prior_correct_unconfirmed": 0,
            "band_regression": {k: [0, 0] for k in ("0_3mm", "3_6mm", "over_6mm")},
        }
        if verdict == "conflict" and args.method != "no_revocation" and replay is None:
            pending = {"type": "revoke", "coordinates": np.argwhere(support).tolist()}
        if verdict == "correct":
            error_components, _ = ndi.label(old_error, np.ones((3, 3)))
            ids = np.unique(error_components[support])
            selected = np.isin(error_components, ids[ids > 0])
            if any(
                (mask & selected).sum() / max((mask | selected).sum(), 1) > 0.5
                for mask in repaired
                if (mask & old_error).any()
            ):
                rework += 1
            last_update["timing"] = infer(
                initial,
                editor,
                image,
                state,
                spacing,
                args.method,
                args.device,
                cfg["roi_mm"],
                features,
                support,
                cfg["band_mm"],
            )
            error_now = (state.probability >= 0.5) != truth.astype(bool)
            for component in ids[ids > 0]:
                region = error_components == component
                if not (region & error_now).any():
                    repaired.append(region)
            prior_correct = ~old_error & (state.labels < 0)
            regression = prior_correct & error_now
            last_update["regression_numerator"] = int(regression.sum())
            last_update["prior_correct_unconfirmed"] = int(prior_correct.sum())
            edge = approved ^ ndi.binary_erosion(approved)
            distance_band = (
                ndi.distance_transform_edt(~edge, sampling=spacing)
                if edge.any()
                else np.full(old.shape, np.inf)
            )
            for name, low, high in [
                ("0_3mm", 0, 3),
                ("3_6mm", 3, 6),
                ("over_6mm", 6, np.inf),
            ]:
                subset = (distance_band >= low) & (distance_band < high)
                last_update["band_regression"][name] = [
                    int((regression & subset).sum()),
                    int((prior_correct & subset).sum()),
                ]
            last_update["changed_pixels"] = int(
                ((old >= 0.5) != (state.probability >= 0.5)).sum()
            )
            last_update["beneficial_remote_pixels"] = int(
                (old_error & ~error_now & ~selected).sum()
            )
            last_update["harmful_remote_pixels"] = int(
                (~old_error & error_now & ~selected).sum()
            )
            last_update["corrected_target_pixels"] = int((selected & ~error_now).sum())
    return (
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "reader": reader,
            "initial_dice": initial_dice,
            "budget": cfg["budget"],
            "charged_actions": state.actions,
            "actions90": actions90,
            "rework": rework,
            "reviewed_area_fraction": state.reviewed_area / truth.size,
            "actions": actions,
            "snapshots": snapshots,
        },
        image,
        truth,
        state.probability,
    )


def run(args):
    cfg = load(args.config)
    seed_all(args.seed)
    initial, editor = load_models(args.initial, args.checkpoint, args.device)
    replay = (
        {
            r["id"] + "_" + str(r["reader"]): r["actions"]
            for r in load(args.replay)["episodes"]
        }
        if args.replay
        else {}
    )
    episodes = []
    output = Path(args.output)
    for row in records(args.manifest, args.split):
        with np.load(row["path"], allow_pickle=False) as obj:
            readers = len(obj["masks"])
        for reader in range(readers):
            result, image, truth, prediction = episode(
                row,
                reader,
                initial,
                editor,
                cfg,
                args,
                replay.get(row["id"] + "_" + str(reader)),
            )
            if args.stratum:
                bounds = {
                    "low": (0.65, 0.75),
                    "middle": (0.75, 0.85),
                    "high": (0.85, 0.95),
                }[args.stratum]
                if not bounds[0] <= result["initial_dice"] < bounds[1]:
                    continue
            episodes.append(result)
            if len(episodes) <= args.overlays:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                output.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                for ax, mask, title in zip(
                    axes,
                    (
                        truth,
                        prediction >= 0.5,
                        (prediction >= 0.5) != truth.astype(bool),
                    ),
                    ("Target", "Prediction", "Error"),
                ):
                    ax.imshow(image, cmap="gray")
                    ax.imshow(
                        np.ma.masked_where(~mask.astype(bool), mask),
                        cmap="Blues",
                        alpha=0.5,
                    )
                    ax.set_title(title)
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(output / f"overlay_{len(episodes)}.png", dpi=180)
                plt.close(fig)
    if not episodes:
        raise ValueError("No eligible evaluation episodes")
    summary = summarize(episodes, args.method)
    summary.update(peak_memory_mib())
    summary["trainable_parameters"] = sum(
        p.numel()
        for p in (
            editor.candidate.parameters()
            if args.method
            in (
                "global",
                "hard_lock",
                "fixed_roi",
                "progressive_merge",
                "entropy_roi",
                "dmp",
            )
            else editor.parameters()
        )
    )
    curves = []
    for budget in range(1, cfg["budget"] + 1):
        selected = []
        for e in episodes:
            clone = dict(
                e, snapshots=[r for r in e["snapshots"] if r["action"] <= budget]
            )
            selected.append(clone)
        curves.append(dict(summarize(selected, args.method), budget=budget))
    save(
        output / "results.json",
        {
            "summary": [summary],
            "curves": curves,
            "episodes": episodes,
            "config": cfg,
            "method": args.method,
            "seed": args.seed,
            "policy_seed": args.policy_seed,
            "condition": {
                "wrong_approval": args.wrong_approval,
                "stratum": args.stratum,
                "replay": bool(args.replay),
            },
        },
    )
    save(output / "curves.json", {"summary": curves})


def main():
    parser = argparse.ArgumentParser(prog="confirmseg")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", choices=["acdc", "lidc", "manifest"], required=True)
    prep.add_argument("--input")
    prep.add_argument("--output", required=True)
    training = commands.add_parser("train")
    training.add_argument("--stage", choices=["initial", "correction"], required=True)
    training.add_argument("--initial")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--method", choices=METHODS, default="confirmseg")
    execution = commands.add_parser("run")
    execution.add_argument("--manifest", required=True)
    execution.add_argument("--initial", required=True)
    execution.add_argument("--checkpoint", required=True)
    execution.add_argument("--method", choices=METHODS, default="confirmseg")
    execution.add_argument("--split", choices=["validation", "test"], default="test")
    execution.add_argument("--output", required=True)
    execution.add_argument("--policy-seed", type=int, default=31415)
    execution.add_argument("--wrong-approval", type=float, default=0)
    execution.add_argument("--replay")
    execution.add_argument("--stratum", choices=["low", "middle", "high"])
    execution.add_argument("--overlays", type=int, default=3)
    for item in (training, execution):
        item.add_argument("--config", default="configs/default.json")
        item.add_argument("--seed", type=int, default=17)
        item.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--inputs", nargs="+", required=True)
    evaluation.add_argument("--output", required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source == "lidc":
            prepare_lidc(args.output)
        elif args.source == "acdc":
            prepare_acdc(args.input, args.output)
        else:
            prepare_manifest(args.input, args.output)
    elif args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        rows = []
        for path in args.inputs:
            result = load(path)
            for dataset in sorted({e["dataset"] for e in result["episodes"]}):
                episodes = [e for e in result["episodes"] if e["dataset"] == dataset]
                row = summarize(episodes, result["method"])
                for key in (
                    "peak_process_memory_mib",
                    "peak_gpu_memory_mib",
                    "trainable_parameters",
                ):
                    row[key] = result["summary"][0].get(key)
                row.update(
                    seed=result["seed"],
                    dataset=dataset,
                    condition=result.get("condition", {}),
                )
                rows.append(row)
        save(args.output, {"summary": rows})
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
