import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import (
    evaluation_tasks,
    fixed_field,
    prepare_acdc,
    prepare_lidc,
    prepare_manifest,
    records,
)
from .method import (
    Editor,
    InitialModel,
    METHODS,
    SPATIAL_METHODS,
    State,
    footprint_target,
    infer,
    loss_terms,
    physical_dilation,
    simulate_action,
)
from .metrics import (
    BANDS,
    assd,
    budget_integral,
    counts,
    episode_statistics,
    overlap,
    paired_intervals,
    rework_update,
    seed_statistics,
    summarize,
    transition,
)
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
    """Load inference modules; no checkpoint code or optimizer objects are executed."""
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
        if args.method in SPATIAL_METHODS + ("fixed_gate",):
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
                    edited, candidate, gate = model(
                        features,
                        channels,
                        args.method,
                        spacing=spacing,
                        support=support,
                        radius=cfg["roi_mm"],
                        fixed_gate=cfg.get("fixed_gate", 0.5),
                    )
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
                    state.probability = edited.detach()[0, 0].cpu().numpy()
                    state.revoked.fill(0)
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            (loss / cfg["batch_size"]).backward()
            loss_total += float(loss.detach())
            accumulated += 1
            if accumulated % cfg["batch_size"] == 0:
                optimizer.step()
                optimizer.zero_grad()
        if not accumulated:
            raise ValueError("No training episodes produced a decoded correction loss")
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
                    validation_cfg = dict(cfg, budget=cfg.get("validation_actions", 4))
                    validation_args = SimpleNamespace(
                        device=args.device,
                        method=args.method,
                        policy_seed=31415,
                        wrong_approval=0,
                        wrong_mode="event",
                    )
                    record, _, _, prediction = episode(
                        row, 0, initial, model, validation_cfg, validation_args
                    )
                values.append(
                    (row["patient_id"], overlap(counts(prediction, masks[0]))[0])
                )
        score = float(
            np.mean(
                [
                    np.mean([value for key, value in values if key == patient])
                    for patient in sorted({key for key, _ in values})
                ]
            )
        )
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
                    "stage": args.stage,
                    "validation_actions": cfg.get("validation_actions", 4)
                    if args.stage == "correction"
                    else 1,
                    "epoch": epoch + 1,
                },
                output / "best.pt",
            )
        save(output / "history.json", history)
        print(
            f"Epoch {epoch + 1}: loss={history[-1]['loss']:.6f} validation Dice={score:.6f}",
            flush=True,
        )
    return model


def snapshot(state, truth, spacing, rework, update=None):
    values = counts(state.probability, truth)
    distance, failure = assd(state.probability, truth, spacing)
    approved = state.labels >= 0
    values.update(
        action=state.actions,
        assd_mm=distance,
        empty_failure=failure,
        violation_numerator=int(
            ((state.probability >= 0.5) != state.labels)[approved].sum()
        ),
        confirmation_denominator=int(approved.sum()),
        regression_numerator=0,
        prior_correct_unconfirmed=0,
        band_regression={name: [0, 0] for name, _, _ in BANDS},
        reviewed_area_fraction=state.reviewed_area / truth.size,
        rework=rework,
    )
    values.update(update or {})
    return values


def episode(row, reader, initial, editor, cfg, args, replay=None):
    """Offline simulation uses labels for feedback/scoring, never as network inputs."""
    image, masks, spacing = fixed_field(row, initial=reader, size=cfg["size"])
    truth = masks[reader]
    localization = localization_prompt(truth, spacing)
    initial.eval()
    editor.eval()
    with torch.inference_mode():
        features = initial.encoder(tensor(image, args.device))
        probability = (
            initial.decoder(features, tensor(localization, args.device))
            .sigmoid()[0, 0]
            .cpu()
            .numpy()
        )
    state = State.initial(probability, localization)
    initial_dice = overlap(counts(probability, truth))[0]
    rng = np.random.default_rng(args.policy_seed)
    snapshots, actions, repaired, pending = [], [], [], []
    rework = 0
    snapshots.append(snapshot(state, truth, spacing, rework))
    if replay is not None and len(replay) != cfg["budget"] - 1:
        raise ValueError("Replay must contain exactly budget-1 charged actions")
    while state.actions < cfg["budget"]:
        old = state.probability.copy()
        if replay is not None:
            action = replay[len(actions)]
        elif pending:
            action = pending.pop(0)
        else:
            action = simulate_action(
                state, truth, spacing, rng, args.wrong_approval, args.wrong_mode
            )
        actions.append(action)
        disable_revocation = (
            getattr(args, "disable_revocation", False) or args.method == "no_revocation"
        )
        verdict, support = state.act(action, spacing, args.method, disable_revocation)
        update = {"verdict": verdict}
        if verdict == "conflict" and not disable_revocation and replay is None:
            pending = [
                {"type": "revoke", "coordinates": np.argwhere(support).tolist()},
                action,
            ] + pending
        if verdict == "correct":
            update["timing"] = infer(
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
                cfg.get("fixed_gate", 0.5),
            )
            update.update(
                transition(
                    old, state.probability, truth, state.labels, support, spacing
                )
            )
            rework += rework_update(repaired, old, state.probability, truth, support)
        snapshots.append(snapshot(state, truth, spacing, rework, update))
    result = {
        "id": row["id"],
        "dataset": row["dataset"],
        "patient_id": row["patient_id"],
        "reader": reader,
        "initial_dice": initial_dice,
        "budget": cfg["budget"],
        "charged_actions": state.actions,
        "rework": rework,
        "reviewed_area_fraction": state.reviewed_area / truth.size,
        "actions": actions,
        "snapshots": snapshots,
        "purpose": row.get("purpose", "dataset_evaluation"),
    }
    result["actions90"] = episode_statistics(result)["actions90"]
    return result, image, truth, state.probability


def inference_configuration(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != args.method:
        raise ValueError("--method must match the trained correction checkpoint")
    return load(args.config) if args.config else checkpoint["config"]


def run(args):
    """Run the charged offline policy, optionally replaying fixed neutral actions."""
    cfg = inference_configuration(args)
    seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["seed"] != args.seed:
        raise ValueError("--seed must match the correction checkpoint")
    initial, editor = load_models(args.initial, args.checkpoint, args.device)
    replay = None
    if args.replay:
        archived = load(args.replay)
        if (
            archived["config"]["size"] != cfg["size"]
            or archived["config"]["budget"] != cfg["budget"]
        ):
            raise ValueError("Replay must use the same spatial grid and charged budget")
        replay = {
            r["id"] + "_" + str(r["reader"]): r["actions"] for r in archived["episodes"]
        }
    episodes = []
    output = Path(args.output)
    tasks = evaluation_tasks(
        records(args.manifest, args.split), args.protocol, args.policy_seed
    )
    for row, reader in tasks:
        key = row["id"] + "_" + str(reader)
        if replay is not None and key not in replay:
            raise ValueError(f"Missing task in fixed replay: {key}")
        result, image, truth, prediction = episode(
            row,
            reader,
            initial,
            editor,
            cfg,
            args,
            replay[key] if replay is not None else None,
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
                (truth, prediction >= 0.5, (prediction >= 0.5) != truth.astype(bool)),
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
    parameters = (
        editor.candidate.parameters()
        if args.method in SPATIAL_METHODS + ("fixed_gate",)
        else editor.parameters()
    )
    summary["trainable_parameters"] = sum(p.numel() for p in parameters)
    curves = []
    for budget in range(1, cfg["budget"] + 1):
        selected = [
            dict(
                e,
                budget=budget,
                snapshots=[r for r in e["snapshots"] if r["action"] <= budget],
            )
            for e in episodes
        ]
        curves.append(dict(summarize(selected, args.method), budget=budget))
    save(
        output / "results.json",
        {
            "purpose": episodes[0]["purpose"],
            "summary": [summary],
            "curves": curves,
            "episodes": episodes,
            "config": cfg,
            "method": args.method,
            "seed": args.seed,
            "policy_seed": args.policy_seed,
            "condition": {
                "wrong_approval": args.wrong_approval,
                "wrong_mode": args.wrong_mode,
                "stratum": args.stratum,
                "replay": bool(args.replay),
                "protocol": args.protocol,
                "disable_revocation": args.disable_revocation
                or args.method == "no_revocation",
            },
        },
    )
    save(output / "curves.json", {"summary": curves, "purpose": episodes[0]["purpose"]})


def evaluate(args):
    """Compute per-seed pooled rates, seed mean/SD, patient costs, and paired intervals."""
    runs = [load(path) for path in args.inputs]
    rows, groups, grouped_episodes = [], {}, {}
    for run in runs:
        for dataset in sorted({e["dataset"] for e in run["episodes"]}):
            episodes = [e for e in run["episodes"] if e["dataset"] == dataset]
            row = summarize(episodes, run["method"])
            row.update(
                seed=run["seed"], dataset=dataset, condition=run.get("condition", {})
            )
            for key in (
                "peak_process_memory_mib",
                "peak_gpu_memory_mib",
                "trainable_parameters",
            ):
                row[key] = run["summary"][0].get(key)
            rows.append(row)
            key = dataset, run["method"], json.dumps(row["condition"], sort_keys=True)
            if any(previous["seed"] == row["seed"] for previous in groups.get(key, [])):
                raise ValueError("Duplicate method/dataset/condition/seed input")
            groups.setdefault(key, []).append(row)
            grouped_episodes.setdefault(key, []).extend(episodes)
    pooled_curves = []
    for key, episodes in grouped_episodes.items():
        if len({episode["budget"] for episode in episodes}) != 1:
            raise ValueError("Cannot pool runs with different charged budgets")
        for budget in range(1, episodes[0]["budget"] + 1):
            totals = {
                name: sum(
                    [row for row in episode["snapshots"] if row["action"] <= budget][
                        -1
                    ][name]
                    for episode in episodes
                )
                for name in ("tp", "fp", "fn")
            }
            dice, iou = overlap(totals)
            pooled_curves.append(
                dict(
                    method=key[1],
                    dataset=key[0],
                    condition=groups[key][0]["condition"],
                    budget=budget,
                    dice=dice,
                    iou=iou,
                    **totals,
                )
            )
    result = {
        "summary": rows,
        "curves": pooled_curves,
        "pooled_summary": [
            dict(
                summarize(episodes, key[1]),
                dataset=key[0],
                condition=groups[key][0]["condition"],
            )
            for key, episodes in grouped_episodes.items()
        ],
        "purpose": "functional_check"
        if all(run.get("purpose") == "functional_check" for run in runs)
        else "dataset_evaluation",
        "seed_summary": [
            dict(
                dataset=key[0],
                method=key[1],
                condition=values[0]["condition"],
                metrics=seed_statistics(values),
            )
            for key, values in groups.items()
        ],
        "budget_summaries": [
            {
                "method": run["method"],
                "seed": run["seed"],
                **{
                    f"mean_dice_1_{end}": budget_integral(
                        [
                            row
                            for row in run["curves"]
                            if row["budget"] in (1, 3, 5, 7, 9, 11, 12)
                        ],
                        end,
                    )
                    for end in (5, 12)
                    if run["config"]["budget"] >= end
                },
                "success_only_actions90": float(
                    np.mean(
                        [
                            e["actions90"]
                            for e in run["episodes"]
                            if e["actions90"] <= e["budget"]
                        ]
                    )
                )
                if any(e["actions90"] <= e["budget"] for e in run["episodes"])
                else None,
            }
            for run in runs
        ],
    }
    if args.paired:
        conditions = {
            str(run.get("condition", {}))
            for run in runs
            if run["method"] in args.paired
        }
        if len(conditions) > 1:
            raise ValueError("Paired comparison requires a common evaluation condition")
        if (
            len({run["policy_seed"] for run in runs if run["method"] in args.paired})
            != 1
        ):
            raise ValueError("Paired methods must share the feedback-policy seed")
        result["paired"] = paired_intervals(
            runs, args.paired, args.bootstrap, args.bootstrap_seed
        )
    save(args.output, result)


def input_state(args, initial, cfg):
    """Load a prepared image and explicit localization/state, without a mask reference."""
    with np.load(args.input, allow_pickle=False) as obj:
        image, spacing = obj["image"].astype(np.float32), obj["spacing"].astype(float)
        if image.shape != (cfg["size"], cfg["size"]) or not np.isfinite(image).all():
            raise ValueError("Input image must be finite and match the checkpoint grid")
        if (
            spacing.shape != (2,)
            or not np.isfinite(spacing).all()
            or (spacing <= 0).any()
        ):
            raise ValueError(
                "Input spacing must have two positive finite millimetre values"
            )
        with torch.inference_mode():
            features = initial.encoder(tensor(image, args.device))
            if "probability" in obj:
                probability = obj["probability"].astype(np.float32)
            else:
                localization = obj["localization"].astype(np.float32)
                if (
                    localization.shape != image.shape
                    or not np.isin(localization, [0, 1]).all()
                ):
                    raise ValueError(
                        "Localization must be a binary map on the image grid"
                    )
                probability = (
                    initial.decoder(features, tensor(localization, args.device))
                    .sigmoid()[0, 0]
                    .cpu()
                    .numpy()
                )
        if (
            probability.shape != image.shape
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError(
                "Initial probabilities must be on the image grid and in [0,1]"
            )
        localization = (
            obj["localization"].astype(np.float32)
            if "localization" in obj
            else np.zeros_like(image)
        )
        state = State.initial(probability, localization)
        for key in ("positive", "negative", "labels", "revoked"):
            if key in obj:
                value = obj[key]
                allowed = [-1, 0, 1] if key == "labels" else [0, 1]
                if value.shape != image.shape or not np.isin(value, allowed).all():
                    raise ValueError(f"Invalid saved state channel: {key}")
                setattr(state, key, value.copy())
        state.actions = int(obj["actions"]) if "actions" in obj else 1
        state.reviewed_area = int(obj["reviewed_area"]) if "reviewed_area" in obj else 0
    return image, spacing, state, features


def predict(args):
    """Apply user-supplied actions with no oracle, preserving conflict/revocation state."""
    cfg = inference_configuration(args)
    initial, editor = load_models(args.initial, args.checkpoint, args.device)
    image, spacing, state, features = input_state(args, initial, cfg)
    before = state.probability >= 0.5
    trace = []
    for action in load(args.actions):
        if state.actions >= cfg["budget"]:
            raise ValueError(
                "Interaction actions exceed the checkpoint's charged budget"
            )
        old = state.probability >= 0.5
        verdict, support = state.act(action, spacing, args.method)
        if verdict == "correct":
            infer(
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
                cfg.get("fixed_gate", 0.5),
            )
        overlay = (
            support
            if verdict in ("conflict", "revoke", "revocation_disabled", "area_cap")
            else old != (state.probability >= 0.5)
        )
        trace.append(
            {
                "action": state.actions,
                "verdict": verdict,
                "input": action,
                "overlay_coordinates": np.argwhere(overlay).tolist(),
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "final.npz",
        image=image,
        spacing=spacing,
        probability=state.probability,
        mask=state.probability >= 0.5,
        change=before != (state.probability >= 0.5),
        positive=state.positive,
        negative=state.negative,
        labels=state.labels,
        revoked=state.revoked,
        actions=state.actions,
        reviewed_area=state.reviewed_area,
    )
    save(output / "trace.json", trace)


def benchmark(args):
    """Measure repeated cached-feature correction of one fixed supplied state."""
    cfg = inference_configuration(args)
    initial, editor = load_models(args.initial, args.checkpoint, args.device)
    image, spacing, state, features = input_state(args, initial, cfg)
    support = (state.positive + state.negative) > 0
    if not support.any():
        raise ValueError("Benchmark state needs a correction/localization prompt")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("Need nonnegative warm-up and positive repetitions")
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    timings = []
    for index in range(args.warmup + args.repeats):
        timing = infer(
            initial,
            editor,
            image,
            copy.deepcopy(state),
            spacing,
            args.method,
            args.device,
            cfg["roi_mm"],
            features,
            support,
            cfg["band_mm"],
            cfg.get("fixed_gate", 0.5),
        )
        if index >= args.warmup:
            timings.append(timing)
    parameters = (
        editor.candidate.parameters()
        if args.method in SPATIAL_METHODS + ("fixed_gate",)
        else editor.parameters()
    )
    save(
        args.output,
        {
            "method": args.method,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "device": str(args.device),
            "torch": torch.__version__,
            "hardware": torch.cuda.get_device_name(args.device)
            if str(args.device).startswith("cuda")
            else "CPU",
            "timing_ms": {
                key: float(np.mean([row[key] for row in timings])) for key in timings[0]
            },
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            **peak_memory_mib(),
        },
    )


def argument_parser():
    """Build the command parser shared by the console entry and functional checks."""
    parser = argparse.ArgumentParser(prog="confirmseg")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", choices=["acdc", "lidc", "manifest"], required=True)
    prep.add_argument("--input")
    prep.add_argument("--output", required=True)
    prep.add_argument("--cohort", type=int, default=500)
    prep.add_argument("--split-seed", type=int, default=2027)
    training = commands.add_parser("train")
    training.add_argument("--stage", choices=["initial", "correction"], required=True)
    training.add_argument("--initial")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--method", choices=METHODS, default="confirmseg")
    training.add_argument("--config", default="configs/default.json")
    training.add_argument("--seed", type=int, default=17)
    execution = commands.add_parser(
        "run", help="offline simulated feedback and evaluation"
    )
    execution.add_argument("--manifest", required=True)
    execution.add_argument("--split", choices=["validation", "test"], default="test")
    execution.add_argument("--policy-seed", type=int, default=31415)
    execution.add_argument("--wrong-approval", type=float, default=0)
    execution.add_argument("--wrong-mode", choices=["event", "pixel"], default="event")
    execution.add_argument("--disable-revocation", action="store_true")
    execution.add_argument("--replay")
    execution.add_argument("--stratum", choices=["low", "middle", "high"])
    execution.add_argument("--protocol", choices=["patient", "all"], default="patient")
    execution.add_argument("--overlays", type=int, default=3)
    execution.add_argument("--seed", type=int, default=17)
    prediction = commands.add_parser(
        "predict", help="explicit user actions, no reference labels"
    )
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--actions", required=True)
    timing = commands.add_parser("benchmark", help="cached-feature correction timings")
    timing.add_argument("--input", required=True)
    timing.add_argument("--warmup", type=int, default=20)
    timing.add_argument("--repeats", type=int, default=200)
    for item in (execution, prediction, timing):
        item.add_argument("--initial", required=True)
        item.add_argument("--checkpoint", required=True)
        item.add_argument("--method", choices=METHODS, default="confirmseg")
        item.add_argument("--config", help="override checkpoint configuration")
        item.add_argument("--output", required=True)
    for item in (training, execution, prediction, timing):
        item.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--inputs", nargs="+", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--paired", nargs=2, choices=METHODS)
    evaluation.add_argument("--bootstrap", type=int, default=10000)
    evaluation.add_argument("--bootstrap-seed", type=int, default=271828)
    plot = commands.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.source == "lidc":
            prepare_lidc(args.output, seed=args.split_seed, cohort=args.cohort)
        elif not args.input:
            parser.error("--input is required for ACDC and curated manifests")
        elif args.source == "acdc":
            prepare_acdc(args.input, args.output, args.split_seed)
        else:
            prepare_manifest(args.input, args.output, args.split_seed)
    elif args.command == "run":
        if not 0 <= args.wrong_approval <= 1:
            parser.error("--wrong-approval must be in [0,1]")
        if args.replay and args.wrong_approval:
            parser.error(
                "Wrong approvals must be recorded in the source replay, not reinjected"
            )
        run(args)
    elif args.command == "plot":
        plot_results(args.input, args.output)
    else:
        {
            "train": train,
            "evaluate": evaluate,
            "predict": predict,
            "benchmark": benchmark,
        }[args.command](args)


if __name__ == "__main__":
    main()
