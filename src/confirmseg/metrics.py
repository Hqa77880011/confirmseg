import numpy as np
from scipy import ndimage as ndi


BANDS = (("0_3mm", 0, 3), ("3_6mm", 3, 6), ("over_6mm", 6, np.inf))


def transition(old, updated, truth, labels, support, spacing):
    """Record pooled numerators/denominators for one successfully decoded correction."""
    old_error, error = (
        (old >= 0.5) != truth.astype(bool),
        (updated >= 0.5) != truth.astype(bool),
    )
    components, _ = ndi.label(old_error, np.ones((3, 3)))
    ids = np.unique(components[support])
    selected = np.isin(components, ids[ids > 0])
    editable = labels < 0
    correct = ~old_error & editable
    regression = correct & error
    approved = ~editable
    bands = {name: [0, 0] for name, _, _ in BANDS}
    if approved.any():
        edge = approved ^ ndi.binary_erosion(approved)
        distance = ndi.distance_transform_edt(~edge, sampling=spacing)
        for name, low, high in BANDS:
            subset = correct & (distance >= low) & (distance < high)
            bands[name] = [int((regression & subset).sum()), int(subset.sum())]
    return {
        "regression_numerator": int(regression.sum()),
        "prior_correct_unconfirmed": int(correct.sum()),
        "band_regression": bands,
        "corrected_target_pixels": int((selected & ~error).sum()),
        "target_error_pixels": int(selected.sum()),
        "changed_pixels": int((((old >= 0.5) != (updated >= 0.5)) & editable).sum()),
        "editable_pixels": int(editable.sum()),
        "beneficial_remote_pixels": int((old_error & ~error & ~selected).sum()),
        "harmful_remote_pixels": int((~old_error & error & ~selected).sum()),
    }


def rework_update(repaired, old, updated, truth, support):
    """Match individual 8-connected targets at IoU > .5; count once per action."""
    old_error, error = (
        (old >= 0.5) != truth.astype(bool),
        (updated >= 0.5) != truth.astype(bool),
    )
    components, _ = ndi.label(old_error, np.ones((3, 3)))
    ids = np.unique(components[support])
    selected = [components == index for index in ids if index > 0]
    repeated = any(
        (prior & current).sum() / max((prior | current).sum(), 1) > 0.5
        for current in selected
        for prior in repaired
    )
    repaired.extend(region for region in selected if not (region & error).any())
    return int(repeated)


def episode_statistics(episode):
    """First-hit review area is measured at the hit, or at the final budget on failure."""
    snapshots = episode["snapshots"]
    hit = next((row for row in snapshots if overlap(row)[0] >= 0.9), None)
    area = (hit or snapshots[-1])["reviewed_area_fraction"]
    actions = hit["action"] if hit else episode["budget"] + 1
    return {
        "dice": overlap(snapshots[-1])[0],
        "iou": overlap(snapshots[-1])[1],
        "repeated_corrections": snapshots[-1]["rework"],
        "actions90": actions,
        "reviewed_area_at_hit_fraction": area,
        **{f"cost90_alpha_{alpha}": actions + alpha * area for alpha in (0, 10, 50)},
    }


def patient_macro(episodes):
    """Average episodes within each patient before averaging patients."""
    return {
        patient: {
            key: float(np.mean([row[key] for row in values])) for key in values[0]
        }
        for patient in sorted({episode["patient_id"] for episode in episodes})
        for values in [
            [
                episode_statistics(episode)
                for episode in episodes
                if episode["patient_id"] == patient
            ]
        ]
    }


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def counts(prediction, target):
    p, y = prediction >= 0.5, target.astype(bool)
    return {
        "tp": int((p & y).sum()),
        "fp": int((p & ~y).sum()),
        "fn": int((~p & y).sum()),
    }


def overlap(values):
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    return (
        (2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1))
        if tp + fp + fn
        else (1.0, 1.0)
    )


def assd(prediction, target, spacing):
    p, y = prediction >= 0.5, target.astype(bool)
    if not p.any() and not y.any():
        return 0.0, False
    if not p.any() or not y.any():
        return float(np.linalg.norm(np.asarray(y.shape) * spacing)), True
    a, b = p ^ ndi.binary_erosion(p), y ^ ndi.binary_erosion(y)
    da = ndi.distance_transform_edt(~a, sampling=spacing)[b]
    db = ndi.distance_transform_edt(~b, sampling=spacing)[a]
    return float((da.sum() + db.sum()) / (len(da) + len(db))), False


def summarize(episodes, method):
    """Pool pixel counts; retain separate patient-macro and first-hit summaries."""
    if not episodes:
        raise ValueError("Cannot summarize zero episodes")
    final = [e["snapshots"][-1] for e in episodes]
    totals = {key: sum(r[key] for r in final) for key in ("tp", "fp", "fn")}
    dice, iou = overlap(totals)
    snapshots = [r for e in episodes for r in e["snapshots"]]
    stages = [r.get("timing", {}) for r in snapshots]
    episode_values = [episode_statistics(e) for e in episodes]
    patients = patient_macro(episodes)
    result = {
        "method": method,
        "dice": dice,
        "iou": iou,
        "actions90": float(np.mean([e["actions90"] for e in episode_values])),
        "failures90": sum(
            e["actions90"] > episode["budget"]
            for e, episode in zip(episode_values, episodes)
        ),
        "repeated_corrections": float(
            np.mean([e["repeated_corrections"] for e in episode_values])
        ),
        "regression_percent": percent(
            sum(r["regression_numerator"] for r in snapshots),
            sum(r["prior_correct_unconfirmed"] for r in snapshots),
        ),
        "approval_violation_percent": percent(
            sum(r["violation_numerator"] for r in snapshots),
            sum(r["confirmation_denominator"] for r in snapshots),
        ),
        "assd_mm": float(np.mean([r["assd_mm"] for r in final])),
        "empty_mask_failures": sum(r["empty_failure"] for r in final),
        "reviewed_area_fraction": float(
            np.mean([e["snapshots"][-1]["reviewed_area_fraction"] for e in episodes])
        ),
        "patient_macro": {
            key: float(np.mean([row[key] for row in patients.values()]))
            for key in episode_values[0]
        },
        "first_hit": {
            key: float(np.mean([row[key] for row in episode_values]))
            for key in episode_values[0]
            if key.startswith("cost90_") or key == "reviewed_area_at_hit_fraction"
        },
    }
    result["episodes"] = len(episodes)
    result["patients"] = len({e["patient_id"] for e in episodes})
    for key in (
        "beneficial_remote_pixels",
        "harmful_remote_pixels",
        "corrected_target_pixels",
        "changed_pixels",
        "editable_pixels",
        "target_error_pixels",
    ):
        result[key] = sum(r.get(key, 0) for r in snapshots)
    result["confirmation_evaluations"] = sum(
        r["confirmation_denominator"] for r in snapshots
    )
    result["confirmation_violations"] = sum(r["violation_numerator"] for r in snapshots)
    result["target_repair_percent"] = percent(
        result["corrected_target_pixels"], result["target_error_pixels"]
    )
    result["changed_editable_percent"] = percent(
        result["changed_pixels"], result["editable_pixels"]
    )
    result["conflicts"] = sum(r.get("verdict") == "conflict" for r in snapshots)
    result["revocations"] = sum(r.get("verdict") == "revoke" for r in snapshots)
    for key in (
        "candidate_ms",
        "gate_ms",
        "candidate_and_gate_ms",
        "gate_and_merge_ms",
        "merge_ms",
        "projection_ms",
        "dispatch_ms",
        "total_ms",
    ):
        values = [r[key] for r in stages if key in r]
        result[key] = float(np.mean(values)) if values else None
    result["band_counts"] = {}
    for name, _, _ in BANDS:
        numerator = sum(r["band_regression"][name][0] for r in snapshots)
        denominator = sum(r["band_regression"][name][1] for r in snapshots)
        result["band_counts"][name] = [numerator, denominator]
        result["regression_" + name + "_percent"] = percent(numerator, denominator)
    return result


def seed_statistics(rows):
    """Mean and sample SD of per-seed estimands, never SD inferred from a mean."""
    result = {}
    for key in rows[0]:
        values = [
            row[key]
            for row in rows
            if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
        ]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "seeds": len(values),
            }
    for key in ("patient_macro", "first_hit"):
        if key in rows[0]:
            result[key] = seed_statistics([row[key] for row in rows])
    return result


def paired_intervals(runs, methods, resamples=10000, seed=271828):
    """Pair patient/task/seed records; resample patient clusters jointly over seeds."""
    relevant = [run for run in runs if run["method"] in methods]
    output = {}
    for dataset in sorted({e["dataset"] for run in relevant for e in run["episodes"]}):
        grouped = {}
        for run in relevant:
            episodes = [e for e in run["episodes"] if e["dataset"] == dataset]
            if not episodes:
                continue
            key = run["method"], run["seed"]
            if key in grouped:
                raise ValueError(
                    "Paired comparison needs one run per method/dataset/seed"
                )
            grouped[key] = episodes
        seeds = sorted({key[1] for key in grouped})
        matrices = {key: [] for key in ("dice", "repeated_corrections")}
        patient_ids = None
        for training_seed in seeds:
            if any((method, training_seed) not in grouped for method in methods):
                raise ValueError(
                    "Paired methods must have identical training seed sets"
                )
            a, b = (grouped[method, training_seed] for method in methods)
            identity = lambda episodes: {
                (e["patient_id"], e["id"], e["reader"], e["budget"]) for e in episodes
            }
            if identity(a) != identity(b):
                raise ValueError(
                    "Paired runs must contain the same patient/task/reader/budget records"
                )
            first, second = patient_macro(a), patient_macro(b)
            ids = sorted(first)
            if patient_ids is not None and ids != patient_ids:
                raise ValueError("Patient clusters must be shared across all seeds")
            patient_ids = ids
            for metric in matrices:
                matrices[metric].append(
                    [
                        first[patient][metric] - second[patient][metric]
                        for patient in ids
                    ]
                )
        if resamples < 1:
            raise ValueError("Bootstrap resamples must be positive")
        rng = np.random.default_rng(seed)
        indices = rng.integers(len(patient_ids), size=(resamples, len(patient_ids)))
        result = {
            "methods": list(methods),
            "patients": len(patient_ids),
            "seeds": seeds,
            "bootstrap_seed": seed,
            "resamples": resamples,
        }
        for metric, values in matrices.items():
            matrix = np.asarray(values) * (100 if metric == "dice" else 1)
            per_seed = matrix.mean(1)
            samples = matrix.mean(0)[indices].mean(1)
            result[metric + ("_pp" if metric == "dice" else "")] = {
                "difference": float(matrix.mean()),
                "seed_sd": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
                "patient_interval95": np.quantile(samples, [0.025, 0.975]).tolist(),
            }
        output[dataset] = result
    if not output:
        raise ValueError("No paired datasets found")
    return output


def budget_integral(curves, end):
    """Trapezoid summary over recorded knots, normalized by end minus one."""
    selected = sorted(
        (row for row in curves if row["budget"] <= end), key=lambda row: row["budget"]
    )
    if (
        not selected
        or selected[0]["budget"] != 1
        or selected[-1]["budget"] != end
        or end <= 1
    ):
        raise ValueError("Budget integration needs recorded endpoints 1 and end > 1")
    x, y = (
        np.asarray([row["budget"] for row in selected]),
        np.asarray([row["dice"] for row in selected]),
    )
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (end - 1))
