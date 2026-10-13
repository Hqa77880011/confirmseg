import numpy as np
from scipy import ndimage as ndi


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
    final = [e["snapshots"][-1] for e in episodes]
    totals = {key: sum(r[key] for r in final) for key in ("tp", "fp", "fn")}
    dice, iou = overlap(totals)
    snapshots = [r for e in episodes for r in e["snapshots"]]
    stages = [r.get("timing", {}) for r in snapshots]
    result = {
        "method": method,
        "dice": dice,
        "iou": iou,
        "actions90": float(np.mean([e["actions90"] for e in episodes])),
        "failures90": sum(e["actions90"] > e["budget"] for e in episodes),
        "repeated_corrections": float(np.mean([e["rework"] for e in episodes])),
        "regression_percent": 100
        * sum(r["regression_numerator"] for r in snapshots)
        / max(sum(r["prior_correct_unconfirmed"] for r in snapshots), 1),
        "approval_violation_percent": 100
        * sum(r["violation_numerator"] for r in snapshots)
        / max(sum(r["confirmation_denominator"] for r in snapshots), 1),
        "assd_mm": float(np.mean([r["assd_mm"] for r in final])),
        "empty_mask_failures": sum(r["empty_failure"] for r in final),
        "reviewed_area_fraction": float(
            np.mean([e["reviewed_area_fraction"] for e in episodes])
        ),
        "area_weighted_actions": float(
            np.mean(
                [
                    e["charged_actions"] + 10 * e["reviewed_area_fraction"]
                    for e in episodes
                ]
            )
        ),
    }
    result["episodes"] = len(episodes)
    result["patients"] = len({e["patient_id"] for e in episodes})
    for key in (
        "beneficial_remote_pixels",
        "harmful_remote_pixels",
        "corrected_target_pixels",
        "changed_pixels",
    ):
        result[key] = sum(r.get(key, 0) for r in snapshots)
    result["confirmation_evaluations"] = sum(
        r["confirmation_denominator"] for r in snapshots
    )
    result["confirmation_violations"] = sum(r["violation_numerator"] for r in snapshots)
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
    for name, low, high in [
        ("0_3mm", 0, 3),
        ("3_6mm", 3, 6),
        ("over_6mm", 6, float("inf")),
    ]:
        numerator = sum(r["band_regression"][name][0] for r in snapshots)
        denominator = sum(r["band_regression"][name][1] for r in snapshots)
        result["regression_" + name + "_percent"] = (
            100 * numerator / max(denominator, 1)
        )
    return result
