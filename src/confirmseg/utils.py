import json
import random
from pathlib import Path

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_groups(rows, key, seed=2027):
    groups = sorted({str(r[key]) for r in rows})
    order = np.random.default_rng(seed).permutation(groups)
    first = int(0.6 * len(order))
    second = first + int(0.2 * len(order))
    assigned = {
        g: "train" if i < first else "validation" if i < second else "test"
        for i, g in enumerate(order)
    }
    return [dict(r, split=assigned[str(r[key])]) for r in rows]


def plot_results(path, output, metrics=None):
    """Plot measured summaries, charged-budget curves, spatial rates and paired CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = load(path)
    rows = obj["summary"]
    if not rows:
        raise ValueError("No result rows to plot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = ["#487d96", "#4b8b78", "#6c929a", "#80868b"]

    def finish(fig, name):
        if obj.get("purpose") == "functional_check":
            fig.suptitle("Synthetic functional check")
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=180)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)

    def label(row):
        text = " / ".join(
            value for value in (row.get("dataset", ""), row["method"]) if value
        )
        condition = row.get("condition", {})
        details = []
        if condition.get("wrong_approval"):
            details.append(
                f"wrong={condition['wrong_approval']:.0%} ({condition.get('wrong_mode', 'event')})"
            )
        if condition.get("disable_revocation"):
            details.append("revocation off")
        if condition.get("stratum"):
            details.append(condition["stratum"])
        if condition.get("replay"):
            details.append("replay")
        return text + (" [" + ", ".join(details) + "]" if details else "")

    if obj.get("curves"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_prop_cycle(color=colors)
        curves = obj["curves"]
        for name in sorted({label(row) for row in curves}):
            curve = sorted(
                (row for row in curves if label(row) == name),
                key=lambda row: row["budget"],
            )
            ax.plot(
                [row["budget"] for row in curve],
                [row["dice"] for row in curve],
                "o-",
                label=name,
            )
        ax.set(xlabel="Charged actions", ylabel="Pooled micro-Dice", ylim=(0, 1))
        ax.legend(fontsize=8)
        finish(fig, "pooled_budget_dice")

    if all("budget" in row for row in rows):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_prop_cycle(color=colors)
        for method in sorted({row["method"] for row in rows}):
            curve = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: row["budget"],
            )
            ax.plot(
                [row["budget"] for row in curve],
                [row["dice"] for row in curve],
                "o-",
                label=method,
            )
        ax.set(xlabel="Charged actions", ylabel="Micro-Dice", ylim=(0, 1))
        ax.legend(fontsize=8)
        finish(fig, "budget_dice")
        return

    aggregate = obj.get("seed_summary")
    if aggregate:
        values = []
        for group in aggregate:
            record = dict(
                method=group["method"],
                dataset=group["dataset"],
                condition=group["condition"],
            )
            for key, value in group["metrics"].items():
                if isinstance(value, dict) and "mean" in value:
                    record[key], record[key + "_sd"] = value["mean"], value["sd"]
            for key, value in group["metrics"].get("patient_macro", {}).items():
                record["patient_" + key] = value["mean"]
            values.append(record)
    else:
        values = rows
    metrics = metrics or [
        "dice",
        "iou",
        "actions90",
        "repeated_corrections",
        "regression_percent",
        "target_repair_percent",
        "changed_editable_percent",
        "assd_mm",
        "patient_cost90_alpha_0",
        "patient_cost90_alpha_10",
        "patient_cost90_alpha_50",
    ]
    for metric in metrics:
        valid = [row for row in values if row.get(metric) is not None]
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(valid) + 1)))
        labels = [label(row) for row in valid]
        error = [row.get(metric + "_sd") or 0 for row in valid]
        ax.barh(
            labels,
            [row[metric] for row in valid],
            xerr=error,
            capsize=3,
            color="#487d96",
        )
        ax.set_xlabel(metric.replace("_", " "))
        finish(fig, metric)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_prop_cycle(color=colors)
    available = False
    for row in values:
        keys = [
            "regression_" + band + "_percent" for band in ("0_3mm", "3_6mm", "over_6mm")
        ]
        if all(row.get(key) is not None for key in keys):
            ax.errorbar(
                range(3),
                [row[key] for key in keys],
                yerr=[row.get(key + "_sd") or 0 for key in keys],
                marker="o",
                capsize=3,
                label=label(row),
            )
            available = True
    if available:
        ax.set(
            xticks=range(3),
            xticklabels=["[0,3)", "[3,6)", ">=6"],
            xlabel="Distance to active approval boundary (mm)",
            ylabel="Regression (%)",
        )
        ax.legend(fontsize=8)
        finish(fig, "distance_bands")
    else:
        plt.close(fig)
    for dataset, paired in obj.get("paired", {}).items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, metric, axis_label in zip(
            axes,
            ("dice_pp", "repeated_corrections"),
            ("Dice difference (pp)", "Rework difference"),
        ):
            result = paired[metric]
            low, high = result["patient_interval95"]
            estimate = result["difference"]
            ax.plot([low, high], [0, 0], color="#487d96", linewidth=2)
            ax.plot(estimate, 0, "o", color="#246754")
            ax.axvline(0, color=".6", linestyle="--", linewidth=0.8)
            ax.set(xlabel=axis_label, yticks=[], title=dataset)
        finish(fig, "paired_" + dataset.replace("/", "_"))


def peak_memory_mib():
    import os
    import sys
    import torch

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not api(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        process = counters.peak_working / 1024**2
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process = peak / (1024**2 if sys.platform == "darwin" else 1024)
    gpu = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else None
    )
    return {"peak_process_memory_mib": float(process), "peak_gpu_memory_mib": gpu}
