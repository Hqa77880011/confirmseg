"""Small CPU review-plan/replay check using synthetic, short-fit checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .cli import main
from .data import fixed_field, records
from .utils import load, save


def verify(fixture, output):
    """Exercise shared wrong approvals, trained editors, replay and hand-counted metrics."""
    torch.set_num_threads(1)
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    manifest, config = fixture / "data/manifest.json", fixture / "config.json"
    initial = fixture / "initial/best.pt"
    # Reuse the short initial fit; PM receives its own actual one-epoch correction fit.
    pm = fixture / "progressive_merge/best.pt"
    if not pm.exists():
        main(["train", "--stage", "correction", "--manifest", str(manifest),
              "--config", str(config), "--initial", str(initial), "--method", "progressive_merge",
              "--output", str(pm.parent), "--device", "cpu"])
    plan = output / "review_plan.json"
    main(["review-plan", "--manifest", str(manifest), "--config", str(config),
          "--wrong-approval", "1", "--wrong-mode", "subpatch", "--wrong-subpatch-max-pixels", "2",
          "--output", str(plan)])
    frozen = load(plan)
    common = ["run", "--manifest", str(manifest), "--config", str(config),
              "--initial", str(initial), "--overlays", "0", "--device", "cpu"]
    source = output / "neutral/results.json"
    main(common + ["--method", "hard_lock", "--checkpoint", str(fixture / "hard_lock/best.pt"),
                   "--review-plan", str(plan), "--output", str(source.parent)])
    for method in ("confirmseg", "progressive_merge"):
        main(common + ["--method", method, "--checkpoint", str(fixture / method / "best.pt"),
                       "--replay", str(source), "--output", str(output / method)])
    main(common + ["--method", "progressive_merge", "--checkpoint", str(pm),
                   "--review-plan", str(plan), "--disable-revocation",
                   "--output", str(output / "pm_no_revocation")])
    neutral = load(source)
    annotation = {row["id"]: row for row in records(manifest, "test")}
    wrong_pixels = reviewed_pixels = accepted = events = 0
    for episode in neutral["episodes"]:
        _, masks, _ = fixed_field(annotation[episode["id"]], initial=episode["reader"],
                                   size=neutral["config"]["size"])
        truth = masks[episode["reader"]]
        seen = set()
        for action, snapshot in zip(episode["actions"], episode["snapshots"][1:]):
            if action["type"] != "confirm":
                continue
            coordinates = np.asarray(action["coordinates"])
            wrong = np.asarray(action["labels"]) != truth[tuple(coordinates.T)]
            assert 0 < wrong.sum() <= 2 < len(coordinates)
            index = action["review"]["event_index"]
            expected = frozen["tasks"][episode["id"] + "_" + str(episode["reader"])][index]
            assert action == expected
            seen.add(index)
            if snapshot["verdict"] == "confirm":
                accepted += 1
                wrong_pixels += int(wrong.sum())
                reviewed_pixels += len(coordinates)
        events += len(seen)
        assert episode["charged_actions"] == neutral["config"]["budget"]
        assert episode["reviewed_area_fraction"] <= .12
    summary = neutral["summary"][0]
    assert summary["accepted_wrong_review_pixels"] == wrong_pixels > 0
    assert summary["accepted_review_pixels"] == reviewed_pixels
    assert summary["accepted_review_events"] == accepted
    assert summary["wrong_review_events"] == summary["unique_review_events"] == events
    assert summary["wrong_event_percent"] == 100
    assert abs(summary["accepted_wrong_pixel_percent"] - 100*wrong_pixels/reviewed_pixels) < 1e-12
    for method in ("confirmseg", "progressive_merge"):
        replay = load(output / method / "results.json")
        assert [e["actions"] for e in replay["episodes"]] == [e["actions"] for e in neutral["episodes"]]
        assert replay["condition"]["wrong_approval"] == 1
        assert replay["condition"]["wrong_mode"] == "subpatch"
        assert replay["condition"]["review_protocol"] == "homogeneous_connected_bfs_v1"
    comparison = output / "comparison.json"
    main(["evaluate", "--inputs", str(output / "confirmseg/results.json"),
          str(output / "progressive_merge/results.json"), "--paired", "confirmseg", "progressive_merge",
          "--bootstrap", "100", "--output", str(comparison)])
    result = dict(purpose="functional_check", device="cpu", cpu_threads=1,
                  historical_experiment=False, episodes=len(neutral["episodes"]),
                  charged_budget=neutral["config"]["budget"], unique_wrong_review_events=events,
                  accepted_review_events=accepted, accepted_wrong_pixels=wrong_pixels,
                  accepted_review_pixels=reviewed_pixels, wrong_event_percent=100,
                  accepted_wrong_pixel_percent=100*wrong_pixels/reviewed_pixels,
                  identical_realized_actions=True, replay_condition_preserved=True,
                  paired_bootstrap_resamples=100)
    save(output / "verification.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="output of confirmseg.verify")
    parser.add_argument("--output", default="runs/review-smoke")
    args = parser.parse_args()
    verify(args.fixture, args.output)
