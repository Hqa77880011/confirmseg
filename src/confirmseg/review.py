"""Explicit offline review interventions; no corruption flags enter inference."""

from collections import deque
import zlib

import numpy as np
from scipy import ndimage as ndi

MODES = ("event", "pixel", "subpatch")
SUBPATCH_PROTOCOL = "homogeneous_connected_bfs_v1"


def corrupt_review(coordinates, truth, rng, rate=0., mode="subpatch", fraction=.25, max_pixels=0):
    """Sample one event, then flip one bounded same-class connected sub-patch.

    BFS shape, 8-connectivity, uniform reviewed-pixel anchor and the fraction are
    current implementation choices, not a reconstruction of the historical table.
    """
    if mode not in MODES or not 0 <= rate <= 1:
        raise ValueError("Use event/pixel/subpatch mode and a rate in [0,1]")
    if not 0 < fraction <= 1 or max_pixels < 0:
        raise ValueError("Sub-patch fraction must be in (0,1], max_pixels nonnegative")
    coordinates = np.asarray(coordinates, dtype=int).reshape(-1, 2)
    clean = truth[coordinates[:, 0], coordinates[:, 1]].astype(int)
    flip = np.zeros(len(clean), dtype=bool)
    anchor, source_label = None, None
    selected = rng.random(len(clean) if mode == "pixel" else 1) < rate
    if mode == "pixel":
        flip = np.asarray(selected, dtype=bool)
    elif mode == "event":
        flip[:] = bool(selected[0])
    elif selected[0]:
        anchor = coordinates[int(rng.integers(0, len(coordinates)))]
        source_label = int(truth[tuple(anchor)])
        allowed = set(map(tuple, coordinates[clean == source_label]))
        target = max(1, int(np.ceil(fraction * len(clean))))
        if max_pixels:
            target = min(target, max_pixels)
        queue, visited, chosen = deque([tuple(anchor)]), {tuple(anchor)}, set()
        while queue and len(chosen) < target:
            point = queue.popleft()
            chosen.add(point)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (point[0] + dr, point[1] + dc)
                    if neighbor in allowed and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
        flip = np.array([tuple(point) in chosen for point in coordinates])
    labels = np.where(flip, 1 - clean, clean)
    return {"type": "confirm", "coordinates": coordinates.tolist(), "labels": labels.tolist(),
            "review": {"mode": mode, "event_selected": bool(np.any(selected)),
                       "review_pixels": len(clean), "flipped_pixels": int(flip.sum()),
                       "flipped_coordinates": coordinates[flip].tolist(),
                       "anchor": anchor.tolist() if anchor is not None else None,
                       "source_label": source_label,
                       "subpatch_fraction": fraction if mode == "subpatch" else None,
                       "subpatch_max_pixels": max_pixels if mode == "subpatch" else None,
                       "protocol": SUBPATCH_PROTOCOL if mode == "subpatch" else mode}}


def make_plan(truth, task_key, seed, count, rate, mode="subpatch", fraction=.25, max_pixels=0):
    """Freeze review geometry/errors independently of the model and correction RNG."""
    shape = np.asarray(truth.shape)
    edge = max(1, int(np.sqrt(.009 * truth.size)))
    identity = zlib.crc32(task_key.encode("utf-8"))
    actions = []
    for index in range(count):
        geometry = np.random.default_rng(np.random.SeedSequence([seed, identity, index, 0]))
        corruption = np.random.default_rng(np.random.SeedSequence([seed, identity, index, 1]))
        origin = geometry.integers(0, np.maximum(1, shape - edge + 1))
        coordinates = np.argwhere(np.ones((edge, edge), dtype=bool)) + origin
        action = corrupt_review(coordinates, truth, corruption, rate, mode, fraction, max_pixels)
        action["review"]["event_index"] = index
        action["review"]["task_key"] = task_key
        actions.append(action)
    return actions


def validate_plan(actions, truth, condition):
    """Check externally loaded review labels against their declared intervention."""
    for action in actions:
        coordinates = np.asarray(action["coordinates"])
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates)
                or not np.issubdtype(coordinates.dtype, np.integer)
                or np.any(coordinates < 0) or np.any(coordinates >= np.asarray(truth.shape))
                or len(np.unique(coordinates, axis=0)) != len(coordinates)):
            raise ValueError("Review plan has invalid coordinates")
        labels = np.asarray(action["labels"])
        if action["type"] != "confirm" or labels.shape != (len(coordinates),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Review plan needs one binary label per inspected coordinate")
        clean = truth[coordinates[:, 0], coordinates[:, 1]]
        wrong = labels != clean
        metadata = action["review"]
        if metadata["flipped_pixels"] != int(wrong.sum()):
            raise ValueError("Review metadata disagrees with supplied labels")
        if condition["wrong_mode"] == "subpatch" and wrong.any():
            if len(np.unique(clean[wrong])) != 1:
                raise ValueError("Wrong sub-patch must have one original class")
            mask = np.zeros(truth.shape, bool)
            mask[tuple(coordinates[wrong].T)] = True
            if ndi.label(mask, np.ones((3, 3)))[1] != 1:
                raise ValueError("Wrong sub-patch must be 8-connected")
            limit = max(1, int(np.ceil(condition["wrong_subpatch_fraction"] * len(coordinates))))
            maximum = condition["wrong_subpatch_max_pixels"]
            if maximum:
                limit = min(limit, maximum)
            if wrong.sum() > limit:
                raise ValueError("Wrong sub-patch exceeds its declared area limit")


def statistics(episodes):
    """Report event sampling and accepted wrong-pixel exposure with denominators."""
    attempts, accepted, wrong_pixels, reviewed_pixels = 0, 0, 0, 0
    unique, wrong_events = set(), set()
    for episode_index, episode in enumerate(episodes):
        for index, row in enumerate(episode["snapshots"]):
            if "review_event" not in row:
                continue
            event = row["review_event"]
            key = (episode_index, event.get("event_index", index))
            unique.add(key)
            attempts += 1
            if event["wrong_pixels"]:
                wrong_events.add(key)
            if row["verdict"] == "confirm":
                accepted += 1
                wrong_pixels += event["wrong_pixels"]
                reviewed_pixels += event["review_pixels"]
    return {"review_attempts": attempts, "unique_review_events": len(unique),
            "wrong_review_events": len(wrong_events), "accepted_review_events": accepted,
            "accepted_wrong_review_pixels": wrong_pixels, "accepted_review_pixels": reviewed_pixels,
            "wrong_event_percent": 100 * len(wrong_events) / len(unique) if unique else None,
            "accepted_wrong_pixel_percent": 100 * wrong_pixels / reviewed_pixels if reviewed_pixels else None}
