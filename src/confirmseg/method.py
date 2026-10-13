import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F

from .network import Decoder, Encoder, segmentation_loss


METHODS = (
    "confirmseg",
    "global",
    "hard_lock",
    "fixed_roi",
    "progressive_merge",
    "entropy_roi",
    "dmp",
    "no_approval",
    "no_area",
    "no_regression",
    "no_boundary",
    "no_revocation",
)


def physical_dilation(mask, radius, spacing):
    if not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= radius


def band_map(approved, spacing, radius=3):
    edge = approved ^ ndi.binary_erosion(approved)
    return physical_dilation(edge, radius, spacing).astype(np.float32)


@dataclass
class State:
    probability: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    labels: np.ndarray
    revoked: np.ndarray
    actions: int = 1
    reviewed_area: int = 0

    @classmethod
    def initial(cls, probability, localization=None):
        return cls(
            probability.copy(),
            np.zeros_like(probability) if localization is None else localization.copy(),
            np.zeros_like(probability),
            np.full(probability.shape, -1, dtype=np.int8),
            np.zeros_like(probability),
        )

    def channels(self, spacing, radius=3):
        return np.stack(
            [
                self.probability,
                self.positive,
                self.negative,
                self.labels == 1,
                self.labels == 0,
                self.revoked,
                band_map(self.labels >= 0, spacing, radius),
            ]
        ).astype(np.float32)

    def act(self, action, spacing, method="confirmseg"):
        self.actions += 1
        mask = np.zeros_like(self.labels, dtype=bool)
        coordinates = np.asarray(action["coordinates"], dtype=int).reshape(-1, 2)
        if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(mask.shape)):
            raise ValueError("Action coordinate outside image")
        mask[coordinates[:, 0], coordinates[:, 1]] = True
        if action["type"] == "revoke":
            if method == "no_revocation":
                return "revocation_disabled", mask
            self.labels[mask] = -1
            self.revoked[mask] = 1
            return "revoke", mask
        labels = np.asarray(action["labels"], dtype=np.int8)
        if labels.shape != (len(coordinates),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Each action coordinate needs a binary label")
        supplied = np.full(self.labels.shape, -1, dtype=np.int8)
        supplied[coordinates[:, 0], coordinates[:, 1]] = labels
        conflict = mask & (self.labels >= 0) & (supplied != self.labels)
        if conflict.any():
            return "conflict", conflict
        if action["type"] == "confirm":
            if mask.sum() > int(
                0.01 * mask.size
            ) or self.reviewed_area + mask.sum() > int(0.12 * mask.size):
                return "area_cap", mask
            self.reviewed_area += int(mask.sum())
            self.labels[mask] = supplied[mask]
            if method != "global":
                self.probability[mask] = supplied[mask]
            return "confirm", mask
        if action["type"] != "correct":
            raise ValueError("Unknown action type")
        for label, channel in ((1, self.positive), (0, self.negative)):
            channel[
                physical_dilation(
                    mask & (supplied == label),
                    1 if len(coordinates) > 1 else 2,
                    spacing,
                )
            ] = 1
        return "correct", mask


class InitialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(1)

    def forward(self, image, localization):
        return self.decoder(self.encoder(image), localization).sigmoid()


class Editor(nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate = Decoder(7)
        self.gate = nn.Sequential(
            nn.Conv2d(135, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, features, state, method="confirmseg", candidate=None):
        context = state.clone()
        if method == "no_approval":
            context[:, 3:7] = 0
        q = (
            self.candidate(features, context).sigmoid()
            if candidate is None
            else candidate
        )
        approved = state[:, 3:5].sum(1, keepdim=True).clamp(0, 1)
        if method in (
            "global",
            "hard_lock",
            "fixed_roi",
            "progressive_merge",
            "entropy_roi",
            "dmp",
        ):
            gate = torch.ones_like(q) if method == "global" else 1 - approved
            p = (
                q
                if method == "global"
                else torch.where(approved.bool(), state[:, 3:4], q)
            )
            return p, q, gate
        inputs = torch.cat(
            [
                features[2],
                F.interpolate(
                    context,
                    size=features[2].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ],
            1,
        )
        raw_gate = F.interpolate(
            self.gate(inputs),
            size=state.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        gate = raw_gate * (1 - approved)
        p = state[:, :1] + gate * (q - state[:, :1])
        p = torch.where(approved.bool(), state[:, 3:4], p)
        return p, q, gate


def footprint_target(old, truth, approved, support, spacing, radius):
    components, _ = ndi.label((old >= 0.5) != truth.astype(bool), np.ones((3, 3)))
    ids = np.unique(components[support])
    selected = np.isin(components, ids[ids > 0])
    return physical_dilation(selected, radius, spacing) & ~approved


def loss_terms(p, q, gate, state, truth, target, spacing, weights, method):
    approved = state[:, 3:5].sum(1, keepdim=True) > 0
    editable = ~approved
    old = state[:, :1]
    correct = editable & ((old >= 0.5) == truth.bool())
    base = segmentation_loss(p, truth)
    gate_loss = (
        F.binary_cross_entropy(gate.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        * editable
    ).sum() / editable.sum().clamp_min(1)
    area = gate.sum() / editable.sum().clamp_min(1)
    regression = (gate * (q - old).abs() * correct).sum() / correct.sum().clamp_min(1)
    band = state[:, 6:7] > 0
    boundary = p.new_zeros(())
    count = p.new_zeros(())
    for axis, distance in ((2, float(spacing[0])), (3, float(spacing[1]))):
        a = [slice(None)] * 4
        b = a.copy()
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        selected = band[a] | band[b]
        boundary = (
            boundary
            + (
                ((p[a] - p[b]) - (truth[a] - truth[b])).abs() * selected / distance
            ).sum()
        )
        count = count + selected.sum()
    boundary = boundary / count.clamp_min(1)
    if method == "dmp":
        local = target > 0
        remote = editable & ~local
        local_loss = ((p - truth).abs() * local).sum() / local.sum().clamp_min(1)
        preserve = ((p - old).abs() * remote).sum() / remote.sum().clamp_min(1)
        return base + weights["regression"] * preserve + local_loss
    if method not in (
        "confirmseg",
        "no_approval",
        "no_area",
        "no_regression",
        "no_boundary",
        "no_revocation",
    ):
        return segmentation_loss(q, truth)
    return (
        base
        + weights["gate"] * gate_loss
        + (0 if method == "no_area" else weights["area"]) * area
        + (0 if method == "no_regression" else weights["regression"]) * regression
        + (0 if method == "no_boundary" else weights["boundary"]) * boundary
    )


def infer(
    initial,
    editor,
    image,
    state,
    spacing,
    method,
    device,
    radius=16,
    features=None,
    prompt_support=None,
    band_mm=3,
):
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    total_start = time.perf_counter()
    channels = torch.from_numpy(state.channels(spacing, band_mm))[None].to(device)
    if features is None:
        with torch.inference_mode():
            features = initial.encoder(torch.from_numpy(image)[None, None].to(device))
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        context = channels.clone()
        if method == "no_approval":
            context[:, 3:7] = 0
        q = editor.candidate(features, context).sigmoid()
        sync()
        candidate_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        _, q, gate = editor(features, channels, method, candidate=q)
        sync()
        gate_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    candidate, amplitude = q[0, 0].cpu().numpy(), gate[0, 0].cpu().numpy()
    old = state.probability
    support = (
        prompt_support
        if prompt_support is not None
        else (state.positive + state.negative) > 0
    )
    if method in ("global", "hard_lock", "dmp"):
        amplitude = np.ones_like(old)
    elif method == "fixed_roi":
        amplitude = physical_dilation(support, radius, spacing).astype(float)
    elif method == "progressive_merge":
        difference, _ = ndi.label((candidate >= 0.5) != (old >= 0.5), np.ones((3, 3)))
        touched = np.unique(difference[support])
        amplitude = np.isin(difference, touched[touched > 0]).astype(float)
    elif method == "entropy_roi":
        entropy = -(
            candidate * np.log(candidate + 1e-6)
            + (1 - candidate) * np.log(1 - candidate + 1e-6)
        )
        entropy[state.labels >= 0] = 0
        center = np.unravel_index(
            ndi.gaussian_filter(
                entropy, radius / max(float(np.mean(spacing)), 1e-6)
            ).argmax(),
            old.shape,
        )
        point = np.zeros_like(old, dtype=bool)
        point[center] = True
        amplitude = physical_dilation(point, radius, spacing).astype(float)
    updated = old + amplitude * (candidate - old)
    merge_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    if method != "global":
        approved = state.labels >= 0
        updated[approved] = state.labels[approved]
    projection_ms = (time.perf_counter() - start) * 1000
    state.probability = updated.astype(np.float32)
    state.revoked.fill(0)
    return {
        "candidate_ms": candidate_ms,
        "gate_ms": gate_ms,
        "candidate_and_gate_ms": candidate_ms + gate_ms,
        "gate_and_merge_ms": gate_ms + merge_ms,
        "total_ms": (time.perf_counter() - total_start) * 1000,
        "dispatch_ms": max(
            0.0,
            (time.perf_counter() - total_start) * 1000
            - candidate_ms
            - gate_ms
            - merge_ms
            - projection_ms,
        ),
        "merge_ms": merge_ms,
        "projection_ms": projection_ms,
    }


def simulate_action(state, truth, spacing, rng, wrong_rate=0):
    error = (state.probability >= 0.5) != truth.astype(bool)
    if state.actions % 3 == 0 and state.reviewed_area < 0.12 * error.size:
        edge = max(1, int(np.sqrt(0.009 * error.size)))
        x, y = rng.integers(0, np.maximum(1, np.asarray(error.shape) - edge + 1))
        coordinates = np.argwhere(np.ones((edge, edge), dtype=bool)) + (x, y)
        labels = truth[coordinates[:, 0], coordinates[:, 1]].astype(int)
        flip = rng.random(len(labels)) < wrong_rate
        labels[flip] = 1 - labels[flip]
        return {
            "type": "confirm",
            "coordinates": coordinates.tolist(),
            "labels": labels.tolist(),
        }
    components, number = ndi.label(error, np.ones((3, 3)))
    if number:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        selected = components == sizes.argmax()
        distance = ndi.distance_transform_edt(selected, sampling=spacing)
        point = np.asarray(np.unravel_index(distance.argmax(), error.shape))
    else:
        point = np.asarray(np.unravel_index(state.probability.argmax(), error.shape))
    coordinates = point[None]
    if state.actions % 2 == 0 and number:
        boundary = np.argwhere(selected ^ ndi.binary_erosion(selected))
        anchor = boundary[rng.integers(len(boundary))]
        nearby = np.linalg.norm((boundary - anchor) * spacing, axis=1) <= 4
        coordinates = boundary[nearby]
    return {
        "type": "correct",
        "coordinates": coordinates.tolist(),
        "labels": truth[coordinates[:, 0], coordinates[:, 1]].astype(int).tolist(),
    }
