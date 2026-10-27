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
    "candidate_only",
    "gate_only",
    "fixed_gate",
    "no_area",
    "no_regression",
    "no_boundary",
    "no_revocation",
)

SPATIAL_METHODS = (
    "global",
    "hard_lock",
    "fixed_roi",
    "progressive_merge",
    "entropy_roi",
    "dmp",
)


def head_context(state, method, head):
    """Mask review channels independently, retaining p and both prompt channels."""
    context = state.clone()
    if (
        method == "no_approval"
        or (head == "candidate" and method == "gate_only")
        or (head == "gate" and method == "candidate_only")
    ):
        context[:, 3:7] = 0
    return context


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

    def act(self, action, spacing, method="confirmseg", disable_revocation=False):
        """Charge and apply explicit feedback; conflicts leave review and prompts intact."""
        mask = np.zeros_like(self.labels, dtype=bool)
        supplied_coordinates = np.asarray(action["coordinates"])
        coordinates = supplied_coordinates.astype(int).reshape(-1, 2)
        if not np.array_equal(supplied_coordinates, coordinates):
            raise ValueError("Action coordinates must be integer [row, column] pairs")
        if not len(coordinates) or len(np.unique(coordinates, axis=0)) != len(
            coordinates
        ):
            raise ValueError("Action coordinates must be nonempty and distinct")
        if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(mask.shape)):
            raise ValueError("Action coordinate outside image")
        mask[coordinates[:, 0], coordinates[:, 1]] = True
        if action["type"] == "revoke":
            self.actions += 1
            if method == "no_revocation" or disable_revocation:
                return "revocation_disabled", mask
            self.labels[mask] = -1
            self.revoked[mask] = 1
            return "revoke", mask
        labels = np.asarray(action["labels"])
        if action["type"] not in ("confirm", "correct"):
            raise ValueError("Unknown action type")
        if labels.shape != (len(coordinates),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Each action coordinate needs a binary label")
        labels = labels.astype(np.int8)
        supplied = np.full(self.labels.shape, -1, dtype=np.int8)
        supplied[coordinates[:, 0], coordinates[:, 1]] = labels
        conflict = mask & (self.labels >= 0) & (supplied != self.labels)
        self.actions += 1
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
        for label, channel in ((1, self.positive), (0, self.negative)):
            channel[
                physical_dilation(
                    mask & (supplied == label),
                    1
                    if action.get(
                        "prompt", "stroke" if len(coordinates) > 1 else "click"
                    )
                    == "stroke"
                    else 2,
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

    def forward(
        self,
        features,
        state,
        method="confirmseg",
        candidate=None,
        spacing=(1, 1),
        support=None,
        radius=16,
        fixed_gate=0.5,
        project=True,
    ):
        """Apply the same differentiable merge during training and inference."""
        context = head_context(state, method, "candidate")
        q = (
            self.candidate(features, context).sigmoid()
            if candidate is None
            else candidate
        )
        approved = state[:, 3:5].sum(1, keepdim=True).clamp(0, 1)
        if method in SPATIAL_METHODS:
            if method in ("global", "hard_lock", "dmp"):
                gate = torch.ones_like(q)
            else:
                if support is None or len(q) != 1:
                    raise ValueError(
                        "Spatial controls require one current prompt support"
                    )
                mask = spatial_gate(q, state, support, spacing, method, radius)
                gate = torch.as_tensor(mask, dtype=q.dtype, device=q.device)[None, None]
            if method != "global":
                gate = gate * (1 - approved)
        elif method == "fixed_gate":
            gate = (1 - approved) * fixed_gate
        else:
            context = head_context(state, method, "gate")
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
        if method != "global" and project:
            p = torch.where(approved.bool(), state[:, 3:4], p)
        return p, q, gate


def spatial_gate(q, state, support, spacing, method, radius):
    """Hard supports depend on detached masks; gradients pass through the candidate."""
    candidate = q.detach()[0, 0].cpu().numpy()
    old = state[0, 0].detach().cpu().numpy()
    if method == "fixed_roi":
        return physical_dilation(support, radius, spacing)
    if method == "progressive_merge":
        difference, _ = ndi.label((candidate >= 0.5) != (old >= 0.5), np.ones((3, 3)))
        touched = np.unique(difference[support])
        return np.isin(difference, touched[touched > 0])
    entropy = -(
        candidate * np.log(candidate + 1e-6)
        + (1 - candidate) * np.log(1 - candidate + 1e-6)
    )
    entropy[state[0, 3:5].sum(0).detach().cpu().numpy() > 0] = 0
    center = np.unravel_index(
        ndi.gaussian_filter(entropy, np.asarray([radius, radius]) / spacing).argmax(),
        old.shape,
    )
    point = np.zeros(old.shape, dtype=bool)
    point[center] = True
    return physical_dilation(point, radius, spacing)


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
        "candidate_only",
        "gate_only",
        "fixed_gate",
        "no_area",
        "no_regression",
        "no_boundary",
        "no_revocation",
    ):
        return base
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
    fixed_gate=0.5,
):
    """Decode a correction without labels; timings exclude cached image encoding."""

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    initial.eval()
    editor.eval()
    if features is None:
        with torch.inference_mode():
            features = initial.encoder(torch.from_numpy(image)[None, None].to(device))
    sync()
    total_start = time.perf_counter()
    channels = torch.from_numpy(state.channels(spacing, band_mm))[None].to(device)
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        context = head_context(channels, method, "candidate")
        q = editor.candidate(features, context).sigmoid()
        sync()
        candidate_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        updated, _, _ = editor(
            features,
            channels,
            method,
            candidate=q,
            spacing=spacing,
            support=prompt_support,
            radius=radius,
            fixed_gate=fixed_gate,
            project=False,
        )
        sync()
        gate_merge_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if method != "global":
            approved = channels[:, 3:5].sum(1, keepdim=True) > 0
            updated = torch.where(approved, channels[:, 3:4], updated)
        sync()
        projection_ms = (time.perf_counter() - start) * 1000
    state.probability = updated[0, 0].cpu().numpy().copy()
    state.revoked.fill(0)
    total_ms = (time.perf_counter() - total_start) * 1000
    return {
        "candidate_ms": candidate_ms,
        "gate_and_merge_ms": gate_merge_ms,
        "total_ms": total_ms,
        "dispatch_ms": max(
            0.0, total_ms - candidate_ms - gate_merge_ms - projection_ms
        ),
        "projection_ms": projection_ms,
    }


def simulate_action(state, truth, spacing, rng, wrong_rate=0, wrong_mode="event"):
    """Offline reviewer only: reference labels never enter the learned heads."""
    error = (state.probability >= 0.5) != truth.astype(bool)
    if (state.actions + 1) % 3 == 0 and state.reviewed_area < 0.12 * error.size:
        edge = max(1, int(np.sqrt(0.009 * error.size)))
        x, y = rng.integers(0, np.maximum(1, np.asarray(error.shape) - edge + 1))
        coordinates = np.argwhere(np.ones((edge, edge), dtype=bool)) + (x, y)
        labels = truth[coordinates[:, 0], coordinates[:, 1]].astype(int)
        flip = rng.random(len(labels) if wrong_mode == "pixel" else 1) < wrong_rate
        labels = np.where(flip, 1 - labels, labels)
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
