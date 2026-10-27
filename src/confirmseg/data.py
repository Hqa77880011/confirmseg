import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .utils import load, save, split_groups


def resize(array, shape, order=1):
    return ndi.zoom(
        array, np.asarray(shape) / np.asarray(array.shape), order=order, prefilter=False
    )


def normalize(image, modality):
    image = np.asarray(image, dtype=np.float32)
    if modality == "CT":
        return (np.clip(image, -1000, 400) + 1000) / 1400
    values = image[image != 0]
    if not len(values):
        values = image.reshape(-1)
    return (image - values.mean()) / max(float(values.std()), 1e-6)


def crop(array, center, shape, fill=0):
    shape = np.asarray(shape, dtype=int)
    start = np.asarray(center, dtype=int) - shape // 2
    end = start + shape
    output = np.full(tuple(shape), fill, dtype=array.dtype)
    low, high = np.maximum(start, 0), np.minimum(end, array.shape)
    if np.any(high <= low):
        return output
    output[tuple(slice(int(a), int(b)) for a, b in zip(low - start, high - start))] = (
        array[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
    )
    return output


def prepare_acdc(raw, output, seed=2027):
    """Prepare nonempty organ tasks, retaining phase/structure and patient strata."""
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mask_path in sorted(raw.rglob("patient*_frame*_gt.nii.gz")):
        image_path = mask_path.with_name(mask_path.name.replace("_gt", ""))
        mask_file, image_file = nib.load(mask_path), nib.load(image_path)
        image = normalize(image_file.get_fdata(), "MRI")
        labels = mask_file.get_fdata().astype(np.uint8)
        spacing = image_file.header.get_zooms()[:2]
        patient = mask_path.parent.name
        group = "unknown"
        info = mask_path.parent / "Info.cfg"
        if info.exists():
            metadata = dict(
                r.split(":", 1) for r in info.read_text().splitlines() if ":" in r
            )
            group = next(
                (
                    r.split(":", 1)[1].strip()
                    for r in info.read_text().splitlines()
                    if r.startswith("Group:")
                ),
                "unknown",
            )
        else:
            raise ValueError(f"Missing ACDC diagnostic group/phase metadata: {info}")
        frame = int(mask_path.name.split("_frame")[1].split("_")[0])
        phase = next(
            (name for name in ("ED", "ES") if int(metadata[name]) == frame), None
        )
        if phase is None:
            raise ValueError(f"Frame is neither ED nor ES: {mask_path}")
        for structure in (1, 2, 3):
            for z in range(image.shape[2]):
                target = labels[:, :, z] == structure
                if not target.any():
                    continue
                identifier = f"{mask_path.name.split('.')[0]}_s{structure}_z{z}"
                path = output / (identifier + ".npz")
                shape = (256, 256)
                np.savez_compressed(
                    path,
                    image=resize(image[:, :, z], shape).astype(np.float32),
                    masks=resize(target.astype(np.uint8), shape, 0)[None],
                    spacing=np.asarray(spacing) * np.asarray(target.shape) / shape,
                )
                rows.append(
                    {
                        "id": identifier,
                        "patient_id": patient,
                        "dataset": "ACDC",
                        "group": group,
                        "phase": phase,
                        "structure": structure,
                        "slice": z,
                        "normalized": True,
                        "path": path.name,
                    }
                )
    assigned = []
    for group in sorted({r["group"] for r in rows}):
        assigned.extend(
            split_groups([r for r in rows if r["group"] == group], "patient_id", seed)
        )
    if not assigned:
        raise ValueError("No labeled ACDC volumes found")
    save(output / "manifest.json", assigned)


def prepare_lidc(output, readers=1, seed=2027, cohort=500):
    """Select eligible patients and each patient's largest median-diameter nodule."""
    import pylidc as pl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    eligible = {}
    for scan in sorted(pl.query(pl.Scan).all(), key=lambda scan: scan.id):
        for index, annotations in enumerate(scan.cluster_annotations()):
            if (
                len(annotations) < readers
                or np.median([a.diameter for a in annotations]) < 3
            ):
                continue
            annotations = sorted(annotations, key=lambda annotation: annotation.id)
            diameter = float(
                np.median([annotation.diameter for annotation in annotations])
            )
            eligible.setdefault(scan.patient_id, []).append(
                (diameter, scan, index, annotations)
            )
    patients = np.random.default_rng(seed).permutation(sorted(eligible))
    if cohort and len(patients) < cohort:
        raise ValueError(
            f"Need {cohort} eligible patients, found {len(patients)}; --cohort 0 keeps all"
        )
    patients = patients[:cohort] if cohort else patients
    rows = []
    for patient in sorted(patients):
        diameter, scan, index, annotations = min(
            eligible[patient], key=lambda item: (-item[0], item[1].id, item[2])
        )
        volume = scan.to_volume().astype(np.float32)
        masks = []
        for annotation in annotations:
            mask = np.zeros(volume.shape, dtype=np.uint8)
            mask[annotation.bbox()] = annotation.boolean_mask().astype(np.uint8)
            masks.append(mask)
        identifier = f"{scan.patient_id}_scan{scan.id}_n{index}"
        path = output / (identifier + ".npz")
        np.savez_compressed(
            path,
            image=volume,
            masks=np.stack(masks),
            spacing=np.asarray(
                [scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing]
            ),
        )
        rows.append(
            {
                "id": identifier,
                "patient_id": scan.patient_id,
                "dataset": "LIDC",
                "modality": "CT",
                "diameter_mm": diameter,
                "reader_ids": [annotation.id for annotation in annotations],
                "path": path.name,
            }
        )
    if not rows:
        raise ValueError("No eligible LIDC nodules found; configure pylidc DICOM paths")
    save(output / "manifest.json", split_groups(rows, "patient_id", seed))


def prepare_manifest(source, output, seed=2027):
    """Validate curated arrays and preserve explicit, patient-consistent splits."""
    source, output = Path(source), Path(output)
    rows = load(source)
    if not rows:
        raise ValueError("Manifest is empty")
    if any(not {"id", "patient_id", "dataset", "path"} <= row.keys() for row in rows):
        raise ValueError("Each record needs id, patient_id, dataset, path")
    for row in rows:
        path = (source.parent / row["path"]).resolve()
        with np.load(path, allow_pickle=False) as case:
            if "image" not in case or "masks" not in case or "spacing" not in case:
                raise ValueError(f"Missing image, masks, or spacing: {path}")
            if case["masks"].shape[1:] != case["image"].shape:
                raise ValueError(f"Image and masks have different shapes: {path}")
            if not np.isin(case["masks"], [0, 1]).all():
                raise ValueError(f"Masks must be binary: {path}")
            if case["image"].ndim not in (2, 3) or not np.isfinite(case["image"]).all():
                raise ValueError(f"Image must be a finite 2D/3D array: {path}")
            if len(case["spacing"]) != case["image"].ndim or not np.all(
                case["spacing"] > 0
            ):
                raise ValueError(
                    f"Spacing must have one positive value per image axis: {path}"
                )
        try:
            relative = os.path.relpath(path, output.parent.resolve())
        except ValueError:
            # Windows paths on different drives cannot be made relative.
            relative = str(path)
        row["path"] = relative.replace(os.sep, "/")
    if all("split" not in row for row in rows):
        rows = split_groups(rows, "patient_id", seed)
    validate_splits(rows)
    save(output, rows)


def validate_splits(rows):
    """Reject patient leakage and ambiguous task identifiers at the data boundary."""
    patients, identifiers = {}, set()
    for row in rows:
        split = row.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("All records need valid splits, or all must omit split")
        patient = row["dataset"], row["patient_id"]
        if patient in patients and patients[patient] != split:
            raise ValueError(f"Patient crosses splits: {patient}")
        patients[patient] = split
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate task id: {row['id']}")
        identifiers.add(row["id"])


def records(manifest, split):
    path = Path(manifest)
    rows = load(path)
    validate_splits(rows)
    rows = [r for r in rows if r["split"] == split]
    for row in rows:
        row["path"] = str((path.parent / row["path"]).resolve())
    if not rows:
        raise ValueError(f"No cases in {split} split")
    return rows


def evaluation_tasks(rows, protocol="patient", seed=31415):
    """Select five stratified ACDC tasks or one designated LIDC reader per patient."""
    rng = np.random.default_rng(seed)
    selected = []
    for key in sorted({(row["dataset"], row["patient_id"]) for row in rows}):
        patient = sorted(
            (r for r in rows if (r["dataset"], r["patient_id"]) == key),
            key=lambda r: r["id"],
        )
        if protocol == "all":
            for row in patient:
                with np.load(row["path"], allow_pickle=False) as case:
                    selected.extend(
                        (row, reader) for reader in range(len(case["masks"]))
                    )
        elif key[0] == "ACDC":
            if any("phase" not in r or "structure" not in r for r in patient):
                raise ValueError(
                    "ACDC patient protocol needs phase and structure metadata"
                )
            strata = {}
            for row in patient:
                strata.setdefault((row["phase"], row["structure"]), []).append(row)
            keys = sorted(strata)
            pools = {
                keys[index]: list(rng.permutation(strata[keys[index]]))
                for index in rng.permutation(len(keys))
            }
            chosen = []
            while len(chosen) < 5 and any(pools.values()):
                for stratum in pools.values():
                    if stratum and len(chosen) < 5:
                        chosen.append((stratum.pop(), 0))
            if len(chosen) != 5:
                raise ValueError(f"Need five distinct nonempty ACDC tasks: {key[1]}")
            selected.extend(chosen)
        elif key[0] == "LIDC":
            row = min(patient, key=lambda r: (-r.get("diameter_mm", 0), r["id"]))
            selected.append((row, 0))
        else:
            selected.extend((row, 0) for row in patient)
    return selected


def fixed_field(row, initial=0, size=256, side_mm=96):
    with np.load(row["path"], allow_pickle=False) as obj:
        image = obj["image"].astype(np.float32)
        masks = obj["masks"].astype(np.uint8)
        spacing = obj["spacing"].astype(float)
    if image.ndim == 3:
        reference = masks[initial]
        z = int(reference.sum((0, 1)).argmax())
        image, masks = image[:, :, z], masks[:, :, :, z]
        points = np.argwhere(masks[initial])
        if not len(points):
            raise ValueError("Initial contour is empty")
        center = np.asarray(
            np.unravel_index(
                ndi.distance_transform_edt(
                    masks[initial], sampling=spacing[:2]
                ).argmax(),
                image.shape,
            )
        )
        shape = np.maximum(1, np.rint(side_mm / spacing[:2]).astype(int))
        image = crop(image, center, shape, -1000)
        masks = np.stack([crop(mask, center, shape) for mask in masks])
    old_shape = image.shape
    image = (
        normalize(
            image, row.get("modality", "CT" if row["dataset"] == "LIDC" else "MRI")
        )
        if not row.get("normalized", row["dataset"] == "ACDC")
        else image
    )
    image = resize(image, (size, size)).astype(np.float32)
    masks = np.stack([resize(mask, (size, size), 0) for mask in masks]).astype(
        np.float32
    )
    spacing = spacing[:2] * np.asarray(old_shape) / size
    return image, masks, spacing
