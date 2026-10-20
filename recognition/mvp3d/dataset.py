import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.identity = sha256(self.path)
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        cases = self.document["cases"]
        expected = {"B006_Week0": "train", "B037_Week0": "train",
                    "B038_Week0": "validation"}
        if len(cases) != 3 or {c["case_id"]: c["split"] for c in cases} != expected:
            raise ValueError("MVP requires the fixed three-case split.")
        if self.document["label_ids"] != list(range(6)):
            raise ValueError("Expected fixed label IDs 0-5.")
        if len({c["patient_prefix"] for c in cases}) != 3:
            raise ValueError("Filename patient groups overlap.")
        self.cases = {c["case_id"]: c for c in cases}

    def path_for(self, case_id, kind):
        case = self.cases[case_id]
        path = (self.path.parent / case[kind]).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError("Manifest paths must stay within the subset directory.")
        if sha256(path) != case[f"{kind}_sha256"]:
            raise ValueError(f"Manifest hash mismatch: {path}")
        return path


def read_volume(path):
    image = nib.load(path)
    if len(image.shape) != 3 or any(n < 2 for n in image.shape):
        raise ValueError(f"{path}: expected a nondegenerate 3D array.")
    affine = image.affine
    if (not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12
            or not np.allclose(affine[3], [0, 0, 0, 1])):
        raise ValueError(f"{path}: invalid spatial affine.")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: nonfinite voxel values.")
    return image, data


def class_counts(data):
    if (not np.isfinite(data).all() or np.any(data != np.floor(data))
            or np.any(data < 0) or np.any(data > 5)):
        raise ValueError("Labels must be finite integer IDs 0-5.")
    return np.bincount(data.astype(np.int64).ravel(), minlength=6)


def prepare_image(manifest, case_id, config):
    source, data = read_volume(manifest.path_for(case_id, "image"))
    if source.header.get_xyzt_units()[0] != "mm":
        raise ValueError("MRI spatial units must explicitly be millimetres.")
    dhw = config["grid_dhw"]
    if len(dhw) != 3 or any(type(n) is not int or n < 16 or n % 8 for n in dhw):
        raise ValueError("Grid dimensions must be integers >=16 divisible by eight.")
    shape = (dhw[1], dhw[2], dhw[0])
    scale = np.asarray(source.shape, dtype=float) / np.asarray(shape)
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = (scale - 1) / 2
    target_affine = source.affine @ transform
    target = resample_from_to(source, (shape, target_affine), order=1,
                              mode="nearest").get_fdata(dtype=np.float32)
    if config["normalization"] != "all_voxels_zscore":
        raise ValueError("Unsupported normalization rule.")
    mean, std = float(target.mean(dtype=np.float64)), float(target.std(dtype=np.float64))
    epsilon = float(config["normalization_epsilon"])
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Normalization epsilon must be finite and positive.")
    normalized = ((target - mean) / std if std > epsilon else np.zeros_like(target))
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized image.")
    metadata = {
        "case_id": case_id, "original_shape_hwd": list(source.shape),
        "original_affine": source.affine.tolist(),
        "target_shape_hwd": list(shape), "target_affine": target_affine.tolist(),
        "target_spacing_hwd_mm": nib.affines.voxel_sizes(target_affine).tolist(),
        "axis_conversion": "NIfTI HWD -> tensor DHW: transpose(2,0,1)",
        "extent_convention": "preserve voxel-cell outer edges; centre offset (scale-1)/2",
        "normalization_mean": mean, "normalization_std": std,
        "zero_variance": std <= epsilon,
    }
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))[None]
    return tensor.float(), metadata


def prepare_label(manifest, case_id, metadata):
    source, data = read_volume(manifest.path_for(case_id, "label"))
    if (list(source.shape) != metadata["original_shape_hwd"]
            or not np.allclose(source.affine, metadata["original_affine"],
                               rtol=1e-5, atol=1e-5)):
        raise ValueError(f"{case_id}: image/mask shapes or affines are misaligned.")
    units = source.header.get_xyzt_units()[0]
    if units not in ("mm", "unknown"):
        raise ValueError(f"Unsupported mask units: {units}")
    metadata["mask_units"] = units
    metadata["mask_unit_assumption"] = (
        "Inherited mm from MRI after matching shape and affine."
        if units == "unknown" else "Explicit mm")
    before = class_counts(data)
    target = resample_from_to(
        source, (tuple(metadata["target_shape_hwd"]), np.array(metadata["target_affine"])),
        order=0, mode="nearest").get_fdata(dtype=np.float32)
    after = class_counts(target)
    metadata["class_counts_original"] = before.tolist()
    metadata["class_counts_target"] = after.tolist()
    missing = np.flatnonzero((before > 0) & (after == 0)).tolist()
    if missing:
        raise ValueError(f"{case_id}: classes {missing} disappeared; use a finer grid. "
                         f"Before={before.tolist()}, after={after.tolist()}")
    return torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).long()


class VolumeDataset(Dataset):
    def __init__(self, manifest, case_ids, config):
        self.manifest, self.case_ids, self.config = manifest, list(case_ids), config

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        image, metadata = prepare_image(self.manifest, case_id, self.config)
        mask = prepare_label(self.manifest, case_id, metadata)
        return image, mask, metadata


def collate_pairs(items):
    images, masks, metadata = zip(*items)
    return torch.stack(images), torch.stack(masks), list(metadata)
