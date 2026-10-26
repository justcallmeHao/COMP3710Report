import copy
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def discover(config):
    settings = config["data"]
    root = ROOT / settings["root"]
    cases = []
    for path in sorted((root / settings["image_dir"]).glob("*" + settings["image_suffix"])):
        case_id = path.name.removesuffix(settings["image_suffix"])
        label_path = root / settings["label_dir"] / (case_id + settings["label_suffix"])
        image = nib.load(path)
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt("RAS"))
        affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape)
        cases.append({"case_id": case_id,
                      "patient_id": case_id.split(settings["patient_separator"])[0],
                      "image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT)),
                      "spacing": nib.affines.voxel_sizes(affine).tolist(),
                      "source_identity": [[p.stat().st_size, p.stat().st_mtime_ns]
                                          for p in (path, label_path)]})
    return cases


def split_manifest(config, cases):
    settings = config["data"]
    path = ROOT / settings["splits_file"]
    patients = sorted({case["patient_id"] for case in cases})
    definition = {"seed": config["seed"], "n_folds": settings["n_folds"],
                  "test_fraction": settings["test_fraction"],
                  "official_test_patients": settings["official_test_patients"], "patients": patients}
    if path.exists():
        return read_json(path)
    groups = np.asarray(patients)
    official = settings["official_test_patients"]
    if official:
        development = np.array([p for p in patients if p not in official])
        test = sorted(official)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=settings["test_fraction"],
                                     random_state=config["seed"])
        dev_index, test_index = next(splitter.split(groups, groups=groups))
        development, test = groups[dev_index], groups[test_index].tolist()
    folds = settings["n_folds"]
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=config["seed"])
    manifest = {"definition": definition, "test": test, "folds": [
        {"train": development[train].tolist(), "validation": development[val].tolist()}
        for train, val in splitter.split(development, groups=development)]}
    write_json(path, manifest)
    return manifest


def select_cases(cases, patients):
    return [case for case in cases if case["patient_id"] in patients]


def resolve_spacing(config, training_cases):
    resolved = copy.deepcopy(config)
    if config["preprocessing"]["target_spacing_mm"] is None:
        patients = sorted({case["patient_id"] for case in training_cases})
        per_patient = [np.median([c["spacing"] for c in training_cases if c["patient_id"] == p], axis=0)
                       for p in patients]
        resolved["preprocessing"]["target_spacing_mm"] = np.median(per_patient, axis=0).tolist()
    return resolved


def map_labels(values, id_to_source):
    mapping = {source: index for index, source in enumerate(id_to_source)}
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)


def prepare_volume(case, settings, id_to_source, with_label):
    source = nib.load(ROOT / case["image"])
    oriented = nib.as_closest_canonical(nib.Nifti1Image(source.get_fdata(dtype=np.float32), source.affine))
    grid = resample_to_output(oriented, voxel_sizes=settings["target_spacing_mm"], order=1)
    image = grid.get_fdata(dtype=np.float32)
    foreground = image > settings["foreground_threshold"]
    points = np.where(foreground)
    start = np.maximum(np.array([axis.min() for axis in points]) - settings["crop_margin"], 0)
    end = np.minimum(np.array([axis.max() for axis in points]) + 1 + settings["crop_margin"], image.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    cropped = image[region].copy()
    mask = cropped != 0
    mean, std = float(cropped[mask].mean()), float(cropped[mask].std())
    cropped[mask] = (cropped[mask] - mean) / max(std, settings["normalization_epsilon"])
    item = {"image": cropped,
            "geometry": {"original_shape": list(source.shape), "original_affine": source.affine.tolist(),
                         "grid_shape": list(grid.shape), "grid_affine": grid.affine.tolist(),
                         "crop_start": start.tolist(), "crop_end": end.tolist()}}
    if with_label:
        label = nib.load(ROOT / case["label"])
        resampled = resample_from_to(label, (grid.shape, grid.affine), order=0)
        item["label"] = map_labels(np.asanyarray(resampled.dataobj)[region], id_to_source)
    return item


def load_volume(case, preprocessing, id_to_source, with_label=True):
    fingerprint = identity({"preprocessing": preprocessing,
                            "labels": id_to_source if with_label else None})
    cache = OUTPUTS / "cache" / f"numpy-ras-v1-{fingerprint}"
    cache.mkdir(parents=True, exist_ok=True)
    sources = [case["image"]] + ([case["label"]] if with_label else [])
    fingerprint = identity([[p, (ROOT / p).stat().st_size, (ROOT / p).stat().st_mtime_ns] for p in sources])
    path = cache / f"{case['case_id']}-{fingerprint}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    item = prepare_volume(case, preprocessing, id_to_source, with_label)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(item, temporary)
    temporary.replace(path)
    return item


class PatchDataset(Dataset):
    def __init__(self, cases, sampling, augmentation, preprocessing, id_to_source):
        self.cases = cases
        self.sampling, self.augmentation, self.preprocessing = sampling, augmentation, preprocessing
        self.id_to_source = id_to_source
        self.prepared = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = int(sampling["memory_cache_mb"] * 1024**2)

    def __len__(self):
        return len(self.cases)

    def prepared_volume(self, index):
        if index in self.prepared:
            self.prepared.move_to_end(index)
            return self.prepared[index]
        item = load_volume(self.cases[index], self.preprocessing, self.id_to_source)
        patch = self.sampling["patch_size"]
        padding = [(0, max(0, n - size)) for n, size in zip(patch, item["image"].shape)]
        image, label = np.pad(item["image"], padding), np.pad(item["label"], padding)
        candidates = (np.flatnonzero(label == 0), np.flatnonzero(label > 0))
        prepared = (image, label, *candidates)
        size = sum(array.nbytes for array in prepared)
        if size <= self.cache_limit:
            while self.prepared and self.cache_bytes + size > self.cache_limit:
                _, removed = self.prepared.popitem(last=False)
                self.cache_bytes -= sum(array.nbytes for array in removed)
            self.prepared[index] = prepared
            self.cache_bytes += size
        return prepared

    def __getitem__(self, index):
        image, label, background, foreground = self.prepared_volume(index)
        rng = np.random.default_rng(int(torch.randint(0, 2**32, (1,)).item()))
        sampling, aug = self.sampling, self.augmentation
        patch = np.array(sampling["patch_size"])
        positive = rng.random() < sampling["foreground_fraction"]
        candidates = foreground if positive else background
        if not candidates.size:
            candidates = background if positive else foreground
        center = np.array(np.unravel_index(rng.choice(candidates), label.shape))
        start = np.clip(center - patch // 2, 0, np.array(label.shape) - patch)
        region = tuple(slice(int(a), int(a + size)) for a, size in zip(start, patch))
        image, label = image[region].copy(), label[region].copy()
        if rng.random() < aug["spatial_probability"]:
            rotation = Rotation.from_euler("xyz", rng.uniform(-aug["rotation_degrees"], aug["rotation_degrees"], 3), degrees=True).as_matrix()
            spacing = np.diag(self.preprocessing["target_spacing_mm"])
            matrix = np.linalg.inv(spacing) @ rotation.T @ spacing / rng.uniform(*aug["scale_range"])
            center = (patch - 1) / 2
            offset = center - matrix @ center
            image = affine_transform(image, matrix, offset, order=1, mode="constant", cval=0)
            label = affine_transform(label, matrix, offset, order=0, mode="constant", cval=0)
        for axis in aug["mirror_axes"]:
            if rng.random() < aug["mirror_probability"]:
                image, label = np.flip(image, axis), np.flip(label, axis)
        if rng.random() < aug["gamma_probability"]:
            mean, std, low = image.mean(), image.std(), image.min()
            epsilon = self.preprocessing["normalization_epsilon"]
            span = max(float(image.max() - low), epsilon)
            image = ((image - low) / span) ** rng.uniform(*aug["gamma_range"])
            image = (image - image.mean()) / max(float(image.std()), epsilon) * std + mean
        if rng.random() < aug["noise_probability"]:
            image = image + rng.normal(0, rng.uniform(0, aug["noise_std"]), image.shape)
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                "label": torch.from_numpy(np.ascontiguousarray(label[None], dtype=np.int64))}
