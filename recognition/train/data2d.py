import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover(partition):
    partition = "validation" if partition == "validate" else partition
    patients = read_json(ROOT / "configs/splits.json")[partition]
    cases = []

    for path in sorted((ROOT / "data/semantic_MRs").glob("*_LFOV.nii.gz")):
        case_id = path.name.removesuffix("_LFOV.nii.gz")
        label_path = ROOT / "data/semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"
        cases.append({"case_id": case_id,
                      "patient_id": case_id.split("_Week", 1)[0],
                      "image": str(path.relative_to(ROOT)),
                      "label": str(label_path.relative_to(ROOT))})

    slice_cases = []
    for case in select_cases(cases, patients):
        source = nib.load(ROOT / case["image"])
        geometry = {"original_shape": list(source.shape),
                    "original_affine": source.affine.tolist()}
        
        for index in range(source.shape[2]):
            slice_cases.append({**case, "slice_axis": 2, "slice_index": index,
                                "geometry": geometry})
    return slice_cases


def select_cases(cases, patients):
    return [case for case in cases if case["patient_id"] in patients]


def map_labels(values, id_to_source):
    mapping = {source: index for index, source in enumerate(id_to_source)}
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)


def cached_slices(path, axis, id_to_source=(), reference=None):
    # invalidate cached arrays when source files or label mappings change
    paths = [path] if reference is None else [path, reference]
    signature = [(str(item), item.stat().st_size, item.stat().st_mtime_ns) for item in paths]
    key = hashlib.sha256(json.dumps([signature, axis, id_to_source, 1]).encode()).hexdigest()
    cached = OUTPUTS / "cache2d" / f"{key}.npy"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        source = nib.load(path)
        if reference is None:
            values = np.asarray(source.dataobj, dtype=np.float32)
        else:
            target = nib.load(reference)
            # align the mask once per scan, only when its grid differs
            if source.shape != target.shape or not np.array_equal(source.affine, target.affine):
                source = resample_from_to(source, (target.shape, target.affine), order=0)
            values = map_labels(np.asanyarray(source.dataobj), id_to_source)
        # store adjacent pixels together so shuffled slices read efficiently
        np.save(cached, np.ascontiguousarray(np.moveaxis(values, axis, 0)))
    return np.load(cached, mmap_mode="r")


def load_slice(case, id_to_source, with_label=True):
    axis, index = case["slice_axis"], case["slice_index"]
    image = cached_slices(ROOT / case["image"], axis)[index]

    # training and prediction use the same per slice z score
    mean, std = float(image.mean()), float(image.std())
    image = (image - mean) / (std + 1e-8)

    item = {"image": image}
    if with_label:
        item["label"] = cached_slices(ROOT / case["label"], axis, id_to_source,
                                      ROOT / case["image"])[index]
    return item


class SliceDataset(Dataset):
    def __init__(self, cases, id_to_source):
        self.cases = cases
        self.id_to_source = id_to_source

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_slice(self.cases[index], self.id_to_source)
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
