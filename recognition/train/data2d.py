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


def load_slice(case, id_to_source):
    source = nib.load(ROOT / case["image"])
    axis, index = case["slice_axis"], case["slice_index"]
    region = [slice(None)] * 3
    region[axis] = slice(index, index + 1)
    region = tuple(region)
    values = np.asarray(source.dataobj[region], dtype=np.float32)
    image = values.squeeze(axis)

    # training and prediction use the same per slice z score
    mean, std = float(image.mean()), float(image.std())
    image = (image - mean) / (std + 1e-8)

    label = nib.load(ROOT / case["label"])
    offset = np.eye(4)
    offset[axis, 3] = index
    shape = list(source.shape)
    shape[axis] = 1
    
    resampled = resample_from_to(label, (tuple(shape), source.affine @ offset), order=0)
    label = map_labels(np.asanyarray(resampled.dataobj).squeeze(axis), id_to_source)
    return {"image": image, "label": label}


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
