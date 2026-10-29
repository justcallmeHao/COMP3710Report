import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess2D import PREPROCESSED, ROOT, map_labels, preprocess

OUTPUTS = ROOT / "outputs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover(partition):
    preprocess()
    partition = "validation" if partition == "validate" else partition
    patients = read_json(ROOT / "configs/splits.json")[partition]
    cases = read_json(PREPROCESSED / "cases.json")
    return [{**case, "slice_axis": 2, "slice_index": index}
            for case in cases if case["patient_id"] in patients
            for index in range(case["geometry"]["original_shape"][2])]


def load_slice(case, with_label=True):
    directory = PREPROCESSED / case["patient_id"]
    names = ("image", "label") if with_label else ("image",)
    return {name: np.load(directory / f"{case['case_id']}_{name}.npy", mmap_mode="r")[case["slice_index"]]
            for name in names}


class SliceDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_slice(self.cases[index])
        # copy read-only cached slices into writable training tensors
        image = np.array(item["image"][None], dtype=np.float32, order="C", copy=True)
        label = np.array(item["label"][None], dtype=np.int64, order="C", copy=True)
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
