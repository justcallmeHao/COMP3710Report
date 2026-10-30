import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess3D import PREPROCESSED, ROOT, map_labels, preprocess

OUTPUTS = ROOT / "outputs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover(config):
    preprocess(config)
    return read_json(PREPROCESSED / "cases.json")["cases"]


def select_cases(cases, patients):
    return [case for case in cases if case["patient_id"] in patients]


def resolve_spacing(config):
    # reuse the preprocessing settings that produced the cached volumes
    resolved = copy.deepcopy(config)
    resolved["preprocessing"] = read_json(PREPROCESSED / "cases.json")["preprocessing"]
    return resolved


def load_volume(case, with_label=True):
    directory = PREPROCESSED / case["patient_id"]
    names = ("image", "label") if with_label else ("image",)
    item = {name: np.load(directory / f"{case['case_id']}_{name}.npy") for name in names}
    item["geometry"] = case["geometry"]
    return item


class PatchDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_volume(self.cases[index])
        # add channel dimensions and use contiguous arrays with training dtypes
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
