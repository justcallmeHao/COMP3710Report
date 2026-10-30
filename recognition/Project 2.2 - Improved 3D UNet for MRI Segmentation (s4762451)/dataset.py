import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from helpers.preprocess3D import PREPROCESSED, ROOT, map_labels, preprocess

OUTPUTS = ROOT / "outputs"


def read_json(path):
    """load data from a JSON file"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """save data as indented JSON"""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover(config):
    """prepare cached volumes and return the scan manifest"""
    preprocess(config)
    return read_json(PREPROCESSED / "cases.json")["cases"]


def select_cases(cases, patients):
    """select scans belonging to the requested patients"""
    return [case for case in cases if case["patient_id"] in patients]


def resolve_spacing(config):
    # reuse the preprocessing settings that produced the cached volumes
    """copy the config with cached preprocessing settings"""
    resolved = copy.deepcopy(config)
    resolved["preprocessing"] = read_json(PREPROCESSED / "cases.json")["preprocessing"]
    return resolved


def load_volume(case, with_label=True):
    """load a cached image volume, optional labels, and scan geometry"""
    directory = PREPROCESSED / case["patient_id"]
    names = ("image", "label") if with_label else ("image",)
    item = {name: np.load(directory / f"{case['case_id']}_{name}.npy") for name in names}
    item["geometry"] = case["geometry"]
    return item


class PatchDataset(Dataset):
    def __init__(self, cases):
        """store the scan cases for volume loading"""
        self.cases = cases

    def __len__(self):
        """return the number of scans"""
        return len(self.cases)

    def __getitem__(self, index):
        """load a scan and return image and label tensors with channel dimensions"""
        item = load_volume(self.cases[index])
        # add channel dimensions and use contiguous arrays with training dtypes
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
