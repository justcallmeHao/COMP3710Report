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


def tile_regions(shape, output_size):
    for top in range(0, shape[0], output_size[0]):
        for left in range(0, shape[1], output_size[1]):
            yield top, left, min(output_size[0], shape[0] - top), min(output_size[1], shape[1] - left)


def pad_image(image, input_size, output_size):
    padding = [((before - after) // 2, (before - after) // 2 + (-size) % after)
               for size, before, after in zip(image.shape, input_size, output_size)]
    return np.pad(image, padding, mode="reflect")


class SliceDataset(Dataset):
    def __init__(self, cases, settings):
        self.cases = cases
        self.input_size = settings["input_patch_size"]
        self.output_size = settings["output_patch_size"]
        self.tiles = [(index, region) for index, case in enumerate(cases)
                      for region in tile_regions(case["geometry"]["original_shape"][:2], self.output_size)]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        case_index, (top, left, height, width) = self.tiles[index]
        item = load_slice(self.cases[case_index])
        padded = pad_image(item["image"], self.input_size, self.output_size)
        image = np.array(padded[None, top:top + self.input_size[0], left:left + self.input_size[1]],
                         dtype=np.float32, order="C", copy=True)
        
        label = np.full((1, *self.output_size), -100, dtype=np.int64)
        label[0, :height, :width] = item["label"][top:top + height, left:left + width]
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
