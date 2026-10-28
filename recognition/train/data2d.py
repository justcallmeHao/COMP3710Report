import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover(partition):
    root = ROOT / "data" / "keras_slices_data"
    cases = []

    for path in sorted((root / f"keras_slices_{partition}").glob("case_*.nii.gz")):
        # pair each slice with its segmentation mask
        label_path = root / f"keras_slices_seg_{partition}" / path.name.replace("case_", "seg_", 1)
        cases.append({"image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT))})

    return cases


def load_slice(case):
    source = nib.load(ROOT / case["image"])
    image = source.get_fdata(dtype=np.float32, caching="unchanged")

    # remove the extra dimension used by some nifti slices
    if image.ndim == 3:
        image = image[:, :, 0]

    # normalize pixels while preserving the supplied slice grid
    mean, std = float(image.mean()), float(image.std())
    image = (image - mean) / (std + 1e-8)

    label = nib.load(ROOT / case["label"]).get_fdata(dtype=np.float32, caching="unchanged")
    if label.ndim == 3:
        label = label[:, :, 0]

    return {"image": image, "label": label}


class SliceDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_slice(self.cases[index])

        # add channel dimensions and use contiguous training arrays
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)

        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
