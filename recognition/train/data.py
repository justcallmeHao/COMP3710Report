import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to, resample_to_output
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def discover(config):
    settings = config["data"]
    root = ROOT / settings["root"]
    cases = []

    for path in sorted((root / settings["image_dir"]).glob("*" + settings["image_suffix"])):
        # match each image filename to its label path
        case_id = path.name.removesuffix(settings["image_suffix"])
        label_path = root / settings["label_dir"] / (case_id + settings["label_suffix"])

        # read metadata and calculate spacing in ras orientation
        image = nib.load(path)
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt("RAS"))
        affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape)

        # record patient identity, relative paths, and voxel spacing
        cases.append({"case_id": case_id,
                      "patient_id": case_id.split(settings["patient_separator"])[0],
                      "image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT)),
                      "spacing": nib.affines.voxel_sizes(affine).tolist()})

    # return the discovered scan records
    return cases

def select_cases(cases, patients):
    return [case for case in cases if case["patient_id"] in patients]

def resolve_spacing(config, training_cases):
    resolved = copy.deepcopy(config)

    # derive missing spacing from patient medians using training scans only
    if config["preprocessing"]["target_spacing_mm"] is None:
        patients = sorted({case["patient_id"] for case in training_cases})
        per_patient = [np.median([c["spacing"] for c in training_cases if c["patient_id"] == p], axis=0)
                       for p in patients]
        resolved["preprocessing"]["target_spacing_mm"] = np.median(per_patient, axis=0).tolist()
    return resolved

def map_labels(values, id_to_source):
    # map source label ids to class indexes
    mapping = {source: index for index, source in enumerate(id_to_source)}
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)

def load_volume(case, preprocessing, id_to_source, with_label=True):
    # load image voxels for preprocessing
    settings = preprocessing
    source = nib.load(ROOT / case["image"])
    values = source.get_fdata(dtype=np.float32)

    # orient the image to ras and resample to the target spacing
    oriented = nib.as_closest_canonical(nib.Nifti1Image(values, source.affine))
    grid = resample_to_output(oriented, voxel_sizes=settings["target_spacing_mm"], order=1)
    image = grid.get_fdata(dtype=np.float32)

    # find foreground bounds and expand them by the crop margin
    foreground = image > settings["foreground_threshold"]
    points = np.where(foreground)
    start = np.maximum(np.array([axis.min() for axis in points]) - settings["crop_margin"], 0)
    end = np.minimum(np.array([axis.max() for axis in points]) + 1 + settings["crop_margin"], image.shape)

    # crop the image to the foreground region
    region = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    cropped = image[region].copy()

    # z-score nonzero voxels while keeping zero background unchanged
    mask = cropped != 0
    mean, std = float(cropped[mask].mean()), float(cropped[mask].std())
    cropped[mask] = (cropped[mask] - mean) / max(std, settings["normalization_epsilon"])

    # store the image and geometry needed to restore predictions
    item = {"image": cropped,
            "geometry": {"original_shape": list(source.shape), "original_affine": source.affine.tolist(),
                         "grid_shape": list(grid.shape), "grid_affine": grid.affine.tolist(),
                         "crop_start": start.tolist(), "crop_end": end.tolist()}}

    # resample labels with nearest neighbours and apply the same crop
    if with_label:
        label = nib.load(ROOT / case["label"])
        resampled = resample_from_to(label, (grid.shape, grid.affine), order=0)
        item["label"] = map_labels(np.asanyarray(resampled.dataobj)[region], id_to_source)

    # return the preprocessed image and optional labels
    return item

class PatchDataset(Dataset):
    def __init__(self, cases, preprocessing, id_to_source):
        self.cases = cases
        self.preprocessing = preprocessing
        self.id_to_source = id_to_source

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_volume(self.cases[index], self.preprocessing, self.id_to_source)

        # add channel dimensions and use contiguous arrays with training dtypes
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)

        # convert image and label arrays to torch tensors
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
