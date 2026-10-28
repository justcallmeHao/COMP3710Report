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
        case_id = path.name.removesuffix(settings["image_suffix"])  # match each image to its label
        label_path = root / settings["label_dir"] / (case_id + settings["label_suffix"])
        image = nib.load(path)  # read image metadata without loading voxel values
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt("RAS"))
        affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape)
        cases.append({"case_id": case_id,
                      "patient_id": case_id.split(settings["patient_separator"])[0],
                      "image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT)),
                      "spacing": nib.affines.voxel_sizes(affine).tolist()})
    return cases

def split_patients(patients, fraction, seed):
    count = int(np.ceil(len(patients) * fraction))  # round up the held-out patient count
    indices = np.random.RandomState(seed).permutation(len(patients))  # repeat the shuffle with the same seed
    return patients[np.sort(indices[count:])], patients[np.sort(indices[:count])]

def split_manifest(config, cases):
    settings = config["data"]
    path = ROOT / settings["splits_file"]
    patients = sorted({case["patient_id"] for case in cases})  # keep each patient's scans together
    validation_fraction = settings.get("validation_fraction", 0.2)  # fraction of development patients
    definition = {"seed": config["seed"], "validation_fraction": validation_fraction,
                  "test_fraction": settings["test_fraction"],
                  "official_test_patients": settings["official_test_patients"], "patients": patients}
    groups = np.asarray(patients)
    official = settings["official_test_patients"]
    if official:
        development = np.array([p for p in patients if p not in official])  # reserve official test patients
        test = sorted(official)
    else:
        development, test = split_patients(groups, settings["test_fraction"], config["seed"])
        test = test.tolist()
    train, validation = split_patients(development, validation_fraction, config["seed"])
    manifest = {"definition": definition, "test": test, "folds": [
        {"train": train.tolist(), "validation": validation.tolist()}]}
    write_json(path, manifest)
    return manifest

def select_cases(cases, patients):
    return [case for case in cases if case["patient_id"] in patients]

def resolve_spacing(config, training_cases):
    resolved = copy.deepcopy(config)  # leave the original configuration unchanged
    if config["preprocessing"]["target_spacing_mm"] is None:
        patients = sorted({case["patient_id"] for case in training_cases})
        per_patient = [np.median([c["spacing"] for c in training_cases if c["patient_id"] == p], axis=0)
                       for p in patients]
        resolved["preprocessing"]["target_spacing_mm"] = np.median(per_patient, axis=0).tolist()  # use training data only
    return resolved

def map_labels(values, id_to_source):
    mapping = {source: index for index, source in enumerate(id_to_source)}  # map source labels to class indices
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)

def load_volume(case, preprocessing, id_to_source, with_label=True):
    settings = preprocessing
    source = nib.load(ROOT / case["image"])
    values = source.get_fdata(dtype=np.float32)  # load voxels as float32 before interpolation
    oriented = nib.as_closest_canonical(nib.Nifti1Image(values, source.affine))  # use RAS orientation
    grid = resample_to_output(oriented, voxel_sizes=settings["target_spacing_mm"], order=1)  # use common voxel spacing
    image = grid.get_fdata(dtype=np.float32)
    foreground = image > settings["foreground_threshold"]
    points = np.where(foreground)
    start = np.maximum(np.array([axis.min() for axis in points]) - settings["crop_margin"], 0)
    end = np.minimum(np.array([axis.max() for axis in points]) + 1 + settings["crop_margin"], image.shape)
    region = tuple(slice(int(a), int(b)) for a, b in zip(start, end))  # crop around foreground
    cropped = image[region].copy()
    mask = cropped != 0  # normalize nonzero voxels and keep the background at zero
    mean, std = float(cropped[mask].mean()), float(cropped[mask].std())
    cropped[mask] = (cropped[mask] - mean) / max(std, settings["normalization_epsilon"])
    item = {"image": cropped,
            "geometry": {"original_shape": list(source.shape), "original_affine": source.affine.tolist(),
                         "grid_shape": list(grid.shape), "grid_affine": grid.affine.tolist(),
                         "crop_start": start.tolist(), "crop_end": end.tolist()}}
    if with_label:
        label = nib.load(ROOT / case["label"])
        resampled = resample_from_to(label, (grid.shape, grid.affine), order=0)  # preserve label ids
        item["label"] = map_labels(np.asanyarray(resampled.dataobj)[region], id_to_source)
    return item

class PatchDataset(Dataset):
    def __init__(self, cases, sampling, augmentation, preprocessing, id_to_source):
        self.cases = cases
        self.preprocessing = preprocessing  # keep the constructor compatible with training callers
        self.id_to_source = id_to_source

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = load_volume(self.cases[index], self.preprocessing, self.id_to_source)  # load the full scan and label
        image = np.ascontiguousarray(item["image"][None], dtype=np.float32)  # add the channel dimension
        label = np.ascontiguousarray(item["label"][None], dtype=np.int64)
        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}
