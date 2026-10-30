import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED = ROOT / "data/preprocess3d"


def map_labels(values, id_to_source):
    """map source label IDs to consecutive class indices"""
    # map source label ids to class indexes
    mapping = {source: index for index, source in enumerate(id_to_source)}
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)


def preprocess(config):
    """resolve voxel spacing and cache processed volumes, labels, and geometry"""
    if PREPROCESSED.exists():
        print("Already preprocessed 3D data")
        return

    settings = config["data"]
    root = ROOT / settings["root"]
    cases = []
    paths = sorted((root / settings["image_dir"]).glob("*" + settings["image_suffix"]))
    print(f"Preprocessing {len(paths)} scans into 3D volumes", flush=True)
    # collect scan paths and voxel spacing in ras orientation
    for path in paths:
        case_id = path.name.removesuffix(settings["image_suffix"])
        label_path = root / settings["label_dir"] / (case_id + settings["label_suffix"])
        source = nib.load(path)
        orientation = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(source.affine), nib.orientations.axcodes2ornt("RAS"))
        affine = source.affine @ nib.orientations.inv_ornt_aff(orientation, source.shape)
        cases.append({"case_id": case_id,
                      "patient_id": case_id.split(settings["patient_separator"])[0],
                      "image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT)),
                      "spacing": nib.affines.voxel_sizes(affine).tolist()})

    # resolve voxel spacing using only training patients
    preprocessing = dict(config["preprocessing"])
    if preprocessing["target_spacing_mm"] is None:
        patients = json.loads((ROOT / settings["splits_file"]).read_text(encoding="utf-8"))["train"]
        training = [case for case in cases if case["patient_id"] in patients]
        medians = [np.median([case["spacing"] for case in training if case["patient_id"] == patient], axis=0)
                   for patient in sorted({case["patient_id"] for case in training})]
        preprocessing["target_spacing_mm"] = np.median(medians, axis=0).tolist()

    # save each scan under its patient and retain geometry for prediction
    PREPROCESSED.mkdir(parents=True)
    for number, case in enumerate(cases, 1):
        directory = PREPROCESSED / case["patient_id"]
        directory.mkdir(exist_ok=True)
        item = preprocess_volume(case, preprocessing, config["labels"]["id_to_source"])
        for name in ("image", "label"):
            np.save(directory / f"{case['case_id']}_{name}.npy", np.ascontiguousarray(item[name]))
        case["geometry"] = item["geometry"]
        del case["spacing"]
        print(f"[{number}/{len(cases)}] {case['case_id']}: {item['image'].shape}", flush=True)
    (PREPROCESSED / "cases.json").write_text(
        json.dumps({"preprocessing": preprocessing, "cases": cases}), encoding="utf-8")
    print(f"Saved preprocessed data to {PREPROCESSED}", flush=True)


def preprocess_volume(case, preprocessing, id_to_source):
    # load image voxels for preprocessing
    """resample, crop, and normalize a scan and align its labels"""
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
    label = nib.load(ROOT / case["label"])
    resampled = resample_from_to(label, (grid.shape, grid.affine), order=0)
    item["label"] = map_labels(np.asanyarray(resampled.dataobj)[region], id_to_source)

    return item
