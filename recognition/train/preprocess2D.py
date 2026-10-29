import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED = ROOT / "data/preprocess2d"


def map_labels(values, id_to_source):
    mapping = {source: index for index, source in enumerate(id_to_source)}
    sources, inverse = np.unique(values, return_inverse=True)
    return np.asarray([mapping[source] for source in sources], dtype=np.uint8)[inverse].reshape(values.shape)


def preprocess():
    if PREPROCESSED.exists():
        print("Already preprocessed 2D data")
        return

    PREPROCESSED.mkdir(parents=True)
    id_to_source = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))["id_to_source"]
    paths = sorted((ROOT / "data/semantic_MRs").glob("*_LFOV.nii.gz"))
    cases = []
    print(f"Preprocessing {len(paths)} scans into 2D slices", flush=True)
    for number, path in enumerate(paths, 1):
        case_id = path.name.removesuffix("_LFOV.nii.gz")
        patient_id = case_id.split("_Week", 1)[0]
        directory = PREPROCESSED / patient_id
        directory.mkdir(exist_ok=True)
        label_path = ROOT / "data/semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"
        source, label = nib.load(path), nib.load(label_path)

        # align labels to the image grid using nearest neighbours
        if source.shape != label.shape or not np.array_equal(source.affine, label.affine):
            label = resample_from_to(label, (source.shape, source.affine), order=0)
        labels = map_labels(np.asanyarray(label.dataobj), id_to_source)

        # store slices and normalize each image slice once
        images = np.ascontiguousarray(np.moveaxis(np.asarray(source.dataobj, dtype=np.float32), 2, 0))
        for index, image in enumerate(images):
            mean, std = float(image.mean()), float(image.std())
            images[index] = (image - mean) / (std + 1e-8)
        np.save(directory / f"{case_id}_image.npy", images)
        np.save(directory / f"{case_id}_label.npy", np.ascontiguousarray(np.moveaxis(labels, 2, 0)))

        # retain source paths and geometry for restoring predictions
        cases.append({"case_id": case_id, "patient_id": patient_id,
                      "image": str(path.relative_to(ROOT)), "label": str(label_path.relative_to(ROOT)),
                      "geometry": {"original_shape": list(source.shape),
                                   "original_affine": source.affine.tolist()}})
        print(f"[{number}/{len(paths)}] {case_id}: {len(images)} slices", flush=True)
    (PREPROCESSED / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    print(f"Saved preprocessed data to {PREPROCESSED}", flush=True)
