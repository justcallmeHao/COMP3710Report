"""Predict from image alone; load reference labels only after inference."""
import argparse
import time
from pathlib import Path

# Support both `python predict.py` and `python -m mvp3d.predict`.
if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mvp3d"

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .dataset import Manifest, prepare_image, prepare_label
from .metrics import hard_metrics
from .modules import UNet3D
from .utils import (ROOT, panels, reset_peak, resources, select_device, synchronize,
                    versions, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=["B006_Week0", "B037_Week0", "B038_Week0"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "subset_data/manifest.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest = Manifest(args.manifest)
    if manifest.identity != checkpoint["manifest_sha256"]:
        raise ValueError("Manifest differs from checkpoint.")
    model = UNet3D().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    directory = args.output or ROOT / "outputs/mvp3d" / (
        f"predict-{args.case}-{time.strftime('%Y%m%d-%H%M%S')}")
    directory.mkdir(parents=True, exist_ok=False)
    synchronize(device)
    reset_peak(device)
    start = time.perf_counter()
    image, metadata = prepare_image(manifest, args.case, config)
    with torch.inference_mode():
        prediction = model(image[None].to(device)).argmax(1)[0].cpu()
    target_image = nib.Nifti1Image(
        prediction.numpy().transpose(1, 2, 0).astype(np.uint8),
        np.array(metadata["target_affine"]))
    target_image.header.set_xyzt_units("mm")
    original = resample_from_to(
        target_image, (tuple(metadata["original_shape_hwd"]),
                       np.array(metadata["original_affine"])), order=0, mode="nearest")
    original = nib.Nifti1Image(np.asarray(original.dataobj, dtype=np.uint8),
                              np.array(metadata["original_affine"]))
    original.header.set_xyzt_units("mm")
    synchronize(device)
    elapsed = time.perf_counter() - start
    measured_resources = resources(device)
    # Reference has not been opened anywhere in the inference path above.
    nib.save(target_image, directory / "prediction_target.nii.gz")
    nib.save(original, directory / "prediction_original.nii.gz")
    reference = prepare_label(manifest, args.case, metadata)
    metrics = hard_metrics(prediction, reference)
    panels(directory / "review.png", image[0].numpy(), reference.numpy(), metadata,
           prediction.numpy())
    write_json(directory / "preprocessing.json", metadata)
    write_json(directory / "report.json", {
        "case_id": args.case, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_mode": checkpoint["mode"], "checkpoint_selection": checkpoint["selection"],
        "manifest_sha256": manifest.identity, "config": config,
        "software": versions(device), "inference_precision": "float32",
        "full_case_inference_seconds": elapsed,
        "timing": "Includes image hash/read, preprocessing, transfer, forward, argmax, "
                  "CPU transfer, original-grid remapping and CUDA synchronization. "
                  "Excludes model/checkpoint setup, output writes, reference loading and review.",
        "metric_grid": "target DHW grid", **metrics, **measured_resources,
    })
    print(f"Prediction and review saved to {directory}")


if __name__ == "__main__":
    main()
