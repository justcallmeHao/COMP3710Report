from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .data2d import OUTPUTS, ROOT, discover, load_slice, write_json
from .model2d import ResidualUNet2D


@torch.inference_mode()
def infer_slice(model, item, device, amp):
    image = torch.from_numpy(np.ascontiguousarray(item["image"]))[None, None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(image)
    return logits.argmax(1)[0].cpu().numpy().astype(np.int16)


def save_prediction(path, labels, case):
    original = nib.load(ROOT / case["image"])
    if len(original.shape) == 3:
        labels = labels[:, :, None]
    header = original.header.copy()
    header.set_data_dtype(np.int16)
    output = nib.Nifti1Image(labels, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    nib.save(output, path)


@torch.inference_mode()
def predict(config):
    settings = config["inference"]
    run_id = config["run_id"]
    run_id = run_id if run_id.endswith("_2d") else f"{run_id}_2d"
    run = OUTPUTS / run_id
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = {**checkpoint["config"], "run_id": run_id,
              "inference": settings, "amp": config["amp"]}

    partition = settings["partition"]
    partition = "validate" if partition == "validation" else partition
    cases = discover(partition)
    directory = run / settings["prediction_id"]
    directory.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7)
    model = ResidualUNet2D(config["model"], in_channels=1, classes=6).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions = []
    for case in cases:
        slice_id = Path(case["image"]).name.removesuffix(".nii.gz")
        patient_id = slice_id.removeprefix("case_").split("_week_", 1)[0]
        labels = infer_slice(model, load_slice(case), device, amp)
        filename = f"{slice_id}-prediction.nii.gz"
        save_prediction(directory / filename, labels, case)
        predictions.append({**case, "case_id": slice_id, "slice_id": slice_id,
                            "patient_id": patient_id, "prediction": filename})

    # record prediction paths and checkpoint
    protocol = {"partition": partition, "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_score": checkpoint["best_score"],
                "prediction_id": settings["prediction_id"]}
    write_json(directory / "predictions.json",
               {"protocol": protocol, "config": config, "cases": predictions})
    return directory
