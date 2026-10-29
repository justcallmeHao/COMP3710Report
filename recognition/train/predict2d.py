import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .data2d import OUTPUTS, ROOT, discover, load_slice, write_json
from .model2d import ResidualUNet2D


@torch.inference_mode()
def infer_slice(model, item, device, amp):
    image = torch.from_numpy(np.array(item["image"], order="C", copy=True))[None, None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(image)
    return logits.argmax(1)[0].cpu().numpy().astype(np.int16)


def save_prediction(path, labels, case):
    original = nib.load(ROOT / case["image"])
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
    partition = "validation" if partition == "validate" else partition
    cases = discover(partition)
    directory = run / settings["prediction_id"]
    directory.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7)
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    print(f"Device: {name}; AMP: {amp}", flush=True)
    id_to_source = config["labels"]["id_to_source"]
    model = ResidualUNet2D(config["model"], in_channels=1, classes=len(id_to_source)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scans = {}
    for case in cases:
        scans.setdefault(case["case_id"], []).append(case)

    predictions = []
    inverse = np.asarray(id_to_source, dtype=np.int16)
    for scan in scans.values():
        case = scan[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        labels = np.empty(case["geometry"]["original_shape"], dtype=np.int16)
        # restore each slice at its original axis and index using source label IDs
        for item in scan:
            prediction = infer_slice(model, load_slice(item, with_label=False), device, amp)
            region = [slice(None)] * 3
            region[item["slice_axis"]] = item["slice_index"]
            labels[tuple(region)] = inverse[prediction]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - began
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        stem = Path(case["image"]).name.removesuffix(".gz").removesuffix(".nii")
        filename = f"{stem}-prediction.nii.gz"
        save_prediction(directory / filename, labels, case)
        predictions.append({"case_id": case["case_id"], "patient_id": case["patient_id"],
                            "image": case["image"], "label": case["label"],
                            "geometry": case["geometry"], "prediction": filename,
                            "slices": [{"slice_id": f"{item['case_id']}-slice-{item['slice_index']}",
                                        "slice_axis": item["slice_axis"], "slice_index": item["slice_index"]}
                                       for item in scan],
                            "inference_seconds": seconds, "peak_gpu_allocated_mib": peak_gpu})

    # record prediction paths and checkpoint
    protocol = {"partition": partition, "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_score": checkpoint["best_score"],
                "prediction_id": settings["prediction_id"]}
    write_json(directory / "predictions.json",
               {"protocol": protocol, "config": config, "cases": predictions})
    print(f"Saved {len(predictions)} original-geometry predictions to {directory}", flush=True)
    return directory
