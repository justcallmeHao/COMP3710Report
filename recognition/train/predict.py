import importlib.metadata
import json
import platform
import random
from itertools import product

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.nn import functional as F

from .data import OUTPUTS, ROOT, load_volume, select_cases, write_json
from .evaluate import case_metrics, label_tensor
from .model import ResidualUNet3D


def select_device(use_amp):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    amp = bool(use_amp and cuda and torch.cuda.get_device_capability()[0] >= 7)
    name = torch.cuda.get_device_name() if cuda else "CPU fallback: CUDA unavailable"
    print(f"Device: {name}; AMP: {amp}", flush=True)
    return device, amp


def seed_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def versions(device):
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchmetrics", "scipy", "numpy", "nibabel", "scikit-learn", "matplotlib")},
            "device": str(device), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}


def window_starts(size, patch, overlap):
    step = max(1, int(patch * (1 - overlap)))
    return sorted(set([*range(0, size - patch + 1, step), size - patch]))


def restore_probabilities(probabilities, geometry):
    region = tuple(slice(a, b) for a, b in zip(geometry["crop_start"], geometry["crop_end"]))
    target = (tuple(geometry["original_shape"]), np.asarray(geometry["original_affine"]))
    restored = []
    for channel in probabilities.numpy():
        full = np.zeros(geometry["grid_shape"], dtype=np.float32)
        full[region] = channel
        volume = nib.Nifti1Image(full, np.asarray(geometry["grid_affine"]))
        restored.append(resample_from_to(volume, target, order=1).get_fdata(dtype=np.float32))
    return torch.from_numpy(np.stack(restored))


@torch.inference_mode()
def infer_volume(model, item, config, device, amp):
    settings = config["inference"]
    patch = settings["patch_size"]
    image = torch.from_numpy(item["image"]).unsqueeze(0)
    original_shape = image.shape[1:]
    padding = [value for size, length in reversed(list(zip(original_shape, patch)))
               for value in (0, max(0, length - size))]
    image = F.pad(image, padding)
    shape = image.shape[1:]
    coordinates = list(product(*(window_starts(size, length, settings["overlap"])
                                 for size, length in zip(shape, patch))))
    axes = [torch.arange(length, dtype=torch.float32) - (length - 1) / 2 for length in patch]
    grid = torch.meshgrid(*axes, indexing="ij")
    exponent = sum((axis / (length * settings["sigma_scale"])) ** 2 for axis, length in zip(grid, patch))
    weight = torch.exp(-0.5 * exponent).clamp_min(settings["weight_floor"])
    logits = torch.zeros((len(config["labels"]["names"]), *shape), dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)
    for offset in range(0, len(coordinates), settings["batch_size"]):
        locations = coordinates[offset:offset + settings["batch_size"]]
        regions = [tuple(slice(start, start + length) for start, length in zip(point, patch)) for point in locations]
        batch = torch.stack([image[(slice(None), *region)] for region in regions]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(batch)
        for output, region in zip(outputs.float().cpu(), regions):
            logits[(slice(None), *region)] += output * weight
            weights[region] += weight
    logits /= weights
    region = (slice(None), *(slice(0, size) for size in original_shape))
    logits = logits[region]
    probabilities = restore_probabilities(logits.softmax(0), item["geometry"])
    return logits.unsqueeze(0), probabilities


def save_prediction(path, labels, case, config):
    original = nib.load(ROOT / case["image"])
    inverse = np.asarray(config["labels"]["id_to_source"], dtype=np.int16)
    source_labels = inverse[labels.numpy()]
    header = original.header.copy()
    header.set_data_dtype(np.int16)
    output = nib.Nifti1Image(source_labels, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    nib.save(output, path)


def predict(requested, evaluate_results=False):
    settings = requested["inference"]
    run = OUTPUTS / requested["run_id"]
    checkpoint_path = run / f"{settings['checkpoint']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["inference"] = settings
    config["amp"] = requested["amp"]
    manifest, cases = checkpoint["splits"], checkpoint["cases"]
    partition = settings["partition"]
    patients = manifest[partition]
    selected = select_cases(cases, patients)
    if partition == "validation" and settings["case_ids"]:
        by_id = {case["case_id"]: case for case in selected}
        selected = [by_id[case_id] for case_id in dict.fromkeys(settings["case_ids"])]
    directory = run / settings["prediction_id"]
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"partition": partition, "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_score": checkpoint["best_score"],
                "prediction_id": settings["prediction_id"]}
    if partition == "test":
        with (run / "test-protocol.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(protocol, indent=2))
    device, amp = select_device(config["amp"])
    seed_state(config["seed"])
    model = ResidualUNet3D(config["model"], len(config["data"]["channel_names"]), len(config["labels"]["names"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, rows = [], []
    for case in selected:
        item = load_volume(case, config["preprocessing"], config["labels"]["id_to_source"], with_label=False)
        _, probabilities = infer_volume(model, item, config, device, amp)
        filename = f"{case['case_id']}-prediction.nii.gz"
        labels = probabilities.argmax(0)
        save_prediction(directory / filename, labels, case, config)
        if evaluate_results:
            rows.extend(case_metrics(labels.unsqueeze(0), label_tensor(ROOT / case["label"], config), case, config))
        predictions.append({**case, "prediction": filename})
    manifest = {"protocol": protocol, "config": config, "versions": versions(device), "cases": predictions}
    write_json(directory / "predictions.json", manifest)
    print(f"Saved {len(predictions)} original-geometry predictions to {directory}")

    return manifest, rows
