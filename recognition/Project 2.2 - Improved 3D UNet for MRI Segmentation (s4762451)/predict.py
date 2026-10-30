import importlib.metadata
import platform
import time
from itertools import product

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.nn import functional as F

from dataset import OUTPUTS, ROOT, discover, load_volume, select_cases, write_json
from modules import ResidualUNet3D


def select_device(use_amp):
    """select the compute device and supported mixed precision mode"""
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    amp = bool(use_amp and cuda and torch.cuda.get_device_capability()[0] >= 7)
    name = torch.cuda.get_device_name() if cuda else "CUDA unavailable"
    print(f"Device: {name}; AMP: {amp}", flush=True)
    return device, amp


def versions(device):
    """collect package versions and compute device details"""
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchmetrics", "scipy", "numpy", "nibabel", "scikit-learn", "matplotlib")},
            "device": str(device), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}


def window_starts(size, patch, overlap):
    # space windows by the overlap and include the final boundary
    """return overlapping window positions including the final boundary"""
    step = max(1, int(patch * (1 - overlap)))
    return sorted(set([*range(0, size - patch + 1, step), size - patch]))


def restore_probabilities(probabilities, geometry):
    # recover the crop bounds and original scan geometry
    """restore class probabilities to the original scan geometry"""
    region = tuple(slice(a, b) for a, b in zip(geometry["crop_start"], geometry["crop_end"]))
    target = (tuple(geometry["original_shape"]), np.asarray(geometry["original_affine"]))

    # place each class in the full grid and resample to the original scan
    restored = []
    for channel in probabilities.numpy():
        full = np.zeros(geometry["grid_shape"], dtype=np.float32)
        full[region] = channel
        volume = nib.Nifti1Image(full, np.asarray(geometry["grid_affine"]))
        restored.append(resample_from_to(volume, target, order=1).get_fdata(dtype=np.float32))

    # stack restored class probabilities into one tensor
    return torch.from_numpy(np.stack(restored))


@torch.inference_mode()
def infer_volume(model, item, config, device, amp):
    """blend sliding window predictions and restore probabilities to the original grid"""
    settings = config["inference"]
    patch = settings["patch_size"]
    image = torch.from_numpy(item["image"]).unsqueeze(0)
    original_shape = image.shape[1:]

    # pad small volumes so each axis fits at least one window
    padding = [value for size, length in reversed(list(zip(original_shape, patch)))
               for value in (0, max(0, length - size))]
    image = F.pad(image, padding)
    shape = image.shape[1:]

    # make sliding window positions across the padded volume
    coordinates = list(product(*(window_starts(size, length, settings["overlap"])
                                 for size, length in zip(shape, patch))))

    # build gaussian weights
    axes = [torch.arange(length, dtype=torch.float32) - (length - 1) / 2 for length in patch]
    grid = torch.meshgrid(*axes, indexing="ij")
    exponent = sum((axis / (length * settings["sigma_scale"])) ** 2 for axis, length in zip(grid, patch))
    weight = torch.exp(-0.5 * exponent).clamp_min(settings["weight_floor"])

    logits = torch.zeros((len(config["labels"]["names"]), *shape), dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    # process sliding windows in batches
    for offset in range(0, len(coordinates), settings["batch_size"]):
        # extract the current windows and move them to the device
        locations = coordinates[offset:offset + settings["batch_size"]]
        regions = [tuple(slice(start, start + length) for start, length in zip(point, patch)) for point in locations]
        batch = torch.stack([image[(slice(None), *region)] for region in regions]).to(device)

        # predict window logits with optional mixed precision
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(batch)

        # blend overlapping window logits with gaussian weights
        for output, region in zip(outputs.float().cpu(), regions):
            logits[(slice(None), *region)] += output * weight
            weights[region] += weight

    # average overlapping predictions and remove padding
    logits /= weights
    region = (slice(None), *(slice(0, size) for size in original_shape))
    logits = logits[region]

    # restore class probabilities to the original scan geometry
    probabilities = restore_probabilities(logits.softmax(0), item["geometry"])
    return logits.unsqueeze(0), probabilities


def save_prediction(path, labels, case, config):
    """map class indices to source labels and save them with the original scan geometry"""
    original = nib.load(ROOT / case["image"])
    inverse = np.asarray(config["labels"]["id_to_source"], dtype=np.int16)
    source_labels = inverse[labels.numpy()]

    header = original.header.copy()
    header.set_data_dtype(np.int16)
    output = nib.Nifti1Image(source_labels, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    nib.save(output, path)


def predict(requested):
    # load the selected model checkpoint
    """load a checkpoint and save predictions for the requested patient split"""
    settings = requested["inference"]
    run = OUTPUTS / requested["run_id"]
    checkpoint_path = run / f"{settings['checkpoint']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # reuse training settings with the requested inference options
    config = checkpoint["config"]
    config["inference"] = settings
    config["amp"] = requested["amp"]

    # select scans from the requested patient partition
    manifest, cases = checkpoint["splits"], discover(config)
    partition = settings["partition"]
    patients = manifest[partition]
    selected = select_cases(cases, patients)

    # create the prediction directory and record checkpoint details
    directory = run / settings["prediction_id"]
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"partition": partition, "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_score": checkpoint["best_score"],
                "prediction_id": settings["prediction_id"]}

    # restore model weights and enable evaluation mode
    device, amp = select_device(config["amp"])
    model = ResidualUNet3D(config["model"], len(config["data"]["channel_names"]), len(config["labels"]["names"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # collect saved prediction paths
    predictions = []
    measure_time = settings["partition"] == "test"
    for case in selected:
        # reset GPU usage checker to measure for this case
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        # load each cached scan and predict class probabilities
        started = time.perf_counter() if measure_time else None
        item = load_volume(case, with_label=False)
        _, probabilities = infer_volume(model, item, config, device, amp)

        # save the most likely class at each voxel
        filename = f"{case['case_id']}-prediction.nii.gz"
        labels = probabilities.argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started if measure_time else None
        save_prediction(directory / filename, labels, case, config)

        # GPU usage
        peak_gpu = (torch.cuda.max_memory_allocated(device) / 1024**2
                            if device.type == "cuda" else None)

        # record the saved prediction path for this case
        predictions.append({**case, "prediction": filename,
                            "peak_gpu_allocated_mib": peak_gpu})
        if measure_time:
            predictions[-1]["inference_seconds"] = seconds

    # save metadata for evaluation and plotting
    manifest = {"protocol": protocol, "config": config, "versions": versions(device), "cases": predictions}
    write_json(directory / "predictions.json", manifest)
    print(f"Saved {len(predictions)} original-geometry predictions to {directory}")


def main():
    """predict the configured split, evaluate results, and save figures"""
    from dataset import read_json
    from graphs.plot import plot
    from helpers.evaluate3d import evaluate

    config = read_json(ROOT / "configs/training3d.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    predict(config)
    evaluate(config)
    plot(config, include_training=False)


if __name__ == "__main__":
    main()
