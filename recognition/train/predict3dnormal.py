from itertools import product
import time

import torch

from .data3d import OUTPUTS, discover, load_volume, select_cases, write_json
from .data3dnormal import pad_image
from .model3dnormal import UNet3D
from .predict3d import restore_probabilities, save_prediction, select_device, versions, window_starts


@torch.inference_mode()
def infer_volume(model, item, config, device, amp):
    settings = config["inference"]
    patch, output_size = settings["patch_size"], settings["output_patch_size"]
    original_shape = item["image"].shape
    shape = tuple(max(size, length) for size, length in zip(original_shape, output_size))
    # transfer once, assemble windows and blend predictions on the inference device
    image = torch.from_numpy(pad_image(item["image"], patch, output_size)).unsqueeze(0).to(device)
    # Positions and Gaussian weights describe output regions, not input tiles.
    coordinates = list(product(*(window_starts(size, length, settings["overlap"])
                                 for size, length in zip(shape, output_size))))
    axes = [torch.arange(length, dtype=torch.float32, device=device) - (length - 1) / 2
            for length in output_size]
    grid = torch.meshgrid(*axes, indexing="ij")
    exponent = sum((axis / (length * settings["sigma_scale"])) ** 2
                   for axis, length in zip(grid, output_size))
    weight = torch.exp(-0.5 * exponent).clamp_min(settings["weight_floor"])
    logits = torch.zeros((len(config["labels"]["names"]), *shape), dtype=torch.float32, device=device)
    weights = torch.zeros(shape, dtype=torch.float32, device=device)

    for offset in range(0, len(coordinates), settings["batch_size"]):
        locations = coordinates[offset:offset + settings["batch_size"]]
        regions = [tuple(slice(start, start + size) for start, size in zip(point, patch))
                   for point in locations]
        batch = torch.stack([image[(slice(None), *region)] for region in regions])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(batch)
        for output, point in zip(outputs.float(), locations):
            region = tuple(slice(start, start + size) for start, size in zip(point, output_size))
            logits[(slice(None), *region)] += output * weight
            weights[region] += weight

    logits /= weights

    # relieve workload onto cpu
    logits = logits[(slice(None), *(slice(0, size) for size in original_shape))].cpu()
    probabilities = restore_probabilities(logits.softmax(0), item["geometry"])
    return logits.unsqueeze(0), probabilities


def predict(requested):
    settings = requested["inference"]
    run = OUTPUTS / requested["run_id"]
    checkpoint_path = run / f"{settings['checkpoint']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = {**checkpoint["config"], "inference": settings, "amp": requested["amp"]}
    selected = select_cases(discover(config), checkpoint["splits"][settings["partition"]])
    directory = run / settings["prediction_id"]
    directory.mkdir(parents=True, exist_ok=False)
    device, amp = select_device(config["amp"])
    model = UNet3D(config["model"], len(config["data"]["channel_names"]), len(config["labels"]["names"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions = []
    measure_time = settings["partition"] == "test"
    for case in selected:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter() if measure_time else None
        _, probabilities = infer_volume(model, load_volume(case, with_label=False), config, device, amp)
        filename = f"{case['case_id']}-prediction.nii.gz"
        labels = probabilities.argmax(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started if measure_time else None
        save_prediction(directory / filename, labels, case, config)
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        predictions.append({**case, "prediction": filename, "peak_gpu_allocated_mib": peak_gpu})
        if measure_time:
            predictions[-1]["inference_seconds"] = seconds

    protocol = {"partition": settings["partition"], "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_score": checkpoint["best_score"],
                "prediction_id": settings["prediction_id"]}
    write_json(directory / "predictions.json",
               {"protocol": protocol, "config": config, "versions": versions(device), "cases": predictions})
    print(f"Saved {len(predictions)} original-geometry predictions to {directory}")
