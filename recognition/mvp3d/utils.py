import csv
import json
import platform
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def select_device(name, amp=False):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly.")
        torch.cuda.set_device(device)
    if amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA.")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def resources(device):
    result = {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
              "cuda_device_used_bytes_at_sample": None}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      cuda_device_used_bytes_at_sample=total - free)
    return result


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def versions(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "nibabel": nibabel.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "cuda_build": torch.version.cuda, "device": str(device),
            "device_name": (torch.cuda.get_device_name(device)
                            if device.type == "cuda" else platform.processor())}


def panels(path, image, reference, metadata, prediction=None):
    """Three evenly spaced array-axis slices; class IDs are not anatomical names."""
    image, reference = np.asarray(image), np.asarray(reference)
    depth = image.shape[0]
    slices = sorted({depth // 4, depth // 2, 3 * depth // 4})
    columns = 3 if prediction is None else 5
    fig, axes = plt.subplots(len(slices), columns, figsize=(4 * columns, 3 * len(slices)),
                             squeeze=False, layout="constrained")
    spacing = metadata["target_spacing_hwd_mm"]
    aspect = spacing[0] / spacing[1]
    for row, z in enumerate(slices):
        for ax in axes[row]:
            ax.set_axis_off()
        axes[row, 0].imshow(image[z], cmap="gray", aspect=aspect)
        axes[row, 0].set_title(f"{metadata['case_id']} | D={z} | MRI")
        axes[row, 1].imshow(reference[z], cmap="tab10", vmin=0, vmax=9, aspect=aspect,
                            interpolation="nearest")
        axes[row, 1].set_title("Reference IDs 0-5")
        axes[row, 2].imshow(image[z], cmap="gray", aspect=aspect)
        axes[row, 2].imshow(np.ma.masked_equal(reference[z], 0), cmap="tab10",
                            vmin=0, vmax=9, alpha=0.45, aspect=aspect,
                            interpolation="nearest")
        axes[row, 2].set_title("Reference alignment")
        if prediction is not None:
            pred, truth = prediction[z], reference[z]
            axes[row, 3].imshow(pred, cmap="tab10", vmin=0, vmax=9, aspect=aspect,
                                interpolation="nearest")
            axes[row, 3].set_title("Prediction IDs 0-5")
            difference = np.zeros((*pred.shape, 3), dtype=float)
            difference[..., 0] = (pred != truth) & (pred != 0)
            difference[..., 2] = (pred != truth) & (truth != 0)
            axes[row, 4].imshow(difference, aspect=aspect, interpolation="nearest")
            axes[row, 4].set_title("FP red / FN blue / wrong class magenta")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_logs(directory, rows):
    write_json(directory / "metrics.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for phase in sorted({row["phase"] for row in rows}):
        values = [row for row in rows if row["phase"] == phase]
        axes[0].plot([r["step"] for r in values], [r["loss"] for r in values],
                     marker=".", label=phase)
        axes[1].plot([r["step"] for r in values],
                     [r["foreground_macro_dice"] for r in values], marker=".", label=phase)
    axes[0].set_ylabel("Combined loss")
    axes[1].set_ylabel("Hard foreground macro Dice")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.legend()
    fig.savefig(directory / "learning_curves.png", dpi=120)
    plt.close(fig)
