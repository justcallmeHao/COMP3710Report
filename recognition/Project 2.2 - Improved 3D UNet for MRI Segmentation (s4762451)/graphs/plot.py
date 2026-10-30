import csv
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import OUTPUTS, ROOT, map_labels, read_json


def csv_rows(path):
    """read CSV rows as dictionaries"""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    """extract a numeric column and replace empty values with NaN"""
    return np.asarray([float(row[name]) if row[name] else np.nan for row in rows])


def learning_curves(run, destination, settings):
    """save plots of losses, validation scores, learning rates, and epoch times"""
    rows = csv_rows(run / "epochs.csv")
    epochs = column(rows, "epoch")
    fig, axes = plt.subplots(2, 3, figsize=settings["figure_size"], layout="constrained")
    for axis, metric, title in zip(axes[0], ("loss", "dice_loss", "ce_loss"),
                                   ("Total Dice + CE loss", "Dice loss component", "CE loss component")):
        for phase in ("train", "validation"):
            values = column(rows, f"{phase}_{metric}")
            valid = np.isfinite(values)
            axis.plot(epochs[valid], values[valid], label=phase,
                      marker="o" if phase == "validation" else None)
        axis.set(xlabel="Epoch", ylabel=title)
        axis.legend()
    for metric in ("dice", "precision", "recall", "iou"):
        values = column(rows, f"validation_{metric}")
        valid = np.isfinite(values)
        axes[1, 0].plot(epochs[valid], values[valid], marker="o", label=metric)
    axes[1, 0].set(xlabel="Epoch", ylabel="Validation foreground macro", ylim=(0, 1))
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, column(rows, "learning_rate"))
    axes[1, 1].set(xlabel="Epoch", ylabel="Learning rate")
    axes[1, 2].plot(epochs, column(rows, "seconds"))
    axes[1, 2].set(xlabel="Epoch", ylabel="Epoch seconds")
    fig.savefig(destination / "learning.png", dpi=settings["dpi"])
    plt.close(fig)


def score_boxes(path, destination, settings):
    """save per-class metric box plots with individual scores"""
    rows = csv_rows(path)
    names = list(dict.fromkeys(row["class_name"] for row in rows))
    random = np.random.default_rng()
    fig, axes = plt.subplots(2, 2, figsize=settings["figure_size"], layout="constrained")
    for axis, metric in zip(axes.flat, ("dice", "precision", "recall", "iou")):
        values = [column([r for r in rows if r["class_name"] == name], metric) for name in names]
        values = [v[np.isfinite(v)] for v in values]
        axis.boxplot(values, tick_labels=names, showfliers=False)
        for index, points in enumerate(values, start=1):
            axis.scatter(index + random.uniform(-settings["jitter"], settings["jitter"], len(points)),
                         points, s=settings["point_size"])
        axis.set(ylabel=metric, ylim=(0, 1))
        axis.tick_params(axis="x", rotation=35)
    fig.savefig(destination / (path.stem + "-scores.png"), dpi=settings["dpi"])
    plt.close(fig)


def overlays(prediction_dir, destination, settings):
    """save scan views with reference labels, predictions, and errors"""
    manifest = read_json(prediction_dir / "predictions.json")
    config = manifest["config"]
    cases = manifest["cases"]
    if settings["overlay_cases"]:
        cases = [case for case in cases if case["case_id"] in settings["overlay_cases"]]
    for case in cases[:settings["max_overlay_cases"]]:
        image = nib.as_closest_canonical(nib.load(ROOT / case["image"]))
        truth = nib.as_closest_canonical(nib.load(ROOT / case["label"]))
        prediction = nib.as_closest_canonical(nib.load(prediction_dir / case["prediction"]))
        arrays = [np.asanyarray(volume.dataobj) for volume in (image, truth, prediction)]
        arrays[1:] = [map_labels(array, config["labels"]["id_to_source"]) for array in arrays[1:]]
        fig, axes = plt.subplots(3, 4, figsize=settings["figure_size"], layout="constrained")
        for axis, name in enumerate(("sagittal", "coronal", "axial")):
            index = min(int(image.shape[axis] * settings["slice_fraction"]), image.shape[axis] - 1)
            mri, target, pred = [np.take(array, index, axis=axis).T for array in arrays]
            spacing = np.delete(image.header.get_zooms(), axis)
            aspect = spacing[1] / spacing[0]
            for panel in axes[axis]:
                panel.imshow(mri, cmap=settings["image_cmap"], origin="lower", aspect=aspect)
                panel.set_axis_off()
            for panel, mask in zip(axes[axis, 1:3], (target, pred)):
                panel.imshow(np.ma.masked_equal(mask, 0), cmap=settings["label_cmap"],
                             vmin=0, vmax=len(config["labels"]["names"]) - 1, alpha=settings["alpha"],
                             interpolation="nearest", origin="lower", aspect=aspect)
            difference = np.zeros((*pred.shape, 4))
            difference[..., 0] = (pred != target) & (pred != 0)
            difference[..., 2] = (pred != target) & (target != 0)
            difference[..., 3] = (pred != target) * settings["alpha"]
            axes[axis, 3].imshow(difference, origin="lower", aspect=aspect, interpolation="nearest")
            for panel, title in zip(axes[axis], ("MRI", "Ground truth", "Prediction", "FP red / FN blue / wrong class magenta")):
                panel.set_title(f"{name} {index}: {title}")
        fig.suptitle(case["case_id"])
        fig.savefig(destination / f"{case['case_id']}-overlay.png", dpi=settings["dpi"])
        plt.close(fig)


def inference_times(prediction_dir, destination, settings):
    """save a plot of recorded inference times by case"""
    manifest = read_json(prediction_dir / "predictions.json")
    cases = [case for case in manifest["cases"]
             if case.get("inference_seconds") is not None]
    if not cases:
        return  # Older prediction files do not contain timing measurements.
    fig, axis = plt.subplots(figsize=settings["figure_size"], layout="constrained")
    axis.plot(range(1, len(cases) + 1), [case["inference_seconds"] for case in cases],
              marker="o", markersize=4, linewidth=1.5)
    axis.set(xlabel="Test case index (timed cases in predictions.json order)",
             ylabel="Inference time (seconds)")
    fig.savefig(destination / "inference-times.png", dpi=settings["dpi"])
    plt.close(fig)


def plot(config, include_training=True):
    """save the configured training and prediction figures"""
    settings = config["graphs"]
    run = OUTPUTS / config["run_id"]
    destination = run / settings["figure_id"]
    destination.mkdir(exist_ok=False)
    if include_training:
        learning_curves(run, destination, settings)
        score_boxes(run / "best-patients.csv", destination, settings)
    prediction_dir = run / config["inference"]["prediction_id"]
    if (prediction_dir / "predictions.json").exists():
        overlays(prediction_dir, destination, settings)
        if config["inference"]["partition"] == "test":
            inference_times(prediction_dir, destination, settings)
    if (prediction_dir / "evaluation-patients.csv").exists():
        score_boxes(prediction_dir / "evaluation-patients.csv", destination, settings)
    print(f"Saved figures to {destination}")

