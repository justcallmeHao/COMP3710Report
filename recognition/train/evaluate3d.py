import csv

import nibabel as nib
import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_recall_fscore_support
from torchmetrics.functional.segmentation import dice_score

from .data3d import OUTPUTS, ROOT, map_labels, read_json, write_json

METRICS = ("dice", "precision", "recall", "iou")


def label_tensor(path, config):
    # load source labels and convert them to batched class indices
    values = np.asanyarray(nib.load(path).dataobj)
    return torch.from_numpy(map_labels(values, config["labels"]["id_to_source"])).long().unsqueeze(0)


def case_metrics(prediction, target, case, config):
    # flatten labels and select foreground classes for scoring
    classes = len(config["labels"]["names"])
    pred, truth = prediction.numpy().ravel(), target.numpy().ravel()
    labels = list(range(1, classes))

    # calculate per class metrics
    precision, recall, _, _ = precision_recall_fscore_support(
        truth, pred, labels=labels, average=None, zero_division=np.nan)
    iou = jaccard_score(truth, pred, labels=labels, average=None, zero_division=0)
    dice = dice_score(prediction, target, num_classes=classes, include_background=False,
                      average="none", input_format="index")[0].numpy()

    # count class voxels to identify classes absent from both masks
    pred_counts, target_counts = np.bincount(pred, minlength=classes), np.bincount(truth, minlength=classes)
    results = {"dice": dice, "precision": precision, "recall": recall, "iou": iou}

    # build one metric row for each foreground class
    rows = []
    for index, name in enumerate(config["labels"]["names"][1:]):
        row = {"case_id": case["case_id"], "patient_id": case["patient_id"],
               "class_id": index + 1, "class_name": name}

        # exclude nonfinite scores and classes absent from both masks
        both_empty = pred_counts[index + 1] == 0 and target_counts[index + 1] == 0
        for metric, values in results.items():
            available = bool(np.isfinite(values[index])) and not both_empty
            row[metric] = float(values[index]) if available else None
            row[metric + "_valid"] = int(available)
        rows.append(row)
    return rows


def statistics(values):
    # exclude values before summarising
    valid = [value for value in values if value is not None and np.isfinite(value)]

    # report the mean, spread, and included and excluded counts
    return {"mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid)) if valid else None,
            "valid": len(valid), "excluded": len(values) - len(valid)}


def summarize(rows, config):
    patients = sorted({row["patient_id"] for row in rows})
    patient_rows, classes = [], []

    # aggregate each foreground class across scans and patients
    for class_id, name in enumerate(config["labels"]["names"][1:], start=1):
        # average repeated scans within each patient
        class_patients = []
        for patient in patients:
            scans = [row for row in rows if row["patient_id"] == patient and row["class_id"] == class_id]
            item = {"patient_id": patient, "class_id": class_id, "class_name": name, "scan_count": len(scans)}

            # record each metric mean
            for metric in METRICS:
                stats = statistics([row[metric] for row in scans])
                item[metric] = stats["mean"]
                item[metric + "_valid_scans"] = stats["valid"]
                item[metric + "_excluded_scans"] = stats["excluded"]
            class_patients.append(item)

        # summarise patient scores with weight per patient
        patient_rows.extend(class_patients)
        classes.append({"class_id": class_id, "class_name": name,
                        **{metric: statistics([row[metric] for row in class_patients]) for metric in METRICS}})

    # average class means
    macro = {metric: statistics([row[metric]["mean"] for row in classes])["mean"] for metric in METRICS}

    # calculate each patient score
    distributions = [{"patient_id": patient,
                      **{metric: statistics([row[metric] for row in patient_rows
                                             if row["patient_id"] == patient])["mean"] for metric in METRICS}}
                     for patient in patients]

    # record summary statistics, class availability, and target status
    available = all(row["dice"]["valid"] > 0 for row in classes)
    summary = {"aggregation": "full scan -> mean visits per patient/class -> mean patients per class -> equal class mean",
               "patient_count": len(patients), "scan_count": len({r["case_id"] for r in rows}),
               "classes": classes, "foreground_macro": macro, "patient_macro": distributions,
               "patient_macro_distribution": {m: statistics([r[m] for r in distributions]) for m in METRICS},
               "all_dice_classes_available": available,
               "dice_target": config["evaluation"]["dice_target"],
               "dice_target_met": bool(available and macro["dice"] is not None
                                       and macro["dice"] > config["evaluation"]["dice_target"])}
    
    return patient_rows, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics(directory, prefix, rows, config):
    patient_rows, summary = summarize(rows, config)
    write_csv(directory / f"{prefix}-cases.csv", rows)
    write_json(directory / f"{prefix}-cases.json", rows)
    write_csv(directory / f"{prefix}-patients.csv", patient_rows)
    write_json(directory / f"{prefix}-summary.json", summary)
    return summary


def evaluate(requested):
    # load the saved prediction manifest
    directory = OUTPUTS / requested["run_id"] / requested["inference"]["prediction_id"]
    manifest = read_json(directory / "predictions.json")
    config, rows = manifest["config"], []

    # compare each saved prediction with its reference label
    for case in manifest["cases"]:
        prediction_path, label_path = directory / case["prediction"], ROOT / case["label"]
        rows.extend(case_metrics(label_tensor(prediction_path, config), label_tensor(label_path, config), case, config))

    # save evaluation metrics and report the foreground averages
    summary = save_metrics(directory, "evaluation", rows, config)
    print(f"{manifest['protocol']['partition']} foreground macro: {summary['foreground_macro']}")

