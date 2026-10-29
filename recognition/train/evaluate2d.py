import csv
import json

import nibabel as nib
import numpy as np
import torch

from .data2d import OUTPUTS, ROOT, map_labels, write_json

METRICS = ("dice", "iou", "precision", "recall")


def label_tensor(path, config):
    # keep the full volume and map source IDs to class indices
    values = np.asanyarray(nib.load(path).dataobj)
    values = map_labels(values, config["labels"]["id_to_source"])
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))


@torch.no_grad()
def foreground_counts(prediction, target):
    counts = []
    for label in range(1, 6):
        predicted, truth = prediction == label, target == label
        counts.append(torch.stack(((predicted & truth).sum(), predicted.sum(), truth.sum())))
    return torch.stack(counts)


def scores_from_counts(counts):
    intersection, predicted, truth = counts.to(torch.float64).unbind(dim=1)
    numerators = (2 * intersection, intersection, intersection, intersection)
    denominators = (predicted + truth, predicted + truth - intersection, predicted, truth)
    return {metric: torch.where(denominator > 0, numerator / denominator, float("nan"))
            for metric, numerator, denominator in zip(METRICS, numerators, denominators)}


def case_metrics(prediction, target, case, config):
    scores = scores_from_counts(foreground_counts(prediction, target))
    rows = []
    for index in range(5):
        row = {"case_id": case["case_id"], "patient_id": case["patient_id"],
               "slice_id": f"{case['case_id']}-slice-{case['slice_index']}",
               "class_id": index + 1, "class_name": str(index + 1)}
        for metric in METRICS:
            value = scores[metric][index].item()
            available = bool(np.isfinite(value))
            row[metric] = value if available else None
            row[metric + "_valid"] = int(available)
        rows.append(row)
    return rows


def statistics(values):
    valid = [value for value in values if value is not None and np.isfinite(value)]
    return {"mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid)) if valid else None,
            "valid": len(valid), "excluded": len(values) - len(valid)}


def summarize(rows, config):
    patients = sorted({row["patient_id"] for row in rows})
    patient_rows, classes = [], []
    for class_id in range(1, 6):
        name = str(class_id)
        class_patients = []
        for patient in patients:
            # average slices within each patient before averaging patients
            slices = [row for row in rows if row["patient_id"] == patient and row["class_id"] == class_id]
            item = {"patient_id": patient, "class_id": class_id, "class_name": name, "slice_count": len(slices)}
            for metric in METRICS:
                stats = statistics([row[metric] for row in slices])
                item[metric] = stats["mean"]
                item[metric + "_valid_slices"] = stats["valid"]
                item[metric + "_excluded_slices"] = stats["excluded"]
            class_patients.append(item)
        patient_rows.extend(class_patients)
        classes.append({"class_id": class_id, "class_name": name,
                        **{metric: statistics([row[metric] for row in class_patients]) for metric in METRICS}})

    macro = {metric: statistics([row[metric]["mean"] for row in classes])["mean"] for metric in METRICS}
    distributions = [{"patient_id": patient,
                      **{metric: statistics([row[metric] for row in patient_rows
                                             if row["patient_id"] == patient])["mean"] for metric in METRICS}}
                     for patient in patients]
    summary = {"aggregation": "slice -> mean slices per patient/class -> mean patients per class -> equal class mean",
               "patient_count": len(patients),
               "slice_count": len({row["slice_id"] for row in rows}),
               "scan_count": len({row["case_id"] for row in rows}),
               "classes": classes, "foreground_macro": macro, "patient_macro": distributions,
               "patient_macro_distribution": {metric: statistics([row[metric] for row in distributions])
                                               for metric in METRICS}}
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


def evaluate(config):
    run_id = config["run_id"]
    directory = OUTPUTS / run_id / config["inference"]["prediction_id"]
    manifest = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    rows = []
    for case in manifest["cases"]:
        prediction = label_tensor(directory / case["prediction"], config)
        target = label_tensor(ROOT / case["label"], config)
        for item in case["slices"]:
            axis, index = item["slice_axis"], item["slice_index"]
            rows.extend(case_metrics(prediction.select(axis, index), target.select(axis, index),
                                     {**case, **item}, config))
    summary = save_metrics(directory, "evaluation", rows, config)
    print(f"{manifest['protocol']['partition']} foreground macro: {summary['foreground_macro']}", flush=True)
    return summary
