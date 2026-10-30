import csv
import importlib.metadata
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data2d import OUTPUTS, ROOT, SliceDataset, discover, load_slice, read_json, write_json
from .evaluate2d import METRICS, case_metrics, save_metrics
from .model2d import UNet2D
from .predict2d import slice_logits


def versions(device):
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "scipy", "numpy", "nibabel")},
            "device": str(device), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}


def segmentation_loss(logits, target, settings):
    target = target[:, 0].long()
    valid = (target != -100).unsqueeze(1)
    probabilities = logits.float().softmax(1) * valid
    truth = F.one_hot(target.clamp_min(0), num_classes=logits.shape[1]).movedim(-1, 1).float() * valid

    intersection = (probabilities * truth).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + truth.sum(dim=(2, 3))
    epsilon = settings["probability_floor"]
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    if not settings["include_background"]:
        dice = dice[:, 1:]
    ce = F.cross_entropy(logits.float(), target, ignore_index=-100)
    dice = 1 - dice.mean()
    total = settings["lambda_dice"] * dice + settings["lambda_ce"] * ce
    return total, dice, ce


def train_epoch(model, loader, loss_settings, optimizer, scaler, device, amp):
    model.train()
    totals = np.zeros(3)
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["label"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
        values = segmentation_loss(logits, target, loss_settings)
        scaler.scale(values[0]).backward()
        scaler.step(optimizer)
        scaler.update()
        totals += np.asarray([value.detach().item() for value in values]) * len(image)
        count += len(image)
    return (totals / count).tolist()


@torch.inference_mode()
def validate(model, cases, config, device, amp):
    model.eval()
    rows = []
    totals = np.zeros(3)
    count = 0
    for case in cases:
        item = load_slice(case)
        truth = torch.from_numpy(np.array(item["label"], dtype=np.int64, copy=True))[None, None].to(device)
        logits = slice_logits(model, item, device, amp)
        values = segmentation_loss(logits, truth, config["loss"])
        totals += np.asarray([value.item() for value in values])
        labels = logits.argmax(1)[0].cpu()
        rows.extend(case_metrics(labels, truth[0, 0].cpu(), case, config))
        count += 1
    return (totals / count).tolist(), rows


def run_training(config, training_cases, validation_cases, directory, device, amp):
    settings = config["training"]
    id_to_source = config["labels"]["id_to_source"]
    training = DataLoader(SliceDataset(training_cases, config["model"]), batch_size=settings["batch_size"],
                          shuffle=True, pin_memory=device.type == "cuda")

    model = UNet2D(config["model"], in_channels=1, classes=len(id_to_source)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    write_json(directory / "resolved-config.json", config)
    runtime = versions(device)
    runtime["amp"] = amp
    runtime["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    runtime["datasets"] = {
        partition: {"slices": len(cases), "scans": len({case["case_id"] for case in cases}),
                    "patients": len({case["patient_id"] for case in cases})}
        for partition, cases in (("train", training_cases), ("validation", validation_cases))}
    write_json(directory / "runtime.json", runtime)
    print(f"Run {config['run_id']}: device={device}, GPU={runtime['gpu']}, AMP={amp}, "
          f"parameters={runtime['parameters']:,}", flush=True)
    print(f"Datasets: {runtime['datasets']}; epochs={settings['epochs']}, "
          f"batch={settings['batch_size']}, lr={settings['learning_rate']}", flush=True)
    epoch_log = directory / "epochs.csv"
    best_score = -1.0
    target_count = 0
    for epoch in range(1, settings["epochs"] + 1):
        began = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_epoch(model, training, config["loss"], optimizer, scaler, device, amp)
        validation_losses, metrics, improved = [None] * 3, dict.fromkeys(METRICS), False
        if epoch % settings.get("validation_interval", 1) == 0 or epoch == settings["epochs"]:
            validation_losses, validation_rows = validate(model, validation_cases, config, device, amp)
            summary = save_metrics(directory, "validation", validation_rows, config)
            metrics = summary["foreground_macro"]
            target_count = target_count + 1 if metrics["dice"] is not None and metrics["dice"] > 0.70 else 0
            improved = metrics["dice"] is not None and metrics["dice"] > best_score
            if improved:
                best_score = metrics["dice"]
                save_metrics(directory, "best", validation_rows, config)
        scheduler.step()

        # keep checkpoint fields compatible with prediction
        state = {"model": model.state_dict(), "epoch": epoch, "best_score": best_score,
                 "config": config}
        if improved:
            torch.save(state, directory / "best.pt")

        row = {"epoch": epoch, "learning_rate": learning_rate, "seconds": time.perf_counter() - began,
               **{f"train_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), train_losses)},
               **{f"validation_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), validation_losses)},
               **{f"validation_{metric}": metrics[metric] for metric in METRICS}}
        with epoch_log.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch}/{settings['epochs']}: train={train_losses[0]:.4f}, "
              f"validation={validation_losses[0]}, Dice={metrics['dice']}, "
              f"lr={learning_rate:.6g}, seconds={row['seconds']:.1f}", flush=True)
        if target_count >= 5:
            print("stopped after 5 consecutive validations with Dice > 0.70.", flush=True)
            break
    return directory


def train(config):
    run_id = config["run_id"]
    config = {**config, "run_id": run_id, "labels": read_json(ROOT / "configs/labels.json")}
    directory = OUTPUTS / run_id
    directory.mkdir(parents=True, exist_ok=False)
    training_cases, validation_cases = discover("train"), discover("validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7)
    return run_training(config, training_cases, validation_cases, directory, device, amp)
