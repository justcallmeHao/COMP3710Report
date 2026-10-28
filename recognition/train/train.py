import csv
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional.segmentation import dice_score

from .data import OUTPUTS, ROOT, PatchDataset, discover, load_volume, read_json, resolve_spacing, select_cases, write_json
from .evaluate import METRICS, case_metrics, label_tensor, save_metrics
from .model import ResidualUNet3D
from .predict import infer_volume, select_device, versions


def segmentation_loss(logits, target, settings):
    target = target[:, 0].long()
    classes = logits.shape[1]
    probabilities = logits.float().softmax(1).clamp_min(settings["probability_floor"])
    truth = F.one_hot(target, num_classes=classes).movedim(-1, 1).float()
    dice = 1 - dice_score(probabilities, truth, num_classes=classes,
                          include_background=settings["include_background"],
                          average="macro", input_format="one-hot").mean()
    ce = F.cross_entropy(logits.float(), target)
    total = settings["lambda_dice"] * dice + settings["lambda_ce"] * ce
    return total, dice, ce


def train_epoch(model, loader, loss_settings, optimizer, scaler, device, amp):
    model.train()
    totals = []
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
        totals.append([float(value.detach()) for value in values])
    return np.mean(totals, axis=0).tolist()


@torch.inference_mode()
def validate(model, cases, config, device, amp):
    model.eval()
    losses, rows = [], []
    for case in cases:
        item = load_volume(case, config["preprocessing"], config["labels"]["id_to_source"])
        logits, probabilities = infer_volume(model, item, config, device, amp)
        target = torch.from_numpy(item["label"]).long()[None, None]
        losses.append([float(value) for value in segmentation_loss(logits, target, config["loss"])])
        rows.extend(case_metrics(probabilities.argmax(0, keepdim=True),
                                 label_tensor(ROOT / case["label"], config), case, config))
    return np.mean(losses, axis=0).tolist(), rows


def run_training(config, cases, splits, directory, device, amp):
    training_cases = select_cases(cases, splits["train"])
    validation_cases = select_cases(cases, splits["validation"])
    settings = config["training"]
    resolved = resolve_spacing(config, training_cases)
    batch_size = resolved["training"]["batch_size"]

    counts = Counter(case["patient_id"] for case in training_cases)
    weights = [1 / counts[case["patient_id"]] for case in training_cases]
    sampler = WeightedRandomSampler(weights, config["training"]["steps_per_epoch"] * batch_size,
                                    replacement=True)

    dataset = PatchDataset(training_cases, resolved["preprocessing"],
                           resolved["labels"]["id_to_source"])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=device.type == "cuda")

    model = ResidualUNet3D(resolved["model"], len(resolved["data"]["channel_names"]), len(resolved["labels"]["names"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    latest, best = directory / "latest.pt", directory / "best.pt"
    epoch_log = directory / "epochs.csv"
    best_score = -1.0
    best_rows = None
    target_count = 0

    write_json(directory / "resolved-config.json", resolved)
    write_json(directory / "runtime.json", versions(device))
    print(f"{len(training_cases)} training scans; spacing {resolved['preprocessing']['target_spacing_mm']}")

    for epoch in range(1, settings["epochs"] + 1):
        began = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_epoch(model, loader, resolved["loss"], optimizer, scaler, device, amp)

        validation_losses, metrics, improved = [None] * 3, dict.fromkeys(METRICS), False
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            validation_losses, rows = validate(model, validation_cases, resolved, device, amp)
            summary = save_metrics(directory, f"epoch{epoch:04d}", rows, resolved)
            metrics = summary["foreground_macro"]
            target_count = target_count + 1 if metrics["dice"] is not None and metrics["dice"] > 0.70 else 0
            improved = metrics["dice"] is not None and metrics["dice"] > best_score

            if improved:
                best_score = metrics["dice"]
                best_rows = rows
                save_metrics(directory, "best", rows, resolved)

        scheduler.step()

        row = {"epoch": epoch, "learning_rate": learning_rate, "seconds": time.perf_counter() - began,
               **{f"train_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), train_losses)},
               **{f"validation_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), validation_losses)},
               **{f"validation_{metric}": metrics[metric] for metric in METRICS}}

        state = {"model": model.state_dict(),
                 "epoch": epoch, "best_score": best_score, "config": resolved,
                 "splits": splits, "cases": cases}
        if improved:
            torch.save(state, best)
        torch.save(state, latest)

        exists = epoch_log.exists()
        with epoch_log.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()

        print(f"Epoch {epoch}/{settings['epochs']}: train={train_losses[0]:.4f}, "
              f"validation={validation_losses[0]}, Dice={metrics['dice']}, "
              f"lr={learning_rate:.6g}, seconds={row['seconds']:.1f}", flush=True)

        if target_count >= 2:
            print("stopped after two consecutive validations with Dice > 0.70.", flush=True)
            break

    return best_rows


def train(config):
    directory = OUTPUTS / config["run_id"]
    splits = read_json(ROOT / config["data"]["splits_file"])
    cases = discover(config)
    directory.mkdir(parents=True, exist_ok=False)
    device, amp = select_device(config["amp"])
    print(f"Run {config['run_id']}: "
          f"epochs={config['training']['epochs']}, steps={config['training']['steps_per_epoch']}, "
          f"batch={config['training']['batch_size']}, "
          f"channels={config['model']['channels']}, lr={config['training']['learning_rate']}", flush=True)
    rows = run_training(config, cases, splits, directory, device, amp)
    save_metrics(directory, "validation", rows, config)
