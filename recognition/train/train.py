import csv
import random
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional.segmentation import dice_score

from .data import OUTPUTS, ROOT, PatchDataset, discover, load_volume, read_json, resolve_spacing, select_cases, split_manifest, write_json
from .evaluate import METRICS, case_metrics, label_tensor, save_metrics, summarize, write_csv
from .model import ResidualUNet3D
from .predict import infer_volume, seed_state, select_device, versions


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


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


def run_fold(config, cases, splits, fold, directory, device, amp):
    groups = splits["folds"][fold]
    training_cases = select_cases(cases, groups["train"])
    resolved = resolve_spacing(config, training_cases)
    seed = config["seed"] + fold
    seed_state(seed)
    validation_cases = select_cases(cases, groups["validation"])
    generator = torch.Generator().manual_seed(seed)
    sampling = resolved["sampling"]
    counts = Counter(case["patient_id"] for case in training_cases)
    weights = [1 / counts[case["patient_id"]] for case in training_cases]
    sampler = WeightedRandomSampler(weights, config["training"]["steps_per_epoch"] * sampling["batch_size"],
                                    replacement=True, generator=generator)
    dataset = PatchDataset(training_cases, sampling, resolved["augmentation"], resolved["preprocessing"],
                           resolved["labels"]["id_to_source"])
    loader = DataLoader(dataset, batch_size=sampling["batch_size"], sampler=sampler,
                        pin_memory=device.type == "cuda", generator=generator)
    model = ResidualUNet3D(resolved["model"], len(resolved["data"]["channel_names"]), len(resolved["labels"]["names"])).to(device)
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    latest, best = directory / f"fold{fold}-latest.pt", directory / f"fold{fold}-best.pt"
    epoch_log = directory / f"fold{fold}-epochs.csv"
    start, best_score, history = 1, -1.0, []
    best_rows = None
    if config["resume"]:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        generator.set_state(checkpoint["loader_rng"])
        start, best_score, history = checkpoint["epoch"] + 1, checkpoint["best_score"], checkpoint["history"]
        write_csv(epoch_log, history)
    write_json(directory / f"fold{fold}-config.json", resolved)
    write_json(directory / f"fold{fold}-runtime.json", versions(device))
    print(f"Fold {fold}: {len(training_cases)} training scans; spacing {resolved['preprocessing']['target_spacing_mm']}")
    for epoch in range(start, settings["epochs"] + 1):
        began = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_epoch(model, loader, resolved["loss"], optimizer, scaler, device, amp)
        validation_losses, metrics, improved = [None] * 3, dict.fromkeys(METRICS), False
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            validation_losses, rows = validate(model, validation_cases, resolved, device, amp)
            summary = save_metrics(directory, f"fold{fold}-epoch{epoch:04d}", rows, resolved)
            metrics = summary["foreground_macro"]
            improved = metrics["dice"] is not None and metrics["dice"] > best_score
            if improved:
                best_score = metrics["dice"]
                best_rows = rows
                save_metrics(directory, f"fold{fold}-best", rows, resolved)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": learning_rate, "seconds": time.perf_counter() - began,
               **{f"train_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), train_losses)},
               **{f"validation_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), validation_losses)},
               **{f"validation_{metric}": metrics[metric] for metric in METRICS}}
        history.append(row)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_score": best_score, "config": resolved, "fold": fold,
                 "splits": splits, "cases": cases, "rng": rng_state(), "loader_rng": generator.get_state(),
                 "history": history, "versions": versions(device)}
        if improved:
            save_checkpoint(best, state)
        save_checkpoint(latest, state)
        exists = epoch_log.exists()
        with epoch_log.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
        print(f"Fold {fold} epoch {epoch}/{settings['epochs']}: train={train_losses[0]:.4f}, "
              f"validation={validation_losses[0]}, Dice={metrics['dice']}, "
              f"lr={learning_rate:.6g}, seconds={row['seconds']:.1f}", flush=True)

    return best_rows if best_rows is not None else read_json(directory / f"fold{fold}-best-cases.json")


def train(config, prepared=None):
    directory = OUTPUTS / config["run_id"]
    if config["resume"]:
        folds = config["data"]["folds_to_run"]
        config = read_json(directory / "config.json")
        config["data"]["folds_to_run"] = folds
        config["resume"] = True
        cases = read_json(directory / "cases.json")
        splits = read_json(directory / "splits.json")
    else:
        if prepared is None:
            cases = discover(config)
            splits = split_manifest(config, cases)
        else:
            cases, splits = prepared
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", config)
        write_json(directory / "cases.json", cases)
        write_json(directory / "splits.json", splits)
    device, amp = select_device(config["amp"])
    print(f"Run {config['run_id']}: folds={config['data']['folds_to_run']}, "
          f"epochs={config['training']['epochs']}, steps={config['training']['steps_per_epoch']}, "
          f"patch={config['sampling']['patch_size']}, batch={config['sampling']['batch_size']}, "
          f"channels={config['model']['channels']}, lr={config['training']['learning_rate']}, "
          f"seed={config['seed']}", flush=True)
    rows, folds = [], []
    for fold in config["data"]["folds_to_run"]:
        fold_rows = run_fold(config, cases, splits, fold, directory, device, amp)
        rows.extend(fold_rows)
        folds.append({"fold": fold, **summarize(fold_rows, config)[1]["foreground_macro"]})
    summary = save_metrics(directory, "development-oof", rows, config)
    summary["folds"] = folds
    summary["complete_development_coverage"] = len(folds) == len(splits["folds"])
    write_json(directory / "development-oof-summary.json", summary)

