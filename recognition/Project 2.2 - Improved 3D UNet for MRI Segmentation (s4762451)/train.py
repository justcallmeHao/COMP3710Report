import csv
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional.segmentation import dice_score

from dataset import OUTPUTS, ROOT, PatchDataset, discover, load_volume, read_json, resolve_spacing, select_cases, write_json
from helpers.evaluate3d import METRICS, case_metrics, label_tensor, save_metrics
from modules import ResidualUNet3D
from predict import infer_volume, select_device, versions


def segmentation_loss(logits, target, settings):
    # remove the label channel dimension for cross-entropy
    target = target[:, 0].long()

    # convert predictions to probabilities and labels to one-hot channels
    classes = logits.shape[1]
    probabilities = logits.float().softmax(1).clamp_min(settings["probability_floor"])
    truth = F.one_hot(target, num_classes=classes).movedim(-1, 1).float()

    # calculate the mean dice loss across the selected classes
    dice = 1 - dice_score(probabilities, truth, num_classes=classes,
                          include_background=settings["include_background"],
                          average="macro", input_format="one-hot").mean()

    # combine dice and cross-entropy with the configured weights
    ce = F.cross_entropy(logits.float(), target)
    total = settings["lambda_dice"] * dice + settings["lambda_ce"] * ce
    return total, dice, ce


def train_epoch(model, loader, loss_settings, optimizer, scaler, device, amp):
    # enable training mode and collect batch losses
    model.train()
    totals = []
    for batch in loader:
        # move the image and label batch to the selected device
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["label"].to(device, non_blocking=device.type == "cuda")

        # clear gradients and predict with optional mixed precision
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)

        # backpropagate the loss and update model weights
        values = segmentation_loss(logits, target, loss_settings)
        scaler.scale(values[0]).backward()
        scaler.step(optimizer)
        scaler.update()

        # store detached losses for the epoch average
        totals.append([float(value.detach()) for value in values])

    # return the mean total, dice, and cross-entropy losses
    return np.mean(totals, axis=0).tolist()


@torch.inference_mode()
def validate(model, cases, config, device, amp):
    # disable training behaviour and collect validation results
    model.eval()
    losses, rows = [], []
    for case in cases:
        # load the cached scan and run inference
        item = load_volume(case)
        logits, probabilities = infer_volume(model, item, config, device, amp)

        # calculate loss on the preprocessed label grid
        target = torch.from_numpy(item["label"]).long()[None, None]
        losses.append([float(value) for value in segmentation_loss(logits, target, config["loss"])])

        # score predictions against labels in the original scan geometry
        rows.extend(case_metrics(probabilities.argmax(0, keepdim=True),
                                 label_tensor(ROOT / case["label"], config), case, config))

    # return mean validation losses and per-case metrics
    return np.mean(losses, axis=0).tolist(), rows


def run_training(config, cases, splits, directory, device, amp):
    # select training and validation scans and resolve voxel spacing
    training_cases = select_cases(cases, splits["train"])
    validation_cases = select_cases(cases, splits["validation"])
    settings = config["training"]
    resolved = resolve_spacing(config)
    batch_size = resolved["training"]["batch_size"]

    # give each patient equal total sampling weight across their scans
    counts = Counter(case["patient_id"] for case in training_cases)
    weights = [1 / counts[case["patient_id"]] for case in training_cases]
    sampler = WeightedRandomSampler(weights, config["training"]["steps_per_epoch"] * batch_size,
                                    replacement=True)

    # load training batches with the weighted sampler
    dataset = PatchDataset(training_cases)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=device.type == "cuda")

    # create the model, optimiser, learning-rate scheduler, and gradient scaler
    model = ResidualUNet3D(resolved["model"], len(resolved["data"]["channel_names"]), len(resolved["labels"]["names"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # set output paths and initialise best-score tracking
    best = directory / "best.pt"
    epoch_log = directory / "epochs.csv"
    best_score = -1.0
    best_rows = None
    target_count = 0

    # record the resolved configuration and runtime environment
    write_json(directory / "resolved-config.json", resolved)
    write_json(directory / "runtime.json", versions(device))
    print(f"{len(training_cases)} training scans; spacing {resolved['preprocessing']['target_spacing_mm']}")

    # train and validate until the epoch limit or early stopping
    for epoch in range(1, settings["epochs"] + 1):
        # time the epoch and train using the current learning rate
        began = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_epoch(model, loader, resolved["loss"], optimizer, scaler, device, amp)

        # validate at the configured interval and on the final epoch
        validation_losses, metrics, improved = [None] * 3, dict.fromkeys(METRICS), False
        if epoch % settings["validation_interval"] == 0 or epoch == settings["epochs"]:
            validation_losses, rows = validate(model, validation_cases, resolved, device, amp)
            summary = save_metrics(directory, f"epoch{epoch:04d}", rows, resolved)
            metrics = summary["foreground_macro"]

            # track consecutive target scores and check for a new best score
            target_count = target_count + 1 if metrics["dice"] is not None and metrics["dice"] > 0.70 else 0
            improved = metrics["dice"] is not None and metrics["dice"] > best_score

            # save metrics when validation dice improves
            if improved:
                best_score = metrics["dice"]
                best_rows = rows
                save_metrics(directory, "best", rows, resolved)

        # decay the learning rate for the next epoch
        scheduler.step()

        # collect losses, metrics, learning rate, and elapsed time
        row = {"epoch": epoch, "learning_rate": learning_rate, "seconds": time.perf_counter() - began,
               **{f"train_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), train_losses)},
               **{f"validation_{name}": value for name, value in zip(("loss", "dice_loss", "ce_loss"), validation_losses)},
               **{f"validation_{metric}": metrics[metric] for metric in METRICS}}

        # save the best model for prediction
        state = {"model": model.state_dict(),
                 "epoch": epoch, "best_score": best_score, "config": resolved,
                 "splits": splits}
        if improved:
            torch.save(state, best)

        # append the epoch results and write the header for a new log
        exists = epoch_log.exists()
        with epoch_log.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()

        # print the epoch results
        print(f"Epoch {epoch}/{settings['epochs']}: train={train_losses[0]:.4f}, "
              f"validation={validation_losses[0]}, Dice={metrics['dice']}, "
              f"lr={learning_rate:.6g}, seconds={row['seconds']:.1f}", flush=True)

        # early stopping
        if target_count >= 5:
            print("stopped after 5 consecutive validations with Dice > 0.70.", flush=True)
            break

    # return per-case metrics from the best validation
    return best_rows


def train(config):
    # discover scans and create a new training output directory
    directory = OUTPUTS / config["run_id"]
    splits = read_json(ROOT / config["data"]["splits_file"])
    cases = discover(config)
    directory.mkdir(parents=True, exist_ok=False)

    # select the compute device and report the training settings
    device, amp = select_device(config["amp"])
    print(f"Run {config['run_id']}: "
          f"epochs={config['training']['epochs']}, steps={config['training']['steps_per_epoch']}, "
          f"batch={config['training']['batch_size']}, "
          f"channels={config['model']['channels']}, lr={config['training']['learning_rate']}", flush=True)

    # run training and save the best validation results
    rows = run_training(config, cases, splits, directory, device, amp)
    save_metrics(directory, "validation", rows, config)


def main():
    from graphs.plot import plot
    from helpers.evaluate3d import evaluate
    from helpers.pipeline import run_pipeline
    from predict import predict

    config = read_json(ROOT / "configs/training3d.json")
    config["labels"] = read_json(ROOT / config["data"]["labels_file"])
    run_pipeline(config, train=train, predict=predict, evaluate=evaluate, plot=plot)


if __name__ == "__main__":
    main()
