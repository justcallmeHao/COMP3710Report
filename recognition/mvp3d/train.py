"""Explicit smoke, one-pair and subset training entry points."""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

# Direct execution has no package context; resolve it from this file, not cwd.
if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mvp3d"

import torch
from torch.utils.data import DataLoader

from .dataset import Manifest, VolumeDataset, collate_pairs
from .metrics import hard_metrics, segmentation_loss
from .modules import UNet3D
from .utils import (ROOT, panels, reset_peak, resources, save_logs, select_device,
                    setup_seed, synchronize, versions, write_json)


def run_batch(model, batch, device, amp, optimizer, scaler):
    synchronize(device)
    reset_peak(device)
    start = time.perf_counter()
    image, target, metadata = batch
    image, target = image.to(device), target.to(device)
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
            logits = model(image)
        # Outside autocast: softmax, one-hot, reductions and CE are float32.
        loss, dice_loss, ce = segmentation_loss(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss.")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Missing or nonfinite gradients; optimizer not updated.")
            scaler.step(optimizer)
            scaler.update()
    metrics = hard_metrics(logits.detach().argmax(1), target)
    synchronize(device)
    return {
        "case_id": metadata[0]["case_id"], "loss": float(loss.detach()),
        "soft_foreground_dice_loss": float(dice_loss.detach()),
        "cross_entropy": float(ce.detach()), **metrics,
        "step_seconds": time.perf_counter() - start, **resources(device),
        "output_shape": str(list(logits.shape)),
        "gradients_finite": True if training else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["step", "one-pair", "subset"], required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "subset_data/manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("device", "amp", "steps", "epochs"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["num_classes"] != 6 or config["batch_size"] != 1 or config["num_workers"] != 0:
        raise ValueError("MVP requires six classes, batch size 1, num_workers 0.")
    if config["steps"] < 1 or config["epochs"] < 1:
        raise ValueError("Budgets must be positive.")
    device = select_device(config["device"], config["amp"])
    setup_seed(config["seed"])
    manifest = Manifest(args.manifest)
    directory = args.output or ROOT / "outputs/mvp3d" / (
        f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    write_json(directory / "manifest.json", manifest.document)
    model = UNet3D().to(device)
    write_json(directory / "run.json", {
        "mode": args.mode, "seed": config["seed"],
        "manifest_sha256": manifest.identity, "manifest_source": str(manifest.path),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "software": versions(device),
        "timing": "step excludes loading/resampling; includes transfers, model, loss, "
                  "backward/update when training, metrics and CUDA synchronization",
        "memory": "allocator peaks per step; device usage is a point sample, not a peak",
    })
    train_ids = ([c["case_id"] for c in manifest.cases.values() if c["split"] == "train"]
                 if args.mode == "subset" else ["B006_Week0"])
    validation_ids = [c["case_id"] for c in manifest.cases.values()
                      if c["split"] == "validation"]
    train_data = VolumeDataset(manifest, train_ids, config)
    validation_data = VolumeDataset(manifest, validation_ids, config)
    # Inspect each selected pair independently, without keeping all volumes in RAM.
    for dataset in ([train_data, validation_data] if args.mode == "subset" else [train_data]):
        for index in range(len(dataset)):
            image, target, metadata = dataset[index]
            write_json(directory / f"{metadata['case_id']}_preprocessing.json", metadata)
            panels(directory / f"{metadata['case_id']}_alignment.png",
                   image[0].numpy(), target.numpy(), metadata)
    loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_pairs)
    validation_loader = DataLoader(validation_data, batch_size=1, num_workers=0,
                                   collate_fn=collate_pairs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    rows, step, best = [], 0, -1.0

    def checkpoint(name, selection, score=None):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "config": config,
                    "manifest_sha256": manifest.identity, "manifest": manifest.document,
                    "mode": args.mode, "step": step, "selection": selection,
                    "validation_foreground_dice": score}, directory / name)

    epochs = config["epochs"] if args.mode == "subset" else (
        1 if args.mode == "step" else config["steps"])
    for epoch in range(1, epochs + 1):
        for batch in loader:
            step += 1
            row = run_batch(model, batch, device, config["amp"], optimizer, scaler)
            row.update(phase="train_pre_update", step=step, epoch=epoch)
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.mode == "subset":
            scores = []
            for batch in validation_loader:
                row = run_batch(model, batch, device, config["amp"], None, scaler)
                row.update(phase="validation", step=step, epoch=epoch)
                rows.append(row)
                if row["foreground_macro_dice"] is not None:
                    scores.append(row["foreground_macro_dice"])
            score = sum(scores) / len(scores) if scores else None
            if score is not None and score > best:
                best = score
                checkpoint("best.pt", "validation foreground macro Dice", score)
        save_logs(directory, rows)
        checkpoint("last.pt", "last optimizer update; not validation-selected")
    if args.mode != "subset":
        row = run_batch(model, next(iter(loader)), device, config["amp"], None, scaler)
        row.update(phase="training_pair_final_eval", step=step, epoch=epochs)
        rows.append(row)
        save_logs(directory, rows)
    print(f"Artifacts: {directory}")


if __name__ == "__main__":
    main()
