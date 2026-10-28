import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data2d import OUTPUTS, ROOT, SliceDataset, discover, read_json, write_json
from .model2d import ResidualUNet2D


def segmentation_loss(logits, target, settings):
    target = target[:, 0].long()
    probabilities = logits.float().softmax(1)
    truth = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).float()

    intersection = (probabilities * truth).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + truth.sum(dim=(2, 3))
    epsilon = settings["probability_floor"]
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    if not settings["include_background"]:
        dice = dice[:, 1:]
    ce = F.cross_entropy(logits.float(), target)
    return settings["lambda_dice"] * (1 - dice.mean()) + settings["lambda_ce"] * ce


def train_epoch(model, loader, loss_settings, optimizer, scaler, device, amp):
    model.train()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["label"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(image)
            loss = segmentation_loss(logits, target, loss_settings)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


@torch.inference_mode()
def validate(model, cases, config, device, amp):
    model.eval()
    id_to_source = config["labels"]["id_to_source"]
    scans, patients = {}, {}
    for case in cases:
        scans.setdefault(case["case_id"], []).append(case)

    for scan in scans.values():
        loader = DataLoader(SliceDataset(scan, id_to_source),
                            batch_size=config["training"]["batch_size"],
                            pin_memory=device.type == "cuda")
        prediction = torch.empty(scan[0]["geometry"]["original_shape"], dtype=torch.int64)
        target = torch.empty_like(prediction)
        start = 0
        # restore all slices to their native volume positions before scoring
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                labels = model(image).argmax(1).cpu()
            for index, case in enumerate(scan[start:start + len(labels)]):
                prediction.select(case["slice_axis"], case["slice_index"]).copy_(labels[index])
                target.select(case["slice_axis"], case["slice_index"]).copy_(batch["label"][index, 0])
            start += len(labels)

        scores = []
        for label in range(1, len(id_to_source)):
            predicted, truth = prediction == label, target == label
            intersection = (predicted & truth).sum().double()
            denominator = predicted.sum() + truth.sum()
            scores.append(torch.where(denominator > 0, 2 * intersection / denominator, float("nan")))
        patients.setdefault(scan[0]["patient_id"], []).append(torch.stack(scores))

    # match 3D aggregation
    patient_scores = torch.stack([torch.stack(scores).nanmean(0) for scores in patients.values()])
    score = patient_scores.nanmean(0).nanmean().item()
    return score


def run_training(config, training_cases, validation_cases, directory, device, amp):
    settings = config["training"]
    id_to_source = config["labels"]["id_to_source"]
    training = DataLoader(SliceDataset(training_cases, id_to_source), batch_size=settings["batch_size"],
                          shuffle=True, pin_memory=device.type == "cuda")

    model = ResidualUNet2D(config["model"], in_channels=1, classes=len(id_to_source)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    write_json(directory / "resolved-config.json", config)
    best_score = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        train_epoch(model, training, config["loss"], optimizer, scaler, device, amp)
        score = validate(model, validation_cases, config, device, amp)
        improved = score > best_score
        if improved:
            best_score = score
        scheduler.step()

        # save prediction-ready checkpoints without metric reports
        state = {"model": model.state_dict(), "epoch": epoch, "best_score": best_score,
                 "config": config}
        if improved:
            torch.save(state, directory / "best.pt")
        torch.save(state, directory / "latest.pt")
    return directory


def train(config):
    run_id = config["run_id"]
    run_id = run_id if run_id.endswith("_2d") else f"{run_id}_2d"
    config = {**config, "run_id": run_id, "labels": read_json(ROOT / "configs/labels.json")}
    directory = OUTPUTS / run_id
    directory.mkdir(parents=True, exist_ok=False)
    training_cases, validation_cases = discover("train"), discover("validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7)
    return run_training(config, training_cases, validation_cases, directory, device, amp)
