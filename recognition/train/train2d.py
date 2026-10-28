import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import OUTPUTS, write_json
from .data2d import SliceDataset, discover
from .model2d import ResidualUNet2D


def segmentation_loss(logits, target, settings):
    target = target[:, 0].long()
    probabilities = logits.float().softmax(1)
    truth = F.one_hot(target, num_classes=6).movedim(-1, 1).float()

    # keep soft probabilities differentiable for Dice loss
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
def validate(model, loader, device, amp):
    model.eval()
    intersection = torch.zeros(5, device=device, dtype=torch.float64)
    denominator = torch.zeros_like(intersection)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["label"][:, 0].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction = model(image).argmax(1)
        # accumulate foreground counts across the entire validation set
        for index, label in enumerate(range(1, 6)):
            predicted, truth = prediction == label, target == label
            intersection[index] += (predicted & truth).sum()
            denominator[index] += predicted.sum() + truth.sum()

    present = denominator > 0
    return (2 * intersection[present] / denominator[present]).mean().item()


def train(config):
    # separate 2D results from the corresponding 3D run
    run_id = config["run_id"]
    run_id = run_id if run_id.endswith("_2d") else f"{run_id}_2d"
    config = {**config, "run_id": run_id}
    directory = OUTPUTS / run_id
    directory.mkdir(parents=True, exist_ok=False)
    settings = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7)
    training = DataLoader(SliceDataset(discover("train")), batch_size=settings["batch_size"],
                          shuffle=True, pin_memory=device.type == "cuda")
    validation = DataLoader(SliceDataset(discover("validate")), batch_size=settings["batch_size"],
                            pin_memory=device.type == "cuda")

    model = ResidualUNet2D(config["model"], in_channels=1, classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    write_json(directory / "resolved-config.json", config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    (directory / "model.txt").write_text(f"{model}\n\nParameters: {parameters}\n", encoding="utf-8")
    best_score = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        train_epoch(model, training, config["loss"], optimizer, scaler, device, amp)
        score = validate(model, validation, device, amp)
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
