"""Training procedure for the object detection transformer."""

import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config
from data.dataset import create_train_val_dataloaders
from model.hungarian import HungarianMatcher
from model.model import TransformerModel
from model.positional_encoding import FixedPositionalEncoding2D
from model.utils import get_resnet_backbone


def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for training, but no CUDA device is available."
        )

    return torch.device("cuda")


def build_model() -> TransformerModel:
    return TransformerModel(
        backbone_builder=get_resnet_backbone,
        feature_num_layers=config.BACKBONE_NUM_LAYERS,
        positional_encoding_builder=FixedPositionalEncoding2D,
        d_model=config.D_MODEL,
        height=config.FEATURE_HEIGHT,
        width=config.FEATURE_WIDTH,
        max_objects=config.NUM_QUERIES,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    )


def move_targets_to_device(
    targets: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        "boxes": targets["boxes"].to(device),
        "labels": targets["labels"].to(device),
        "object_mask": targets["object_mask"].to(device),
    }


def normalize_target_boxes(targets: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        **targets,
        "boxes": targets["boxes"] / float(config.IMAGE_SIZE),
    }


def compute_losses(
    pred_class_logits: Tensor,
    pred_boxes: Tensor,
    targets: dict[str, Tensor],
    matcher: HungarianMatcher,
) -> tuple[Tensor, dict[str, float]]:
    pred_class_probs = pred_class_logits.softmax(dim=-1)

    with torch.no_grad():
        target_classes, pred_bbox_indices, target_bbox_indices = matcher(
            pred_class_probs,
            pred_boxes,
            targets,
        )

    class_loss = F.cross_entropy(
        pred_class_logits.transpose(1, 2),
        target_classes,
    )

    if pred_bbox_indices[0].numel() == 0:
        bbox_loss = pred_boxes.sum() * 0.0
    else:
        matched_pred_boxes = pred_boxes[pred_bbox_indices]
        matched_target_boxes = targets["boxes"][target_bbox_indices]
        bbox_loss = F.smooth_l1_loss(matched_pred_boxes, matched_target_boxes)

    loss = (
        config.CLASS_LOSS_WEIGHT * class_loss
        + config.BBOX_LOSS_WEIGHT * bbox_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "class_loss": float(class_loss.detach().cpu()),
        "bbox_loss": float(bbox_loss.detach().cpu()),
    }

    return loss, metrics


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    matcher: HungarianMatcher,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "class_loss": 0.0,
        "bbox_loss": 0.0,
    }
    num_batches = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for images, targets in data_loader:
            images = images.to(device)
            targets = move_targets_to_device(targets, device)
            targets = normalize_target_boxes(targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            pred_class_logits, pred_boxes = model(images)
            loss, metrics = compute_losses(
                pred_class_logits,
                pred_boxes,
                targets,
                matcher,
            )

            if is_train:
                loss.backward()
                optimizer.step()

            for key, value in metrics.items():
                totals[key] += value
            num_batches += 1

    if num_batches == 0:
        return totals

    return {key: value / num_batches for key, value in totals.items()}


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    is_best: bool,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": {
            name: getattr(config, name)
            for name in dir(config)
            if name.isupper()
        },
    }

    torch.save(checkpoint, checkpoint_dir / "last.pt")

    if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")

    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best.pt")


def main() -> None:
    device = get_cuda_device()
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = create_train_val_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = build_model().to(device)
    matcher = HungarianMatcher(
        class_weight=config.MATCHER_CLASS_WEIGHT,
        bbox_weight=config.MATCHER_BBOX_WEIGHT,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            matcher=matcher,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            matcher=matcher,
            device=device,
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        save_checkpoint(
            checkpoint_dir=Path(__file__).resolve().parent / config.CHECKPOINT_DIR,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_best=is_best,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_class={train_metrics['class_loss']:.4f} "
            f"train_bbox={train_metrics['bbox_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_class={val_metrics['class_loss']:.4f} "
            f"val_bbox={val_metrics['bbox_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
