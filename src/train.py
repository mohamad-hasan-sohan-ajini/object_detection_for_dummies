"""Training procedure for the object detection transformer."""

from pathlib import Path
from typing import Any


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from data.dataset import create_train_val_dataloaders
from model.hungarian import HungarianMatcher
from model.model import TransformerModel
from model.positional_encoding import FixedPositionalEncoding2D
from model.utils import get_resnet_backbone


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
    class_predictions = pred_class_logits.argmax(dim=-1)
    class_accuracy = (class_predictions == target_classes).float().mean()

    matched_pred_boxes = pred_boxes[pred_bbox_indices]
    matched_target_boxes = targets["boxes"][target_bbox_indices]
    bbox_loss = F.smooth_l1_loss(matched_pred_boxes, matched_target_boxes)

    loss = config.CLASS_LOSS_WEIGHT * class_loss + config.BBOX_LOSS_WEIGHT * bbox_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "class_loss": float(class_loss.detach().cpu()),
        "bbox_loss": float(bbox_loss.detach().cpu()),
        "class_accuracy": float(class_accuracy.detach().cpu()),
    }

    return loss, metrics


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    matcher: HungarianMatcher,
    device: torch.device,
    optimizer: AdamW | None = None,
    description: str = "epoch",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "class_loss": 0.0,
        "bbox_loss": 0.0,
        "class_accuracy": 0.0,
    }
    num_batches = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        progress_bar = tqdm(
            data_loader,
            desc=description,
            dynamic_ncols=True,
            leave=False,
        )
        for images, targets in progress_bar:
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
            progress_bar.set_postfix(
                loss=totals["loss"] / num_batches,
                cls=totals["class_loss"] / num_batches,
                bbox=totals["bbox_loss"] / num_batches,
                acc=totals["class_accuracy"] / num_batches,
            )

    return {key: value / num_batches for key, value in totals.items()}


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
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
            name: getattr(config, name) for name in dir(config) if name.isupper()
        },
    }

    torch.save(checkpoint, checkpoint_dir / "last.pt")

    if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")


def log_metrics(
    writer: Any,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    optimizer: AdamW,
) -> None:
    writer.add_scalar("loss/train", train_metrics["loss"], epoch)
    writer.add_scalar("loss/val", val_metrics["loss"], epoch)
    writer.add_scalar("class_loss/train", train_metrics["class_loss"], epoch)
    writer.add_scalar("class_loss/val", val_metrics["class_loss"], epoch)
    writer.add_scalar("bbox_loss/train", train_metrics["bbox_loss"], epoch)
    writer.add_scalar("bbox_loss/val", val_metrics["bbox_loss"], epoch)
    writer.add_scalar("class_accuracy/train", train_metrics["class_accuracy"], epoch)
    writer.add_scalar("class_accuracy/val", val_metrics["class_accuracy"], epoch)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)


def main() -> None:
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device("cuda", index=0)
    torch.backends.cudnn.benchmark = True
    project_dir = Path(__file__).resolve().parent

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

    log_dir = project_dir / config.LOG_DIR

    with SummaryWriter(log_dir=log_dir) as writer:
        writer.add_text("device", str(device), 0)
        writer.add_text(
            "config",
            "\n".join(
                f"{name}: {getattr(config, name)}"
                for name in dir(config)
                if name.isupper()
            ),
            0,
        )

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_metrics = run_epoch(
                model=model,
                data_loader=train_loader,
                matcher=matcher,
                device=device,
                optimizer=optimizer,
                description=f"epoch {epoch:03d}/{config.NUM_EPOCHS:03d} train",
            )
            val_metrics = run_epoch(
                model=model,
                data_loader=val_loader,
                matcher=matcher,
                device=device,
                description=f"epoch {epoch:03d}/{config.NUM_EPOCHS:03d} val",
            )

            log_metrics(
                writer=writer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=optimizer,
            )
            writer.flush()

            save_checkpoint(
                checkpoint_dir=project_dir / config.CHECKPOINT_DIR,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_class={train_metrics['class_loss']:.4f} "
                f"train_bbox={train_metrics['bbox_loss']:.4f} "
                f"train_acc={train_metrics['class_accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_class={val_metrics['class_loss']:.4f} "
                f"val_bbox={val_metrics['bbox_loss']:.4f} "
                f"val_acc={val_metrics['class_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
