"""Train a Mask R-CNN baseline on the CarDD manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional


def _device(torch, requested: str) -> object:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, image_size: int, pretrained: bool = False):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=image_size,
        max_size=image_size,
    )
    box_predictor = model.roi_heads.box_predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(box_predictor.cls_score.in_features, num_classes)
    mask_predictor = model.roi_heads.mask_predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_predictor.conv5_mask.in_channels,
        mask_predictor.conv5_mask.out_channels,
        num_classes,
    )
    return model


def _run_epoch(model, loader, device, optimizer=None) -> float:
    import torch

    training = optimizer is not None
    # Detection models return loss dictionaries only in train mode.
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()} for target in targets]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / batches if batches else 0.0


def train(
    manifest: Path,
    output_dir: Path,
    epochs: int = 20,
    image_size: int = 800,
    batch_size: int = 2,
    learning_rate: float = 0.005,
    pretrained: bool = False,
    device: str = "auto",
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    num_workers: int = 0,
) -> dict:
    try:
        import torch
        from torch.optim import SGD
        from torch.optim.lr_scheduler import StepLR
        from torch.utils.data import DataLoader, Subset
        try:
            from training.cardd_dataset import CarDDInstanceDataset, collate_fn
        except ModuleNotFoundError:  # Direct execution: python training/train_mask_rcnn.py
            from cardd_dataset import CarDDInstanceDataset, collate_fn
    except ImportError as exc:
        raise RuntimeError("Install torch, torchvision, and pillow to run Mask R-CNN training") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = CarDDInstanceDataset(manifest, "train")
    validation_dataset = CarDDInstanceDataset(manifest, "validation")
    if max_train_images is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_images, len(train_dataset))))
    if max_val_images is not None:
        validation_dataset = Subset(validation_dataset, range(min(max_val_images, len(validation_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    runtime_device = _device(torch, device)
    model = build_model(len(CarDDInstanceDataset(manifest, "train").class_to_index) + 1, image_size, pretrained).to(runtime_device)
    optimizer = SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    history = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, runtime_device, optimizer)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, runtime_device)
        scheduler.step()
        checkpoint = output_dir / f"mask_rcnn_epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "mask_rcnn_best.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "seconds": time.perf_counter() - started})
        print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")

    summary = {
        "method": "mask_rcnn",
        "manifest": str(manifest),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "pretrained": pretrained,
        "device": str(runtime_device),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "best_validation_loss": best_val_loss,
        "history": history,
        "status": "completed",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision COCO weights; may download weights")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
