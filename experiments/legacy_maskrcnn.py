"""Mask R-CNN training entrypoint for the CarDD instance-segmentation track.

Reads the manifest produced by ``dataset/prepare_cardd.py`` (COCO-style
polygons), trains ``torchvision.models.detection.maskrcnn_resnet50_fpn`` on
the local CarDD copy, evaluates it on a held-out split, and writes
``training_summary.json`` with the same top-level keys/order as
``training/train_yolo.py``'s summary (``method``, ``seed``, ``device``,
``train_seconds``, ``status``, ``evaluation_split``, ``box_metrics``,
``mask_metrics``, ``latency_ms_per_image``, ``hardware``, ``note``), so
``evaluation/merge_results.py`` can treat both uniformly. Mask-R-CNN-only
fields (``batch_size``, ``train_history``, ...) are appended after.

Device auto-detects CUDA, then MPS, then CPU (same convention as
``train_yolo.py``'s ``--device``). ``pretrained`` defaults to True (COCO
backbone, ~170MB one-time download to ``~/.cache/torch/hub/checkpoints/``),
matching train_yolo's fine-tuned baseline so the comparison stays fair; pass
``--no-pretrained`` for a from-scratch run.

Optional deps, not in the base requirements: ``pip install torch torchvision
pillow numpy``. Importing this module works without them; only ``train()``
raises.

Precision/recall/mAP are this script's own COCO-style matcher (no
pycocotools) at ``--confidence-threshold`` / ``--iou-match-threshold``,
101-point interpolation per class -- for consistent comparison with the YOLO
baseline, not a drop-in for official COCO eval. Metrics that can't be
computed (e.g. a class with 0 ground-truth instances) report as ``"TBD"``,
never a fabricated number.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Optional

# Some detection ops (NMS/ROI Align) have had correctness bugs on MPS; this
# falls back to CPU for anything MPS doesn't support. No-op on CUDA/CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import box_iou
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    _IMPORT_ERROR: Optional[ImportError] = exc
    Dataset = object  # placeholder base class so CarDDInstanceSegmentation can still be *defined*
else:
    _IMPORT_ERROR = None


def _require_torch() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Install the optional 'torch', 'torchvision', 'pillow' and 'numpy' "
            "dependencies to run Mask R-CNN training/evaluation"
        ) from _IMPORT_ERROR


# --------------------------------------------------------------------------- #
# Dataset adapter
# --------------------------------------------------------------------------- #


class CarDDInstanceSegmentation(Dataset):
    """torch ``Dataset`` adapter over the CarDD manifest.

    Expects COCO-polygon segmentation (RLE is rejected with a clear error).
    Boxes are derived from the rasterized mask, not the manifest's ``bbox``
    field (which may be a placeholder) -- the mask is the source of truth.
    Zero-area/empty-polygon annotations are dropped (image kept as a
    possibly-negative example); ``iscrowd`` annotations are excluded
    entirely. Category ids are remapped to a contiguous ``1..num_classes``
    label space (0 = background), stable across splits.
    """

    def __init__(
        self,
        manifest_path: Path,
        images_root: Path,
        split: str,
        transform: Optional[Callable] = None,
        min_instance_area: float = 4.0,
        category_id_to_label: Optional[dict[int, int]] = None,
        classes: Optional[list[str]] = None,
        max_images: Optional[int] = None,
    ) -> None:
        _require_torch()
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.manifest_path = Path(manifest_path)
        self.images_root = Path(images_root)
        self.split = split
        self.transform = transform
        self.min_instance_area = min_instance_area
        self.classes: list[str] = classes if classes is not None else manifest["classes"]

        # Reuse the train split's label map so val/test use identical label ids.
        self.category_id_to_label = category_id_to_label or self._build_label_map(manifest)

        self.records = [img for img in manifest["images"] if img["split"] == split]
        if not self.records:
            raise ValueError(f"No images found for split={split!r} in {manifest_path}")
        if max_images is not None:
            self.records = self.records[:max_images]

    @staticmethod
    def _build_label_map(manifest: dict) -> dict[int, int]:
        ids = sorted({ann["category_id"] for img in manifest["images"] for ann in img["annotations"]})
        return {cid: i + 1 for i, cid in enumerate(ids)}  # 1-indexed; 0 == background

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _rasterize(segmentation: Any, height: int, width: int) -> Optional["np.ndarray"]:
        if isinstance(segmentation, dict):
            raise NotImplementedError(
                "RLE segmentation is not supported by this adapter; convert CarDD "
                "annotations to polygons in dataset/prepare_cardd.py before training."
            )
        polygons = segmentation or []
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        drew_any = False
        for polygon in polygons:
            if len(polygon) < 6:  # need >= 3 points
                continue
            points = list(zip(polygon[0::2], polygon[1::2]))
            draw.polygon(points, outline=1, fill=1)
            drew_any = True
        if not drew_any:
            return None
        return np.array(mask_img, dtype=np.uint8)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = self.images_root / record["relative_path"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}. Check that --images-root points at your "
                f"local CarDD_COCO directory (manifest relative_path={record['relative_path']!r})."
            )
        image = Image.open(image_path).convert("RGB")
        width = record.get("width", image.width)
        height = record.get("height", image.height)

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list["np.ndarray"] = []
        areas: list[float] = []

        for ann in record["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            mask = self._rasterize(ann.get("segmentation"), height, width)
            if mask is None or mask.sum() < self.min_instance_area:
                continue
            ys, xs = np.where(mask)
            x1, x2 = float(xs.min()), float(xs.max() + 1)
            y1, y2 = float(ys.min()), float(ys.max() + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            category_id = ann["category_id"]
            if category_id not in self.category_id_to_label:
                # Category never seen in the split this map was built from; skip
                # rather than silently invent a label id.
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[category_id])
            masks.append(mask)
            areas.append(float(mask.sum()))

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, height, width), dtype=torch.uint8)
            areas_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": areas_t,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


class DetectionTransform:
    """Minimal train-time augmentation: to-tensor plus optional horizontal flip.

    Resizing is intentionally *not* done here: ``maskrcnn_resnet50_fpn``'s
    built-in ``GeneralizedRCNNTransform`` resizes images (and, during
    training, boxes/masks) to ``min_size``/``max_size`` internally and
    un-resizes predictions back to the original resolution at inference
    time, so the dataset adapter always works in original-image coordinates.
    """

    def __init__(self, hflip_prob: float = 0.0) -> None:
        self.hflip_prob = hflip_prob

    def __call__(self, image, target):
        image_t = TF.to_tensor(image)
        if self.hflip_prob and torch.rand(1).item() < self.hflip_prob:
            image_t = image_t.flip(-1)
            width = image_t.shape[-1]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-1)
        return image_t, target


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(num_classes: int, image_size: int, pretrained: bool = True):
    """Build Mask R-CNN with replaced box/mask heads for ``num_classes`` (incl. background).

    ``pretrained`` defaults to True to mirror train_yolo.py's baseline, which
    fine-tunes from a pretrained checkpoint (``yolo11n-seg.pt``) rather than
    training from scratch -- keeping both methods on the same "pretrained
    baseline, fine-tuned on CarDD" footing is what makes the comparison fair.
    The COCO-pretrained backbone is a ~170MB one-time download, cached under
    ``~/.cache/torch/hub/checkpoints/`` after the first run. Pass
    ``--no-pretrained`` if you specifically want a from-scratch run instead.
    """
    weights = "DEFAULT" if pretrained else None
    # weights_backbone downloads ImageNet weights even when weights=None; disable
    # both so --no-pretrained is fully from-scratch.
    weights_backbone = "DEFAULT" if pretrained else None
    # Keep the standard COCO 800/1333 min/max-size ratio for non-default --image-size.
    max_size = max(image_size, round(image_size * 1333 / 800))
    model = maskrcnn_resnet50_fpn(
        weights=weights, weights_backbone=weights_backbone, min_size=image_size, max_size=max_size
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# --------------------------------------------------------------------------- #
# Optimization loop
# --------------------------------------------------------------------------- #


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device: str,
    epoch: int,
    grad_clip_norm: float = 0.0,
    log_every: int = 20,
    warmup: bool = True,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    num_steps = 0

    # Mixed-precision (AMP) only works on CUDA; GradScaler is a no-op concept on CPU/MPS.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # 1-epoch linear LR warmup (torchvision detection reference recipe): without it,
    # loss can spike to NaN in the first few steps, especially at small --batch-size.
    warmup_scheduler = None
    if warmup and epoch == 1:
        warmup_iters = min(1000, len(data_loader) - 1)
        if warmup_iters > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
            )

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        if not torch.isfinite(losses):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch} step {step}: {loss_dict}. "
                "This is usually the LR ramping up too fast for a small --batch-size even "
                "with warmup active -- try a lower --lr (e.g. 0.001) or enable "
                "--grad-clip-norm 5.0; also double-check the dataset adapter's boxes/masks "
                "aren't degenerate for this batch."
            )

        if scaler is not None:
            scaler.scale(losses).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        totals["total_loss"] = totals.get("total_loss", 0.0) + float(losses.item())
        for name, value in loss_dict.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())
        num_steps += 1

        if step % log_every == 0 or step == len(data_loader):
            print(f"[epoch {epoch}] step {step}/{len(data_loader)} loss={losses.item():.4f}")

    return {k: v / max(num_steps, 1) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# Evaluation: box + mask precision/recall/mAP50/mAP50:95
# --------------------------------------------------------------------------- #


def _synchronize(device: str) -> None:
    """Block until pending GPU work finishes, so wall-clock timing is accurate.

    Needed for both CUDA and MPS -- ops on either backend run asynchronously,
    so timing without a sync point would understate real latency.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _run_inference(model, data_loader, device: str):
    # NOTE: intentionally not decorated with @torch.no_grad() so this module stays
    # importable (e.g. for --help / tests) even without torch installed; the
    # no_grad context is entered explicitly inside the function body instead.
    model.eval()
    predictions = []
    targets_out = []
    latencies_s = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            _synchronize(device)
            start = time.perf_counter()
            outputs = model(images)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            latencies_s.append(elapsed / max(len(images), 1))

            for out, tgt in zip(outputs, targets):
                masks = out["masks"]
                if masks.numel():
                    masks_bool = (masks.squeeze(1) > 0.5).cpu()
                else:
                    h, w = tgt["masks"].shape[-2:] if tgt["masks"].numel() else (0, 0)
                    masks_bool = torch.zeros((0, h, w), dtype=torch.bool)
                predictions.append(
                    {
                        "boxes": out["boxes"].cpu(),
                        "labels": out["labels"].cpu(),
                        "scores": out["scores"].cpu(),
                        "masks": masks_bool,
                    }
                )
                targets_out.append(
                    {
                        "boxes": tgt["boxes"],
                        "labels": tgt["labels"],
                        "masks": tgt["masks"].bool(),
                    }
                )

    return predictions, targets_out, latencies_s


def _mask_iou(masks_a: "torch.Tensor", masks_b: "torch.Tensor") -> "torch.Tensor":
    """Pairwise IoU between boolean mask sets: (Na,H,W) x (Nb,H,W) -> (Na,Nb)."""
    if masks_a.shape[0] == 0 or masks_b.shape[0] == 0:
        return torch.zeros((masks_a.shape[0], masks_b.shape[0]))
    a = masks_a.flatten(1).float()
    b = masks_b.flatten(1).float()
    intersection = a @ b.T
    area_a = a.sum(1).unsqueeze(1)
    area_b = b.sum(1).unsqueeze(0)
    union = area_a + area_b - intersection
    return intersection / union.clamp(min=1e-6)


def _collect_class_matches(predictions, targets, class_id: int, iou_fn, item_key: str):
    """Precompute per-image score order + IoU matrix vs GT for one class (reused across IoU thresholds)."""
    per_image = []
    total_gt = 0
    for pred, tgt in zip(predictions, targets):
        p_mask = pred["labels"] == class_id
        g_mask = tgt["labels"] == class_id
        p_items = pred[item_key][p_mask]
        p_scores = pred["scores"][p_mask]
        g_items = tgt[item_key][g_mask]
        total_gt += g_items.shape[0]

        if p_items.shape[0] and g_items.shape[0]:
            ious = iou_fn(p_items, g_items)
        else:
            ious = torch.zeros((p_items.shape[0], g_items.shape[0]))
        order = torch.argsort(p_scores, descending=True) if p_scores.numel() else torch.empty(0, dtype=torch.long)
        per_image.append({"order": order, "scores": p_scores, "ious": ious, "num_gt": g_items.shape[0]})
    return per_image, total_gt


def _match_at_threshold(per_image, iou_threshold: float) -> list[tuple[float, bool]]:
    """Greedy per-image matching (highest score first) -> list of (score, is_true_positive)."""
    records: list[tuple[float, bool]] = []
    for entry in per_image:
        matched_gt = torch.zeros(entry["num_gt"], dtype=torch.bool)
        for local_i in entry["order"].tolist():
            score = float(entry["scores"][local_i].item())
            is_tp = False
            if entry["num_gt"]:
                row = entry["ious"][local_i].clone()
                row[matched_gt] = -1.0
                best_j = int(torch.argmax(row))
                if float(row[best_j]) >= iou_threshold:
                    matched_gt[best_j] = True
                    is_tp = True
            records.append((score, is_tp))
    return records


def _precision_recall(records: list[tuple[float, bool]], total_gt: int, score_threshold: float):
    tp = sum(1 for score, is_tp in records if score >= score_threshold and is_tp)
    fp = sum(1 for score, is_tp in records if score >= score_threshold and not is_tp)
    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return precision, recall


def _average_precision(records: list[tuple[float, bool]], total_gt: int) -> Optional[float]:
    """COCO-style 101-point interpolated AP for a single class/IoU threshold."""
    if total_gt == 0:
        return None
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda r: -r[0])
    tp_cum = 0
    fp_cum = 0
    precisions = np.empty(len(ordered))
    recalls = np.empty(len(ordered))
    for i, (_, is_tp) in enumerate(ordered):
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        precisions[i] = tp_cum / (tp_cum + fp_cum)
        recalls[i] = tp_cum / total_gt

    recall_levels = np.linspace(0, 1, 101)
    interpolated = np.zeros(101)
    for i, r in enumerate(recall_levels):
        eligible = precisions[recalls >= r]
        interpolated[i] = eligible.max() if eligible.size else 0.0
    return float(interpolated.mean())


def _detection_metrics(
    predictions,
    targets,
    class_ids: list[int],
    iou_fn,
    item_key: str,
    score_threshold: float,
    iou_match_threshold: float,
) -> dict[str, Optional[float]]:
    map_iou_range = np.arange(0.5, 1.0, 0.05)
    per_class_precision, per_class_recall = [], []
    per_class_ap50, per_class_ap_range = [], []

    for class_id in class_ids:
        per_image, total_gt = _collect_class_matches(predictions, targets, class_id, iou_fn, item_key)
        if total_gt == 0:
            continue  # class absent from this split's GT: excluded, never fabricated as 0

        pr_records = _match_at_threshold(per_image, iou_match_threshold)
        precision, recall = _precision_recall(pr_records, total_gt, score_threshold)
        if precision is not None:
            per_class_precision.append(precision)
        if recall is not None:
            per_class_recall.append(recall)

        ap50_records = _match_at_threshold(per_image, 0.5)
        ap50 = _average_precision(ap50_records, total_gt)
        if ap50 is not None:
            per_class_ap50.append(ap50)

        aps_over_range = []
        for thr in map_iou_range:
            recs = _match_at_threshold(per_image, float(thr))
            ap = _average_precision(recs, total_gt)
            if ap is not None:
                aps_over_range.append(ap)
        if aps_over_range:
            per_class_ap_range.append(float(np.mean(aps_over_range)))

    def _avg(values: list[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    return {
        "precision": _avg(per_class_precision),
        "recall": _avg(per_class_recall),
        "mAP50": _avg(per_class_ap50),
        "mAP50_95": _avg(per_class_ap_range),
    }


def evaluate(
    model,
    data_loader,
    device: str,
    class_ids: list[int],
    confidence_threshold: float,
    iou_match_threshold: float,
):
    predictions, targets, latencies_s = _run_inference(model, data_loader, device)
    box_metrics = _detection_metrics(
        predictions, targets, class_ids, box_iou, "boxes", confidence_threshold, iou_match_threshold
    )
    mask_metrics = _detection_metrics(
        predictions, targets, class_ids, _mask_iou, "masks", confidence_threshold, iou_match_threshold
    )
    latency_ms_per_image = float(np.mean(latencies_s)) * 1000.0 if latencies_s else None
    return box_metrics, mask_metrics, latency_ms_per_image


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _default_images_root(manifest_path: Path) -> Path:
    """Read images_root from the manifest's own ``source_dir`` field, so
    manifest.json and images_root can't silently drift out of sync."""
    manifest_path = Path(manifest_path)
    try:
        source_dir = json.loads(manifest_path.read_text(encoding="utf-8"))["source_dir"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{manifest_path} has no 'source_dir' field -- rebuild it with "
            "dataset/prepare_cardd.py (which always records it), or pass "
            "--images-root explicitly to override."
        ) from exc
    return Path(source_dir)


def _tbd(value: Optional[float]):
    """Per experiments.json's report_unknown_values_as convention: never fabricate a number."""
    return value if value is not None else "TBD"


def _hardware_info() -> dict:
    """Same shape as train_yolo.py's hardware block, for side-by-side comparison."""
    return {"platform": platform.platform(), "processor": platform.processor() or None}


def train(
    manifest: Path,
    output_dir: Path,
    images_root: Optional[Path] = None,
    epochs: int = 20,
    image_size: int = 800,
    batch_size: int = 2,
    eval_batch_size: int = 1,
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    num_workers: int = 2,
    hflip_prob: float = 0.5,
    grad_clip_norm: float = 0.0,
    confidence_threshold: float = 0.5,
    iou_match_threshold: float = 0.5,
    evaluation_split: str = "validation",
    checkpoint_every: int = 5,
    seed: int = 42,
    pretrained: bool = True,
    device: Optional[str] = None,
    resume_from: Optional[Path] = None,
    max_images_per_split: Optional[int] = None,
    eval_every: int = 1,
    early_stop_patience: int = 0,
    use_amp: bool = False,
) -> dict:
    _require_torch()
    torch.manual_seed(seed)

    manifest = Path(manifest)
    output_dir = Path(output_dir)
    images_root = Path(images_root) if images_root else _default_images_root(manifest)
    if not images_root.exists():
        raise FileNotFoundError(
            f"images_root does not exist: {images_root}. Pass --images-root explicitly "
            "pointing at your local CarDD_COCO directory (this dataset is not committed to git)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Auto-detect CUDA, then MPS, then CPU (same convention as train_yolo.py --device).
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cpu":
        print("Warning: training on CPU will be slow. Consider --max-images-per-split for a smoke test.")

    debug_subset = max_images_per_split is not None
    if debug_subset:
        print(
            f"DEBUG SUBSET: using at most {max_images_per_split} images per split. "
            "Do not report these numbers as real results (see README.md)."
        )

    train_transform = DetectionTransform(hflip_prob=hflip_prob)
    train_dataset = CarDDInstanceSegmentation(
        manifest, images_root, split="train", transform=train_transform, max_images=max_images_per_split
    )

    eval_transform = DetectionTransform(hflip_prob=0.0)
    eval_dataset = CarDDInstanceSegmentation(
        manifest,
        images_root,
        split=evaluation_split,
        transform=eval_transform,
        category_id_to_label=train_dataset.category_id_to_label,
        classes=train_dataset.classes,
        max_images=max_images_per_split,
    )
    if evaluation_split == "test":
        print("Evaluating on the held-out TEST split -- use only for final course reporting.")

    # pin_memory + persistent_workers: free speed wins, don't affect training dynamics.
    use_pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    class_ids = list(range(1, len(train_dataset.classes) + 1))
    num_classes = len(train_dataset.classes) + 1  # +1 for background

    model = build_model(num_classes, image_size, pretrained=pretrained).to(device)

    start_epoch = 1
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    for _ in range(start_epoch - 1):
        lr_scheduler.step()

    history = []
    best_mask_map50 = -1.0
    patience_counter = 0
    best_ckpt_path = checkpoints_dir / "best.pt"
    early_stopped = False
    train_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            grad_clip_norm=grad_clip_norm, use_amp=use_amp,
        )
        lr_scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["total_loss"],
                "loss_components": {k: v for k, v in train_stats.items() if k != "total_loss"},
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
            }
        )
        print(f"[epoch {epoch}/{epochs}] loss={train_stats['total_loss']:.4f} ({epoch_seconds:.1f}s)")

        # --- periodic validation, best-checkpoint tracking, early stopping ---
        if epoch % eval_every == 0 or epoch == epochs:
            val_start = time.perf_counter()
            val_box, val_mask, _ = evaluate(
                model, eval_loader, device, class_ids, confidence_threshold, iou_match_threshold
            )
            val_seconds = time.perf_counter() - val_start
            val_mask_map50 = val_mask.get("mAP50") or 0.0
            val_box_map50 = val_box.get("mAP50") or 0.0
            history[-1]["val_mask_mAP50"] = val_mask_map50
            history[-1]["val_box_mAP50"] = val_box_map50
            history[-1]["val_seconds"] = val_seconds
            print(
                f"[epoch {epoch}/{epochs}] val mask_mAP50={val_mask_map50:.4f} "
                f"box_mAP50={val_box_map50:.4f} (val {val_seconds:.1f}s)"
            )

            if val_mask_map50 > best_mask_map50:
                best_mask_map50 = val_mask_map50
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "classes": train_dataset.classes,
                        "category_id_to_label": train_dataset.category_id_to_label,
                        "mask_mAP50": val_mask_map50,
                    },
                    best_ckpt_path,
                )
            else:
                patience_counter += 1
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch}: val mask_mAP50 did not improve "
                        f"for {patience_counter} evals. Best mask_mAP50={best_mask_map50:.4f}."
                    )
                    early_stopped = True
                    break

        if epoch % checkpoint_every == 0 or epoch == epochs:
            ckpt_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "classes": train_dataset.classes,
                    "category_id_to_label": train_dataset.category_id_to_label,
                },
                ckpt_path,
            )

    train_seconds = time.time() - train_start

    # Load best checkpoint for final evaluation when periodic validation was active
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(
            f"Loaded best checkpoint from epoch {best_ckpt.get('epoch', '?')} "
            f"(mask_mAP50={best_ckpt.get('mask_mAP50', 'N/A')}) for final eval."
        )

    eval_start = time.perf_counter()
    box_metrics, mask_metrics, latency_ms_per_image = evaluate(
        model, eval_loader, device, class_ids, confidence_threshold, iou_match_threshold
    )
    eval_seconds = time.perf_counter() - eval_start

    final_ckpt = checkpoints_dir / "final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "category_id_to_label": train_dataset.category_id_to_label,
            "image_size": image_size,
        },
        final_ckpt,
    )

    summary = {
        # --- fields kept in the same order/name as train_yolo.py's summary ---
        "method": "mask_rcnn",
        "model_weights": "maskrcnn_resnet50_fpn (pretrained)" if pretrained else "maskrcnn_resnet50_fpn (from scratch)",
        "epochs": epochs,
        "image_size": image_size,
        "seed": seed,
        "device": device,
        "confidence_threshold": confidence_threshold,
        "iou_match_threshold": iou_match_threshold,
        "train_seconds": round(train_seconds, 1),
        "status": "debug_subset_do_not_report" if debug_subset else "completed",
        "evaluation_split": evaluation_split,
        "box_metrics": {k: _tbd(v) for k, v in box_metrics.items()},
        "mask_metrics": {k: _tbd(v) for k, v in mask_metrics.items()},
        "latency_ms_per_image": _tbd(latency_ms_per_image),
        "hardware": _hardware_info(),
        "note": (
            "precision/recall/mAP are computed by this script's own COCO-style matcher "
            "(no pycocotools), for consistent comparison with the YOLO baseline; see the "
            "module docstring for the matching protocol."
        ),
        # --- mask_rcnn-specific fields, not present in the YOLO summary ---
        "manifest": str(manifest),
        "images_root": str(images_root),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "classes": train_dataset.classes,
        "train_images": len(train_dataset),
        "evaluation_images": len(eval_dataset),
        "eval_seconds": round(eval_seconds, 1),
        "train_history": history,
        "final_checkpoint": str(final_ckpt),
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path.exists() else None,
        "best_mask_mAP50": best_mask_map50 if best_mask_map50 >= 0 else None,
        "early_stopped": early_stopped,
        "debug_subset": debug_subset,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to cardd_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write checkpoints + summary")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Local CarDD_COCO directory (default: read from --manifest's own 'source_dir' field)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005, dest="learning_rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="0 disables gradient clipping")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="see experiments.json")
    parser.add_argument("--iou-match-threshold", type=float, default=0.5, help="see experiments.json")
    parser.add_argument(
        "--evaluation-split", type=str, default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="see experiments.json dataset.seed")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", default=True)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. 'cpu', 'cuda', 'mps'; default auto-detects CUDA, then MPS, then CPU "
        "(same convention as training/train_yolo.py's --device)",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="DEBUG ONLY: cap images per split for a smoke test; results must not be reported",
    )
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs (default 1)")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N consecutive validations without val mask_mAP50 improvement; 0 disables early stopping",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        dest="use_amp",
        default=False,
        help="Enable mixed-precision (AMP) training — requires CUDA GPU (e.g. Colab), speeds up ~30-50%%",
    )
    args = parser.parse_args()

    summary = train(
        manifest=args.manifest,
        output_dir=args.output_dir,
        images_root=args.images_root,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        hflip_prob=args.hflip_prob,
        grad_clip_norm=args.grad_clip_norm,
        confidence_threshold=args.confidence_threshold,
        iou_match_threshold=args.iou_match_threshold,
        evaluation_split=args.evaluation_split,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        pretrained=args.pretrained,
        device=args.device,
        resume_from=args.resume_from,
        max_images_per_split=args.max_images_per_split,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        use_amp=args.use_amp,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
