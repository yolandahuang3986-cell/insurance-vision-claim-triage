# CarDD Instance Segmentation: YOLO vs Mask R-CNN

Course project code for CA6129 (Computer Vision & Applications). Compares a
YOLO segmentation baseline against a Mask R-CNN baseline on the CarDD car
damage instance-segmentation dataset, under a fixed fair-comparison protocol
(same seed, same held-out test split, same confidence / IoU-match thresholds).

Run via `CarDD_YOLO_vs_MaskRCNN.ipynb` (Colab notebook, kept separately) which
orchestrates the scripts below end-to-end on a GPU runtime. Each script also
runs standalone from the command line.

## Layout

```
dataset/
  prepare_cardd.py        official CarDD COCO annotations -> manifest JSON
  convert_coco_to_yolo.py manifest -> YOLO segmentation format (images + labels + data.yaml)

training/
  train_yolo.py           YOLO segmentation training + test-split evaluation (box/mask precision, recall, mAP50, mAP50:95, latency)
  train_mask_rcnn.py       Mask R-CNN (torchvision) training + test-split evaluation, same metric schema as train_yolo.py

evaluation/
  mask_iou.py              mean matched-pair mask IoU for the YOLO model on the test split
  mask_iou_maskrcnn.py     same protocol, for the Mask R-CNN checkpoint
  error_analysis.py        shared error-category classification (missed_damage, wrong_damage_class, poor_localization, ...)
  merge_results.py         merges a training_summary.json + a mask_iou_summary.json into one comparison-table row
  metrics.py                dependency-light IoU/precision/recall helpers
```

## Self-contained data reference

`dataset/prepare_cardd.py` records the exact directory it read images from
(as `source_dir`) inside the manifest it writes. Both `training/train_mask_rcnn.py`
and `evaluation/mask_iou*.py` read that field back automatically, so -- like
`train_yolo.py`'s single `data.yaml` -- passing `--manifest` alone is enough;
you don't have to separately track and re-pass a matching `--images-root` /
`--source-dir` by hand. Pass it explicitly only if you've moved the images
since the manifest was built.

## Protocol

- Seed 42, CarDD official train/validation/test split (2816 / 810 / 374 images).
- Both models fine-tune from a pretrained checkpoint (`yolo11n-seg.pt` /
  COCO-pretrained `maskrcnn_resnet50_fpn`) -- not trained from scratch --
  so the comparison is "pretrained baseline fine-tuned on CarDD" for both.
- Final numbers are reported on the held-out **test** split only, at
  confidence threshold 0.5 and IoU-match threshold 0.5.
- `evaluation/merge_results.py` produces one row per method with the same
  columns: mask_iou, precision, recall, mAP50, mAP50_95, latency_ms.

## Standalone usage

```bash
pip install -r requirements.txt

python -m dataset.prepare_cardd --source-dir data/raw/CarDD_release --output data/manifests/cardd_manifest.json
python -m dataset.convert_coco_to_yolo --source-dir data/raw/CarDD_release --output-dir data/yolo --link-mode copy

python -m training.train_yolo --data-yaml data/yolo/data.yaml --output-dir runs/yolo
python -m training.train_mask_rcnn --manifest data/manifests/cardd_manifest.json --output-dir runs/mask_rcnn --evaluation-split test

python -m evaluation.mask_iou --manifest data/manifests/cardd_manifest.json --weights runs/yolo/yolo_segmentation/weights/best.pt
python -m evaluation.mask_iou_maskrcnn --manifest data/manifests/cardd_manifest.json --checkpoint runs/mask_rcnn/checkpoints/final.pt

python -m evaluation.merge_results --training-summary runs/yolo/training_summary.json --mask-iou-summary evaluation/mask_iou_summary.json --method-label "YOLO Segmentation" --output evaluation/final_results_row_yolo.json
python -m evaluation.merge_results --training-summary runs/mask_rcnn/training_summary.json --mask-iou-summary evaluation/mask_iou_summary_mask_rcnn.json --method-label "Mask R-CNN" --output evaluation/final_results_row_mask_rcnn.json
```
