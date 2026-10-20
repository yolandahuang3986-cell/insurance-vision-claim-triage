# Third-party and provenance notices

This inventory identifies dependencies and source provenance; it does not grant a new project-wide license, replace required upstream license text when redistributing it, or certify legal compliance.

| Component | Source and license evidence | Project use |
|---|---|---|
| Ultralytics YOLO11 | AGPL-3.0 (LICENSE file of the official ultralytics/ultralytics repository); separate enterprise terms available | optional external dependency; recorded version 8.4.138 |
| torchvision detection | BSD-3-Clause (LICENSE file of the official pytorch/vision repository) | external Mask R-CNN/ResNet-50 FPN dependency |
| pycocotools / COCO API | official `cocodataset/cocoapi` project; verify the installed package's notices before redistribution | external evaluator dependency, not vendored here |
| CarDD | official project and licensing form, research consent and redistribution restrictions | separately obtained runtime data, not included in the ZIP |
| Recovered Mask trainer | team-provided Colab source retained in `experiments/legacy_maskrcnn.py` | historical provenance and corrective training wrapper |
| Corrective scripts | original local experiment scripts now packaged under `experiments/` | executed algorithm snapshots, with a documented input-root portability change |

The audit does not bundle upstream Python packages, pretrained weights or datasets. Do not remove copyright/license headers from any copied code. Before public release, confirm team ownership/permission and select a compatible license with the relevant rights holders; no MIT/Apache/AGPL label is assigned to all project code automatically.
