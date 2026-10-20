# Corrective CarDD experiments

For the completed nine-configuration workflow, use [docs/reproduction.md](../docs/reproduction.md) and [docs/experiment_map.md](../docs/experiment_map.md). Later executed scripts now included here are `validate_retina_and_time.py`, `calibrate_followup.py`, `bootstrap_calibrated_f1.py`, `bootstrap_selected_ap.py` and `bootstrap_yolo_resolution.py` (`--root` selects the analysis directory). `reporting/` provides aggregate-only evidence checking and numerical plots; no GPU training is rerun by that path.

This separate path preserves the existing MVP and training entrypoints. `legacy_maskrcnn.py` is a recovered snapshot of the group-provided Colab trainer, not a replacement for `training/train_mask_rcnn.py`.

## Scientific scope

1. **Common evaluation**: evaluate existing YOLO and Mask R-CNN weights against original COCO instance masks using pycocotools, score floor 0.001, and maxDets 100. This fixes evaluator alignment but cannot undo test-based selection of the old Mask R-CNN checkpoint.
2. **Fixed-weight inference-size sweep**: reuse the same Mask R-CNN weights at 640/1066 and 800/1333. This is inference sensitivity, not a retraining ablation and not an equal-geometry comparison with YOLO letterboxing.
3. **Controlled retraining**: independently retrain the two Mask R-CNN resolutions with seed 42, 20 epochs, batch 2, FP32, the same SGD/warmup/cosine/flipping recipe, and validation-only checkpoint selection. Both variants select official COCO mask AP50 against original validation annotations. The wrapper streams compressed RLE predictions rather than retaining all dense masks in RAM. Final test AP uses the same COCOeval. The official test set has already been observed and is not a pristine confirmatory holdout.

Do not combine old evaluator numbers and new COCOeval numbers in an unlabelled table. Conditional matched-mask IoU excludes misses. Fixed-threshold micro precision/recall are not interchangeable with class-averaged or best-F1 metrics. Scores from a four-image smoke run are not benchmark results.

## Reproduction

Use an isolated GPU environment. The actual experiment environment is recorded with every run; the recovered Colab YOLO log used Ultralytics 8.4.138 and PyTorch 2.11.0+cu128. Install appropriate PyTorch/torchvision CUDA wheels for the GPU driver, then:

```sh
pip install ultralytics==8.4.138 pycocotools==2.0.11 pillow numpy matplotlib
python -m unittest discover -s experiments -p test_diagnostic.py -v

python experiments/unified_eval.py \
  --root /path/to/CarDD_COCO/test2017 \
  --annotations /path/to/CarDD_COCO/annotations/instances_test2017.json \
  --weights /path/to/yolo_best.pt --method yolo --size 640 \
  --output outputs/common_yolo640

python experiments/unified_eval.py \
  --root /path/to/CarDD_COCO/test2017 \
  --annotations /path/to/CarDD_COCO/annotations/instances_test2017.json \
  --weights /path/to/mask_rcnn_final.pt --method mask --size 800 \
  --legacy experiments/legacy_maskrcnn.py --output outputs/common_mask800

python experiments/train_resolution.py \
  --legacy experiments/legacy_maskrcnn.py \
  --manifest /path/to/cardd_manifest.json --root /path/to/CarDD_COCO \
  --size 640 --output outputs/retrain640
```

Repeat the last command with size 800 and a separate output directory. Test only the frozen selected checkpoint. `final.pt` in the recovered trainer is saved **after best weights are reloaded**; file names alone cannot establish weight mismatch.

`unified_eval.py` records hashes, software/GPU, original image IDs, native-sized RLE predictions, common COCO AP, class AP, fixed-threshold counts and timing. Timing uses batch 1, FP32, five warmups, CUDA synchronisation, and prepared RGB image input through CPU mask export/RLE serialization; disk decoding is excluded. This is export-pipeline latency, not isolated model-forward latency. Pin NMS settings as recorded, and do not confuse them with AP matching thresholds.

Only load checkpoints from trusted project sources. YOLO deserialization can execute pickle code; unknown weights should not be loaded.

The exact tested Linux/CUDA environment's direct dependencies are pinned in `requirements-gpu.txt`; install it in a separate GPU environment with `pip install -r experiments/requirements-gpu.txt`, not into a Mac mock-app environment. CUDA wheel compatibility still depends on the host driver. Use `analyze_classes.py --run-root /path/to/local/results` to independently verify class totals; `make_comparison_figures.py --run-root /path/to/local/results` renders the paired report figures. See each script's help for the expected original annotation and image inputs. No external file link substitutes for these source files.

Dataset images, original annotation JSON, trained weights and the environment are external runtime inputs and must not be included in the course code ZIP. The source-code package must remain usable without cloud-hosted substitute code links.

## Additional diagnostics (9 October revision)

`posthoc_analysis.py` implements official COCO area AP and paired image-bootstrap AP differences. Its cached accumulation is checked against official COCOeval, including duplicated image occurrences. Bootstrap intervals describe evaluation-image sampling uncertainty for fixed checkpoints, not training variability or checkpoint-selection bias.

`inference_components.py` implements fixed-weight Mask R-CNN proposal-budget, box-NMS and score-floor sensitivity, staged proposal/box/mask coverage, YOLO retina-mask comparison, validation prediction exports, and five-round interleaved timing. These experiments were executed for the report. Coverage is a per-ground-truth diagnostic, not one-to-one detection recall; downstream misses cannot automatically be attributed exclusively to the mask head.

`calibrate_thresholds.py` selects global and crack/scratch/dent thresholds on validation predictions, freezes the policy before loading test predictions, and compares fixed-threshold precision/recall/F1 and threshold-truncated AP. Filtering does not improve the underlying score ranking; truncated AP must not be presented as a new full-ranking model AP. Validation exports are required before execution.

Run `python -m unittest discover -s experiments -p test_posthoc.py -v` in an environment with NumPy and pycocotools. Each diagnostic script documents its inputs through `--help`. Keep all runtime predictions, bootstrap samples, datasets and weights outside the submission archive.

`train_yolo_resolution.py` adds a paired YOLO11n-seg 640/800 experiment: identical COCO-pretrained initialization, 50 epochs, batch 16, seed 42, AMP, fixed AdamW and augmentation settings. Only input size differs. It removes the test entry from its training YAML and explicitly selects maximum native validation mask AP50 (not Ultralytics' default joint fitness). This selection metric uses YOLO labels and is not identical to original-mask COCOeval. Save `training/results.csv`, `training/args.yaml` and `run_manifest.json`; evaluate frozen weights with `unified_eval.py` only after both selections have completed. This is a new single-seed controlled experiment, not an exact replication of historical auto-batch training or a matched-training-budget cross-architecture experiment.
