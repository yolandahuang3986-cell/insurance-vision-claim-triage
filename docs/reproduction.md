# Reproduction guide

## Environments and authorized inputs

Run commands from the repository root. Basic mock tests need Python 3.9+ and pytest. CPU post-processing/plotting additionally needs `experiments/requirements-analysis.txt`. The delivery audit uses a fresh Python 3.12 environment; it does not claim CUDA equivalence.

For recorded GPU experiments: Linux, Python 3.13.5, NVIDIA A40, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.138. Install `experiments/requirements-gpu.txt` in a separate environment after checking GPU driver compatibility. Run inference and training only on an allocated GPU, never a login node. Only deserialize trusted project weights.

CarDD must be obtained under its license, not from this ZIP. Expected layout:

```text
CarDD_COCO/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  annotations/instances_test2017.json
  train2017/
  val2017/
  test2017/
```

Set these task-specific shell variables to your local locations:

```sh
CARDD=/path/to/authorized/CarDD_COCO
WORK=/path/to/new/analysis_workspace
GPU_WORK=/path/to/new/gpu_input_workspace
MANIFEST=/path/to/cardd_manifest.json
YOLO_DATA=/path/to/cardd_yolo/data.yaml
```

`WORK` stores evaluation exports and histories. `GPU_WORK` stores local input links expected by the historical multi-model scripts. Neither belongs in the submission archive. Directory names containing dates or `40682` below are compatibility labels, not scheduler job instructions.

## A. Data preparation and training

```sh
python -m dataset.prepare_cardd --source-dir "$CARDD" --output "$MANIFEST"
python -m dataset.convert_coco_to_yolo --source-dir "$CARDD" --output-dir /path/to/cardd_yolo
python experiments/train_resolution.py --legacy experiments/legacy_maskrcnn.py --manifest "$MANIFEST" --root "$CARDD" --size 640 --output "$WORK/retrain640"
python experiments/train_resolution.py --legacy experiments/legacy_maskrcnn.py --manifest "$MANIFEST" --root "$CARDD" --size 800 --output "$WORK/retrain800"
python experiments/train_yolo_resolution.py --data "$YOLO_DATA" --initial /path/to/trusted/yolo11n-seg.pt --size 640 --output "$WORK/yolo_pair_20261009/yolo640"
python experiments/train_yolo_resolution.py --data "$YOLO_DATA" --initial /path/to/trusted/yolo11n-seg.pt --size 800 --output "$WORK/yolo_pair_20261009/yolo800"
```

These train independent size variants with validation-only selection. Mask uses 20 epochs, batch two, FP32, seed 42 and official validation COCO mask AP50. YOLO uses 50 epochs, batch 16, AMP, seed 42 and maximum native validation mask AP50 (not default combined fitness). The historical auto-batch YOLO run is a different recipe; its missing epoch CSV cannot be reconstructed from later runs. The historical Mask checkpoint was test-selected; use its records for the historical comparison, not as a recommended selection protocol.

## B. Shared COCO evaluation

```sh
python experiments/unified_eval.py --root "$CARDD/test2017" --annotations "$CARDD/annotations/instances_test2017.json" --weights /path/to/historical_yolo_best.pt --method yolo --size 640 --output "$WORK/legacy_yolo640"
python experiments/unified_eval.py --root "$CARDD/test2017" --annotations "$CARDD/annotations/instances_test2017.json" --weights /path/to/historical_mask_final.pt --method mask --size 800 --legacy experiments/legacy_maskrcnn.py --output "$WORK/legacy_mask_inference800"
```

Repeat Mask with the **same historical weights**, size 640 and destination `legacy_mask_inference640` for inference-only resizing. Repeat using each retrained Mask checkpoint with its own size into `retrain640` and `retrain800`. Repeat YOLO using each retrained checkpoint/size into `yolo_pair_20261009/yolo{size}/test_unified`.

Outputs: `summary.json`, native-image `predictions.json` and `cases.json`. AP uses original COCO masks, score floor 0.001 and maxDets 100; fixed-threshold errors use confidence 0.5. These dense prediction exports remain runtime inputs, excluded from the archive. The single-pass timing recorded here is exploratory, not the final paired benchmark.

## C. Multi-model layout and component experiments

Copy `configs/reproduction_inputs.example.json` to a local file, fill in the dataset and all six trusted weight paths, then:

```sh
python experiments/prepare_workspace.py --config /path/to/local_inputs.json --output "$GPU_WORK"
```

The destination must be fresh. Paths in the config are relative to its directory unless absolute. The helper links `train_mask_rcnn.py` to the packaged legacy trainer, historical weights to `yolo_best.pt`/`mask_rcnn_final.pt`, retrained weights to the original `results/.../best.pt` layout, and authorized annotations under `evidence/`. It records the dataset directory in `data/dataset_root.txt`; it does not download or deserialize weights.

```sh
python experiments/inference_components.py --root "$GPU_WORK" --output "$WORK/components_smoke" --mode smoke
python experiments/inference_components.py --root "$GPU_WORK" --output "$WORK/components_20261009" --mode full
python experiments/validate_retina_and_time.py --root "$GPU_WORK" --output "$WORK/validation_followup_40682" --rounds 5
```

The component job writes `components/components.json`, stage records, validation predictions and a historical five-configuration timing cohort. It changes only proposal cap, box NMS or score floor for the frozen historical Mask checkpoint. Coverage is per GT, not one-to-one detection recall or proof that the mask head alone causes misses.

The follow-up requires A40 for timing comparability. It compares reconstruction True/False on all 810 validation images for three YOLO checkpoints, freezes every choice before selected-test quality evaluation, then times six retrained configurations. Outputs include `frozen_validation_selection.json`, `selected_test_metrics.json`, `validation/`, `selected_test/` and `paired_timing.json`. Use a distinct fresh directory for `--smoke`; it cannot be resumed as a full run.

Both timing cohorts use FP32 batch one, five warmups, five rotating/interleaved rounds, CUDA synchronization, prepared RGB through CPU mask transfer and RLE encoding, score floor 0.001. Disk decode and diagnostic hooks are excluded. Do not pool cohorts or describe these as model-forward speed or calibrated-cutoff latency.

## D. CPU post-processing, calibration and intervals

Link authorized validation/test annotations into `WORK/evidence/` for the analysis scripts:

```sh
mkdir -p "$WORK/evidence"
ln -s "$CARDD/annotations/instances_val2017.json" "$WORK/evidence/instances_val2017.json"
ln -s "$CARDD/annotations/instances_test2017.json" "$WORK/evidence/instances_test2017.json"
python experiments/posthoc_analysis.py --root "$WORK" --replicates 2000
python experiments/calibrate_thresholds.py --root "$WORK" --validation "$WORK/components_20261009/validation"
python experiments/bootstrap_yolo_resolution.py --root "$WORK"
python experiments/bootstrap_calibrated_f1.py --root "$WORK" --output "$WORK/calibrated_f1_bootstrap.json" --replicates 2000
python experiments/calibrate_followup.py --root "$WORK" --followup "$WORK/validation_followup_40682" --output "$WORK/followup_revision/calibration"
python experiments/bootstrap_selected_ap.py --root "$WORK" --followup "$WORK/validation_followup_40682" --output "$WORK/followup_revision/selected_ap_bootstrap.json"
```

`calibrate_followup.py` expands the original four policies to all nine configurations and verifies previous-policy equality. The score grid, max-validation-F1 rule and higher-cutoff tie break are retained. Class-specific changes apply to dent/scratch/crack; other classes keep the global cutoff. Filtering is not probability calibration or an improved full-ranking AP.

Bootstrap uses complete paired images and re-aggregates micro F1 or COCO AP, 2,000 draws, seed 6129. Cached AP is checked against COCOeval including duplicated images. Intervals condition on frozen checkpoints, thresholds and reconstruction; they exclude training/tuning variance. Earlier test exposure remains a limitation.

## E. Small evidence bundle and figures

```sh
python reporting/verify_metrics.py
python reporting/plot_metrics.py --output outputs/report_figures
python -m pytest -q
python -m unittest discover -s experiments -p 'test_*.py' -v
```

The bundle is a curated aggregate snapshot with source hashes, not all result directories. Numeric plots reproduce the evidence behind Figures 1, 2, 4, 5 and 6, not pixel-identical layout. Figure 3 additionally needs authorized images, predictions and cases: `make_comparison_figures.py --run-root "$WORK"` reads the first four image files from `WORK/evidence/smoke_images/`. Its older latency scatter is exploratory and must not replace the final paired timing table.

Build the offline package with `python scripts/build_submission_zip.py --group-id 15`. Review its verified manifest, then test the extracted ZIP independently. No dataset, weights, runtime predictions or cloud-hosted substitute source file should be added to fix a reproduction gap.
