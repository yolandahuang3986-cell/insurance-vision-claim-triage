# Report experiments and source-code coverage

The authoritative numerical evidence is the frozen experiment exports summarized in `reporting/metrics.json`. A new code audit is not a rerun of training. Report/PPT/contribution files remain separate course documents, not generated from these templates automatically.

| Report evidence | Code | Included aggregate evidence |
|---|---|---|
| Part 2 historical training and native evaluation | `CarDD project/cardd_code/training/`, `CarDD project/cardd_code/evaluation/` | native records retained upstream; compact common metrics are kept separate |
| Shared evaluator, Table 5, class errors/Table 6 | `experiments/unified_eval.py`, `analyze_classes.py` | `quality`, `class_diagnostics` |
| Mask size retraining, Tables 7–8/Figure 4 | `train_resolution.py`, `legacy_maskrcnn.py` | `quality`, `training_histories` |
| YOLO size retraining, Table 9/Figure 5 | `train_yolo_resolution.py`, `unified_eval.py` | `quality`, `training_histories` |
| Components and staged crack coverage, Table 10 | `inference_components.py` | `components` |
| Validation reconstruction choice, Table 11 | `validate_retina_and_time.py` | `reconstruction_selection`, `selected_test` |
| Four original / nine final calibrated settings, Table 13/Figure 6 | `calibrate_thresholds.py`, `calibrate_followup.py` | `calibration`, `validation_policies` |
| Area AP, historical and Mask-size AP intervals | `posthoc_analysis.py` | `scale_metrics`, `bootstrap_ap` |
| YOLO True size AP interval | `bootstrap_yolo_resolution.py` | `bootstrap_yolo_true` |
| Selected reconstruction / size AP intervals | `bootstrap_selected_ap.py` | `bootstrap_selected_ap` |
| Calibrated F1 intervals | `bootstrap_calibrated_f1.py`, `calibrate_followup.py` | `bootstrap_calibrated_f1` |
| Historical paired timing and Table 12 retrained cohort | `inference_components.py`, `validate_retina_and_time.py` | `timing`, two separately named cohorts |
| Figures 1, 2, 4, 5, 6 numerical reproduction | `reporting/plot_metrics.py` | compact histories, curves, AP and CI |
| Figure 3 photographic comparisons | `experiments/make_comparison_figures.py` | intentionally requires separately authorized photos and instance exports |

## Known boundaries

- Historical native metrics (Part 2 / original comparison) and common COCO metrics (Part 3) are not interchangeable. Historical Mask was test-selected; later training is validation-selected.
- Historical YOLO's epoch CSV is unavailable. Later complete training histories do not retroactively reconstruct it.
- `make_comparison_figures.py` retains an exploratory single-pass latency scatter. The final evidence uses the paired timing cohorts, not that scatter.
- The later copied follow-up scripts preserve executed algorithms. Only the YOLO size-bootstrap input root was changed to a CLI argument for portability.
- An inference-component sensitivity study is not a retrained FPN/RoIAlign ablation. Downstream crack loss is not exclusive attribution to one head.
- A confidence interval spanning zero is not proof of equivalence. The selected complete configuration is a pilot candidate, not a validated claim-decision system.

See `reproduction.md` for ordering, required local-input layouts and exact output names.
