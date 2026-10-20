# Compact evidence bundle

`metrics.json` is a curated aggregate-only export. The 24 corrective result/history sources are identified by SHA-256 under `source_sha256`. `historical_native` separately retains the two upstream Colab aggregate and matched-IoU summaries, so their original scores are not confused with common COCO evaluation. The bundle retains quality, class diagnostics, scale metrics, intervals, stage coverage, validation choices, nine calibrated operating points, two timing cohorts and compact histories. No images, original annotations, per-image instance exports, mask RLE, weights or credentials are included.

```sh
python reporting/verify_metrics.py
python reporting/plot_metrics.py --output outputs/report_figures
```

The first command verifies arithmetic and protocol totals without inference. The second recreates source-driven numerical evidence behind Figures 1, 2, 4, 5 and 6, not exact Word layout. Figure 3's photographic comparison needs authorized images and predictions; see `docs/reproduction.md`.

Use `docs/experiment_map.md` to locate table/figure sources. This bundle is frozen evidence, not synthetic test data or a substitute for full model reproduction. Raw experiment outputs stay outside the package. Preserve image-sampling uncertainty, prior test exposure and single-seed limitations when interpreting it.
