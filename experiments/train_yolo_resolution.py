"""Paired YOLO11n-seg resolution retraining; validation-only mask AP50 selection.

Native YOLO validation metrics select checkpoints. Final original-mask COCOeval
is a separate diagnostic and does not influence checkpoint selection.
"""
import argparse
import csv
import json
import platform
import time
from copy import copy
from pathlib import Path

from unified_eval import digest


def main():
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator

    class MaskAP50Validator(SegmentationValidator):
        def get_stats(self):
            stats = super().get_stats()
            stats['fitness'] = stats['metrics/mAP50(M)']
            return stats

    class MaskAP50Trainer(SegmentationTrainer):
        def get_validator(self):
            return MaskAP50Validator(self.test_loader, save_dir=self.save_dir,
                                     args=copy(self.args), _callbacks=self.callbacks)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--size', type=int, choices=[640, 800], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run on an allocated school GPU, not the login node.')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; no overwrite or automatic resume.')
    args.output.mkdir(parents=True)
    data = yaml.safe_load(args.data.read_text())
    data['path'] = str(args.data.resolve().parent)
    # Training never has a test entry available; test evaluation is separate.
    data.pop('test', None)
    train_yaml = args.output / 'train_val.yaml'
    train_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    config = dict(data=str(train_yaml.resolve()), epochs=1 if args.smoke else args.epochs,
                  imgsz=args.size, batch=16, seed=42, deterministic=True,
                  device=0, workers=4, cache=False, amp=True, optimizer='AdamW',
                  lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                  warmup_epochs=3.0, cos_lr=False, patience=0,
                  multi_scale=False, rect=False, close_mosaic=10,
                  mosaic=1.0, mixup=0.0, copy_paste=0.0, fliplr=0.5, flipud=0.0,
                  mask_ratio=4, overlap_mask=True, save=True, save_period=-1,
                  plots=True, val=True, project=str(args.output.resolve()),
                  name='training', exist_ok=False, fraction=0.01 if args.smoke else 1.0)
    metadata = {'status': 'running', 'config': config,
                'selection': 'maximum native validation mask AP50; ties latest epoch',
                'initial_sha256': digest(args.initial), 'data_yaml_sha256': digest(args.data),
                'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                'test_previously_exposed': True, 'smoke': args.smoke}
    record = args.output / 'run_manifest.json'
    record.write_text(json.dumps(metadata, indent=2))
    started = time.time()
    YOLO(str(args.initial)).train(trainer=MaskAP50Trainer, **config)
    training = args.output / 'training'
    with (training / 'results.csv').open() as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(enumerate(rows), key=lambda x: (float(x[1]['metrics/mAP50(M)']), x[0]))[1]
    checkpoint = training / 'weights/best.pt'
    assert checkpoint.exists()
    metadata.update(status='complete', training_seconds=time.time()-started,
                    best_epoch=int(float(best['epoch'])), selected_validation_row=best,
                    best_sha256=digest(checkpoint), results_csv_sha256=digest(training/'results.csv'))
    record.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
