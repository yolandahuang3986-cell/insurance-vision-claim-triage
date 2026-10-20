"""Common COCO mask evaluation for trusted, user-provided CarDD checkpoints.

Diagnostic evaluation of exposed test data, NOT a new confirmatory benchmark.
Inference resolution sweep keeps weights fixed: not a retraining ablation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def rle(binary):
    value = mask_util.encode(np.asfortranarray(binary.astype(np.uint8)))
    value['counts'] = value['counts'].decode('ascii')
    return value


def diagnostic(preds, gts, floor=0.5, threshold=0.5):
    """Score-sorted same-class one-to-one matches; counts include empty images."""
    selected = sorted((p for p in preds if p['score'] >= floor),
                      key=lambda p: (-p['score'], p['category_id']))
    used, overlaps = set(), []
    for pred in selected:
        eligible = [(i, gt) for i, gt in enumerate(gts)
                    if i not in used and gt['category_id'] == pred['category_id']]
        if not eligible:
            continue
        ious = mask_util.iou([pred['segmentation']],
                            [g['segmentation'] for _, g in eligible], [0]*len(eligible))[0]
        best = int(np.argmax(ious))
        if ious[best] >= threshold:
            used.add(eligible[best][0])
            overlaps.append(float(ious[best]))
    return {'matched': len(used), 'missed': len(gts)-len(used),
            'unmatched': len(selected)-len(used), 'predictions': len(selected),
            'ious': overlaps}


def verified_mask_mapping(checkpoint, categories):
    ordered = sorted(categories)
    expected = {cid:i+1 for i,cid in enumerate(ordered)}
    provided = {int(k):int(v) for k,v in checkpoint.get('category_id_to_label',{}).items()}
    names = [categories[cid]['name'] for cid in ordered]
    if provided != expected or list(checkpoint.get('classes',[])) != names:
        raise ValueError('Checkpoint class mapping does not match original COCO categories.')
    return {label:cid for cid,label in expected.items()}


def coco_metrics(coco, preds, image_ids):
    if preds:
        dt = coco.loadRes(copy.deepcopy(preds))
    else:
        dt = COCO()
        dt.dataset = {'images': coco.dataset['images'], 'categories': coco.dataset['categories'],
                      'annotations': [], 'info': {}}
        dt.createIndex()
    ev = COCOeval(coco, dt, 'segm')
    ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        values = ev.eval['precision'][:, :, k, 0, -1]
        valid = values[values >= 0]
        per_class[coco.cats[cid]['name']] = float(valid.mean()) if len(valid) else None
    return {'AP50_95': float(ev.stats[0]), 'AP50': float(ev.stats[1]),
            'AR100': float(ev.stats[8]), 'per_class_AP50_95': per_class}


def main():
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--method', choices=['yolo', 'mask'], required=True)
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--legacy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('GPU evaluation must run on an allocated GPU node.')
    coco = COCO(str(args.annotations))
    ids = sorted(coco.imgs)
    if args.limit:
        ids = ids[:args.limit]
    if args.method == 'mask':
        spec = importlib.util.spec_from_file_location('legacy_mask', args.legacy)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ckpt = torch.load(args.weights, map_location='cpu', weights_only=True)
        model = module.build_model(len(coco.cats)+1, args.size, pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 100
        model.to(device).eval()
        class_map = verified_mask_mapping(ckpt, coco.cats)
        def infer(im, image_id):
            with torch.inference_mode():
                o = model([to_tensor(im).to(device)])[0]
            return [{'image_id': image_id, 'category_id': class_map[int(label)],
                     'score': float(score), 'segmentation': rle(mask[0].cpu().numpy() >= 0.5)}
                    for mask, label, score in zip(o['masks'], o['labels'], o['scores'])]
    else:
        import ultralytics
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        categories = {v['name']: k for k, v in coco.cats.items()}
        class_map = {k: categories[v] for k, v in model.names.items()}
        def infer(im, image_id):
            o = model.predict(im, imgsz=args.size, conf=0.001, iou=0.5, max_det=100,
                              retina_masks=True, device=0, half=False, verbose=False)[0]
            if o.masks is None:
                return []
            assert tuple(o.masks.data.shape[-2:]) == (im.height, im.width)
            return [{'image_id': image_id, 'category_id': class_map[int(label)],
                     'score': float(score), 'segmentation': rle(mask.cpu().numpy() >= 0.5)}
                    for mask, label, score in zip(o.masks.data, o.boxes.cls, o.boxes.conf)]
    first = Image.open(args.root / coco.imgs[ids[0]]['file_name']).convert('RGB')
    for _ in range(5):
        infer(first, ids[0])
    preds, timing, cases = [], [], []
    totals = {'matched': 0, 'missed': 0, 'unmatched': 0, 'predictions': 0}
    overlaps = []
    for n, image_id in enumerate(ids):
        im = Image.open(args.root / coco.imgs[image_id]['file_name']).convert('RGB')
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = infer(im, image_id)
        torch.cuda.synchronize()
        timing.append((time.perf_counter()-start)*1000)
        preds.extend(result)
        gts = [dict(a, segmentation=coco.annToRLE(a)) for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not a.get('iscrowd', 0)]
        d = diagnostic(result, gts)
        for key in totals:
            totals[key] += d[key]
        overlaps.extend(d['ious'])
        cases.append({'image_id': image_id, **{k:v for k,v in d.items() if k != 'ious'}})
        if n % 25 == 0:
            print(f'{args.method} {args.size}: {n+1}/{len(ids)}', flush=True)
    totals['micro_precision'] = totals['matched']/max(totals['predictions'], 1)
    totals['micro_recall'] = totals['matched']/max(totals['matched']+totals['missed'], 1)
    totals['matched_mask_IoU'] = float(np.mean(overlaps)) if overlaps else None
    summary = {'status': 'complete', 'diagnostic_only': True, 'test_previously_exposed': True,
               'method': args.method, 'size_setting': args.size, 'images': len(ids),
               'weights_sha256': digest(args.weights), 'annotations_sha256': digest(args.annotations),
               'torch': torch.__version__, 'torchvision': torchvision.__version__,
               'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
               'AMP': False, 'batch': 1, 'warmup': 5, 'score_floor': 0.001,
               'timing_scope': 'RGB PIL input through prediction, CPU transfer and RLE serialization; excludes disk decode',
               'latency_ms_mean': float(np.mean(timing)), 'latency_ms_median': float(np.median(timing)),
               'latency_ms_p95': float(np.percentile(timing, 95)),
               'coco_mask': coco_metrics(coco, preds, ids), 'fixed_threshold': totals}
    (args.output/'predictions.json').write_text(json.dumps(preds))
    (args.output/'cases.json').write_text(json.dumps(cases, indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
