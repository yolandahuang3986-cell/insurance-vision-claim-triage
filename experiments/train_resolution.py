"""Independently retrain two Mask R-CNN resolution settings; no test selection."""
import argparse
import importlib.util
import json
import random
from pathlib import Path


def main():
    import numpy as np
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--size', type=int, required=True, choices=[640,800])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    spec = importlib.util.spec_from_file_location('legacy_train', args.legacy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_build = module.build_model
    def build(*pos, **kw):
        model = original_build(*pos, **kw)
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 100
        return model
    module.build_model = build
    # The legacy evaluator retains dense full-resolution masks for the entire
    # split. Use compressed per-image RLE and official COCO validation instead.
    from unified_eval import rle, coco_metrics
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import time
    def streaming_evaluate(model, loader, device, class_ids, confidence_threshold, iou_match_threshold):
        assert loader.dataset.split == 'validation', 'Selection may not access test.'
        coco = COCO(str(args.root/'annotations/instances_val2017.json'))
        inverse = {label:cid for cid,label in loader.dataset.category_id_to_label.items()}
        predictions, boxes, image_ids, elapsed = [], [], [], []
        model.eval()
        with torch.inference_mode():
            for n,(images,targets) in enumerate(loader):
                images = [im.to(device) for im in images]
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = model(images)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)/len(images)*1000)
                for out,target in zip(outputs,targets):
                    idx = int(target['image_id'].item())
                    iid = loader.dataset.records[idx]['source_id']
                    assert iid in coco.imgs, 'Manifest/COCO image mapping mismatch.'
                    image_ids.append(iid)
                    masks = out['masks'].detach().cpu().numpy()
                    for mask,label,score,box in zip(masks,out['labels'].tolist(),out['scores'].tolist(),out['boxes'].tolist()):
                        common = {'image_id':iid,'category_id':inverse[label],'score':score}
                        predictions.append(dict(common,segmentation=rle(mask[0]>=.5)))
                        x1,y1,x2,y2=box
                        boxes.append(dict(common,bbox=[x1,y1,x2-x1,y2-y1]))
                del outputs
                if n%100==0:
                    print(f'[validation COCO] {n+1}/{len(loader)}',flush=True)
        metrics = coco_metrics(coco,predictions,image_ids)
        if boxes:
            ev=COCOeval(coco,coco.loadRes(boxes),'bbox')
            ev.params.imgIds=image_ids
            ev.evaluate();ev.accumulate();ev.summarize()
            box_ap50=float(ev.stats[1])
        else:
            box_ap50=0.0
        return {'mAP50':box_ap50}, {'mAP50':metrics['AP50'],'mAP50_95':metrics['AP50_95']}, float(np.mean(elapsed))
    module.evaluate = streaming_evaluate
    args.output.mkdir(parents=True,exist_ok=True)
    config = {'epochs':20,'batch_size':2,'seed':42,'SGD_lr':.005,'momentum':.9,
              'weight_decay':.0005,'hflip':.5,'AMP':False,'eval_every':2,
              'selection_split':'validation','selection_metric':'official COCO mask AP50',
              'test_used_for_selection':False,'prior_test_exposure':True,
              'min_size':args.size,'max_size':round(args.size*1333/800)}
    (args.output/'fixed_protocol.json').write_text(json.dumps(config,indent=2))
    module.train(manifest=args.manifest,images_root=args.root,output_dir=args.output,
                 epochs=20,image_size=args.size,batch_size=2,eval_batch_size=1,
                 num_workers=4,evaluation_split='validation',eval_every=2,
                 seed=42,pretrained=True,device='cuda',use_amp=False,
                 confidence_threshold=.001)


if __name__ == '__main__':
    main()
