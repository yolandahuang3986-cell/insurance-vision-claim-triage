"""Paired image bootstrap of micro F1 at frozen validation-selected thresholds."""
import argparse,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from unified_eval import diagnostic,digest

def image_counts(gt,preds,threshold):
    ids=sorted(gt.imgs);byimage={i:[] for i in ids}
    for pred in preds:byimage[pred['image_id']].append(pred)
    rows=[]
    for i in ids:
        gts=[dict(a,segmentation=gt.annToRLE(a)) for a in gt.loadAnns(gt.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0)]
        result=diagnostic(byimage[i],gts,floor=threshold,threshold=.5)
        rows.append([result['matched'],result['unmatched'],result['missed']])
    return np.asarray(rows,dtype=np.int64)

def micro_f1(counts):
    total=counts.sum(axis=0);tp,fp,fn=total
    return float(2*tp/max(2*tp+fp+fn,1))

def interval_pair(a,b,replicates,seed):
    assert a.shape==b.shape
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        sample=rng.integers(0,len(a),size=len(a))
        samples.append(micro_f1(a[sample])-micro_f1(b[sample]))
    ci=np.quantile(samples,[.025,.975])
    return {'point_difference':micro_f1(a)-micro_f1(b),'ci95':ci.tolist(),
            'contains_zero':bool(ci[0]<=0<=ci[1]),'replicates':replicates,'seed':seed}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replicates',type=int,default=2000)
    args=p.parse_args();root=args.root
    policy_path=root/'posthoc/calibration/frozen_validation_policy.json';policy=json.loads(policy_path.read_text())
    gt=COCO(str(root/'evidence/instances_test2017.json'))
    configs={'yolo640':'legacy_yolo640','mask800_retrained':'retrain800'};counts={};metrics={}
    reference=json.loads((root/'posthoc/calibration/test_results.json').read_text())
    for name,folder in configs.items():
        predpath=root/folder/'predictions.json';preds=json.loads(predpath.read_text())
        counts[name]=image_counts(gt,preds,policy[name]['global'])
        value=micro_f1(counts[name])
        assert abs(value-reference[name]['validation_global']['test_micro']['F1'])<1e-12
        metrics[name]={'threshold':policy[name]['global'],'F1':value,
                       'counts_TP_FP_FN':counts[name].sum(axis=0).tolist(),'predictions_sha256':digest(predpath)}
    result={'comparison':'retrained Mask R-CNN 800 minus historical YOLO 640, baseline retina_masks=True',
            'protocol':'paired image bootstrap; aggregate TP/FP/FN then recompute micro F1; fixed checkpoints and validation-selected thresholds; no threshold or training variance',
            'test_previously_exposed':True,'policy_sha256':digest(policy_path),'models':metrics,
            **interval_pair(counts['mask800_retrained'],counts['yolo640'],args.replicates,6129)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
