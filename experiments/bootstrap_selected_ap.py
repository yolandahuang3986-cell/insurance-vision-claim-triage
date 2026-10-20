"""Image-bootstrap checks for the validation-selected reconstruction protocol."""
import argparse,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from posthoc_analysis import evaluate,bootstrap_ap,validate_duplicate_resampling

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--followup',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gt=COCO(str(a.root/'evidence/instances_test2017.json'))
    paths={
        'selected_yolo640':a.followup/'selected_test/yolo640_retrained/predictions.json',
        'selected_yolo800':a.followup/'selected_test/yolo800_retrained/predictions.json',
        'mask800_retrained':a.root/'retrain800/predictions.json',
        'baseline_yolo800_true':a.root/'yolo_pair_20261009/yolo800/test_unified/predictions.json',
    }
    evs={}
    for name,path in paths.items():
        preds=json.loads(path.read_text());ev=evaluate(gt,preds)
        assert abs(bootstrap_ap(ev,np.arange(len(gt.imgs)))-ev.stats[0])<1e-12
        validate_duplicate_resampling(gt,preds,ev);evs[name]=ev
        print('COCO cache verified',name,flush=True)
    pairs={
        'selected_yolo800_minus_selected_yolo640':('selected_yolo800','selected_yolo640'),
        'selected_yolo800_minus_retrained_mask800':('selected_yolo800','mask800_retrained'),
        'yolo800_selected_false_minus_baseline_true':('selected_yolo800','baseline_yolo800_true'),
    }
    rng=np.random.default_rng(6129);draws={name:[] for name in pairs}
    for n in range(2000):
        sample=rng.integers(0,len(gt.imgs),size=len(gt.imgs))
        values={name:bootstrap_ap(ev,sample) for name,ev in evs.items()}
        for name,(left,right) in pairs.items():draws[name].append(values[left]-values[right])
        if n%250==0:print('AP bootstrap',n,'/ 2000',flush=True)
    result={'protocol':'2,000 paired image draws, seed 6129, percentile 95%; fixed checkpoints and validation-selected reconstruction; no training variance; exposed test set',
            'comparisons':{}}
    for name,(left,right) in pairs.items():
        ci=np.quantile(draws[name],[.025,.975]);result['comparisons'][name]={
            'models':[left,right],'point_difference':float(evs[left].stats[0]-evs[right].stats[0]),
            'ci95':ci.tolist(),'contains_zero':bool(ci[0]<=0<=ci[1])}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
