import json
import argparse
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from posthoc_analysis import evaluate,bootstrap_ap,validate_duplicate_resampling

parser=argparse.ArgumentParser(description='Reproduce the frozen True-reconstruction YOLO 800-minus-640 AP interval.')
parser.add_argument('--root',type=Path,required=True,help='Analysis root containing evidence/ and yolo_pair_20261009/.')
args=parser.parse_args()
root=args.root
(root/'final_revision').mkdir(parents=True,exist_ok=True)
gt=COCO(str(root/'evidence/instances_test2017.json'));evs={}
for size in [640,800]:
    preds=json.loads((root/f'yolo_pair_20261009/yolo{size}/test_unified/predictions.json').read_text())
    ev=evaluate(gt,preds);evs[size]=ev
    assert abs(bootstrap_ap(ev,np.arange(len(gt.imgs)))-ev.stats[0])<1e-12
    validate_duplicate_resampling(gt,preds,ev)
    print('validated',size,flush=True)
rng=np.random.default_rng(6129);diffs=[]
for n in range(2000):
    sample=rng.integers(0,len(gt.imgs),len(gt.imgs))
    diffs.append(bootstrap_ap(evs[800],sample)-bootstrap_ap(evs[640],sample))
    if n%500==0:print('replicate',n,flush=True)
out={'comparison':'retrained YOLO 800 minus 640','point':float(evs[800].stats[0]-evs[640].stats[0]),'ci95':np.quantile(diffs,[.025,.975]).tolist(),'replicates':2000,'seed':6129,'scope':'paired image resampling conditional on fixed checkpoints; no training variance'}
(root/'final_revision/yolo_bootstrap.json').write_text(json.dumps(out,indent=2));print(out)
