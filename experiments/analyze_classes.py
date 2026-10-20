"""Validate class diagnostic totals from frozen predictions; no threshold tuning."""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from unified_eval import diagnostic

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,required=True)
root=parser.parse_args().run_root
coco=COCO(str(root/'evidence/instances_test2017.json'))
out={}
for method in ['legacy_yolo640','legacy_mask_inference800','legacy_mask_inference640']:
    predictions=json.loads((root/method/'predictions.json').read_text())
    byimage=defaultdict(list)
    for p in predictions:
        assert p['image_id'] in coco.imgs and p['category_id'] in coco.cats
        assert 0<=p['score']<=1
        byimage[p['image_id']].append(p)
    result={}
    for cid,c in coco.cats.items():
        totals={k:0 for k in ['matched','missed','unmatched','predictions']}
        overlaps=[]
        for iid in sorted(coco.imgs):
            gt=[dict(a,segmentation=coco.annToRLE(a)) for a in coco.imgToAnns[iid] if a['category_id']==cid and not a.get('iscrowd',0)]
            preds=[p for p in byimage[iid] if p['category_id']==cid]
            d=diagnostic(preds,gt)
            for k in totals: totals[k]+=d[k]
            overlaps.extend(d['ious'])
        totals['ground_truth']=totals['matched']+totals['missed']
        totals['precision']=totals['matched']/totals['predictions'] if totals['predictions'] else None
        totals['recall']=totals['matched']/totals['ground_truth']
        totals['matched_IoU']=sum(overlaps)/len(overlaps) if overlaps else None
        result[c['name']]=totals
    summary=json.loads((root/method/'summary.json').read_text())
    for k in ['matched','missed','unmatched','predictions']:
        assert sum(x[k] for x in result.values())==summary['fixed_threshold'][k]
    assert sum(x['ground_truth'] for x in result.values())==785
    out[method]=result
(root/'class_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
