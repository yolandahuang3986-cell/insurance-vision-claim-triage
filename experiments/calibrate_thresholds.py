"""Validation-only threshold selection; one frozen test application per policy.

Select thresholds for micro F1 on original COCO masks at IoU .5. This is an
operating-point improvement, not a retrained model or a claim of increased
unrestricted ranked AP. Filtered AP is explicitly threshold-truncated.
"""
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from posthoc_analysis import evaluate

GRID=np.r_[.001,.01,np.arange(.05,1.,.05)]
def records(gt,preds):
    byimage={i:[] for i in gt.imgs}
    for p in preds:byimage[p['image_id']].append(p)
    rows=[]; totals={cid:0 for cid in gt.cats}
    for i in sorted(gt.imgs):
        gts=[a for a in gt.loadAnns(gt.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0)]
        used=set()
        for a in gts:totals[a['category_id']]+=1
        for p in sorted(byimage[i],key=lambda p:(-p['score'],p['category_id'])):
            eligible=[(j,a) for j,a in enumerate(gts) if j not in used and a['category_id']==p['category_id']]
            matched=False
            if eligible:
                ious=mask_util.iou([p['segmentation']],[gt.annToRLE(a) for j,a in eligible],[0]*len(eligible))[0]
                best=int(np.argmax(ious))
                if ious[best]>=.5:used.add(eligible[best][0]); matched=True
            rows.append((p['category_id'],p['score'],int(matched)))
    return np.asarray(rows).reshape(-1,3),totals

def metric(rows,totals,thresholds,category=None):
    selected=[r for r in rows if r[1]>=thresholds[int(r[0])] and (category is None or int(r[0])==category)]
    tp=sum(int(r[2]) for r in selected); num=len(selected); gt=sum(totals.values()) if category is None else totals[category]
    precision=tp/max(num,1); recall=tp/max(gt,1)
    return {'precision':precision,'recall':recall,'F1':2*tp/max(num+gt,1),'TP':tp,'FP':num-tp,'FN':gt-tp}

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--validation',type=Path,required=True); args=a.parse_args()
    root=args.root;output=root/'posthoc/calibration';output.mkdir(parents=True,exist_ok=True)
    vg=COCO(str(root/'evidence/instances_val2017.json'));tg=COCO(str(root/'evidence/instances_test2017.json'))
    configs={'yolo640':'legacy_yolo640','mask800_historical':'legacy_mask_inference800','mask640_retrained':'retrain640','mask800_retrained':'retrain800'}
    results={};policy={'grid':GRID.tolist(),'selection':'maximum validation F1; ties prefer higher threshold','mask_IoU':.5,'test_not_used_to_choose_threshold':True,'classes_calibrated':['crack','scratch','dent']}
    for model,folder in configs.items():
        valpreds=json.loads((args.validation/model/'predictions.json').read_text()); rows,totals=records(vg,valpreds)
        curves=[{'threshold':float(t),**metric(rows,totals,{c:t for c in vg.cats})} for t in GRID]
        globalbest=max(curves,key=lambda x:(x['F1'],x['threshold']))['threshold']
        thresholds={c:globalbest for c in vg.cats}; classcurves={}
        for cid,cat in vg.cats.items():
            cc=[{'threshold':float(t),**metric(rows,totals,{c:t for c in vg.cats},cid)} for t in GRID];classcurves[cat['name']]=cc
            if cat['name'] in ['crack','scratch','dent']:thresholds[cid]=max(cc,key=lambda x:(x['F1'],x['threshold']))['threshold']
        policy[model]={'global':globalbest,'class_specific':thresholds,'validation_global_curve':curves,'validation_class_curves':classcurves}
    # Save and freeze all policies before reading any test prediction file.
    (output/'frozen_validation_policy.json').write_text(json.dumps(policy,indent=2))
    for model,folder in configs.items():
        preds=json.loads((root/folder/'predictions.json').read_text()); rows,totals=records(tg,preds)
        globalt=policy[model]['global'];classmap=policy[model]['class_specific']
        results[model]={}
        for label,ts in [('fixed_0.5',{c:.5 for c in tg.cats}),('validation_global',{c:globalt for c in tg.cats}),('validation_class_specific',classmap)]:
            filtered=[p for p in preds if p['score']>=ts[p['category_id']]]
            ev=evaluate(tg,filtered)
            results[model][label]={'thresholds':ts,'test_micro':metric(rows,totals,ts),'test_threshold_truncated_AP':float(ev.stats[0]),'test_threshold_truncated_AP50':float(ev.stats[1]),'per_class':{tg.cats[c]['name']:metric(rows,totals,ts,c) for c in tg.cats}}
    (output/'test_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
