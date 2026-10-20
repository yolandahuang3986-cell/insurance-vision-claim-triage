"""Extend frozen validation operating thresholds to all six baseline checkpoints.

Also calibrate the validation-selected YOLO reconstruction variants. All policy
choices are written before reading any test predictions in this run.
"""
import argparse,json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from calibrate_thresholds import GRID,records,metric
from posthoc_analysis import evaluate
from bootstrap_calibrated_f1 import image_counts,micro_f1,interval_pair
from unified_eval import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--followup',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;gpu=a.followup
    a.output.mkdir(parents=True,exist_ok=False)
    vg=COCO(str(root/'evidence/instances_val2017.json'))
    selection=json.loads((gpu/'frozen_validation_selection.json').read_text())
    oldval=root/'components_20261009/validation'
    paths={
        'yolo640':(oldval/'yolo640/predictions.json',root/'legacy_yolo640/predictions.json'),
        'mask800_historical':(oldval/'mask800_historical/predictions.json',root/'legacy_mask_inference800/predictions.json'),
        'mask640_retrained':(oldval/'mask640_retrained/predictions.json',root/'retrain640/predictions.json'),
        'mask800_retrained':(oldval/'mask800_retrained/predictions.json',root/'retrain800/predictions.json'),
    }
    for size in [640,800]:
        name=f'yolo{size}_retrained'
        paths[name]=(gpu/f'validation/{name}/true/predictions.json',root/f'yolo_pair_20261009/yolo{size}/test_unified/predictions.json')
    selected_keys={}
    for name,cfg in selection['selected'].items():
        if cfg['retina_masks']:
            selected_keys[name]='yolo640' if name=='yolo640_historical' else name
        else:
            key=name+'_selected_false';selected_keys[name]=key
            paths[key]=(gpu/f'validation/{name}/false/predictions.json',gpu/f'selected_test/{name}/predictions.json')
    policy={'grid':GRID.tolist(),'selection':'max validation F1; ties higher cutoff',
            'mask_IoU':.5,'unit':'micro operating point','selected_reconstruction_keys':selected_keys,
            'reconstruction_policy_sha256':digest(gpu/'frozen_validation_selection.json'),'models':{}}
    for name,(vp,tp) in paths.items():
        rows,totals=records(vg,json.loads(vp.read_text()))
        curve=[{'threshold':float(t),**metric(rows,totals,{c:t for c in vg.cats})} for t in GRID]
        globalbest=max(curve,key=lambda z:(z['F1'],z['threshold']))['threshold']
        classmap={c:globalbest for c in vg.cats};classcurves={}
        for c,cat in vg.cats.items():
            curve_c=[{'threshold':float(t),**metric(rows,totals,{k:t for k in vg.cats},c)} for t in GRID]
            classcurves[cat['name']]=curve_c
            if cat['name'] in ['dent','scratch','crack']:
                classmap[c]=max(curve_c,key=lambda z:(z['F1'],z['threshold']))['threshold']
        policy['models'][name]={'global':globalbest,'class_specific':classmap,
                               'validation_global_curve':curve,'validation_class_curves':classcurves,
                               'validation_predictions_sha256':digest(vp)}
        print('policy selected',name,globalbest,flush=True)
    # Freeze every reconstruction-conditional threshold policy before test use.
    policyfile=a.output/'frozen_validation_policy.json'
    policyfile.write_text(json.dumps(policy,indent=2))
    tg=COCO(str(root/'evidence/instances_test2017.json'));results={};counts={}
    for name,(vp,tp) in paths.items():
        preds=json.loads(tp.read_text());rows,totals=records(tg,preds)
        cfg=policy['models'][name];results[name]={}
        for label,thresholds in [
            ('fixed_0.5',{c:.5 for c in tg.cats}),
            ('validation_global',{c:cfg['global'] for c in tg.cats}),
            ('validation_class_specific',cfg['class_specific'])
        ]:
            filtered=[pred for pred in preds if pred['score']>=thresholds[pred['category_id']]]
            ev=evaluate(tg,filtered)
            results[name][label]={'thresholds':thresholds,'test_micro':metric(rows,totals,thresholds),
                                 'test_threshold_truncated_AP':float(ev.stats[0])}
        counts[name]=image_counts(tg,preds,cfg['global'])
        assert abs(micro_f1(counts[name])-results[name]['validation_global']['test_micro']['F1'])<1e-12
        print('test evaluated',name,results[name]['validation_global']['test_micro'],flush=True)
    # Existing four policies must reproduce their frozen records exactly.
    previous=json.loads((root/'posthoc/calibration/test_results.json').read_text())
    for name in previous:
        for rule in previous[name]:
            assert results[name][rule]['test_micro']==previous[name][rule]['test_micro'],(name,rule)
    pairs={'retrained_mask800_minus_historical_yolo640':('mask800_retrained','yolo640'),
           'retrained_mask800_minus_retrained_yolo640':('mask800_retrained','yolo640_retrained'),
           'retrained_mask800_minus_retrained_yolo800':('mask800_retrained','yolo800_retrained')}
    for name,key in selected_keys.items():
        pairs['retrained_mask800_minus_'+key]=('mask800_retrained',key)
    bootstrap={label:{'models':[left,right],**interval_pair(counts[left],counts[right],2000,6129)}
               for label,(left,right) in pairs.items()}
    (a.output/'test_results.json').write_text(json.dumps(results,indent=2))
    (a.output/'f1_bootstrap.json').write_text(json.dumps({'protocol':'paired image percentile bootstrap; fixed weights and validation cutoffs; test previously exposed',
                                                       'policy_sha256':digest(policyfile),'comparisons':bootstrap},indent=2))
    print(json.dumps(bootstrap,indent=2))

if __name__=='__main__':main()
