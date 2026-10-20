"""Fixed-checkpoint COCO scale metrics and paired image bootstrap, no inference.

Bootstrap resamples images (including their complete annotations and predictions),
not instances or scalar per-image AP. COCO matching is cached per image; ranking
and 101-point precision accumulation are recomputed in every replicate.
"""
import argparse, contextlib, copy, io, json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(gt, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            dt=gt.loadRes(copy.deepcopy(predictions))
        else:
            dt=COCO(); dt.dataset={'images':list(gt.imgs.values()),'categories':list(gt.cats.values()),'annotations':[],'info':{}};dt.createIndex()
        ev=COCOeval(gt,dt,'segm'); ev.params.imgIds=sorted(gt.imgs)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev

def bootstrap_ap(ev, sample):
    """Exact COCO all-area maxDets=100 AP from cached per-image matching."""
    n=len(ev.params.imgIds); a=len(ev.params.areaRng); values=[]
    for k in range(len(ev.params.catIds)):
        entries=[ev.evalImgs[k*a*n+int(i)] for i in sample]
        entries=[e for e in entries if e is not None]
        if not entries:continue
        positives=sum(np.count_nonzero(np.asarray(e['gtIgnore'])==0) for e in entries)
        if not positives:continue
        scores=np.concatenate([e['dtScores'][:100] for e in entries])
        order=np.argsort(-scores,kind='mergesort')
        matches=np.concatenate([e['dtMatches'][:,:100] for e in entries],axis=1)[:,order]
        ignore=np.concatenate([e['dtIgnore'][:,:100] for e in entries],axis=1)[:,order]
        tp=np.cumsum((matches>0)&~ignore,axis=1,dtype=float)
        fp=np.cumsum((matches==0)&~ignore,axis=1,dtype=float)
        interpolated=np.zeros((10,101))
        for t in range(10):
            recall=tp[t]/positives
            precision=tp[t]/(tp[t]+fp[t]+np.spacing(1))
            precision=np.maximum.accumulate(precision[::-1])[::-1]
            idx=np.searchsorted(recall,ev.params.recThrs,side='left')
            valid=idx<len(precision); interpolated[t,valid]=precision[idx[valid]]
        values.append(interpolated.mean())
    return float(np.mean(values)) if values else float('nan')

def validate_duplicate_resampling(gt,preds,ev):
    sample=np.array([0,1,1,2,7,12,12,20,30,50,80,100,150,200,250,300,350,373])
    ids=ev.params.imgIds; images=[]; annotations=[]; predictions=[]
    byimage={i:[] for i in ids}
    for pred in preds:byimage[pred['image_id']].append(pred)
    for j,i in enumerate(sample):
        old=ids[int(i)]; im=copy.deepcopy(gt.imgs[old]); im['id']=j; images.append(im)
        for ann in gt.loadAnns(gt.getAnnIds(imgIds=[old])):
            ann=copy.deepcopy(ann); ann['image_id']=j; ann['id']=len(annotations)+1; annotations.append(ann)
        for pred in byimage[old]:
            pred=copy.deepcopy(pred); pred['image_id']=j; predictions.append(pred)
    duplicate=COCO(); duplicate.dataset={'images':images,'annotations':annotations,'categories':list(gt.cats.values()),'info':{}}
    with contextlib.redirect_stdout(io.StringIO()):duplicate.createIndex()
    expected=evaluate(duplicate,predictions).stats[0]
    actual=bootstrap_ap(ev,sample)
    assert abs(expected-actual)<1e-12,(expected,actual)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--replicates',type=int,default=2000)
    args=p.parse_args(); root=args.root; output=root/'posthoc'; output.mkdir(exist_ok=True)
    gt=COCO(str(root/'evidence/instances_test2017.json'))
    configs={'yolo640':'legacy_yolo640','mask800_historical':'legacy_mask_inference800','mask640_retrained':'retrain640','mask800_retrained':'retrain800'}
    evaluators={}; result={'protocol':{'replicates':args.replicates,'seed':6129,'unit':'image','paired':True,'interval':'percentile 95%','conditioning':'fixed checkpoints; exposed test set; not training variance'},'scale_metrics':{},'bootstrap':{}}
    counts={gt.cats[c]['name']:{'small':0,'medium':0,'large':0} for c in gt.cats}
    for ann in gt.anns.values():
        if ann.get('iscrowd',0):continue
        area=ann['area']; label='small' if area<32**2 else 'medium' if area<96**2 else 'large'
        counts[gt.cats[ann['category_id']]['name']][label]+=1
    result['gt_area_counts']=counts
    for label,folder in configs.items():
        preds=json.loads((root/folder/'predictions.json').read_text()); ev=evaluate(gt,preds); evaluators[label]=ev
        actual=bootstrap_ap(ev,np.arange(len(gt.imgs))); assert abs(actual-ev.stats[0])<1e-12
        validate_duplicate_resampling(gt,preds,ev)
        result['scale_metrics'][label]={'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP_small':float(ev.stats[3]),'AP_medium':float(ev.stats[4]),'AP_large':float(ev.stats[5]),'per_class_area_AP':{}}
        for k,cid in enumerate(ev.params.catIds):
            row={}
            for ai,name in enumerate(['all','small','medium','large']):
                v=ev.eval['precision'][:,:,k,ai,-1]; v=v[v>=0]
                row[name]=float(v.mean()) if len(v) else None
            result['scale_metrics'][label]['per_class_area_AP'][gt.cats[cid]['name']]=row
        print('Verified COCO cache and duplicate resampling:',label,flush=True)
    rng=np.random.default_rng(6129); samples={name:[] for name in configs}
    for rep in range(args.replicates):
        draw=rng.integers(0,len(gt.imgs),size=len(gt.imgs))
        for name,ev in evaluators.items():samples[name].append(bootstrap_ap(ev,draw))
        if rep%100==0:print('Bootstrap',rep,'/',args.replicates,flush=True)
    for a,b,name in [('mask800_historical','yolo640','historical_mask_minus_yolo'),('mask640_retrained','mask800_retrained','retrained_640_minus_800')]:
        diffs=np.array(samples[a])-np.array(samples[b]); interval=np.quantile(diffs,[.025,.975])
        result['bootstrap'][name]={'point_difference':float(evaluators[a].stats[0]-evaluators[b].stats[0]),'ci95':interval.tolist(),'contains_zero':bool(interval[0]<=0<=interval[1])}
    (output/'analysis.json').write_text(json.dumps(result,indent=2)); (output/'bootstrap_samples.json').write_text(json.dumps(samples))
    print(json.dumps(result['bootstrap'],indent=2))
if __name__=='__main__':main()
