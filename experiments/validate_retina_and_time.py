"""Validation-only mask reconstruction selection and paired inference timing.

The test set was exposed earlier. This improves the selection protocol without
turning the existing test set into a new unseen holdout.
"""
import argparse,datetime,json,time,platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from inference_components import load_models,infer
from unified_eval import coco_metrics,digest

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def dump(path,obj):path.write_text(json.dumps(obj,indent=2))

def export_yolo(model,name,mapping,gt,images,path,size,retina):
    path.mkdir(parents=True,exist_ok=False);predictions=[]
    for j,i in enumerate(sorted(gt.imgs)):
        with Image.open(images/gt.imgs[i]['file_name']) as original:im=original.convert('RGB')
        predictions.extend(infer(model,name,mapping,im,i,retina=retina,size=size))
        if j%100==0:print('export',name,'retina',retina,j,'/',len(gt.imgs),flush=True)
    dump(path/'predictions.json',predictions)
    metrics=coco_metrics(gt,predictions,sorted(gt.imgs))
    dump(path/'summary.json',metrics);return metrics

def paired_timing(models,maps,gt,images,out,selected,rounds):
    configs={
        'yolo640_retrained_true':('yolo640_retrained',640,True),
        'yolo800_retrained_true':('yolo800_retrained',800,True),
        'mask640_retrained':('mask640_retrained',640,True),
        'mask800_retrained':('mask800_retrained',800,True),
    }
    for name in ['yolo640_retrained','yolo800_retrained']:
        if not selected[name]['retina_masks']:
            configs[name+'_selected_false']=(name,selected[name]['size'],False)
    ids=sorted(gt.imgs);prepared=[]
    for i in ids:
        with Image.open(images/gt.imgs[i]['file_name']) as original:prepared.append((i,original.convert('RGB')))
    names=list(configs);history={n:[] for n in names};raw={n:[] for n in names}
    def execute(n,im,i):
        name,size,retina=configs[n]
        return infer(models[name],name,maps[name],im,i,size=size,retina=retina)
    for n in names:
        for _ in range(5):execute(n,prepared[0][1],prepared[0][0])
    for round_id in range(rounds):
        samples={n:[] for n in names};mask_counts={n:[] for n in names}
        for j,(i,im) in enumerate(prepared):
            shift=(j+round_id)%len(names);order=names[shift:]+names[:shift]
            if (j+round_id)%2:order=order[::-1]
            for n in order:
                torch.cuda.synchronize();start=time.perf_counter()
                predictions=execute(n,im,i);torch.cuda.synchronize()
                samples[n].append((time.perf_counter()-start)*1000)
                mask_counts[n].append(len(predictions))
            if j%100==0:print('timing',round_id+1,j,'/',len(ids),flush=True)
        for n in names:
            history[n].append({'round':round_id+1,'images':len(ids),'mean_ms':float(np.mean(samples[n])),
                               'median_ms':float(np.median(samples[n])),'p95_ms':float(np.quantile(samples[n],.95)),
                               'exported_masks':sum(mask_counts[n])})
            raw[n].append(samples[n])
        dump(out/'paired_timing.json',{'gpu':torch.cuda.get_device_name(),
            'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'image_ids':ids,
            'scope':'prepared RGB PIL to CPU RLE; excludes disk decode; FP32 batch 1; co-resident models; CUDA synchronised',
            'score_floor':.001,'mask_threshold':.5,'warmup_calls_per_config':5,
            'order':'rotate and reverse by image and round','configs':configs,'rounds':history})
        dump(out/'paired_timing_raw.json',raw)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--rounds',type=int,default=5);a=p.parse_args()
    assert torch.cuda.is_available(),'Run inside an allocated GPU job'
    assert 'A40' in torch.cuda.get_device_name(),'Use A40 for the paired timing comparison'
    assert not a.output.exists(),'Use a fresh directory; no automatic overwrite or resume'
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    root=a.root.resolve();data=Path((root/'data/dataset_root.txt').read_text().strip())
    vg=COCO(str(data/'annotations/instances_val2017.json'))
    models,maps=load_models(root,vg)
    from ultralytics import YOLO
    import ultralytics
    categories={v['name']:k for k,v in vg.cats.items()}
    weights={'yolo640_historical':root/'yolo_best.pt'}
    models['yolo640_historical']=models['yolo640'];maps['yolo640_historical']=maps['yolo640']
    for size in [640,800]:
        name=f'yolo{size}_retrained';weight=root/f'results/yolo_pair_20261009/yolo{size}/training/weights/best.pt'
        weights[name]=weight;models[name]=YOLO(str(weight))
        maps[name]={k:categories[v] for k,v in models[name].names.items()}
    protocol={'created_utc':stamp(),'weights_sha256':{n:digest(w) for n,w in weights.items()},
        'torch':torch.__version__,'python':platform.python_version(),'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(),'validation_annotations_sha256':digest(data/'annotations/instances_val2017.json'),
        'selection':'maximum original-mask COCO validation AP50:95; exact ties prefer True',
        'candidates':[True,False],'score_floor':.001,'NMS_IoU':.5,'max_det':100,
        'mask_threshold':.5,'test_previously_exposed':True,
        'test_rule':'freeze all validation choices before one quality evaluation of each selected configuration'}
    dump(a.output/'protocol.json',protocol)
    names=list(weights);sizes={'yolo640_historical':640,'yolo640_retrained':640,'yolo800_retrained':800}
    if a.smoke:
        i=sorted(vg.imgs)[0];im=Image.open(data/'val2017'/vg.imgs[i]['file_name']).convert('RGB')
        for name in names:
            for retina in [True,False]:
                result=infer(models[name],name,maps[name],im,i,retina=retina,size=sizes[name])
                assert all(r['segmentation']['size']==[im.height,im.width] for r in result)
                print('smoke',name,retina,len(result),flush=True)
        print('SMOKE PASSED',flush=True);return
    # Candidate selection reads validation data only. No test predictions or test labels below until freeze.
    selected={};validation={}
    for name in names:
        validation[name]={}
        for retina in [True,False]:
            destination=a.output/'validation'/name/str(retina).lower()
            validation[name][str(retina).lower()]=export_yolo(models[name],name,maps[name],vg,data/'val2017',destination,sizes[name],retina)
        choice=max([True,False],key=lambda x:(validation[name][str(x).lower()]['AP50_95'],x))
        selected[name]={'retina_masks':choice,'size':sizes[name],
                        'validation_AP50_95':validation[name][str(choice).lower()]['AP50_95']}
    frozen={'frozen_utc':stamp(),'selection':protocol['selection'],'selected':selected,'validation':validation,
            'test_previously_exposed':True,'weights_sha256':protocol['weights_sha256']}
    dump(a.output/'frozen_validation_selection.json',frozen)
    print('ALL VALIDATION SELECTIONS FROZEN',json.dumps(selected),flush=True)
    tg=COCO(str(data/'annotations/instances_test2017.json'));test={}
    for name in names:
        cfg=selected[name]
        test[name]=export_yolo(models[name],name,maps[name],tg,data/'test2017',a.output/'selected_test'/name,cfg['size'],cfg['retina_masks'])
    dump(a.output/'selected_test_metrics.json',{'evaluated_utc':stamp(),'selected':selected,'metrics':test,
                                             'test_annotations_sha256':digest(data/'annotations/instances_test2017.json')})
    paired_timing(models,maps,tg,data/'test2017',a.output,selected,a.rounds)
    (a.output/'COMPLETE').write_text('complete\n')

if __name__=='__main__':main()
