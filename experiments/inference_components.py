"""Frozen-checkpoint inference-component experiments, validation export and paired timing.

No training. Run only on an allocated GPU. RPN box coverage is a diagnostic,
not a mask-AP metric or exclusive causal attribution of a missed instance.
"""
import argparse, importlib.util, json, time, copy
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from unified_eval import rle, coco_metrics, diagnostic, verified_mask_mapping, digest

def load_models(root,gt):
    spec=importlib.util.spec_from_file_location('legacy',root/'train_mask_rcnn.py')
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    models={}; maps={}
    for name,path,size in [('mask800_historical',root/'mask_rcnn_final.pt',800),('mask640_retrained',root/'results/retrain640/checkpoints/best.pt',640),('mask800_retrained',root/'results/retrain800/checkpoints/best.pt',800)]:
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        model=legacy.build_model(len(gt.cats)+1,size,pretrained=False)
        model.load_state_dict(ckpt['model_state_dict']); model.to('cuda').eval()
        model.roi_heads.score_thresh=.001; model.roi_heads.detections_per_img=100
        models[name]=model; maps[name]=verified_mask_mapping(ckpt,gt.cats)
    from ultralytics import YOLO
    models['yolo640']=YOLO(str(root/'yolo_best.pt'))
    categories={v['name']:k for k,v in gt.cats.items()}
    maps['yolo640']={k:categories[v] for k,v in models['yolo640'].names.items()}
    return models,maps

def infer(model,name,mapping,im,image_id,retina=True,probe=None,size=640):
    if name.startswith('mask'):
        with torch.inference_mode():out=model([to_tensor(im).cuda()])[0]
        records=[{'image_id':image_id,'category_id':mapping[int(c)],'score':float(s),'segmentation':rle(m[0].cpu().numpy()>=.5)} for m,c,s in zip(out['masks'],out['labels'],out['scores'])]
        if probe is not None:probe['final']=out
        return records
    out=model.predict(im,imgsz=size,conf=.001,iou=.5,max_det=100,retina_masks=retina,device=0,half=False,verbose=False)[0]
    if out.masks is None:return []
    # retina=False masks live on the letterboxed inference grid, not original RGB coordinates.
    if retina:
        assert tuple(out.masks.data.shape[-2:])==(im.height,im.width)
        masks=out.masks.data
    else:
        from ultralytics.utils.ops import scale_masks
        masks=scale_masks(out.masks.data[None].float(),(im.height,im.width))[0]
    return [{'image_id':image_id,'category_id':mapping[int(c)],'score':float(s),'segmentation':rle(m.cpu().numpy()>=.5)} for m,c,s in zip(masks,out.boxes.cls,out.boxes.conf)]

def export(models,maps,gt,image_root,output,names,retina=True):
    for name in names:
        preds=[]
        for j,i in enumerate(sorted(gt.imgs)):
            im=Image.open(image_root/gt.imgs[i]['file_name']).convert('RGB')
            preds.extend(infer(models[name],name,maps[name],im,i,retina))
            if j%100==0:print('export',name,j,flush=True)
        dest=output/name; dest.mkdir(parents=True,exist_ok=True)
        (dest/'predictions.json').write_text(json.dumps(preds))
        (dest/'summary.json').write_text(json.dumps(coco_metrics(gt,preds,sorted(gt.imgs)),indent=2))

def components(models,maps,gt,image_root,output,limit=0):
    name='mask800_historical'; model=models[name]; probe={}
    def transform_hook(module,args,out):probe['sizes']=out[0].image_sizes
    def rpn_hook(module,args,out):probe['proposals']=out[0]
    handles=[model.transform.register_forward_hook(transform_hook),model.rpn.register_forward_hook(rpn_hook)]
    defaults={'post_nms':model.rpn._post_nms_top_n['testing'],'nms':model.roi_heads.nms_thresh,'score':.001}
    configs=[('baseline',{})]+[(f'rpn_{n}',{'post_nms':n}) for n in [100,300,2000]]+[(f'box_nms_{n}',{'nms':n}) for n in [.3,.7]]+[(f'box_score_{n}',{'score':n}) for n in [.05,.1,.5]]
    allresults={}; ids=sorted(gt.imgs)[:limit or None]
    for label,changes in configs:
        cfg={**defaults,**changes}; model.rpn._post_nms_top_n['testing']=cfg['post_nms']; model.roi_heads.nms_thresh=cfg['nms']; model.roi_heads.score_thresh=cfg['score']
        preds=[]; stages=[]; proposal_counts=[]
        for j,i in enumerate(ids):
            im=Image.open(image_root/gt.imgs[i]['file_name']).convert('RGB')
            result=infer(model,name,maps[name],im,i,probe=probe); preds.extend(result)
            anns=[a for a in gt.loadAnns(gt.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0)]
            boxes=torch.tensor([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns],device='cuda',dtype=torch.float32).reshape(-1,4)
            h,w=probe['sizes'][0]; proposals=probe['proposals'][0].clone(); proposal_counts.append(len(proposals)); proposals*=torch.tensor([im.width/w,im.height/h,im.width/w,im.height/h],device='cuda')
            proposal_iou=box_iou(boxes,proposals).max(dim=1).values if len(proposals) else torch.zeros(len(anns),device='cuda')
            final=probe['final']; finaliou=box_iou(boxes,final['boxes'])
            for ai,a in enumerate(anns):
                same=torch.tensor([maps[name][int(v)]==a['category_id'] for v in final['labels']],device='cuda',dtype=torch.bool)
                boxmatch=float(finaliou[ai,same].max()) if same.any() else 0.
                eligible=[x for x in result if x['category_id']==a['category_id']]
                maskmatch=float(mask_util.iou([gt.annToRLE(a)],[x['segmentation'] for x in eligible],[0]*len(eligible)).max()) if eligible else 0.
                stages.append({'image_id':i,'annotation_id':a['id'],'category_id':a['category_id'],'area':a['area'],'proposal_box_iou':float(proposal_iou[ai]),'final_same_class_box_iou':boxmatch,'final_same_class_mask_iou':maskmatch})
            if j%100==0:print('component',label,j,flush=True)
        dest=output/label; dest.mkdir(parents=True,exist_ok=True); (dest/'predictions.json').write_text(json.dumps(preds)); (dest/'stage_diagnostics.json').write_text(json.dumps(stages))
        counts={}
        for cid,cat in gt.cats.items():
            rows=[s for s in stages if s['category_id']==cid]; denom=len(rows)
            counts[cat['name']]={'instances':denom,'proposal_recall50':sum(s['proposal_box_iou']>=.5 for s in rows)/max(denom,1),'proposal_recall75':sum(s['proposal_box_iou']>=.75 for s in rows)/max(denom,1),'final_box_coverage50':sum(s['final_same_class_box_iou']>=.5 for s in rows)/max(denom,1),'mask_coverage50':sum(s['final_same_class_mask_iou']>=.5 for s in rows)/max(denom,1),'covered_proposal_but_missing_final_box':sum(s['proposal_box_iou']>=.5 and s['final_same_class_box_iou']<.5 for s in rows),'covered_final_box_but_missing_mask':sum(s['final_same_class_box_iou']>=.5 and s['final_same_class_mask_iou']<.5 for s in rows)}
        allresults[label]={'config':cfg,'images':len(ids),'proposal_count_mean':float(np.mean(proposal_counts)),'proposal_count_range':[min(proposal_counts),max(proposal_counts)],'stage_protocol':'per-GT coverage, not one-to-one recall; boxes use original COCO bbox; masks use original COCO RLE; score floor follows each config; absence at a stage is not exclusive causal attribution','mask_metrics':coco_metrics(gt,preds,ids),'stage_coverage':counts}
        (output/'components.json').write_text(json.dumps(allresults,indent=2))
    for handle in handles:handle.remove()
    model.rpn._post_nms_top_n['testing']=defaults['post_nms']; model.roi_heads.nms_thresh=defaults['nms']; model.roi_heads.score_thresh=defaults['score']
    export(models,maps,gt,image_root,output/'yolo_retina_false',['yolo640'],retina=False)

def timing(models,maps,gt,image_root,output,rounds=5):
    configs=['yolo640','mask800_historical','mask800_historical_inference640','mask640_retrained','mask800_retrained']
    ims=[(i,Image.open(image_root/gt.imgs[i]['file_name']).convert('RGB')) for i in sorted(gt.imgs)]
    history={n:[] for n in configs}; raw={n:[] for n in configs}
    historical=models['mask800_historical']; original=(historical.transform.min_size,historical.transform.max_size)
    def execute(n,im,i):
        name='mask800_historical' if n.endswith('inference640') else n
        if name=='mask800_historical':
            historical.transform.min_size=(640,) if n.endswith('inference640') else original[0]
            historical.transform.max_size=1066 if n.endswith('inference640') else original[1]
        return infer(models[name],name,maps[name],im,i)
    for n in configs:
        for _ in range(5):execute(n,ims[0][1],ims[0][0])
    for r in range(rounds):
        samples={n:[] for n in configs}
        # Rotate and reverse order by round/image to distribute position and thermal effects.
        for j,(i,im) in enumerate(ims):
            shift=(j+r)%len(configs); order=configs[shift:]+configs[:shift]
            if (j+r)%2:order=order[::-1]
            for name in order:
                torch.cuda.synchronize(); start=time.perf_counter(); execute(name,im,i); torch.cuda.synchronize()
                samples[name].append((time.perf_counter()-start)*1000)
            if j%100==0:print('paired timing',r,j,flush=True)
        for n in configs:history[n].append({'round':r+1,'mean_ms':float(np.mean(samples[n])),'median_ms':float(np.median(samples[n]))}); raw[n].append(samples[n])
        (output/'paired_timing.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'scope':'prepared RGB PIL to CPU RLE; disk decode excluded; FP32 batch 1; co-resident models; no diagnostic hooks','rounds':history},indent=2))
    (output/'paired_timing_raw.json').write_text(json.dumps(raw))
    historical.transform.min_size,historical.transform.max_size=original

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--mode',choices=['smoke','full'],default='full');args=p.parse_args()
    assert torch.cuda.is_available(),'Use an allocated GPU, never login-node inference'
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    data=Path((args.root/'data/dataset_root.txt').read_text().strip()); gt=COCO(str(data/'annotations/instances_test2017.json'))
    models,maps=load_models(args.root,gt)
    config={'weight_hashes':{str(path.relative_to(args.root)):digest(path) for path in [args.root/'yolo_best.pt',args.root/'mask_rcnn_final.pt',args.root/'results/retrain640/checkpoints/best.pt',args.root/'results/retrain800/checkpoints/best.pt']},'torch':torch.__version__,'images_test':len(gt.imgs)}
    (args.output/'protocol.json').write_text(json.dumps(config,indent=2))
    if args.mode=='smoke':
        first=sorted(gt.imgs)[0]; im=Image.open(data/'test2017'/gt.imgs[first]['file_name']).convert('RGB')
        for n in models:
            result=infer(models[n],n,maps[n],im,first);print(n,len(result),flush=True)
        infer(models['yolo640'],'yolo640',maps['yolo640'],im,first,retina=False)
        print('SMOKE PASSED',flush=True);return
    components(models,maps,gt,data/'test2017',args.output/'components',args.limit)
    validation=COCO(str(data/'annotations/instances_val2017.json'))
    export(models,maps,validation,data/'val2017',args.output/'validation',list(models))
    timing(models,maps,gt,data/'test2017',args.output)
    (args.output/'COMPLETE').write_text('complete\n')
if __name__=='__main__':main()
