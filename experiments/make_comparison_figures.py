import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pycocotools.coco import COCO
from pycocotools import mask as maskutil

parser=argparse.ArgumentParser(description='Render report figures from completed local evaluations.')
parser.add_argument('--run-root',type=Path,required=True,
    help='Directory with legacy_* result folders and evidence/{instances_test2017.json,smoke_images}.')
ROOT=parser.parse_args().run_root
(ROOT/'figures').mkdir(parents=True,exist_ok=True)
co=COCO(str(ROOT/'evidence/instances_test2017.json'))
methods=['legacy_yolo640','legacy_mask_inference800','legacy_mask_inference640']
summaries=[json.loads((ROOT/m/'summary.json').read_text()) for m in methods]
names=['YOLO 640','Mask R-CNN 800','Mask R-CNN 640*']
cats=list(co.cats.values())
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for i,s in enumerate(summaries):
    vals=[s['coco_mask']['per_class_AP50_95'][c['name']] for c in cats]
    axes[0].bar(np.arange(6)+(i-1)*.24,vals,width=.24,label=names[i])
axes[0].set_xticks(range(6),['Dent','Scratch','Crack','Glass','Lamp','Tire'],rotation=25)
axes[0].set_ylim(0,1); axes[0].set_ylabel('COCO mask AP50:95'); axes[0].legend(fontsize=8)
axes[0].set_title('(a) Class-level AP, same evaluator')
for i,s in enumerate(summaries):
    axes[1].scatter(s['latency_ms_mean'],s['coco_mask']['AP50_95'],s=90)
    axes[1].annotate(names[i],(s['latency_ms_mean'],s['coco_mask']['AP50_95']),xytext=(5,5),textcoords='offset points',fontsize=9)
axes[1].set_xlim(65,123);axes[1].set_ylim(.498,.515)
axes[1].set_xlabel('Mean RGB-to-mask-export latency (ms, A40)')
axes[1].set_ylabel('COCO mask AP50:95');axes[1].set_title('(b) Fixed-checkpoint quality / latency')
for ax in axes: ax.grid(axis='y',alpha=.2)
fig.savefig(ROOT/'figures/common_evaluation.png',dpi=200);plt.close(fig)

pred=[json.loads((ROOT/m/'predictions.json').read_text()) for m in methods[:2]]
cases=[{x['image_id']:x for x in json.loads((ROOT/m/'cases.json').read_text())} for m in methods[:2]]
colors=plt.get_cmap('tab10').colors
ids=sorted(co.imgs)[:4]
fig,axes=plt.subplots(4,3,figsize=(9,10.6),layout='constrained')
for row,iid in enumerate(ids):
    im=co.imgs[iid];image=np.asarray(Image.open(ROOT/'evidence/smoke_images'/im['file_name']).convert('RGB'))
    for col,ax in enumerate(axes[row]):
        ax.imshow(image)
        anns=co.imgToAnns[iid] if col==0 else [p for p in pred[col-1] if p['image_id']==iid and p['score']>=.5]
        for a in anns:
            mask=co.annToMask(a) if col==0 else maskutil.decode(a['segmentation'])
            overlay=np.zeros((*mask.shape,4)); overlay[mask.astype(bool)]=(*colors[(a['category_id']-1)%10],.42)
            ax.imshow(overlay)
        title=f"Image {iid}: ground truth" if col==0 else f"{'YOLO' if col==1 else 'Mask 800'}: matched {cases[col-1][iid]['matched']}, missed {cases[col-1][iid]['missed']}, extra {cases[col-1][iid]['unmatched']}"
        ax.set_title(title,fontsize=9);ax.axis('off')
fig.legend(handles=[Patch(color=colors[(c['id']-1)%10],label=c['name']) for c in cats],loc='outside lower center',ncol=3,fontsize=9)
fig.savefig(ROOT/'figures/paired_predictions.png',dpi=180);plt.close(fig)
