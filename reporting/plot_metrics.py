"""Replot numeric report evidence from the compact bundle; no photos or GPU.

These are source-driven reproductions, not a claim of pixel-identical formatting.
The paired photographic Figure 3 additionally requires authorized CarDD images
and per-instance predictions, deliberately excluded from the course archive.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(metrics,output):
    d=json.loads(Path(metrics).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2B5FA8','#5B6170','#A0A4AB']
    def save(fig,name):
        fig.savefig(output/name,dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    hist=d['training_histories']['historical_mask800']
    for key in hist[0]['loss_components']:
        axs[0].plot([r['epoch'] for r in hist],[r['loss_components'][key] for r in hist],label=key)
    rows=[r for r in hist if 'val_mask_mAP50' in r]
    axs[1].plot([r['epoch'] for r in rows],[r['val_mask_mAP50'] for r in rows],label='Historical test mask AP50')
    axs[0].set_title('Historical Mask R-CNN losses');axs[1].set_title('Historical test-selected trajectory')
    for ax in axs:ax.set_xlabel('Epoch');ax.legend(fontsize=6);ax.grid(alpha=.15)
    save(fig,'figure_1_historical_training.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    names=['historical_yolo640','historical_mask800','historical_mask640_inference_only']
    cats=list(d['quality'][names[0]]['coco_mask']['per_class_AP50_95'])
    for j,name in enumerate(names):
        vals=d['quality'][name]['coco_mask']['per_class_AP50_95']
        axs[0].bar(np.arange(6)+(j-1)*.25,[vals[c] for c in cats],width=.25,color=colors[j],label=name)
    axs[0].set_xticks(range(6),cats,rotation=25);axs[0].set_ylim(0,1);axs[0].legend(fontsize=6)
    ci=d['bootstrap_ap']['historical_mask_minus_yolo'];point=ci['point_difference'];lo,hi=ci['ci95']
    axs[1].errorbar(point,0,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4,color=colors[0])
    axs[1].axvline(0,color='gray',linestyle='--');axs[1].set_yticks([0],['Mask minus YOLO']);axs[1].set_xlabel('AP50:95 difference and 95% CI')
    save(fig,'figure_2_common_quality_ci.png')
    for model,name,losskey,apkey in [('mask','figure_4_mask_retraining.png','train_loss','val_mask_mAP50'),('yolo','figure_5_yolo_retraining.png','train/seg_loss','metrics/mAP50(M)')]:
        fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
        for j,size in enumerate([640,800]):
            rows=d['training_histories'][f'retrained_{model}{size}']
            axs[0].plot([r['epoch'] for r in rows],[r[losskey] for r in rows],color=colors[j],label=str(size))
            val=[r for r in rows if apkey in r]
            axs[1].plot([r['epoch'] for r in val],[r[apkey] for r in val],color=colors[j],label=str(size))
        axs[0].set_title('Training loss');axs[1].set_title('Validation mask AP50')
        for ax in axs:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.15)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ['yolo640','mask800_historical','mask640_retrained','mask800_retrained']:
        cfg=d['validation_policies'][name];curve=cfg['validation_global_curve']
        ax.plot([r['recall'] for r in curve],[r['precision'] for r in curve],label=name)
        selected=min(curve,key=lambda r:abs(r['threshold']-cfg['global']))
        ax.scatter(selected['recall'],selected['precision'],marker='s')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Validation micro recall',ylabel='Validation micro precision');ax.legend(fontsize=7)
    save(fig,'figure_6_validation_pr.png')
    return sorted(p.name for p in output.glob('*.png'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,default=Path(__file__).with_name('metrics.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(plot(args.metrics,args.output),indent=2))
