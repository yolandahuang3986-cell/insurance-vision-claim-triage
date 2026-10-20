"""Check compact frozen evidence without datasets, weights or GPU inference."""
import argparse
import json
import math
from pathlib import Path


def verify(path):
    data=json.loads(Path(path).read_text())
    checks=[]
    def check(condition,label):
        if not condition:raise AssertionError(label)
        checks.append(label)
    for model,policies in data['calibration'].items():
        for rule,row in policies.items():
            m=row['test_micro'];tp,fp,fn=m['TP'],m['FP'],m['FN']
            check(tp+fn==785,f'{model}/{rule}: GT denominator')
            check(math.isclose(m['F1'],2*tp/(2*tp+fp+fn),abs_tol=1e-12),f'{model}/{rule}: F1 arithmetic')
            check(math.isclose(m['precision'],tp/(tp+fp),abs_tol=1e-12),f'{model}/{rule}: precision')
            check(math.isclose(m['recall'],tp/(tp+fn),abs_tol=1e-12),f'{model}/{rule}: recall')
        check(policies['validation_global']['test_micro']['F1']>policies['fixed_0.5']['test_micro']['F1'],f'{model}: global F1 improves')
    check(len(data['calibration'])==9,'nine configurations')
    for method in ['legacy_yolo640','legacy_mask_inference800']:
        classes=data['class_diagnostics'][method]
        check(sum(v['ground_truth'] for v in classes.values())==785,f'{method}: class GT total')
    baseline=data['components']['baseline']['stage_coverage']['crack']
    for key,wanted in [('proposal_recall50',67),('final_box_coverage50',60),('mask_coverage50',51)]:
        check(round(baseline[key]*baseline['instances'])==wanted,'crack '+key)
    selected=data['selected_test']['yolo800_retrained']
    check(round(selected['AP50_95'],4)==.5328,'selected YOLO 800 AP')
    check(round(selected['per_class_AP50_95']['crack'],4)==.1408,'selected YOLO 800 crack AP')
    check(data['validation_policies']['yolo800_retrained_selected_false']['global']==.4,'selected cutoff')
    for cohort,record in data['timing'].items():
        check(all(len(rounds)==5 for rounds in record['rounds'].values()),cohort+' timing five rounds')
    for key,row in data['reconstruction_selection']['selected'].items():
        check(row['retina_masks'] is False,key+' validation reconstruction')
    return {'status':'passed','checks':len(checks),'scope':'Frozen aggregate arithmetic and protocol checks, not retraining or independent AP recomputation'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,default=Path(__file__).with_name('metrics.json'))
    print(json.dumps(verify(parser.parse_args().metrics),indent=2))
