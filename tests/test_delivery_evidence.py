from pathlib import Path
import json
from reporting.verify_metrics import verify
from experiments.prepare_workspace import prepare, WEIGHT_LAYOUT


def test_frozen_metric_arithmetic():
    result=verify(Path(__file__).resolve().parents[1]/'reporting/metrics.json')
    assert result['status']=='passed'
    assert result['checks']>100


def test_runtime_layout_links_only_local_inputs(tmp_path):
    data=tmp_path/'authorized_data'
    for part in ['train2017','val2017','test2017','annotations']:(data/part).mkdir(parents=True)
    for split in ['train','val','test']:(data/f'annotations/instances_{split}2017.json').write_text('{}')
    weights={}
    for key in WEIGHT_LAYOUT:
        p=tmp_path/(key+'.pt');p.write_bytes(b'never deserialized');weights[key]=str(p)
    cfg=tmp_path/'inputs.json';cfg.write_text(json.dumps({'dataset_root':str(data),'weights':weights}))
    out=prepare(cfg,tmp_path/'runtime')
    assert (out/'train_mask_rcnn.py').is_symlink()
    assert (out/'yolo_best.pt').is_symlink()
    assert (out/'data/dataset_root.txt').read_text().strip()==str(data)
