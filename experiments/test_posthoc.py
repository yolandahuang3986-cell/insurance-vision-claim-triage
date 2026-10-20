import unittest
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
from posthoc_analysis import evaluate,bootstrap_ap
from calibrate_thresholds import metric,records

class PosthocTests(unittest.TestCase):
    def fixture(self):
        mask=np.zeros((10,10),np.uint8);mask[2:5,2:5]=1;rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
        gt=COCO();gt.dataset={'info':{},'images':[{'id':1,'height':10,'width':10},{'id':2,'height':10,'width':10}],'categories':[{'id':1,'name':'crack'}],'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[2,2,3,3],'area':9,'iscrowd':0,'segmentation':rle}]};gt.createIndex()
        return gt,[{'image_id':1,'category_id':1,'score':.9,'segmentation':rle},{'image_id':2,'category_id':1,'score':.6,'segmentation':rle}]
    def test_empty_predictions(self):
        gt,preds=self.fixture();ev=evaluate(gt,[]);self.assertEqual(ev.stats[0],0)
    def test_bootstrap_identity(self):
        gt,preds=self.fixture();ev=evaluate(gt,preds);self.assertAlmostEqual(bootstrap_ap(ev,[0,1]),ev.stats[0],places=12)
    def test_empty_images_count_as_false_positive(self):
        gt,preds=self.fixture();r,t=records(gt,preds);m=metric(r,t,{1:.5});self.assertEqual((m['TP'],m['FP'],m['FN']),(1,1,0))
    def test_threshold_only_filters(self):
        gt,preds=self.fixture();r,t=records(gt,preds);m=metric(r,t,{1:.8});self.assertEqual((m['TP'],m['FP'],m['FN']),(1,0,0))
if __name__=='__main__':unittest.main()
