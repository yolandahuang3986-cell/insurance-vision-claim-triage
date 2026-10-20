import unittest
import json
import numpy as np
from unified_eval import diagnostic, rle, coco_metrics, verified_mask_mapping
from pycocotools.coco import COCO


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.seg = rle(np.ones((3,3),dtype=np.uint8))
        self.gt = [{'category_id':1,'segmentation':self.seg}]

    def test_duplicate_and_confidence_sort(self):
        p = [{'score':s,'category_id':1,'segmentation':self.seg} for s in [.6,.9]]
        result = diagnostic(p,self.gt)
        self.assertEqual((result['matched'],result['unmatched'],result['missed']),(1,1,0))

    def test_wrong_class_does_not_match(self):
        result = diagnostic([{'score':.9,'category_id':2,'segmentation':self.seg}],self.gt)
        self.assertEqual((result['matched'],result['unmatched'],result['missed']),(0,1,1))

    def test_low_confidence_excluded(self):
        result = diagnostic([{'score':.49,'category_id':1,'segmentation':self.seg}],self.gt)
        self.assertEqual((result['predictions'],result['missed']),(0,1))

    def test_empty_gt_retains_false_positive(self):
        result = diagnostic([{'score':.9,'category_id':1,'segmentation':self.seg}],[])
        self.assertEqual(result['unmatched'],1)

    def test_empty_predictions_retain_false_negative(self):
        self.assertEqual(diagnostic([],self.gt)['missed'],1)

    def test_checkpoint_class_mapping_is_verified(self):
        self.assertEqual(verified_mask_mapping({'classes':['dent'], 'category_id_to_label':{3:1}}, {3:{'name':'dent'}}),{1:3})

    def test_checkpoint_mapping_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            verified_mask_mapping({'classes':['scratch'],'category_id_to_label':{1:1}}, {1:{'name':'dent'}})

    def test_coco_does_not_mutate_exported_predictions(self):
        c = COCO()
        c.dataset = {'info':{},'images':[{'id':1,'height':3,'width':3}],
                     'categories':[{'id':1,'name':'dent'}],
                     'annotations':[{'id':1,'image_id':1,'category_id':1,
                                     'segmentation':self.seg,'area':9,'bbox':[0,0,3,3],'iscrowd':0}]}
        c.createIndex()
        preds = [{'image_id':1,'category_id':1,'segmentation':self.seg,'score':.9}]
        metrics = coco_metrics(c,preds,[1])
        self.assertAlmostEqual(metrics['AP50'],1)
        self.assertNotIn('area',preds[0])
        json.dumps(preds)


if __name__ == '__main__':
    unittest.main()
