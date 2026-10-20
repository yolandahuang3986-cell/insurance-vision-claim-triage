import unittest
import numpy as np
from bootstrap_calibrated_f1 import micro_f1, interval_pair


class CalibratedBootstrapTests(unittest.TestCase):
    def test_recompute_from_pooled_counts(self):
        rows=np.array([[1,0,0],[0,5,2]])
        self.assertAlmostEqual(micro_f1(rows),2/9)

    def test_paired_identical_counts_have_zero_gap(self):
        rows=np.array([[1,0,0],[0,1,1],[2,1,0]])
        result=interval_pair(rows,rows,100,6129)
        self.assertEqual(result['ci95'],[0.,0.])
        self.assertEqual(result['point_difference'],0.)

    def test_seed_is_reproducible(self):
        a=np.array([[1,0,1],[1,1,0],[2,0,0]])
        b=np.array([[0,1,2],[1,0,0],[1,0,1]])
        self.assertEqual(interval_pair(a,b,100,6129),interval_pair(a,b,100,6129))


if __name__=='__main__':unittest.main()
