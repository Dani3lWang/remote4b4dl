"""Metric fixtures for the controlled experiment, with known masks and counts."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('e2c_fixture',Path(__file__).resolve().parents[2]/'scripts/run_maskpls_e2c.py')
e2c=importlib.util.module_from_spec(spec);spec.loader.exec_module(e2c)


class E2CMetricTests(unittest.TestCase):
    def test_matching_preserves_cardinality_before_iou(self):
        self.assertEqual(len(e2c.pairs(np.array([[.9,.6],[.6,0]]),[0,0],[0,0])),2)

    def test_real_mask_fixture_distinguishes_geometry_class_and_final_fp(self):
        gt=np.array([[1,1,1,0,0,0],[0,0,0,1,1,1]],bool)
        raw=np.concatenate([gt,np.array([[0,0,0,0,0,1]],bool)])
        pc=np.array([0,16,0]);gc=np.array([0,1]);prob=np.zeros((3,18))
        prob[np.arange(3),pc+1]=1
        final=raw[[0,2]];valid=np.ones(6,bool)
        row,targets=e2c.score_arrays(gt,gc,raw,pc,final,np.array([0,0]),prob,valid,valid,final,
                                    np.full(6,1000),'scene',0)
        self.assertEqual((row['raw_geometry_tp'],row['raw_class_tp'],row['tp'],row['fp']),(2,1,1,1))
        self.assertTrue(targets[1]['all_qualified_no_object'])
        total=e2c.summary([row],targets)
        self.assertEqual((total['precision'],total['recall']),(.5,.5))
        self.assertEqual(total['strata']['bucket']['1-10']['raw_geometry_tp'],2)


if __name__=='__main__':unittest.main()
