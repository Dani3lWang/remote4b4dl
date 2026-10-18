"""Trace must explain native selection including stuff competition and tiny masks."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from run_maskpls_trace import trace_extraction
from maskpls_b4dl.config import MaskPLSConfig


class TraceTests(unittest.TestCase):
    def test_stuff_wins_points_and_overlap_drops_thing(self):
        cfg = MaskPLSConfig()
        # q0 car loses one of two points to stuff, q2 retains a single point.
        logits = torch.full((4,18), -20.)
        logits[0,4] = 5
        logits[1,11] = 5
        logits[2,1] = 5
        logits[3,17] = 5
        masks = torch.tensor([[2.,2.,-5.,5.], [4.,-5.,-5.,5.],
                              [-5.,-5.,4.,5.], [8.,8.,8.,8.]])
        output = dict(class_logits=logits, mask_logits=masks, queries=torch.zeros(4,2),
                      valid_points=torch.tensor([True,True,True,False]))
        native, trace = trace_extraction(output, cfg)
        self.assertEqual(native['query_indices'].tolist(), [2])
        self.assertEqual(trace['rows'][0]['reason'], 'below_overlap_threshold')
        self.assertEqual(trace['rows'][1]['reason'], 'stuff_competitor_not_exported')
        self.assertEqual(trace['rows'][3]['reason'], 'no_object')
        self.assertEqual(trace['rows'][0]['lost_to_queries'], {'1':1})
        self.assertFalse(trace['pre'][:,-1].any())

    def test_no_eligible_and_nonfinite(self):
        cfg = MaskPLSConfig()
        output = dict(class_logits=torch.zeros(2,18), mask_logits=torch.zeros(2,3),
                      queries=torch.zeros(2,2), valid_points=torch.ones(3,dtype=torch.bool))
        native, trace = trace_extraction(output, cfg)
        self.assertEqual(len(native['masks']), 0)
        self.assertTrue((trace['winner_query'] == -1).all())
        output['mask_logits'][0,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            trace_extraction(output, cfg)


if __name__ == '__main__':
    unittest.main()
