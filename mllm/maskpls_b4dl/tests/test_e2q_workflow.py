"""Independent fixtures for E2-Q distance metrics and output preservation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("e2q_workflow_fixture",Path(__file__).resolve().parents[2]/"scripts/run_maskpls_e2q.py")
e2q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2q)


class WorkflowTests(unittest.TestCase):
    def test_distance_and_purity_are_computed_from_full_target_denominators(self):
        gt = np.eye(4,dtype=bool)
        raw = np.array([[1,1,0,0],[0,0,1,1]],dtype=bool)
        valid = np.ones(4,dtype=bool);classes = np.zeros(4,dtype=int)
        prob = np.zeros((2,18));prob[:,1] = 1
        row,targets = e2q.h.score_arrays(gt,classes,raw,np.zeros(2,dtype=int),raw,np.zeros(2,dtype=int),
            prob,valid,valid,raw,np.full(4,1000),"scene",0)
        xyz = np.array([[5,0,0],[15,0,0],[30,0,0],[45,0,0]],dtype=np.float32)
        e2q.enrich_targets(targets,gt,raw,xyz)
        result = e2q.summary([row],targets)
        self.assertEqual(result['gt'],4)
        self.assertEqual(result['raw_geometry_tp'],2)
        self.assertEqual([t['distance_bin'] for t in targets],['0-10','10-20','20-40','40+'])
        self.assertTrue(all(t['best_query_coverage']==1 and t['best_query_purity']==.5 for t in targets))
        self.assertEqual(result['geometry_diagnostics']['all']['median_best_query_purity'],.5)
        self.assertEqual(sum(v['gt'] for v in result['strata']['distance'].values()),4)

    def test_prepare_rejection_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);marker = root/'completion.json'
            marker.write_text('{"original":true}',encoding='utf-8')
            argv = ['run_maskpls_e2q.py','prepare','--parent-root',directory,'--out',directory,
                    '--reference',directory,'--source-commit','fixture']
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'NEW output directory'):
                e2q.main()
            self.assertEqual(marker.read_text(),'{"original":true}')
            self.assertEqual([p.name for p in root.iterdir()],['completion.json'])


if __name__=='__main__':unittest.main()
