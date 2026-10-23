from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from b4dl_core import b3
from b4dl_core.experiment import validate_experiment
from b4dl_core.io import digest, read_json, sha256, write_json
from b4dl_core.six_task import prediction_arrays
from b4dl_core.inputs import GeometryDataset
from test_core_geometry import raw_fixture
from test_core_b3_integration import tiny, TinyTokenizer, small_config
from b4dl_core.data import build_dataset
from dataclasses import asdict


class ExperimentTests(unittest.TestCase):
    def test_frozen_recipe_and_material_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/'baseline.json', {'fixture': True})
            write_json(root/'material.json', {'frozen': True})
            protocol = {'protocol_id': 'fixture', 'files_sha256': {str(root/'material.json'): sha256(root/'material.json')}}
            write_json(root/'protocol.json', protocol)
            recipe = {'lr': .0001, 'batch_size': 1, 'grad_accum': 4, 'steps': 3, 'lora_rank': 0,
                      'save_every': 1, 'keep_last': 2, 'train_examples': 12, 'epochs': 1}
            plan = {'schema': 'b4dl-core-experiment-v1', 'dataset_id': 'fixture',
                'baseline_manifest_sha256': sha256(root/'baseline.json'), 'recipe': recipe,
                'trainable_arms': ['A1', 'A2', 'A3'], 'seeds': [42, 43, 44],
                'six_task_protocol': {'path': str(root/'protocol.json'), 'sha256': sha256(root/'protocol.json'), 'protocol_id': 'fixture'}}
            plan['experiment_id'] = digest(plan)
            write_json(root/'experiment.json', plan)
            args = SimpleNamespace(**{k: v for k, v in recipe.items() if k not in ('train_examples', 'epochs')},
                experiment=str(root/'experiment.json'), baseline=str(root/'baseline.json'), arm='A3', seed=42)
            preparation = {'experiment_sha256': sha256(root/'experiment.json')}
            dataset = SimpleNamespace(manifest={'dataset_id': 'fixture'}, requests=[{}]*12)
            validate_experiment(args, preparation, {}, dataset)
            args.lr = .0002
            with self.assertRaises(ValueError): validate_experiment(args, preparation, {}, dataset)
            args.lr = .0001
            write_json(root/'material.json', {'changed': True})
            with self.assertRaises(ValueError): validate_experiment(args, preparation, {}, dataset)

    def test_prediction_errors_remain_in_six_task_denominator(self):
        rows = [{'task': 'existence', 'request_id': str(i), 'input_fingerprint': str(i),
                 'target': 'Yes', 'query': '<video>\nQ', 'scene_id': 's'} for i in range(2)]
        predictions = [{'request_id': str(i), 'input_fingerprint': str(i), 'status': 'error' if i else 'ok',
                        'text': '' if i else 'Yes'} for i in range(2)]
        result = prediction_arrays(rows, predictions)['existence']
        self.assertEqual(result['predictions'], ['Yes', ''])
        self.assertEqual(len(result['ground_truths']), 2)
        with self.assertRaises(ValueError): prediction_arrays(rows, predictions[:1])
        with self.assertRaises(ValueError): prediction_arrays(rows, list(reversed(predictions)))

    def test_first_and_latest_checkpoints_survive_retention(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            build_dataset(tables, metadata, features, splits, root/'dataset')
            dataset = GeometryDataset(root/'dataset', 'train')
            write_json(root/'baseline.json', {'fixture': True})
            output = root/'training'; output.mkdir()
            (output/'keep-me.pt').write_text('unrelated')
            args = SimpleNamespace(device='cpu', arm='A3', lora_rank=0, lr=.0001, seed=42,
                batch_size=1, grad_accum=1, steps=4, save_every=1, keep_last=2, resume=None,
                engineering=True, output=str(output), baseline=str(root/'baseline.json'))
            with patch.object(b3, 'load_original', lambda *unused: (TinyTokenizer(), tiny())):
                b3.train(args, {'frame_adapter': asdict(small_config())}, {}, dataset)
            self.assertEqual(sorted(p.name for p in output.glob('step-*.pt')), ['step-1.pt', 'step-3.pt', 'step-4.pt'])
            latest = read_json(output/'latest.json')
            self.assertEqual(latest['step'], 4)
            self.assertEqual(latest['checkpoint_sha256'], sha256(output/'step-4.pt'))
            self.assertEqual((output/'keep-me.pt').read_text(), 'unrelated')


if __name__ == '__main__': unittest.main()
