import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vtimellm.segmentation.config import ReasonSegConfig
from vtimellm.segmentation.training_state import EpochRandomSampler, resolve_training_config, resume_position
from vtimellm.segmentation.checkpoint import save_spatial_encoder_checkpoint, load_semantic_checkpoint
from vtimellm.segmentation.heads import HierarchicalMaskDecoder
from vtimellm.segmentation.data import ReasonSegDataset
from scripts.reasonseg_experiments.prepare_audited_manifests import prepare_records


class TrainingSafetyTests(unittest.TestCase):
    def test_negative_language_and_mask_failures_are_separate(self):
        from vtimellm.segmentation.negative_metrics import NegativeQueryMetrics
        metric = NegativeQueryMetrics()
        self.assertIsNone(metric.compute()['negative_noobj_accuracy'])
        metric.update(0, np.zeros((0, 4), dtype=bool), 'no_object')
        metric.update(0, np.zeros((0, 4), dtype=bool), 'missing_segmentation_tokens')
        metric.update(0, np.ones((1, 4), dtype=bool), 'ok')
        metric.update(1, np.ones((1, 4), dtype=bool), 'ok')
        self.assertEqual(metric.compute()['negative_queries'], 3)
        self.assertEqual(metric.compute()['negative_noobj_accuracy'], 1/3)
        self.assertEqual(metric.compute()['negative_mask_false_positive_rate'], 1/3)

    def test_resume_keeps_tversky_and_rejects_changed_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            config = ReasonSegConfig(region_loss='tversky', use_loc_prior=False)
            (Path(folder) / 'reasonseg_config.json').write_text(json.dumps({'config': config.to_dict()}))
            args = SimpleNamespace(resume_from_checkpoint=folder, eval_checkpoint=None,
                                   no_loc_prior=False, region_loss=None)
            self.assertEqual(resolve_training_config(args), config)
            args.region_loss = 'dice'
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                resolve_training_config(args)

    def test_cursor_distinguishes_step_save_from_completed_epoch(self):
        contract = {'batches_per_epoch': 7}
        state = dict(resume_format_version=2, resume_contract=contract, epoch=2,
                     next_batch_index=3, epoch_completed=False)
        self.assertEqual(resume_position(state, contract), (2, 3))
        state['next_batch_index'] = 7
        self.assertEqual(resume_position(state, contract), (2, 7))
        state['epoch_completed'] = True
        self.assertEqual(resume_position(state, contract), (3, 0))
        with self.assertRaisesRegex(ValueError, 'changed'):
            resume_position(state, {'batches_per_epoch': 8})
        with self.assertRaisesRegex(ValueError, 'legacy'):
            resume_position({'epoch': 2}, contract)

    def test_accelerate_interrupted_run_matches_data_weights_lr_and_rng(self):
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        with tempfile.TemporaryDirectory() as folder:
            def setup():
                torch.manual_seed(42)
                model = nn.Sequential(nn.Linear(1, 4), nn.Dropout(0.2), nn.Linear(4, 1))
                opt = torch.optim.AdamW(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
                data = torch.arange(14, dtype=torch.float32).view(-1, 1)
                sampler = EpochRandomSampler(data, 87)
                loader = DataLoader(data, batch_size=2, sampler=sampler,
                                    generator=torch.Generator().manual_seed(87))
                return accelerator.prepare(model, opt, loader, scheduler)
            def run(model, opt, loader, scheduler, stop=None, save=False):
                seen = []
                for i, batch in enumerate(loader):
                    seen.extend(batch.flatten().tolist())
                    with accelerator.accumulate(model):
                        accelerator.backward((model(batch) - batch).square().mean())
                        opt.step(); scheduler.step(); opt.zero_grad()
                    if stop is not None and i + 1 == stop:
                        if save:
                            accelerator.save_state(folder)
                        break
                return seen
            model, opt, loader, scheduler = setup()
            expected_seen = run(model, opt, loader, scheduler)
            expected = copy.deepcopy(model.state_dict())
            expected_lr = scheduler.get_last_lr()
            expected_rng = torch.get_rng_state().clone()
            accelerator.free_memory()
            model, opt, loader, scheduler = setup()
            first = run(model, opt, loader, scheduler, stop=4, save=True)
            accelerator.free_memory()
            model, opt, loader, scheduler = setup()
            accelerator.load_state(folder)
            rest = run(model, opt, accelerator.skip_first_batches(loader, 4), scheduler)
            self.assertEqual(expected_seen, first + rest)
            self.assertEqual(expected_lr, scheduler.get_last_lr())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            accelerator.free_memory()

    def test_semantic_checkpoint_round_trip_and_legacy_rejection(self):
        config = ReasonSegConfig(point_feature_dim=8, decoder_heads=2)
        def model():
            return SimpleNamespace(point_encoder=nn.Linear(4, 8), classifier=nn.Linear(8, 3))
        source, restored = model(), model()
        protocol = {'sample_tokens_sha256': 'abc', 'label_mapping': {'car': 1}}
        with tempfile.TemporaryDirectory() as folder:
            save_spatial_encoder_checkpoint(source.point_encoder, folder, config=config, epoch=1,
                                            validation_miou=0.2, num_classes=3,
                                            classifier=source.classifier, semantic_protocol=protocol)
            load_semantic_checkpoint(restored, folder, config=config, semantic_protocol=protocol)
            points = torch.randn(5, 4)
            torch.testing.assert_close(source.classifier(source.point_encoder(points)),
                                       restored.classifier(restored.point_encoder(points)))
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                load_semantic_checkpoint(restored, folder, config=config, semantic_protocol={})
            (Path(folder) / 'semantic_classifier.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'legacy'):
                load_semantic_checkpoint(restored, folder, config=config, semantic_protocol=protocol)

    def test_unreachable_target_has_no_seg_loc_or_class_gradient(self):
        config = ReasonSegConfig(hidden_size=8, point_feature_dim=8, decoder_heads=2,
                                 decoder_layers=1, decoder_ffn_dim=16)
        head = HierarchicalMaskDecoder(config)
        logits = torch.zeros(1, 1, 4, requires_grad=True)
        loc = torch.zeros_like(logits, requires_grad=True)
        classes = torch.zeros(1, 1, config.num_classes, requires_grad=True)
        losses = head.compute_losses(loc_logits=loc, mask_logits=logits, class_logits=classes,
                                     target_masks=torch.tensor([[[False, False, False, True]]]),
                                     target_loc_masks=torch.ones_like(logits, dtype=torch.bool),
                                     target_classes=torch.zeros(1, 1, dtype=torch.long),
                                     point_valid_mask=torch.tensor([[True, True, True, False]]),
                                     object_valid_mask=torch.ones(1, 1, dtype=torch.bool))
        sum(losses.values()).backward()
        for tensor in (logits, loc, classes):
            self.assertEqual(tensor.grad.abs().sum().item(), 0)

    def test_manifest_drops_whole_record_and_checks_tiny_outside_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            np.array([[0, 0, 0, 1, 0], [99, 0, 0, 1, 0]], dtype=np.float32).tofile(folder/'points.bin')
            np.savez(folder/'labels.npz', data=np.array([1001, 2001]))
            base = dict(sample_token='s', scene_token='scene', split='train', lidar_path='points.bin',
                        panoptic_path='labels.npz', query='q', answer='<LOC><SEG>',
                        targets=[dict(panoptic_id=1001, class_id=3, class_name='car')])
            bad = {**base, 'targets': base['targets'] + [dict(panoptic_id=2001, class_id=9, class_name='truck')]}
            categories = {1: 'vehicle.car', 2: 'vehicle.truck'}
            output, report = prepare_records([base, bad], folder, categories, [-1,-1,-1,1,1,1], 0.5, 12)
            self.assertEqual(report['dropped_positive_records'], 1)
            self.assertEqual(report['positive_records'], 1)
            negatives = [r for r in output if not r['targets']]
            self.assertEqual(len(negatives), 1)
            self.assertNotIn('truck', negatives[0]['query'])
            self.assertNotIn('car', negatives[0]['query'])
            self.assertEqual(negatives[0]['answer'], '<NOOBJ>')
            manifest=folder/'manifest.jsonl'
            manifest.write_text(json.dumps(bad)+'\n')
            config=ReasonSegConfig(point_cloud_range=[-1,-1,-1,1,1,1])
            with self.assertRaisesRegex(ValueError, 'unreachable'):
                ReasonSegDataset(str(manifest), dataroot=str(folder), config=config, require_reachable=True)[0]

    def test_auc_is_not_a_top_k_success_gate(self):
        # Ten negatives above all twelve positives: excellent pairwise AUC, poor top-K.
        auc, n, k, tp = 0.99, 1012, 12, 2
        self.assertGreater(auc, 1-k/n)
        self.assertLess(tp/(2*k-tp), 0.5)


if __name__ == '__main__':
    unittest.main()
