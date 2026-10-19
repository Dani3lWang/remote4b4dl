"""E2-Q coordinate/initialization/gradient contracts; no native CUDA dependency."""
import unittest
import copy
import numpy as np
import torch
from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.decoder import MaskedTransformerDecoder
from maskpls_b4dl.query_init import farthest_point_indices
from maskpls_b4dl.e2q import (ARMS,QUERY_KEYS,initialize_model,
                             preserve_model_modes,selection_gates,tensor_digest)
from maskpls_b4dl.model import MaskPLS
from maskpls_b4dl.tests.test_port import ToyBackbone


def decoder(mode):
    bb, dec, data = MaskPLSConfig(num_queries=4, query_initialization=mode).upstream()
    return MaskedTransformerDecoder(dec, bb, data)


class QueryInitializationTests(unittest.TestCase):
    def test_fps_original_order_duplicates_ties_and_shortfall(self):
        xyz = np.array([[1,0,0],[-1,0,0],[1,0,0],[0,0,0]], dtype=np.float32)
        self.assertEqual(farthest_point_indices(xyz,5).tolist(), [0,1,3,0,1])
        self.assertEqual(farthest_point_indices(xyz,2).tolist(), [0,1])
        with self.assertRaises(ValueError):
            farthest_point_indices(np.empty((0,3)), 4)

    def test_default_and_invalid_mode(self):
        self.assertEqual(MaskPLSConfig.from_dict({}).query_initialization,'learned')
        with self.assertRaises(ValueError):
            MaskPLSConfig.from_dict({'query_initialization':'gt_centers'})

    def test_position_and_content_are_separate_without_new_parameters(self):
        xyz=torch.tensor([[[1.,0,0],[-1,0,0],[0,0,0],[900.,0,0]]])
        padding=torch.tensor([[False,False,False,True]])
        torch.manual_seed(100)
        q0=decoder('learned')
        q1=decoder('zero_learned');q1.load_state_dict(q0.state_dict())
        q2=decoder('zero_fps');q2.load_state_dict(q0.state_dict())
        p0,c0=q0.initialize_queries(xyz,padding)
        p1,c1=q1.initialize_queries(xyz,padding)
        p2,c2=q2.initialize_queries(xyz,padding)
        self.assertTrue(torch.equal(p0,p1))
        self.assertTrue(c1.eq(0).all() and c2.eq(0).all())
        self.assertFalse(c0.eq(0).all())
        self.assertEqual(q2.last_query_point_indices.tolist(),[[0,1,2,0]])
        self.assertTrue(torch.equal(p2,q2.pe_layer(xyz[:,[0,1,2,0]])))
        (p0.sum()+c0.sum()).backward();p1.sum().backward()
        self.assertIsNotNone(q0.query_feat.weight.grad)
        self.assertIsNone(q1.query_feat.weight.grad)
        self.assertFalse(q2.query_feat.weight.requires_grad or q2.query_embed.weight.requires_grad)
        self.assertEqual(set(q0.state_dict()),set(q2.state_dict()))
        self.assertTrue(torch.equal(p2,q2.initialize_queries(xyz,padding)[0]))

    def test_fps_rejects_bad_inputs_and_does_not_consume_rng(self):
        for bad in (np.array([[float('nan'),0,0]]),np.zeros((2,2)),np.empty((0,3))):
            with self.assertRaises(ValueError):farthest_point_indices(bad,4)
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError):farthest_point_indices(np.zeros((2,3)),count)
        np.random.seed(42);before=copy.deepcopy(np.random.get_state())
        xyz=np.array([[0,0,0],[5,0,0],[0,3,0],[0,0,1]],dtype=np.float32)
        first=farthest_point_indices(xyz,4)
        self.assertEqual(first.tolist(),[1,2,3,0])
        self.assertTrue(np.array_equal(before[1],np.random.get_state()[1]))


def toy_model(mode):
    return MaskPLS(MaskPLSConfig(num_queries=4,query_initialization=mode,
                              checkpoint_decoder=False,knn_backend='torch'),backbone=ToyBackbone())


class MigrationAndGradientTests(unittest.TestCase):
    def test_paired_reset_preserves_shared_weights_and_private_rng(self):
        torch.manual_seed(23);parent=toy_model('learned').state_dict()
        identities={};models={}
        for arm,mode in ARMS.items():
            model=toy_model(mode);before=torch.get_rng_state().clone()
            identities[arm]=initialize_model(model,parent,arm)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for name,tensor in model.state_dict().items():
                if name not in QUERY_KEYS:self.assertTrue(torch.equal(tensor,parent[name]),name)
            models[arm]=model
        position=QUERY_KEYS[1]
        self.assertEqual(identities['Q0']['query_sha256'][position],identities['Q1']['query_sha256'][position])
        self.assertFalse(torch.equal(models['Q0'].state_dict()[position],parent[position]))
        self.assertTrue(models['Q1'].state_dict()[QUERY_KEYS[0]].eq(0).all())
        self.assertTrue(models['Q2'].state_dict()[position].eq(0).all())
        self.assertEqual(len({i['shared_state_sha256'] for i in identities.values()}),1)

    def test_migration_rejects_key_shape_or_arm_mismatch(self):
        model=toy_model('learned');parent=model.state_dict()
        with self.assertRaises(ValueError):initialize_model(model,parent,'Q2')
        missing=dict(parent);missing.pop(QUERY_KEYS[0])
        with self.assertRaises(ValueError):initialize_model(model,missing,'Q0')
        bad=dict(parent);bad[QUERY_KEYS[0]]=bad[QUERY_KEYS[0]][:1]
        with self.assertRaises(ValueError):initialize_model(model,bad,'Q0')

    def test_eval_restores_flags_and_module_modes_even_on_failure(self):
        model=toy_model('zero_fps');initialize_model(model,model.state_dict(),'Q2')
        model.train();model.backbone.eval()
        flags={n:p.requires_grad for n,p in model.named_parameters()}
        modes={n:m.training for n,m in model.named_modules()}
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with preserve_model_modes(model):
                model.eval();model.requires_grad_(True)
                raise RuntimeError('fixture')
        self.assertEqual(flags,{n:p.requires_grad for n,p in model.named_parameters()})
        self.assertEqual(modes,{n:m.training for n,m in model.named_modules()})

    def test_real_decoder_forward_backward_updates_shared_not_fixed_queries(self):
        torch.manual_seed(27);parent=toy_model('learned').state_dict()
        points=torch.tensor([[1.,0,0,1],[0,2,0,2],[-3,0,0,3],[0,-1,1,4],[99,0,0,5]])
        for arm,mode in ARMS.items():
            with self.subTest(arm=arm):
                model=toy_model(mode).train();identity=initialize_model(model,parent,arm)
                optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
                output=model(points)
                loss=output['class_logits'].square().mean()+output['mask_logits'][:,output['valid_points']].square().mean()
                self.assertTrue(torch.isfinite(loss));loss.backward()
                shared=[p.grad for n,p in model.named_parameters() if n not in QUERY_KEYS and p.grad is not None]
                self.assertTrue(shared and all(torch.isfinite(g).all() for g in shared))
                self.assertTrue(any(g.abs().sum()>0 for g in shared))
                for key in QUERY_KEYS:
                    p=dict(model.named_parameters())[key]
                    self.assertEqual(p.grad is not None,identity['trainable'][key])
                if arm=='Q2':
                    self.assertTrue((model.decoder.last_query_point_indices<4).all())
                optimizer.step()
                for key in QUERY_KEYS:
                    if not identity['trainable'][key]:
                        self.assertEqual(tensor_digest({key:model.state_dict()[key]}),identity['query_sha256'][key])


def metric(tp=100,precision=.5,small_geometry=100):
    return dict(tp=tp,precision=precision,ignored_fraction=.01,ego_points=0,
                strata=dict(bucket={'1-10':dict(tp=20,raw_geometry_tp=small_geometry),
                                    '11-49':dict(recall=.2),'50+':dict(recall=.3)},
                            scene_token={str(i):dict(recall=.2) for i in range(8)}))


class SelectionTests(unittest.TestCase):
    def test_fps_must_beat_both_contemporaneous_controls(self):
        f=metric();q0=metric(105);q1=metric(115,small_geometry=120)
        q2=metric(120,small_geometry=140)
        gates=selection_gates(q2,[f,q0,q1])
        self.assertFalse(gates['final_tp'])
        q2['tp']=125
        self.assertTrue(all(selection_gates(q2,[f,q0,q1]).values()))
        q2['strata']['bucket']['1-10']['raw_geometry_tp']=139
        self.assertFalse(selection_gates(q2,[f,q0,q1])['small_geometry'])

    def test_precision_and_scene_constraints_cannot_be_hidden_by_tp(self):
        f=metric();q0=metric();q1=metric();q2=metric(150,.47,150)
        self.assertFalse(selection_gates(q2,[f,q0,q1])['precision'])
        q2['precision']=.5
        for key in ('0','1','2','3'):q2['strata']['scene_token'][key]['recall']=.1
        self.assertFalse(selection_gates(q2,[f,q0,q1])['scenes'])
        with self.assertRaises(ValueError):selection_gates(q2,[f])


if __name__=='__main__':unittest.main()
