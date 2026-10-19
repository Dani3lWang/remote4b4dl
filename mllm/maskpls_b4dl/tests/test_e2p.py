"""E2-P architecture, causal isolation and migration contracts on CPU."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.decoder import MaskedTransformerDecoder
from maskpls_b4dl.e2p import ARMS, initialize_model, selection_gates
from maskpls_b4dl.e2q import tensor_digest
from maskpls_b4dl.model import MaskPLS
from maskpls_b4dl.tests.test_port import ToyBackbone
from maskpls_b4dl.tests.test_e2q import metric
from maskpls_b4dl.runner import load_warm_start


def model(mode="none"):
    return MaskPLS(MaskPLSConfig(num_queries=4,mask_residual_mode=mode,
                                checkpoint_decoder=False,knn_backend="torch"),backbone=ToyBackbone())


def points():
    return torch.tensor([[1.,0,0,1],[0,2,0,2],[-3,0,0,3],[0,-1,1,4],[99,0,0,5]])


def flatten(output):
    result = {k:output[k] for k in ("class_logits","mask_logits","queries")}
    result.update({f"aux_{i}_{k}":v[k] for i,v in enumerate(output["auxiliary"])
                   for k in ("class_logits","mask_logits")})
    return result


class ResidualMaskTests(unittest.TestCase):
    def test_default_has_no_new_parameters_and_modes_validate(self):
        self.assertEqual(MaskPLSConfig.from_dict({}).mask_residual_mode,"none")
        self.assertFalse(any("residual" in k for k in model().state_dict()))
        with self.assertRaises(ValueError):MaskPLSConfig.from_dict({"mask_residual_mode":"gt_center"})

    def test_mature_queries_inherited_exactly_and_new_heads_paired_without_rng_consumption(self):
        parent = model().state_dict();identities = {}
        for arm,mode in ARMS.items():
            m = model(mode);before = torch.get_rng_state().clone()
            identities[arm] = initialize_model(m,parent,arm)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for key,tensor in parent.items():self.assertTrue(torch.equal(m.state_dict()[key],tensor),key)
        self.assertEqual(identities["P1"]["added_state_sha256"],identities["P2"]["added_state_sha256"])
        self.assertEqual(identities["P1"]["added_parameter_count"],197376)
        self.assertEqual(identities["P2"]["added_parameter_count"],197376)
        self.assertEqual(identities["P0"]["added_parameter_count"],0)

    def test_zero_branch_reproduces_all_ten_heads_and_attention_masks_exactly(self):
        torch.manual_seed(80);base = model().eval();parent = base.state_dict()
        with torch.no_grad():expected = flatten(base(points()))
        for arm,mode in ARMS.items():
            m = model(mode).eval();initialize_model(m,parent,arm);attention = []
            handle = m.decoder.transformer_cross_attention_layers[0].register_forward_pre_hook(
                lambda module,args,kwargs:attention.append(kwargs["attn_mask"].clone()),with_kwargs=True)
            with torch.no_grad():actual = flatten(m(points()))
            handle.remove()
            self.assertEqual(set(expected),set(actual))
            for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]),key)
            if arm=="P0":expected_attention = attention
            else:self.assertTrue(torch.equal(attention[0],expected_attention[0]))

    def test_position_residual_is_feature_independent_at_fixed_query_and_coordinates(self):
        torch.manual_seed(81);m = model("position").decoder
        # Activate the head to avoid proving independence with a zero branch.
        torch.nn.init.normal_(m.mask_residual_embed.layers[-1].weight,std=.02)
        out = torch.randn(1,4,256);feat = torch.randn(1,5,256)
        xyz = points()[None,:,:3];basis = m.pe_layer(xyz);pad = torch.zeros(1,5,dtype=torch.bool)
        for f in (feat,feat*2+10):
            _,mask,_ = m.pred_heads(out,f,pad_mask=pad,residual_basis=basis)
            base = torch.einsum("bqc,bpc->bpq",m.mask_embed(m.decoder_norm(out)),f)
            residual = mask-base
            if f is feat:expected = residual
            else:torch.testing.assert_close(residual,expected,atol=1e-5,rtol=1e-4)
        shifted = m.pe_layer(xyz+torch.tensor([[[.3,.4,0.]]]))
        self.assertFalse(torch.allclose(basis,shifted))
        with self.assertRaises(ValueError):m.pred_heads(out,feat,pad_mask=pad)

    def test_position_branch_uses_coordinate_basis_in_forward(self):
        # Observe actual forward wiring, rather than only exercising pred_heads.
        for mode in ("feature","position"):
            m = model(mode);observed = []
            original = m.decoder.pred_heads
            def record(*args,**kwargs):
                observed.append(kwargs["residual_basis"].detach().clone())
                return original(*args,**kwargs)
            with patch.object(m.decoder,"pred_heads",side_effect=record):m(points())
            self.assertEqual(len(observed),10)
            if mode=="position":expected = m.decoder.pe_layer(points()[None,:4,:3])
            else:expected = m.decoder.mask_feat_proj(m.backbone(points()[:4])[0][-1])
            for basis in observed:torch.testing.assert_close(basis,expected)

    def test_gradients_open_new_head_then_reach_hidden_layers(self):
        torch.manual_seed(82);parent = model().state_dict()
        for arm,mode in (("P1","feature"),("P2","position")):
            m = model(mode).train();initialize_model(m,parent,arm)
            optimizer = torch.optim.AdamW(m.parameters(),lr=1e-4)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);output = m(points())
                loss = sum(v["mask_logits"][:,:4].square().mean() for v in output["auxiliary"])
                loss = loss+output["mask_logits"][:,:4].square().mean();loss.backward()
                final = m.decoder.mask_residual_embed.layers[-1]
                self.assertTrue(final.weight.grad.abs().sum()>0)
                self.assertTrue(torch.isfinite(final.weight.grad).all())
                if step:
                    self.assertTrue(m.decoder.mask_residual_embed.layers[0].weight.grad.abs().sum()>0)
                optimizer.step()
            self.assertFalse(torch.equal(m.decoder.query_feat.weight,parent["decoder.query_feat.weight"]))

    def test_point_permutation_roi_and_checkpointed_gradients(self):
        torch.manual_seed(83);m = model("position").train()
        torch.nn.init.normal_(m.decoder.mask_residual_embed.layers[-1].weight,std=.01)
        checked = copy.deepcopy(m);checked.decoder.checkpoint_decoder = True
        permutation = torch.tensor([2,4,0,3,1])
        a = m(points());b = checked(points()[permutation])
        torch.testing.assert_close(a["mask_logits"][:,permutation],b["mask_logits"],atol=1e-4,rtol=1e-4)
        self.assertTrue((a["mask_logits"][:,4]==-1e4).all())
        for output in (a,b):output["mask_logits"].square().mean().backward()
        for (name,left),right in zip(m.named_parameters(),checked.parameters()):
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=1e-4,rtol=1e-4,msg=name)

    def test_migration_and_generic_warm_start_reject_unapproved_architecture(self):
        parent = model().state_dict()
        bad = dict(parent);bad.pop("decoder.query_feat.weight")
        with self.assertRaises(ValueError):initialize_model(model("position"),bad,"P2")
        with self.assertRaises(ValueError):initialize_model(model("feature"),parent,"P2")
        from maskpls_b4dl.config import SCHEMA,UPSTREAM_COMMIT
        checkpoint = dict(metadata=dict(schema_version=SCHEMA,implementation_sha256="a"*64,
            upstream_commit=UPSTREAM_COMMIT,config=MaskPLSConfig().to_dict()))
        with patch("maskpls_b4dl.runner.torch.load",return_value=checkpoint),self.assertRaisesRegex(ValueError,"only explicit loss"):
            load_warm_start("fixture",MaskPLSConfig(mask_residual_mode="position"),"a"*64)


spec = importlib.util.spec_from_file_location("e2p_workflow_fixture",Path(__file__).resolve().parents[2]/"scripts/run_maskpls_e2p.py")
workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_small_medians_include_all_misses_and_both_small_buckets(self):
        # Existing score helpers establish the mandatory common denominator.
        gt = np.eye(4,dtype=bool);raw = np.array([[1,1,0,0],[0,0,1,1]],dtype=bool)
        valid = np.ones(4,dtype=bool);prob = np.zeros((2,18));prob[:,1] = 1
        row,targets = workflow.h.score_arrays(gt,np.zeros(4,dtype=int),raw,np.zeros(2,dtype=int),
            raw,np.zeros(2,dtype=int),prob,valid,valid,raw,np.full(4,1000),"scene",0)
        workflow.q.enrich_targets(targets,gt,raw,np.array([[5,0,0],[15,0,0],[30,0,0],[45,0,0]],dtype=np.float32))
        result = workflow.summary([row],targets);d = result["geometry_diagnostics"]["1-10"]
        self.assertEqual(d["gt"],4);self.assertEqual(d["median_best_query_purity"],.5)
        self.assertEqual(d["median_best_iou"],.5)

    def test_capacity_control_and_purity_coverage_gates_are_required(self):
        def metrics(tp=100,purity=.1,coverage=.8,iou=.05):
            value = metric(tp=tp,small_geometry=100)
            value["geometry_diagnostics"] = {"1-10":dict(median_best_query_purity=purity,
                median_best_query_coverage=coverage,median_best_iou=iou)}
            return value
        controls = [metrics(),metrics(),metrics(tp=120)]
        treatment = metrics(tp=130,purity=.12);treatment["strata"]["bucket"]["1-10"].update(tp=30,raw_geometry_tp=120)
        self.assertTrue(all(selection_gates(treatment,controls).values()))
        treatment["geometry_diagnostics"]["1-10"]["median_best_query_coverage"] = .7
        self.assertFalse(selection_gates(treatment,controls)["small_coverage"])
        treatment["tp"] = 125
        self.assertFalse(selection_gates(treatment,controls)["final_tp"])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"completion.json";path.write_text('{"original":true}')
            argv = ["e2p","prepare","--out",directory,"--parent-root",directory,"--reference",directory,"--source-commit","fixture"]
            with patch("sys.argv",argv),self.assertRaisesRegex(ValueError,"NEW output directory"):workflow.main()
            self.assertEqual(path.read_text(),'{"original":true}')
            self.assertEqual(len(list(Path(directory).iterdir())),1)


if __name__=="__main__":unittest.main()
