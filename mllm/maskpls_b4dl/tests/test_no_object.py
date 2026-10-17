"""CPU contracts for E2-C's only changed objective and paired samples."""
import tempfile
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from maskpls_b4dl.config import MaskPLSConfig, SCHEMA, UPSTREAM_COMMIT
from maskpls_b4dl.criterion import MaskPLSCriterion, validate_sample_indices
from maskpls_b4dl.runner import load_warm_start, load_native


class NoObjectTests(unittest.TestCase):
    def test_default_validation_and_legacy_config(self):
        self.assertEqual(MaskPLSConfig.from_dict({}).no_object_weight, .1)
        for value in (0, -1, 1.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                MaskPLSConfig.from_dict(dict(no_object_weight=value))

    def test_only_classification_changes_and_matches_reference_gradient(self):
        torch.manual_seed(2026)
        output=dict(class_logits=torch.randn(3,18,requires_grad=True),
                    mask_logits=torch.randn(3,5,requires_grad=True))
        labels=torch.tensor([[1.,1.,0.,0.,0.]])
        classes=torch.tensor([4]);indices=torch.arange(5);assignment=[1]
        results=[]
        for weight in (.1,.05):
            criterion=MaskPLSCriterion(MaskPLSConfig(no_object_weight=weight))
            parts=criterion.layer_loss(output,labels,classes,indices,assignment)
            weights=torch.ones(18);weights[-1]=weight
            reference=F.cross_entropy(output['class_logits'],torch.tensor([17,4,17]),weight=weights)
            self.assertTrue(torch.equal(parts['classification'],reference))
            actual_grad=torch.autograd.grad(parts['classification'],output['class_logits'],retain_graph=True)[0]
            reference_grad=torch.autograd.grad(reference,output['class_logits'],retain_graph=True)[0]
            self.assertTrue(torch.equal(actual_grad,reference_grad))
            results.append(parts)
        for key in ('mask_bce','mask_dice'):
            self.assertTrue(torch.equal(results[0][key],results[1][key]))
        a=MaskPLSCriterion(MaskPLSConfig(no_object_weight=.1)).match(output,labels,classes,indices)
        b=MaskPLSCriterion(MaskPLSConfig(no_object_weight=.05)).match(output,labels,classes,indices)
        self.assertEqual(a,b)

    def test_explicit_samples_reject_missing_targets_duplicates_and_ignored_points(self):
        labels=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]])
        valid=torch.tensor([True,True,True,False])
        validate_sample_indices(torch.tensor([0,1,2]),labels,valid,3)
        for indices in ([0,2],[0,0,1],[0,1,3],[-1,0,1],[0,1,4]):
            with self.assertRaises(ValueError):
                validate_sample_indices(torch.tensor(indices),labels,valid,3)

    def test_audited_migration_allows_weight_but_exact_resume_stays_strict(self):
        config=MaskPLSConfig(mask_ignore_policy='background').to_dict()
        config.pop('no_object_weight')
        source=dict(metadata=dict(schema_version=SCHEMA,upstream_commit=UPSTREAM_COMMIT,
                    implementation_sha256='a'*64,config=config))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt';torch.save(source,path)
            result=load_warm_start(path,MaskPLSConfig(mask_ignore_policy='background',no_object_weight=.05),'a'*64)
            self.assertEqual(result,source)
            with self.assertRaises(ValueError):load_native(path)
            with self.assertRaises(ValueError):
                load_warm_start(path,MaskPLSConfig(num_queries=100,no_object_weight=.05),'a'*64)

    def test_explicit_samples_do_not_consume_model_rng(self):
        output=dict(class_logits=torch.zeros(3,18,requires_grad=True),
                    mask_logits=torch.zeros(3,4,requires_grad=True),
                    semantic_logits=torch.zeros(4,20,requires_grad=True),
                    valid_points=torch.tensor([True,True,True,False]),auxiliary=[])
        frame=dict(labels=torch.tensor([[1.,0.,-1.,-1.]]),class_ids=torch.tensor([0]),
                   semantic_labels=torch.tensor([1,11,0,0]))
        cfg=MaskPLSConfig(mask_ignore_policy='background');indices=torch.tensor([0,1,2])
        losses=[]
        for value in (7,2026):
            torch.manual_seed(value);before=torch.get_rng_state().clone()
            loss,_,_=MaskPLSCriterion(cfg)(output,frame,match_indices=indices,loss_indices=indices)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            losses.append(loss)
        self.assertTrue(torch.equal(losses[0],losses[1]))


if __name__=='__main__':unittest.main()
