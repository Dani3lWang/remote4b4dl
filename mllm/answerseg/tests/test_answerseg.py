"""CPU integration tests with real VTimeLLM/PEFT; sparse backbone fixtures are explicit."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from answerseg import SCHEMA
from answerseg.data import (
    AnswerSegDataset,
    digest,
    load_features,
    prepare_records,
    validate_request,
)
from answerseg.language import (
    make_prompt,
    mark_answer,
    prompt_ids,
    register_tokens,
    token_rows,
    training_sequence,
)
from answerseg.model import AnswerSegModel, candidate_assignment
from answerseg.runner import (
    initialize_stage,
    input_points,
    score_prediction,
)
from answerseg.runtime import (
    artifact_identity,
    build_model,
    configure_lora,
    extension_state,
    load_bundle,
    optimizer_for,
    restore_extension,
    save_bundle,
)
from maskpls_b4dl.config import SCHEMA as SPATIAL_SCHEMA
from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.protocol import implementation_digest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM

torch.set_num_threads(1)


def tiny_language():
    vocab = {
        word: index
        for index, word in enumerate(
            (
                "<unk>",
                "<s>",
                "</s>",
                "<pad>",
                "truck",
                "car",
                "ahead",
                "<4DLiDAR>",
                "<meta>",
                "The",
                ".",
                "no",
                "target",
            )
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=["<4DLiDAR>", "<meta>"],
    )
    config = VTimeLLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        use_cache=False,
    )
    model = VTimeLLMLlamaForCausalLM(config)
    model.get_model().mm_projector = nn.Linear(768, 32)
    return tokenizer, model


class SpatialFixture(nn.Module):
    """Only the optional sparse backbone/checkpoint is replaced in CPU integration."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or MaskPLSConfig(
            num_queries=2, knn_backend="torch", checkpoint_decoder=False
        )
        self.queries = nn.Parameter(torch.randn(2, 256))
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, points):
        masks = points.new_tensor([[1, 1, -1, -1], [-1, -1, 1, 1]]) * self.scale
        classes = points.new_full((2, 18), -12.0)
        classes[:, 4] = 12.0
        return {
            "mask_logits": masks,
            "class_logits": classes,
            "queries": self.queries,
            "valid_points": torch.ones(4, dtype=torch.bool, device=points.device),
        }


def model_fixture(stage="binding", tune=False):
    torch.manual_seed(73)
    tokenizer, language = tiny_language()
    tokens = register_tokens(tokenizer, language)
    if stage == "joint":
        language = configure_lora(language, tokens, rank=2, alpha=4)
    model = AnswerSegModel(
        language, tokenizer, tokens, SpatialFixture(), stage=stage, tune_spatial=tune
    )
    return model


def example_fixture():
    return {
        "prompt": make_prompt("Which truck is ahead?"),
        "answer": "The truck ahead.",
        "targets": [
            {"span": [4, 15], "status": "matched", "instance_token": "truck-1"}
        ],
        "features": torch.randn(3, 768),
        "frame": {
            "points": torch.zeros(4, 4),
            "labels": torch.tensor([[1.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0]]),
            "instance_tokens": ["truck-1", "truck-2"],
            "point_sha256": "a" * 64,
        },
    }


class LanguageTests(unittest.TestCase):
    def test_registration_preserves_b3_and_adds_two_tokens_after_restore(self):
        tokenizer, model = tiny_language()
        before = model.get_input_embeddings().weight.detach().clone()
        tokens = register_tokens(tokenizer, model)
        torch.testing.assert_close(
            before, model.get_input_embeddings().weight[: len(before)], rtol=0, atol=0
        )
        self.assertIn("<meta>", tokenizer.additional_special_tokens)
        self.assertIn("<4DLiDAR>", tokenizer.additional_special_tokens)
        self.assertEqual(
            tokenizer.encode("<SEG>", add_special_tokens=False), [tokens.seg]
        )
        with self.assertRaisesRegex(ValueError, "unextended"):
            register_tokens(tokenizer, model)

    def test_true_visual_expansion_aligns_assistant_markers_and_masks_prompt(self):
        model = model_fixture()
        example = example_fixture()
        marked, _ = mark_answer(example["answer"], example["targets"])
        ids, labels = training_sequence(
            model.bridge.tokenizer, example["prompt"], marked
        )
        prefix = prompt_ids(model.bridge.tokenizer, example["prompt"])
        self.assertTrue(labels[:, : prefix.shape[1]].eq(-100).all())
        with patch.object(
            model.language, "forward", wraps=model.language.forward
        ) as forward:
            loss, states, kinds = model.bridge.forward(ids, labels, example["features"])
        call = forward.call_args.kwargs
        self.assertEqual(call["inputs_embeds"].shape[1], ids.shape[1] + 2)
        location = call["labels"].eq(model.tokens.seg).nonzero()[0, 1]
        raw_location = labels.eq(model.tokens.seg).nonzero()[0, 1]
        self.assertEqual(int(location - raw_location), 2)
        self.assertEqual(states.shape, (1, 32))
        self.assertEqual(kinds.tolist(), [model.tokens.seg])
        self.assertTrue(torch.isfinite(loss))

    def test_context_overflow_cannot_silently_truncate_targets(self):
        model = model_fixture()
        example = example_fixture()
        marked, _ = mark_answer(example["answer"], example["targets"])
        ids, labels = training_sequence(
            model.bridge.tokenizer, example["prompt"], marked
        )
        model.language.config.tokenizer_model_max_length = ids.shape[1]
        with self.assertRaisesRegex(ValueError, "truncation"):
            model.bridge.forward(ids, labels, example["features"])

    def test_actual_transformers_generation_retains_negative_visual_prefix(self):
        model = model_fixture()
        model.language.generation_config.forced_eos_token_id = (
            model.bridge.tokenizer.eos_token_id
        )
        model.eval()
        generated = model.bridge.generate(make_prompt("truck"), torch.randn(3, 768), 2)
        self.assertTrue(generated["finished"])
        self.assertEqual(int(generated["output_ids"].eq(-200).sum()), 1)

    def test_span_overlap_and_unknown_target_are_rejected(self):
        target = {"span": [0, 5], "status": "matched", "instance_token": "truck-1"}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            mark_answer("truck ahead", [target, dict(target, span=[4, 8])])
        with self.assertRaisesRegex(ValueError, "NOOBJ"):
            mark_answer("truck", [dict(target, status="pending")])


class TrainingTests(unittest.TestCase):
    def test_binding_freezes_geometry_language_and_batch_state(self):
        model = model_fixture()
        before_lm = {
            key: value.clone() for key, value in model.language.state_dict().items()
        }
        before_spatial = {
            key: value.clone() for key, value in model.spatial.state_dict().items()
        }
        model.train()
        self.assertFalse(model.language.training)
        self.assertFalse(model.spatial.training)
        result = model(example_fixture())
        self.assertEqual(result["covered_targets"], 1)
        result["loss"].backward()
        optimizer_for(model, 1e-3).step()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.head.parameters()
            )
        )
        for key, value in model.language.state_dict().items():
            torch.testing.assert_close(value, before_lm[key], rtol=0, atol=0)
        for key, value in model.spatial.state_dict().items():
            torch.testing.assert_close(value, before_spatial[key], rtol=0, atol=0)

    def test_binding_signal_alone_reaches_actual_b3_lora(self):
        model = model_fixture("joint")
        model.train()
        example = example_fixture()
        marked, _ = mark_answer(example["answer"], example["targets"])
        ids, labels = training_sequence(
            model.bridge.tokenizer, example["prompt"], marked
        )
        _, states, _ = model.bridge.forward(ids, labels, example["features"])
        _, _, queries = model.candidates(example["frame"]["points"])
        # Deliberately discard language CE: this proves the binding gradient is online.
        F.cross_entropy(model.head(states, queries), torch.tensor([0])).backward()
        gradients = [
            p.grad for key, p in model.language.named_parameters() if "lora_" in key
        ]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
        self.assertIsNone(model.language.get_model().mm_projector.weight.grad)

    def test_joint_updates_lora_small_token_rows_and_optional_spatial(self):
        model = model_fixture("joint", tune=True)
        old_input = model.language.get_input_embeddings().weight.detach().clone()
        old_output = model.language.get_output_embeddings().weight.detach().clone()
        new_output = token_rows(
            model.language.get_output_embeddings(), model.tokens.original_vocab
        ).clone()
        model.train()
        result = model(example_fixture())
        result["loss"].backward()
        self.assertIsNotNone(model.spatial.scale.grad)
        self.assertGreater(float(model.spatial.scale.grad.abs()), 0)
        optimizer_for(model, 1e-3).step()
        torch.testing.assert_close(
            model.language.get_input_embeddings().weight, old_input, rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.language.get_output_embeddings().weight, old_output, rtol=0, atol=0
        )
        self.assertFalse(
            torch.equal(
                token_rows(
                    model.language.get_output_embeddings(), model.tokens.original_vocab
                ),
                new_output,
            )
        )
        self.assertEqual(
            model.language.get_output_embeddings().new_rows.dtype, torch.float32
        )
        self.assertFalse(
            any(
                p.requires_grad
                for key, p in model.language.named_parameters()
                if "lora_" not in key and "new_rows" not in key
            )
        )

    def test_uncovered_positive_is_skipped_instead_of_false_noobj(self):
        model = model_fixture()
        example = example_fixture()
        example["frame"]["labels"][0] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        model.train()
        result = model(example)
        self.assertEqual(result["covered_targets"], 0)
        self.assertEqual(result["supervised_bindings"], 0)
        self.assertEqual(result["skipped_uncovered"], 1)
        self.assertEqual(float(result["loss"].detach()), 0)

    def test_explicit_no_target_trains_a_separate_rejection_target(self):
        model = model_fixture()
        example = example_fixture()
        example["targets"] = [
            {
                "span": [0, len(example["answer"])],
                "status": "no_target",
                "instance_token": None,
            }
        ]
        model.train()
        result = model(example)
        self.assertEqual(result["positive_targets"], 0)
        self.assertEqual(result["supervised_bindings"], 1)
        self.assertGreater(float(result["binding_loss"]), 0)

    def test_candidate_assignment_ignores_original_ignored_points(self):
        labels = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        masks = torch.tensor([[True, True, True, True]])
        self.assertEqual(
            candidate_assignment(masks, labels, torch.ones(4, dtype=torch.bool), 1),
            {0: 0},
        )


class InferenceTests(unittest.TestCase):
    def generated(self, model, text):
        prefix = prompt_ids(model.bridge.tokenizer, make_prompt("truck"))
        suffix = model.bridge.tokenizer.encode(text, add_special_tokens=False) + [
            model.bridge.tokenizer.eos_token_id
        ]
        return torch.cat((prefix, torch.tensor([suffix])), 1)

    def test_free_answer_seg_selects_mask_in_original_point_order_without_gt(self):
        model = model_fixture()
        model.eval()
        generated = self.generated(model, "The truck <SEG>")
        with (
            patch.object(model.language, "generate", return_value=generated),
            patch.object(
                model.head, "forward", return_value=torch.tensor([[20.0, -20.0, -20.0]])
            ),
        ):
            prediction = model.predict_native(
                make_prompt("truck"),
                torch.randn(3, 768),
                torch.zeros(4, 4),
                max_new_tokens=8,
            )
        self.assertEqual(prediction["objects"][0]["query_index"], 0)
        self.assertEqual(prediction["masks"][0].tolist(), [True, True, False, False])
        self.assertIsNone(prediction["objects"][0]["track_id"])

    def test_missing_token_explicit_noobj_and_truncation_are_distinct(self):
        model = model_fixture()
        model.eval()
        for text, expected in (
            ("truck", "missing_seg_token"),
            ("no target <NOOBJ>", "ok"),
        ):
            with patch.object(
                model.language, "generate", return_value=self.generated(model, text)
            ):
                prediction = model.predict_native(
                    make_prompt("truck"),
                    torch.randn(3, 768),
                    torch.zeros(4, 4),
                    max_new_tokens=8,
                )
            self.assertEqual(prediction["status"], expected)
            if expected == "ok":
                self.assertEqual(
                    prediction["objects"][0]["status"], "explicit_no_target"
                )
                self.assertFalse(prediction["masks"][0].any())
        ids = self.generated(model, "truck")[:, :-1]
        with patch.object(model.language, "generate", return_value=ids):
            result = model.predict_native(
                make_prompt("truck"),
                torch.randn(3, 768),
                torch.zeros(4, 4),
                max_new_tokens=8,
            )
        self.assertEqual(result["status"], "generation_truncated")

    def test_evaluation_keeps_parse_failures_in_denominator_and_noobj_must_be_explicit(
        self,
    ):
        totals = Counter()
        failed = {"status": "missing_seg_token", "masks": [], "objects": []}
        score_prediction(failed, example_fixture(), totals)
        self.assertEqual(totals["positive_targets"], 1)
        self.assertEqual(totals["false_negatives"], 1)
        example = example_fixture()
        example["targets"] = [
            {"status": "no_target", "instance_token": None, "span": [0, 1]}
        ]
        score_prediction(failed, example, totals)
        self.assertEqual(totals["correct_no_target_records"], 0)


def request_fixture():
    return {
        "request_id": "q1",
        "scene_token": "scene1",
        "sample_data_token": "frame1",
        "split": "train",
        "question": "Which truck?",
        "features": {
            "path": "features.npy",
            "sha256": "a" * 64,
            "input_policy": "explicit_frames",
            "frame_tokens": ["frame1"],
            "encoder_checkpoint_id": "sst-hash",
        },
    }


class DataTests(unittest.TestCase):
    def test_feature_mapping_hash_order_and_no_answer_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "features.npy"
            np.save(path, np.ones((2, 768), dtype=np.float32))
            request = request_fixture()
            request["features"].update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), indices=[1]
            )
            self.assertEqual(tuple(load_features(root, request).shape), (1, 768))
            request["features"]["input_policy"] = "answer_frames"
            with self.assertRaisesRegex(ValueError, "answer_frames"):
                validate_request(request)
            request["features"]["input_policy"] = "explicit_frames"
            request["features"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_features(root, request)

    def test_generated_answers_require_unchanged_fingerprint_and_explicit_review(self):
        source = dict(
            request_fixture(),
            schema_version=SCHEMA + "-generated",
            answer="truck ahead",
            answer_model_id="b3-hash",
            finished=True,
            review_status="pending",
        )
        source["record_fingerprint"] = digest(source)
        review = {
            "request_id": "q1",
            "record_fingerprint": source["record_fingerprint"],
            "decision": "accept",
            "reviewer": "reviewer",
            "reviewed_at": "2026-10-21",
            "evidence": "checked camera and point cloud",
            "targets": [
                {"span": [0, 5], "status": "matched", "instance_token": "truck-1"}
            ],
        }
        index = [
            {
                "scene_token": "scene1",
                "sample_data_token": "frame1",
                "features": source["features"],
            }
        ]
        records = prepare_records([source], index, [review])
        self.assertEqual(records[0]["answer_source"], "prediction")
        self.assertEqual(records[0]["answer"], source["answer"])
        with self.assertRaisesRegex(ValueError, "accepted reviews"):
            prepare_records([source], index)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_records([dict(source, answer="another truck")], index, [review])

    def test_pending_legacy_candidates_never_enter_binding_training(self):
        with self.assertRaisesRegex(ValueError, "pending"):
            prepare_records(
                [
                    {
                        "schema_version": "b4dl-answer-binding-v1",
                        "grounding_verified": False,
                    }
                ],
                [],
            )

    def test_dataset_rejects_split_relabel_and_unknown_target(self):
        source = dict(
            request_fixture(),
            schema_version=SCHEMA + "-reviewed",
            answer="truck ahead",
            targets=[
                {"span": [0, 5], "status": "matched", "instance_token": "unknown"}
            ],
            reviews=[
                {
                    "decision": "accept",
                    "reviewer": "a",
                    "reviewed_at": "today",
                    "evidence": "checked",
                }
            ],
            source_fingerprint="a" * 64,
            review_status="accepted",
            grounding_verified=True,
            answer_source="dataset_ground_truth",
        )
        spatial = SimpleNamespace(
            records=[
                {
                    "scene_token": "scene1",
                    "split": "internal_dev",
                    "frames": [{"sample_data_token": "frame1"}],
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split"):
                AnswerSegDataset(path, spatial, directory)

    def test_prediction_points_preserve_native_rows_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.asarray([[2, 0, 1, 15, 1], [1, 2, 3, 4, 0]], dtype="<f4")
            path = Path(directory) / "points.bin"
            path.write_bytes(values.tobytes())
            request = {
                "point_file": {
                    "path": "points.bin",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
            torch.testing.assert_close(
                input_points(directory, request), torch.from_numpy(values[:, :4].copy())
            )


class CheckpointTests(unittest.TestCase):
    def test_cpu_language_offload_overrides_gpu_request_without_cuda_allocation(self):
        from answerseg.runtime import load_b3

        tokenizer, language = tiny_language()
        runtime = {
            "model_base": "vicuna",
            "stage2": "b3",
            "pretrain_mm_mlp_adapter": "projector",
            "language_device": "cpu",
        }
        with patch(
            "vtimellm.model.builder.load_pretrained_model",
            return_value=(tokenizer, language, 512),
        ):
            _, restored = load_b3(runtime, "cuda:0")
        self.assertEqual(restored.get_input_embeddings().weight.device.type, "cpu")
        self.assertEqual(restored.get_input_embeddings().weight.dtype, torch.float32)

    def test_roundtrip_retains_joint_parameters_and_rejects_wrong_stage(self):
        model = model_fixture("joint", tune=True)
        model.train()
        optimizer = optimizer_for(model, 1e-3)
        model(example_fixture())["loss"].backward()
        optimizer.step()
        state = extension_state(model)
        restored = model_fixture("joint", tune=True)
        restore_extension(restored, state)
        for key, parameter in model.state_dict().items():
            torch.testing.assert_close(
                parameter, restored.state_dict()[key], rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "parameter names"):
            restore_extension(model_fixture("binding"), state)

    def test_bundle_load_checks_sources_and_tamper_before_constructing_model(self):
        model = model_fixture("joint")
        options = {
            "stage": "joint",
            "tune_spatial": False,
            "minimum_iou": 0.5,
            "rank": 2,
            "alpha": 4,
        }
        identity = {"b3": "b3-hash", "spatial": "spatial-hash"}
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            save_bundle(bundle, model, options, identity, {"global_step": 1})
            with (
                patch("answerseg.runtime.source_identity", return_value=identity),
                patch(
                    "answerseg.runtime.build_model",
                    return_value=(model_fixture("joint"), options),
                ),
            ):
                restored, _ = load_bundle(bundle, {}, "cpu")
                for key, value in extension_state(model)["parameters"].items():
                    torch.testing.assert_close(
                        value,
                        extension_state(restored)["parameters"][key],
                        rtol=0,
                        atol=0,
                    )
            with (
                patch("answerseg.runtime.source_identity", return_value={}),
                self.assertRaisesRegex(ValueError, "base identity"),
            ):
                load_bundle(bundle, {}, "cpu")
            (bundle / "extension.pt").write_bytes(b"tampered")
            with (
                patch("answerseg.runtime.source_identity", return_value=identity),
                self.assertRaisesRegex(ValueError, "integrity"),
            ):
                load_bundle(bundle, {}, "cpu")

    def test_binding_to_joint_carries_selector_without_loading_second_b3(self):
        binding, joint = model_fixture(), model_fixture("joint")
        with torch.no_grad():
            binding.head.no_referent.fill_(3)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            save_bundle(bundle, binding, {"stage": "binding"}, {}, {})
            initialize_stage(joint, bundle, {})
        torch.testing.assert_close(joint.head.no_referent, binding.head.no_referent)

    def test_production_loader_restores_real_local_b3_before_new_tokens(self):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(15)
        tokenizer, base = tiny_language()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.save_pretrained(root / "base")
            tokenizer.save_pretrained(root / "base")
            torch.save(
                {
                    "mm_projector.weight": base.get_model().mm_projector.weight,
                    "mm_projector.bias": base.get_model().mm_projector.bias,
                },
                root / "projector.bin",
            )
            adapted = get_peft_model(
                base,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            adapted.save_pretrained(root / "b3")
            spatial = SpatialFixture()
            torch.save(
                {
                    "metadata": {
                        "schema_version": SPATIAL_SCHEMA,
                        "implementation_sha256": implementation_digest(),
                        "config": spatial.config.to_dict(),
                    },
                    "model": spatial.state_dict(),
                },
                root / "spatial.pt",
            )
            runtime = {
                "model_base": str(root / "base"),
                "stage2": str(root / "b3"),
                "pretrain_mm_mlp_adapter": str(root / "projector.bin"),
                "spatial_checkpoint": str(root / "spatial.pt"),
            }
            with patch("maskpls_b4dl.model.MaskPLS", SpatialFixture):
                model, _ = build_model(runtime, "cpu", stage="joint", rank=2, alpha=4)
            self.assertEqual(model.tokens.original_vocab, len(tokenizer))
            self.assertEqual(
                model.bridge.tokenizer.encode("<meta>", add_special_tokens=False), [8]
            )
            model.train()
            result = model(example_fixture())
            result["loss"].backward()
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertGreater(len(artifact_identity(root / "b3")), 0)
            optimizer_for(model, 1e-3).step()
            from answerseg.runtime import source_identity

            options = {
                "stage": "joint",
                "tune_spatial": False,
                "minimum_iou": 0.5,
                "rank": 2,
                "alpha": 4,
            }
            save_bundle(
                root / "bundle",
                model,
                options,
                source_identity(runtime),
                {"global_step": 1},
            )
            with patch("maskpls_b4dl.model.MaskPLS", SpatialFixture):
                restored, _ = load_bundle(root / "bundle", runtime, "cpu")
            model.eval()
            restored.eval()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, restored.state_dict()[key], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
