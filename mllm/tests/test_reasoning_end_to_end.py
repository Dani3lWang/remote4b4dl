"""Exercise dataset -> SFT -> GRPO -> load -> inference receipts on CPU."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from b4dl_core.data import build_dataset
from b4dl_core.io import read_json, read_jsonl, write_json
from b4dl_core.reasoning.data import ReasoningDataset, build_data
from b4dl_core.reasoning.evaluation import (checkpoint_lineage, compare_six_task_metrics, evaluate, infer,
                                            infer_six_task, load_free_six_task, load_model, prepare_free_six_task)
from b4dl_core.reasoning.policy import expanded_log_probs, reference_policy, sequence_batch, step_token_ends
from b4dl_core.reasoning.protocol import encode_trace
from b4dl_core.reasoning.rewards import score_trace
from b4dl_core.reasoning.trainer import train
from test_reasoning_data import raw_fixture
from test_reasoning_training import CharacterTokenizer, model


class ReasoningEndToEndTests(unittest.TestCase):
    def test_engineering_sft_grpo_and_reloaded_inference(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables, features, support = raw_fixture(root)
            build_dataset(tables, root/"metadata.json", features, root/"splits.json", root/"geometry", input_support=support)
            build_data(root/"geometry", tables, support, root/"reasoning")
            trainset = ReasoningDataset(root/"reasoning", root/"geometry", "train")
            testset = ReasoningDataset(root/"reasoning", root/"geometry", "test")
            write_json(root/"baseline.json", {"fixture": True})
            write_json(root/"preparation.json", {"fixture": True})
            recipe = {"seed": 42, "lora_rank": 4, "sft_lr": .001, "grpo_lr": .001,
                      "group_size": 2, "grad_accum": 1, "max_new_tokens": 512, "clip": .2,
                      "beta": .04, "process_weight": .5, "mode": "process", "save_every": 1,
                      "engineering": True, "max_updates": 2}
            def prepare(*args): return CharacterTokenizer(), model(), {"arm": "A3", "fixture": True}
            def verified_group(net, tokenizer, model_input, request, target, group_size, max_new_tokens):
                wrong = copy.deepcopy(target["trace"])
                field = next(iter(wrong["steps"][0]["value"]))
                wrong["steps"][0]["value"][field] += 2
                wrong["answer"] = "No" if target["answer"] == "Yes" else "Yes"
                group = []
                net.eval()
                for trace in (target["trace"], wrong):
                    text = encode_trace(trace)
                    ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
                    batch = sequence_batch(net, tokenizer, [model_input], [ids])
                    with torch.no_grad():
                        old = expanded_log_probs(net, batch)[0].cpu()
                        with reference_policy(net): ref = expanded_log_probs(net, batch)[0].cpu()
                    score = score_trace(request, target, text)
                    group.append({"ids": ids, "text": text, "status": "ok", "old": old, "ref": ref,
                                  "score": score, "ends": step_token_ends(tokenizer, ids, score["step_char_ends"]),
                                  "logprob_max_difference": 0.})
                return group
            with patch("b4dl_core.reasoning.trainer.prepare_model", prepare), \
                 patch("b4dl_core.reasoning.trainer.rollout_group", verified_group):
                sft = train(trainset, root/"baseline.json", root/"preparation.json", root/"sft", "sft", recipe, device="cpu")
                self.assertEqual(sft["status"], "engineering_budget_complete")
                resumed = train(trainset, root/"baseline.json", root/"preparation.json", root/"sft-resumed", "sft", recipe,
                                device="cpu", resume=root/"sft/step-1.pt")
                self.assertEqual(resumed["updates"], 2)
                with patch("b4dl_core.reasoning.trainer.runtime_identity", return_value={"different": True}):
                    with self.assertRaises(ValueError):
                        train(trainset, root/"baseline.json", root/"preparation.json", root/"bad-resume", "sft", recipe,
                              device="cpu", resume=root/"sft/step-1.pt")
                first = torch.load(root/"sft/step-2.pt", map_location="cpu", weights_only=True)
                second = torch.load(root/"sft-resumed/step-2.pt", map_location="cpu", weights_only=True)
                for key, value in first["parameters"].items():
                    torch.testing.assert_close(value, second["parameters"][key], atol=0, rtol=0)
                grpo_recipe = {**recipe, "max_updates": 1}
                grpo = train(trainset, root/"baseline.json", root/"preparation.json", root/"grpo", "grpo", grpo_recipe,
                             device="cpu", sft_checkpoint=root/"sft/step-2.pt")
                self.assertEqual(grpo["status"], "engineering_budget_complete")
                tokenizer, net, checkpoint = load_model(testset, root/"baseline.json", root/"preparation.json",
                                                       root/"grpo/step-1.pt", "cpu")
                self.assertEqual(checkpoint["stage"], "grpo")
                from b4dl_core.io import sha256
                sft_lineage = checkpoint_lineage(first, sha256(root/"sft/step-2.pt"), trainset.manifest["dataset_id"])
                grpo_lineage = checkpoint_lineage(checkpoint, sha256(root/"grpo/step-1.pt"), trainset.manifest["dataset_id"])
                self.assertEqual(sft_lineage, grpo_lineage)
                protocol = {"protocol_id": "six-fixture", "generation": {"do_sample": False}}
                with patch("b4dl_core.reasoning.evaluation.load_free_six_task", return_value=(protocol, [])), \
                     patch("b4dl_core.six_task.infer_predictions"):
                    six_run = infer_six_task(testset, net, tokenizer, root/"grpo/step-1.pt", root/"unused-protocol",
                        root/"six-inference", baseline=root/"baseline.json", preparation=root/"preparation.json",
                        checkpoint_metadata=checkpoint)
                self.assertEqual(six_run["lineage"], sft_lineage)
                self.assertEqual(six_run["baseline_sha256"], sha256(root/"baseline.json"))
                answers = {r["prompt"]: encode_trace(testset.targets[r["request_id"]]["trace"]) for r in testset.requests}
                def sample(current, tok, model_input, max_new_tokens, do_sample=False):
                    text = answers[model_input["prompt"]]
                    return {"text": text, "status": "ok", "ids": tok.encode(text) + [tok.eos_token_id]}
                with patch("b4dl_core.reasoning.trainer.sample_one", sample):
                    infer(testset, net, tokenizer, root/"grpo/step-1.pt", root/"predictions", max_new_tokens=512)
                    infer(testset, net, tokenizer, root/"grpo/step-1.pt", root/"predictions", max_new_tokens=512, resume=True)
            predictions = read_jsonl(root/"predictions/predictions.jsonl")
            report = evaluate(testset, predictions, read_json(root/"predictions/run.json"), repetitions=10)
            self.assertEqual(report["summary"]["accuracy"], 1.)
            self.assertEqual(report["summary"]["whole_trace_pass_rate"], 1.)
            self.assertEqual(len(predictions), len(testset.requests))

    def test_free_six_task_protocol_binds_sources_and_never_uses_gt_for_input(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root/"mllm/b4dl_dataset"
            assets.mkdir(parents=True)
            item = {"scene_id": "s0", "task": "time_grounding", "conversations": [
                {"from": "human", "value": "When does the car stop?"},
                {"from": "gpt", "value": "from frame 6 to frame 8"}]}
            write_json(assets/"test_qa.json", [item])
            write_json(assets/"ego_metadata.json", {"s0": {"first_frame": "start", "last_frame": "end"}})
            write_json(assets/"ego_frame_motion.json", {})
            write_json(root/"historical.json", {"fixture": True})
            geometry = SimpleNamespace(manifest={"dataset_id": "fixture", "scenes": {"s0": {"split": "test", "sha256": "a"*64}}})
            old = {"generation": {"do_sample": False}, "metrics": {}, "backend_sources": {}, "files_sha256": {}}
            rows = [{"source_index": 0, "scene_id": "s0", "task": "time_grounding", "target": item["conversations"][1]["value"]}]
            with patch("b4dl_core.six_task.validate_protocol", return_value=(old, rows)):
                protocol = prepare_free_six_task(root, geometry, root/"historical.json", root/"free")
            loaded, queries = load_free_six_task(root/"free/manifest.json", geometry)
            self.assertEqual(protocol["protocol_id"], loaded["protocol_id"])
            self.assertNotIn("frame 6", queries[0]["query"])
            self.assertFalse(loaded["input_protocol"]["oracle_assisted_metadata"])
            write_json(assets/"test_qa.json", [{**item, "scene_id": "changed"}])
            with self.assertRaises(ValueError): load_free_six_task(root/"free/manifest.json", geometry)

    def test_six_task_comparison_requires_matching_completed_metrics(self):
        from b4dl_core.io import sha256
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = {"schema": "b4dl-reasoning-six-task-inference-v1", "protocol_id": "same",
                   "model_kind": "reasoning", "model_artifact_sha256": "artifact",
                   "lineage": {"common": {"baseline": "same"}, "sft_origin_sha256": "same"},
                   "training": {"stage": "grpo", "recipe": {"mode": "process", "seed": 42}, "step": 1,
                                "execution": {"device": "cpu"}, "planned_full_budget": True},
                   "baseline_sha256": "baseline", "preparation_sha256": "prep", "implementation": "code",
                   "generation": {"do_sample": False}, "runtime": {"torch": "same"},
                   "device": "cpu", "dtype": "torch.float32", "gpu_name": None,
                   "oracle_assisted_metadata": False}
            for name, value in (("base", .5), ("candidate", .6)):
                folder = root/name
                folder.mkdir()
                write_json(folder/"metrics.json", {"per_task_metrics": {"binary_qa": {"accuracy": value}},
                                                   "metric_backend": {"fixture": True}})
                write_json(folder/"completion.json", {"status": "six_task_metrics_complete", "protocol_id": "same",
                    "requests": 1, "metrics_sha256": sha256(folder/"metrics.json"), "inference_run": run})
            result = compare_six_task_metrics(root/"base", root/"candidate")
            self.assertAlmostEqual(result["per_task_metrics"]["binary_qa"]["accuracy"]["delta"], .1)
            receipt = read_json(root/"candidate/completion.json")
            b3_receipt = read_json(root/"base/completion.json")
            write_json(root/"base/completion.json", {**b3_receipt, "inference_run":
                {**run, "model_kind": "original_b3", "model_artifact_sha256": "baseline", "lineage": None, "training": None}})
            self.assertEqual(compare_six_task_metrics(root/"base", root/"candidate")["base_model"]["kind"], "original_b3")
            write_json(root/"base/completion.json", b3_receipt)
            for change in ({"dtype": "torch.bfloat16"}, {"baseline_sha256": "other"},
                           {"lineage": {**run["lineage"], "sft_origin_sha256": "other"}},
                           {"training": {**run["training"], "step": 2}}):
                write_json(root/"candidate/completion.json", {**receipt, "inference_run": {**run, **change}})
                with self.assertRaises(ValueError): compare_six_task_metrics(root/"base", root/"candidate")
            write_json(root/"candidate/completion.json", receipt)
            (root/"candidate/metrics.json").write_text("{}")
            with self.assertRaises(ValueError): compare_six_task_metrics(root/"base", root/"candidate")


if __name__ == "__main__":
    unittest.main()
