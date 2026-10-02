import subprocess
import sys
import unittest
import argparse
import tempfile
from unittest.mock import patch
from pathlib import Path

MLLM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLLM_ROOT))
from vtimellm.conversation import conv_templates
from vtimellm.demo_inference import prepare_chat_prompt


class ChatContextTests(unittest.TestCase):
    def make_conversation(self):
        conversation = conv_templates["v1"].copy()
        conversation.messages = [
            [conversation.roles[0], "<4DLiDAR>\n<video>\n" + "old " * 80],
            [conversation.roles[1], "old answer"],
            [conversation.roles[0], "new question"],
            [conversation.roles[1], None],
        ]
        return conversation

    def test_prunes_full_turns_and_keeps_lidar_prefix(self):
        original = self.make_conversation()
        prepared, tokens, dropped = prepare_chat_prompt(
            original, lambda text: text.split(), 40, 100, 16,
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(len(prepared.messages), 2)
        self.assertEqual(prepared.get_prompt().count("<video>"), 1)
        self.assertLessEqual(len(tokens) - 1 + 40 + 16, 100)
        self.assertEqual(len(original.messages), 4)

    def test_rejects_oversize_current_question_without_mutating_state(self):
        original = self.make_conversation()
        before = original.dict()
        with self.assertRaisesRegex(ValueError, "超过上下文上限"):
            prepare_chat_prompt(original, lambda text: text.split(), 40, 50, 16)
        self.assertEqual(original.dict(), before)

    def test_module_cli_does_not_import_model_stack(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import runpy, sys; sys.argv=['demo', '--help']; "
                "\ntry: runpy.run_module('vtimellm.demo_gradio', run_name='__main__')"
                "\nexcept SystemExit as e: assert e.code == 0"
                "\nassert 'torch' not in sys.modules"
                "\nassert 'transformers' not in sys.modules"
            )], cwd=MLLM_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max_context_tokens", result.stdout)


class ModelStackTests(unittest.TestCase):
    def test_loader_precision_attention_and_cached_multimodal_generation(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
            from vtimellm.model.builder import load_pretrained_model
        except ImportError as exc:
            self.skipTest(f"optional inference dependencies: {exc}")
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )

        class TinyTokenizer:
            def add_special_tokens(self, value):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            LlamaForCausalLM(config).save_pretrained(directory)
            for dtype, attention in (("float16", "eager"), ("bfloat16", "sdpa")):
                with self.subTest(dtype=dtype, attention=attention):
                    args = argparse.Namespace(model_base=directory, pretrain_mm_mlp_adapter=None,
                                              dtype=dtype, attn_implementation=attention)
                    with patch("vtimellm.model.builder.AutoTokenizer.from_pretrained",
                               return_value=TinyTokenizer()):
                        _, model, context = load_pretrained_model(args)
                    model = model.to(getattr(torch, dtype)).eval()
                    self.assertEqual(context, 128)
                    self.assertEqual(model.dtype, getattr(torch, dtype))
                    self.assertEqual(model.config._attn_implementation, attention)
                    input_ids = torch.tensor([[1, 5, -200, 6, 7]])
                    images = torch.randn(1, 4, 768, dtype=model.dtype)
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=input_ids, images=images,
                            max_new_tokens=4, do_sample=False, use_cache=True, eos_token_id=None,
                        )
                        uncached = model.generate(
                            input_ids=input_ids, images=images,
                            max_new_tokens=4, do_sample=False, use_cache=False, eos_token_id=None,
                        )
                    self.assertEqual(output.shape[1], input_ids.shape[1] + 4)
                    torch.testing.assert_close(output, uncached)


if __name__ == "__main__":
    unittest.main()
