"""Batch collection must preserve the original single-sample greedy outputs."""

import unittest

import torch
from answerseg.language import LanguageBridge, make_prompt
from scripts.collect_mini_answers import greedy_batch, question_range
from test_answerseg import tiny_language


class MiniCollectionTests(unittest.TestCase):
    def test_batch_matches_serial_with_different_prompt_lengths(self):
        torch.manual_seed(203)
        tokenizer, model = tiny_language()
        model.eval()
        prompts = [make_prompt("truck"), make_prompt("The car ahead. The truck ahead.")]
        features = [torch.randn(3, 768), torch.randn(3, 768)]
        batch = greedy_batch(model, tokenizer, prompts, features, 6)
        serial = [
            greedy_batch(model, tokenizer, [p], [f], 6)[0]
            for p, f in zip(prompts, features)
        ]
        self.assertEqual(
            [r["generated_token_ids"] for r in batch],
            [r["generated_token_ids"] for r in serial],
        )
        self.assertEqual(
            [r["finished"] for r in batch], [r["finished"] for r in serial]
        )
        self.assertFalse(hasattr(model.config, "tokenizer_padding_side"))
        legacy = LanguageBridge(model, tokenizer, None)
        for prompt, feature, value in zip(prompts, features, serial):
            expected = legacy.generate(prompt, feature, 6)
            self.assertEqual(value["answer"], expected["answer"])

    def test_visual_length_mismatch_rejected_before_cached_generation(self):
        tokenizer, model = tiny_language()
        with self.assertRaisesRegex(ValueError, "equal visual lengths"):
            greedy_batch(
                model,
                tokenizer,
                [make_prompt("truck")] * 2,
                [torch.randn(3, 768), torch.randn(4, 768)],
                4,
            )

    def test_context_cannot_be_silently_truncated_by_multimodal_expansion(self):
        tokenizer, model = tiny_language()
        model.config.max_position_embeddings = 16
        with self.assertRaisesRegex(ValueError, "expanded multimodal context"):
            greedy_batch(
                model, tokenizer, [make_prompt("truck")], [torch.randn(3, 768)], 8
            )

    def test_metadata_range_is_from_question_or_whole_scene_only(self):
        self.assertEqual(
            question_range("What happens between frame 002 and frame 004?", 40),
            (2, 4, "question_frames"),
        )
        self.assertEqual(
            question_range("When does the car turn?", 40),
            (0, 39, "whole_scene_fallback"),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            question_range("Describe frame 040", 40)


if __name__ == "__main__":
    unittest.main()
