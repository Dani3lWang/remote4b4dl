from importlib import metadata
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from b4dl_core.runtime import VERSIONS, check_runtime
from test_core_b3_integration import tiny, tensors
from vtimellm.model.vtimellm_llama import VTimeLLMLlamaForCausalLM


class RuntimeTests(unittest.TestCase):
    def test_installed_stack_and_requirements_agree(self):
        report = check_runtime()
        self.assertEqual(report['packages']['transformers'], '4.47.0')
        requirements = (Path(__file__).parents[1]/'requirements.txt').read_text()
        for name, version in VERSIONS.items():
            self.assertIn(name+'=='+version, requirements)
        self.assertNotIn('flash_attn', requirements)

    def test_old_missing_and_broken_api_fail_before_loading_weights(self):
        actual = metadata.version
        for bad in ('4.31.0', None):
            def version(name):
                if name == 'transformers':
                    if bad is None: raise metadata.PackageNotFoundError(name)
                    return bad
                return actual(name)
            with patch('b4dl_core.runtime.metadata.version', version):
                with self.assertRaisesRegex(RuntimeError, 'transformers==4.47.0 required'): check_runtime()
        with patch('b4dl_core.runtime.inspect.signature', return_value=inspect.Signature()):
            with self.assertRaisesRegex(RuntimeError, 'cache_position'): check_runtime()
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'): check_runtime(require_cuda=True)

    def test_small_vtimellm_forward_generation_and_full_save_reload(self):
        torch.set_num_threads(2)
        model, batch = tiny(), tensors()
        model.config.mm_hidden_size = 768
        with torch.no_grad(): logits = model(**batch).logits
        tokens = model.generate(**batch, max_new_tokens=3, do_sample=False)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            # The legacy architecture creates its projector from Stage 1 explicitly.
            restored = VTimeLLMLlamaForCausalLM(model.config)
            restored.get_model().initialize_vision_modules(type('Args', (), {'pretrain_mm_mlp_adapter': None})())
            from safetensors.torch import load_file
            restored.load_state_dict(load_file(str(Path(directory)/'model.safetensors')), strict=True)
            restored.eval()
            with torch.no_grad(): after = restored(**batch).logits
            torch.testing.assert_close(logits, after, rtol=0, atol=0)
            torch.testing.assert_close(tokens, restored.generate(**batch, max_new_tokens=3, do_sample=False), rtol=0, atol=0)

    def test_six_task_tokenization_import_does_not_require_video_or_image_packages(self):
        from vtimellm.mm_utils import tokenizer_image_token
        self.assertTrue(callable(tokenizer_image_token))


if __name__ == '__main__': unittest.main()
