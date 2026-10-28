from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from b4dl_core import b3, six_task
from b4dl_core.data import build_dataset
from b4dl_core.inputs import GeometryDataset, control_plan
from b4dl_core.io import read_json, read_jsonl, sha256, write_json, write_jsonl
from b4dl_core.resume import (append_jsonl, incomplete_object, output_lease,
                              recover_jsonl, validate_predictions)
from test_core_b3_integration import tiny, TinyTokenizer, small_config
from test_core_geometry import raw_fixture


def requests(count=3):
    return [{'request_id': str(i), 'input_fingerprint': 'fp-'+str(i)} for i in range(count)]


def predictions(rows):
    return [{**r, 'status': 'ok', 'text': '测试'} for r in rows]


class DurableIOTests(unittest.TestCase):
    def test_atomic_json_and_jsonl_keep_previous_file_on_write_or_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            write_jsonl(path, predictions(requests()))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_jsonl(path, [{'good': 1}, {'bad': float('nan')}])
            self.assertEqual(path.read_bytes(), before)
            with patch('b4dl_core.io.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError): write_json(path, {'new': True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_tail_recovery_preserves_original_and_keeps_every_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            expected = predictions(requests())
            prefix = (json.dumps(expected[0], ensure_ascii=False)+'\n').encode()
            tails = [json.dumps(expected[1], ensure_ascii=False).encode(),
                     b'{"request_id": "1", "text": "partial',
                     b'{"request_id": "1", "text": "\xe6\xb5',
                     b'{"request_id": "1", "extra": tru',
                     b'{"request_id": "1", "extra": 1e+']
            for tail in tails:
                with self.subTest(tail=tail):
                    original = prefix+tail
                    path.write_bytes(original)
                    found = recover_jsonl(path, lambda rows: validate_predictions(requests(), rows))
                    self.assertEqual(found, expected[:2] if tail == tails[0] else expected[:1])
                    self.assertTrue(path.read_bytes().endswith(b'\n'))
                    backups = list(Path(directory).glob('*.bak'))
                    self.assertTrue(any(p.read_bytes() == original for p in backups))
                    append_jsonl(path, expected[len(found):])
                    self.assertEqual(read_jsonl(path), expected)

    def test_corruption_and_wrong_identity_do_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            good = predictions(requests())
            bad_contents = [
                b'{"bad": @', b'{"bad": "\\q', b'{"bad": 01', b'{"bad": truX',
                b'{"bad": 1,}', b'{"bad": 1}\n{broken}\n',
                b'{"bad": 1\n', b'{"bad": "\xff',
                (json.dumps(good[1])+'\n{"text": "cut').encode(),
                (json.dumps(good[0])+'\n'+json.dumps(good[0])).encode(),
                json.dumps({**good[0], 'input_fingerprint': 'wrong'}).encode(),
                (json.dumps(good[0])+json.dumps(good[1])).encode()]
            for content in bad_contents:
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaises(ValueError):
                        recover_jsonl(path, lambda rows: validate_predictions(requests(), rows))
                    self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(Path(directory).glob('*.bak')), [])

    def test_every_serialized_object_prefix_is_recoverable_but_malformed_json_is_not(self):
        value = {'text': '引号"和换行\n', 'items': [None, True, False, -1.2e-5, {}, []]}
        encoded = json.dumps(value, ensure_ascii=False)
        for size in range(1, len(encoded)):
            self.assertTrue(incomplete_object(encoded[:size]), encoded[:size])
        self.assertFalse(incomplete_object(encoded))
        for text in ('{"x": [1,]}', '{"x": [}', '{"x": NaN', '{"x": "\\uqq', '{}junk'):
            self.assertFalse(incomplete_object(text), text)

    def test_concurrent_writer_is_rejected_and_lock_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with output_lease(directory):
                with self.assertRaises(OSError):
                    with output_lease(directory): pass
            with output_lease(directory): pass


class GeometryResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tables, metadata, features, splits = raw_fixture(self.root)
        build_dataset(tables, metadata, features, splits, self.root/'dataset')
        self.dataset = GeometryDataset(self.root/'dataset', 'train')
        self.dataset.requests = self.dataset.requests[:3]
        write_json(self.root/'baseline.json', {'fixture': True})
        self.output = self.root/'output'
        self.output.mkdir()
        self.args = SimpleNamespace(device='cpu', arm='A0', lora_rank=0, checkpoint=None,
            baseline=str(self.root/'baseline.json'), split='train', condition='identity',
            output=str(self.output), batch_size=1, max_new_tokens=2, seed=42, resume_inference=False)
        self.preparation = {'frame_adapter': asdict(small_config())}
        self.loader = patch.object(b3, 'load_original', side_effect=lambda *unused: (TinyTokenizer(), tiny()))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def infer(self):
        return b3.infer(self.args, self.preparation, {'baseline_id': 'test'}, self.dataset)

    def interrupt_after_first_batch(self):
        calls = []
        def generate(model, **batch):
            calls.append(1)
            if len(calls) == 2: raise RuntimeError('interrupted generation')
            return torch.tensor([[5, 2]])
        with patch('vtimellm.model.vtimellm_llama.VTimeLLMLlamaForCausalLM.generate', generate):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'): self.infer()
        self.assertTrue((self.output/'run.json').exists())
        self.assertTrue((self.output/'resume.json').exists())
        self.assertFalse((self.output/'completion.json').exists())

    def test_interruption_resume_and_completed_noop_have_no_duplicate_or_omission(self):
        self.interrupt_after_first_batch()
        first = read_jsonl(self.output/'predictions.jsonl')[0]
        self.args.resume_inference = True
        self.infer()
        result = read_jsonl(self.output/'predictions.jsonl')
        self.assertEqual(result[0], first)
        self.assertEqual([r['request_id'] for r in result], [r['request_id'] for r in self.dataset.requests])
        complete = read_json(self.output/'completion.json')
        self.assertEqual(complete['predictions_sha256'], sha256(self.output/'predictions.jsonl'))
        with patch('vtimellm.model.vtimellm_llama.VTimeLLMLlamaForCausalLM.generate', side_effect=AssertionError('must not generate')):
            self.infer()
        with (self.output/'predictions.jsonl').open('ab') as stream: stream.write(b' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.infer()

    def test_committed_pair_survives_export_failure_and_torn_journal_tail(self):
        real_append = b3.append_jsonl
        def interrupted(path, values):
            if Path(path).name == 'input-receipts.jsonl': raise OSError('export disk failure')
            return real_append(path, values)
        with patch.object(b3, 'append_jsonl', interrupted):
            with self.assertRaises(OSError): self.infer()
        transaction = read_jsonl(self.output/'inference-journal.jsonl')[0]
        # Simulate a process killed during its next journal append and a torn view.
        with (self.output/'inference-journal.jsonl').open('ab') as stream: stream.write(b'{"run_id": "cut')
        (self.output/'predictions.jsonl').write_bytes(b'broken view')
        self.args.resume_inference = True
        self.infer()
        result = read_jsonl(self.output/'predictions.jsonl')
        receipts = read_jsonl(self.output/'input-receipts.jsonl')
        self.assertEqual(result[0], transaction['entries'][0]['prediction'])
        self.assertEqual(receipts[0], transaction['entries'][0]['receipt'])
        self.assertEqual(len(result), 3)
        self.assertEqual(len(receipts), 3)
        self.assertEqual(len(list(self.output.glob('inference-journal.jsonl.*.bak'))), 1)

    def test_changed_run_inputs_weights_options_and_implementation_are_rejected(self):
        self.interrupt_after_first_batch()
        self.args.resume_inference = True
        changes = [('max_new_tokens', 3), ('batch_size', 2), ('condition', 'zero'), ('seed', 43)]
        before = (self.output/'inference-journal.jsonl').read_bytes()
        for key, value in changes:
            with self.subTest(key=key), patch.object(self.args, key, value):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.infer()
        for target, value in [('implementation_id', 'changed')]:
            with patch.object(b3, target, return_value=value):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.infer()
        with patch.object(self.dataset, 'requests', list(reversed(self.dataset.requests))):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.infer()
        write_json(self.root/'baseline.json', {'fixture': 'changed weights'})
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.infer()
        self.assertEqual((self.output/'inference-journal.jsonl').read_bytes(), before)

    def test_mixed_available_batch_retains_request_order(self):
        self.args.batch_size = 3
        controls = control_plan(self.dataset, 'identity')
        controls[0]['status'] = 'unavailable_fixture'
        with patch.object(b3, 'control_plan', return_value=controls): self.infer()
        result = read_jsonl(self.output/'predictions.jsonl')
        self.assertEqual([r['request_id'] for r in result], [r['request_id'] for r in self.dataset.requests])
        self.assertEqual(result[0]['status'], 'unavailable')

    def test_infer_cli_resumes_same_directory_and_keeps_training_resume_separate(self):
        write_json(self.root/'baseline.json', {'schema': 'b4dl-core-baseline-v1',
            'baseline_id': 'test', 'components': {'stage2': {'path': 'fixture'}}})
        write_json(self.root/'preparation.json', {**self.preparation, 'baseline_checkpoint': 'fixture'})
        output = self.root/'cli'
        argv = ['b3', 'infer', '--dataset', str(self.root/'dataset'), '--baseline', self.args.baseline,
            '--preparation', str(self.root/'preparation.json'), '--output', str(output),
            '--device', 'cpu', '--arm', 'A0', '--split', 'train', '--max-new-tokens', '2']
        generator = 'vtimellm.model.vtimellm_llama.VTimeLLMLlamaForCausalLM.generate'
        with patch.object(b3, 'GeometryDataset', return_value=self.dataset):
            with patch('sys.argv', argv), patch(generator, side_effect=[torch.tensor([[5, 2]]), RuntimeError('interrupt')]):
                with self.assertRaisesRegex(RuntimeError, 'interrupt'): b3.main()
            with patch('sys.argv', argv+['--resume-inference']), patch(generator, return_value=torch.tensor([[5, 2]])) as generate:
                b3.main()
                self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(read_jsonl(output/'predictions.jsonl')), 3)
        self.assertTrue((output/'completion.json').exists())

    def test_corrupt_committed_journal_is_rejected_without_rewriting_exports(self):
        self.interrupt_after_first_batch()
        self.args.resume_inference = True
        original = (self.output/'predictions.jsonl').read_bytes()
        path = self.output/'inference-journal.jsonl'
        journal_bytes = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing inference journal'): self.infer()
        self.assertEqual((self.output/'predictions.jsonl').read_bytes(), original)
        path.write_bytes(journal_bytes)
        journal = read_jsonl(path)
        journal[0]['entries'][0]['prediction']['text'] = 'tampered'
        write_jsonl(self.output/'inference-journal.jsonl', journal)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'): self.infer()
        self.assertEqual((self.output/'predictions.jsonl').read_bytes(), original)


class SixTaskResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.rows = requests()
        self.run = {'identity': {'implementation': 'test'}, 'protocol_id': 'test',
                    'checkpoint_sha256': 'test', 'generation': {'do_sample': False, 'max_new_tokens': 2}}
        self.model = Mock()
        self.model.generate.return_value = torch.tensor([[5, 2]])
        self.batch = patch.object(six_task, 'query_batch', return_value={})
        self.batch.start()
        self.addCleanup(self.batch.stop)

    def infer(self, resume=True):
        six_task.infer_predictions(self.output, self.run, self.rows, None, self.model, TinyTokenizer(), resume)

    def test_resume_without_newline_or_with_truncated_tail_is_complete_and_auditable(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                (self.output/'completion.json').unlink(missing_ok=True)
                write_json(self.output/'run.json', self.run)
                first = predictions(self.rows[:1])[0]
                content = json.dumps(first, ensure_ascii=False).encode()
                if truncated: content += b'\n{"request_id": "1", "text": "half'
                (self.output/'predictions.jsonl').write_bytes(content)
                self.model.generate.reset_mock()
                self.infer()
                self.assertEqual(self.model.generate.call_count, 2)
                result = read_jsonl(self.output/'predictions.jsonl')
                self.assertEqual(result[0], first)
                validate_predictions(self.rows, result, complete=True)
                self.assertEqual(read_json(self.output/'completion.json')['predictions_sha256'], sha256(self.output/'predictions.jsonl'))

    def test_corrupt_middle_wrong_id_duplicate_extra_and_unidentified_file_are_rejected(self):
        write_json(self.output/'run.json', self.run)
        valid = predictions(self.rows)
        for values in ([valid[1]], [valid[0], valid[0]], valid+[valid[0]]):
            write_jsonl(self.output/'predictions.jsonl', values)
            with self.assertRaises(ValueError): self.infer()
        (self.output/'predictions.jsonl').write_text('{bad}\n{}\n', encoding='utf-8')
        with self.assertRaises(ValueError): self.infer()
        (self.output/'run.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing resume identity'): self.infer()
        self.model.generate.assert_not_called()
        self.assertFalse((self.output/'completion.json').exists())

    def test_changed_parameters_or_weights_are_rejected_before_prediction_write(self):
        write_json(self.output/'run.json', self.run)
        write_jsonl(self.output/'predictions.jsonl', predictions(self.rows[:1]))
        before = (self.output/'predictions.jsonl').read_bytes()
        for key, value in [('checkpoint_sha256', 'other'), ('generation', {'max_new_tokens': 99}),
                           ('protocol_id', 'other'), ('identity', {'implementation': 'other'})]:
            with patch.dict(self.run, {key: value}):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.infer()
        self.assertEqual((self.output/'predictions.jsonl').read_bytes(), before)
        self.model.generate.assert_not_called()

    def test_final_revalidation_blocks_false_completion(self):
        real_append = six_task.append_jsonl
        def corrupt(path, values):
            real_append(path, values)
            if values[0]['request_id'] == self.rows[-1]['request_id']:
                with Path(path).open('ab') as stream: stream.write(b'{bad}\n')
        with patch.object(six_task, 'append_jsonl', corrupt):
            with self.assertRaises(ValueError): self.infer(resume=False)
        self.assertFalse((self.output/'completion.json').exists())

    def test_oom_retains_prefix_and_resume_completion_is_idempotent(self):
        self.model.generate.side_effect = [torch.tensor([[5, 2]]), torch.cuda.OutOfMemoryError('test')]
        with self.assertRaises(torch.cuda.OutOfMemoryError): self.infer(resume=False)
        self.assertEqual(len(read_jsonl(self.output/'predictions.jsonl')), 1)
        self.assertFalse((self.output/'completion.json').exists())
        self.model.generate.side_effect = None
        self.infer()
        self.model.generate.reset_mock()
        self.infer()
        self.model.generate.assert_not_called()

    def test_six_task_cli_recovers_unterminated_last_record(self):
        write_json(self.output/'baseline.json', {'fixture': True})
        write_json(self.output/'preparation.json', {'frame_adapter': asdict(small_config())})
        output = self.output/'cli'
        argv = ['six_task', 'infer', '--dataset', 'fixture', '--protocol', 'fixture', '--output', str(output),
            '--baseline', str(self.output/'baseline.json'), '--preparation', str(self.output/'preparation.json'), '--device', 'cpu']
        dataset = SimpleNamespace(manifest={'dataset_id': 'test'})
        protocol = {'protocol_id': 'test', 'generation': self.run['generation']}
        generator = 'vtimellm.model.vtimellm_llama.VTimeLLMLlamaForCausalLM.generate'
        with patch.object(six_task, 'GeometryDataset', return_value=dataset), \
             patch.object(six_task, 'validate_protocol', return_value=(protocol, self.rows)), \
             patch.object(six_task, 'load_original', side_effect=lambda *unused: (TinyTokenizer(), tiny())):
            with patch('sys.argv', argv), patch(generator, side_effect=[torch.tensor([[5, 2]]), torch.cuda.OutOfMemoryError('test')]):
                with self.assertRaises(torch.cuda.OutOfMemoryError): six_task.main()
            path = output/'predictions.jsonl'
            path.write_bytes(path.read_bytes().rstrip(b'\n'))
            with patch('sys.argv', argv+['--resume']), patch(generator, return_value=torch.tensor([[5, 2]])) as generate:
                six_task.main()
                self.assertEqual(generate.call_count, 2)
        validate_predictions(self.rows, read_jsonl(path), complete=True)
        self.assertEqual(read_json(output/'completion.json')['predictions_sha256'], sha256(path))


if __name__ == '__main__': unittest.main()
