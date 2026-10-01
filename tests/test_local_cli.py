import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import laya_local


class LocalCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'laya'
        entries = []
        for name, data in [('model.safetensors', b'unit-test fixture weights'), ('tokenizer/tokenizer_config.json', b'{"tokenizer_class":"TokenizersBackend"}')]:
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entries.append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        self.model = {'name': 'english', 'directory': 'laya', 'files': entries}
        self.agent = Mock()
        self.agent.device = types.SimpleNamespace(type='cpu')
        self.agent.predict.return_value = {'answers': {'test': {'type': 'noul', 'noul': 0.75}}}
        self.laya = types.ModuleType('laya')
        self.laya.load = Mock(return_value=self.agent)
        self.presets = types.ModuleType('laya.presets')
        for function, field in laya_local.PRESETS.values():
            setattr(self.presets, function, Mock(return_value={'test': {'type': 'noul', 'instructions': f'Read {field}'}}))
        self.torch = types.ModuleType('torch')
        self.torch.set_num_threads = Mock()
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict('sys.modules', {'torch': self.torch, 'laya': self.laya, 'laya.presets': self.presets}))
        self.stack.enter_context(patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}))
        self.stack.enter_context(patch.object(laya_local, 'load_manifest', return_value={'models': [self.model]}))
        self.stack.enter_context(contextlib.redirect_stdout(self.stdout))
        self.stack.enter_context(contextlib.redirect_stderr(self.stderr))

    def run_cli(self, *args):
        return laya_local.main(['--models-dir', str(self.root), *args])

    def test_all_presets_use_absolute_local_model_and_correct_state_field(self):
        for preset, (function, field) in laya_local.PRESETS.items():
            with self.subTest(preset=preset):
                self.assertEqual(self.run_cli('--preset', preset, '--json', 'My request'), 0)
                self.laya.load.assert_called_with(str(self.directory.resolve()), device='cpu', fast=False, compile=False)
                self.agent.predict.assert_called_with({field: 'My request'}, getattr(self.presets, function)())
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')

    def test_interactive_reuses_model_for_multiple_prompts(self):
        with patch('builtins.input', side_effect=['First request', '', 'Second request', 'quit']):
            self.assertEqual(self.run_cli('--preset', 'triage'), 0)
        self.assertEqual(self.laya.load.call_count, 1)
        self.assertEqual(self.agent.predict.call_count, 2)
        self.assertIn('Preset: triage', self.stdout.getvalue())

    def test_missing_local_model_does_not_fall_back_to_hub(self):
        self.assertEqual(self.run_cli('--models-dir', str(self.root / 'missing'), 'My request'), 1)
        self.laya.load.assert_not_called()
        self.assertIn('restore.py --model english', self.stderr.getvalue())

    def test_changed_model_is_rejected_before_loading(self):
        (self.directory / 'model.safetensors').write_bytes(b'changed')
        self.assertEqual(self.run_cli('My request'), 1)
        self.laya.load.assert_not_called()

    def test_network_is_blocked_during_load_and_prediction(self):
        def blocked():
            with socket.socket() as connection:
                with self.assertRaisesRegex(OSError, 'Network access is disabled'):
                    connection.connect(('127.0.0.1', 1))
                with self.assertRaisesRegex(OSError, 'Network access is disabled'):
                    connection.connect_ex(('127.0.0.1', 1))
            with self.assertRaisesRegex(OSError, 'Network access is disabled'):
                socket.create_connection(('127.0.0.1', 1))
        def load(*args, **kwargs):
            blocked()
            return self.agent
        def predict(*args, **kwargs):
            blocked()
            return {'answers': {'test': {'noul': 0.75}}}
        self.laya.load.side_effect = load
        self.agent.predict.side_effect = predict
        self.assertEqual(self.run_cli('My request'), 0)

    def test_tokenizer_patch_is_reverted_after_success_and_failure(self):
        path = self.directory / 'tokenizer/tokenizer_config.json'
        original = path.read_bytes()
        def load(*args, **kwargs):
            path.write_bytes(b'SDK normalization')
            return self.agent
        self.laya.load.side_effect = load
        self.assertEqual(self.run_cli('My request'), 0)
        self.assertEqual(path.read_bytes(), original)
        def failure(*args, **kwargs):
            path.write_bytes(b'SDK normalization')
            raise RuntimeError('load failed')
        self.laya.load.side_effect = failure
        self.assertEqual(self.run_cli('My request'), 1)
        self.assertEqual(path.read_bytes(), original)

    def test_json_output_is_machine_readable(self):
        self.assertEqual(self.run_cli('--json', 'My request'), 0)
        self.assertEqual(json.loads(self.stdout.getvalue()), self.agent.predict.return_value)

    def test_interactive_inference_error_keeps_session_open(self):
        self.agent.predict.side_effect = [ValueError('bad request'), {'answers': {'test': {'noul': 0.75}}}]
        with patch('builtins.input', side_effect=['First', 'Second', 'exit']):
            self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.agent.predict.call_count, 2)
        self.assertIn('bad request', self.stderr.getvalue())

    def test_plain_text_buckets_print_all_probabilities_and_reuse_local_model(self):
        choice = {'answers': {'bucket': {'type': 'choice', 'choice': 'A', 'probabilities': {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}}}}
        self.agent.predict.side_effect = [choice, {'answers': {'test': {'noul': 0.75}}}]
        instruction = 'From the following options A=billing | B=technical support | C=sales | D=other categorize the following message on each one: “I was charged twice.”'
        with patch('builtins.input', side_effect=[instruction, 'Ordinary preset message', 'quit']):
            self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.laya.load.call_count, 1)
        first = self.agent.predict.call_args_list[0]
        self.assertEqual(first.args[0], {'message': 'I was charged twice.'})
        self.assertEqual(first.args[1]['bucket']['criteria'], {'A': 'billing', 'B': 'technical support', 'C': 'sales', 'D': 'other'})
        self.assertEqual(first.args[1]['bucket']['type'], 'choice')
        second = self.agent.predict.call_args_list[1]
        self.assertEqual(second.args[1], self.presets.triage_questions())
        for line in ['A: 0.4000', 'B: 0.3000', 'C: 0.2000', 'D: 0.1000']:
            self.assertIn(line, self.stdout.getvalue())

    def test_bad_bucket_instruction_does_not_reach_inference(self):
        self.assertEqual(self.run_cli('From the following options A B categorize the following message on each one: ""'), 1)
        self.agent.predict.assert_not_called()
        self.assertIn('message to categorize is empty', self.stderr.getvalue())


class PlainTextBucketTests(unittest.TestCase):
    def parse(self, options, message='“I was charged twice.”'):
        return laya_local.parse_bucket_request(f'From the following options {options} categorize the following message on each one: {message}')

    def test_exact_requested_format_with_letters_and_curly_quotes(self):
        text, questions = self.parse('A B C and D', '“message”')
        self.assertEqual(text, 'message')
        self.assertEqual(questions['bucket']['criteria'], {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'})

    def test_named_buckets_with_natural_final_and(self):
        for options in ['billing, technical, sales and other', 'billing, technical, sales, and other', 'billing technical sales and other']:
            with self.subTest(options=options):
                _, questions = self.parse(options)
                self.assertEqual(list(questions['bucket']['criteria']), ['billing', 'technical', 'sales', 'other'])

    def test_explicit_descriptions_and_multiword_categories(self):
        for delimiter in [', ', '; ', ' | ']:
            with self.subTest(delimiter=delimiter):
                _, questions = self.parse(delimiter.join(['A=billing and refunds', 'B: technical support', 'C=sales', 'D=research and development']))
                self.assertEqual(questions['bucket']['criteria'], {'A': 'billing and refunds', 'B': 'technical support', 'C': 'sales', 'D': 'research and development'})
        _, questions = self.parse('customer support | sales | research and development')
        self.assertEqual(list(questions['bucket']['criteria']), ['customer support', 'sales', 'research and development'])

    def test_message_quotes_and_punctuation_are_preserved(self):
        for quoted in ['"Say \'hello\': A, B, C."', '“Say \'hello\': A, B, C.”', "‘Say 'hello': A, B, C.’", "Say 'hello': A, B, C."]:
            with self.subTest(quoted=quoted):
                message, _ = self.parse('A B', quoted)
                self.assertEqual(message, "Say 'hello': A, B, C.")

    def test_invalid_options_and_malformed_instruction(self):
        for options in ['A', 'A a', 'A= | B=technical', '=billing | B=technical', 'A | | B']:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.parse(options)
        with self.assertRaises(ValueError):
            laya_local.parse_bucket_request('From the following options A B missing the message instruction')

    def test_ordinary_text_uses_preset_and_format_is_case_insensitive(self):
        self.assertIsNone(laya_local.parse_bucket_request('Please refund the duplicate charge.'))
        message, questions = laya_local.parse_bucket_request('FROM THE FOLLOWING OPTIONS: billing, other CLASSIFY THE FOLLOWING MESSAGE: Duplicate charge')
        self.assertEqual(message, 'Duplicate charge')
        self.assertEqual(list(questions['bucket']['criteria']), ['billing', 'other'])


if __name__ == '__main__':
    unittest.main()
