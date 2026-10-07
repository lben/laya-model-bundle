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
from laya_chunking import TextPlan


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
        self.stack.enter_context(patch('laya_chunking.analyze_text', return_value=TextPlan([], 0, 8192, False, 512, 192)))

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

    def test_message_first_with_options_prints_every_probability(self):
        self.agent.predict.return_value = {'answers': {'bucket': {'choice': 'billing', 'probabilities': {'billing': 0.8, 'technical support': 0.2}}}}
        self.assertEqual(self.run_cli('I was charged twice.', '--options', ' billing , technical support '), 0)
        state, questions = self.agent.predict.call_args.args
        self.assertEqual(state, {'message': 'I was charged twice.'})
        self.assertEqual(questions['bucket']['criteria'], {'billing': 'billing', 'technical support': 'technical support'})
        self.assertIn('billing: 0.8000', self.stdout.getvalue())
        self.assertIn('technical support: 0.2000', self.stdout.getvalue())
        self.presets.triage_questions.assert_not_called()

    def test_options_json_and_literal_message(self):
        message = 'From the following options this is the literal message'
        self.assertEqual(self.run_cli(message, '--options', 'A=billing,B=technical support', '--json'), 0)
        self.assertEqual(self.agent.predict.call_args.args[0], {'message': message})
        self.assertEqual(self.agent.predict.call_args.args[1]['bucket']['criteria'], {'A': 'billing', 'B': 'technical support'})
        self.assertEqual(json.loads(self.stdout.getvalue()), self.agent.predict.return_value)

    def test_interactive_options_reuse_categories_and_model(self):
        with patch('builtins.input', side_effect=['First message', 'Second message', 'quit']):
            self.assertEqual(self.run_cli('--options', 'billing,other'), 0)
        self.assertEqual(self.laya.load.call_count, 1)
        self.assertEqual(self.agent.predict.call_count, 2)
        for call in self.agent.predict.call_args_list:
            self.assertEqual(call.args[1]['bucket']['criteria'], {'billing': 'billing', 'other': 'other'})
        self.assertIn('Options: billing, other', self.stdout.getvalue())
        self.presets.triage_questions.assert_not_called()

    def test_invalid_options_fail_before_loading_model(self):
        for options in ['', 'billing', ',other', 'billing,', 'billing,,other', 'billing,BILLING', 'A=,B=other', '=billing,B=other']:
            with self.subTest(options=options):
                self.assertEqual(self.run_cli('My message', '--options', options), 1)
        self.laya.load.assert_not_called()
        self.agent.predict.assert_not_called()

    def test_options_and_preset_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_cli('My message', '--options', 'billing,other', '--preset', 'triage')
        self.assertEqual(raised.exception.code, 2)
        self.laya.load.assert_not_called()

    def test_large_message_and_options_files_with_message_first(self):
        message = ('A lengthy customer message with quotes " and punctuation & |.\r\n' * 2000)
        description = 'billing and refund requests ' * 1000
        message_file = self.root / 'message with spaces.txt'
        options_file = self.root / 'options with spaces.txt'
        message_file.write_bytes(message.encode('utf-8'))
        options_file.write_bytes(f'A={description},\r\n B=other'.encode('utf-8'))
        self.assertEqual(self.run_cli(f'@{message_file}', '--options', f'@{options_file}', '--json'), 0)
        state, questions = self.agent.predict.call_args.args
        self.assertEqual(state, {'message': message})
        self.assertEqual(questions['bucket']['criteria'], {'A': description.strip(), 'B': 'other'})
        self.assertEqual(json.loads(self.stdout.getvalue()), self.agent.predict.return_value)

    def test_explicit_file_flags_support_powershell_encodings(self):
        message = '  Crème brûlée, 中文, and a refund request.\r\n'
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16']:
            with self.subTest(encoding=encoding):
                message_file = self.root / 'message.txt'
                options_file = self.root / 'options.txt'
                message_file.write_bytes(message.encode(encoding))
                options_file.write_bytes('billing, other\r\n'.encode(encoding))
                self.assertEqual(self.run_cli('--text-file', str(message_file), '--options-file', str(options_file)), 0)
                self.assertEqual(self.agent.predict.call_args.args[0], {'message': message})
                self.assertEqual(self.agent.predict.call_args.args[1]['bucket']['criteria'], {'billing': 'billing', 'other': 'other'})

    def test_file_and_inline_inputs_can_be_mixed(self):
        message_file = self.root / 'message.txt'
        options_file = self.root / 'options.txt'
        message_file.write_text('Refund please', encoding='utf-8')
        options_file.write_text('billing,other', encoding='utf-8')
        self.assertEqual(self.run_cli(f'@{message_file}', '--options', 'billing,other'), 0)
        self.assertEqual(self.agent.predict.call_args.args[0], {'message': 'Refund please'})
        self.assertEqual(self.run_cli('Inline message', '--options-file', str(options_file)), 0)
        self.assertEqual(self.agent.predict.call_args.args[0], {'message': 'Inline message'})
        self.assertEqual(self.run_cli('--text-file', str(message_file), '--preset', 'triage'), 0)
        self.assertEqual(self.agent.predict.call_args.args[0], {'message': 'Refund please'})

    def test_interactive_options_file_reuses_categories(self):
        options_file = self.root / 'options.txt'
        options_file.write_text('billing,other', encoding='utf-8')
        with patch('builtins.input', side_effect=['First', 'Second', 'quit']):
            self.assertEqual(self.run_cli('--options-file', str(options_file)), 0)
        self.assertEqual(self.laya.load.call_count, 1)
        self.assertEqual(self.agent.predict.call_count, 2)
        for call in self.agent.predict.call_args_list:
            self.assertEqual(call.args[1]['bucket']['criteria'], {'billing': 'billing', 'other': 'other'})

    def test_missing_empty_and_invalid_files_fail_before_loading(self):
        empty = self.root / 'empty.txt'
        empty.write_text(' \r\n', encoding='utf-8')
        invalid = self.root / 'invalid.txt'
        invalid.write_bytes(b'\x80')
        bad_options = self.root / 'bad-options.txt'
        bad_options.write_text('billing,BILLING', encoding='utf-8')
        cases = [
            (f'@{self.root / "missing.txt"}',),
            ('message', '--options', f'@{self.root / "missing.txt"}'),
            ('--text-file', str(empty)),
            (f'@{empty}',),
            ('message', '--options-file', str(empty)),
            (f'@{invalid}',),
            ('message', '--options-file', str(invalid)),
            ('message', '--options-file', str(bad_options)),
            ('@',),
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args), 1)
        self.laya.load.assert_not_called()
        self.agent.predict.assert_not_called()

    def test_conflicting_file_inputs_are_rejected(self):
        for args in [('message', '--text-file', 'unused.txt'), ('@unused.txt', 'extra')]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args), 1)
        for args in [('--options', 'A,B', '--options-file', 'unused.txt'), ('--options-file', 'unused.txt', '--preset', 'triage')]:
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                self.run_cli(*args)
            self.assertEqual(raised.exception.code, 2)
        self.laya.load.assert_not_called()

    def test_literal_leading_at_sign_can_be_escaped(self):
        self.assertEqual(self.run_cli('@@customer requests a refund', '--options', '@@account,other'), 0)
        self.assertEqual(self.agent.predict.call_args.args[0], {'message': '@customer requests a refund'})
        self.assertEqual(self.agent.predict.call_args.args[1]['bucket']['criteria'], {'@account': '@account', 'other': 'other'})

    def test_token_budgets_reach_options_plain_text_and_presets(self):
        cases = [
            ('message', '--options', 'billing,other'),
            ('From the following options A B categorize the following message: message',),
            ('message', '--preset', 'triage'),
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args, '--max-len', '4096', '--head-max-len', '3072'), 0)
                self.assertEqual(self.agent.predict.call_args.kwargs, {'max_len': 4096, 'head_max_len': 3072})

    def test_interactive_token_budgets_are_reused(self):
        with patch('builtins.input', side_effect=['First', 'Second', 'quit']):
            self.assertEqual(self.run_cli('--options', 'A,B', '--max-len', '4096', '--head-max-len', '3072'), 0)
        for call in self.agent.predict.call_args_list:
            self.assertEqual(call.kwargs, {'max_len': 4096, 'head_max_len': 3072})

    def test_invalid_token_budgets_fail_before_model_load(self):
        for args in [('--max-len', '0'), ('--head-max-len', '-1'), ('--max-len', '512', '--head-max-len', '512')]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli('message', *args), 1)
        self.laya.load.assert_not_called()

    def test_379_file_options_are_preserved_with_larger_budget(self):
        options = [f'category_{i}' for i in range(379)]
        path = self.root / 'options.txt'
        path.write_text(','.join(options), encoding='utf-8')
        self.assertEqual(self.run_cli('message', '--options-file', str(path), '--max-len', '4096', '--head-max-len', '3072'), 0)
        self.assertEqual(list(self.agent.predict.call_args.args[1]['bucket']['criteria']), options)

    def test_top_n_text_ranks_and_preserves_full_inference_options(self):
        self.agent.predict.return_value = {'answers': {'bucket': {'choice': 'C', 'probabilities': {'A': 0.1, 'B': 0.3, 'C': 0.5, 'D': 0.1}}}}
        self.assertEqual(self.run_cli('message', '--options', 'A,B,C,D', '--top-n', '2'), 0)
        output = self.stdout.getvalue()
        self.assertIn('C: 0.5000', output)
        self.assertIn('B: 0.3000', output)
        self.assertLess(output.index('  C:'), output.index('  B:'))
        self.assertNotIn('  A:', output)
        self.assertNotIn('  D:', output)
        self.assertEqual(list(self.agent.predict.call_args.args[1]['bucket']['criteria']), ['A', 'B', 'C', 'D'])
        self.assertEqual(self.agent.predict.call_args.kwargs, {})

    def test_top_n_json_does_not_renormalize_or_mutate_sdk_result(self):
        original = {'answers': {'bucket': {'choice': 'B', 'confidence': 0.6, 'probabilities': {'A': 0.2, 'B': 0.5, 'C': 0.3}},
                                'urgency': {'score': 1, 'probabilities': {'0': 0.2, '1': 0.8}}}, 'usage': {'input_tokens': 123}}
        self.agent.predict.return_value = original
        self.assertEqual(self.run_cli('message', '--options', 'A,B,C', '--top-n', '2', '--json'), 0)
        result = json.loads(self.stdout.getvalue())
        self.assertEqual(result['answers']['bucket']['probabilities'], {'B': 0.5, 'C': 0.3})
        self.assertEqual(list(result['answers']['bucket']['probabilities']), ['B', 'C'])
        self.assertEqual(result['answers']['bucket']['choice'], 'B')
        self.assertEqual(result['answers']['bucket']['confidence'], 0.6)
        self.assertEqual(result['answers']['urgency'], original['answers']['urgency'])
        self.assertEqual(result['usage'], original['usage'])
        self.assertEqual(original['answers']['bucket']['probabilities'], {'A': 0.2, 'B': 0.5, 'C': 0.3})

    def test_top_n_ties_and_larger_than_option_count(self):
        result = {'answers': {'bucket': {'choice': 'C', 'probabilities': {'A': 0.4, 'B': 0.2, 'C': 0.4}}}}
        self.assertEqual(laya_local.limit_probabilities(result, 1)['answers']['bucket']['probabilities'], {'C': 0.4})
        self.assertEqual(list(laya_local.limit_probabilities(result, 10)['answers']['bucket']['probabilities']), ['C', 'A', 'B'])
        self.assertIs(laya_local.limit_probabilities(result, None), result)

    def test_top_n_applies_to_presets_plain_text_and_interactive_mode(self):
        self.agent.predict.return_value = {'answers': {'bucket': {'choice': 'A', 'probabilities': {'A': 0.7, 'B': 0.3}}}}
        with patch('builtins.input', side_effect=['From the following options A B categorize the following message: First', 'Ordinary preset entry', 'quit']):
            self.assertEqual(self.run_cli('--top-n', '1'), 0)
        self.assertEqual(self.agent.predict.call_count, 2)
        self.assertEqual(self.stdout.getvalue().count('  A: 0.7000'), 2)
        self.assertNotIn('  B:', self.stdout.getvalue())

    def test_invalid_top_n_is_rejected_before_loading(self):
        for n in ['0', '-1']:
            with self.subTest(n=n):
                self.assertEqual(self.run_cli('message', '--top-n', n), 1)
        with self.assertRaises(SystemExit) as raised:
            self.run_cli('message', '--top-n', '1.5')
        self.assertEqual(raised.exception.code, 2)
        self.laya.load.assert_not_called()

    def test_chunking_controls_forwarded_and_top_n_after_scan(self):
        result = {'answers': {'bucket': {'choice': 'B', 'probabilities': {'A': 0.2, 'B': 0.8},
                                         'window': {'index': 4, 'count': 5}}},
                  'chunking': {'mode': 'on', 'chunks': 5, 'covered_tokens': 30000}, 'usage': {'truncated': False}}
        with patch.object(laya_local, 'predict_text', return_value=result) as scan:
            self.assertEqual(self.run_cli('message', '--options', 'A,B', '--chunking', 'on', '--chunk-tokens', '1000',
                                          '--chunk-overlap', '64', '--top-n', '1', '--json'), 0)
        self.assertEqual(scan.call_args.kwargs['chunking'], 'on')
        self.assertEqual(scan.call_args.kwargs['chunk_tokens'], 1000)
        self.assertEqual(scan.call_args.kwargs['chunk_overlap'], 64)
        output = json.loads(self.stdout.getvalue())
        self.assertEqual(output['answers']['bucket']['probabilities'], {'B': 0.8})
        self.assertEqual(output['chunking']['covered_tokens'], 30000)
        self.assertEqual(output['answers']['bucket']['window']['index'], 4)

    def test_invalid_chunk_parameters_fail_before_loading(self):
        for args in [('--chunk-tokens', '0'), ('--chunk-overlap', '-1'), ('--chunk-tokens', '10', '--chunk-overlap', '10')]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli('message', *args), 1)
        self.laya.load.assert_not_called()


class OptionsFlagTests(unittest.TestCase):
    def test_comma_list_preserves_multiword_names_and_and(self):
        questions = laya_local.parse_options(' customer support , research and development , and , and more ')
        self.assertEqual(list(questions['bucket']['criteria']), ['customer support', 'research and development', 'and', 'and more'])

    def test_labels_and_descriptions(self):
        questions = laya_local.parse_options(' A = billing and refunds , B: technical support ')
        self.assertEqual(questions['bucket']['criteria'], {'A': 'billing and refunds', 'B': 'technical support'})


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
