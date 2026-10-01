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


if __name__ == '__main__':
    unittest.main()
