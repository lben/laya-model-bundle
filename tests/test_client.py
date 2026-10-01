import contextlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
import types
import unittest
from unittest.mock import Mock, patch

from laya_client import LocalLaya, options_questions
from laya_api import make_handler


class ClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / 'laya'
        for name in ['model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
                     'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']:
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}')
        self.agent = Mock()
        self.agent.device = types.SimpleNamespace(type='cpu')
        self.answer = {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 0.8, 'other': 0.2}, 'confidence': 0.5}
        self.agent.predict.return_value = {'answers': {'bucket': self.answer}}
        self.laya = types.ModuleType('laya')
        self.laya.load = Mock(return_value=self.agent)
        self.torch = types.ModuleType('torch')
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict('sys.modules', {'torch': self.torch, 'laya': self.laya}))
        stack.enter_context(patch.dict(os.environ))

    def test_load_once_and_classify_options_formats(self):
        model = LocalLaya(self.directory)
        for options in ['billing,other', ['billing', 'other'], {'billing': 'billing', 'other': 'other'}]:
            self.assertEqual(model.classify('Message', options), self.answer)
            state, questions = self.agent.predict.call_args.args
            self.assertEqual(state, {'message': 'Message'})
            self.assertEqual(questions['bucket']['criteria'], {'billing': 'billing', 'other': 'other'})
        self.laya.load.assert_called_once_with(str(self.directory.resolve()), device='cpu', fast=False, compile=False)
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')

    def test_missing_checkpoint_cannot_use_hub(self):
        with self.assertRaises(FileNotFoundError):
            LocalLaya(self.directory / 'missing')
        (self.directory / 'encoder/config.json').unlink()
        with self.assertRaises(FileNotFoundError):
            LocalLaya(self.directory)
        self.laya.load.assert_not_called()

    def test_invalid_requests_do_not_predict(self):
        model = LocalLaya(self.directory)
        for message, options in [('', ['A', 'B']), (None, ['A', 'B']), ('message', []),
                                 ('message', ['A', 'a']), ('message', None), ('message', [1, 2])]:
            with self.subTest(message=message, options=options), self.assertRaises(ValueError):
                model.classify(message, options)
        self.agent.predict.assert_not_called()

    def test_tokenizer_normalization_is_reverted_on_load_failure(self):
        config = self.directory / 'tokenizer/tokenizer_config.json'
        original = config.read_bytes()
        def fail(*args, **kwargs):
            config.write_bytes(b'SDK normalization')
            raise RuntimeError('failed')
        self.laya.load.side_effect = fail
        with self.assertRaises(RuntimeError):
            LocalLaya(self.directory)
        self.assertEqual(config.read_bytes(), original)

    def test_cuda_unavailable_fails_before_loading(self):
        with self.assertRaises(ValueError):
            LocalLaya(self.directory, device='cuda')
        self.laya.load.assert_not_called()

    def test_advanced_predict_returns_complete_sdk_result(self):
        model = LocalLaya(self.directory)
        questions = {'urgent': {'type': 'noul', 'instructions': 'Is this urgent?'}}
        result = model.predict('message', questions, max_len=512)
        self.assertEqual(result, self.agent.predict.return_value)
        self.agent.predict.assert_called_once_with('message', questions, max_len=512)

    def test_description_mapping_preserves_delimiters_in_description(self):
        question = options_questions({'A': 'billing, invoices: refunds = payments', 'B': 'other'})
        self.assertEqual(question['bucket']['criteria']['A'], 'billing, invoices: refunds = payments')
        with self.assertRaises(ValueError):
            options_questions({'A': 123, 'B': 'other'})


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.classify.return_value = {'choice': 'billing', 'probabilities': {'billing': 0.8, 'other': 0.2}}
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.model))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, body=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_and_repeated_classification(self):
        self.assertEqual(self.request('GET', '/health'), (200, {'ready': True}))
        for message in ['First', 'Second']:
            status, result = self.request('POST', '/classify', json.dumps({'message': message, 'options': ['billing', 'other']}))
            self.assertEqual(status, 200)
            self.assertEqual(result, self.model.classify.return_value)
            self.model.classify.assert_called_with(message, ['billing', 'other'])

    def test_bad_json_and_invalid_options(self):
        for body in ['{invalid', '[]', '']:
            with self.subTest(body=body):
                self.assertEqual(self.request('POST', '/classify', body)[0], 400)
        self.model.classify.assert_not_called()
        self.model.classify.side_effect = ValueError('Provide at least two options.')
        self.assertEqual(self.request('POST', '/classify', '{"message":"x","options":[]}')[0], 400)

    def test_unknown_route_and_inference_failure(self):
        self.assertEqual(self.request('POST', '/unknown', '{}')[0], 404)
        self.model.classify.side_effect = RuntimeError('internal details')
        status, result = self.request('POST', '/classify', '{"message":"x","options":["A","B"]}')
        self.assertEqual(status, 500)
        self.assertNotIn('internal details', result['error'])

    def test_body_size_limit(self):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request('POST', '/classify', headers={'Content-Length': str(1024 * 1024 + 1)})
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            connection.close()
        self.model.classify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
