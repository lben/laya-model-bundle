import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from scripts.bundle import check_chunks, digest_file, load_manifest, pack_stream, restore_model, safe_path, verify_model


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Many chunk boundaries, including a tiny empty gzip member.
        source = bytes(range(256)) * 200 + b'original Hugging Face bytes'
        files = []
        for i, (path, data) in enumerate([('model.safetensors', source), ('encoder/config.json', b'{"offline":true}'), ('empty', b'')]):
            packed = pack_stream(io.BytesIO(data), self.root / 'chunks', f'file{i}', limit=79)
            for part in packed['parts']:
                part['path'] = Path(part['path']).relative_to(self.root).as_posix()
            packed['path'] = path
            files.append(packed)
        self.model = {'name': 'english', 'directory': 'laya', 'files': files}
        self.manifest = {'format_version': 1, 'compression': 'gzip', 'models': [self.model]}
        self.write_manifest()
        self.output = self.root / 'models'

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_roundtrip_and_idempotency(self):
        load_manifest(self.root)
        restore_model(self.model, self.output, self.root)
        verify_model(self.model, self.output)
        restore_model(self.model, self.output, self.root)
        self.assertEqual((self.output / 'laya/encoder/config.json').read_bytes(), b'{"offline":true}')

    def test_missing_chunk(self):
        (self.root / self.model['files'][0]['parts'][1]['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            restore_model(self.model, self.output, self.root)
        self.assertFalse((self.output / 'laya').exists())

    def test_corrupt_chunk(self):
        (self.root / self.model['files'][0]['parts'][0]['path']).write_bytes(b'broken')
        with self.assertRaises(ValueError):
            check_chunks(self.model, self.root)

    def test_wrong_original_hash_cleans_stage(self):
        self.model['files'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            restore_model(self.model, self.output, self.root)
        self.assertFalse((self.output / 'laya').exists())
        self.assertEqual(list(self.output.glob('.laya-restore-*')), [])

    def test_existing_modified_file_is_preserved(self):
        restore_model(self.model, self.output, self.root)
        path = self.output / 'laya/model.safetensors'
        path.write_bytes(b'user changes')
        with self.assertRaises(ValueError):
            restore_model(self.model, self.output, self.root)
        self.assertEqual(path.read_bytes(), b'user changes')

    def test_unsafe_paths(self):
        for path in ['../escape', '/absolute', 'C:/windows', 'a\\b', 'a/../b', 'a//b', './a']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_path(self.root, path)

    def test_manifest_traversal_and_duplicate(self):
        self.model['files'][0]['path'] = '../escape'
        self.write_manifest()
        with self.assertRaises(ValueError):
            load_manifest(self.root)
        self.model['files'][0]['path'] = 'empty'
        self.write_manifest()
        with self.assertRaises(ValueError):
            load_manifest(self.root)

    def test_upstream_validation(self):
        with self.assertRaises(ValueError):
            pack_stream(io.BytesIO(b'test'), self.root / 'other', 'test', expected_size=5)
        with self.assertRaises(ValueError):
            pack_stream(io.BytesIO(b'test'), self.root / 'more', 'test', expected_size=4, expected_sha256='0' * 64)

    def test_decompression_limit(self):
        self.model['files'][0]['size'] = 2
        with self.assertRaises(ValueError):
            restore_model(self.model, self.output, self.root)
        self.assertFalse((self.output / 'laya').exists())


if __name__ == '__main__':
    unittest.main()
