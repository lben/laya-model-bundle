"""Lossless, bounded-memory gzip chunking and verified offline restoration."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

BLOCK = 1024 * 1024
CHUNK_SIZE = 45 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(base, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('..', '.', '') for x in name.split('/')) or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name!r}')
    result = Path(base).joinpath(*p.parts)
    if not result.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError(f'Path escapes destination: {name!r}')
    return result


class ChunkWriter:
    def __init__(self, directory, prefix, limit=CHUNK_SIZE, sink=None):
        if not 0 < limit < 100 * 1024 * 1024:
            raise ValueError('Chunk size must be positive and below 100 MiB')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix, self.limit, self.sink = prefix, limit, sink
        self.parts, self.file, self.size, self.hash = [], None, 0, None

    def write(self, data):
        total = len(data)
        while data:
            if self.file is None:
                self.path = self.directory / f'{self.prefix}.gz.part{len(self.parts):04d}'
                if self.path.exists():
                    raise FileExistsError(self.path)
                self.file = self.path.open('wb')
                self.hash = hashlib.sha256()
                self.size = 0
            block = data[:self.limit - self.size]
            self.file.write(block)
            self.hash.update(block)
            self.size += len(block)
            data = data[len(block):]
            if self.size == self.limit:
                self.finish_part()
        return total

    def flush(self):
        if self.file:
            self.file.flush()

    def finish_part(self):
        if self.file:
            self.file.close()
            self.file = None
            part = {'path': self.path.as_posix(), 'size': self.size, 'sha256': self.hash.hexdigest()}
            if self.sink:
                self.sink(self.path)
            self.parts.append(part)

    def close(self):
        self.finish_part()


def pack_stream(stream, directory, prefix, limit=CHUNK_SIZE, sink=None, expected_size=None, expected_sha256=None, expected_git_oid=None):
    writer = ChunkWriter(directory, prefix, limit, sink)
    sha = hashlib.sha256()
    git_sha = hashlib.sha1()
    if expected_size is not None:
        git_sha.update(f'blob {expected_size}\0'.encode())
    size = 0
    try:
        with gzip.GzipFile(filename='', mode='wb', fileobj=writer, compresslevel=6, mtime=0) as compressor:
            for block in iter(lambda: stream.read(BLOCK), b''):
                size += len(block)
                sha.update(block)
                git_sha.update(block)
                compressor.write(block)
    finally:
        writer.close()
    if expected_size is not None and expected_size != size:
        raise ValueError(f'Upstream size mismatch for {prefix}: {size} != {expected_size}')
    if expected_sha256 and expected_sha256 != sha.hexdigest():
        raise ValueError(f'Upstream SHA-256 mismatch for {prefix}')
    if expected_git_oid and expected_git_oid != git_sha.hexdigest():
        raise ValueError(f'Upstream Git blob mismatch for {prefix}')
    return {'size': size, 'sha256': sha.hexdigest(), 'parts': writer.parts}


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__()
        self.paths, self.current = iter(paths), None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def load_manifest(root=ROOT):
    manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or manifest.get('compression') != 'gzip':
        raise ValueError('Unsupported bundle format')
    if not manifest.get('models'):
        raise ValueError('Manifest contains no models')
    names, dirs, all_parts = set(), set(), set()
    for model in manifest['models']:
        if model['name'] in names or model['directory'] in dirs or not model.get('files'):
            raise ValueError('Duplicate or empty model')
        names.add(model['name'])
        dirs.add(model['directory'])
        safe_path(root, model['directory'])
        paths = set()
        for entry in model['files']:
            safe_path(root, entry['path'])
            if entry['path'] in paths or entry['size'] < 0 or not entry.get('parts'):
                raise ValueError('Duplicate or invalid file entry')
            paths.add(entry['path'])
            for sha in [entry['sha256']] + [p['sha256'] for p in entry['parts']]:
                if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('Invalid SHA-256')
            for part in entry['parts']:
                safe_path(root, part['path'])
                if not part['path'].startswith('chunks/') or part['path'] in all_parts or not 0 < part['size'] < 100 * 1024 * 1024:
                    raise ValueError('Invalid or duplicate chunk')
                all_parts.add(part['path'])
    return manifest


def check_chunks(model, root=ROOT):
    for entry in model['files']:
        for part in entry['parts']:
            p = safe_path(root, part['path'])
            if not p.is_file():
                raise FileNotFoundError(f'Missing chunk: {p}. Clone the complete repository.')
            if p.stat().st_size != part['size'] or digest_file(p) != part['sha256']:
                raise ValueError(f'Corrupt chunk: {p}')


def verify_model(model, destination):
    directory = safe_path(destination, model['directory'])
    for entry in model['files']:
        p = safe_path(directory, entry['path'])
        if not p.is_file() or p.stat().st_size != entry['size'] or digest_file(p) != entry['sha256']:
            raise ValueError(f'Missing or changed restored file: {p}')


def restore_model(model, destination, root=ROOT):
    destination = Path(destination)
    directory = safe_path(destination, model['directory'])
    check_chunks(model, root)
    if directory.exists():
        verify_model(model, destination)
        print(f'{model["name"]}: already restored and verified', flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    required = sum(f['size'] for f in model['files']) + 64 * BLOCK
    if shutil.disk_usage(destination).free < required:
        raise OSError(f'Insufficient disk space: need at least {required / 1024**3:.2f} GiB free')
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.laya-restore-', dir=directory.parent))
    try:
        for entry in model['files']:
            target = safe_path(stage, entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            h, size = hashlib.sha256(), 0
            with PartsReader([safe_path(root, p['path']) for p in entry['parts']]) as raw:
                with io.BufferedReader(raw) as buffered, gzip.GzipFile(fileobj=buffered) as decoder, target.open('wb') as out:
                    for block in iter(lambda: decoder.read(BLOCK), b''):
                        size += len(block)
                        if size > entry['size']:
                            raise ValueError(f'Decompressed size exceeds manifest: {entry["path"]}')
                        h.update(block)
                        out.write(block)
            if size != entry['size'] or h.hexdigest() != entry['sha256']:
                raise ValueError(f'Restored checksum mismatch: {entry["path"]}')
        stage.rename(directory)
        print(f'{model["name"]}: restored and verified -> {directory}', flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
