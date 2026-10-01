"""Maintainer: stream a pinned HF snapshot into gzip chunks (no full downloads)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bundle import ROOT, pack_stream

REPO = 'convaiinnovations/laya'
GROUPS = {'english': ('', 'laya'), 'multilingual': ('multilingual/', 'laya-multilingual'), 'typed-decisions': ('typed-decisions/', 'laya-typed-decisions')}


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(GROUPS), required=True)
    p.add_argument('--git-index', action='store_true', help='Store chunks directly in Git index and delete working copies to save space')
    a = p.parse_args()
    os.chdir(ROOT)
    lock_path = ROOT / 'source-lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        revision = fetch_json(f'https://huggingface.co/api/models/{REPO}')['sha']
        files = []
        url = f'https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=true&expand=false&limit=1000'
        while url:
            with urllib.request.urlopen(url, timeout=120) as response:
                files.extend(x for x in json.load(response) if x['type'] == 'file')
                link = response.headers.get('Link', '')
            url = next((part.split(';')[0].strip().strip('<>') for part in link.split(',') if 'rel="next"' in part), None)
        lock = {'repository': REPO, 'revision': revision, 'files': files}
        lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    manifest_path = ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'format_version': 1, 'compression': 'gzip', 'chunk_limit_bytes': 45 * 1024 * 1024,
        'source': {'repository': REPO, 'revision': lock['revision'], 'license': 'Apache-2.0'}, 'models': []}
    if any(m['name'] == a.model for m in manifest['models']):
        p.error('Model already packaged; use a clean staging directory for an update')
    prefix, directory = GROUPS[a.model]
    selected = [f for f in lock['files'] if (f['path'].startswith(prefix) if prefix else not f['path'].startswith(('multilingual/', 'typed-decisions/')))]
    model = {'name': a.model, 'directory': directory, 'source_subfolder': prefix.rstrip('/'), 'files': []}

    def sink(path):
        oid = subprocess.check_output(['git', 'hash-object', '-w', str(path)], text=True).strip()
        subprocess.run(['git', 'update-index', '--add', '--cacheinfo', '100644', oid, path.as_posix()], check=True)
        path.unlink()

    for index, f in enumerate(sorted(selected, key=lambda f: f['path'])):
        url = f'https://huggingface.co/{REPO}/resolve/{lock["revision"]}/{f["path"]}'
        print(f'{a.model}: {index + 1}/{len(selected)} {f["path"]} ({f["size"]:,} bytes)', flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'laya-model-bundle/1.0'})
        with urllib.request.urlopen(req, timeout=300) as stream:
            entry = pack_stream(stream, f'chunks/{a.model}', f'file{index:03d}', sink=sink if a.git_index else None,
                                expected_size=f['size'], expected_sha256=f.get('lfs', {}).get('oid'),
                                expected_git_oid=None if f.get('lfs') else f['oid'])
        entry['path'] = f['path'][len(prefix):]
        entry['source_path'] = f['path']
        model['files'].append(entry)
    manifest['models'].append(model)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Packaged {a.model}: {sum(p["size"] for f in model["files"] for p in f["parts"]):,} compressed bytes', flush=True)


if __name__ == '__main__':
    main()
