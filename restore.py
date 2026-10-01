"""Restore official Laya checkpoints, with no dependencies or network access."""
import argparse
from pathlib import Path
from scripts.bundle import ROOT, check_chunks, load_manifest, restore_model, verify_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all', 'english', 'multilingual', 'typed-decisions'], default='all')
    p.add_argument('--output', type=Path, default=ROOT / 'models')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true', help='Verify restored files without writing')
    mode.add_argument('--check-chunks', action='store_true', help='Verify compressed chunks without restoring')
    a = p.parse_args()
    try:
        manifest = load_manifest()
        models = [m for m in manifest['models'] if a.model == 'all' or m['name'] == a.model]
        if not models:
            raise ValueError(f'Checkpoint absent from manifest: {a.model}')
        for model in models:
            if a.check_chunks:
                check_chunks(model)
                print(f'{model["name"]}: all compressed chunks verified')
            elif a.verify_only:
                verify_model(model, a.output)
                print(f'{model["name"]}: all restored files verified')
            else:
                restore_model(model, a.output)
    except (OSError, ValueError, EOFError) as e:
        p.exit(1, f'ERROR: {e}\n')


if __name__ == '__main__':
    main()
