"""Offline prompt and interactive CLI for the restored Laya checkpoints."""
import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

from scripts.bundle import ROOT, load_manifest, safe_path, verify_model

# Each upstream preset asks about a specific field in the input state.
PRESETS = {
    'triage': ('triage_questions', 'message'),
    'email': ('email_questions', 'body'),
    'guard': ('guard_questions', 'prompt'),
    'moderation': ('moderation_questions', 'post'),
    'router': ('router_questions', 'request'),
}


def offline_mode():
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['USE_TF'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    guard = ExitStack()
    message = 'Network access is disabled. This command uses restored local models only.'
    for target, attribute in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'), (socket, 'create_connection')):
        guard.enter_context(patch.object(target, attribute, side_effect=OSError(message)))
    return guard


def load_agent(args):
    model = next((m for m in load_manifest()['models'] if m['name'] == args.model), None)
    if model is None:
        raise ValueError(f'Checkpoint absent from manifest: {args.model}')
    directory = safe_path(args.models_dir, model['directory']).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f'Local checkpoint is missing: {directory}\n'
            f'Restore it with .\\.venv\\Scripts\\python.exe restore.py --model {args.model}'
        )
    verify_model(model, args.models_dir)
    import torch
    import laya
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.')
    config = directory / 'tokenizer/tokenizer_config.json'
    original = config.read_bytes()
    try:
        # Absolute existing path prevents the SDK from interpreting it as a Hub model ID.
        agent = laya.load(str(directory), device=args.device, fast=False, compile=False)
    finally:
        # The SDK normalizes tokenizer compatibility in place; keep the snapshot unchanged.
        if config.read_bytes() != original:
            config.write_bytes(original)
    if agent.device.type != args.device:
        raise RuntimeError(f'Laya could not use the requested device: {args.device}')
    return agent, directory


def show_result(result, as_json):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for name, answer in result['answers'].items():
        if 'choice' in answer:
            value = answer['choice']
            detail = f'{value} (probability {answer["probabilities"][value]:.3f})'
        elif 'score' in answer:
            detail = f'{answer["score"]:.3f}'
        else:
            detail = f'yes probability {answer["noul"]:.3f}'
        print(f'{name}: {detail}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='*', help='text to classify; omit for interactive mode')
    parser.add_argument('--preset', choices=list(PRESETS), default='triage')
    parser.add_argument('--model', choices=['english', 'multilingual', 'typed-decisions'], default='english')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--json', action='store_true', help='print complete structured results')
    args = parser.parse_args(argv)
    try:
        with offline_mode():
            agent, directory = load_agent(args)
            function, state_key = PRESETS[args.preset]
            questions = getattr(importlib.import_module('laya.presets'), function)()

            def predict(text):
                show_result(agent.predict({state_key: text}, questions), args.json)

            text = ' '.join(args.text).strip()
            if text:
                predict(text)
            else:
                print(f'Local model: {directory}\nPreset: {args.preset}; device: {args.device}')
                print('Enter text to classify. Type quit or exit to finish.')
                while True:
                    try:
                        text = input('laya> ').strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if text.lower() in ('quit', 'exit'):
                        break
                    if text:
                        try:
                            predict(text)
                        except (OSError, RuntimeError, ValueError) as error:
                            print(f'ERROR: {error}', file=sys.stderr)
        return 0
    except ImportError as error:
        print(f'ERROR: Missing runtime dependency ({error}). Run test_windows.ps1 -Install using your configured pip registry.', file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
