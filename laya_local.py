"""Offline prompt and interactive CLI for the restored Laya checkpoints."""
import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import re
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

BUCKET_REQUEST = re.compile(
    r'^from\s+the\s+following\s+options\s*:?\s*(?P<options>.*?)\s+'
    r'(?:categorize|classify)\s+the\s+following\s+message'
    r'(?:\s+on\s+each\s+one)?\s*:\s*(?P<message>.*)$',
    re.IGNORECASE | re.DOTALL,
)


def bucket_questions(pieces):
    """Validate category names/descriptions and build an SDK choice question."""
    criteria, seen = {}, set()
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            raise ValueError('Each option needs a name; empty options are not allowed.')
        pair = re.split(r'\s*[=:]\s*', piece, maxsplit=1)
        label = pair[0].strip()
        description = pair[1].strip() if len(pair) == 2 else label
        if not label or not description:
            raise ValueError('Use a name or a label with a description, such as A=billing.')
        if label.casefold() in seen:
            raise ValueError(f'Duplicate option: {label}')
        seen.add(label.casefold())
        criteria[label] = description
    if len(criteria) < 2:
        raise ValueError('Provide at least two options.')
    return {'bucket': {'type': 'choice', 'instructions': 'Which option best categorizes `message`?', 'criteria': criteria}}


def parse_options(options):
    """Parse the flag's comma-separated list, preserving multiword option names."""
    return bucket_questions(options.split(','))


def parse_bucket_request(text):
    """Convert the supported plain-text instruction into an SDK choice question."""
    text = text.strip()
    if not re.match(r'^from\s+the\s+following\s+options\b', text, re.IGNORECASE):
        return None
    match = BUCKET_REQUEST.fullmatch(text)
    if not match:
        raise ValueError('Use: From the following options A, B, C and D categorize the following message on each one: "message"')
    options = match['options'].strip().rstrip(',;|').strip()
    pieces = re.split(r'[,;|]', options) if re.search(r'[,;|]', options) else options.split()
    # A natural comma list can finish with "sales and other" rather than another comma.
    # Explicit |/semicolon separators and label descriptions preserve their own wording.
    if ',' in options and not re.search(r'[;|]', options) and not re.search(r'[=:]', pieces[-1]):
        pieces[-1:] = re.split(r'\s+and\s+', pieces[-1].strip(), flags=re.IGNORECASE)
    questions = bucket_questions([
        re.sub(r'^and\s+', '', piece.strip(), flags=re.IGNORECASE)
        for piece in pieces if piece.strip().lower() != 'and'
    ])
    message = match['message'].strip()
    for opening, closing in [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’')]:
        if len(message) >= 2 and message.startswith(opening) and message.endswith(closing):
            message = message[1:-1].strip()
            break
    if not message:
        raise ValueError('The message to categorize is empty.')
    return message, questions


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


def show_result(result, as_json, all_probabilities=False):
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
        if all_probabilities and 'choice' in answer:
            for label, probability in answer['probabilities'].items():
                print(f'  {label}: {probability:.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='*', help='text to classify; omit for interactive mode')
    classification = parser.add_mutually_exclusive_group()
    classification.add_argument('--preset', choices=list(PRESETS), help='preset to use (default: triage)')
    classification.add_argument('--options', help='comma-separated category names, optionally label=description; omit text for interactive mode')
    parser.add_argument('--model', choices=['english', 'multilingual', 'typed-decisions'], default='english')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--json', action='store_true', help='print complete structured results')
    args = parser.parse_args(argv)
    args.preset = args.preset or 'triage'
    try:
        fixed_questions = parse_options(args.options) if args.options is not None else None
        with offline_mode():
            agent, directory = load_agent(args)
            if fixed_questions is None:
                function, state_key = PRESETS[args.preset]
                questions = getattr(importlib.import_module('laya.presets'), function)()

            def predict(text):
                if fixed_questions is not None:
                    show_result(agent.predict({'message': text}, fixed_questions), args.json, all_probabilities=True)
                    return
                custom = parse_bucket_request(text)
                if custom is not None:
                    message, bucket_questions = custom
                    show_result(agent.predict({'message': message}, bucket_questions), args.json, all_probabilities=True)
                else:
                    show_result(agent.predict({state_key: text}, questions), args.json)

            text = ' '.join(args.text).strip()
            if text:
                predict(text)
            else:
                mode = f'Options: {", ".join(fixed_questions["bucket"]["criteria"])}' if fixed_questions is not None else f'Preset: {args.preset}'
                print(f'Local model: {directory}\n{mode}; device: {args.device}')
                print('Enter text to classify. Type quit or exit to finish.')
                if fixed_questions is None:
                    print('Custom buckets: From the following options A, B, C and D categorize the following message on each one: "message"')
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
