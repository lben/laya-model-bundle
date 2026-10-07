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
from laya_client import bucket_questions, token_budgets

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


def parse_options(options):
    """Parse the flag's comma-separated list, preserving multiword option names."""
    return bucket_questions(options.split(','))


def read_text_file(filename):
    """Read UTF-8 or BOM-marked UTF-16 files, including PowerShell output."""
    if not filename:
        raise ValueError('A file path is required after @.')
    path = Path(filename).expanduser()
    data = path.read_bytes()
    encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    try:
        return data.decode(encoding)
    except UnicodeError as error:
        raise ValueError(f'Cannot decode {path}. Save the file as UTF-8 or UTF-16 with a BOM.') from error


def resolve_input(value):
    """@path reads a file; @@ escapes a literal leading @."""
    if value.startswith('@@'):
        return value[1:]
    if value.startswith('@'):
        return read_text_file(value[1:])
    return value


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


def limit_probabilities(result, top_n):
    """Limit choice output only, preserving scores and the original SDK result."""
    if top_n is None:
        return result
    answers = {}
    for name, answer in result['answers'].items():
        if 'choice' in answer:
            # Prefer the selected option within ties, so it survives the cutoff.
            ranked = sorted(answer['probabilities'].items(), key=lambda item: (-item[1], item[0] != answer['choice']))
            answers[name] = {**answer, 'probabilities': dict(ranked[:top_n])}
        else:
            answers[name] = answer
    return {**result, 'answers': answers}


def show_result(result, as_json, all_probabilities=False, top_n=None):
    result = limit_probabilities(result, top_n)
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
        if (all_probabilities or top_n is not None) and 'choice' in answer:
            for label, probability in answer['probabilities'].items():
                print(f'  {label}: {probability:.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='*', help='text or @path to a text file; omit for interactive mode')
    parser.add_argument('--text-file', type=Path, help='read the message from a file instead of positional text')
    classification = parser.add_mutually_exclusive_group()
    classification.add_argument('--preset', choices=list(PRESETS), help='preset to use (default: triage)')
    classification.add_argument('--options', help='comma-separated category names or @path to a file; optionally label=description')
    classification.add_argument('--options-file', type=Path, help='read comma-separated category names/descriptions from a file')
    parser.add_argument('--model', choices=['english', 'multilingual', 'typed-decisions'], default='english')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-len', type=int, help='total input token limit (default: checkpoint setting)')
    parser.add_argument('--head-max-len', type=int, help='question/options token budget (default: checkpoint setting)')
    parser.add_argument('--top-n', type=int, help='show only the N highest-probability options per choice answer')
    parser.add_argument('--json', action='store_true', help='print structured results (optionally limited by --top-n)')
    args = parser.parse_args(argv)
    args.preset = args.preset or 'triage'
    try:
        if args.top_n is not None and args.top_n <= 0:
            raise ValueError('top-n must be a positive integer.')
        budgets = token_budgets(args.max_len, args.head_max_len)
        if args.text_file is not None and args.text:
            raise ValueError('Use positional text or --text-file, not both.')
        file_message = args.text_file is not None or bool(args.text and args.text[0].startswith('@') and not args.text[0].startswith('@@'))
        if file_message and len(args.text) > 1:
            raise ValueError('Pass @path as the only message argument, or use --text-file.')
        if args.text_file is not None:
            text = read_text_file(args.text_file)
        elif len(args.text) == 1:
            text = resolve_input(args.text[0])
        else:
            text = ' '.join(args.text).strip()
        if not file_message:
            text = text.strip()
        if file_message and not text.strip():
            raise ValueError('The message file is empty.')
        options = read_text_file(args.options_file) if args.options_file is not None else (
            resolve_input(args.options) if args.options is not None else None
        )
        fixed_questions = parse_options(options) if options is not None else None
        with offline_mode():
            agent, directory = load_agent(args)
            if fixed_questions is None:
                function, state_key = PRESETS[args.preset]
                questions = getattr(importlib.import_module('laya.presets'), function)()

            def predict(text):
                if fixed_questions is not None:
                    show_result(agent.predict({'message': text}, fixed_questions, **budgets), args.json, all_probabilities=True, top_n=args.top_n)
                    return
                custom = parse_bucket_request(text)
                if custom is not None:
                    message, bucket_questions = custom
                    show_result(agent.predict({'message': message}, bucket_questions, **budgets), args.json, all_probabilities=True, top_n=args.top_n)
                else:
                    show_result(agent.predict({state_key: text}, questions, **budgets), args.json, top_n=args.top_n)

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
