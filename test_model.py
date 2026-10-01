"""Real offline inference smoke test. Exit 0 means load + all three types work."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
from unittest.mock import patch

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key] = '1'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from scripts.bundle import ROOT, load_manifest, verify_model

QUESTIONS = {
    'department': {'type': 'choice', 'instructions': 'Which department should handle this request?',
                   'criteria': {'billing': 'invoices, payments, refunds', 'technical': 'bugs, outages, errors', 'other': 'everything else'}},
    'urgency': {'type': 'score', 'instructions': 'How urgent is this request?', 'criteria': ['not urgent', 'soon', 'blocking']},
    'refund': {'type': 'noul', 'instructions': 'Does the customer explicitly request a refund?'}
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value):
    require(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1, f'Invalid probability: {value}')


def validate(result):
    answers = result['answers']
    require(set(answers) == set(QUESTIONS), 'Missing answers')
    for name, question in QUESTIONS.items():
        answer = answers[name]
        require(answer['type'] == question['type'], f'Wrong answer type: {name}')
        probability(answer['confidence'])
        if name == 'refund':
            probability(answer['noul'])
        else:
            probs = answer['probabilities']
            require(bool(probs), 'Empty distribution')
            for value in probs.values():
                probability(value)
            require(abs(sum(probs.values()) - 1) < 0.002, 'Distribution does not sum to 1')
    require(set(answers['department']['probabilities']) == set(QUESTIONS['department']['criteria']), 'Wrong choice labels')
    require(answers['department']['choice'] in QUESTIONS['department']['criteria'], 'Invalid choice')
    require(set(answers['urgency']['probabilities']) == {'0', '1', '2'}, 'Wrong score levels')
    score = answers['urgency']['score']
    require(math.isfinite(score) and 0 <= score <= 2, 'Score outside rubric')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['english', 'multilingual', 'typed-decisions'], default='english')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--report', type=Path)
    a = parser.parse_args()
    model = next(m for m in load_manifest()['models'] if m['name'] == a.model)
    verify_model(model, a.models_dir)
    import torch
    import laya
    if a.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable; install a CUDA PyTorch wheel and check your NVIDIA driver')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    directory = (a.models_dir / model['directory']).resolve()
    config = directory / 'tokenizer/tokenizer_config.json'
    original = config.read_bytes()
    states = ['Hi, we were billed twice for March. Please refund the duplicate today.']
    if a.model == 'multilingual':
        states.append('Me cobraron dos veces. Por favor, devuelvan el cargo duplicado.')
    results = []
    start = time.perf_counter()
    try:
        # Offline flags plus a socket guard ensure a missing model component cannot download silently.
        with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled during inference test')), patch.object(socket, 'create_connection', side_effect=RuntimeError('Network disabled during inference test')):
            agent = laya.load(str(directory), device=a.device, fast=False, compile=False)
            require(agent.device.type == a.device, 'Runtime fell back to another device')
            load_seconds = time.perf_counter() - start
            for state in states:
                started = time.perf_counter()
                result = agent.predict(state, QUESTIONS)
                validate(result)
                results.append({'state': state, 'seconds': time.perf_counter() - started, 'result': result})
    finally:
        # Laya fixes tokenizer compatibility in-place. Preserve the exact restored snapshot.
        if config.read_bytes() != original:
            config.write_bytes(original)
    report = {'passed': True, 'model': a.model, 'device': a.device, 'platform': platform.platform(),
              'python': platform.python_version(), 'torch': torch.__version__, 'laya': laya.__version__,
              'source': load_manifest()['source'], 'load_seconds': load_seconds, 'tests': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {a.model} loaded offline and returned valid choice, score, and noul answers.')


if __name__ == '__main__':
    main()
