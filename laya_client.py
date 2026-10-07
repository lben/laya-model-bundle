"""Small importable API for a restored, local Laya checkpoint."""
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
from threading import Lock
from laya_chunking import chunk_settings, predict_text


def bucket_questions(pieces):
    """Validate category names/descriptions and build an SDK choice question."""
    criteria, seen = {}, set()
    for piece in pieces:
        if not isinstance(piece, str):
            raise ValueError('Option names and descriptions must be strings.')
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


def options_questions(options):
    """Accept a comma list, a sequence of names, or a label-to-description dict."""
    if isinstance(options, str):
        return bucket_questions(options.split(','))
    if isinstance(options, Mapping):
        pieces = []
        for label, description in options.items():
            if not isinstance(label, str) or not isinstance(description, str):
                raise ValueError('Option names and descriptions must be strings.')
            if '=' in label or ':' in label:
                raise ValueError('Mapping labels must not contain = or :.')
            pieces.append(f'{label}={description}')
        return bucket_questions(pieces)
    if isinstance(options, Sequence) and not isinstance(options, (bytes, bytearray)):
        return bucket_questions(options)
    raise ValueError('Options must be a comma-separated string, a list of names, or a description dictionary.')


def token_budgets(max_len=None, head_max_len=None):
    """Validate optional per-call token limits without changing checkpoint defaults."""
    budgets = {}
    for name, value in [('max_len', max_len), ('head_max_len', head_max_len)]:
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
            budgets[name] = value
    if max_len is not None and head_max_len is not None and head_max_len >= max_len:
        raise ValueError('head_max_len must be smaller than max_len to leave room for the message.')
    return budgets


_LOAD_LOCK = Lock()


class LocalLaya:
    """Load once, then classify repeatedly. No model download or server required.

    model_path is the checkpoint directory (for example C:/models/laya), not
    the bundle root. The caller owns dependency installation and restoration.
    Predictions on one instance are serialized for safe sharing by API handlers.
    """

    def __init__(self, model_path, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or cuda.')
        self.model_path = Path(model_path).expanduser().resolve()
        required = ['model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
                    'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']
        for name in required:
            if not (self.model_path / name).is_file():
                raise FileNotFoundError(f'Local checkpoint file is missing: {self.model_path / name}. Restore the model first.')
        # Set flags before importing the ML runtime; never pass a Hub identifier.
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
            os.environ[key] = '1'
        os.environ['USE_TF'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        with _LOAD_LOCK:
            import torch
            import laya
            if device == 'cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA is unavailable. Use device="cpu" or a CUDA-enabled PyTorch build.')
            config = self.model_path / 'tokenizer/tokenizer_config.json'
            original = config.read_bytes()
            try:
                self._agent = laya.load(str(self.model_path), device=device, fast=False, compile=False)
            finally:
                if config.read_bytes() != original:
                    config.write_bytes(original)
            if self._agent.device.type != device:
                raise RuntimeError(f'Laya could not use the requested device: {device}')
        self._prediction_lock = Lock()

    def classify(self, message, options, *, max_len=None, head_max_len=None,
                 chunking='auto', chunk_tokens=None, chunk_overlap=None):
        """Return a dict with choice, probabilities, and the SDK's confidence fields."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError('Message must be a nonempty string.')
        questions = options_questions(options)
        budgets = token_budgets(max_len, head_max_len)
        chunk_settings(chunking, chunk_tokens, chunk_overlap)
        with self._prediction_lock:
            result = predict_text(self._agent, message, questions, budgets=budgets,
                                  chunking=chunking, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
        answer = dict(result['answers']['bucket'])
        if 'chunking' in result:
            answer['chunking'] = result['chunking']
            answer['usage'] = result['usage']
        return answer

    def predict(self, state, questions, **kwargs):
        """Advanced interface: return the complete SDK result for typed questions."""
        with self._prediction_lock:
            return self._agent.predict(state, questions, **kwargs)
