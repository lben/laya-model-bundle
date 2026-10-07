"""Token-aware local text scanning with every question/option kept in each chunk."""
from dataclasses import dataclass


@dataclass
class TextPlan:
    token_ids: list
    state_tokens: int
    room: int
    truncated: bool
    max_len: int
    head_max_len: int


def chunk_settings(chunking='auto', chunk_tokens=None, chunk_overlap=None):
    if chunking not in ('auto', 'on', 'off'):
        raise ValueError('chunking must be auto, on, or off.')
    for name, value, minimum in [('chunk_tokens', chunk_tokens, 1), ('chunk_overlap', chunk_overlap, 0)]:
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < minimum):
            raise ValueError(f'{name} must be an integer of at least {minimum}.')
    if chunk_tokens is not None and chunk_overlap is not None and chunk_overlap >= chunk_tokens:
        raise ValueError('chunk_overlap must be smaller than chunk_tokens.')


def analyze_text(agent, text, questions, state_key, budgets):
    # Use the pinned SDK's own renderer, so option overhead and tokenizer behavior
    # are measured rather than guessed from characters or head_max_len alone.
    from laya.common import build_sequence, encode_text, render_options, serialize_state
    maximum = getattr(agent.model.encoder.config, 'max_position_embeddings', 8192)
    max_len = budgets.get('max_len', agent.cfg.get('max_len', 512))
    head = budgets.get('head_max_len', agent.cfg.get('head_max_len', 192))
    if max_len > maximum:
        raise ValueError(f'max_len={max_len} exceeds this encoder\'s {maximum}-token limit.')
    tok = agent.tok
    # Full tokenization is needed to scan/count a long document. verbose=False avoids
    # the misleading tokenizer warning; every actual model row is checked below.
    def encode(value):
        return encode_text(tok, value.replace(tok.mask_token, ' '), add_special_tokens=False, verbose=False)['input_ids']
    message_ids = encode(text)
    state_ids = encode(serialize_state({state_key: text}))
    overhead = len(encode(serialize_state({state_key: ''})))
    room, truncated = max_len, False
    for name, question in questions.items():
        internal = agent._to_internal(question)
        seq, markers, stats = build_sequence(tok, {state_key: text}, internal, max_len, head,
                                            state_ids=state_ids, return_truncation_stats=True)
        count = len(render_options(internal))
        if len(markers) != count:
            raise ValueError(f'question {name!r}: only {len(markers)} of its {count} option markers fit in max_len={max_len}. '
                             'Raise --max-len and --head-max-len; message chunking keeps every option.')
        truncated = truncated or stats['truncated']
        # On a fitting state stats only reports its length, so calculate head room
        # separately with an empty state and no serialization tokens.
        empty, _ = build_sequence(tok, '', internal, max_len, head, state_ids=[])
        room = min(room, max_len - len(empty) - overhead)
    return TextPlan(message_ids, len(state_ids), room, truncated, max_len, head)


def aggregate_chunks(results, spans, plan, mode, overlap):
    answers = {}
    for name in results[0]['answers']:
        def strength(index):
            answer = results[index]['answers'][name]
            if 'noul' in answer:
                return answer['noul']
            return answer.get('answer_confidence', max(answer.get('probabilities', {}).values(), default=answer.get('confidence', 0)))
        best = max(range(len(results)), key=strength)
        start, end = spans[best]
        answers[name] = {**results[best]['answers'][name],
                         'window': {'index': best, 'token_start': start, 'token_end': end, 'count': len(results)}}
    usage = {}
    for result in results:
        for key, value in result.get('usage', {}).items():
            if key in ('state_tokens', 'state_tokens_dropped', 'truncated', 'truncated_questions'):
                continue
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            elif isinstance(value, dict):
                usage[key] = {**usage.get(key, {}), **value}
            else:
                usage[key] = value
    usage.update(state_tokens=plan.state_tokens, state_tokens_dropped=0, truncated=False,
                 truncated_questions=[], windows=len(results))
    return {**results[0], 'answers': answers, 'usage': usage,
            'chunking': {'mode': mode, 'chunks': len(results), 'message_tokens': len(plan.token_ids),
                         'covered_tokens': len(plan.token_ids), 'overlap_tokens': overlap,
                         'aggregation': 'strongest-window', 'max_len': plan.max_len, 'head_max_len': plan.head_max_len}}


def predict_text(agent, text, questions, *, state_key='message', budgets=None,
                 chunking='auto', chunk_tokens=None, chunk_overlap=None, progress=None):
    chunk_settings(chunking, chunk_tokens, chunk_overlap)
    budgets = budgets or {}
    plan = analyze_text(agent, text, questions, state_key, budgets)
    needs_scan = plan.truncated or (chunk_tokens is not None and len(plan.token_ids) > chunk_tokens)
    if chunking == 'off' or (chunking == 'auto' and not needs_scan):
        result = agent.predict({state_key: text}, questions, **budgets)
        if chunking == 'off' and plan.truncated:
            result = {**result, 'chunking': {'mode': 'off', 'chunks': 1, 'message_tokens': len(plan.token_ids),
                                           'warning': 'Only the first part of the message was evaluated.'}}
        return result
    # Leave a small margin for BPE decode/re-encode and JSON escaping. Each candidate
    # is still checked with the renderer; if necessary its end shrinks until it fits.
    capacity = plan.room - 16
    size = min(chunk_tokens or capacity, capacity)
    if size < 1:
        raise ValueError('The questions/options leave no usable space for message chunks. Raise --max-len.')
    overlap = chunk_overlap if chunk_overlap is not None else min(64, size // 4)
    if overlap >= size:
        raise ValueError(f'chunk_overlap must be smaller than the effective chunk size ({size}).')
    chunks, spans = [], []
    start, total = 0, len(plan.token_ids)
    while start < total:
        end = min(start + size, total)
        while True:
            decoded = agent.tok.decode(plan.token_ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            candidate = analyze_text(agent, decoded, questions, state_key, budgets)
            if not candidate.truncated:
                break
            end -= max(1, min(end - start - 1, candidate.state_tokens - candidate.room))
            if end <= start:
                raise ValueError('A message chunk cannot fit in the configured token budget.')
        if end < total and end - start <= overlap:
            raise ValueError('The fitted chunk is smaller than the overlap. Lower --chunk-overlap.')
        chunks.append(decoded)
        spans.append((start, end))
        if end == total:
            break
        start = end - overlap
    if not chunks:
        chunks, spans = [text], [(0, 0)]
    results = []
    for index, decoded in enumerate(chunks):
        if progress:
            progress(index + 1, len(chunks))
        result = agent.predict({state_key: decoded}, questions, **budgets)
        if result.get('usage', {}).get('truncated'):
            raise RuntimeError('Laya truncated a planned chunk; no partial document result was returned.')
        results.append(result)
    return aggregate_chunks(results, spans, plan, chunking, overlap)
