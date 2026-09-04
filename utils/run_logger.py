from __future__ import annotations

import json, time, uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import contextvars

_current_log_dir: contextvars.ContextVar[str | None] = contextvars.ContextVar('run_log_dir', default=None)
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('run_id', default=None)

SUBDIRS = ['prompts', 'llm_calls', 'retrieval']

def start_run_log(objective: str, mode: str = 'v31_clean', run_root: str | Path = 'runs/logs') -> str:
    rid = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = Path(run_root) / rid
    root.mkdir(parents=True, exist_ok=True)
    for d in SUBDIRS:
        (root / d).mkdir(exist_ok=True)
    (root / 'run.json').write_text(json.dumps({'run_id': rid, 'objective': objective, 'mode': mode}, indent=2), encoding='utf-8')
    _current_run_id.set(rid); _current_log_dir.set(str(root))
    log_event('run', 'start', {'objective': objective, 'mode': mode, 'run_id': rid})
    return rid

def current_log_dir() -> Path | None:
    value = _current_log_dir.get()
    return Path(value) if value else None

def current_run_id() -> str | None:
    return _current_run_id.get()

def log_event(category: str, action: str, payload: dict[str, Any] | None = None, *, status: str = 'ok') -> None:
    root = current_log_dir()
    if not root:
        return
    event = {'ts': datetime.now().isoformat(timespec='seconds'), 'run_id': current_run_id(), 'category': category, 'action': action, 'status': status, 'payload': _safe(payload or {})}
    with (root / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')

def log_gemini_call(*, caller: str, model: str, prompt: str, response_text: str | None = None, usage: dict[str, Any] | None = None, duration_s: float | None = None, error: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    root = current_log_dir()
    safe_metadata = _safe(metadata or {})
    payload = {
        'caller': caller,
        'agent': safe_metadata.get('agent'),
        'purpose': safe_metadata.get('purpose'),
        'stage': safe_metadata.get('stage') or caller,
        'model': model,
        'temperature': safe_metadata.get('temperature'),
        'llm_call_index': safe_metadata.get('llm_call_index'),
        'prompt_sha256': safe_metadata.get('prompt_sha256'),
        'prompt_chars': len(prompt or ''),
        'response_chars': len(response_text or ''),
        'estimated_prompt_tokens': _estimate_tokens(len(prompt or '')),
        'estimated_response_tokens': _estimate_tokens(len(response_text or '')),
        'usage': usage or {},
        'duration_s': round(duration_s, 3) if duration_s is not None else None,
    }
    if safe_metadata:
        payload.update(safe_metadata)
    if error:
        payload['error'] = error
    if root:
        idx = len(list((root / 'llm_calls').glob('*.json'))) + 1
        stem = f'{idx:04d}_{_safe_name(caller)}'
        ppath = root / 'prompts' / f'{stem}.txt'
        ppath.write_text(prompt or '', encoding='utf-8')
        payload['prompt_file'] = str(ppath)
        if response_text is not None:
            rpath = root / 'llm_calls' / f'{stem}.response.txt'
            rpath.write_text(response_text or '', encoding='utf-8')
            payload['response_file'] = str(rpath)
        (root / 'llm_calls' / f'{stem}.json').write_text(json.dumps(_safe(payload), indent=2, ensure_ascii=False), encoding='utf-8')
    log_event('gemini', 'generate_content', payload, status='error' if error else 'ok')

def finalize_run(extra_summary: dict[str, Any] | None = None) -> None:
    root = current_log_dir()
    if not root:
        return
    events = []
    ep = root / 'events.jsonl'
    if ep.exists():
        for line in ep.read_text(encoding='utf-8').splitlines():
            try: events.append(json.loads(line))
            except Exception: pass
    gem = [e for e in events if e.get('category') == 'gemini']
    api = [e for e in events if e.get('category') == 'api']
    usage_summary = collect_llm_usage_summary()
    retrieval = [e for e in events if e.get('category') == 'retrieval']
    cache_events = [e for e in events if (e.get('action') or '').endswith('cache_hit') or (e.get('action') or '').endswith('cache_miss') or e.get('action') in {'cache_hit', 'cache_miss'}]
    summary = {
        'run_id': current_run_id(),
        'gemini_calls': len(gem),
        'gemini_errors': sum(1 for e in gem if e.get('status') == 'error'),
        'api_events': len(api),
        'retrieval_events': len(retrieval),
        'cache_events': len(cache_events),
        'estimated_prompt_tokens': sum((e.get('payload') or {}).get('estimated_prompt_tokens',0) for e in gem),
        'estimated_response_tokens': sum((e.get('payload') or {}).get('estimated_response_tokens',0) for e in gem),
        'llm_usage_totals': usage_summary.get('totals', {}),
        'llm_usage_by_stage': usage_summary.get('by_stage', {}),
        'llm_usage_by_model': usage_summary.get('by_model', {}),
        'top_prompt_calls': usage_summary.get('top_prompt_calls', []),
    }
    try:
        from runtime.context import current_llm_call_count, current_runtime
        rt = current_runtime()
        summary['budget_usage'] = {'llm_calls_used': current_llm_call_count(), 'llm_call_limit': rt.limits.max_llm_calls_per_run, 'mode': rt.mode}
    except Exception:
        pass
    if extra_summary: summary.update(_safe(extra_summary))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log_event('run', 'finalize', summary)

def _estimate_tokens(chars: int) -> int:
    return max(1, round(chars / 4)) if chars else 0

def _safe_name(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)[:80] or 'unknown'

def _safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except Exception:
        if isinstance(x, dict): return {str(k): _safe(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [_safe(v) for v in x]
        return str(x)


def collect_llm_usage_summary() -> dict[str, Any]:
    root = current_log_dir()
    if not root:
        return {"calls": [], "totals": {}, "by_stage": {}, "by_model": {}, "top_prompt_calls": []}
    calls = []
    for path in sorted((root / 'llm_calls').glob('*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        usage = payload.get('usage') or {}
        input_tokens = usage.get('prompt_token_count') or usage.get('input_tokens') or payload.get('estimated_prompt_tokens') or 0
        output_tokens = usage.get('candidates_token_count') or usage.get('output_tokens') or payload.get('estimated_response_tokens') or 0
        total_tokens = usage.get('total_token_count') or (input_tokens + output_tokens)
        call = {
            'caller': payload.get('caller'),
            'agent': payload.get('agent'),
            'purpose': payload.get('purpose'),
            'stage': payload.get('stage') or payload.get('caller'),
            'model': payload.get('model'),
            'temperature': payload.get('temperature'),
            'llm_call_index': payload.get('llm_call_index'),
            'prompt_sha256': payload.get('prompt_sha256'),
            'prompt_chars': payload.get('prompt_chars'),
            'response_chars': payload.get('response_chars'),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'thoughts_token_count': usage.get('thoughts_token_count'),
            'duration_s': payload.get('duration_s'),
            'error': payload.get('error'),
            'prompt_file': payload.get('prompt_file'),
            'response_file': payload.get('response_file'),
            'mock': payload.get('mock'),
        }
        call['estimated_cost_usd'] = _estimate_cost(call['model'], input_tokens, output_tokens)
        calls.append(call)

    totals = _summarize_calls(calls)
    by_stage = _group_call_summary(calls, 'stage')
    by_model = _group_call_summary(calls, 'model')
    top_prompt_calls = sorted(
        calls,
        key=lambda c: int(c.get('input_tokens') or 0),
        reverse=True,
    )[:10]
    compact_top = [
        {
            'stage': c.get('stage'),
            'model': c.get('model'),
            'input_tokens': c.get('input_tokens'),
            'output_tokens': c.get('output_tokens'),
            'estimated_cost_usd': c.get('estimated_cost_usd'),
            'prompt_chars': c.get('prompt_chars'),
            'prompt_file': c.get('prompt_file'),
        }
        for c in top_prompt_calls
    ]
    budget_usage = {}
    try:
        from runtime.context import current_llm_call_count, current_runtime
        rt = current_runtime()
        budget_usage = {'llm_calls_used': current_llm_call_count(), 'llm_call_limit': rt.limits.max_llm_calls_per_run, 'mode': rt.mode}
    except Exception:
        pass
    return {'calls': calls, 'totals': totals, 'by_stage': by_stage, 'by_model': by_model, 'top_prompt_calls': compact_top, 'budget_usage': budget_usage}


def _summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'calls': len(calls),
        'errors': sum(1 for c in calls if c.get('error')),
        'input_tokens': sum(int(c.get('input_tokens') or 0) for c in calls),
        'output_tokens': sum(int(c.get('output_tokens') or 0) for c in calls),
        'total_tokens': sum(int(c.get('total_tokens') or 0) for c in calls),
        'estimated_cost_usd': round(sum(float(c.get('estimated_cost_usd') or 0) for c in calls), 6),
        'duration_s': round(sum(float(c.get('duration_s') or 0) for c in calls), 3),
    }


def _group_call_summary(calls: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[str(call.get(key) or 'unknown')].append(call)
    return {name: _summarize_calls(items) for name, items in sorted(grouped.items())}

def _estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    try:
        from runtime.context import current_runtime
        pricing = current_runtime().pricing or {}
        p = pricing.get(model or '') or {}
        return round((input_tokens / 1_000_000) * float(p.get('input_per_million', 0)) + (output_tokens / 1_000_000) * float(p.get('output_per_million', 0)), 6)
    except Exception:
        return 0.0
