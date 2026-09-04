from __future__ import annotations

import re


def build_literature_query_variants(query: str) -> list[str]:
    """Return source-execution variants for deterministic literature APIs.

    The original LLM query is preserved in logs. For deterministic literature
    databases such as PubMed/EuropePMC/OpenAlex/Crossref, broad OR expressions
    are executed as separate query variants because one OR string can mix two
    different search intents and dilute ranked results.

    Variant policy:
    - if OR is present: execute each OR-separated segment as its own variant;
      if there are two ORs, this yields three segment variants, etc.
    - when quotes are present in a segment/query: also execute a relaxed
      unquoted variant for that segment/query.
    - also keep the original full query as a later variant for provenance and
      recall, without making it the only route by which records can enter.
    - if OR is absent: execute the raw query, plus a relaxed variant if quoted.
    """
    raw = _collapse_spaces(query or "")
    variants: list[str] = []
    if not raw:
        return variants

    parts = _split_or_segments(raw)
    if len(parts) > 1:
        for part in parts:
            _append_with_relaxed(variants, part)
        _append_with_relaxed(variants, raw)
    else:
        _append_with_relaxed(variants, raw)

    return variants


def _split_or_segments(query: str) -> list[str]:
    # PubMed/EuropePMC/OpenAlex/Crossref adapters use this as a source-specific
    # execution strategy. It does not rewrite the globally stored LLM query.
    if not re.search(r"\bOR\b", query):
        return [query]
    return [_strip_outer_parens(p.strip()) for p in re.split(r"\bOR\b", query) if p.strip()]


def _append_with_relaxed(items: list[str], query: str) -> None:
    query = _collapse_spaces(query)
    _append_unique(items, query)
    relaxed = _relax_quotes(query)
    if relaxed != query:
        _append_unique(items, relaxed)


def _relax_quotes(query: str) -> str:
    q = query.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    q = q.replace('"', "").replace("'", "")
    return _collapse_spaces(q)


def _strip_outer_parens(text: str) -> str:
    t = text.strip()
    while t.startswith("(") and t.endswith(")") and _balanced_parens(t[1:-1]):
        t = t[1:-1].strip()
    return t


def _balanced_parens(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)
