#!/usr/bin/env bash
set -euo pipefail

for i in 1 2 3 4 5; do
  python -u scripts/run_single_axis_literature_generation.py \
    --out-dir "runs/v67_1_query_reviewer_family_only_rep${i}" \
    --subtopic-mode v2 \
    --max-axis-query-families 6 \
    --cutoff-year 2023 \
    --stop-after-query-families

done

python - <<'PY'
import json
from pathlib import Path
for d in sorted(Path('runs').glob('v67_1_query_reviewer_family_only_rep*')):
    p = d / '00c_subtopic_generation_context.json'
    if not p.exists():
        print(f'\n{d}: missing query-family context')
        continue
    data = json.loads(p.read_text())
    raw = data.get('raw_query_families_payload', {}).get('query_families', [])
    final = data.get('query_families_payload', {}).get('query_families', [])
    notes = data.get('query_reviewer_payload', {}).get('reviewed_payload', {}).get('review_notes', [])
    print(f'\n=== {d.name} ===')
    print('RAW:')
    for f in raw:
        print(f"  {f.get('family_id')} | {f.get('name') or f.get('family_name')} | {f.get('query')}")
    print('REVIEWED:')
    for f in final:
        print(f"  {f.get('family_id')} | {f.get('name') or f.get('family_name')} | {f.get('query')} | {f.get('revision_reason','')}")
    if notes:
        print('NOTES:')
        for n in notes:
            print(f"  - {n.get('issue')}: {n.get('change_made')}")
PY
