from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.pubtator_api import PubTatorAPI


def main() -> None:
    pmids = sys.argv[1:] or ["37012202", "38908793", "40169588"]
    api = PubTatorAPI(batch_size=5)
    annotations = api.annotate_pmids(pmids)
    payload = {
        "requested_pmids": pmids,
        "annotated_count": len(annotations),
        "annotations": {pmid: ann.compact() for pmid, ann in annotations.items()},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:12000])


if __name__ == "__main__":
    main()
