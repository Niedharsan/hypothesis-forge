from __future__ import annotations

from dataclasses import dataclass

from retrieval.crossref_api import CrossrefAPI
from retrieval.europepmc_api import EuropePMCAPI
from retrieval.openalex_api import OpenAlexAPI
from retrieval.pubmed_api import PubMedAPI
from retrieval.semantic_scholar_api import SemanticScholarAPI
from utils.config import load_config


@dataclass
class LiteratureAdapters:
    pubmed: PubMedAPI
    europepmc: EuropePMCAPI
    openalex: OpenAlexAPI
    crossref: CrossrefAPI
    semantic_scholar: SemanticScholarAPI


def build_adapters(config_path: str = "configs/config.yaml") -> LiteratureAdapters:
    return LiteratureAdapters(
        pubmed=PubMedAPI(),
        europepmc=EuropePMCAPI(),
        openalex=OpenAlexAPI(),
        crossref=CrossrefAPI(),
        semantic_scholar=SemanticScholarAPI(
            allow_unauthenticated=_semantic_scholar_allow_unauthenticated(config_path)
        ),
    )


def _semantic_scholar_allow_unauthenticated(config_path: str) -> bool:
    try:
        config = load_config(config_path)
    except Exception:
        return False
    return bool(
        (((config.get("retrieval") or {}).get("semantic_scholar") or {}).get("allow_unauthenticated"))
    )
