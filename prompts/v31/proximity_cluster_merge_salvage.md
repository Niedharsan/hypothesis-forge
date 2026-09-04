You analyze similarity between hypotheses.

Goal: {{ supervisor_config_json }}

Generated hypotheses to analyze:
{{ hypotheses_json }}

Use the supplied hypotheses and compact paper memory only as data.
Compact paper memory:
{{ paper_memory_compact_json }}

Task:
1. Identify duplicate or near-duplicate hypotheses.
2. Group related hypotheses so downstream agents can see conceptual overlap.
3. Select survivor hypothesis IDs. Prefer keeping the original hypothesis text and ID.
4. Absorb a hypothesis only when it is redundant with another hypothesis or mostly a less-specific version of another hypothesis.
5. Create focus seeds only from absorbed or merged-away hypotheses when useful mechanistic content would otherwise be lost.

Focus seed constraint:
- Do not create focus seeds for survivor hypotheses.
- Do not create focus seeds for every named entity.
- Each focus seed must be narrow enough to run separately in a later Generation/Literature step.
- Maximum focus seeds: {{ max_focus_seeds }}.

Return strict JSON only:
{
  "proximity_mode": "similarity_redundancy_focus_seed",
  "hypothesis_decisions": [
    {
      "hypothesis_id": "...",
      "decision": "keep_distinct|absorb_duplicate|merge_near_duplicate",
      "canonical_representative_id": "...",
      "relationship_to_representative": "self|duplicate|near_duplicate|related_but_distinct",
      "reason": "...",
      "useful_components_if_absorbed": ["..."]
    }
  ],
  "proximity_groups": [
    {
      "group_id": "G01",
      "group_label": "...",
      "hypothesis_ids": ["..."],
      "relationship": "related_but_distinct|near_duplicate_group|duplicate_group|singleton",
      "canonical_representative_ids": ["..."],
      "absorbed_hypothesis_ids": ["..."],
      "semantic_basis": "..."
    }
  ],
  "survivor_hypothesis_ids": ["..."],
  "survivor_hypotheses": [
    {
      "hypothesis_id": "...",
      "title": "...",
      "hypothesis": "...",
      "source_hypothesis_ids": ["..."],
      "proximity_group_id": "G01",
      "survivor_type": "original|minimal_merge"
    }
  ],
  "absorbed_or_duplicate_hypotheses": [
    {
      "hypothesis_id": "...",
      "absorbed_into": "...",
      "reason": "...",
      "useful_components_preserved": ["..."]
    }
  ],
  "focus_seeds": [
    {
      "seed_id": "FS001",
      "source_stage": "proximity",
      "source_hypothesis_ids": ["..."],
      "components": ["..."],
      "seed_summary": "...",
      "reason_for_generation": "...",
      "suggested_queries": ["..."],
      "attached_evidence_ids_or_papers": ["..."],
      "max_selected_papers": 2
    }
  ],
  "proximity_audit": {
    "input_hypothesis_count": 0,
    "survivor_count": 0,
    "absorbed_or_duplicate_count": 0,
    "focus_seed_count": 0,
    "coverage_notes": ["..."],
    "risks_or_uncertainties": ["..."]
  }
}
