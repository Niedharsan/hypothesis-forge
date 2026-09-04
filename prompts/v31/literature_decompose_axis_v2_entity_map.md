You are a biomedical research strategist creating a targeted literature-retrieval mechanism map.

This is VERSION 2: AXIS-FIRST + ENTITY-MAP-INFORMED COVERAGE MAP.

You are given:
1. the original biological axis,
2. broad query-family retrieval summaries,
3. candidate paper titles/abstract snippets,
4. PubTator or fallback entity/concept counts from the candidate pool.

Discovery axis:
{{ axis_json }}

Entity/concept inventory:
{{ concept_inventory_json }}

Maximum parent subtopics: {{ max_subtopics }}
Maximum search queries per parent subtopic: {{ max_queries_per_subtopic }}

Create a compact set of parent subtopics for targeted literature review.

Purpose of this version:
- Start from the same axis-faithful coverage logic as v1.
- Use broad retrieval and entity/concept evidence only to refine the map, add missing branches, and detect overlooked distinct mechanisms.
- Prevent retrieval popularity bias from replacing explicit axis concepts.

Instructions:
- First identify the explicit mechanisms, intervention handles, biological processes, cell-state contexts, and experimental angles in the original axis.
- Treat these axis-explicit concepts as mandatory coverage items.
- Then use the retrieved entity/concept inventory to add retrieval-supported branches, candidate proteins/genes/drugs/pathways, and possible overlooked mechanisms.
- The retrieved inventory may enrich, split, merge, or add subtopics, but it must not erase an explicit axis concept unless that concept is preserved under a parent subtopic or excluded with a clear reason.
- Create parent subtopics for distinct axis-relevant biological routes. Concepts that do not define their own route should be represented as branches, contexts, or actionable handles under the closest parent subtopic.
- Use query-family provenance to avoid popularity bias from one overrepresented field.
- Do not rank only by frequency. Prioritize: axis coverage, mechanistic distinctness, relevance to the axis, query-family diversity, and experimental/hypothesis separability.
- Merge overlapping concepts into parent subtopics when they share evidence base, perturbation logic, biological function, or readout.
- Preserve important distinct mechanisms inside `covered_branches` so they are not lost during merging.
- Do not use fixed disease-domain categories. Let the axis and retrieved concept inventory determine the biology.
- Every explicit axis concept and every important retrieved concept must be represented as a parent, represented as a branch, merged with a reason, or excluded with a reason.
- Generate search queries from both parent names and covered branches.
- Keep search queries short, biomedical, and searchable: 3-8 key terms.

Return strict JSON only:
{
  "method": "v2_axis_first_entity_map_informed_coverage_map",
  "axis_id": "A01",
  "axis_explicit_concepts": ["..."],
  "retrieval_supported_concepts_used": ["..."],
  "subtopics": [
    {
      "subtopic_id": "A01_T01",
      "name": "parent subtopic name",
      "covered_branches": ["branch/mechanism/entity to preserve"],
      "supporting_query_families": ["QF01"],
      "supporting_entities_or_terms": ["gene/protein/drug/mechanism term"],
      "rationale": "why this is a distinct literature route and how retrieved evidence changed or refined the v1-style map",
      "search_queries": ["3-8 key terms", "..."]
    }
  ],
  "coverage_audit": [
    {
      "concept": "explicit or retrieved concept",
      "concept_source": "axis_explicit | retrieved_entity_map | both",
      "status": "covered_as_parent | covered_as_branch | merged | excluded",
      "parent_subtopic": "A01_T01 or null",
      "reason": "brief reason"
    }
  ],
  "retrieval_bias_audit": [
    {
      "possible_bias": "overrepresented query family, popular field, or common AML concept",
      "handling": "how you prevented it from dominating the map"
    }
  ],
  "excluded_or_merged_concepts": [
    {"concept": "...", "decision": "merged/excluded", "reason": "..."}
  ],
  "coverage_risk_notes": ["possible missing branch or ambiguity after reconciling axis concepts with retrieved concepts"]
}
