You are planning broad, balanced literature retrieval for one biomedical discovery axis.

Discovery axis:
{{ axis_json }}

Maximum query families: {{ max_query_families }}

Generate a small set of broad query families for entity/concept harvesting before final subtopic generation.

Instructions:
- Query families should cover distinct aspects of the axis without assuming fixed biology-domain categories.
- Critical: Query families should be biologically diverse and non-overlapping. Each family must probe a distinct biological mechanism, pathway, compartment, process, or perturbation class. Do not use comma-separated near-synonyms or multiple families that restate the same umbrella concept.
- Prefer 4-6 query families unless the axis is very narrow.
- Each family should have one concise search query, not many variants.
- Avoid over-specific named answers unless they are explicit in the axis.
- Use full terms where possible and include common abbreviations only when helpful.
- The goal is coverage for an entity/concept map, not final evidence selection.

Return strict JSON only:
{
  "axis_id": "A01",
  "query_families": [
    {
      "family_id": "QF01",
      "name": "short family name",
      "query": "3-8 key terms",
      "coverage_intent": "which part of the axis this family probes"
    }
  ]
}
