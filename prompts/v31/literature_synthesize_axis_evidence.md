You are synthesizing literature evidence for one biological research axis.

Your goal is to create a compact, evidence-preserving synthesis for downstream hypothesis generation. Do not write a general review. Preserve only information that can support, weaken, or shape scientific hypotheses.

You are given:
1. the original research goal,
2. the selected biological axis,
3. the final parent subtopics and covered branches,
4. selected evidence packets from literature retrieval,
5. entity annotations when available.

Discovery axis:
{{ axis_json }}

Literature subtopics:
{{ subtopics_json }}

Retrieved evidence packets:
{{ abstracts_context }}

Synthesis rules:
- Be concise. Each string should usually be one sentence.
- Use evidence IDs whenever a claim comes from retrieved evidence.
- Preserve direct evidence and transferable/background evidence, but label the directness clearly.
- Keep branch-level distinctions only when they affect hypothesis generation.
- Do not repeat the same claim in multiple sections.
- Do not output long narrative reviews, long abstract summaries, or paper-by-paper summaries unless a paper contains a distinct usable finding.
- Prefer fewer, higher-value records over exhaustive coverage.
- If evidence is weak, missing, crowded, or contradictory, say so directly.
- Include no more than the caps shown in the schema comments.

Return valid JSON only using this compact schema:
{
  "axis_id": "string",
  "axis_name": "string",
  "evidence_records": [
    {
      "evidence_record_id": "ER01",
      "finding": "One concise synthesis finding useful for hypothesis generation.",
      "source_evidence_ids": ["A04_T01_E001"],
      "subtopic_ids": ["A04_T01"],
      "directness": "direct|related|transferable|background|unclear",
      "system_or_context": "Disease, organism, cell type, model, assay, dataset, or context if stated.",
      "key_entities_or_variables": ["entity/process/variable"],
      "limitations": ["Concise limitation, if important."]
    }
  ],
  "subtopic_summaries": [
    {
      "subtopic_id": "string",
      "subtopic_name": "string",
      "supported_findings": ["Up to 3 concise findings."],
      "useful_handles_or_variables": ["Up to 4 entities, processes, variables, readouts, perturbations, contexts, or methods."],
      "gaps_or_tensions": ["Up to 2 concise gaps, tensions, or missing branches."]
    }
  ],
  "cross_subtopic_connections": [
    {
      "connection": "Concise connection across subtopics that may support a hypothesis.",
      "subtopic_ids": ["string"],
      "supporting_evidence_record_ids": ["ER01"]
    }
  ],
  "hypothesis_relevant_gaps": [
    {
      "gap": "Concise gap, underexplored context, contradiction, or unresolved question.",
      "why_it_matters": "Why this could shape a hypothesis.",
      "related_evidence_record_ids": ["ER01"]
    }
  ],
  "weak_or_missing_areas": [
    {
      "area": "Input branch/subtopic/concept with weak, missing, merged, or contradictory evidence.",
      "reason": "Concise reason."
    }
  ],
  "additional_search_queries": [
    {
      "query": "string",
      "purpose": "string"
    }
  ],
  "axis_level_summary": "Very concise integrated summary for hypothesis generation."
}

Caps:
- evidence_records: max 12 total
- subtopic_summaries: max 1 per input subtopic
- cross_subtopic_connections: max 5
- hypothesis_relevant_gaps: max 6
- weak_or_missing_areas: max 5
- additional_search_queries: max 3
