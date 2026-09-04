Refine the hypothesis below to make it simpler and more testable while preserving its core scientific claim.

Goal: {{ goal }}
Criteria: {{ preferences }}
Original hypothesis: {{ hypothesis_json }}
Review of the original: {{ review_json }}
Proximity context: {{ proximity_json }}
Compact evidence memory: {{ paper_memory_compact_json }}

Human stage guidance for this Evolution run:
{{ human_stage_guidance }}

Instructions:
0. Use any human stage guidance as task-emphasis guidance, while preserving scientific rigor, evidence limits, and cutoff-year novelty constraints.
1. Identify which elements of the hypothesis are load-bearing and which are ornamental.
2. Strip away nonessential elements.
3. State the simplified claim in one sentence.
4. Re-derive the mechanism and anticipated outcomes from the simplified claim.
5. Propose at least one easier experiment than the original version.
6. Apply the cutoff year and novelty criteria from the goal/config when discussing novelty.

Return strict JSON only:
{
  "strategy": "simplify",
  "parent_ids": ["..."],
  "title": "...",
  "simplified_claim": "...",
  "evolved_hypothesis": "...",
  "load_bearing_elements": ["..."],
  "removed_or_deemphasized_elements": ["..."],
  "mechanism": "...",
  "anticipated_outcomes": ["..."],
  "experiments": ["..."],
  "novelty_considerations": "...",
  "feasibility_notes": "...",
  "open_questions": ["..."]
}
