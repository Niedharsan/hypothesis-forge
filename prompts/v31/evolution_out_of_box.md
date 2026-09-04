You are an expert researcher tasked with generating one novel, singular hypothesis inspired by analogous elements from the provided concepts.

Goal: {{ goal }}
Criteria: {{ preferences }}
Inspiring hypotheses: {{ hypotheses_json }}
Reviews: {{ reviews_json }}
Proximity context: {{ proximity_json }}
Compact evidence memory: {{ paper_memory_compact_json }}

Human stage guidance for this Evolution run:
{{ human_stage_guidance }}

Instructions:
0. Use any human stage guidance as task-emphasis guidance, while preserving scientific rigor, evidence limits, and cutoff-year novelty constraints.
1. Use analogy and inspiration, not direct aggregation.
2. Identify a promising avenue that is not merely a restatement of the supplied hypotheses.
3. Develop one original and specific hypothesis aligned to the goal.
4. The hypothesis must be testable and mechanistically articulated.
5. Apply the cutoff year and novelty criteria from the goal/config.
6. If the goal asks for therapeutic routes or drug candidates, name candidate classes or probes only when justified by supplied context; mark uncertain candidates as candidates to verify.

Return strict JSON only:
{
  "strategy": "out_of_box",
  "parent_ids": ["..."],
  "title": "...",
  "inspiration_used": ["..."],
  "evolved_hypothesis": "...",
  "candidate_entities_or_interventions": ["..."],
  "target_or_mechanistic_node": "...",
  "disease_context_or_subset": "...",
  "mechanism": "...",
  "anticipated_outcomes": ["..."],
  "experiments": ["..."],
  "novelty_considerations": "...",
  "prior_art_risk": "low|medium|high|unknown",
  "feasibility_notes": "...",
  "open_questions": ["..."]
}
