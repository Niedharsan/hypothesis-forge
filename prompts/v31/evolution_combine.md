You are an expert in scientific synthesis. Combine the best parts of the two hypotheses below into a new, stronger hypothesis.

Goal: {{ goal }}
Criteria: {{ preferences }}
Hypothesis A: {{ hypothesis_a_json }}
Review of hypothesis A: {{ review_a_json }}
Hypothesis B: {{ hypothesis_b_json }}
Review of hypothesis B: {{ review_b_json }}
Proximity context: {{ proximity_json }}
Compact evidence memory: {{ paper_memory_compact_json }}

Human stage guidance for this Evolution run:
{{ human_stage_guidance }}

Instructions:
0. Use any human stage guidance as task-emphasis guidance, while preserving scientific rigor, evidence limits, and cutoff-year novelty constraints.
1. Identify the strongest mechanism in A and the strongest mechanism in B.
2. State whether there are contradictions or tensions between A and B.
3. Resolve those tensions explicitly if combining is justified.
4. If the hypotheses are merely adjacent but not usefully combinable, say so and produce the best narrower synthesis.
5. The final hypothesis should be more specific and testable than either parent.
6. Apply the cutoff year and novelty criteria from the goal/config.

Return strict JSON only:
{
  "strategy": "combine",
  "parent_ids": ["...", "..."],
  "title": "...",
  "strongest_parts_from_parents": ["..."],
  "contradictions_or_tensions": ["..."],
  "resolution": "...",
  "evolved_hypothesis": "...",
  "candidate_entities_or_interventions": ["..."],
  "mechanism": "...",
  "anticipated_outcomes": ["..."],
  "experiments": ["..."],
  "novelty_considerations": "...",
  "feasibility_notes": "...",
  "open_questions": ["..."]
}
