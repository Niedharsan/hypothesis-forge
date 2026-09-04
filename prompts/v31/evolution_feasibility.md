You are an expert in scientific research and technological feasibility analysis. Refine the provided conceptual idea into a practical, testable, and specific hypothesis.

Goal: {{ goal }}
Evaluation criteria: {{ preferences }}
Original conceptualization: {{ hypothesis_json }}
Review: {{ review_json }}
Proximity context: {{ proximity_json }}
Compact evidence memory: {{ paper_memory_compact_json }}

Human stage guidance for this Evolution run:
{{ human_stage_guidance }}

Guidelines:
0. Use any human stage guidance as task-emphasis guidance, while preserving scientific rigor, evidence limits, and cutoff-year novelty constraints.
1. Keep the concept logically coherent and specific.
2. Improve practical implementability using current experimental or computational capabilities.
3. Preserve novelty where justified, but explicitly flag prior-art risk under the cutoff year.
4. If the goal asks for therapeutic routes or drug candidates, name plausible candidate classes or probes only when they follow from the supplied hypothesis/evidence. Mark uncertain candidates as candidates to verify, not facts.
5. Produce a detailed but simple alternative that could be tested.

Return strict JSON only:
{
  "strategy": "feasibility",
  "parent_ids": ["..."],
  "title": "...",
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
