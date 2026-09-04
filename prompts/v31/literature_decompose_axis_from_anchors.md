You are a research specialist at literature decomposition.

Your task is to convert one discovery axis into a search-ready set of literature subtopics using the axis text and axis-level anchor papers.

Discovery axis:
{{ axis_json }}

Axis-level anchor papers/summaries:
{{ anchor_context }}

Create a compact set of literature subtopics for this axis. Do not force a fixed number; use only as many subtopics as needed to cover distinct mechanisms/processes/intervention handles in the axis. A code safety cap of {{ max_subtopics }} subtopics applies.

Rules:
- Use the anchor papers only to calibrate the map of mechanisms/processes inside the axis.
- Merge subtopics that mainly differ by context, upstream framing, resistance framing, or wording but would rely on mostly the same evidence, mechanisms, perturbations, or readouts.
- Keep subtopics separate when they represent distinct mechanisms, intervention handles, assays, readouts, or hypothesis paths.
- Add a subtopic only if the axis text or anchor papers reveal a major mechanism/process needed to cover the axis.
- Preserve important explicit mechanisms, pathways, systems, processes, or intervention handles named in the axis unless they are clearly redundant.
- Keep subtopics narrow enough for targeted literature search and broad enough to yield useful findings.
- For each subtopic, generate up to {{ max_queries_per_subtopic }} focused search queries.
- Prefer clear full terms over ambiguous abbreviations.
- Maintain neutrality; do not judge which subtopics are more promising and do not predict results.

Return strict JSON:
{
  "axis_id": "A01",
  "anchor_informed": true,
  "subtopics": [
    {
      "subtopic_id": "A01_T01",
      "question": "focused research question",
      "rationale": "why this subtopic is needed",
      "distinct_angle": "what makes this non-overlapping within the axis",
      "merged_or_covered_inputs": ["mechanisms/processes/framing covered by this subtopic"],
      "search_queries": ["query 1", "query 2", "query 3"]
    }
  ],
  "collapsed_or_merged_notes": ["brief notes on what was merged and why"],
  "coverage_audit": "brief note on how the subtopics cover the axis"
}
