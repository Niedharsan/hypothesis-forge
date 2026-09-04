#!/usr/bin/env bash
set -euo pipefail
mkdir -p runs
export GEMINI_RETRY_ATTEMPTS=${GEMINI_RETRY_ATTEMPTS:-6}
export GEMINI_RETRY_DELAYS=${GEMINI_RETRY_DELAYS:-"10,20,40,80,160,300"}
GOAL="Identify novel drug candidates for acute myeloid leukemia that have not previously been used for AML."
for REP in 1 2 3; do
  OUT="runs/v73_subtopics_cutoff2023_rep${REP}"
  LOG="runs/v73_subtopics_cutoff2023_rep${REP}_terminal.log"
  echo "=== v73 subtopic replicate ${REP} ==="
  python scripts/run_axis_first_literature_generation.py \
    --goal "$GOAL" \
    --sources PubMed,EuropePMC,OpenAlex,Crossref \
    --cutoff-year 2023 \
    --use-pubtator \
    --max-subtopics-per-axis 5 \
    --max-queries-per-subtopic 5 \
    --raw-papers-per-source-query 5 \
    --ai-papers-per-subtopic 3 \
    --ai-papers-per-axis 15 \
    --max-llm-calls 120 \
    --stop-after subtopics \
    --clean-out-dir \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG"
done
zip -r v73_triplicate_subtopics_cutoff2023.zip \
  runs/v73_subtopics_cutoff2023_rep1 \
  runs/v73_subtopics_cutoff2023_rep2 \
  runs/v73_subtopics_cutoff2023_rep3 \
  runs/v73_subtopics_cutoff2023_rep1_terminal.log \
  runs/v73_subtopics_cutoff2023_rep2_terminal.log \
  runs/v73_subtopics_cutoff2023_rep3_terminal.log \
  runs/logs
