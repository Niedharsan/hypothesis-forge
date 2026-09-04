from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.supervisor_config_agent import SupervisorConfigAgent
from agents.generation_rewired import RewiredGenerationAgent
from runtime.context import configure_runtime
from utils.config import load_config, validate_config
from utils.run_logger import start_run_log, finalize_run, log_event, collect_llm_usage_summary

DEFAULT_SOURCES = 'PubMed,EuropePMC,OpenAlex,Crossref,SemanticScholar'
SUPPORTED_SOURCES = {'PubMed', 'EuropePMC', 'OpenAlex', 'Crossref', 'SemanticScholar'}


def parse_sources(raw: str) -> list[str]:
    values = [s.strip() for s in raw.split(',') if s.strip()]
    out = []
    for s in values:
        if s not in SUPPORTED_SOURCES:
            raise SystemExit(f'Unsupported source: {s}. Supported: {sorted(SUPPORTED_SOURCES)}')
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Run clean v31 Supervisor+Generation workflow')
    parser.add_argument('--goal', required=True)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--out-dir', default='runs/rewired_supervisor_generation')
    parser.add_argument('--sources', default=DEFAULT_SOURCES)
    parser.add_argument('--axes', type=int, default=10)
    parser.add_argument('--enable-literature', action='store_true')
    parser.add_argument('--max-queries-per-axis', type=int, default=2)
    parser.add_argument('--papers-per-axis', type=int, default=5)
    parser.add_argument('--model', default='gemini-2.5-flash-lite')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.axes != 10:
        raise SystemExit('This cleaned diagnostic intentionally fixes --axes at 10. Use --axes 10.')
    if args.max_queries_per_axis < 1 or args.max_queries_per_axis > 3:
        raise SystemExit('--max-queries-per-axis must be 1–3 for this diagnostic.')
    if args.papers_per_axis < 1 or args.papers_per_axis > 10:
        raise SystemExit('--papers-per-axis must be 1–10 for this diagnostic.')

    config = load_config(args.config)
    warnings = validate_config(config, runtime_mode='dry_run' if args.dry_run else None, strict=False)
    configure_runtime(config, mode_override='dry_run' if args.dry_run else None)
    run_id = start_run_log(args.goal, 'v31_clean_supervisor_generation')
    for warning in warnings:
        log_event('config', 'warning', {'warning': warning}, status='warning')

    selected_sources = parse_sources(args.sources)
    supervisor = SupervisorConfigAgent().configure(
        args.goal,
        axes=args.axes,
        use_literature=args.enable_literature,
        model=args.model,
    )
    generator = RewiredGenerationAgent(model=args.model)
    output = generator.run(
        objective=args.goal,
        supervisor_config=supervisor.to_dict(),
        sources=selected_sources,
        config_path=args.config,
        enable_literature=args.enable_literature,
        max_queries_per_axis=args.max_queries_per_axis,
        papers_per_axis=args.papers_per_axis,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / '01_supervisor_config.json').write_text(json.dumps(supervisor.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
    (out / '02_generation_axes.json').write_text(json.dumps(output.axes, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / '03_generation_routes.json').write_text(json.dumps(output.routes, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / '04_literature_results.json').write_text(json.dumps([
        {
            'axis_id': r.axis_id,
            'queries': r.queries,
            'evidence_packets': [p.to_dict() for p in r.evidence_packets],
            'warnings': r.warnings,
        }
        for r in output.literature_results
    ], indent=2, ensure_ascii=False), encoding='utf-8')
    (out / '05_generation_hypotheses_payload.json').write_text(json.dumps(output.hypotheses_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'initial_candidates.json').write_text(json.dumps([s.to_dict() for s in output.strategies], indent=2, ensure_ascii=False), encoding='utf-8')

    llm_usage = collect_llm_usage_summary()
    (out / 'llm_usage.json').write_text(json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = {
        'run_id': run_id,
        'goal': args.goal,
        'model': args.model,
        'axes_fixed': 10,
        'axes_returned': len(output.axes.get('axes', [])) if isinstance(output.axes, dict) else None,
        'routes_returned': len(output.routes.get('routes', [])) if isinstance(output.routes, dict) else None,
        'hypotheses_returned': len(output.strategies),
        'hypothesis_count_policy': 'dynamic; no requested hypothesis count',
        'literature_enabled': args.enable_literature,
        'sources': selected_sources,
        'max_queries_per_axis': args.max_queries_per_axis,
        'papers_per_axis': args.papers_per_axis,
        'old_workflow_removed': True,
        'llm_usage_totals': llm_usage.get('totals', {}),
        'llm_usage_by_stage': llm_usage.get('by_stage', {}),
        'llm_usage_by_model': llm_usage.get('by_model', {}),
        'top_prompt_calls': llm_usage.get('top_prompt_calls', []),
        'budget_usage': llm_usage.get('budget_usage', {}),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log_event('workflow', 'v31_clean_complete', summary)
    finalize_run(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
