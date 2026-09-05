from pathlib import Path

from agents.generation_rewired import RewiredGenerationAgent


ROOT = Path(__file__).resolve().parents[1]


def test_active_generation_prompts_are_domain_general() -> None:
    prompt_paths = [
        ROOT / "prompts/v31/generation_decompose.md",
        ROOT / "prompts/v31/generation_hypotheses.md",
        ROOT / "prompts/v31/literature_decompose_axis_v2_entity_map.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in prompt_paths)

    assert "AML" not in combined
    assert "candidate_drug_or_class" not in combined
    assert "candidate_intervention_or_focus" in combined


def test_generic_hypothesis_payload_coerces_without_disease_or_drug_assumptions() -> None:
    payload = {
        "hypotheses": [
            {
                "hypothesis_id": "S001",
                "axis_id": "A01",
                "title": "Root microbiome composition modulates drought tolerance",
                "hypothesis": "A defined rhizosphere community changes Arabidopsis drought tolerance through root stress signaling.",
                "candidate_intervention_or_focus": "rhizosphere microbial community composition",
                "mechanistic_rationale": "Microbial metabolites can alter root signaling and water-stress responses.",
                "falsifiable_predictions": ["Changing community composition changes drought-survival phenotypes."],
                "key_assumptions": ["The community can be experimentally manipulated."],
                "first_validation_step": "Compare defined communities in a controlled drought assay.",
                "novelty_risk": "medium",
            }
        ]
    }

    strategies = RewiredGenerationAgent()._coerce_hypotheses(
        payload,
        "How does root microbiome composition influence drought tolerance in Arabidopsis?",
        [],
    )

    assert len(strategies) == 1
    assert strategies[0].proposed_intervention == "rhizosphere microbial community composition"
    assert "Arabidopsis" in strategies[0].objective_relevance_rationale
