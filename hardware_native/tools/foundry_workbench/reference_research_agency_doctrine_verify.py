#!/usr/bin/env python3
"""Focused executable receipt for Cyber Lagoon research-before-decision agency.

This is documentation/workflow verification only. It does not promote AGI capability.
"""
from __future__ import annotations

import json
from pathlib import Path

semantics_free_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True

ROOT = Path(__file__).resolve().parents[3]
PATHS = {
    "agents": ROOT / "AGENTS.md",
    "prompt": ROOT / "hardware_native/tools/foundry_workbench/NEW_AGENT_LANGUAGE_MASTERY_PROMPT.md",
    "scope": ROOT / "hardware_native/tools/foundry_workbench/WORKBENCH_SCOPE.md",
    "claude": ROOT / "CLAUDE.md",
    "controllability": ROOT / "docs/research/sapolsky/2026-08-31-controllability-contingency-hardware-ethology.md",
    "control_assay": ROOT / "hardware_native/tools/foundry_workbench/reference_agi_controllability_language_verify.py",
    "destructive_audit": ROOT / "hardware_native/tools/foundry_workbench/reference_productive_leaf_sapolsky_destructive_audit_verify.py",
    "leaf_factor": ROOT / "hardware_native/tools/foundry_workbench/reference_mathematical_adult_leaf_surface_factorization_verify.py",
    "language_fast": ROOT / "hardware_native/tools/foundry_workbench/run_language_mastery_fast.sh",
    "grammar_plan": ROOT / "plans/2026-08-30-h-compositional-grammar-adult-gap.md",
    "grammar_diary": ROOT / "docs/diary/2026-08-31/2026-08-31-mathematical-adult-productive-leaf-factorization.md",
    "graph": ROOT / "docs/constitutional_dependency_graph.json",
}


def main() -> int:
    b = {k: p.read_text(encoding="utf-8") for k, p in PATHS.items()}
    checks = {
        "primary_and_sapolsky_are_hard_minimums": (
            "Mechanism-specific primary grounding" in b["prompt"]
            and "Sapolsky-style cross-level grounding" in b["prompt"]
            and "two non-negotiable grounding passes" in b["agents"]
        ),
        "research_agency_is_explicit": all("research agency" in b[k].lower() for k in ("agents", "prompt", "scope")),
        "grounding_is_destructive_not_supporting_context": (
            "destructive research law" in b["agents"].lower()
            and "use both passes destructively" in b["prompt"].lower()
            and "destructive causal audit" in b["claude"].lower()
        ),
        "sapolsky_axes_are_explicit_counterfactuals": (
            all(term in b["agents"].lower() for term in ("prior development", "recent history", "body/resource", "controllability", "consequence", "timescale/recovery"))
            and all(term in b["prompt"].lower() for term in ("prior development", "recent history", "body/resource", "controllability", "consequence", "recovery"))
        ),
        "novel_synthesis_still_requires_resident_competition": (
            "resident competition" in b["agents"].lower()
            and "resident competition" in b["prompt"].lower()
            and "resident competition" in b["claude"].lower()
        ),
        "active_frontier_executes_destructive_history_audit": (
            "matched_current_contingency_cells" in b["control_assay"]
            and "focal_history_lesion_removes_only_advantage" in b["control_assay"]
            and "continued_disconfirmation_extinguishes_history_advantage" in b["control_assay"]
            and "ordinary_control_evidence_reacquires_history_advantage" in b["control_assay"]
            and "same_current_context_different_consequence_history_changes_winner" in b["destructive_audit"]
            and "focal_control_history_lesion_destroys_only_history_advantage" in b["destructive_audit"]
            and "timescale_disconfirmation_extinguishes_and_lived_control_recovers" in b["destructive_audit"]
        ),
        "framing_only_cannot_reenter_active_grammar_authority": (
            "SAPOLSKY_FRAMING_ONLY_CROSS_LEVEL_STATE_HISTORY_CAUSATION_PRESERVED" not in b["leaf_factor"]
            and "`SAPOLSKY_FRAMING_ONLY` for this narrow encoding" not in b["grammar_plan"]
            and "**Sapolsky check:** `SAPOLSKY_FRAMING_ONLY`" not in b["grammar_diary"]
            and "Sapolsky disposition is FRAMING_ONLY for the narrow encoding" not in b["graph"]
            and "DESTRUCTIVE_AUDIT_REQUIRED_AND_PARTIALLY_PAID" in b["graph"]
            and "reference_productive_leaf_sapolsky_destructive_audit_verify.py" in b["leaf_factor"]
            and "sapolsky-destructive-audit:reference_productive_leaf_sapolsky_destructive_audit_verify.py" in b["language_fast"]
        ),
        "random_cross_field_papers_are_allowed": (
            "apparently unrelated fields" in b["agents"]
            and "apparently unrelated fields" in b["prompt"]
            and "apparently unrelated fields" in b["scope"]
        ),
        "named_competitors_are_examples_not_closed_canon": (
            "Pim de Witte/General Intuition" in b["agents"]
            and "Pim de Witte/General Intuition" in b["prompt"]
            and "Pim de Witte/General Intuition" in b["scope"]
            and "not a closed canon" in b["agents"]
            and "not a closed canon" in b["prompt"]
            and "examples rather than a closed list" in b["scope"]
        ),
        "competitor_ideas_are_filtered_through_architecture": (
            "engineering donors" in b["agents"]
            and "architecture-compatible functions" in b["agents"]
            and "never wholesale training stacks" in b["prompt"]
        ),
        "research_propagates_to_issue_graph_genome": (
            "Research-propagation law" in b["scope"]
            and "GitHub issues" in b["prompt"]
            and "constitutional graph" in b["prompt"]
            and "Genome/Network" in b["prompt"]
        ),
        "current_frontier_demonstrates_free_research_synthesis": (
            "USED — Rich Sutton" in b["controllability"]
            and "USED — LeCun / JEPA" in b["controllability"]
            and "USED — Pim de Witte / General Intuition" in b["controllability"]
            and "LIMITED USE — DeepMind" in b["controllability"]
            and "NOVEL SYNTHESIS" in b["controllability"]
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    result = {
        "schema": "cyber-lagoon.reference-research-agency-doctrine.v1",
        "pass": not failed,
        "reference_only": True,
        "capability_promotion": False,
        "economic_gain": {"closed_mandatory_donor_checklists": 0, "research_entrypoints_checked": len(PATHS)},
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "checks": checks,
        "failed": failed,
    }
    print("FOUNDRY_RESEARCH_AGENCY_DOCTRINE_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
