#!/usr/bin/env python3
"""Executable receipt for Cyber Lagoon mechanic-doctrine recovery.

This is a documentation/architecture-consistency receipt, not an AGI capability
claim. It prevents the live mechanic entrypoints from simultaneously prescribing
retired Target-v5/worktree/Network-first doctrine and the Mathematical AGI model.
"""
from __future__ import annotations

import json
from pathlib import Path

semantics_free_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True

ROOT = Path(__file__).resolve().parents[3]
FILES = {
    "agents": ROOT / "AGENTS.md",
    "claude": ROOT / "CLAUDE.md",
    "prompt": ROOT / "hardware_native/tools/foundry_workbench/NEW_AGENT_LANGUAGE_MASTERY_PROMPT.md",
    "scope": ROOT / "hardware_native/tools/foundry_workbench/WORKBENCH_SCOPE.md",
    "hypothesis": ROOT / "hardware_native/tools/foundry_workbench/MATHEMATICAL_SUBSTRATE_HYPOTHESIS.md",
    "playbook": ROOT / "hardware_native/tools/foundry_workbench/MATHEMATICAL_ADULT_REFACTOR_PLAYBOOK.md",
    "autotrans": ROOT / "hardware_native/tools/foundry_workbench/AUTOTRANS_EXECUTABLE_SPEC.md",
    "grounding": ROOT / "docs/research/foundry_mathematical_cognitive_quantity_grounding_2026-08-31.md",
}


def text(name: str) -> str:
    return FILES[name].read_text(encoding="utf-8")


def main() -> int:
    bodies = {name: text(name) for name in FILES}
    forbidden = {
        "target_v5_design_authority": "Revision 12 remains the semantic authority",
        "network_first_ontology": "Network-first rule",
        "direct_target_architecture": "The **target architecture** is the one continuing Direct",
        "representation_no_exception": "There is no representation, state-minimization",
        "all_commits_need_language": "Every substantive commit must contain a **measured visible language",
        "old_product_inference": "Engineering inference: 0x1",
        "gamma_curriculum_plane": "not compiled into Gamma",
        "frozen_lm2_migration": "The first candidate migration is LM2",
        "old_handoff_name": "`0X1_JOB` — one coherent engineering mission",
    }
    joined = "\n".join(bodies.values())
    checks = {
        "cyber_lagoon_and_agi_are_current_identity": (
            "# Cyber Lagoon engineering workflow" in bodies["agents"]
            and "The AI under construction is **AGI**" in bodies["agents"]
            and "# Cyber Lagoon — New frontier-model mechanic prompt" in bodies["prompt"]
        ),
        "whole_agi_stateful_operator_is_explicit": (
            "Phi_G" in bodies["prompt"]
            and "future-causally-sufficient" in bodies["prompt"]
            and "future update authority" in bodies["prompt"]
        ),
        "target_v5_is_direct_donor_not_design_authority": (
            "not design authority over AGI" in bodies["agents"]
            and "not architecture authority" in bodies["scope"]
        ),
        "network_and_recipe_are_replaceable_roles": (
            "Active-interaction role law" in bodies["scope"]
            and "not a required primitive" in bodies["scope"]
            and "not mandatory persistence ontology" in bodies["scope"]
        ),
        "relation_solver_lowering_are_separate": all(
            "solver" in bodies[name].lower() and "lowering" in bodies[name].lower()
            for name in ("prompt", "playbook", "autotrans")
        ),
        "economic_only_exception_is_semantics_free": (
            "Only genuinely semantics-free" in bodies["scope"]
            and "Only genuinely semantics-free" in bodies["playbook"]
            and "representation/state refactor inside AGI cognition" in bodies["playbook"]
        ),
        "current_depth_six_language_floor_is_protected": (
            "64 distinct held-out clauses" in bodies["prompt"]
            and "2,495 public bytes" in bodies["prompt"]
            and "depth 6" in bodies["playbook"]
            and "2495 public bytes" in bodies["grounding"]
        ),
        "zero_composite_closure_factor_owner_is_recorded": (
            "zero composite closures" in bodies["prompt"]
            and "Adult has no second surface shadow namespace" in bodies["hypothesis"]
        ),
        "protected_prediction_valence_agency_social_laws_present": (
            "unexpected != bad" in bodies["prompt"]
            and "agency/controllability" in bodies["prompt"]
            and "belief_of_user" in bodies["prompt"]
        ),
        "research_lenses_are_mandatory_and_role_bounded": (
            "Sapolsky / organism causation" in bodies["prompt"]
            and "Chomsky / psycholinguistics" in bodies["prompt"]
            and "Rich Sutton" in bodies["prompt"]
            and "Tony Robbins" in bodies["prompt"]
            and "zero mechanistic authority" in bodies["prompt"]
            and "Do not import LLM, transformer, token, next-token" in bodies["prompt"]
            and "Research-function law" in bodies["scope"]
            and "Mandatory research-role boundaries" in bodies["claude"]
        ),
        "organism_research_conclusions_are_explicit": (
            "Intrinsic motivation / exploration" in bodies["prompt"]
            and "host intrinsic-reward head" in bodies["prompt"]
            and "McEwen / allostasis" in bodies["prompt"]
            and "Berridge / motivation" in bodies["prompt"]
            and "pursuit/incentive salience" in bodies["prompt"]
            and "Biology-to-silicon translation rule" in bodies["prompt"]
            and "evolutionary accidents" in bodies["prompt"]
        ),
        "new_handoff_namespace_is_cyber_lagoon": (
            "CYBER_LAGOON_JOB" in bodies["agents"]
            and "Historical `0X1_*` handoff labels" in bodies["agents"]
        ),
        "no_retired_contradiction_patterns": not any(value in joined for value in forbidden.values()),
    }
    failed = [name for name, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mechanic-doctrine-recovery.v1",
        "pass": not failed,
        "reference_only": True,
        "capability_promotion": False,
        "economic_gain": {
            "complexity_reduction": len(forbidden),
            "contradictory_doctrine_patterns_remaining": sum(
                int(value in joined) for value in forbidden.values()
            ),
            "doctrine_entrypoints_checked": len(FILES),
        },
        "checks": checks,
        "failed": failed,
    }
    status = "GREEN" if result["pass"] else "RED"
    print(f"FOUNDRY_MECHANIC_DOCTRINE_RECOVERY_{status} contradictions_remaining={result['economic_gain']['contradictory_doctrine_patterns_remaining']} entrypoints={len(FILES)}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
