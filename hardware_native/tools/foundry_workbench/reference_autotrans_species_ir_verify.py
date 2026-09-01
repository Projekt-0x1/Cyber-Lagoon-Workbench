#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from autotrans_species_ir_v0 import (
    BackendLawMappingV0,
    FoundrySpeciesProgramV0,
    SpeciesLawV0,
    direct_translation_readiness,
)


def rejects(**kwargs):
    try:
        FoundrySpeciesProgramV0.build([
            SpeciesLawV0("test", 1, tuple(sorted((str(k), v) for k, v in kwargs.items())))
        ])
        return False
    except ValueError:
        return True


def main():
    started = time.perf_counter()
    laws = (
        SpeciesLawV0("authenticated_external_contact"),
        SpeciesLawV0("source_conditioned_access_evidence"),
        SpeciesLawV0("independent_consequence_revision"),
        SpeciesLawV0("contradiction_reopens_access"),
        SpeciesLawV0("independent_reacquisition"),
        SpeciesLawV0("focal_injury_rematerialization"),
        SpeciesLawV0("finite_plastic_work_preemption", parameters=(("front_budget", 24),)),
        SpeciesLawV0("distributed_constructor_consolidation", parameters=(("minimum_distinct_ticks", 2),)),
    )
    program = FoundrySpeciesProgramV0.build(
        reversed(laws),
        resource_bounds=(("persistent_matter", 1), ("plastic_work_budget", 24)),
    )
    same = FoundrySpeciesProgramV0.build(
        laws,
        resource_bounds=(("plastic_work_budget", 24), ("persistent_matter", 1)),
    )

    mappings = (
        BackendLawMappingV0(
            "authenticated_external_contact", "direct",
            "Direct boundary/contact receipt + exact history",
            ("direct_adult_core.cu",), "MAPPED"),
        BackendLawMappingV0(
            "source_conditioned_access_evidence", "direct",
            "ResidentRawContactBinding/ResidentRecipeIncidence",
            ("direct_network_postbirth_constructor_ecology.inl",), "MAPPED"),
        # Direct currently learns verified action returns and freezes causal-world
        # predictions, but does not yet lower the Workbench law that a changed
        # authenticated predicted return reopens the exact source-conditioned access.
        BackendLawMappingV0(
            "independent_consequence_revision", "direct", "", (), "UNLOWERED"),
        BackendLawMappingV0(
            "contradiction_reopens_access", "direct", "", (), "UNLOWERED"),
        BackendLawMappingV0(
            "independent_reacquisition", "direct", "", (), "UNLOWERED"),
        BackendLawMappingV0(
            "focal_injury_rematerialization", "direct",
            "accounted exact-route injury + resident derivation rematerialization",
            ("direct_network_tract_lesion.cu", "direct_network_resident_development.cu"), "MAPPED"),
        BackendLawMappingV0(
            "finite_plastic_work_preemption", "direct",
            "bounded construction-front injury preemption",
            ("direct_network_resident_development.cu",), "MAPPED"),
        BackendLawMappingV0(
            "distributed_constructor_consolidation", "direct",
            "constructor meta-update recurrence gate",
            ("direct_adult_constructor_meta_update.cuh",), "MAPPED"),
    )
    readiness = direct_translation_readiness(program, mappings)

    checks = {
        "canonical_species_root_ignores_declaration_order": program.root() == same.root(),
        "species_ir_rejects_surface_content": rejects(surface=123),
        "species_ir_rejects_expected_output": rejects(expected_output=1),
        "species_ir_rejects_trained_checkpoint": rejects(checkpoint=1),
        "species_ir_rejects_final_route_identity": rejects(route_identity=7),
        "curriculum_and_adult_state_not_in_species_document": all(
            key not in json.dumps(program.canonical_document(), sort_keys=True)
            for key in ("curriculum", "adult_state", "checkpoint", "expected_output")
        ),
        "translation_audit_fails_closed_on_unlowered_laws": (
            not readiness["translation_ready"]
            and readiness["unlowered"] == (
                "contradiction_reopens_access",
                "independent_consequence_revision",
                "independent_reacquisition",
            )
        ),
        "already_proven_generic_laws_have_named_direct_lowerings": len(readiness["mapped"]) == 5,
    }
    result = {
        "schema": "0x1.reference-autotrans-species-ir-audit.v0",
        "pass": all(checks.values()),
        "checks": checks,
        "translation_ready": readiness["translation_ready"],
        "mapped": readiness["mapped"],
        "unlowered": readiness["unlowered"],
        "species_root": readiness["species_root"],
        "claim": "AUTOTRANS_COMPILER_BOUNDARY_AND_FAIL_CLOSED_TRANSLATION_AUDIT_REFERENCE_ONLY",
        "graph_flip": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_AUTOTRANS_SPECIES_IR " + ("GREEN" if result["pass"] else "RED")
        + f" translation_ready={int(result['translation_ready'])}"
        + f" mapped={len(result['mapped'])} unlowered={len(result['unlowered'])}"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
