#!/usr/bin/env python3
"""Fast falsifier for lifetime-scaled construction exemplar rebuilding.

A new language construction should update its already identified structural factor,
not replay every construction contact the Adult has lived. The assay also requires
that the compact exemplar factor remains causally necessary for held-out productive
composition.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1  # noqa: E402

CLAUSE = 9001
A1, A2, G1, G2, V1, V2, O1, O2 = 101, 102, 201, 202, 301, 302, 401, 402
X = (A1, G1, V1, O1)
PARTIAL = (A2, G1, V1, O1)
Y = (A2, G2, V2, O2)
HELD = (A1, G2, V1, O2)
SX = b"the careful engineer tests the sensor."
SP = b"the quiet engineer tests the sensor."
SY = b"the quiet technician inspects the valve."
NAMES = {
    A1: "careful", A2: "quiet", G1: "engineer", G2: "technician",
    V1: "tests", V2: "inspects", O1: "sensor", O2: "valve",
}


def exemplar_mass(adult, tid):
    return len(adult._template_exemplars.get(int(tid), ()))


def raw_observation_bytes(adult):
    rows = getattr(adult, "_construction_observations", ())
    return sum(len(row[2]) for row in rows)


def main():
    started = time.perf_counter()
    adult = LanguageMasteryAdultV1()
    for feature, text in NAMES.items():
        adult.observe_surface_item(feature, text.encode(), 1000 + feature)
        adult.observe_surface_item(feature, text.encode(), 2000 + feature)

    # Preserve the existing developmental phenotype: supported concrete pattern,
    # partial variation, then full cross-port diversity.
    adult.observe_surface_construction(CLAUSE, X, SX, 3001)
    adult.observe_surface_construction(CLAUSE, X, SX, 3002)
    concrete = adult.construction_productivity(CLAUSE, X)
    held_before_diversity = adult.construction_productivity(CLAUSE, HELD)
    adult.observe_surface_construction(CLAUSE, PARTIAL, SP, 3003)
    partial = adult.construction_productivity(CLAUSE, HELD)
    adult.observe_surface_construction(CLAUSE, Y, SY, 3004)
    productive = adult.construction_productivity(CLAUSE, HELD)
    held_surface = tuple(adult.leaf(CLAUSE, HELD).surface) if productive else ()

    template = adult.language.template(CLAUSE, 4)
    if template is None:
        raise RuntimeError("construction-factor-locality:no_template")
    tid = int(template.identity[:15], 16)

    # Inflate lifetime contacts without adding a new structural distinction.
    for _ in range(64):
        adult.observe_surface_construction(CLAUSE, X, SX, 3001)
    raw_bytes_before = raw_observation_bytes(adult)
    mass_before = exemplar_mass(adult, tid)

    # Count the work performed by one more identical contact. The incumbent rebuild
    # calls template_candidates once per historical observation; the target local
    # factor update needs no whole-history template-candidate pass.
    calls = {"template_candidates": 0}
    original = adult.language.template_candidates
    def counted(context, arity):
        calls["template_candidates"] += 1
        return original(context, arity)
    adult.language.template_candidates = counted
    adult.observe_surface_construction(CLAUSE, X, SX, 3001)
    adult.language.template_candidates = original

    mass_after = exemplar_mass(adult, tid)
    raw_bytes_after = raw_observation_bytes(adult)
    compact_rows=adult._template_exemplars.get(tid,set())
    compact_projection={'template':tid,'examples':[list(row) for row in sorted(compact_rows)]}
    legacy_sources={X:(3001,3002),PARTIAL:(3003,),Y:(3004,)}
    legacy_projection={'template':tid,'examples':[
        {'concepts':list(row),'sources':list(legacy_sources[row])} for row in sorted(compact_rows)]}
    compact_factor_bytes=len(json.dumps(compact_projection,sort_keys=True,separators=(',',':')).encode())
    legacy_factor_bytes=len(json.dumps(legacy_projection,sort_keys=True,separators=(',',':')).encode())

    saved = copy.deepcopy(adult._template_exemplars)
    adult._template_exemplars.pop(tid, None)
    lesion_blocks_productivity = adult.construction_productivity(CLAUSE, HELD) == 0
    adult._template_exemplars = saved
    recovered = adult.construction_productivity(CLAUSE, HELD)

    checks = {
        "concrete_supported_before_abstraction": concrete == 1,
        "heldout_refused_before_diversity": held_before_diversity == 0,
        "partial_diversity_still_insufficient": partial == 0,
        "diversity_unlocks_heldout_productivity": productive >= 2,
        "heldout_surface_preserved": bytes(held_surface) == b"the careful technician tests the valve.",
        "raw_construction_replay_deleted": not hasattr(adult, "_construction_observations") and raw_bytes_after == 0,
        "single_update_has_no_lifetime_template_rescan": calls["template_candidates"] <= 1,
        "same_source_repeat_does_not_grow_exemplar_factor": mass_after == mass_before,
        "exemplar_factor_has_no_nested_source_sets": isinstance(compact_rows,set) and all(isinstance(row,tuple) for row in compact_rows),
        "compact_exemplar_factor_smaller_than_legacy_provenance": compact_factor_bytes < legacy_factor_bytes,
        "exemplar_factor_lesion_blocks_productivity": lesion_blocks_productivity,
        "exemplar_factor_restore_recovers_productivity": recovered == productive,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-construction-factor-locality.v1",
        "pass": not failed,
        "reference_only": True,
        "representation": "TEMPLATE_FACTOR_TO_CONCEPT_TUPLE_SET",
        "legacy_nested_factor_bytes": legacy_factor_bytes,
        "compact_factor_bytes": compact_factor_bytes,
        "repeated_contacts": 65,
        "raw_observation_bytes_before_final_contact": raw_bytes_before,
        "raw_observation_bytes_after_final_contact": raw_bytes_after,
        "final_contact_template_candidate_calls": calls["template_candidates"],
        "exemplar_mass_before_final_contact": mass_before,
        "exemplar_mass_after_final_contact": mass_after,
        "productivity": {
            "concrete": concrete,
            "held_before_diversity": held_before_diversity,
            "partial": partial,
            "held_after_diversity": productive,
        },
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_CONSTRUCTION_FACTOR_LOCALITY_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
