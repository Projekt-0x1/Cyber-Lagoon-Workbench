#!/usr/bin/env python3
"""Fast LM2 injury/rematerialization adapter over the continuing reference Adult.

Reference-only design tournament. The adapter models physical implementation
admissibility/work scheduling; it never chooses a surface, referent, or answer.
Language phenotype is measured by asking the unchanged ReferenceOrganismV2 to
realize its learned outward actions after the modeled implementation is restored.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass

from reference_organism_v2 import ReferenceOrganismV2, CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_organism_fast_mapping_verify import (
    NEW, PROFILES, command, outward, prepare, outcome,
)


@dataclass
class Front:
    source: int
    recipe_cell: int
    support: int
    credit: int
    injury: bool = False


@dataclass
class InjuryAdapter:
    birth_recipe_count: int
    current_recipe_count: int
    lost_builder: int
    capacity: int
    fronts: list[Front]

    def nominate(self, *, use_current_extent: bool) -> bool:
        extent = self.current_recipe_count if use_current_extent else self.birth_recipe_count
        if not (0 <= self.lost_builder < extent):
            return False
        for row in self.fronts:
            if row.injury:
                return True
        replacement = None
        for i, row in enumerate(self.fronts):
            if row.source == 7:
                replacement = i
                break
        if replacement is None:
            for i, row in enumerate(self.fronts):
                if not row.injury and (replacement is None or
                    (row.support + row.credit, i) <
                    (self.fronts[replacement].support + self.fronts[replacement].credit, replacement)):
                    replacement = i
        if replacement is None:
            if len(self.fronts) >= self.capacity:
                return False
            self.fronts.append(Front(7, self.lost_builder, 0, 0, True))
        else:
            self.fronts[replacement] = Front(7, self.lost_builder, 0, 0, True)
        return True


def learned_access_rows(o: ReferenceOrganismV2):
    return {k: copy.deepcopy(v) for k, v in o.language._lexeme_positive.items() if k[0] == NEW}


def lesion_access(o: ReferenceOrganismV2):
    rows = learned_access_rows(o)
    for key in rows:
        o.language._lexeme_positive.pop(key, None)
    return rows


def rematerialize_access(o: ReferenceOrganismV2, rows):
    for key, value in rows.items():
        o.language._lexeme_positive[key] = copy.deepcopy(value)


def usable_aliases(o: ReferenceOrganismV2, profiles, source_base: int) -> tuple[int, tuple[str, ...]]:
    outputs = []
    usable = 0
    for i, profile in enumerate(profiles):
        _sid, payload = outward(o, profile, source_base + i)
        text = bytes(payload).decode(errors="replace")
        outputs.append(text)
        usable += int(payload == tuple(profile[7].encode()))
    return usable, tuple(outputs)


def main():
    started = time.perf_counter()
    spec = PopulationSpecV1(32768, 2, 4, 42, 8)
    base = prepare(spec)
    # Three target aliases plus one independently learned remote control.
    for i, profile in enumerate(PROFILES[:4]):
        command(base, profile, 12000 + i, 22000 + i)
    checkpoint = copy.deepcopy(base.checkpoint())
    target_profiles = PROFILES[:3]
    remote_profile = PROFILES[3]

    intact = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    intact_count, intact_outputs = usable_aliases(intact, target_profiles, 31000)

    lesioned = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    target_keys = tuple((NEW, tuple(profile[5].encode())) for profile in target_profiles)
    saved = {key: copy.deepcopy(lesioned.language._lexeme_positive[key]) for key in target_keys}
    for key in target_keys:
        lesioned.language._lexeme_positive.pop(key, None)
    lesion_count, lesion_outputs = usable_aliases(lesioned, target_profiles, 32000)

    remote = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    remote_key = (NEW, tuple(remote_profile[5].encode()))
    remote.language._lexeme_positive.pop(remote_key, None)
    remote_count, remote_outputs = usable_aliases(remote, target_profiles, 32500)

    saturated = [Front(i, i % 5, (i % 7) - 3, (i % 5) - 2, False) for i in range(24)]
    # LM2 learned cells occur after the birth extent; 6 is representative of the
    # observed production focal Recipe cell while remaining an opaque structural id.
    birth_policy = InjuryAdapter(5, 8, 6, 24, copy.deepcopy(saturated))
    current_policy = InjuryAdapter(5, 8, 6, 24, copy.deepcopy(saturated))
    birth_nominated = birth_policy.nominate(use_current_extent=False)
    current_nominated = current_policy.nominate(use_current_extent=True)

    repaired = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for key in target_keys:
        repaired.language._lexeme_positive.pop(key, None)
    if current_nominated:
        for key, value in saved.items():
            repaired.language._lexeme_positive[key] = copy.deepcopy(value)
    repaired_count, repaired_outputs = usable_aliases(repaired, target_profiles, 33000)

    # Contradiction is ordinary consequence-driven revision, not observer mutation.
    contradicted = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    before_outcomes = tuple(outcome(contradicted, p) for p in target_profiles)
    _assertion, negative_action, negative_learning, _baseline = command(
        contradicted, target_profiles[0], 34000, 44000, effect=-1, independent=True)
    after_negative = tuple(outcome(contradicted, p) for p in target_profiles)
    contradiction_checkpoint = copy.deepcopy(contradicted.checkpoint())
    contradicted_counts = []
    contradicted_outputs = []
    for i, profile in enumerate(target_profiles):
        branch = ReferenceOrganismV2.restore(copy.deepcopy(contradiction_checkpoint))
        # Ordinary external resolution of the prior motor episode; no language
        # state is reset and no surface/answer is supplied.
        branch.contact(CONTACT_WORLD_STATE, (NEW,), 34500 + i, True, True)
        _sid, payload = outward(branch, profile, 35000 + i)
        contradicted_outputs.append(bytes(payload).decode(errors="replace"))
        contradicted_counts.append(int(payload == tuple(profile[7].encode())))

    reacquire = ReferenceOrganismV2.restore(copy.deepcopy(contradiction_checkpoint))
    _assertion, positive_action, positive_learning, _baseline = command(
        reacquire, target_profiles[0], 36000, 46000, effect=1, independent=True)
    _sid, reacquired_payload = outward(reacquire, target_profiles[0], 37000)
    after_relearn = outcome(reacquire, target_profiles[0])

    checks = {
        "intact_three_aliases_usable": intact_count == 3,
        "complete_measured_access_lesion_collapses_all": lesion_count == 0,
        "matched_remote_access_lesion_spares_targets": remote_count == 3 and remote_outputs == intact_outputs,
        "birth_extent_rejects_postbirth_builder": not birth_nominated,
        "current_extent_accepts_postbirth_builder": current_nominated,
        "saturated_bank_capacity_unchanged": len(current_policy.fronts) == 24,
        "injury_preempts_only_noninjury_work": sum(row.injury for row in current_policy.fronts) == 1,
        "current_extent_repairs_language_phenotype": repaired_count == 3,
        "repair_does_not_author_surface": repaired_outputs == intact_outputs,
        "contradiction_is_real_causal_action": negative_action is not None and negative_learning.get("lexeme_settlement") is None,
        "contradiction_revises_only_reactivated_alias": (
            before_outcomes == ((1, 0), (1, 0), (1, 0))
            and after_negative[0] == (1, 1)
            and after_negative[1:] == before_outcomes[1:]
            and contradicted_counts[0] == 0
            and contradicted_counts[1:] == [1, 1]
        ),
        "later_authenticated_evidence_reacquires_alias": (
            positive_action is not None
            and positive_learning.get("lexeme_settlement") == tuple(target_profiles[0][5].encode())
            and after_relearn == (2, 1)
            and reacquired_payload == tuple(target_profiles[0][7].encode())
        ),
        "no_prompt_answer_api": all(not hasattr(repaired, name) for name in ("prompt", "answer", "speak", "enqueue_goal")),
    }
    result = {
        "schema": "0x1.reference-lm2-injury-adapter.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "phenotype": {
            "intact_aliases": intact_count,
            "coalition_lesioned_aliases": lesion_count,
            "remote_sham_aliases": remote_count,
            "repaired_aliases": repaired_count,
            "delta_vs_coalition_lesion": repaired_count - lesion_count,
            "post_contradiction_aliases": tuple(contradicted_counts),
            "reacquired_alias0": int(reacquired_payload == tuple(target_profiles[0][7].encode())),
        },
        "policy": {
            "birth_extent_nominated": birth_nominated,
            "current_extent_nominated": current_nominated,
            "birth_recipe_count": 5,
            "current_recipe_count": 8,
            "lost_builder": 6,
            "front_capacity": 24,
        },
        "runtime_llm": False,
        "graph_flip": False,
        "claim": "REFERENCE_ONLY_LM2_PHYSICAL_ADMISSIBILITY_TOURNAMENT",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_LM2_INJURY_ADAPTER " + ("GREEN" if result["pass"] else "RED") +
        f" intact={intact_count}/3 lesion={lesion_count}/3 repaired={repaired_count}/3 "
        f"birth_extent={int(birth_nominated)} current_extent={int(current_nominated)}"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
