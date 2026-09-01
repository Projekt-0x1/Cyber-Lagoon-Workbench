#!/usr/bin/env python3
"""Fast LM2 contradiction/relearning discriminator over the continuing reference Adult.

Research-grounded reference-only tournament.  Three distinct learned surface forms
access one nonlinguistic feature.  An independently returned contradiction for one
surface must reopen only that lived access path, leave the other two usable, and
later authenticated confirmation must restore the contradicted path.  No semantic
alias table, expected answer, transformer/backprop, token objective, or host-selected
revision target is resident state.
"""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import ReferenceOrganismV2, MotorActionV2, CONTACT_PARTNER_CONTEXT
from reference_population_v1 import PopulationSpecV1
from reference_organism_fast_mapping_verify import (
    NEW, PROFILES, command, outward, outcome, prepare, u,
)


def main():
    started = time.perf_counter()
    spec = PopulationSpecV1(32768, 2, 4, 42, 8)
    adult = prepare(spec)

    # One continuing organism earns three distinct access paths to the same
    # nonlinguistic NEW feature through ordinary action -> independent consequence.
    learned = []
    for i, profile in enumerate(PROFILES[:3]):
        assertion, action, settlement, baseline = command(
            adult, profile, 12000 + i, 22000 + i, 1, True)
        learned.append((assertion, action, settlement, baseline))

    checkpoint = copy.deepcopy(adult.checkpoint())
    pre_outputs = []
    for i, profile in enumerate(PROFILES[:3]):
        _sid, payload = outward(adult, profile, 31000 + i)
        pre_outputs.append(payload)

    # Contradict only alias 0 through the ordinary lived causal path.  The
    # evaluator chooses the experimental episode, not the resident revision.
    contradicted = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    before = [outcome(contradicted, p) for p in PROFILES[:3]]
    _a, negative_action, negative_learning, _baseline = command(
        contradicted, PROFILES[0], 13000, 23000, -1, True)
    after_negative = [outcome(contradicted, p) for p in PROFILES[:3]]

    negative_checkpoint = copy.deepcopy(contradicted.checkpoint())
    post_negative_access = []
    for i, profile in enumerate(PROFILES[:3]):
        branch = ReferenceOrganismV2.restore(copy.deepcopy(negative_checkpoint))
        branch.contact(CONTACT_PARTNER_CONTEXT, (1, 7, profile[1]), 32000 + i, True, True)
        realized = branch._realized_lexeme(NEW)
        post_negative_access.append(None if realized is None else realized[1])

    # Reacquisition is the same ordinary causal path, not a special repair API.
    _a, reconfirm_action, reconfirm_learning, _baseline = command(
        contradicted, PROFILES[0], 14000, 24000, 1, True)
    after_reconfirm = [outcome(contradicted, p) for p in PROFILES[:3]]
    contradicted.contact(CONTACT_PARTNER_CONTEXT, (1, 7, PROFILES[0][1]), 33000, True, True)
    restored = contradicted._realized_lexeme(NEW)
    restored_output = None if restored is None else restored[1]

    replay = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    replay_before = [outcome(replay, p) for p in PROFILES[:3]]

    checks = {
        "three_distinct_aliases_initially_usable": all(
            payload == u(profile[7])
            for payload, profile in zip(pre_outputs, PROFILES[:3])
        ),
        "three_aliases_are_independently_consequence_confirmed": before == [(1, 0)] * 3,
        "contradiction_comes_from_real_resident_action": (
            isinstance(negative_action, MotorActionV2)
            and bool(negative_action.lexical_occurrences)
        ),
        "contradicted_alias_reopens_without_history_erasure": (
            negative_learning.get("lexeme_settlement") is None
            and after_negative[0] == (1, 1)
            and post_negative_access[0] is None
        ),
        "other_aliases_keep_shared_cognitive_access": (
            after_negative[1:] == [(1, 0), (1, 0)]
            and post_negative_access[1] == u(PROFILES[1][5])
            and post_negative_access[2] == u(PROFILES[2][5])
        ),
        "later_independent_confirmation_reacquires_alias": (
            isinstance(reconfirm_action, MotorActionV2)
            and reconfirm_learning.get("lexeme_settlement") == u(PROFILES[0][5])
            and after_reconfirm[0] == (2, 1)
            and restored_output == u(PROFILES[0][5])
        ),
        "reacquisition_does_not_touch_other_alias_evidence": after_reconfirm[1:] == after_negative[1:],
        "checkpoint_replay_preserves_precontradiction_state": replay_before == before,
        "no_language_specific_negative_table_added": not any(
            hasattr(contradicted.language, name)
            for name in ("alias_negative", "semantic_alias", "contradicted_words")
        ),
        "no_prompt_answer_runtime": all(
            not hasattr(contradicted, name)
            for name in ("prompt", "answer", "enqueue_goal")
        ),
    }
    result = {
        "schema": "0x1.reference-lm2-contradiction-adapter.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "evidence": {
            "before": before,
            "after_negative": after_negative,
            "after_reconfirm": after_reconfirm,
            "negative_surface_usable": post_negative_access[0] == u(PROFILES[0][5]),
            "uncontradicted_survivors": sum(
                post_negative_access[i] == u(PROFILES[i][5]) for i in (1, 2)
            ),
        },
        "runtime_llm": False,
        "transformer": False,
        "backprop": False,
        "token_objective": False,
        "graph_flip": False,
        "claim": "REFERENCE_ONLY_LM2_SOURCE_CONDITIONED_CONTRADICTION_AND_REACQUISITION",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_LM2_CONTRADICTION_ADAPTER "
        + ("GREEN" if result["pass"] else "RED")
        + f" before={before} negative={after_negative} reconfirm={after_reconfirm}"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
