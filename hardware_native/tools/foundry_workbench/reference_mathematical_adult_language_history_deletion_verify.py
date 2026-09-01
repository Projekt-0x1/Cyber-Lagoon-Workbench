#!/usr/bin/env python3
"""Fast A/B falsifier for deleting lifetime language-event shadow state.

The learned relation banks already retain source-qualified lexical/template/span
state.  `_history` records only a coarse duplicate event tuple and is not read by
future language transitions.  The compact state must therefore survive source
withdrawal/restoration and held-out realization while a causal template deletion
remains distinguishable.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1  # noqa: E402

CAREFUL, QUIET = 101, 102
ENGINEER, TECHNICIAN = 201, 202
TESTS, INSPECTS = 301, 302
SENSOR, VALVE = 401, 402
CLAUSE, SPAN = 9001, 9101
EX1 = (CAREFUL, ENGINEER, TESTS, SENSOR)
EX2 = (QUIET, TECHNICIAN, INSPECTS, VALVE)
HELD = (QUIET, ENGINEER, TESTS, VALVE)
MAPPING = {
    CAREFUL: "careful", QUIET: "quiet", ENGINEER: "engineer",
    TECHNICIAN: "technician", TESTS: "tests", INSPECTS: "inspects",
    SENSOR: "sensor", VALVE: "valve",
}


def u(text):
    return tuple(text.encode("utf-8"))


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def build():
    e = LearnedSurfaceEcologyV1()
    for feature, text in MAPPING.items():
        e.observe_naming(feature, u(text), 1000 + feature)
        e.observe_naming(feature, u(text), 2000 + feature)
    e.observe_construction(CLAUSE, EX1, u("the careful engineer tests the sensor."), 3001)
    e.observe_construction(CLAUSE, EX2, u("the quiet technician inspects the valve."), 3002)
    c1 = e.realize(CLAUSE, EX1); c2 = e.realize(CLAUSE, EX2); held = e.realize(CLAUSE, HELD)
    if c1 is None or c2 is None or held is None:
        raise RuntimeError("language-history-deletion:setup")
    e.observe_span(SPAN, (c1, c2, held), c1 + u(" After that, ") + c2 + u(" ") + held, 5001)
    e.observe_span(SPAN, (c2, held, c1), c2 + u(" After that, ") + held + u(" ") + c1, 5002)
    # Inflate only the obsolete coordinate: same-source repeats do not add
    # independent support to the source-indexed learned relations.
    for _ in range(256):
        e.observe_naming(CAREFUL, u("careful"), 1000 + CAREFUL)
    for _ in range(64):
        e.observe_construction(CLAUSE, EX1, u("the careful engineer tests the sensor."), 3001)
    return e


def future_trace(checkpoint):
    try:
        e = LearnedSurfaceEcologyV1.restore(copy.deepcopy(checkpoint))
    except Exception as exc:
        return {"restore_error": type(exc).__name__, "message": str(exc)}
    before = e.realize(CLAUSE, HELD)
    paragraph_before = None
    if before is not None:
        c1 = e.realize(CLAUSE, EX1); c2 = e.realize(CLAUSE, EX2)
        if c1 is not None and c2 is not None:
            paragraph_before = e.realize_span(SPAN, (before, c1, c2))
    e.withdraw_source(3002)
    withdrawn = e.realize(CLAUSE, HELD)
    e.restore_source(3002)
    restored = e.realize(CLAUSE, HELD)
    return {
        "heldout_before": before,
        "paragraph_before": paragraph_before,
        "heldout_after_source_withdrawal": withdrawn,
        "heldout_after_source_restore": restored,
        "template_support": None if e.template(CLAUSE, 4) is None else e.template(CLAUSE, 4).support,
        "span_support": None if e.span_template(SPAN, 3) is None else e.span_template(SPAN, 3).support,
    }


def main():
    started = time.perf_counter()
    e = build()
    current = e.checkpoint()

    if "history" in current:
        legacy = copy.deepcopy(current)
        compact = copy.deepcopy(current)
        compact.pop("history", None)
    else:
        compact = copy.deepcopy(current)
        legacy = copy.deepcopy(current)
        legacy["schema"] = 2
        legacy["history"] = [[1000 + CAREFUL, CAREFUL, "lexeme"] for _ in range(320)]

    legacy_trace = future_trace(legacy)
    compact_trace = future_trace(compact)

    causal_cut = copy.deepcopy(compact)
    causal_cut["templates"] = []
    causal_cut_trace = future_trace(causal_cut)

    repeat_no_growth = False
    repeat_before = repeat_after = 0
    try:
        repeated = LearnedSurfaceEcologyV1.restore(copy.deepcopy(compact))
        repeat_before = wire_bytes(repeated.checkpoint())
        for _ in range(128):
            repeated.observe_naming(CAREFUL, u("careful"), 1000 + CAREFUL)
        repeat_after = wire_bytes(repeated.checkpoint())
        repeat_no_growth = repeat_after == repeat_before
    except Exception:
        pass

    legacy_bytes = wire_bytes(legacy); compact_bytes = wire_bytes(compact)
    checks = {
        "current_checkpoint_excludes_lifetime_history": "history" not in current,
        "live_ecology_has_no_lifetime_history_bank": not hasattr(e, "_history"),
        "legacy_and_compact_future_language_are_equivalent": legacy_trace == compact_trace and "restore_error" not in compact_trace,
        "source_withdrawal_still_changes_future_language": compact_trace.get("heldout_before") is not None and compact_trace.get("heldout_after_source_withdrawal") is None,
        "source_restore_recovers_heldout_language": compact_trace.get("heldout_after_source_restore") == compact_trace.get("heldout_before"),
        "causal_template_negative_control_differs": causal_cut_trace != compact_trace,
        "legacy_shadow_state_is_larger": legacy_bytes > compact_bytes,
        "same_source_repetition_does_not_grow_resident_language_state": repeat_no_growth,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-language-history-deletion.v1",
        "pass": not failed,
        "reference_only": True,
        "deleted_state": "LearnedSurfaceEcologyV1._history",
        "state_role": "OBSERVER_SHADOW_NOT_FUTURE_CAUSAL_STATE",
        "equivalence": "HELDOUT_SURFACE_PLUS_SOURCE_INTERVENTION_TRACE",
        "legacy_checkpoint_bytes": legacy_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved_in_fixture": legacy_bytes - compact_bytes,
        "same_source_repeat_checkpoint_growth_bytes": repeat_after - repeat_before,
        "negative_control": "delete learned construction templates",
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_LANGUAGE_HISTORY_DELETION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True, default=list))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
