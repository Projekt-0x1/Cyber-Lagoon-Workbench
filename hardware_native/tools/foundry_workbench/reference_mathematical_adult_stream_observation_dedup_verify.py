#!/usr/bin/env python3
"""Fast falsifier for duplicate raw stream-observation state in language learning.

Exact duplicate `(surface, source)` contacts add neither an independent source nor a
new boundary context to the undelimited segmentation law. Distinct episodes from the
same source can still add boundary diversity and therefore remain causal evidence.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1  # noqa: E402

FEATURE = 9101
S1, S2, S3 = 7001, 7002, 7003
A = "akmipzo"
B = "bxmipyc"
C = "cwmipvd"
D = "qkmipzt"
E = "rkmipzv"
REPEATS = 256


def u(text):
    return tuple(text.encode())


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def stream_rows(state):
    return tuple(
        (int(row["feature"]), tuple(map(int, row["raw"])), int(row["source"]))
        for row in state.get("streams", ())
    )


def future_segment(checkpoint):
    e = LearnedSurfaceEcologyV1.restore(copy.deepcopy(checkpoint))
    learned = e.observe_stream_naming(FEATURE, u(E), S3)
    return {
        "learned": None if learned is None else tuple(learned),
        "lexeme": e.lexeme(FEATURE),
        "touches": int(e.last_segment_touches),
        "checkpoint": e.checkpoint(),
    }


def main():
    started = time.perf_counter()
    e = LearnedSurfaceEcologyV1()

    # Three distinct S1 contexts are intentionally causal for the boundary-diversity
    # rule; S2/S3 use the same boundary context. Exact duplicates of A are not.
    for raw in (A, B, C):
        e.observe_stream_naming(FEATURE, u(raw), S1)
    e.observe_stream_naming(FEATURE, u(D), S2)
    before_repeat = e.checkpoint()
    before_bytes = wire_bytes(before_repeat)

    for _ in range(REPEATS):
        e.observe_stream_naming(FEATURE, u(A), S1)
    after_repeat = e.checkpoint()
    after_bytes = wire_bytes(after_repeat)
    rows_after = stream_rows(after_repeat)

    future = future_segment(after_repeat)

    # Legacy donor reconstructs the old duplicate representation. Current restore
    # must canonicalize it to the same compact state before future contact.
    legacy = copy.deepcopy(after_repeat)
    legacy["schema"] = min(int(legacy.get("schema", 0)), 3) or 3
    legacy["streams"].extend(
        {"feature": FEATURE, "raw": list(u(A)), "source": S1}
        for _ in range(REPEATS)
    )
    restored_legacy = LearnedSurfaceEcologyV1.restore(copy.deepcopy(legacy))
    restored_compact = LearnedSurfaceEcologyV1.restore(copy.deepcopy(after_repeat))
    legacy_future = future_segment(legacy)

    # Negative control: B and C are distinct same-source episodes. Removing them
    # leaves only two distinct left/right contexts, so the later third source must
    # no longer establish `mip`.
    causal_cut = copy.deepcopy(after_repeat)
    causal_cut["streams"] = [
        row for row in causal_cut["streams"]
        if not (int(row["source"]) == S1 and tuple(row["raw"]) in (u(B), u(C)))
    ]
    cut_future = future_segment(causal_cut)

    checks = {
        "current_checkpoint_has_no_exact_duplicate_stream_rows": len(rows_after) == len(set(rows_after)),
        "exact_duplicate_repetition_does_not_grow_checkpoint": after_bytes == before_bytes,
        "exact_duplicate_repetition_does_not_grow_stream_rows": len(rows_after) == len(stream_rows(before_repeat)),
        "compact_future_learns_expected_undelimited_chunk": future["learned"] == u("mip") and future["lexeme"] == u("mip"),
        "legacy_duplicates_restore_to_same_causal_state": restored_legacy.digest() == restored_compact.digest(),
        "legacy_duplicates_have_identical_future": legacy_future == future,
        "distinct_same_source_contexts_remain_causal": cut_future["learned"] != future["learned"] and cut_future["lexeme"] != future["lexeme"],
        "future_segmentation_work_is_bounded_by_unique_rows": future["touches"] < 128,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-stream-observation-dedup.v1",
        "pass": not failed,
        "reference_only": True,
        "deleted_representation": "EXACT_DUPLICATE_STREAM_SURFACE_SOURCE_ROWS",
        "retained_evidence": "DISTINCT_SURFACE_SOURCE_EPISODES",
        "repeat_count": REPEATS,
        "checkpoint_bytes_before_repeats": before_bytes,
        "checkpoint_bytes_after_repeats": after_bytes,
        "checkpoint_growth_bytes": after_bytes - before_bytes,
        "stream_rows_before_repeats": len(stream_rows(before_repeat)),
        "stream_rows_after_repeats": len(rows_after),
        "future_segment_touches": future["touches"],
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_STREAM_OBSERVATION_DEDUP_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True, default=list))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
