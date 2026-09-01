#!/usr/bin/env python3
"""Cheap integrity smoke for the retained public Workbench Adult checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / "hardware_native" / "tools" / "foundry_workbench"
sys.path.insert(0, str(WORKBENCH))

from reference_life_function_curriculum_v1 import (  # noqa: E402
    ReferenceLifeFunctionRuntimeV2,
    canonical_species_program_v2,
)


def main() -> int:
    checkpoint_path = ROOT / ".state" / "adult.json"
    provenance_path = ROOT / ".state" / "adult.provenance.json"
    if not checkpoint_path.is_file() or not provenance_path.is_file():
        raise RuntimeError("public-adult-smoke:run-build-first")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    runtime = ReferenceLifeFunctionRuntimeV2.restore(canonical_species_program_v2(), checkpoint)
    mark = str(provenance["final_mark"])
    checks = {
        "checkpoint_roundtrip_canonical": json.dumps(runtime.checkpoint(), sort_keys=True, separators=(",", ":")) == json.dumps(checkpoint, sort_keys=True, separators=(",", ":")),
        "cursor_matches_provenance": runtime.cursor == int(provenance["final_cursor"]),
        "final_mark_names_current_cursor": runtime.marks.get(mark) == runtime.cursor,
        "history_root_nonempty": bool(runtime.history_root()),
        "adult_digest_nonempty": bool(runtime.adult.digest()),
        "tail_status_declared": provenance.get("experimental_tail_status") in {"FULL", "RED"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(
        json.dumps(
            {
                "status": "PASS" if not failed else "RED",
                "contract": "CYBER_LAGOON_PUBLIC_ADULT_SMOKE",
                "final_mark": mark,
                "cursor": runtime.cursor,
                "experimental_tail_status": provenance.get("experimental_tail_status"),
                "checks": checks,
                "failed": failed,
            },
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
