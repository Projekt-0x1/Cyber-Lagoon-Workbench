#!/usr/bin/env python3
"""Materialize the last completed canonical Workbench Adult checkpoint mark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / "hardware_native" / "tools" / "foundry_workbench"
sys.path.insert(0, str(WORKBENCH))

from reference_life_function_curriculum_v1 import (  # noqa: E402
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
    source_semantics_root_v2,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".state")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--strict-tail", action="store_true", help="fail if the current experimental curriculum tail refuses")
    args = parser.parse_args()

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_dir / "adult.json"
    provenance_path = state_dir / "adult.provenance.json"
    if checkpoint_path.exists() and provenance_path.exists() and not args.reset:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "PUBLIC_ADULT_EXISTS", **provenance}, sort_keys=True))
        return 0
    checkpoint_path.unlink(missing_ok=True)
    provenance_path.unlink(missing_ok=True)

    species = canonical_species_program_v2()
    curriculum = canonical_life_function_curriculum_v2()
    runtime = ReferenceLifeFunctionRuntimeV2(species)
    last_checkpoint = None
    last_mark = None
    last_cursor = 0
    failure = None

    for event in curriculum.events:
        try:
            runtime.apply(event)
        except Exception as error:  # deliberate frontier receipt, not silent recovery
            failure = {
                "sequence": int(event.sequence),
                "lane": str(event.lane),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            break
        if event.lane == "checkpoint_mark":
            last_mark = str(event.payload[0])
            last_cursor = int(runtime.cursor)
            last_checkpoint = runtime.checkpoint()

    if last_checkpoint is None or last_mark is None:
        raise RuntimeError("public-adult:no-completed-checkpoint-mark")
    if failure is not None and args.strict_tail:
        raise RuntimeError(f"public-adult:experimental-tail-red:{failure['sequence']}:{failure['lane']}:{failure['error_type']}")

    temporary = checkpoint_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(last_checkpoint, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(checkpoint_path)

    provenance = {
        "schema": "cyber-lagoon.public-adult-state.v2",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest(last_checkpoint),
        "species_root": species.root(),
        "curriculum_root": curriculum.root(),
        "source_semantics_root": source_semantics_root_v2(),
        "curriculum_events": len(curriculum.events),
        "applied_events": last_cursor,
        "final_mark": last_mark,
        "final_cursor": last_cursor,
        "experimental_tail_status": "FULL" if failure is None else "RED",
        "experimental_tail_failure": failure,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PUBLIC_ADULT_MATERIALIZED", **provenance}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
