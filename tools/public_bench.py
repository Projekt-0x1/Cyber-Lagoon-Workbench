#!/usr/bin/env python3
"""Fast Workbench loop: falsify first, then reuse already-built Direct evaluators."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = (sys.executable, "-I")
WORKBENCH = ROOT / "hardware_native" / "tools" / "foundry_workbench"
DEFAULT_REFERENCE = ROOT / "tools" / "public_adult_smoke.py"
DEFAULT_IR = ROOT / "experiments" / "current_recipe.ir"
DIRECT_IR_LAB = ROOT / ".build" / "direct" / "direct_recipe_ir_lab"
DIRECT_SITDOWN = ROOT / ".build" / "direct" / "direct_adult_sitdown"
DIRECT_CHECKPOINT = ROOT / ".state" / "direct-adult.xcb"
DIRECT_ADULT_LAB = ROOT / "tools" / "direct_adult_lab.py"


def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def reference_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    local = ROOT / candidate
    if local.is_file():
        return local.resolve()
    return (WORKBENCH / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE.relative_to(ROOT)), help="public smoke, focused Foundry verifier filename, or path")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR, help="numeric Resident Recipe IR candidate data")
    parser.add_argument("--no-ir", action="store_true")
    parser.add_argument("--contact", type=Path, help="raw body contact for a disposable branch of the retained Direct Adult")
    parser.add_argument("--require-direct", action="store_true")
    args = parser.parse_args()

    failed = False
    if not args.no_reference:
        script = reference_path(args.reference)
        if not script.is_file():
            raise FileNotFoundError(script)
        failed = run([*PYTHON, str(script)]) != 0

    direct_seen = False
    direct_skipped = False
    if not args.no_ir:
        if DIRECT_IR_LAB.is_file():
            direct_seen = True
            code = run([str(DIRECT_IR_LAB), str(args.ir.resolve())])
            direct_skipped = code == 77
            failed = failed or code not in (0, 77)
        else:
            print("DIRECT_RECIPE_IR_FASTPATH unavailable; run ./build once", flush=True)

    if args.contact:
        if not (DIRECT_SITDOWN.is_file() and DIRECT_CHECKPOINT.is_file()):
            raise RuntimeError("public-bench:retained-direct-adult-unavailable; run ./build once")
        direct_seen = True
        code = run(
            [
                *PYTHON,
                str(DIRECT_ADULT_LAB),
                "--checkpoint",
                str(DIRECT_CHECKPOINT),
                "--artifact",
                str(DIRECT_SITDOWN),
                "--input",
                str(args.contact.resolve()),
            ]
        )
        failed = failed or code != 0

    if args.require_direct and (not direct_seen or direct_skipped):
        failed = True
    print(
        f"CYBER_LAGOON_BENCH status={'RED' if failed else 'PASS'} "
        f"reference={0 if args.no_reference else 1} direct_seen={int(direct_seen)} direct_skipped={int(direct_skipped)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
