#!/usr/bin/env python3
"""Fast Workbench loop: falsify first, then reuse already-built Direct evaluators."""
from __future__ import annotations

import argparse
import os
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
DIRECT_ENV_ALLOW = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_MAXSIZE",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
)


def run(command: list[str], env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def direct_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in DIRECT_ENV_ALLOW if name in os.environ}


def symlink_free_local(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return False
    current = ROOT
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def trusted_local_file(path: Path, executable: bool = False) -> bool:
    return (
        symlink_free_local(path)
        and path.is_file()
        and not path.is_symlink()
        and (not executable or os.access(path, os.X_OK))
    )


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
        if not symlink_free_local(DIRECT_IR_LAB):
            raise RuntimeError("public-bench:direct-artifact-refused")
        if trusted_local_file(DIRECT_IR_LAB, executable=True):
            direct_seen = True
            code = run(
                [str(DIRECT_IR_LAB), str(args.ir.resolve())],
                env=direct_environment(),
            )
            direct_skipped = code == 77
            failed = failed or code not in (0, 77)
        elif DIRECT_IR_LAB.exists():
            raise RuntimeError("public-bench:direct-artifact-refused")
        else:
            print("DIRECT_RECIPE_IR_FASTPATH unavailable; run ./build once", flush=True)

    if args.contact:
        for path in (DIRECT_SITDOWN, DIRECT_CHECKPOINT, DIRECT_ADULT_LAB):
            if not symlink_free_local(path):
                raise RuntimeError("public-bench:direct-artifact-refused")
        if not (
            trusted_local_file(DIRECT_SITDOWN, executable=True)
            and trusted_local_file(DIRECT_CHECKPOINT)
            and trusted_local_file(DIRECT_ADULT_LAB)
        ):
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
                "--timeout",
                "8",
            ],
            env=direct_environment(),
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
