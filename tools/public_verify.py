#!/usr/bin/env python3
"""Verify the public workshop boundary without requiring a GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "build",
    "adult",
    "bench",
    "verify",
    "README.md",
    "CONTRIBUTING.md",
    "experiments/current_recipe.ir",
    "tools/public_adult.py",
    "tools/public_adult_smoke.py",
    "tools/public_bench.py",
    "tools/public_materialize_adult.py",
    "hardware_native/tools/direct_recipe_ir_lab.cu",
    "hardware_native/tools/direct_adult_sitdown.cu",
    "tools/public_direct_adult_gateway.py",
    "hardware_native/tools/foundry_workbench/reference_language_mastery_claude_gateway_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_manifest() -> None:
    path = ROOT / "EXPORT_MANIFEST.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "cyber-lagoon.public-workbench-export.v1", "verify:manifest-schema")
    for row in manifest.get("entries", ()):
        candidate = ROOT / row["path"]
        require(candidate.is_file(), f"verify:manifest-missing:{row['path']}")
        require(candidate.stat().st_size == row["bytes"], f"verify:manifest-size:{row['path']}")
        require(sha256(candidate) == row["sha256"], f"verify:manifest-hash:{row['path']}")


def main() -> int:
    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"verify:missing:{relative}")
    for relative in ("build", "adult", "bench", "verify"):
        require((ROOT / relative).stat().st_mode & 0o111 != 0, f"verify:not-executable:{relative}")

    build = (ROOT / "build").read_text(encoding="utf-8")
    for required in (
        "umask 077",
        '[[ -L "$directory" ]]',
        "chmod 700 .state",
        "mktemp -d .state/.direct-birth.XXXXXX",
        '[[ -L .state/direct-adult.xcb ]]',
    ):
        require(required in build, f"verify:build-hardening:{required}")

    bench = (ROOT / "tools" / "public_bench.py").read_text(encoding="utf-8").lower()
    for forbidden in ("cmake", "nvcc", "build_concurrency_guard", "public_materialize_adult"):
        require(forbidden not in bench, f"verify:bench-crosses-build-boundary:{forbidden}")

    materialize = (ROOT / "tools" / "public_materialize_adult.py").read_text(encoding="utf-8")
    for required in (
        "tempfile.NamedTemporaryFile",
        "os.fsync",
        "os.replace",
        "public-adult:state-symlink-refused",
        "os.chmod(state_dir, 0o700)",
    ):
        require(required in materialize, f"verify:state-hardening:{required}")

    adult = (ROOT / "tools" / "public_adult.py").read_text(encoding="utf-8")
    require("os.environ.copy(" not in adult, "verify:adult-inherits-ambient-environment")
    for required in (
        "GATEWAY_BOOTSTRAP",
        "stdin=subprocess.PIPE",
        "secrets.token_hex(32)",
        "cwd=body_dir",
        "CLAUDE_CONFIG_DIR",
        "NO_PROXY",
        "public-adult:repo-local-claude-refused",
    ):
        require(required in adult, f"verify:adult-body-isolation:{required}")

    direct_lab = (ROOT / "hardware_native" / "tools" / "direct_recipe_ir_lab.cu").read_text(encoding="utf-8")
    for required in (
        "DirectPreinstantiatedRecipeFamily",
        "semantic_authority=0",
        "current_thought_authority=0",
        "claim_scope=experiment_only",
    ):
        require(required in direct_lab, f"verify:direct-lab-boundary:{required}")

    gateway = (
        ROOT
        / "hardware_native"
        / "tools"
        / "foundry_workbench"
        / "reference_language_mastery_claude_gateway_v1.py"
    ).read_text(encoding="utf-8")
    require(
        "CLAUDE_SILENCE_FRAME if claude_source else ''" in gateway,
        "verify:claude-code-silence-boundary",
    )
    direct_gateway = (ROOT / "tools" / "public_direct_adult_gateway.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "body:nonappend_transcript",
        "body:duplicate_boundary",
        '"usage": {"input_tokens": 0, "output_tokens": 0}',
        "SILENCE_FRAME",
    ):
        require(required in direct_gateway, f"verify:direct-gateway-boundary:{required}")

    scripts = (
        ROOT / "tools" / "public_adult.py",
        ROOT / "tools" / "public_adult_smoke.py",
        ROOT / "tools" / "public_bench.py",
        ROOT / "tools" / "public_direct_adult_gateway.py",
        ROOT / "tools" / "public_materialize_adult.py",
    )
    subprocess.run([sys.executable, "-m", "py_compile", *map(str, scripts)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "public_bench.py"), "--no-ir"], cwd=ROOT, check=True)
    require((ROOT / ".state" / "adult.json").is_file(), "verify:run-build-first")
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "public_adult.py"), "--print-backend"],
        cwd=ROOT,
        check=True,
    )
    verify_manifest()
    print("CYBER_LAGOON_PUBLIC_VERIFY status=PASS build_boundary=closed adult=continuing hardened=1")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
