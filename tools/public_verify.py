#!/usr/bin/env python3
"""Verify the public workshop boundary without requiring a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

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
MANIFEST_NAME = "EXPORT_MANIFEST.json"
MANIFEST_V1 = "cyber-lagoon.public-workbench-export.v1"
MANIFEST_V2 = "cyber-lagoon.public-workbench-export.v2"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git(*args: str, check: bool = True) -> bytes:
    try:
        run = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=check,
            capture_output=True,
        )
    except OSError as error:
        raise RuntimeError("verify:git-unavailable") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"verify:git-failed:{args[0]}") from error
    return run.stdout


def tracked_paths(required: bool = False) -> list[str] | None:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ) if shutil.which("git") else None
    if probe is None or probe.returncode != 0 or probe.stdout.strip() != "true":
        if required:
            raise RuntimeError("verify:git-checkout-required")
        return None
    paths = sorted(
        os.fsdecode(raw)
        for raw in git("ls-files", "-z").split(b"\0")
        if raw and os.fsdecode(raw) != MANIFEST_NAME
    )
    require(len(paths) == len(set(paths)), "verify:tracked-files-duplicate")
    return paths


def validate_v1(manifest: object) -> dict:
    require(isinstance(manifest, dict), "verify:manifest-object")
    require(manifest.get("schema") == MANIFEST_V1, "verify:manifest-v1-schema")
    entries = manifest.get("entries")
    require(isinstance(entries, list), "verify:manifest-entries")
    paths: list[str] = []
    total = 0
    for row in entries:
        require(isinstance(row, dict), "verify:manifest-row")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        require(isinstance(relative, str), "verify:manifest-path")
        require(isinstance(size, int) and size >= 0, f"verify:manifest-size-field:{relative}")
        require(isinstance(digest, str) and len(digest) == 64, f"verify:manifest-hash-field:{relative}")
        paths.append(relative)
        total += size
    require(paths == sorted(paths) and len(paths) == len(set(paths)), "verify:manifest-order")
    require(manifest.get("files") == len(entries), "verify:manifest-files")
    require(manifest.get("bytes") == total, "verify:manifest-bytes")
    return manifest


def load_manifest() -> tuple[Path, dict]:
    path = ROOT / MANIFEST_NAME
    require(path.is_file() and not path.is_symlink(), "verify:manifest-file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("verify:manifest-json") from error
    require(isinstance(manifest, dict), "verify:manifest-object")
    require(manifest.get("schema") in {MANIFEST_V1, MANIFEST_V2}, "verify:manifest-schema")
    return path, manifest


def load_compact_base(manifest: dict) -> dict:
    public_commit = manifest.get("base_public_commit")
    blob_sha1 = manifest.get("base_manifest_blob_sha1")
    require(isinstance(public_commit, str) and SHA1_RE.fullmatch(public_commit) is not None, "verify:manifest-base-commit")
    require(isinstance(blob_sha1, str) and SHA1_RE.fullmatch(blob_sha1) is not None, "verify:manifest-base-blob")
    tracked_paths(required=True)
    git("cat-file", "-e", f"{public_commit}^{{commit}}")
    anchored_blob = git("rev-parse", f"{public_commit}:{MANIFEST_NAME}").decode().strip()
    require(anchored_blob == blob_sha1, "verify:manifest-base-anchor")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", public_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor.returncode == 0, "verify:manifest-base-not-ancestor")
    try:
        base = json.loads(git("cat-file", "blob", blob_sha1).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("verify:manifest-base-json") from error
    base = validate_v1(base)
    require(base.get("source_commit") == manifest.get("source_commit"), "verify:manifest-source-commit")
    return base


def rows_by_path(entries: list[dict]) -> dict[str, dict]:
    return {row["path"]: dict(row) for row in entries}


def validate_compact(manifest: dict) -> tuple[dict[str, dict], dict]:
    require(manifest.get("schema") == MANIFEST_V2, "verify:manifest-v2-schema")
    base = load_compact_base(manifest)
    expected = rows_by_path(base["entries"])

    removed = manifest.get("removed", [])
    overrides = manifest.get("overrides", [])
    require(isinstance(removed, list) and all(isinstance(path, str) for path in removed), "verify:manifest-removed")
    require(removed == sorted(set(removed)), "verify:manifest-removed-order")
    for relative in removed:
        require(relative in expected, f"verify:manifest-removed-unknown:{relative}")
        expected.pop(relative)

    require(isinstance(overrides, list), "verify:manifest-overrides")
    override_paths: list[str] = []
    for row in overrides:
        require(isinstance(row, dict), "verify:manifest-override-row")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        require(isinstance(relative, str), "verify:manifest-override-path")
        require(isinstance(size, int) and size >= 0, f"verify:manifest-override-size:{relative}")
        require(isinstance(digest, str) and len(digest) == 64, f"verify:manifest-override-hash:{relative}")
        require(relative not in removed, f"verify:manifest-override-removed:{relative}")
        override_paths.append(relative)
        expected[relative] = {"path": relative, "bytes": size, "sha256": digest}
    require(override_paths == sorted(set(override_paths)), "verify:manifest-override-order")

    expected_paths = sorted(expected)
    require(manifest.get("files") == len(expected_paths), "verify:manifest-files")
    require(manifest.get("bytes") == sum(expected[path]["bytes"] for path in expected_paths), "verify:manifest-bytes")
    current_tracked = tracked_paths(required=True)
    require(current_tracked == expected_paths, "verify:manifest-tracked-set")
    return expected, base


def manifest_candidate(relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and relative not in {"", "."}, f"verify:manifest-path:{relative}")
    candidate = ROOT / path
    require(candidate.is_file() and not candidate.is_symlink(), f"verify:manifest-missing:{relative}")
    return candidate


def current_row(relative: str) -> dict:
    candidate = manifest_candidate(relative)
    return {"path": relative, "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def atomic_manifest_write(path: Path, manifest: dict) -> None:
    temporary: Path | None = None
    mode = path.stat().st_mode & 0o777
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".EXPORT_MANIFEST.",
            delete=False,
        ) as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refresh_manifest() -> None:
    path, manifest = load_manifest()
    if manifest["schema"] == MANIFEST_V1:
        base = validate_v1(manifest)
        current_blob = git("rev-parse", f"HEAD:{MANIFEST_NAME}").decode().strip()
        require(SHA1_RE.fullmatch(current_blob) is not None, "verify:manifest-base-blob")
        base_public_commit = git("log", "-1", "--format=%H", "--", MANIFEST_NAME).decode().strip()
        require(SHA1_RE.fullmatch(base_public_commit) is not None, "verify:manifest-base-commit")
        base_blob = current_blob
    else:
        base = load_compact_base(manifest)
        base_public_commit = manifest["base_public_commit"]
        base_blob = manifest["base_manifest_blob_sha1"]

    base_rows = rows_by_path(base["entries"])
    current = tracked_paths(required=True)
    current_set = set(current)
    removed = sorted(set(base_rows) - current_set)
    overrides = []
    total = 0
    for relative in current:
        row = current_row(relative)
        total += row["bytes"]
        if base_rows.get(relative) != row:
            overrides.append(row)

    compact = {
        "base_manifest_blob_sha1": base_blob,
        "base_public_commit": base_public_commit,
        "bytes": total,
        "files": len(current),
        "overrides": overrides,
        "removed": removed,
        "schema": MANIFEST_V2,
        "source_commit": base.get("source_commit"),
    }
    atomic_manifest_write(path, compact)
    print(
        "CYBER_LAGOON_MANIFEST_REFRESH "
        f"status=PASS files={len(current)} bytes={total} overrides={len(overrides)} removed={len(removed)}"
    )


def verify_manifest() -> None:
    _path, manifest = load_manifest()
    if manifest["schema"] == MANIFEST_V1:
        manifest = validate_v1(manifest)
        expected = rows_by_path(manifest["entries"])
        current_tracked = tracked_paths(required=False)
        if current_tracked is not None:
            require(current_tracked == sorted(expected), "verify:manifest-tracked-set")
    else:
        expected, _base = validate_compact(manifest)

    for relative in sorted(expected):
        row = expected[relative]
        candidate = manifest_candidate(relative)
        require(candidate.stat().st_size == row["bytes"], f"verify:manifest-size:{relative}")
        require(sha256(candidate) == row["sha256"], f"verify:manifest-hash:{relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="explicitly re-pin tracked public deltas against the immutable base receipt and exit",
    )
    args = parser.parse_args()
    if args.refresh_manifest:
        refresh_manifest()
        return 0

    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"verify:missing:{relative}")
    for relative in ("build", "adult", "bench", "verify"):
        require((ROOT / relative).stat().st_mode & 0o111 != 0, f"verify:not-executable:{relative}")

    build = (ROOT / "build").read_text(encoding="utf-8")
    require("--state-dir" not in build, "verify:configurable-state-root")
    for required in (
        "umask 077",
        "[[ -L .state ]]",
        "if [[ -e .build ]]",
        "chmod 700 .state",
        "chmod 700 .build",
        '[[ -L .build/direct ]]',
        "mktemp -d .state/.direct-birth.XXXXXX",
        '[[ -L .state/direct-adult.xcb ]]',
        "chmod 600 .state/direct-adult.xcb",
    ):
        require(required in build, f"verify:build-hardening:{required}")

    bench = (ROOT / "tools" / "public_bench.py").read_text(encoding="utf-8").lower()
    for forbidden in ("cmake", "nvcc", "build_concurrency_guard", "public_materialize_adult"):
        require(forbidden not in bench, f"verify:bench-crosses-build-boundary:{forbidden}")

    materialize = (ROOT / "tools" / "public_materialize_adult.py").read_text(encoding="utf-8")
    require("--state-dir" not in materialize, "verify:configurable-materializer-state-root")
    for required in (
        'STATE = ROOT / ".state"',
        "tempfile.NamedTemporaryFile",
        "os.fsync",
        "os.replace",
        "public-adult:state-symlink-refused",
        "os.chmod(STATE, 0o700)",
        "os.chmod(checkpoint_path, 0o600)",
        "os.chmod(provenance_path, 0o600)",
    ):
        require(required in materialize, f"verify:state-hardening:{required}")

    adult = (ROOT / "tools" / "public_adult.py").read_text(encoding="utf-8")
    require("os.environ.copy(" not in adult, "verify:adult-inherits-ambient-environment")
    for required in (
        "GATEWAY_BOOTSTRAP",
        "stdin=subprocess.PIPE",
        "secrets.token_hex(32)",
        "GATEWAY_STARTUP_SECONDS",
        "private_directory(STATE)",
        "tempfile.TemporaryDirectory",
        "cwd=body_dir",
        "TMPDIR",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
        "NO_PROXY",
        '"--bare"',
        '"--restricted"',
        '"--no-chrome"',
        '"--no-session-persistence"',
        "input=prompt",
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
    direct_gateway = (ROOT / "tools" / "public_direct_adult_gateway.py").read_text(encoding="utf-8")
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
    print("CYBER_LAGOON_PUBLIC_VERIFY status=PASS build_boundary=closed adult=continuing")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
