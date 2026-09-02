#!/usr/bin/env python3
"""Verify the public workshop boundary without requiring a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PYTHON = (sys.executable, "-I")
CLEAN_CHILD_ENV: dict[str, str] = {}
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
    "hardware_native/tools/foundry_workbench/reference_persistent_ambient_language_stream_v1.py",
    "hardware_native/tools/foundry_workbench/reference_authorized_ambient_feed_body_v1.py",
    "hardware_native/tools/foundry_workbench/reference_public_ambient_bystander_stream_verify.py",
    "hardware_native/tools/foundry_workbench/reference_authorized_ambient_feed_body_verify.py",
    "hardware_native/tools/foundry_workbench/reference_authorized_ambient_gateway_verify.py",
    "docs/research/2026-09-02-ambient-bystander-language-stream-preregistration.md",
)
MANIFEST_SCHEMA = "cyber-lagoon.public-workbench-export.v1"
MANIFEST_NAME = "EXPORT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tracked_paths() -> list[str] | None:
    if not (ROOT / ".git").exists():
        return None
    git = shutil.which("git", path=os.defpath)
    if not git:
        raise RuntimeError("verify:git-unavailable")
    git_path = Path(git).resolve()
    if git_path == ROOT or ROOT in git_path.parents:
        raise RuntimeError("verify:repo-local-git-refused")
    try:
        run = subprocess.run(
            [str(git_path), "ls-files", "-z"],
            cwd=ROOT,
            env={"LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1"},
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("verify:tracked-files-unavailable") from error
    paths = sorted(
        os.fsdecode(raw)
        for raw in run.stdout.split(b"\0")
        if raw and os.fsdecode(raw) != MANIFEST_NAME
    )
    require(len(paths) == len(set(paths)), "verify:tracked-files-duplicate")
    return paths


def load_manifest() -> tuple[Path, dict, list[dict]] | None:
    path = ROOT / MANIFEST_NAME
    if not path.exists():
        return None
    require(path.is_file() and not path.is_symlink(), "verify:manifest-file")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == MANIFEST_SCHEMA, "verify:manifest-schema")
    entries = manifest.get("entries")
    require(isinstance(entries, list), "verify:manifest-entries")
    paths = []
    for row in entries:
        require(isinstance(row, dict), "verify:manifest-entry")
        relative = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        require(isinstance(relative, str), "verify:manifest-paths")
        require(type(size) is int and size >= 0, f"verify:manifest-entry-size:{relative}")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"verify:manifest-entry-hash:{relative}",
        )
        paths.append(relative)
    require(len(paths) == len(set(paths)) and paths == sorted(paths), "verify:manifest-order")
    require(manifest.get("files") == len(entries), "verify:manifest-files")
    return path, manifest, entries


def manifest_candidate(relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and relative not in {"", "."}, f"verify:manifest-path:{relative}")
    candidate = ROOT
    for part in path.parts:
        candidate /= part
        require(not candidate.is_symlink(), f"verify:manifest-symlink:{relative}")
    require(candidate.is_file(), f"verify:manifest-missing:{relative}")
    return candidate


def refresh_manifest() -> None:
    loaded = load_manifest()
    require(loaded is not None, "verify:manifest-missing")
    path, manifest, _entries = loaded
    tracked = tracked_paths()
    require(tracked is not None, "verify:manifest-refresh-requires-git")
    entries = []
    total = 0
    for relative in tracked:
        candidate = manifest_candidate(relative)
        size = candidate.stat().st_size
        entries.append({"bytes": size, "path": relative, "sha256": sha256(candidate)})
        total += size
    manifest["entries"] = entries
    manifest["files"] = len(entries)
    manifest["bytes"] = total

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
            json.dump(manifest, stream, indent=2)
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
    print(f"CYBER_LAGOON_MANIFEST_REFRESH status=PASS files={len(entries)} bytes={total}")


def verify_manifest() -> None:
    loaded = load_manifest()
    if loaded is None:
        return
    _path, manifest, entries = loaded
    total = 0
    for row in entries:
        candidate = manifest_candidate(row["path"])
        require(candidate.stat().st_size == row["bytes"], f"verify:manifest-size:{row['path']}")
        require(sha256(candidate) == row["sha256"], f"verify:manifest-hash:{row['path']}")
        total += row["bytes"]
    require(manifest.get("bytes") == total, "verify:manifest-bytes")

    tracked = tracked_paths()
    if tracked is not None:
        manifest_paths = [row["path"] for row in entries]
        require(manifest_paths == tracked, "verify:manifest-tracked-set")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="explicitly re-pin the current tracked public file set and exit",
    )
    args = parser.parse_args()
    if args.refresh_manifest:
        refresh_manifest()
        return 0

    for relative in REQUIRED:
        require((ROOT / relative).is_file(), f"verify:missing:{relative}")
    for relative in ("build", "adult", "bench", "verify"):
        require((ROOT / relative).stat().st_mode & 0o111 != 0, f"verify:not-executable:{relative}")
    for relative in ("adult", "bench", "verify"):
        launcher = (ROOT / relative).read_text(encoding="utf-8")
        for required in (
            "#!/usr/bin/python3 -I",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "LD_AUDIT",
            "runpy.run_path",
        ):
            require(required in launcher, f"verify:isolated-launcher:{relative}:{required}")

    # Authenticate the pinned tree before executing any Workbench subprocess.
    verify_manifest()

    build = (ROOT / "build").read_text(encoding="utf-8")
    require("--state-dir" not in build, "verify:configurable-state-root")
    for required in (
        "#!/bin/bash -p",
        "umask 077",
        "unset LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT",
        'PYTHON="$(command -p -v python3)"',
        "materialize_args=()",
        'command -p env -i "$PYTHON" -I tools/public_materialize_adult.py',
        "canonical_direct_tool",
        "type -P c++",
        "command -p readlink -f",
        "command -p env -i",
        '"HOME=$ROOT/.build/home"',
        '"TMPDIR=$ROOT/.build/tmp"',
        "-DCMAKE_CXX_COMPILER=",
        "-DCMAKE_CUDA_COMPILER=",
        "-DCMAKE_CUDA_HOST_COMPILER=",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=",
        "[[ -L .state ]]",
        "if [[ -e .build ]]",
        "chmod 700 .state",
        "chmod 700 .build",
        "refusing non-directory Direct build boundary",
        "refusing non-regular Direct build artifact",
        "mktemp -d .state/.direct-birth.XXXXXX",
        '[[ -L .state/direct-adult.xcb ]]',
        "refusing non-regular Direct Adult checkpoint",
        "chmod 600 .state/direct-adult.xcb",
    ):
        require(required in build, f"verify:build-hardening:{required}")
    require(
        "LC_CTYPE LD_LIBRARY_PATH" not in build,
        "verify:build-loader-path-authority",
    )
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8").lower()
    require("ccache" not in cmake, "verify:implicit-compiler-launcher")

    bench = (ROOT / "tools" / "public_bench.py").read_text(encoding="utf-8").lower()
    require('python = (sys.executable, "-i")' in bench, "verify:bench-ambient-python")
    for required in (
        "direct_env_allow",
        "direct_environment()",
        "symlink_free_local",
        "trusted_local_file",
        "public-bench:direct-artifact-refused",
    ):
        require(required in bench, f"verify:bench-direct-boundary:{required}")
    for forbidden in ("cmake", "nvcc", "build_concurrency_guard", "public_materialize_adult", "ld_library_path"):
        require(forbidden not in bench, f"verify:bench-crosses-build-boundary:{forbidden}")
    for required in ('"--timeout"', '"8"'):
        require(required in bench, f"verify:bench-contact-bound:{required}")

    direct_contact_lab = (ROOT / "tools" / "direct_adult_lab.py").read_text(encoding="utf-8")
    for required in (
        "os.link(source, branch)",
        "forced_disposable",
        "accepted_input_bytes",
        "require_clean_stop",
        "green_eligible",
    ):
        require(required in direct_contact_lab, f"verify:direct-contact-lab:{required}")

    materialize = (ROOT / "tools" / "public_materialize_adult.py").read_text(encoding="utf-8")
    require("--state-dir" not in materialize, "verify:configurable-materializer-state-root")
    for required in (
        'STATE = ROOT / ".state"',
        "tempfile.NamedTemporaryFile",
        "os.fsync",
        "os.replace",
        "public-adult:state-symlink-refused",
        "public-adult:state-file-refused",
        "public-adult:state-busy",
        "fcntl.LOCK_EX",
        "os.chmod(STATE, 0o700)",
        "os.chmod(checkpoint_path, 0o600)",
        "os.chmod(provenance_path, 0o600)",
    ):
        require(required in materialize, f"verify:state-hardening:{required}")

    adult = (ROOT / "tools" / "public_adult.py").read_text(encoding="utf-8")
    require("os.environ.copy(" not in adult, "verify:adult-inherits-ambient-environment")
    for required in (
        "GATEWAY_BOOTSTRAP",
        '"-I"',
        "GATEWAY_ENV_ALLOW",
        "stdin=subprocess.PIPE",
        "secrets.token_hex(32)",
        "GATEWAY_STARTUP_SECONDS",
        "private_directory(STATE)",
        "symlink_free(path)",
        "fcntl.LOCK_EX",
        "public-adult:state-busy",
        "os.set_inheritable(lock_descriptor, True)",
        "body_checkpoint",
        "os.link(DIRECT_CHECKPOINT, branch",
        ".direct-claude.",
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
    require("LD_LIBRARY_PATH" not in adult, "verify:adult-loader-path-authority")

    direct_recipe_lab = (ROOT / "hardware_native" / "tools" / "direct_recipe_ir_lab.cu").read_text(encoding="utf-8")
    for required in (
        "DirectPreinstantiatedRecipeFamily",
        "semantic_authority=0",
        "current_thought_authority=0",
        "claim_scope=experiment_only",
    ):
        require(required in direct_recipe_lab, f"verify:direct-recipe-lab-boundary:{required}")

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
    for required in (
        "'/v1/ambient'",
        "set(request)!={'candidates'}",
        "entropy=None",
        "MAX_AUTHORIZED_AMBIENT_POOL",
        "AuthorizedAmbientFeedBodyV1.pump",
    ):
        require(required in gateway, f"verify:ambient-gateway-boundary:{required}")
    ambient_body = (
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_authorized_ambient_feed_body_v1.py"
    ).read_text(encoding="utf-8")
    for required in (
        "MAX_AUTHORIZED_AMBIENT_POOL=512",
        "MAX_AUTHORIZED_AMBIENT_POOL_BYTES=8*1024*1024",
        "secrets.randbits(128)",
        "pool_sha256",
        "semantically_blind_index",
    ):
        require(required in ambient_body, f"verify:ambient-feed-boundary:{required}")
    for forbidden in (
        "requests.", "urllib", "http://", "https://", "reddit", "twitter", "twitch", "x.com",
        "topic_score", "toxicity_score", "language_router", "feed_memory", "while true",
    ):
        require(forbidden not in ambient_body.lower(), f"verify:ambient-feed-provider-authority:{forbidden}")
    direct_gateway = (ROOT / "tools" / "public_direct_adult_gateway.py").read_text(
        encoding="utf-8"
    )
    for required in (
        "body:nonappend_transcript",
        "body:duplicate_boundary",
        '"usage": {"input_tokens": 0, "output_tokens": 0}',
        "SILENCE_FRAME",
        'f"S {TERMINAL_CHANNEL:08x} {word:08x}',
        '"text/event-stream"',
        'request.get("stream")',
        "self.process.kill()",
        "signal.signal(signal.SIGTERM, graceful_stop)",
    ):
        require(required in direct_gateway, f"verify:direct-gateway-boundary:{required}")
    for forbidden in ("contact_context", "last_motor", "LD_LIBRARY_PATH"):
        require(forbidden not in direct_gateway, f"verify:direct-gateway-obsolete:{forbidden}")

    sitdown = (ROOT / "hardware_native" / "tools" / "direct_adult_sitdown.cu").read_text(encoding="utf-8")
    for required in (
        "motors[i].trajectory.trajectory_identity",
        "motors[i].trajectory.cursor",
        "motors[i].trajectory.extent",
        "DIRECT_ADULT_SITDOWN_INPUT total=",
        "if (stop_requested) return;",
    ):
        require(required in sitdown, f"verify:direct-sitdown-transport:{required}")

    scripts = (
        ROOT / "tools" / "public_adult.py",
        ROOT / "tools" / "public_adult_smoke.py",
        ROOT / "tools" / "direct_adult_lab.py",
        ROOT / "tools" / "public_bench.py",
        ROOT / "tools" / "public_direct_adult_gateway.py",
        ROOT / "tools" / "public_materialize_adult.py",
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_persistent_ambient_language_stream_v1.py",
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_authorized_ambient_feed_body_v1.py",
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_public_ambient_bystander_stream_verify.py",
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_authorized_ambient_feed_body_verify.py",
        ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_authorized_ambient_gateway_verify.py",
    )
    for script in scripts:
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
        except SyntaxError as error:
            raise RuntimeError(
                f"verify:python-syntax:{script.relative_to(ROOT)}:{error.msg}"
            ) from error
    subprocess.run(
        [*PYTHON, str(ROOT / "tools" / "public_bench.py"), "--no-ir"],
        cwd=ROOT,
        env=CLEAN_CHILD_ENV,
        check=True,
    )
    require((ROOT / ".state" / "adult.json").is_file(), "verify:run-build-first")
    subprocess.run(
        [*PYTHON, str(ROOT / "tools" / "public_adult.py"), "--print-backend"],
        cwd=ROOT,
        env=CLEAN_CHILD_ENV,
        check=True,
    )
    print("CYBER_LAGOON_PUBLIC_VERIFY status=PASS build_boundary=closed adult=continuing")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
