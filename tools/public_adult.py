#!/usr/bin/env python3
"""Start the one public Adult through a terminal or Claude Code body."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".state"
REFERENCE_CHECKPOINT = STATE / "adult.json"
DIRECT_CHECKPOINT = STATE / "direct-adult.xcb"
DIRECT_SITDOWN = ROOT / ".build" / "direct" / "direct_adult_sitdown"
REFERENCE_GATEWAY = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_claude_gateway_v1.py"
REFERENCE_TERMINAL = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_terminal_v1.py"
DIRECT_GATEWAY = ROOT / "tools" / "public_direct_adult_gateway.py"
BODY_ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "COLORTERM", "SHELL", "TMPDIR", "USER", "LOGNAME")


def regular_file(path: Path, executable: bool = False) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and (not executable or os.access(path, os.X_OK))
    )


def backend_available(name: str) -> bool:
    if name == "direct":
        return regular_file(DIRECT_CHECKPOINT) and regular_file(DIRECT_SITDOWN, executable=True)
    return regular_file(REFERENCE_CHECKPOINT)


def choose_backend(requested: str) -> str:
    if requested == "auto":
        # The continuing Claude dialogue path is reference-backed until Direct
        # partner/world consequence settlement is proved end to end. Real Direct
        # remains available explicitly and through ./bench without recompilation.
        requested = "reference"
    if not backend_available(requested):
        raise RuntimeError(f"public-adult:{requested}-backend-unavailable; run ./build")
    return requested


def gateway_command(backend: str) -> list[str]:
    if backend == "direct":
        return [
            sys.executable,
            str(DIRECT_GATEWAY),
            "--sitdown",
            str(DIRECT_SITDOWN),
            "--resume",
            str(DIRECT_CHECKPOINT),
            "--credential-stdin",
            "--port",
            "0",
        ]
    return [
        sys.executable,
        str(REFERENCE_GATEWAY),
        "--resume",
        str(REFERENCE_CHECKPOINT),
        "--auth-token-stdin",
        "--port",
        "0",
    ]


def isolated_body_environment(port: int, credential: str, config: Path, body_dir: Path) -> dict[str, str]:
    env = {name: os.environ[name] for name in BODY_ENV_ALLOW if os.environ.get(name)}
    path_entries = []
    for raw in env.get("PATH", "").split(os.pathsep):
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if candidate == ROOT or ROOT in candidate.parents:
            continue
        path_entries.append(str(candidate))
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    else:
        env.pop("PATH", None)
    env["HOME"] = str(config)
    env["PWD"] = str(body_dir)
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_" + "AUTH_TOKEN"] = credential
    env["ANTHROPIC_" + "API_KEY"] = credential
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CONFIG_DIR"] = str(config)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def run_claude(backend: str, model: str, prompt: str | None) -> int:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("public-adult:claude-code-not-found; use ./adult --terminal")
    claude_path = Path(claude).resolve()
    if claude_path == ROOT or ROOT in claude_path.parents:
        raise RuntimeError("public-adult:repo-local-claude-refused")

    credential = "local-" + secrets.token_hex(32)
    gateway = subprocess.Popen(
        gateway_command(backend),
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert gateway.stdin is not None
        try:
            gateway.stdin.write(credential + "\n")
            gateway.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError("public-adult:gateway-credential-refused") from error
        finally:
            gateway.stdin.close()

        assert gateway.stdout is not None
        ready = gateway.stdout.readline().strip().split()
        if len(ready) < 2 or not ready[-1].isdigit():
            raise RuntimeError("public-adult:gateway-startup-refused")
        port = int(ready[-1])

        config = STATE / "claude-config"
        body_dir = config / "body"
        config.mkdir(parents=True, exist_ok=True, mode=0o700)
        body_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(config, 0o700)
        os.chmod(body_dir, 0o700)
        env = isolated_body_environment(port, credential, config, body_dir)

        print(f"CYBER_LAGOON_ADULT backend={backend} body=claude-code", flush=True)
        command = [str(claude_path), "--model", model]
        if prompt is not None:
            command = [
                str(claude_path),
                "-p",
                "--model",
                model,
                "--max-turns",
                "1",
                "--output-format",
                "json",
                prompt,
            ]
        return subprocess.run(command, cwd=body_dir, env=env, check=False).returncode
    finally:
        if gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "reference", "direct"), default="auto")
    parser.add_argument("--terminal", action="store_true", help="use the raw reference terminal body")
    parser.add_argument("--model", default="sonnet", help="Claude Code client label; cognition still comes from the Adult gateway")
    parser.add_argument("--print-backend", action="store_true")
    parser.add_argument("--prompt", help="send one normal-sentence contact through Claude Code and exit")
    args = parser.parse_args()

    backend = choose_backend(args.backend)
    if args.print_backend:
        print(backend)
        return 0
    if args.terminal:
        if backend != "reference":
            raise RuntimeError("public-adult:raw-terminal-currently-reference-only")
        os.execv(
            sys.executable,
            [sys.executable, str(REFERENCE_TERMINAL), "--resume", str(REFERENCE_CHECKPOINT)],
        )
    if backend == "direct" and args.prompt is None:
        raise RuntimeError(
            "public-adult:direct-claude-is-one-shot-until-world-consequence-settlement-is-proved"
        )
    return run_claude(backend, args.model, args.prompt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
