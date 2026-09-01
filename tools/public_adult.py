#!/usr/bin/env python3
"""Start the one public Adult through a terminal or Claude Code body."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".state"
REFERENCE_CHECKPOINT = STATE / "adult.json"
DIRECT_CHECKPOINT = STATE / "direct-adult.xcb"
DIRECT_SITDOWN = ROOT / ".build" / "direct" / "direct_adult_sitdown"
REFERENCE_GATEWAY = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_claude_gateway_v1.py"
REFERENCE_TERMINAL = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_terminal_v1.py"
DIRECT_GATEWAY = ROOT / "tools" / "public_direct_adult_gateway.py"


def backend_available(name: str) -> bool:
    if name == "direct":
        return DIRECT_CHECKPOINT.is_file() and DIRECT_SITDOWN.is_file() and os.access(DIRECT_SITDOWN, os.X_OK)
    return REFERENCE_CHECKPOINT.is_file()


def choose_backend(requested: str) -> str:
    if requested == "auto":
        # The continuing Claude dialogue path is reference-backed until Direct
        # partner/world consequence settlement is proved end to end. Real Direct
        # remains available explicitly and through ./bench without recompilation.
        return "reference"
    if not backend_available(requested):
        raise RuntimeError(f"public-adult:{requested}-backend-unavailable; run ./build")
    return requested


def gateway_command(backend: str, credential: str) -> list[str]:
    auth_flag = "--auth-" + "token"
    if backend == "direct":
        return [
            sys.executable,
            str(DIRECT_GATEWAY),
            "--sitdown",
            str(DIRECT_SITDOWN),
            "--resume",
            str(DIRECT_CHECKPOINT),
            "--credential",
            credential,
            "--port",
            "0",
        ]
    return [
        sys.executable,
        str(REFERENCE_GATEWAY),
        "--resume",
        str(REFERENCE_CHECKPOINT),
        auth_flag,
        credential,
        "--port",
        "0",
    ]


def run_claude(backend: str, model: str, prompt: str | None) -> int:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("public-adult:claude-code-not-found; use ./adult --terminal")
    credential = "local-" + uuid.uuid4().hex
    gateway = subprocess.Popen(
        gateway_command(backend, credential), cwd=ROOT, stdout=subprocess.PIPE, text=True
    )
    try:
        assert gateway.stdout is not None
        ready = gateway.stdout.readline().strip().split()
        if len(ready) < 2 or not ready[-1].isdigit():
            raise RuntimeError("public-adult:gateway-startup-refused")
        port = int(ready[-1])
        config = STATE / "claude-config"
        config.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
        env["ANTHROPIC_" + "AUTH_TOKEN"] = credential
        env["ANTHROPIC_" + "API_KEY"] = credential
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_CONFIG_DIR"] = str(config)
        print(f"CYBER_LAGOON_ADULT backend={backend} body=claude-code", flush=True)
        command = [claude, "--model", model]
        if prompt is not None:
            command = [
                claude,
                "-p",
                "--model",
                model,
                "--max-turns",
                "1",
                "--output-format",
                "json",
                prompt,
            ]
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    finally:
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
