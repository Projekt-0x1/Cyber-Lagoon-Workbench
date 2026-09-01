#!/usr/bin/env python3
"""Literal Claude carries one resident intention through repeated body reafference."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from life_function_factory_v1 import build_cache, load_mark
from reference_claude_body_causal_uptake_verify import _focus_and_paraphrase
from reference_language_mastery_claude_gateway_v1 import (
    CLAUDE_MCP_BODY_ACTION_TOOL, body_source_identity,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_settled_intention_causal_continuation_verify import REQUEST, _arm


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _claude(binary, environment, config, prompt, session=None):
    command = [binary, "-p"]
    if session:
        command.extend(("--resume", session))
    command.extend((
        "--model", "sonnet", "--max-turns", "8", "--verbose",
        "--output-format", "stream-json", "--mcp-config", str(config),
        "--strict-mcp-config", "--allowedTools=" + CLAUDE_MCP_BODY_ACTION_TOOL,
        prompt.decode(),
    ))
    run = subprocess.run(command, capture_output=True, text=True,
                         timeout=45, env=environment)
    payloads = []
    for line in run.stdout.splitlines():
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    objects = tuple(item for payload in payloads for item in _objects(payload))
    surfaces = tuple(
        item["input"]["surface"].encode()
        for item in objects
        if item.get("type") == "tool_use"
        and item.get("name") == CLAUDE_MCP_BODY_ACTION_TOOL
        and isinstance(item.get("input"), dict)
        and isinstance(item["input"].get("surface"), str)
    )
    # Stream events can repeat a materialized content block; retain physical order
    # while suppressing only immediately duplicated observer copies.
    observed = tuple(surface for index, surface in enumerate(surfaces)
                     if index == 0 or surface != surfaces[index - 1])
    sessions = tuple(str(item["session_id"]) for item in objects
                     if item.get("session_id"))
    results = tuple({key: item.get(key) for key in
                     ("subtype", "is_error", "result", "error", "num_turns")
                     if key in item}
                    for item in objects if item.get("type") == "result")
    return run.returncode, (sessions[-1] if sessions else session or ""), \
        observed, run.stderr[-1024:], results


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    gateway_errors = ""
    with tempfile.TemporaryDirectory(prefix="foundry-claude-tool-loop-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        _effect, question, *_ = _focus_and_paraphrase(base)
        direct = _arm(
            type(base).restore(base.program, copy.deepcopy(base.checkpoint())),
            body_source_identity("literal-tool-loop-control"),
        )
        expected = (direct["requested"], *(row[0] for row in direct["rows"]))
        checkpoint = root / "adult.json"
        checkpoint.write_text(json.dumps(
            base.checkpoint(), separators=(",", ":"), sort_keys=True))
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"agi_body": {
            "type": "stdio", "command": sys.executable,
            "args": [str(Path(__file__).with_name(
                "reference_claude_body_action_mcp_v1.py"))],
        }}}))
        gateway = None
        try:
            if binary:
                gateway = subprocess.Popen(
                    (sys.executable, str(Path(__file__).with_name(
                        "reference_language_mastery_claude_gateway_v1.py")),
                     "--resume", str(checkpoint), "--auth-token", "literal-body"),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                port = int(gateway.stdout.readline().split()[-1])
                environment = os.environ.copy()
                environment.update({
                    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                    "ANTHROPIC_AUTH_TOKEN": "literal-body",
                    "ANTHROPIC_API_KEY": "literal-body",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                })
                session = ""
                for prompt in (question, REQUEST):
                    row = _claude(binary, environment, config, prompt, session or None)
                    rows.append(row)
                    session = row[1]
        finally:
            if gateway is not None:
                gateway.terminate(); gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-2048:]

        checkpoint_text = checkpoint.read_text()
        restored = ReferenceLifeFunctionRuntimeV2.restore(
            canonical_species_program_v2(), json.loads(checkpoint_text))
        pending = len(restored.adult.pending_causal_dialogue_actions)
        literal_channel = (body_source_identity("literal-body") if rows else 0)
        receipts = tuple(
            row for row in restored.adult.recent_causal_dialogue_actions.values()
            if int(row.source) == literal_channel and int(row.channel) == literal_channel)
        coordinates = tuple(
            restored.adult._causal_action_leading_coordinate(row)
            for row in receipts)
        mcp_source = Path(__file__).with_name(
            "reference_claude_body_action_mcp_v1.py").read_text()

    checks = {
        "installed_claude_binary_exists": bool(binary),
        "two_literal_processes_complete_one_session":
            len(rows) == 2 and tuple(row[0] for row in rows) == (0, 0)
            and rows[0][1] == rows[1][1] and bool(rows[0][1]),
        "first_contact_is_one_resident_body_action":
            len(rows) == 2 and len(rows[0][2]) == 1,
        "one_request_carries_full_resident_action_chain":
            len(rows) == 2 and rows[1][2] == expected,
        "every_emitted_action_returns_before_resident_silence": pending == 0,
        "literal_actions_retain_resident_relation_certificates":
            len(coordinates) >= len(expected) + 1
            and all(len(row) == 3 and min(row) > 0 for row in coordinates),
        "checkpoint_contains_no_contact_or_public_surface":
            question.decode() not in checkpoint_text
            and REQUEST.decode() not in checkpoint_text
            and all(surface.decode() not in checkpoint_text
                    for row in rows for surface in row[2]),
        "transport_does_not_supply_expected_surfaces":
            REQUEST.decode() not in mcp_source
            and all(surface.decode() not in mcp_source for surface in expected),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-settled-intention-tool-loop.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "expected_surfaces": tuple(surface.decode() for surface in expected),
        "observed_process_surfaces": tuple(
            tuple(surface.decode() for surface in row[2]) for row in rows),
        "process_codes_and_sessions": tuple((row[0], row[1]) for row in rows),
        "process_errors": tuple(row[3] for row in rows),
        "process_results": tuple(row[4] for row in rows),
        "certified_leading_coordinates": coordinates,
        "gateway_error_tail": gateway_errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "language_phenotype_improved": not failed,
        "visible_language_gain":
            "ONE_LITERAL_CLAUDE_REQUEST_CARRIES_RESIDENT_CAUSAL_COMPOSITION_UNTIL_SILENCE",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
