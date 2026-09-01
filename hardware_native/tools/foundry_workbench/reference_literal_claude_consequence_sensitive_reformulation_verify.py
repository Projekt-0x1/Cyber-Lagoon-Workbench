#!/usr/bin/env python3
"""Installed Claude transports a consequence-selected certified reformulation."""
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
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude
from reference_natural_dialogue_action_consequence_verify import _focus
from reference_predictive_credit_profile_v1 import Q

# Commit-gate declaration: this executable receipt proves a resident-visible
# change of certified causal composition through the installed body surface.
visible_language_gain = True


def coordinates(runtime, action):
    receipt = runtime.adult.pending_causal_dialogue_actions.get(int(action))
    return (() if receipt is None else
            tuple(runtime.adult._causal_action_coordinates(receipt)))


def train_adverse(runtime, question, body_source):
    surfaces = []
    for index in range(3):
        surface, action = runtime.contact_utterance(
            question, body_source, body_source)
        receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
        if receipt is None:
            return tuple(surfaces), False
        surfaces.append(bytes(surface))
        if not runtime.settle_contact_consequence(
                action, 0xFD80 + index, -Q, -Q, True):
            return tuple(surfaces), False
        if not runtime.observe_contact_background(
                action, 0xFD90 + index, False):
            return tuple(surfaces), False
    return tuple(surfaces), True


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    gateway_errors = ""

    with tempfile.TemporaryDirectory(
            prefix="foundry-literal-certified-reformulation-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        _effect, question = _focus(base)
        body_source = body_source_identity("literal-body")

        baseline = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        baseline_surface, baseline_action = baseline.contact_utterance(
            question, body_source, body_source)
        baseline_receipt = baseline.adult.pending_causal_dialogue_actions.get(
            baseline_action)
        baseline_coordinates = coordinates(baseline, baseline_action)

        adverse_surfaces, trained = train_adverse(base, question, body_source)
        trained_checkpoint = copy.deepcopy(base.checkpoint())
        direct = type(base).restore(base.program, copy.deepcopy(trained_checkpoint))
        expected_surface, expected_action = direct.contact_utterance(
            question, body_source, body_source)
        expected_receipt = direct.adult.pending_causal_dialogue_actions.get(
            expected_action)
        expected_coordinates = coordinates(direct, expected_action)

        checkpoint = root / "adult.json"
        checkpoint.write_text(json.dumps(
            trained_checkpoint, separators=(",", ":"), sort_keys=True))
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"agi_body": {
            "type": "stdio", "command": sys.executable,
            "args": [str(Path(__file__).with_name(
                "reference_claude_body_action_mcp_v1.py"))],
        }}}))
        gateway = None
        try:
            if binary and trained and expected_receipt is not None:
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
                rows.append(_claude(binary, environment, config, question, None))
        finally:
            if gateway is not None:
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-2048:]

        checkpoint_text = checkpoint.read_text()
        restored = ReferenceLifeFunctionRuntimeV2.restore(
            canonical_species_program_v2(), json.loads(checkpoint_text))
        restored_surface, _restored_programs = restored.adult.compose_causal_component(
            _effect, channel=body_source)

    observed = tuple(row[2] for row in rows)
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "repeated_exact_adverse_history_precedes_reformulation": bool(
            trained and len(adverse_surfaces) == 3 and all(adverse_surfaces)),
        "direct_reformulation_changes_form_not_causal_coordinates": bool(
            baseline_receipt is not None and expected_receipt is not None
            and baseline_surface != expected_surface
            and baseline_receipt.factors[:1] != expected_receipt.factors[:1]
            and set(expected_coordinates).issubset(set(baseline_coordinates))),
        "installed_claude_transports_resident_reformulation": bool(
            len(rows) == 1 and rows[0][0] == 0 and rows[0][1]
            and observed == ((bytes(expected_surface),),)),
        "checkpoint_keeps_competition_not_transcript": bool(
            question.decode() not in checkpoint_text
            and expected_surface.decode() not in checkpoint_text),
        "same_adult_continues_after_body_return": bool(restored_surface),
        "gateway_has_no_runtime_error": not gateway_errors.strip(),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-certified-reformulation.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "question": question.decode(),
        "baseline": bytes(baseline_surface).decode(),
        "adverse_history": [surface.decode() for surface in adverse_surfaces],
        "reformulation": bytes(expected_surface).decode(),
        "observed": [[surface.decode() for surface in row] for row in observed],
        "post_return_surface": bytes(restored_surface).decode(),
        "baseline_factors": [] if baseline_receipt is None else list(baseline_receipt.factors),
        "reformulation_factors": [] if expected_receipt is None else list(expected_receipt.factors),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "runtime_llm": False,
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
