#!/usr/bin/env python3
"""Installed Claude carries resident ambiguity through clarification and composition."""
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

from reference_language_mastery_claude_gateway_v1 import (
    CLAUDE_MCP_BODY_ACTION_TOOL, body_source_identity,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude
from reference_resident_lexical_ambiguity_clarification_verify import (
    ANCHOR, KNOWN, OTHER, RAW, develop_to_mark,
)


CONFIRMATION = b"sunlight heats the glasshouse aka glimmer heats the glasshouse"


def prepare(runtime):
    language = runtime.adult.language_adult.language
    for source in (0xFA10, 0xFA11):
        language.observe_naming(OTHER, ANCHOR, source)
    return runtime


def experience(runtime, source):
    clarification, action = runtime.contact_utterance(RAW, source, source)
    receipt = runtime.adult.pending_endogenous_inquiry_actions.get(action)
    public = bool(receipt and runtime.adult.settle_endogenous_inquiry_motor_return(
        receipt, source, True))
    confirmation_surface, confirmation_action = runtime.contact_utterance(
        CONFIRMATION, source, source)
    settled = bool(public and action not in
                   runtime.adult.pending_endogenous_inquiry_actions)
    effect = runtime.adult.language_adult.leaf(100, (0xA105,))
    question = b"why is it the case that " + bytes(effect.surface).lower() + b"?"
    reply, action = runtime.contact_utterance(question, source + 1, source + 1)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    certificate = (() if receipt is None else
                   runtime.adult._causal_action_coordinates(receipt))
    return (bytes(clarification), bytes(confirmation_surface),
            int(confirmation_action), settled, question, bytes(reply), certificate)


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    base = prepare(develop_to_mark("lexical_carrier_ready"))
    direct = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    (expected_clarification, expected_confirmation_surface,
     expected_confirmation_action, settled, question, expected_reply,
     certificate) = (
        experience(direct, body_source_identity("literal-ambiguity-control")))
    rows = []
    gateway_errors = ""

    with tempfile.TemporaryDirectory(
            prefix="foundry-literal-ambiguity-") as directory:
        root = Path(directory)
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
            if binary and settled and certificate:
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
                for prompt in (RAW, CONFIRMATION):
                    row = _claude(binary, environment, config, prompt,
                                  session or None)
                    rows.append(row)
                    session = row[1]
                rows.append(_claude(binary, environment, config, question, None))
        finally:
            if gateway is not None:
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-2048:]

        checkpoint_text = checkpoint.read_text()
        restored = ReferenceLifeFunctionRuntimeV2.restore(
            canonical_species_program_v2(), json.loads(checkpoint_text))
        revised = bytes(restored.adult.language_adult.language.lexeme(KNOWN) or ())

    observed = tuple(row[2] for row in rows)
    sessions = tuple(row[1] for row in rows)
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "direct_resident_clarification_and_certified_composition_exist":
            bool(expected_clarification) and settled and bool(certificate)
            and not expected_confirmation_surface
            and expected_confirmation_action == 0
            and revised.startswith(b"glimmer"),
        "literal_answer_settles_then_fresh_session_composes":
            len(rows) == 3 and all(row[0] == 0 for row in rows)
            and len(set(sessions[:2])) == 1 and bool(sessions[0])
            and sessions[2] and sessions[2] != sessions[0],
        "literal_body_returns_resident_clarification_action":
            len(observed) == 3 and observed[0] == (expected_clarification,),
        "literal_partner_confirmation_is_evidence_not_public_acknowledgement":
            len(observed) == 3 and not observed[1],
        "literal_body_returns_heldout_certified_composition":
            len(observed) == 3 and expected_reply in observed[2]
            and revised in expected_reply,
        "checkpoint_keeps_learned_relation_not_transcript":
            RAW.decode() not in checkpoint_text
            and CONFIRMATION.decode() not in checkpoint_text
            and question.decode() not in checkpoint_text,
        "gateway_has_no_runtime_error": not gateway_errors.strip(),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-lexical-ambiguity.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "clarification": expected_clarification.decode(),
        "partner_confirmation": CONFIRMATION.decode(),
        "heldout_question": question.decode(),
        "heldout_reply": expected_reply.decode(),
        "observed": [[surface.decode() for surface in row] for row in observed],
        "revised_surface": revised.decode(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
