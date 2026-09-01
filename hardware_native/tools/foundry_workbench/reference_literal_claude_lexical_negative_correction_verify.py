#!/usr/bin/env python3
"""Installed Claude carries a corrected resident referent into composition."""
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

from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude
from reference_resident_lexical_negative_correction_verify import OTHER, RAW, prepare


CORRECTION = b"steady wind closes the vent aka glimmer heats the glasshouse"
UNRELATED = b"the heater warms the greenhouse"


def stage(runtime, source, *, reafference=True):
    surface, action = runtime.contact_utterance(RAW, source, source)
    receipt = runtime.adult.pending_endogenous_inquiry_actions.get(action)
    public = bool(receipt and reafference and
                  runtime.adult.settle_endogenous_inquiry_motor_return(
                      receipt, source, True))
    return bytes(surface), int(action), receipt, public


def expected_sequence(runtime, source):
    first, first_action = runtime.contact_utterance(RAW, source, source)
    first_receipt = runtime.adult.pending_endogenous_inquiry_actions.get(
        first_action)
    first_public = bool(first_receipt and
                        runtime.adult.settle_endogenous_inquiry_motor_return(
                            first_receipt, source, True))
    second, second_action = runtime.contact_utterance(
        CORRECTION, source, source)
    second_receipt = runtime.adult.pending_endogenous_inquiry_actions.get(
        second_action)
    second_public = bool(second_receipt and
                         runtime.adult.settle_endogenous_inquiry_motor_return(
                             second_receipt, source, True))
    settled_surface, settled_action = runtime.contact_utterance(
        CORRECTION, source, source)
    effect = runtime.adult.language_adult.leaf(100, (OTHER,))
    question = b"why is it the case that " + bytes(effect.surface).lower() + b"?"
    reply, action = runtime.contact_utterance(question, source + 1, source + 1)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    certificate = (() if receipt is None else
                   runtime.adult._causal_action_coordinates(receipt))
    return {
        "first": bytes(first), "first_public": first_public,
        "second": bytes(second), "second_public": second_public,
        "settled_surface": bytes(settled_surface),
        "settled_action": int(settled_action), "question": question,
        "reply": bytes(reply), "certificate": certificate,
    }


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    body_source = body_source_identity("literal-body")
    base = prepare()
    direct = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    expected = expected_sequence(direct, body_source)
    question = expected["question"]

    pre_reafference = prepare()
    _pre_surface, pre_action, _pre_receipt, _ = stage(
        pre_reafference, body_source, reafference=False)
    pre_reafference.contact_utterance(CORRECTION, body_source, body_source)

    wrong_partner = prepare()
    _wrong_surface, wrong_action, _wrong_receipt, wrong_public = stage(
        wrong_partner, body_source)
    wrong_partner.contact_utterance(
        CORRECTION, body_source + 1, body_source + 1)

    unrelated = prepare()
    _unrelated_surface, unrelated_action, _unrelated_receipt, unrelated_public = stage(
        unrelated, body_source)
    unrelated.contact_utterance(UNRELATED, body_source, body_source)

    ambiguous = prepare()
    _ambiguous_surface, ambiguous_action, _ambiguous_receipt, ambiguous_public = stage(
        ambiguous, body_source)
    ambiguous.contact_utterance(RAW, body_source, body_source)

    pending_restart = prepare()
    _restart_surface, restart_action, _restart_receipt, restart_public = stage(
        pending_restart, body_source)
    restored_pending = ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), copy.deepcopy(pending_restart.checkpoint()))
    restored_second, restored_second_action = restored_pending.contact_utterance(
        CORRECTION, body_source, body_source)
    rows = []
    gateway_errors = ""

    with tempfile.TemporaryDirectory(prefix="foundry-literal-correction-") as directory:
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
            if (binary and expected["first_public"] and expected["second_public"]
                    and expected["certificate"]):
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
                for prompt in (RAW, CORRECTION, CORRECTION):
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
        revised = bytes(restored.adult.language_adult.language.lexeme(OTHER) or ())

    observed = tuple(row[2] for row in rows)
    sessions = tuple(row[1] for row in rows)
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "direct_recast_counters_first_then_supports_second_candidate":
            expected["first_public"] and expected["second_public"]
            and expected["first"] and expected["second"]
            and expected["first"] != expected["second"]
            and not expected["settled_surface"]
            and expected["settled_action"] == 0 and revised.startswith(b"glimmer"),
        "contact_before_public_motor_reafference_cannot_answer":
            pre_action in pre_reafference.adult.pending_endogenous_inquiry_actions,
        "wrong_partner_cannot_answer_pending_inquiry":
            wrong_public and wrong_action in
            wrong_partner.adult.pending_endogenous_inquiry_actions,
        "unrelated_contact_cannot_answer_pending_inquiry":
            unrelated_public and unrelated_action in
            unrelated.adult.pending_endogenous_inquiry_actions,
        "ambiguous_contact_cannot_answer_pending_inquiry":
            ambiguous_public and ambiguous_action in
            ambiguous.adult.pending_endogenous_inquiry_actions,
        "checkpoint_preserves_public_unanswered_inquiry_for_later_repair":
            restart_public and restart_action not in
            restored_pending.adult.pending_endogenous_inquiry_actions
            and bytes(restored_second) == expected["second"]
            and restored_second_action > 0,
        "literal_three_contacts_repair_then_fresh_session_composes":
            len(rows) == 4 and all(row[0] == 0 for row in rows)
            and len(set(sessions[:3])) == 1 and bool(sessions[0])
            and sessions[3] and sessions[3] != sessions[0],
        "literal_motor_success_does_not_accept_first_candidate":
            len(observed) == 4 and observed[0] == (expected["first"],)
            and observed[1] == (expected["second"],),
        "literal_partner_recast_settles_without_scripted_acknowledgement":
            len(observed) == 4 and not observed[2],
        "fresh_session_returns_heldout_certified_composition":
            len(observed) == 4 and expected["reply"] in observed[3]
            and revised in expected["reply"] and bool(expected["certificate"]),
        "checkpoint_keeps_correction_not_transcript":
            RAW.decode() not in checkpoint_text
            and CORRECTION.decode() not in checkpoint_text
            and question.decode() not in checkpoint_text,
        "gateway_has_no_runtime_error": not gateway_errors.strip(),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-lexical-negative-correction.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "rejected_surface": expected["first"].decode(),
        "corrected_clarification": expected["second"].decode(),
        "partner_recast": CORRECTION.decode(),
        "heldout_question": question.decode(),
        "heldout_reply": expected["reply"].decode(),
        "observed": [[surface.decode() for surface in row] for row in observed],
        "revised_surface": revised.decode(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
