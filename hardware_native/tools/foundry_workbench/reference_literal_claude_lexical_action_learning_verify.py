#!/usr/bin/env python3
"""Literal Claude consequences revise resident words and causal composition."""
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
from reference_language_mastery_claude_gateway_v1 import (
    CLAUDE_MCP_BODY_ACTION_TOOL, body_source_identity, settle_tool_result,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude


CONTACTS = (
    b"morning sunlight warms the greenhouse aka sunbeams heat the glasshouse",
    b"steady wind closes the vent aka airflow seals the vent",
)


def _experience(runtime, source, contacts):
    surfaces = []
    confirmations = []
    for index, contact in enumerate(contacts):
        surface, action = runtime.contact_utterance(contact, source, source)
        receipt = runtime.adult.pending_endogenous_inquiry_actions.get(action)
        if not surface or receipt is None:
            return tuple(surfaces), tuple(confirmations), b"", (), False
        surfaces.append(bytes(surface))
        if not runtime.adult.settle_endogenous_inquiry_motor_return(
                receipt, source, True):
            return tuple(surfaces), tuple(confirmations), b"", (), False
        confirmation, confirmation_action = runtime.contact_utterance(
            contact, source, source)
        confirmations.append((bytes(confirmation), int(confirmation_action)))
        if confirmation or confirmation_action:
            return tuple(surfaces), tuple(confirmations), b"", (), False
    effect = runtime.adult.language_adult.leaf(100, (0xA105,))
    question = b"why is it the case that " + bytes(effect.surface).lower() + b"?"
    reply, action = runtime.contact_utterance(question, source, source)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    certificate = (() if receipt is None else
                   runtime.adult._causal_action_coordinates(receipt))
    return (tuple(surfaces), tuple(confirmations), question,
            (bytes(reply), certificate), bool(receipt))


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    stages = []
    gateway_errors = ""
    with tempfile.TemporaryDirectory(
            prefix="foundry-literal-lexical-action-") as directory:
        root = Path(directory)
        build_cache(directory)
        base = load_mark(directory, "lexical_carrier_ready")
        direct = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        (expected_surfaces, expected_confirmations, question,
         expected_reply, direct_ok) = _experience(
            direct, body_source_identity("literal-lexical-control"), CONTACTS)

        adverse = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        adverse_surface, adverse_action = adverse.contact_utterance(
            CONTACTS[0], 0xFE01, 0xFE01)
        adverse_receipt = adverse.adult.pending_endogenous_inquiry_actions.get(
            adverse_action)
        adverse_feature = (0 if adverse_receipt is None else
                           int(adverse_receipt.obligation_effect))
        adverse_before = adverse.adult.language_adult.language.lexeme(adverse_feature)
        adverse_ok = bool(adverse_receipt and adverse.settle_contact_consequence(
            adverse_action, 0xFE02, -1, 0, True))
        adverse_after = adverse.adult.language_adult.language.lexeme(adverse_feature)

        qualified = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        qualified_source = 0xFE11
        _qualified_surface, qualified_action = qualified.contact_utterance(
            CONTACTS[0], qualified_source, qualified_source)
        qualified = type(base).restore(
            base.program, copy.deepcopy(qualified.checkpoint()))
        qualified_receipt = qualified.adult.pending_endogenous_inquiry_actions.get(
            qualified_action)
        wrong_session = settle_tool_result(
            qualified, "toolu_agi_" + format(qualified_action, "x"),
            qualified_source + 1, False)
        zero_source = qualified.settle_contact_consequence(
            qualified_action, 0, 1, 0, True)
        yoked = qualified.settle_contact_consequence(
            qualified_action, qualified_source, 1, 0, False)
        withdrawn = type(base).restore(
            base.program, copy.deepcopy(qualified.checkpoint()))
        withdrawn.adult.language_adult.language.withdraw_source(qualified_source)
        withdrawn_return = withdrawn.settle_contact_consequence(
            qualified_action, qualified_source, 1, 0, True)

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
            if binary and direct_ok:
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
                for prompt in (CONTACTS[0], CONTACTS[0],
                               CONTACTS[1], CONTACTS[1]):
                    row = _claude(binary, environment, config, prompt,
                                  session or None)
                    rows.append(row); session = row[1]
                    stage = ReferenceLifeFunctionRuntimeV2.restore(
                        canonical_species_program_v2(),
                        json.loads(checkpoint.read_text()))
                    stages.append({
                        "cause": bytes(stage.adult.language_adult.leaf(
                            100, (0xA103,)).surface).decode(),
                        "effect": bytes(stage.adult.language_adult.leaf(
                            100, (0xA105,)).surface).decode(),
                        "pending_causal": len(
                            stage.adult.pending_causal_dialogue_actions),
                        "pending_inquiry": len(
                            stage.adult.pending_endogenous_inquiry_actions),
                        "continuations": len(
                            stage.adult.last_causal_dialogue_contact_continuations),
                    })
                rows.append(_claude(binary, environment, config, question, None))
        finally:
            if gateway is not None:
                gateway.terminate(); gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-2048:]

        checkpoint_text = checkpoint.read_text()
        restored = ReferenceLifeFunctionRuntimeV2.restore(
            canonical_species_program_v2(), json.loads(checkpoint_text))
        cause = bytes(restored.adult.language_adult.leaf(100, (0xA103,)).surface)
        effect = bytes(restored.adult.language_adult.leaf(100, (0xA105,)).surface)
        observed = tuple(row[2] for row in rows)
        sessions = tuple(row[1] for row in rows)
        expected_composition, certificate = expected_reply
        mcp_source = Path(__file__).with_name(
            "reference_claude_body_action_mcp_v1.py").read_text()

    checks = {
        "installed_claude_binary_exists": bool(binary),
        "direct_resident_actions_and_composition_exist":
            direct_ok and len(expected_surfaces) == 2 and bool(expected_composition)
            and len(expected_confirmations) == 2
            and all(not surface and action == 0
                    for surface, action in expected_confirmations)
            and bool(certificate),
        "literal_answers_two_inquiries_then_fresh_session_composes":
            len(rows) == 5 and all(row[0] == 0 for row in rows)
            and len(set(sessions[:4])) == 1 and bool(sessions[0])
            and sessions[4] and sessions[4] != sessions[0],
        "literal_body_returns_both_resident_lexical_actions":
            len(observed) == 5
            and observed[0] == (expected_surfaces[0],)
            and not observed[1]
            and observed[2] == (expected_surfaces[1],)
            and not observed[3],
        "heldout_question_recomposes_causal_discourse_with_new_words":
            len(observed) == 5 and expected_composition in observed[4]
            and expected_surfaces[0] in expected_composition
            and expected_surfaces[1] in expected_composition,
        "successful_actions_revise_same_resident_world_coordinates":
            cause == expected_surfaces[0] and effect == expected_surfaces[1],
        "adverse_action_does_not_promote_hypothesis":
            bool(adverse_surface) and adverse_ok and adverse_before == adverse_after,
        "pending_action_restart_rejects_wrong_yoked_and_source_less_returns":
            qualified_receipt is not None and not wrong_session
            and not zero_source and not yoked,
        "withdrawn_hypothesis_cannot_borrow_later_consequence":
            not withdrawn_return,
        "checkpoint_retains_no_teaching_contact_or_heldout_question":
            all(contact.decode() not in checkpoint_text for contact in CONTACTS)
            and question.decode() not in checkpoint_text,
        "transport_contains_no_curriculum_or_expected_language":
            all(contact.decode() not in mcp_source for contact in CONTACTS)
            and all(surface.decode() not in mcp_source
                    for surface in expected_surfaces),
        "no_lexical_specific_consequence_lane_remains":
            "lexical_consequence" not in Path(__file__).with_name(
                "reference_life_function_curriculum_v1.py").read_text(),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-lexical-action-learning.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "resident_lexical_actions": tuple(x.decode() for x in expected_surfaces),
        "heldout_question": question.decode(),
        "resident_composition": expected_composition.decode(),
        "literal_observed_actions": tuple(
            tuple(x.decode() for x in row) for row in observed),
        "post_process_stages": stages,
        "gateway_error_tail": gateway_errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "language_phenotype_improved": not failed,
        "visible_language_gain":
            "LITERAL_CLAUDE_CONSEQUENCES_REVISE_RESIDENT_WORDS_AND_CAUSAL_COMPOSITION",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
