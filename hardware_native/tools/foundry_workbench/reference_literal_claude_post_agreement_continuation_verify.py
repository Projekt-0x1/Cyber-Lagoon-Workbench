#!/usr/bin/env python3
"""One literal Claude body carries learned, somatic, and causal life across restart."""
from __future__ import annotations

import copy
import hashlib
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
    body_source_identity, save_runtime,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import (
    _claude as _tool_claude,
)
from reference_literal_claude_lexical_action_learning_verify import (
    CONTACTS as LEXICAL_EXPERIENCE,
)
from reference_settled_intention_causal_continuation_verify import REQUEST

PRESENCE_CONTACT = b"I am still here."


def _restore(path):
    return ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), json.loads(path.read_text()))


def _causal_topic_questions(runtime):
    """Select broad topology contrasts; learned world state supplies their content."""
    candidates = []
    for row in runtime.adult.language_adult.world_causal_learning.current_resolutions():
        effect = int(row[3])
        proposition = bytes(runtime.adult.language_adult._leaf_surface(effect) or b"")
        closure = tuple(runtime.adult.causal_focus_rows(effect))
        if proposition and closure:
            question = (b"why is it the case that " +
                        proposition.rstrip(b".?").lower() + b"?")
            candidates.append((len(closure), effect, question, int(row[5])))
    candidates = sorted(set(candidates), reverse=True)
    if len(candidates) < 3:
        return ()
    # Maximum and median closures cover breadth; the most recently consequence-
    # settled closure proves that the continuing Life exposes later acquisition.
    selected = (candidates[0], candidates[len(candidates) // 2],
                max(candidates, key=lambda row: (row[3], row[1])))
    return (tuple(row[:3] for row in selected)
            if len({row[1] for row in selected}) == 3 else ())


def _direct_arm(base, contact, observation_channel, lesion=False):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    _effect, question, _paraphrase, causal_receipt, _reversed = \
        _focus_and_paraphrase(runtime)
    action_channel = body_source_identity("literal-control-action")
    first, action = runtime.contact_utterance(
        question, action_channel, action_channel)
    settled = runtime.settle_contact_consequence(
        action, action_channel, 0, 0, True)
    if lesion:
        runtime.adult.recent_causal_dialogue_actions.clear()
        runtime.adult._settled_causal_action_lineage.clear()
        runtime.adult._settled_causal_action_lineage_index.clear()
    changed = runtime.adult.observe_authenticated_causal_dialogue_contact(
        contact, observation_channel, observation_channel)
    surface, identity = runtime.contact_utterance(
        contact, observation_channel, observation_channel)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(identity)
    coordinates = (() if receipt is None else
                   runtime.adult._causal_action_coordinates(receipt))
    return {
        "first": first,
        "settled": settled,
        "changed": changed,
        "surface": surface,
        "programs": 0 if receipt is None else len(receipt.programs),
        "causal_receipts": tuple(sorted(int(row[0]) for row in coordinates)),
        "accepted_receipt": causal_receipt,
        "uptake": runtime.adult.causal_dialogue_uptake_support(
            observation_channel, causal_receipt),
        "dispute": runtime.adult.causal_dialogue_dispute_support(
            observation_channel, causal_receipt),
    }


def _apply_authenticated_body_pressure(runtime):
    """Deliver bounded nonlinguistic load through the incumbent body ingress."""
    source = "literal-claude-somatic-body"
    for sequence in range(1, 7):
        reafference = hashlib.sha256(
            f"{source}:{sequence}:load".encode()).hexdigest()
        runtime.adult.language_adult.settle_body_ingress(
            source, sequence, reafference, 1 << 15)
    return runtime.adult.language_adult.slow_resource_history.pressure_q16()


def _role_local_inquiry_arm(base):
    """Keep distinct commitments concurrent while inhibiting same-role repeats."""
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    channel = body_source_identity("literal-role-local-control")
    pressure = _apply_authenticated_body_pressure(runtime)
    state_surface = runtime.quiet_public_opportunity(channel, channel)
    state_digest = hashlib.sha256(state_surface).hexdigest()
    state_receipt = next((receipt for receipt in
                          runtime.adult.pending_endogenous_inquiry_actions.values()
                          if receipt.surface_digest == state_digest), None)
    state_return = bool(state_receipt and
                        runtime.adult.settle_endogenous_inquiry_motor_return(
                            state_receipt, channel, True))
    _effect, question, _paraphrase, _causal_receipt, reversal = \
        _focus_and_paraphrase(runtime)
    _surface, action = runtime.contact_utterance(question, channel, channel)
    causal_return = runtime.settle_contact_consequence(
        action, channel, 0, 0, True)
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        reversal, channel, channel)
    immediate, _identity = runtime.contact_utterance(
        reversal, channel, channel)
    repair = runtime.quiet_public_opportunity(channel, channel)
    roles = runtime.adult._pending_endogenous_inquiry_roles(channel)
    restored = type(base).restore(
        base.program, copy.deepcopy(runtime.checkpoint()))
    restored_roles = restored.adult._pending_endogenous_inquiry_roles(channel)
    duplicate = restored.quiet_public_opportunity(channel, channel)
    return {
        "pressure": pressure, "state_surface": bytes(state_surface),
        "state_return": state_return, "causal_return": causal_return,
        "immediate": bytes(immediate), "repair": bytes(repair),
        "roles": tuple(sorted(roles)),
        "restored_roles": tuple(sorted(restored_roles)),
        "duplicate": bytes(duplicate),
    }


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    stages = []
    uptake = dispute = pressure = 0
    question = paraphrase = b""
    state_question = reversal = b""
    topics = ()
    checkpoint_text = gateway_errors = ""
    correct_control = reversal_control = other_control = lesion_control = {}
    actionless_change = actionless_surface = actionless_identity = 0
    transient_absent = restored_transient_surface = delayed_transient_surface = False
    before_words = after_words = ()
    role_local = {}
    pre_silence_roles = post_silence_roles = ()
    state_resolution = False
    with tempfile.TemporaryDirectory(prefix="foundry-literal-claude-continuation-") as directory:
        root = Path(directory)
        build_cache(directory)
        # Start before the two held-out lexical relations are acquired.  This is
        # one earlier moment of the same canonical Life, not a hand-built fixture.
        base = load_mark(directory, "lexical_carrier_ready")
        role_local = _role_local_inquiry_arm(base)
        _effect, question, paraphrase, causal_receipt, reversal = \
            _focus_and_paraphrase(base)
        channel_a = body_source_identity("literal-control-action")
        channel_b = body_source_identity("literal-control-other")
        correct_control = _direct_arm(base, paraphrase, channel_a)
        reversal_control = _direct_arm(base, reversal, channel_a)
        other_control = _direct_arm(base, paraphrase, channel_b)
        lesion_control = _direct_arm(base, paraphrase, channel_a, lesion=True)
        actionless = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        actionless_change = actionless.adult.observe_authenticated_causal_dialogue_contact(
            paraphrase, channel_a, channel_a)
        actionless_surface, actionless_identity = actionless.contact_utterance(
            paraphrase, channel_a, channel_a)
        transient = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        _initial, transient_action = transient.contact_utterance(
            question, channel_a, channel_a)
        transient.settle_contact_consequence(
            transient_action, channel_a, 0, 0, True)
        transient.adult.observe_authenticated_causal_dialogue_contact(
            paraphrase, channel_a, channel_a)
        transient_checkpoint = transient.checkpoint()
        transient_text = json.dumps(transient_checkpoint, sort_keys=True)
        transient_absent = "last_causal_dialogue_contact_continuations" not in transient_text
        restored_transient = type(base).restore(
            base.program, copy.deepcopy(transient_checkpoint))
        restored_transient_surface = not restored_transient.contact_utterance(
            paraphrase, channel_a, channel_a)[0]
        delayed_transient = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        _delayed_initial, delayed_action = delayed_transient.contact_utterance(
            question, channel_a, channel_a)
        delayed_transient.settle_contact_consequence(
            delayed_action, channel_a, 0, 0, True)
        delayed_transient.adult.observe_authenticated_causal_dialogue_contact(
            paraphrase, channel_a, channel_a)
        delayed_transient.adult.language_adult._advance()
        delayed_transient_surface = not delayed_transient.contact_utterance(
            paraphrase, channel_a, channel_a)[0]
        checkpoint = root / "adult.json"
        checkpoint.write_text(json.dumps(
            base.checkpoint(), separators=(",", ":"), sort_keys=True))
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"agi_body": {
            "type": "stdio", "command": sys.executable,
            "args": [str(Path(__file__).with_name(
                "reference_claude_body_action_mcp_v1.py"))],
        }}}))
        gateways = []
        try:
            if binary:
                def start_gateway():
                    gateway = subprocess.Popen(
                        (sys.executable, str(Path(__file__).with_name(
                            "reference_language_mastery_claude_gateway_v1.py")),
                         "--resume", str(checkpoint), "--auth-token", "literal-body"),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    gateways.append(gateway)
                    port = int(gateway.stdout.readline().split()[-1])
                    environment = os.environ.copy()
                    environment.update({
                        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                        "ANTHROPIC_AUTH_TOKEN": "literal-body",
                        "ANTHROPIC_API_KEY": "literal-body",
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                        "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                    })
                    return gateway, environment

                # First sitting: consequences of the Adult's own clarification
                # actions revise two relations.  The repeated contacts are the
                # partner's answers; tool success alone cannot settle semantics.
                gateway, environment = start_gateway()
                session = ""
                for prompt in (LEXICAL_EXPERIENCE[0], LEXICAL_EXPERIENCE[0],
                               LEXICAL_EXPERIENCE[1], LEXICAL_EXPERIENCE[1]):
                    row = _tool_claude(
                        binary, environment, config, prompt, session or None)
                    rows.append(row)
                    session = row[1]
                    if row[0] != 0:
                        break
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors += gateway.stderr.read()[-1024:]

                learned = _restore(checkpoint)
                before_words = tuple(bytes(base.adult.language_adult.leaf(
                    100, atoms).surface) for atoms in ((0xA103,), (0xA105,)))
                after_words = tuple(bytes(learned.adult.language_adult.leaf(
                    100, atoms).surface) for atoms in ((0xA103,), (0xA105,)))

                # Between sittings the same organism receives authenticated body
                # load, not a semantic prompt.  Its learned state construction then
                # competes for the next genuinely quiet public opportunity.
                pressure = _apply_authenticated_body_pressure(learned)
                partner = body_source_identity("literal-body")
                state_question = bytes(learned.adult.endogenous_state_inquiry(
                    partner) or b"")
                _effect, question, paraphrase, causal_receipt, reversal = \
                    _focus_and_paraphrase(learned)
                topics = _causal_topic_questions(learned)
                if not topics or topics[0][2] != question:
                    raise RuntimeError("literal-dialogue:topic-frontier")
                save_runtime(checkpoint, learned)

                # Second sitting: causal explanation, accepted common ground,
                # deictic continuation, topic interruption, resident body-state
                # inquiry, contradiction repair, then a fresh Claude session.
                gateway, environment = start_gateway()
                session = ""
                conversation = (
                    PRESENCE_CONTACT, question, paraphrase, REQUEST, topics[1][2],
                    topics[2][2], reversal,
                )
                for prompt in conversation:
                    row = _tool_claude(
                        binary, environment, config, prompt, session or None)
                    rows.append(row)
                    session = row[1]
                    restored = _restore(checkpoint)
                    stages.append({
                        "pressure_q16": restored.adult.language_adult.
                            slow_resource_history.pressure_q16(),
                        "inquiry_public_count":
                            restored.adult.endogenous_inquiry_public_count,
                        "pending_inquiry": len(
                            restored.adult.pending_endogenous_inquiry_actions),
                        "settled_causal_coordinates": sum(map(
                            len, restored.adult._settled_causal_action_lineage_index.
                            values())),
                    })
                    if row[0] != 0:
                        break
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors += gateway.stderr.read()[-1024:]

                # No contact is not nonexistence.  The same checkpoint receives
                # resident time, rematerializes transient work, and recovers from
                # recent body pressure before another transport attaches.
                resting = _restore(checkpoint)
                pre_silence_roles = tuple(sorted(
                    resting.adult._pending_endogenous_inquiry_roles(partner)))
                for _ in range(64):
                    resting.adult.language_adult.internal_tick()
                    resting.adult.resident_silent_wave()
                state_receipts = tuple(receipt for receipt in
                    resting.adult.pending_endogenous_inquiry_actions.values()
                    if int(receipt.channel) == partner
                    and resting.adult._endogenous_inquiry_role(receipt) == "state")
                state_resolution = bool(len(state_receipts) == 1 and
                    resting.adult.settle_endogenous_inquiry_resolution(
                        state_receipts[0], partner))
                post_silence_roles = tuple(sorted(
                    resting.adult._pending_endogenous_inquiry_roles(partner)))
                save_runtime(checkpoint, resting)

                # Third sitting: a matched causal question exposes body-history
                # recovery, followed by a held-out learned lexical relation.
                if len(rows) == 11:
                    gateway, environment = start_gateway()
                    recovered = _tool_claude(
                        binary, environment, config, question, None)
                    rows.append(recovered)
                    rows.append(_tool_claude(
                        binary, environment, config, topics[2][2], recovered[1]))
                runtime = _restore(checkpoint)
                uptake = runtime.adult.causal_dialogue_uptake_support(
                    partner, causal_receipt)
                dispute = runtime.adult.causal_dialogue_dispute_support(
                    partner, causal_receipt)
        finally:
            for gateway in gateways:
                if gateway.poll() is None:
                    gateway.terminate()
                    gateway.wait(timeout=3)
                if gateway.stderr is not None:
                    gateway_errors += gateway.stderr.read()[-1024:]
        checkpoint_text = checkpoint.read_text()

    codes = tuple(row[0] for row in rows)
    action_groups = tuple(row[2] for row in rows)
    surfaces = tuple(surface for group in action_groups for surface in group)
    sessions = tuple(row[1] for row in rows)
    lexical_groups = action_groups[:4]
    dialogue_groups = action_groups[4:]
    body_inquiry_seen = any(state_question in group for group in dialogue_groups)
    clarification = (dialogue_groups[-3][-1] if len(dialogue_groups) >= 3
                     and dialogue_groups[-3] else b"")
    recovered_surface = (dialogue_groups[-2][-1] if len(dialogue_groups) >= 2
                         and dialogue_groups[-2] else b"")
    fresh_surface = (dialogue_groups[-1][-1] if dialogue_groups
                     and dialogue_groups[-1] else b"")
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "all_literal_processes_complete": bool(rows and all(code == 0 for code in codes)),
        "one_adult_learns_two_relations_before_broad_dialogue": bool(
            len(lexical_groups) == 4
            and tuple(map(len, lexical_groups)) == (1, 0, 1, 0)
            and before_words != after_words
            and all(new in surfaces for new in after_words)),
        "restart_preserves_partner_but_not_transport_context": bool(
            len(sessions) == 13 and len(set(sessions[:4])) == 1
            and len(set(sessions[4:11])) == 1
            and sessions[0] != sessions[4]
            and len(set(sessions[11:])) == 1
            and sessions[11] not in (sessions[0], sessions[4])),
        "authenticated_body_history_recruits_resident_inquiry": bool(
            pressure > 0 and state_question and body_inquiry_seen
            and any(stage["inquiry_public_count"] > 0 for stage in stages)),
        "role_local_commitments_survive_restart_without_duplicate_storm": bool(
            role_local["pressure"] > 0 and role_local["state_surface"]
            and role_local["state_return"] and role_local["causal_return"]
            and not role_local["immediate"] and role_local["repair"].endswith(b"?")
            and role_local["roles"] == ("causal", "state")
            and role_local["restored_roles"] == role_local["roles"]
            and not role_local["duplicate"]),
        "causal_discourse_survives_learning_body_state_and_interruptions": bool(
            len(dialogue_groups) == 9 and uptake >= 1
            and len({surface for group in dialogue_groups for surface in group}) >= 6
            and sum(len(surface) for group in dialogue_groups for surface in group) > 600),
        "same_adult_crosses_broad_causal_topologies": bool(
            len(topics) == 3 and topics[0][0] > topics[1][0] > topics[2][0]
            and all(any(bytes(question_part).split(b"that ", 1)[-1].rstrip(b"?")
                        in surface.lower() for surface in surfaces)
                    for _depth, _effect, question_part in topics)),
        "contradiction_recruits_resident_clarification_question": bool(
            dispute >= 1 and clarification.endswith(b"?")),
        "silence_recovers_discourse_depth_in_same_adult": bool(
            recovered_surface and len(recovered_surface) >
            max((len(surface) for surface in dialogue_groups[1]), default=0)),
        "recovered_body_evidence_deactivates_only_stale_state_inquiry": bool(
            pre_silence_roles == ("causal", "state") and state_resolution
            and post_silence_roles == ("causal",)),
        "fresh_session_reconstructs_learned_causal_language": bool(
            fresh_surface and any(word in fresh_surface for word in after_words)),
        "checkpoint_contains_no_question_or_partner_transcript": bool(
            question and paraphrase and question.decode() not in checkpoint_text
            and paraphrase.decode() not in checkpoint_text
            and state_question.decode() not in checkpoint_text
            and all(contact.decode() not in checkpoint_text
                    for contact in LEXICAL_EXPERIENCE)
            and PRESENCE_CONTACT.decode() not in checkpoint_text
            and all(topic[2].decode() not in checkpoint_text for topic in topics)
            and REQUEST.decode() not in checkpoint_text
            and reversal.decode() not in checkpoint_text
            and all(surface.decode() not in checkpoint_text for surface in surfaces)),
        "one_valid_acceptance_recruits_only_remaining_certified_programs": bool(
            correct_control["settled"] and correct_control["changed"] == 1
            and correct_control["uptake"] == 1
            and correct_control["programs"] > 0
            and correct_control["accepted_receipt"] not in
                correct_control["causal_receipts"]
            and correct_control["surface"]),
        "reversed_relation_recruits_dispute_not_continuation": bool(
            reversal and reversal_control["settled"]
            and reversal_control["changed"] == 0
            and reversal_control["uptake"] == 0
            and reversal_control["dispute"] >= 1
            and not reversal_control["surface"]),
        "other_session_cannot_continue_the_acted_contribution": bool(
            other_control["settled"] and other_control["changed"] == 0
            and other_control["uptake"] == 0 and not other_control["surface"]),
        "contact_without_resident_action_cannot_recruit_continuation": bool(
            actionless_change == 0 and not actionless_surface
            and actionless_identity == 0),
        "acted_lineage_lesion_selectively_removes_continuation": bool(
            lesion_control["settled"] and lesion_control["changed"] == 0
            and lesion_control["uptake"] == 0 and not lesion_control["surface"]),
        "disposable_continuation_plan_is_not_checkpoint_state": bool(
            transient_absent and restored_transient_surface),
        "intervening_resident_time_expires_disposable_continuation": bool(
            delayed_transient_surface),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema": "cyber-lagoon.literal-claude-continuing-organism.v2",
        "contract": "FOUNDRY_LITERAL_CLAUDE_CONTINUING_ORGANISM_" +
                    ("GREEN" if not failed else "RED"),
        "pass": not failed,
        "language_phenotype_improved": not failed,
        "reference_only": True,
        "runtime_llm": False,
        "claude_binary": binary or "",
        "same_question": question.decode(errors="replace"),
        "causal_topic_questions": [row[2].decode(errors="replace") for row in topics],
        "causal_topic_depths": [row[0] for row in topics],
        "partner_paraphrase": paraphrase.decode(errors="replace"),
        "learned_words_before": [row.decode(errors="replace") for row in before_words],
        "learned_words_after": [row.decode(errors="replace") for row in after_words],
        "resident_state_inquiry": state_question.decode(errors="replace"),
        "body_pressure_q16": pressure,
        "process_exit_codes": list(codes),
        "public_bytes": [len(surface) for surface in surfaces],
        "public_actions_by_process": list(map(len, action_groups)),
        "visible_lived_conversation": [surface.decode(errors="replace") for surface in surfaces],
        "resident_clarification": clarification.decode(errors="replace"),
        "post_silence_recovered_reply": recovered_surface.decode(errors="replace"),
        "pending_roles_across_silence": {
            "before": pre_silence_roles, "after": post_silence_roles,
            "state_resolution": state_resolution,
        },
        "fresh_session_reply": fresh_surface.decode(errors="replace"),
        "uptake_support": uptake,
        "dispute_support": dispute,
        "post_contact_state": stages,
        "role_local_inquiry_control": {
            key: (value.decode(errors="replace") if isinstance(value, bytes) else value)
            for key, value in role_local.items()
        },
        "direct_controls": {
            "correct": {key: value if not isinstance(value, bytes) else len(value)
                        for key, value in correct_control.items()},
            "reversal": {key: value if not isinstance(value, bytes) else len(value)
                         for key, value in reversal_control.items()},
            "other_session": {key: value if not isinstance(value, bytes) else len(value)
                              for key, value in other_control.items()},
            "lineage_lesion": {key: value if not isinstance(value, bytes) else len(value)
                              for key, value in lesion_control.items()},
        },
        "checks": checks,
        "failed": failed,
        "gateway_error_tail": gateway_errors,
        "remaining_red": ["DIRECT_PARITY", "WALL_CLOCK_UNPROMPTED_INITIATIVE",
                          "OPEN_DOMAIN_CONVERSATION"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(result["contract"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
