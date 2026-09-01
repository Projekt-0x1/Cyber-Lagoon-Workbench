#!/usr/bin/env python3
"""Literal Claude RED: a held-out deictic follow-up must recruit lived causality."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import copy
from pathlib import Path

from life_function_factory_v1 import build_cache, load_mark
from reference_claude_body_causal_uptake_verify import _focus_and_paraphrase
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import (
    _claude,
)
from reference_slow_resource_history_v1 import (
    LOAD_SAMPLE_CAP_Q16, SUSTAINED_MIN_CONTACTS,
)


FOLLOWUP = b"What else happens because of that?"
SHUFFLED = b"Because of that, what happens else?"


def _restore(path):
    return ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), json.loads(path.read_text()))


def _pending_programs(runtime, surface, channel):
    digest = hashlib.sha256(surface).hexdigest()
    rows = {int(row.identity): row for row in (
        *runtime.adult.pending_causal_dialogue_actions.values(),
        *runtime.adult.recent_causal_dialogue_actions.values())
        if int(row.source) == channel and int(row.channel) == channel
        and row.surface_digest == digest}
    return 0 if len(rows) != 1 else len(next(iter(rows.values())).programs)


def _direct_followup(base, question, contact=FOLLOWUP, action_channel=1,
                     observation_channel=None, lesion=False, delay=False,
                     pressure=False):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    observation_channel = int(action_channel if observation_channel is None
                              else observation_channel)
    first, first_action = runtime.contact_utterance(
        question, action_channel, action_channel)
    first_settled = runtime.settle_contact_consequence(
        first_action, action_channel, 0, 0, True)
    if lesion:
        runtime.adult.recent_causal_dialogue_actions.clear()
        runtime.adult._settled_causal_action_lineage.clear()
        runtime.adult._settled_causal_action_lineage_index.clear()
    if pressure:
        for sequence in range(1, SUSTAINED_MIN_CONTACTS + 1):
            runtime.adult.language_adult.settle_body_ingress(
                "deictic-pressure", sequence, format(sequence, "064x"),
                LOAD_SAMPLE_CAP_Q16)
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        contact, observation_channel, observation_channel)
    disposable = tuple(runtime.adult.last_causal_dialogue_contact_continuations)
    if delay:
        runtime.adult.language_adult._advance()
    surface, action = runtime.contact_utterance(
        contact, observation_channel, observation_channel)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    programs = 0 if receipt is None else len(receipt.programs)
    wrong_return = (False if receipt is None else runtime.settle_contact_consequence(
        action + 1, observation_channel, 0, 0, True))
    correct_return = (False if receipt is None else runtime.settle_contact_consequence(
        action, observation_channel, 0, 0, True))
    return {
        "first": first, "first_settled": first_settled,
        "surface": surface, "programs": programs,
        "disposable": disposable, "wrong_return": wrong_return,
        "correct_return": correct_return,
        "pressure": runtime.adult.language_adult.slow_resource_history.pressure_q16(),
    }


def _actionless(base, contact, channel):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        contact, channel, channel)
    return runtime.contact_utterance(contact, channel, channel)[0]


def _ambiguous_action(base, effect, contact, channel):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    surface, first = runtime.adult.externalize_causal_component(
        effect, channel, channel)
    second = (None if first is None else runtime.adult.stage_causal_dialogue_action(
        first.programs, surface, channel, channel, factors=first.factors,
        episode=first.episode))
    receipts = tuple(receipt for receipt in (first, second) if receipt is not None)
    settled = tuple(runtime.adult.settle_causal_dialogue_return(
        receipt, channel, 0, 0, True) for receipt in receipts)
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        contact, channel, channel)
    surface = runtime.contact_utterance(contact, channel, channel)[0]
    return len(receipts), all(settled), surface


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    question = b""
    checkpoint_text = gateway_errors = ""
    controls = {}
    with tempfile.TemporaryDirectory(prefix="foundry-literal-claude-followup-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        effect, question, _paraphrase, _receipt, _reversal = \
            _focus_and_paraphrase(base)
        channel = body_source_identity("literal-deictic-control")
        other_channel = body_source_identity("literal-deictic-other")
        controls["correct"] = _direct_followup(
            base, question, action_channel=channel)
        controls["wrong_session"] = _direct_followup(
            base, question, action_channel=channel,
            observation_channel=other_channel)
        controls["lesion"] = _direct_followup(
            base, question, action_channel=channel, lesion=True)
        controls["delay"] = _direct_followup(
            base, question, action_channel=channel, delay=True)
        controls["pressure"] = _direct_followup(
            base, question, action_channel=channel, pressure=True)
        controls["shuffled"] = _direct_followup(
            base, question, contact=SHUFFLED, action_channel=channel)
        controls["actionless"] = _actionless(base, FOLLOWUP, channel)
        controls["ambiguous"] = _ambiguous_action(
            base, effect, FOLLOWUP, channel)
        transient = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        _first, first_action = transient.contact_utterance(
            question, channel, channel)
        transient.settle_contact_consequence(
            first_action, channel, 0, 0, True)
        transient.adult.observe_authenticated_causal_dialogue_contact(
            FOLLOWUP, channel, channel)
        transient_checkpoint = transient.checkpoint()
        restored = type(base).restore(
            base.program, copy.deepcopy(transient_checkpoint))
        controls["transient_checkpoint_absent"] = (
            "last_causal_dialogue_contact_continuations" not in
            json.dumps(transient_checkpoint, sort_keys=True)
            and not restored.contact_utterance(FOLLOWUP, channel, channel)[0])
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
        if binary:
            gateway = subprocess.Popen(
                (sys.executable, str(Path(__file__).with_name(
                    "reference_language_mastery_claude_gateway_v1.py")),
                 "--resume", str(checkpoint), "--auth-token", "literal-body"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
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
                for prompt in (question, FOLLOWUP):
                    code, session, observed, error, _results = _claude(
                        binary, environment, config, prompt, session or None)
                    surface = observed[0] if observed else b""
                    runtime = _restore(checkpoint)
                    channel = body_source_identity("literal-body")
                    rows.append((code, surface,
                                 _pending_programs(runtime, surface, channel), error))
                    if code != 0:
                        break
            finally:
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-1024:]
        checkpoint_text = checkpoint.read_text()

    codes = tuple(row[0] for row in rows)
    surfaces = tuple(row[1] for row in rows)
    programs = tuple(row[2] for row in rows)
    correct = controls["correct"]
    pressure = controls["pressure"]
    ambiguous_count, ambiguous_settled, ambiguous_surface = controls["ambiguous"]
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "two_literal_processes_complete_one_session": codes == (0, 0),
        "initial_question_recruits_resident_causal_action": bool(
            len(surfaces) >= 1 and surfaces[0] and programs[0] > 1),
        "heldout_deictic_followup_recruits_nonreplayed_remainder": bool(
            len(surfaces) == 2 and surfaces[1] and surfaces[1] != surfaces[0]
            and 0 < programs[1] < programs[0]),
        "same_followup_without_resident_action_stays_silent":
            not controls["actionless"],
        "other_session_cannot_borrow_deictic_referent":
            not controls["wrong_session"]["surface"],
        "same_episode_action_ambiguity_stays_silent": bool(
            ambiguous_count > 1 and ambiguous_settled and not ambiguous_surface),
        "shuffled_same_words_do_not_recruit_learned_role":
            not controls["shuffled"]["surface"],
        "recent_action_lineage_lesion_removes_followup":
            not controls["lesion"]["surface"],
        "one_intervening_tick_expires_disposable_projection":
            not controls["delay"]["surface"],
        "sustained_body_pressure_contracts_not_replays_action": bool(
            pressure["pressure"] > 0 and pressure["surface"]
            and 0 < pressure["programs"] < correct["programs"]
            and pressure["surface"] != correct["surface"]),
        "continuation_requires_exact_action_consequence_return": bool(
            correct["surface"] and not correct["wrong_return"]
            and correct["correct_return"]),
        "disposable_projection_is_not_checkpoint_state":
            controls["transient_checkpoint_absent"],
        "checkpoint_contains_neither_question_followup_nor_transcript": bool(
            question and question.decode() not in checkpoint_text
            and FOLLOWUP.decode() not in checkpoint_text
            and "transcript" not in checkpoint_text.lower()),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema": "cyber-lagoon.literal-claude-deictic-followup.v2",
        "contract": "FOUNDRY_LITERAL_CLAUDE_DEICTIC_FOLLOWUP_" +
                    ("GREEN" if not failed else "RED"),
        "pass": not failed,
        "reference_only": True,
        "runtime_llm": False,
        "language_phenotype_improved": not failed,
        "followup": FOLLOWUP.decode(),
        "process_exit_codes": list(codes),
        "public_bytes": [len(surface) for surface in surfaces],
        "resident_programs": list(programs),
        "direct_controls": {
            name: ({key: (len(value) if key in ("first", "surface") else value)
                    for key, value in row.items() if key != "disposable"}
                   if isinstance(row, dict) else
                   ([row[0], row[1], len(row[2])] if isinstance(row, tuple)
                    else (len(row) if isinstance(row, bytes) else row)))
            for name, row in controls.items()
        },
        "checks": checks,
        "failed": failed,
        "gateway_error_tail": gateway_errors,
        "remaining_red": [
            "UNSEEN_FOLLOWUP_FORM_GENERALIZATION",
            "DIRECT_PARITY",
            "BROAD_HUMAN_DIALOGUE",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(result["contract"])
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
