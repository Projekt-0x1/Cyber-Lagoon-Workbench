#!/usr/bin/env python3
"""RED: resident causal common ground must survive one unrelated topic turn."""
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
from reference_claude_body_causal_uptake_verify import (
    _focus_and_paraphrase, _state_contact,
)
from reference_language_mastery_claude_gateway_v1 import (
    CLAUDE_SILENCE_FRAME, body_source_identity,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude
from reference_slow_resource_history_v1 import (
    LOAD_SAMPLE_CAP_Q16, SUSTAINED_MIN_CONTACTS,
)


def _restore(path):
    return ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), json.loads(path.read_text()))


def _receipt(runtime, surface, channel):
    digest = hashlib.sha256(surface).hexdigest()
    rows = {int(row.identity): row for row in (
        *runtime.adult.pending_causal_dialogue_actions.values(),
        *runtime.adult.recent_causal_dialogue_actions.values())
        if int(row.source) == channel and int(row.channel) == channel
        and row.surface_digest == digest}
    return max(rows.values(), key=lambda row: (int(row.born_tick),
                                               int(row.identity))) if rows else None


def _public(row):
    if row[2]:
        return row[2][0]
    for result in reversed(row[4]):
        surface = result.get("result")
        if isinstance(surface, str) and surface != CLAUDE_SILENCE_FRAME:
            return surface.encode()
    return b""


def _direct_return(base, question, state_prompt, paraphrase, *, settle=True,
                   observation_channel=None, lesion=False, pressure=False):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    channel = body_source_identity("topic-return-control")
    observation_channel = int(channel if observation_channel is None
                              else observation_channel)
    _first, first_action = runtime.contact_utterance(question, channel, channel)
    first_settled = bool(settle and first_action and
                         runtime.settle_contact_consequence(
                             first_action, channel, 0, 0, True))
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        state_prompt, channel, channel)
    _state, state_action = runtime.contact_utterance(
        state_prompt, channel, channel)
    if state_action:
        runtime.settle_contact_consequence(state_action, channel, 0, 0, True)
    if lesion:
        runtime.adult.recent_causal_dialogue_actions.clear()
        runtime.adult._settled_causal_action_lineage.clear()
        runtime.adult._settled_causal_action_lineage_index.clear()
    if pressure:
        for sequence in range(1, SUSTAINED_MIN_CONTACTS + 1):
            runtime.adult.language_adult.settle_body_ingress(
                "topic-return-pressure", sequence, format(sequence, "064x"),
                LOAD_SAMPLE_CAP_Q16)
    changed = runtime.adult.observe_authenticated_causal_dialogue_contact(
        paraphrase, observation_channel, observation_channel)
    surface, action = runtime.contact_utterance(
        paraphrase, observation_channel, observation_channel)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    return {
        "first_settled": first_settled, "changed": changed,
        "surface": bytes(surface), "receipt": receipt,
        "programs": 0 if receipt is None else len(receipt.programs),
    }


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    literal_receipts = []
    gateway_errors = checkpoint_text = ""
    initial_receipt = final_receipt = resumed_receipt = None
    final_coordinates = resumed_coordinates = ()
    with tempfile.TemporaryDirectory(prefix="foundry-claude-topic-return-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        _effect, question, paraphrase, *_ = _focus_and_paraphrase(base)
        state_prompt, _state_sources = _state_contact(base)
        direct = _direct_return(base, question, state_prompt, paraphrase)
        unsettled = _direct_return(
            base, question, state_prompt, paraphrase, settle=False)
        wrong_session = _direct_return(
            base, question, state_prompt, paraphrase,
            observation_channel=body_source_identity("topic-return-other"))
        lesioned = _direct_return(
            base, question, state_prompt, paraphrase, lesion=True)
        pressured = _direct_return(
            base, question, state_prompt, paraphrase, pressure=True)
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
                channel = body_source_identity("literal-body")
                for prompt in (question, state_prompt, paraphrase,
                               question, paraphrase):
                    row = _claude(binary, environment, config, prompt, session or None)
                    rows.append(row)
                    session = row[1]
                    snapshot = _restore(checkpoint)
                    literal_receipts.append(_receipt(
                        snapshot, _public(row), channel))
                # A new Claude process/session is a new transport episode over
                # the same authenticated body and checkpointed Adult.
                rows.append(_claude(binary, environment, config, question, None))
                runtime = _restore(checkpoint)
                literal_receipts.append(_receipt(
                    runtime, _public(rows[-1]), channel))
                surfaces = tuple(_public(row) for row in rows)
                initial_receipt = literal_receipts[0]
                final_receipt = literal_receipts[2]
                final_coordinates = (() if final_receipt is None else
                                     runtime.adult._causal_action_coordinates(
                                         final_receipt))
                resumed_receipt = literal_receipts[5]
                resumed_coordinates = (() if resumed_receipt is None else
                                       runtime.adult._causal_action_coordinates(
                                           resumed_receipt))
                checkpoint_text = checkpoint.read_text()
        finally:
            if gateway is not None:
                gateway.terminate(); gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-2048:]

    surfaces = tuple(_public(row) for row in rows)
    sessions = tuple(row[1] for row in rows)
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "five_processes_build_history_then_fresh_session_resumes_same_body":
            len(rows) == 6 and all(row[0] == 0 for row in rows)
            and len(set(sessions[:5])) == 1 and bool(sessions[0])
            and sessions[5] and sessions[5] != sessions[0],
        "initial_causal_action_is_resident_certified":
            initial_receipt is not None and bool(initial_receipt.programs),
        "intervening_body_topic_is_public_and_not_causal_replay":
            len(surfaces) == 6 and bool(surfaces[1])
            and surfaces[1] not in (surfaces[0], surfaces[2]),
        "earlier_common_ground_recruits_distinct_certified_return":
            final_receipt is not None and bool(final_receipt.programs)
            and surfaces[2] and surfaces[2] != surfaces[0],
        "return_remains_bound_to_resident_causal_coordinates":
            bool(final_coordinates),
        "fresh_transport_session_retains_learned_partner_discourse":
            resumed_receipt is not None and bool(resumed_coordinates)
            and surfaces[5] == surfaces[4] and surfaces[5] != surfaces[0],
        "direct_same_history_reproduces_interrupted_return":
            direct["first_settled"] and direct["programs"] > 0
            and direct["surface"] == surfaces[2],
        "unsettled_action_cannot_author_topic_return":
            not unsettled["first_settled"] and unsettled["programs"] == 0,
        "wrong_session_cannot_borrow_earlier_common_ground":
            wrong_session["programs"] == 0,
        "lineage_lesion_abolishes_return": lesioned["programs"] == 0,
        "resource_pressure_modulates_without_inventing_truth":
            pressured["programs"] > 0
            and pressured["programs"] <= direct["programs"],
        "checkpoint_contains_no_contacts_or_public_surfaces":
            all(raw.decode(errors="replace") not in checkpoint_text
                for raw in (question, state_prompt, paraphrase, *surfaces) if raw),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-topic-interruption-return.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "public_surfaces": tuple(surface.decode(errors="replace") for surface in surfaces),
        "resident_programs": (
            0 if initial_receipt is None else len(initial_receipt.programs),
            0 if final_receipt is None else len(final_receipt.programs),
            0 if resumed_receipt is None else len(resumed_receipt.programs),
        ),
        "direct_control_programs": {
            "intact": direct["programs"], "unsettled": unsettled["programs"],
            "wrong_session": wrong_session["programs"],
            "lineage_lesion": lesioned["programs"],
            "pressure": pressured["programs"],
        },
        "gateway_error_tail": gateway_errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "language_phenotype_improved": not failed,
        "visible_language_gain":
            "LEARNED_CAUSAL_COMMON_GROUND_SURVIVES_TOPIC_INTERRUPTION_AND_FRESH_CLAUDE_SESSION",
    }, indent=2))
    raise SystemExit(bool(failed))
if __name__ == "__main__":
    main()
