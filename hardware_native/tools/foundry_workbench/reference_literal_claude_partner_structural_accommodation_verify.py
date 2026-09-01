#!/usr/bin/env python3
"""Installed Claude distinguishes motor error from structural partner feedback."""
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
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import _claude
from reference_natural_dialogue_action_consequence_verify import _focus
from reference_partner_structural_accommodation_transfer_verify import _relations


visible_language_gain = True
ATTEMPTS = 2


def _config(path):
    path.write_text(json.dumps({"mcpServers": {"agi_body": {
        "type": "stdio", "command": sys.executable,
        "args": [str(Path(__file__).with_name(
            "reference_claude_body_action_mcp_v1.py"))],
    }}}))


def _arm(root, checkpoint, prompts, binary, label):
    checkpoint_path = root / (label + "-adult.json")
    checkpoint_path.write_text(json.dumps(
        checkpoint, separators=(",", ":"), sort_keys=True))
    config = root / (label + "-mcp.json")
    _config(config)
    rows = []
    gateway_errors = ""
    gateway = subprocess.Popen(
        (sys.executable, str(Path(__file__).with_name(
            "reference_language_mastery_claude_gateway_v1.py")),
         "--resume", str(checkpoint_path), "--auth-token", "literal-body"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = int(gateway.stdout.readline().split()[-1])
        environment = os.environ.copy()
        environment.update({
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_AUTH_TOKEN": "literal-body",
            "ANTHROPIC_API_KEY": "literal-body",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CONFIG_DIR": str(root / (label + "-claude-config")),
        })
        session = ""
        for prompt in prompts:
            row = _claude(binary, environment, config, prompt, session or None)
            rows.append(row)
            session = row[1]
    finally:
        gateway.terminate()
        gateway.wait(timeout=3)
        gateway_errors = gateway.stderr.read()[-2048:]
    text = checkpoint_path.read_text()
    runtime = ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), json.loads(text))
    return tuple(rows), text, runtime, gateway_errors


def _coordinates(runtime, surface, channel):
    digest = hashlib.sha256(bytes(surface)).hexdigest()
    rows = tuple(row for row in runtime.adult.recent_causal_dialogue_actions.values()
                 if int(row.source) == int(channel)
                 and int(row.channel) == int(channel)
                 and row.surface_digest == digest)
    return (() if not rows else
            tuple(runtime.adult._causal_action_coordinates(rows[-1])))


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    with tempfile.TemporaryDirectory(
            prefix="foundry-literal-online-action-repair-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        _effect, question = _focus(base)
        relations, _factors = _relations(base)
        train, heldout = relations
        partner_alternative = train[3][1]
        partner_prompts = (train[2], partner_alternative,
                           partner_alternative, heldout[2])
        checkpoint = copy.deepcopy(base.checkpoint())
        channel = body_source_identity("literal-body")
        baseline = type(base).restore(base.program, copy.deepcopy(checkpoint))
        baseline_surface, baseline_action = baseline.contact_utterance(
            question, channel, channel)
        baseline_receipt = baseline.adult.pending_causal_dialogue_actions.get(
            baseline_action)
        baseline_coordinates = (() if baseline_receipt is None else
            tuple(baseline.adult._causal_action_coordinates(baseline_receipt)))
        heldout_baseline = type(base).restore(
            base.program, copy.deepcopy(checkpoint))
        heldout_surface, heldout_action = heldout_baseline.contact_utterance(
            heldout[2], channel, channel)
        heldout_receipt = heldout_baseline.adult.pending_causal_dialogue_actions.get(
            heldout_action)
        heldout_coordinates = (() if heldout_receipt is None else tuple(
            heldout_baseline.adult._causal_action_coordinates(heldout_receipt)))

        success_rows = partner_rows = ()
        success_text = partner_text = ""
        success_runtime = partner_runtime = None
        success_errors = partner_errors = ""
        if binary:
            success_rows, success_text, success_runtime, success_errors = _arm(
                root, checkpoint, (question,) * ATTEMPTS, binary, "success")
            partner_rows, partner_text, partner_runtime, partner_errors = _arm(
                root, checkpoint, partner_prompts, binary, "partner")

        success_surfaces = tuple(
            row[2][0] if len(row[2]) == 1 else b"" for row in success_rows)
        success_sessions = tuple(row[1] for row in success_rows)
        partner_sessions = tuple(row[1] for row in partner_rows)
        partner_process_surfaces = tuple(row[2] for row in partner_rows)
        partner_final = (b"" if not partner_process_surfaces
                         or len(partner_process_surfaces[-1]) != 1
                         else partner_process_surfaces[-1][0])
        success_coordinates = (() if success_runtime is None else
            _coordinates(success_runtime, success_surfaces[-1], channel))
        partner_coordinates = (() if partner_runtime is None or not partner_final else
            _coordinates(partner_runtime, partner_final, channel))
        actuator_source = Path(__file__).with_name(
            "reference_claude_body_action_mcp_v1.py").read_text()

    checks = {
        "installed_claude_binary_exists": bool(binary),
        "matched_success_and_partner_arms_complete_one_session_each": bool(
            len(success_rows) == ATTEMPTS
            and len(partner_rows) == len(partner_prompts)
            and all(row[0] == 0 for row in (
                *success_rows, *partner_rows))
            and len(set(success_sessions)) == 1
            and len(set(partner_sessions)) == 1
            and success_sessions[0] and partner_sessions[0]),
        "matched_success_contacts_are_resident_public_actions": bool(
            all(success_surfaces)),
        "success_retains_incumbent_certified_formulation": bool(
            baseline_surface and success_surfaces
            and all(surface == bytes(baseline_surface)
                    for surface in success_surfaces)),
        "partner_certified_form_transfers_to_heldout_content": bool(
            partner_final == heldout[3][1]
            and partner_final != bytes(heldout_surface)),
        "partner_change_preserves_resident_causal_coordinates": bool(
            baseline_coordinates and success_coordinates
            and set(success_coordinates).issubset(set(baseline_coordinates))
            and partner_coordinates == heldout_coordinates),
        "checkpoint_keeps_credit_not_transcript_or_surface": bool(
            all(prompt.decode() not in success_text + partner_text
                for prompt in (question, *partner_prompts))
            and all(surface.decode() not in success_text + partner_text
                    for surface in (*success_surfaces, partner_final) if surface)),
        "actuator_outcome_is_content_free": bool(
            question.decode() not in actuator_source
            and bytes(baseline_surface).decode() not in actuator_source
            and "surface" in actuator_source),
        "gateway_has_no_runtime_error": not success_errors.strip()
            and not partner_errors.strip(),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.literal-claude-online-action-repair.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "question": question.decode(),
        "baseline": bytes(baseline_surface).decode(),
        "success_surfaces": [surface.decode() for surface in success_surfaces],
        "success_process_codes": [row[0] for row in success_rows],
        "partner_process_codes": [row[0] for row in partner_rows],
        "partner_prompts": [prompt.decode() for prompt in partner_prompts],
        "partner_process_surfaces": [[surface.decode() for surface in row]
                                     for row in partner_process_surfaces],
        "heldout_before": bytes(heldout_surface).decode(),
        "heldout_after": bytes(partner_final).decode(),
        "baseline_coordinates": baseline_coordinates,
        "success_coordinates": success_coordinates,
        "heldout_coordinates": heldout_coordinates,
        "partner_coordinates": partner_coordinates,
        "gateway_error_tails": {
            "success": success_errors,
            "partner": partner_errors,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime_llm": False,
        "visible_language_gain":
            "PARTNER_CERTIFIED_FORM_FEEDBACK_TRAINS_HELDOUT_INSTALLED_COMPOSITION",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
