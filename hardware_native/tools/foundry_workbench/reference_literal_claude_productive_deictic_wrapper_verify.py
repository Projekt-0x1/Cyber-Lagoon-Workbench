#!/usr/bin/env python3
"""Literal Claude N+1: an unheard wrapper composes with a learned deictic role."""
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
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_literal_claude_deictic_followup_verify import (
    _actionless, _ambiguous_action, _direct_followup,
)
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2, canonical_species_program_v2,
)
from reference_literal_claude_settled_intention_tool_loop_verify import (
    _claude,
)


TARGET = b"Please continue: What else happens because of that?"
MISSING_BOUNDARY = b"Please continue What else happens because of that?"
SHUFFLED = b"What else happens Please continue: because of that?"
WRAPPER_SOURCES = (0xFA41, 0xFA42)
CHILD_SOURCES = (0xFA31, 0xFA32)


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


def _withdraw(base, sources):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    for source in sources:
        runtime.adult.language.withdraw_source(source)
    return runtime


def main():
    started = time.perf_counter()
    binary = os.environ.get("AGI_CLAUDE_BIN") or shutil.which("claude")
    rows = []
    gateway_errors = checkpoint_text = ""
    controls = {}
    with tempfile.TemporaryDirectory(prefix="foundry-productive-deictic-") as directory:
        root = Path(directory)
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        effect, question, _paraphrase, _receipt, _reversal = _focus_and_paraphrase(base)
        channel = body_source_identity("productive-deictic-control")
        other = body_source_identity("productive-deictic-other")
        controls["correct"] = _direct_followup(
            base, question, TARGET, channel)
        controls["missing_boundary"] = _direct_followup(
            base, question, MISSING_BOUNDARY, channel)
        controls["shuffled"] = _direct_followup(
            base, question, SHUFFLED, channel)
        controls["wrong_session"] = _direct_followup(
            base, question, TARGET, channel, observation_channel=other)
        controls["actionless"] = _actionless(base, TARGET, channel)
        controls["ambiguous"] = _ambiguous_action(base, effect, TARGET, channel)
        controls["wrapper_lesion"] = _direct_followup(
            _withdraw(base, WRAPPER_SOURCES), question, TARGET, channel)
        controls["child_lesion"] = _direct_followup(
            _withdraw(base, CHILD_SOURCES), question, TARGET, channel)
        before = load_mark(directory, "relational_surplus_recovered")
        _before_effect, before_question, *_ = _focus_and_paraphrase(before)
        controls["pre_wrapper"] = _direct_followup(
            before, before_question, TARGET, channel)

        checkpoint = root / "adult.json"
        checkpoint.write_text(json.dumps(
            base.checkpoint(), separators=(",", ":"), sort_keys=True))
        checkpoint_text = checkpoint.read_text()
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
                for prompt in (question, TARGET):
                    code, session, observed, error, _results = _claude(
                        binary, environment, config, prompt, session or None)
                    surface = observed[0] if observed else b""
                    runtime = _restore(checkpoint)
                    body_channel = body_source_identity("literal-body")
                    rows.append((code, surface,
                                 _pending_programs(runtime, surface, body_channel), error))
                    if code != 0:
                        break
            finally:
                gateway.terminate()
                gateway.wait(timeout=3)
                gateway_errors = gateway.stderr.read()[-1024:]

    codes = tuple(row[0] for row in rows)
    surfaces = tuple(row[1] for row in rows)
    programs = tuple(row[2] for row in rows)
    correct = controls["correct"]
    ambiguous_count, ambiguous_settled, ambiguous_surface = controls["ambiguous"]
    checks = {
        "installed_claude_binary_exists": bool(binary),
        "two_literal_processes_complete_one_session": codes == (0, 0),
        "unheard_wrapper_child_combination_recruits_continuation": bool(
            correct["surface"] and correct["programs"] > 0
            and len(surfaces) == 2 and surfaces[1] == correct["surface"]
            and len(programs) == 2 and 0 < programs[1] < programs[0]
            and programs[1] == correct["programs"]),
        "exact_target_is_absent_from_checkpoint": TARGET.decode() not in checkpoint_text,
        "wrapper_requires_prior_development": not controls["pre_wrapper"]["surface"],
        "wrapper_evidence_lesion_removes_transfer":
            not controls["wrapper_lesion"]["surface"],
        "child_role_evidence_lesion_removes_transfer":
            not controls["child_lesion"]["surface"],
        "missing_boundary_and_shuffled_order_refuse": bool(
            not controls["missing_boundary"]["surface"]
            and not controls["shuffled"]["surface"]),
        "same_surface_without_action_or_in_other_session_refuses": bool(
            not controls["actionless"]
            and not controls["wrong_session"]["surface"]),
        "same_episode_action_ambiguity_refuses": bool(
            ambiguous_count > 1 and ambiguous_settled and not ambiguous_surface),
        "exact_continuation_return_is_action_bound": bool(
            not correct["wrong_return"] and correct["correct_return"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema": "cyber-lagoon.literal-claude-productive-deictic-wrapper.v1",
        "contract": "FOUNDRY_LITERAL_CLAUDE_PRODUCTIVE_DEICTIC_WRAPPER_" +
                    ("GREEN" if not failed else "RED"),
        "pass": not failed,
        "reference_only": True,
        "runtime_llm": False,
        "language_phenotype_improved": not failed,
        "unheard_contact": TARGET.decode(),
        "process_exit_codes": list(codes),
        "public_bytes": [len(surface) for surface in surfaces],
        "resident_programs": list(programs),
        "checks": checks,
        "failed": failed,
        "gateway_error_tail": gateway_errors,
        "remaining_red": [
            "MULTIPLE_PRODUCTIVE_DEICTIC_CONSTRUCTIONS",
            "LONG_DELAY_AND_INTERRUPTION_GENERALIZATION",
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
