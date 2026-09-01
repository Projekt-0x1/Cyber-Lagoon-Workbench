#!/usr/bin/env python3
"""Reject fixed Claude turn ladders while preserving physical reafference."""
from __future__ import annotations

import inspect
import json
import time

import reference_language_mastery_claude_gateway_v1 as gateway


def refused(messages, cursor, commitment):
    try:
        gateway.claude_contact_suffix(messages, cursor, commitment)
    except ValueError:
        return True
    return False


def main():
    started = time.perf_counter()
    initial = [
        {"role": "user", "content": [
            {"type": "text", "text": "body-state preamble"},
            {"type": "text", "text": "the careful technician tests the dax."},
        ]},
        {"role": "system", "content": "transport control"},
        {"role": "system", "content": "additional transport control"},
    ]
    first_frames, first_cursor = gateway.claude_contact_suffix(initial, 0, "")

    motor = b"where do you think it is?"
    commitment = gateway.reafference_commitment((motor,))
    continued = [
        {"role": "user", "content": "altered inert old prefix"},
        {"role": "system", "content": "altered inert old control"},
        {"role": "system", "content": "another inert old control"},
        {"role": "assistant", "content": motor.decode()},
        {"role": "system", "content": "new control before contact"},
        {"role": "user", "content": [
            {"type": "text", "text": "the sensor"},
            {"type": "text", "text": "is in locker two"},
        ]},
        {"role": "system", "content": "new control after contact"},
        {"role": "system", "content": "another new control"},
    ]
    next_frames, next_cursor = gateway.claude_contact_suffix(
        continued, first_cursor, commitment
    )

    forged = list(continued)
    forged[first_cursor] = {"role": "assistant", "content": "forged motor"}
    extra_assistant = list(continued)
    extra_assistant.insert(first_cursor + 2, {
        "role": "assistant", "content": motor.decode(),
    })
    no_contact = continued[:first_cursor + 1] + [
        {"role": "system", "content": "control only"},
    ]
    nontext = list(continued)
    nontext[first_cursor + 2] = {
        "role": "user",
        "content": [{"type": "tool_result", "content": "semantic side channel"}],
    }
    source_swap = [
        {"role": "assistant", "content": motor.decode()},
        {"role": "user", "content": "old contact"},
    ]

    source = inspect.getsource(gateway)
    fixed_fragments = (
        "if cursor" + "==0:",
        "len(messages)" + "==cursor+3",
        "messages[cursor+2]" + ".get('role')=='system'",
    )
    checks = {
        "initial_body_preamble_is_not_human_contact": (
            first_frames == (b"the careful technician tests the dax.",)
        ),
        "initial_control_row_count_does_not_set_cognitive_phase": first_cursor == 3,
        "variable_width_suffix_preserves_all_new_user_blocks": (
            next_frames == (b"the sensor", b"is in locker two")
            and next_cursor == 8
        ),
        "altered_old_prefix_is_causally_inert": (
            continued[0]["content"] not in tuple(x.decode() for x in next_frames)
        ),
        "own_motor_commitment_is_required": refused(
            forged, first_cursor, commitment
        ),
        "duplicate_boundary_is_refused": refused(
            continued, next_cursor, gateway.reafference_commitment((b"next",))
        ),
        "second_assistant_mind_is_refused": refused(
            extra_assistant, first_cursor, commitment
        ),
        "control_rows_without_external_contact_are_refused": refused(
            no_contact, first_cursor, commitment
        ),
        "nontext_tool_result_cannot_become_cognition": refused(
            nontext, first_cursor, commitment
        ),
        "new_source_cannot_adopt_old_assistant_history": refused(
            source_swap, 0, ""
        ),
        "source_contains_no_fixed_resume_width_ladder": all(
            fragment not in source for fragment in fixed_fragments
        ),
        "bounded_transport_work": time.perf_counter() - started < 0.1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-claude-body-variable-suffix.v1",
        "contract": "FOUNDRY_CLAUDE_BODY_VARIABLE_SUFFIX_GREEN",
        "pass": not failed,
        "reference_only": True,
        "graph_flip": False,
        "visible_body_gain": "VARIABLE_WIDTH_APPEND_ONLY_CLAUDE_CONTACT",
        "first_frames": [frame.decode() for frame in first_frames],
        "next_frames": [frame.decode() for frame in next_frames],
        "checks": checks,
        "failed": failed,
        "remaining_red": ["DIRECT_CONTINUING_ADULT_PARITY"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    if failed:
        print("FOUNDRY_CLAUDE_BODY_VARIABLE_SUFFIX_RED " + ",".join(failed))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(result["contract"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
