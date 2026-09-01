#!/usr/bin/env python3
"""Exact Claude reafference lets resident structural partner history change speech."""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path

from life_function_factory_v1 import build_cache, load_mark
from reference_language_mastery_claude_gateway_v1 import (
    AdultMessagesHandler, BODY_ACTION_INPUT_SCHEMA, BODY_ACTION_TOOL,
    CONTACT_IDENTITY_HEADER, body_source_identity,
)


TOKEN = "claude-body-causal-uptake"
SESSION_A = "claude-body-partner-a"
SESSION_B = "claude-body-partner-b"
BODY_TOOL = [{"name": BODY_ACTION_TOOL,
              "description": "opaque physical action surface",
              "input_schema": BODY_ACTION_INPUT_SCHEMA}]
CONTINUE_CONTACT = b"Please continue: What else happens because of that?"


class Body:
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.server = None
        self.thread = None

    def start(self, runtime):
        self.server = HTTPServer(("127.0.0.1", 0), AdultMessagesHandler)
        self.server.runtime = runtime
        self.server.checkpoint = self.checkpoint
        self.server.auth_token = TOKEN
        self.server.body_identity = TOKEN
        self.server.idle_seconds = 0.001
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.server = None
        self.thread = None

    def post_payload(self, messages, session, tools=None,
                     contact_identity="transport-session"):
        request = {
            "model": "agi",
            "system": "physical Claude Code body envelope",
            "tools": (tools if tools is not None else
                      [{"name": "body", "description": "physical body capability"}]),
            "messages": messages,
        }
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=10)
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer " + TOKEN,
            "x-claude-code-session-id": session,
        }
        if contact_identity == "transport-session":
            headers[CONTACT_IDENTITY_HEADER] = session
        elif contact_identity:
            headers[CONTACT_IDENTITY_HEADER] = contact_identity
        connection.request("POST", "/v1/messages", json.dumps(request), headers)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        connection.close()
        payload = json.loads(raw)
        return status, payload

    def post(self, messages, session, tools=None,
             contact_identity="transport-session"):
        status, payload = self.post_payload(
            messages, session, tools, contact_identity)
        text = ""
        if status == 200:
            text = "".join(block.get("text", "") for block in payload["content"])
        return status, text

    def post_stream(self, messages, session, tools=None):
        request = {"model": "agi", "system": "physical Claude Code body envelope",
                   "tools": tools or BODY_TOOL, "messages": messages, "stream": True}
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=10)
        connection.request("POST", "/v1/messages", json.dumps(request), {
            "content-type": "application/json", "authorization": "Bearer " + TOKEN,
            "x-claude-code-session-id": session,
            CONTACT_IDENTITY_HEADER: session})
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        connection.close()
        return status, raw.decode()


def _initial(question):
    return [{"role": "user", "content": [
        {"type": "text", "text": "opaque Claude Code body state"},
        {"type": "text", "text": question.decode()},
    ]}]


def _continue(messages, prior, contact):
    return [*messages,
            {"role": "assistant", "content": prior},
            {"role": "user", "content": contact.decode()}]


def _focus_and_paraphrase(runtime):
    adult = runtime.adult
    candidates = []
    for row in adult.language_adult.world_causal_learning.current_resolutions():
        effect = int(row[3])
        surface = bytes(adult.language_adult._leaf_surface(effect) or b"")
        closure = tuple(adult.causal_focus_rows(effect))
        if surface and closure:
            candidates.append((len(closure), effect, surface, closure))
    if not candidates:
        return 0, b"", b"", 0, b""
    _depth, effect, proposition, rows = max(candidates)
    first = rows[0]
    factors = tuple(adult._causal_self_contained_factors())
    selected = next((factor for factor in factors
                     if adult.causal_program_for_row(first, factor, False) > 0), 0)
    alternatives = tuple(factor for factor in factors if factor != selected)
    paraphrase = (bytes(adult._causal_self_contained_surface(first, alternatives[0]) or b"")
                  if len(alternatives) == 1 else b"")
    factor = alternatives[0] if len(alternatives) == 1 else 0
    pieces = adult.language.historical_span_pieces(factor) if factor else ()
    orientation = adult.language_adult.world_causal_learning.grounding.orientation(factor)
    cause, row_effect = int(first[2]), int(first[3])
    children = ((row_effect, cause) if orientation > 0 else (cause, row_effect))
    reversed_relation = (bytes(adult.language_adult._render_pieces(
        pieces, tuple(tuple(adult.language_adult._leaf_surface(child))
                      for child in reversed(children)))) if pieces else b"")
    question = b"why is it the case that " + proposition.rstrip(b".?").lower() + b"?"
    return effect, question, paraphrase, int(first[4]), reversed_relation


def _action_for_surface(runtime, surface, channel):
    digest = hashlib.sha256(surface).hexdigest()
    rows = tuple(row for row in runtime.adult.pending_causal_dialogue_actions.values()
                 if row.channel == channel and row.surface_digest == digest)
    return rows[0] if len(rows) == 1 else None


def _state_contact(runtime):
    """Recover one learned public-state affordance without naming its wording."""
    adult = runtime.adult
    target = adult.language_adult._somatic_appraisal_language_context()
    candidates = []
    for (space, candidate), sources in adult._context_affordance_sources.items():
        if candidate != target:
            continue
        live = tuple(source for source in sources
                     if source not in adult.language._withdrawn)
        binding = adult._relation_language_spaces.get(space)
        if len(live) < 2 or binding is None:
            continue
        try:
            surface = bytes(adult.language_adult.leaf(*binding).surface)
        except RuntimeError:
            continue
        candidates.append((surface, tuple(live)))
    return candidates[0] if len(candidates) == 1 else (b"", ())


def _run_state_arm(runtime, checkpoint, session=SESSION_A, prompt=None):
    resident_prompt, _sources = _state_contact(runtime)
    prompt = bytes(resident_prompt if prompt is None else prompt)
    bindings = runtime.adult.language_adult.language.invert_surface(prompt)
    expected_target = (runtime.adult.context_affordance(
        bindings[0].context, bindings[0].atoms) if len(bindings) == 1 else 0)
    expected = bytes(runtime.adult.current_context_surface(
        expected_target, body_source_identity(session)) or b"")
    body = Body(checkpoint).start(runtime)
    messages = _initial(prompt)
    status, surface = body.post(messages, session)
    status_2, surface_2 = ((0, "") if not surface else body.post(
        _continue(messages, surface, prompt), session))
    checkpoint_text = Path(checkpoint).read_text()
    subject = body.server.runtime
    body.stop()
    restored = type(subject).restore(subject.program, json.loads(checkpoint_text))
    return {
        "status": status, "status_2": status_2, "prompt": prompt,
        "expected": expected, "surface": surface.encode(),
        "surface_2": surface_2.encode(), "checkpoint_text": checkpoint_text,
        "runtime": subject,
        "checkpoint_pending_context_actions": len(
            restored.adult.pending_context_affordance_actions),
    }


def _channel_control_state(runtime, session=SESSION_A):
    adult = runtime.adult
    channel = body_source_identity(session)
    context = int(adult._causal_dialogue_context_by_channel.get(channel, 0))
    totals = [0, 0, 0, 0, 0]
    for profile in adult.language_adult.credit.rows.values():
        local = profile.contexts.get(context)
        if local is None:
            continue
        totals[0] += int(local.control_attempts)
        totals[1] += int(local.control_successes)
        totals[2] += int(local.background_attempts)
        totals[3] += int(local.background_successes)
        totals[4] += int(local.control_history_q16)
    return tuple(totals)


def _run_partner_control_arm(runtime, checkpoint, effect, confirmed):
    """Live matched resident actions; only independent consequences differ."""
    adult = runtime.adult
    channel = body_source_identity(SESSION_A)
    actions = []
    last = None
    for _ in range(2):
        surface, receipt = adult.externalize_causal_component(
            effect, channel, channel)
        if not surface or receipt is None:
            raise RuntimeError("claude-body:control-action-refused")
        actions.append(bytes(surface))
        if not adult.settle_causal_dialogue_return(
                receipt, channel, 1 << 16, 0, bool(confirmed)):
            raise RuntimeError("claude-body:control-return-refused")
        last = receipt
    if not adult.observe_causal_dialogue_background(last.identity, channel, False):
        raise RuntimeError("claude-body:control-background-refused")
    Path(checkpoint).write_text(json.dumps(runtime.checkpoint()))
    row = _run_state_arm(runtime, checkpoint)
    row["actions"] = tuple(actions)
    row["control"] = _channel_control_state(row["runtime"])
    restored = type(runtime).restore(runtime.program, json.loads(row["checkpoint_text"]))
    prompt, _sources = _state_contact(restored)
    bindings = restored.adult.language_adult.language.invert_surface(prompt)
    target = restored.adult.context_affordance(
        bindings[0].context, bindings[0].atoms) if len(bindings) == 1 else 0
    row["restored_surface"] = bytes(restored.adult.current_context_surface(
        target, channel) or b"")
    return row


def _run_tool_result_arm(runtime, checkpoint, question, is_error, restart=True,
                         tool_id_override=None, session=SESSION_A):
    """Literal Anthropic client-tool loop over one resident public-action ticket."""
    state_prompt, _sources = _state_contact(runtime)
    body = Body(checkpoint).start(runtime)
    messages = _initial(question)
    first_status, first = body.post_payload(messages, session, BODY_TOOL)
    blocks = first.get("content", ()) if first_status == 200 else ()
    tool = blocks[0] if len(blocks) == 1 and blocks[0].get("type") == "tool_use" else {}
    pending_before = len(body.server.runtime.adult.pending_causal_dialogue_actions)
    first_checkpoint = Path(checkpoint).read_text()
    if restart:
        body.stop()
        restored = type(runtime).restore(runtime.program, json.loads(first_checkpoint))
        body = Body(checkpoint).start(restored)
    returned_id = tool_id_override if tool_id_override is not None else tool.get("id", "")
    continued = [*messages, {"role": "assistant", "content": blocks},
                 {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": returned_id,
                      "content": "opaque body result", "is_error": bool(is_error)},
                     {"type": "text", "text": state_prompt.decode()},
                 ]}]
    final_status, final = body.post_payload(continued, session, BODY_TOOL)
    final_surface = (b"" if final_status != 200 else "".join(
        block.get("text", "") for block in final.get("content", ())).encode())
    subject = body.server.runtime
    final_checkpoint = Path(checkpoint).read_text()
    pending_after = len(subject.adult.pending_causal_dialogue_consequences)
    control = _channel_control_state(subject, session)
    body.stop()
    restored = type(subject).restore(subject.program, json.loads(final_checkpoint))
    restored_prompt, _ = _state_contact(restored)
    bindings = restored.adult.language_adult.language.invert_surface(restored_prompt)
    target = (restored.adult.context_affordance(
        bindings[0].context, bindings[0].atoms) if len(bindings) == 1 else 0)
    restored_surface = bytes(restored.adult.current_context_surface(
        target, body_source_identity(session)) or b"")
    return {
        "statuses": (first_status, final_status), "tool": tool,
        "first_stop": first.get("stop_reason", ""), "surface": final_surface,
        "control": control, "pending_before": pending_before,
        "pending_after": pending_after, "first_checkpoint": first_checkpoint,
        "final_checkpoint": final_checkpoint, "runtime": subject,
        "continued": continued, "restored_surface": restored_surface,
    }


def _run_settled_tool_chain(runtime, checkpoint, question, request=True,
                            fail_requested=False):
    """Drive only Anthropic alternation while resident state owns every successor."""
    body = Body(checkpoint).start(runtime)
    messages = _initial(question)
    status, response = body.post_payload(messages, SESSION_A, BODY_TOOL)
    rows = []
    checkpoints = []
    final = response
    channel = body_source_identity(SESSION_A)
    for index in range(8):
        blocks = response.get("content", ()) if status == 200 else ()
        tool = (blocks[0] if len(blocks) == 1
                and blocks[0].get("type") == "tool_use" else None)
        if tool is None:
            final = response
            break
        surface = tool.get("input", {}).get("surface", "").encode()
        receipt = _action_for_surface(body.server.runtime, surface, channel)
        coordinate = (() if receipt is None else
                      body.server.runtime.adult._causal_action_leading_coordinate(
                          receipt))
        checkpoint_text = Path(checkpoint).read_text()
        checkpoints.append(checkpoint_text)
        rows.append({
            "surface": surface,
            "programs": 0 if receipt is None else len(receipt.programs),
            "coordinate": tuple(int(value) for value in coordinate),
            "identity": 0 if receipt is None else int(receipt.identity),
            "tool_id": tool.get("id", ""),
            "stop": response.get("stop_reason", ""),
        })
        is_error = bool(fail_requested and index == 1)
        result = [{"type": "tool_result", "tool_use_id": tool.get("id", ""),
                   "content": "", "is_error": is_error}]
        if index == 0 and request:
            result.append({"type": "text", "text": CONTINUE_CONTACT.decode()})
        messages = [*messages, {"role": "assistant", "content": blocks},
                    {"role": "user", "content": result}]
        body.stop()
        restored = type(runtime).restore(runtime.program, json.loads(checkpoint_text))
        body = Body(checkpoint).start(restored)
        status, response = body.post_payload(messages, SESSION_A, BODY_TOOL)
        final = response
    subject = body.server.runtime
    final_checkpoint = Path(checkpoint).read_text()
    body.stop()
    return {
        "status": status, "rows": tuple(rows), "final": final,
        "checkpoints": tuple(checkpoints), "final_checkpoint": final_checkpoint,
        "runtime": subject,
    }


def _run_focus_arm(runtime, checkpoint, question, paraphrase, restart=False):
    body = Body(checkpoint).start(runtime)
    messages = _initial(question)
    status, first = body.post(messages, SESSION_A)
    channel = body_source_identity(SESSION_A)
    first_receipt = _action_for_surface(body.server.runtime, first.encode(), channel)
    if restart:
        body.stop()
        restored = type(runtime).restore(
            runtime.program, json.loads(Path(checkpoint).read_text()))
        body = Body(checkpoint).start(restored)
    status_2, second = body.post(_continue(messages, first, paraphrase), SESSION_A)
    second_receipt = _action_for_surface(
        body.server.runtime, second.encode(), channel)
    messages = _continue(messages, first, paraphrase)
    status_3, third = body.post(_continue(messages, second, question), SESSION_A)
    messages = _continue(messages, second, question)
    status_4, fourth = body.post(_continue(messages, third, paraphrase), SESSION_A)
    messages = _continue(messages, third, paraphrase)
    status_5, final = body.post(_continue(messages, fourth, question), SESSION_A)
    final_receipt = _action_for_surface(
        body.server.runtime, final.encode(), channel)
    uptake = body.server.runtime.adult.causal_dialogue_uptake_support(
        channel, int(_focus_and_paraphrase(body.server.runtime)[3]))
    checkpoint_text = Path(checkpoint).read_text()
    subject = body.server.runtime
    body.stop()
    return {
        "statuses": (status, status_2, status_3, status_4, status_5),
        "first": first.encode(), "second": second.encode(),
        "third": third.encode(), "fourth": fourth.encode(), "final": final.encode(),
        "first_programs": 0 if first_receipt is None else len(first_receipt.programs),
        "second_programs": 0 if second_receipt is None else len(second_receipt.programs),
        "final_programs": 0 if final_receipt is None else len(final_receipt.programs),
        "uptake": uptake, "checkpoint_text": checkpoint_text,
        "runtime": subject,
    }


def _run_yoked_arm(runtime, checkpoint, question, paraphrase):
    body = Body(checkpoint).start(runtime)
    messages = _initial(question)
    status_b1, first_b = body.post(messages, SESSION_B)
    status_b2, second_b = body.post(
        _continue(messages, first_b, paraphrase), SESSION_B)
    status_a, first_a = body.post(_initial(question), SESSION_A)
    channel_a = body_source_identity(SESSION_A)
    channel_b = body_source_identity(SESSION_B)
    receipt = int(_focus_and_paraphrase(body.server.runtime)[3])
    action_a = _action_for_surface(body.server.runtime, first_a.encode(), channel_a)
    result = {
        "statuses": (status_b1, status_b2, status_a),
        "session_a_surface": first_a.encode(),
        "session_a_programs": 0 if action_a is None else len(action_a.programs),
        "session_a_uptake": body.server.runtime.adult.causal_dialogue_uptake_support(
            channel_a, receipt),
        "session_b_uptake": body.server.runtime.adult.causal_dialogue_uptake_support(
            channel_b, receipt),
    }
    body.stop()
    return result


def _run_reversal_arm(runtime, checkpoint, question, reversed_relation, causal_receipt):
    body = Body(checkpoint).start(runtime)
    messages = _initial(question)
    status_first, first = body.post(messages, SESSION_A)
    reversed_messages = _continue(messages, first, reversed_relation)
    status_repair, repair = body.post(reversed_messages, SESSION_A)
    status_followup, followup = body.post(
        _continue(reversed_messages, repair, question), SESSION_A)
    channel = body_source_identity(SESSION_A)
    dispute = body.server.runtime.adult.causal_dialogue_dispute_support(
        channel, causal_receipt)
    result = {
        "statuses": (status_first, status_repair, status_followup),
        "repair": repair.encode(),
        "followup": followup.encode(),
        "dispute": dispute,
        "uptake": body.server.runtime.adult.causal_dialogue_uptake_support(
            channel, causal_receipt),
    }
    body.stop()
    return result


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="foundry-claude-body-uptake-") as directory:
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        effect, question, paraphrase, causal_receipt, reversed_relation = \
            _focus_and_paraphrase(base)
        checkpoint = Path(directory) / "claude-life.json"
        checkpoint.write_text(json.dumps(base.checkpoint(), separators=(",", ":"), sort_keys=True))
        focused = _run_focus_arm(
            type(base).restore(base.program, copy.deepcopy(base.checkpoint())),
            checkpoint, question, paraphrase, restart=True)

        yoked_path = Path(directory) / "yoked.json"
        yoked_runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        yoked_path.write_text(json.dumps(yoked_runtime.checkpoint()))
        yoked = _run_yoked_arm(
            yoked_runtime, yoked_path, question, paraphrase)

        reversal_path = Path(directory) / "reversal.json"
        reversal_runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        reversal_path.write_text(json.dumps(reversal_runtime.checkpoint()))
        reversal = _run_reversal_arm(
            reversal_runtime, reversal_path, question, reversed_relation,
            causal_receipt)

        # Claude session identifiers are replay cursors, not social identities.
        # The first session earns partner-local uptake through exact reafference;
        # a fresh transport session under the same authenticated body must see
        # that changed individual.  An explicit other contact must not inherit it.
        stable_path = Path(directory) / "stable-body-identity.json"
        stable_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        stable_path.write_text(json.dumps(stable_runtime.checkpoint()))
        stable_body = Body(stable_path).start(stable_runtime)
        stable_messages = _initial(question)
        stable_a1_status, stable_a1 = stable_body.post(
            stable_messages, SESSION_A, contact_identity=None)
        stable_a2_messages = _continue(stable_messages, stable_a1, paraphrase)
        stable_a2_status, stable_a2 = stable_body.post(
            stable_a2_messages, SESSION_A,
            contact_identity=None)
        stable_a3_messages = _continue(stable_a2_messages, stable_a2, question)
        stable_a3_status, stable_a3 = stable_body.post(
            stable_a3_messages, SESSION_A, contact_identity=None)
        stable_a4_messages = _continue(stable_a3_messages, stable_a3, paraphrase)
        stable_a4_status, stable_a4 = stable_body.post(
            stable_a4_messages, SESSION_A, contact_identity=None)
        stable_settle_a_status, stable_settle_a = stable_body.post(
            _continue(stable_a4_messages, stable_a4, b""), SESSION_A,
            contact_identity=None)
        stable_b_status, stable_b = stable_body.post(
            _initial(question), SESSION_B, contact_identity=None)
        stable_b_messages = _initial(question)
        stable_settle_b_status, stable_settle_b = stable_body.post(
            _continue(stable_b_messages, stable_b, b""), SESSION_B,
            contact_identity=None)
        explicit_other_status, explicit_other = stable_body.post(
            _initial(question), "claude-body-other-transport",
            contact_identity="authenticated-other-contact")
        stable_channel = body_source_identity(TOKEN)
        stable_cursor_a = stable_body.server.runtime.adult.language_adult.body_ingress_cursor(SESSION_A)
        stable_cursor_b = stable_body.server.runtime.adult.language_adult.body_ingress_cursor(SESSION_B)
        stable_uptake = stable_body.server.runtime.adult.causal_dialogue_uptake_support(
            stable_channel, causal_receipt)
        stable_checkpoint = stable_path.read_text()
        stable_body.stop()

        forged_path = Path(directory) / "forged.json"
        forged_runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        forged_path.write_text(json.dumps(forged_runtime.checkpoint()))
        forged_body = Body(forged_path).start(forged_runtime)
        forged_messages = _initial(question)
        initial_status, initial_surface = forged_body.post(forged_messages, SESSION_A)
        forged_status, _ = forged_body.post(
            _continue(forged_messages, initial_surface + " forged", paraphrase), SESSION_A)
        wrong_session_status, _ = forged_body.post(
            _continue(forged_messages, initial_surface, paraphrase), SESSION_B)
        forged_body.stop()

        state_marks = (
            "open_state_prompt_grounded", "open_state_prompt_loaded",
            "open_state_prompt_recovered", "testimony_revision_old_world",
        )
        state_rows = []
        for index, mark in enumerate(state_marks):
            state_runtime = load_mark(directory, mark)
            state_path = Path(directory) / f"state-{index}.json"
            state_path.write_text(json.dumps(state_runtime.checkpoint()))
            state_rows.append(_run_state_arm(state_runtime, state_path))

        control_rows = []
        for confirmed, name in ((True, "contingent"), (False, "nonconfirming")):
            control_runtime = load_mark(directory, "open_state_prompt_recovered")
            control_path = Path(directory) / f"state-{name}.json"
            control_rows.append(_run_partner_control_arm(
                control_runtime, control_path, effect, confirmed))

        tool_rows = []
        for is_error, name in ((False, "tool-success"), (True, "tool-error")):
            tool_runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
            tool_path = Path(directory) / f"{name}.json"
            tool_path.write_text(json.dumps(tool_runtime.checkpoint()))
            tool_rows.append(_run_tool_result_arm(
                tool_runtime, tool_path, question, is_error))

        chain_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        chain_path = Path(directory) / "tool-settled-chain.json"
        chain_path.write_text(json.dumps(chain_runtime.checkpoint()))
        chain = _run_settled_tool_chain(
            chain_runtime, chain_path, question)

        no_request_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        no_request_path = Path(directory) / "tool-no-continuation-request.json"
        no_request_path.write_text(json.dumps(no_request_runtime.checkpoint()))
        no_request = _run_settled_tool_chain(
            no_request_runtime, no_request_path, question, request=False)

        failed_request_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        failed_request_path = Path(directory) / "tool-failed-continuation.json"
        failed_request_path.write_text(json.dumps(failed_request_runtime.checkpoint()))
        failed_request = _run_settled_tool_chain(
            failed_request_runtime, failed_request_path, question,
            fail_requested=True)

        wrong_tool_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        wrong_tool_path = Path(directory) / "tool-wrong-ticket.json"
        wrong_tool_path.write_text(json.dumps(wrong_tool_runtime.checkpoint()))
        wrong_tool = _run_tool_result_arm(
            wrong_tool_runtime, wrong_tool_path, question, False,
            tool_id_override="toolu_agi_deadbeef")

        replay_runtime = type(base).restore(
            base.program, json.loads(tool_rows[0]["final_checkpoint"]))
        replay_path = Path(directory) / "tool-replay.json"
        replay_path.write_text(tool_rows[0]["final_checkpoint"])
        replay_before = copy.deepcopy(replay_runtime.checkpoint())
        replay_body = Body(replay_path).start(replay_runtime)
        replay_status, _ = replay_body.post_payload(
            tool_rows[0]["continued"], SESSION_A, BODY_TOOL)
        replay_unchanged = replay_body.server.runtime.checkpoint() == replay_before
        replay_body.stop()

        preaction_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        preaction_path = Path(directory) / "tool-before-action.json"
        preaction_path.write_text(json.dumps(preaction_runtime.checkpoint()))
        preaction_body = Body(preaction_path).start(preaction_runtime)
        preaction_status, _ = preaction_body.post_payload([{"role": "user", "content": [
            {"type": "text", "text": "opaque Claude Code body state"},
            {"type": "tool_result", "tool_use_id": "toolu_agi_deadbeef",
             "content": "opaque body result"},
        ]}], SESSION_A, BODY_TOOL)
        preaction_body.stop()

        stream_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        stream_path = Path(directory) / "tool-stream.json"
        stream_path.write_text(json.dumps(stream_runtime.checkpoint()))
        stream_body = Body(stream_path).start(stream_runtime)
        stream_status, stream_events = stream_body.post_stream(
            _initial(question), SESSION_A, BODY_TOOL)
        stream_body.stop()

        malformed_runtime = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        malformed_path = Path(directory) / "tool-malformed-schema.json"
        malformed_path.write_text(json.dumps(malformed_runtime.checkpoint()))
        malformed_body = Body(malformed_path).start(malformed_runtime)
        malformed_status, malformed_payload = malformed_body.post_payload(
            _initial(question), SESSION_A,
            [{"name": BODY_ACTION_TOOL, "input_schema": {"type": "object"}}])
        malformed_body.stop()

        lesion = load_mark(directory, "open_state_prompt_recovered")
        lesion_prompt, lesion_sources = _state_contact(lesion)
        for source in lesion_sources:
            lesion.adult.language.withdraw_source(source)
        lesion_path = Path(directory) / "state-lesion.json"
        lesion_path.write_text(json.dumps(lesion.checkpoint()))
        lesion_row = _run_state_arm(lesion, lesion_path, prompt=lesion_prompt)

    first = focused["first"]
    final = focused["second"]
    chain_surfaces = tuple(row["surface"] for row in chain["rows"])
    chain_programs = tuple(row["programs"] for row in chain["rows"])
    checks = {
        "heldout_natural_question_and_independent_paraphrase_exist": bool(
            effect and question and paraphrase and causal_receipt
            and paraphrase not in first),
        "actual_messages_body_chronology_succeeds": focused["statuses"] == (200, 200, 200, 200, 200),
        "exact_reafference_and_structural_contact_accumulate_uptake": focused["uptake"] >= 1,
        "same_question_lived_partner_history_focuses_resident_programs": bool(
            first and final and final != first
            and focused["first_programs"] > focused["second_programs"] > 0),
        "accepted_relation_is_absent_from_later_public_plan": bool(
            paraphrase not in first and len(final) < len(first)),
        "pending_action_and_body_boundary_survive_process_restart": bool(
            focused["statuses"][1] == 200),
        "forged_or_wrong_session_reafference_refuses": bool(
            initial_status == 200 and forged_status == 422 and wrong_session_status == 422),
        "other_session_exposure_does_not_focus_this_partner": bool(
            yoked["statuses"] == (200, 200, 200)
            and yoked["session_b_uptake"] >= 1
            and yoked["session_a_uptake"] == 0
            and yoked["session_a_programs"] == focused["first_programs"]
            and yoked["session_a_surface"] == first),
        "transport_restart_keeps_partner_history_but_not_replay_cursor": bool(
            (stable_a1_status, stable_a2_status, stable_a3_status,
             stable_a4_status, stable_settle_a_status,
             stable_b_status, stable_settle_b_status,
             explicit_other_status) == (200, 200, 200, 200, 200, 200, 200, 200)
            and stable_uptake >= 1 and stable_a1 and stable_a2 and stable_b
            and stable_b.encode() == focused["final"] and stable_b != stable_a1
            and not stable_settle_a and not stable_settle_b
            and explicit_other == stable_a1
            and stable_cursor_a[0] > 0 and stable_cursor_b[0] > 0
            and stable_cursor_a != stable_cursor_b),
        "stable_body_identity_checkpoint_has_no_transcript_or_raw_credential": bool(
            TOKEN not in stable_checkpoint
            and question.decode() not in stable_checkpoint
            and paraphrase.decode() not in stable_checkpoint),
        "reversed_relation_recruits_specific_repair_without_false_focus": bool(
            reversed_relation and reversal["statuses"] == (200, 200, 200)
            and reversal["dispute"] >= 1 and reversal["uptake"] == 0
            and reversal["repair"] and reversal["repair"].endswith(b"?")
            and reversal["repair"] != first and reversal["followup"] == first),
        "checkpoint_contains_no_partner_surface_or_question_transcript": bool(
            paraphrase.decode() not in focused["checkpoint_text"]
            and question.decode() not in focused["checkpoint_text"]),
        "claude_body_exposes_each_resident_state_affordance": all(
            row["status"] == 200 and row["surface"] == row["expected"]
            and row["surface"] for row in state_rows),
        "state_action_reafference_continues_without_transcript_authority": all(
            row["status_2"] == 200 and row["surface_2"] == row["surface"]
            for row in state_rows),
        "state_action_checkpoint_restores_only_pending_causal_commitment": all(
            row["checkpoint_pending_context_actions"] == 1
            for row in state_rows),
        "same_contact_tracks_current_load_and_recovery": bool(
            state_rows[0]["prompt"] == state_rows[1]["prompt"]
            == state_rows[2]["prompt"] == state_rows[3]["prompt"]
            and state_rows[0]["surface"] != state_rows[1]["surface"]
            and state_rows[0]["surface"] == state_rows[2]["surface"]),
        "byte_reafference_never_manufactures_causal_control": bool(
            _channel_control_state(focused["runtime"]) == (0, 0, 0, 0, 0)),
        "matched_resident_actions_diverge_only_after_lived_consequence": bool(
            control_rows[0]["actions"] == control_rows[1]["actions"]
            and all(control_rows[0]["actions"])
            and control_rows[0]["surface"] != control_rows[1]["surface"]
            and control_rows[0]["control"][0] == control_rows[1]["control"][0]
            and control_rows[0]["control"][2] == control_rows[1]["control"][2]
            and control_rows[0]["control"][1] > control_rows[1]["control"][1]
            and control_rows[0]["control"][4] > control_rows[1]["control"][4]),
        "partner_local_control_survives_checkpoint_and_body_contact": all(
            row["status"] == 200 and row["surface"] == row["expected"]
            and row["surface"] == row["restored_surface"]
            for row in control_rows),
        "resident_public_action_lowers_to_literal_anthropic_tool_use": all(
            row["statuses"] == (200, 200) and row["first_stop"] == "tool_use"
            and row["tool"].get("name") == BODY_ACTION_TOOL
            and row["tool"].get("input", {}).get("surface")
            and row["pending_before"] == 1 and row["pending_after"] == 0
            for row in tool_rows),
        "one_request_sustains_multi_action_anthropic_tool_horizon": bool(
            chain["status"] == 200 and len(chain_surfaces) >= 5
            and len(set(chain_surfaces)) == len(chain_surfaces)
            and all(chain_surfaces)
            and all(row["stop"] == "tool_use" for row in chain["rows"])
            and chain["final"].get("stop_reason") == "end_turn"),
        "every_successor_has_resident_relation_certificate": bool(
            len(chain["rows"]) >= 5
            and all(row["programs"] > 0 and row["coordinate"]
                    and row["tool_id"] == "toolu_agi_" + format(row["identity"], "x")
                    for row in chain["rows"])),
        "body_loop_preserves_broad_changing_argument_structure": bool(
            len(chain_programs) >= 5 and len(set(chain_programs)) >= 3
            and max(chain_programs) >= 3),
        "checkpoint_restart_rematerializes_without_future_surface_storage": bool(
            len(chain["checkpoints"]) == len(chain["rows"])
            and all(CONTINUE_CONTACT.decode() not in checkpoint_text
                    for checkpoint_text in chain["checkpoints"])
            and all(surface.decode() not in checkpoint_text
                    for surface in chain_surfaces
                    for checkpoint_text in chain["checkpoints"])),
        "continuation_requires_request_and_successful_exact_settlement": bool(
            len(no_request["rows"]) == 1
            and no_request["final"].get("stop_reason") == "end_turn"
            and len(failed_request["rows"]) == 2
            and failed_request["final"].get("stop_reason") == "end_turn"),
        "streaming_body_uses_literal_anthropic_tool_events": bool(
            stream_status == 200
            and '"name":"agi_body_action"' in stream_events
            and '"type":"input_json_delta"' in stream_events
            and '"stop_reason":"tool_use"' in stream_events),
        "malformed_body_action_schema_cannot_claim_tool_authority": bool(
            malformed_status == 200
            and malformed_payload.get("stop_reason") == "end_turn"
            and all(block.get("type") != "tool_use"
                    for block in malformed_payload.get("content", ()))),
        "exact_tool_success_and_error_change_later_visible_state": bool(
            tool_rows[0]["tool"]["input"] == tool_rows[1]["tool"]["input"]
            and tool_rows[0]["surface"] and tool_rows[1]["surface"]
            and tool_rows[0]["surface"] != tool_rows[1]["surface"]
            and tool_rows[0]["control"][0] == tool_rows[1]["control"][0] > 0
            and tool_rows[0]["control"][1] > tool_rows[1]["control"][1]),
        "tool_consequence_and_visible_change_survive_process_restart": all(
            row["surface"] == row["restored_surface"] for row in tool_rows),
        "wrong_replayed_or_result_before_action_refuses_without_credit": bool(
            wrong_tool["statuses"] == (200, 422)
            and wrong_tool["control"] == (0, 0, 0, 0, 0)
            and replay_status == 422 and replay_unchanged
            and preaction_status == 422),
        "tool_result_text_is_not_checkpoint_memory": all(
            "opaque body result" not in row["final_checkpoint"]
            and state_rows[0]["prompt"].decode() not in row["final_checkpoint"]
            for row in tool_rows),
        "withdrawn_affordance_evidence_restores_silence": bool(
            lesion_row["status"] == 200 and not lesion_row["surface"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "schema": "cyber-lagoon.reference-claude-body-causal-uptake.v1",
        "contract": "FOUNDRY_CLAUDE_BODY_CAUSAL_UPTAKE_" + ("GREEN" if not failed else "RED"),
        "pass": not failed,
        "language_phenotype_improved": not failed,
        "reference_only": True,
        "runtime_llm": False,
        "same_final_question": question.decode(errors="replace"),
        "independently_learned_partner_paraphrase": paraphrase.decode(errors="replace"),
        "before": first.decode(errors="replace"),
        "after": final.decode(errors="replace"),
        "program_counts": {"before": focused["first_programs"], "after": focused["second_programs"]},
        "uptake_support": focused["uptake"],
        "yoked_session_uptake": {
            "partner_a": yoked["session_a_uptake"],
            "partner_b": yoked["session_b_uptake"],
        },
        "stable_body_sessions": {
            "statuses": [stable_a1_status, stable_a2_status,
                         stable_a3_status, stable_a4_status,
                         stable_settle_a_status, stable_b_status,
                         stable_settle_b_status, explicit_other_status],
            "before": stable_a1,
            "after_learning": stable_a2,
            "after_repeated_evidence": stable_a4,
            "new_transport": stable_b,
            "explicit_other_contact": explicit_other,
            "uptake": stable_uptake,
            "cursor_a": stable_cursor_a,
            "cursor_b": stable_cursor_b,
        },
        "reversed_relation": reversed_relation.decode(errors="replace"),
        "reversal_public_response": reversal["repair"].decode(errors="replace"),
        "reversal_followup": reversal["followup"].decode(errors="replace"),
        "reversal_dispute_support": reversal["dispute"],
        "state_contact": state_rows[0]["prompt"].decode(errors="replace"),
        "state_trajectory": [row["surface"].decode(errors="replace")
                             for row in state_rows],
        "partner_control_trajectory": {
            "contingent": control_rows[0]["surface"].decode(errors="replace"),
            "nonconfirming": control_rows[1]["surface"].decode(errors="replace"),
        },
        "partner_control_counts": {
            "contingent": control_rows[0]["control"],
            "nonconfirming": control_rows[1]["control"],
        },
        "tool_action": tool_rows[0]["tool"].get("input", {}).get("surface", ""),
        "tool_consequence_trajectory": {
            "success": tool_rows[0]["surface"].decode(errors="replace"),
            "error": tool_rows[1]["surface"].decode(errors="replace"),
        },
        "tool_consequence_counts": {
            "success": tool_rows[0]["control"],
            "error": tool_rows[1]["control"],
        },
        "resident_tool_horizon": [
            {"surface": row["surface"].decode(errors="replace"),
             "programs": row["programs"],
             "coordinate": row["coordinate"],
             "tool_id": row["tool_id"]}
            for row in chain["rows"]],
        "resident_tool_horizon_controls": {
            "without_request_actions": len(no_request["rows"]),
            "failed_requested_action_actions": len(failed_request["rows"]),
        },
        "checks": checks,
        "failed": failed,
        "remaining_red": ["LEARNED_TERMINAL_TOOL_SELECTION_AND_ARGUMENT_PLANNING",
                          "BROAD_NATURAL_CLAUDE_DIALOGUE",
                          "DIRECT_PARITY"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(result["contract"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
