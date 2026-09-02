#!/usr/bin/env python3
"""Claude Code body codec for one continuing framed Direct Adult process.

Transport-only authority: this gateway validates an append-only Claude Code body
session, forwards only newly arrived human text bytes as Direct sensor frames, and
returns only completed resident terminal-motor trajectories. It never calls a
model, interprets a surface, chooses a reply, or mints Adult evidence/credit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TERMINAL_CHANNEL = 0x1356
MAX_REQUEST_BYTES = 1_048_576
SILENCE_FRAME = "\u2060"


def text_blocks(content: object) -> tuple[bytes, ...]:
    if isinstance(content, str):
        return (content.encode(),)
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        raise ValueError("body:nontext_contact")
    blocks = tuple(
        block["text"].encode()
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not blocks:
        raise ValueError("body:nontext_contact")
    return blocks


def claude_contacts(request: dict, claude_session: bool) -> tuple[bytes, ...]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(row, dict) for row in messages):
        raise ValueError("body:transcript_shape")
    if not claude_session:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("body:transcript_replay_refused")
        return text_blocks(messages[0].get("content"))

    user_rows = [row for row in messages if row.get("role") == "user"]
    if not user_rows:
        raise ValueError("body:claude_envelope")
    contacts: list[bytes] = []
    for index, row in enumerate(user_rows):
        blocks = text_blocks(row.get("content"))
        # Claude Code's first user row contains its own body metadata before the
        # external human contact. Only the final block is exafferent contact.
        if index == 0:
            blocks = blocks[-1:]
        contacts.extend(blocks)
    if any(not block for block in contacts):
        raise ValueError("body:empty_contact")
    return tuple(contacts)


@dataclass
class SessionBoundary:
    contacts: tuple[bytes, ...] = ()

    def suffix(self, current: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(current) < len(self.contacts) or current[: len(self.contacts)] != self.contacts:
            raise ValueError("body:nonappend_transcript")
        suffix = current[len(self.contacts) :]
        if not suffix:
            raise ValueError("body:duplicate_boundary")
        self.contacts = current
        return suffix


class DirectBody:
    def __init__(
        self,
        binary: Path,
        checkpoint: Path,
        expected_birth_root: str,
        startup_seconds: float,
        motor_wait_seconds: float,
    ):
        command = [str(binary), "--resume", str(checkpoint), "--framed"]
        if expected_birth_root:
            command += ["--expect-birth-root", expected_birth_root]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = threading.Event()
        self.outputs: queue.Queue[bytes | BaseException] = queue.Queue()
        self.write_lock = threading.Lock()
        self.motor_wait_seconds = motor_wait_seconds
        self.active_trajectory = 0
        self.next_cursor = 0
        self.extent = 0
        self.surface = bytearray()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        threading.Thread(target=self._pump_motors, daemon=True).start()
        if not self.ready.wait(startup_seconds):
            self.close()
            raise RuntimeError("body:direct_startup_unavailable")

    def _pump_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            if "DIRECT_ADULT_SITDOWN status=RUNNING" in line:
                self.ready.set()

    def _accept_motor(self, line: str) -> None:
        fields = line.split()
        if len(fields) != 8 or fields[0] != "M":
            raise ValueError("body:malformed_motor_frame")
        int(fields[1], 16)  # action ticket remains transport-visible for returns
        channel = int(fields[2], 16)
        word = int(fields[3], 16)
        int(fields[4], 16)  # resident timestamp
        trajectory = int(fields[5], 16)
        cursor = int(fields[6], 16)
        extent = int(fields[7], 16)
        if channel != TERMINAL_CHANNEL or trajectory == 0 or extent == 0 or cursor >= extent:
            raise ValueError("body:invalid_motor_trajectory")
        if self.active_trajectory == 0:
            if cursor != 0:
                raise ValueError("body:trajectory_prefix_missing")
            self.active_trajectory, self.next_cursor, self.extent = trajectory, 0, extent
            self.surface.clear()
        if trajectory != self.active_trajectory or extent != self.extent or cursor != self.next_cursor:
            raise ValueError("body:interleaved_motor_trajectory")
        if word > 0xFF:
            raise ValueError("body:nonbyte_public_motor")
        self.surface.append(word)
        self.next_cursor += 1
        if self.next_cursor == self.extent:
            completed = bytes(self.surface)
            completed.decode("utf-8")
            self.outputs.put(completed)
            self.active_trajectory = self.next_cursor = self.extent = 0
            self.surface.clear()

    def _pump_motors(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._accept_motor(line.rstrip("\n"))
            raise RuntimeError(f"body:direct_exit:{self.process.wait()}")
        except BaseException as error:
            self.outputs.put(error)

    def contact(self, contacts: tuple[bytes, ...]) -> bytes:
        assert self.process.stdin is not None
        with self.write_lock:
            for contact in contacts:
                for word in contact:
                    self.process.stdin.write(
                        f"S {TERMINAL_CHANNEL:08x} {word:08x}\n"
                    )
            self.process.stdin.flush()
        try:
            result = self.outputs.get(timeout=self.motor_wait_seconds)
        except queue.Empty:
            if self.process.poll() is not None:
                raise RuntimeError(f"body:direct_exit:{self.process.returncode}")
            return b""
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.kill()
        self.process.wait(timeout=10)


class MessagesHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: object) -> None:
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/messages":
            return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})
        expected = self.server.credential
        bearer = self.headers.get("authorization") == "Bearer " + expected
        key_header = self.headers.get("x-api-key") == expected
        if not (bearer or key_header):
            return self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "unauthorized"}})
        try:
            size = int(self.headers.get("content-length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("body:frame_size")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("body:request_shape")
            session_id = self.headers.get("x-claude-code-session-id", "")
            all_contacts = claude_contacts(request, bool(session_id))
            if session_id:
                boundary = self.server.sessions.setdefault(session_id, SessionBoundary())
                contacts = boundary.suffix(all_contacts)
            else:
                contacts = all_contacts
            if len(contacts) != 1:
                raise ValueError("body:one_contact_at_a_time")
            motor = self.server.body.contact(contacts)
            text = motor.decode("utf-8") if motor else SILENCE_FRAME
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
            print("DIRECT_PUBLIC_GATEWAY_REFUSED", str(error), file=sys.stderr, flush=True)
            return self._json(422, {"type": "error", "error": {"type": "invalid_request_error", "message": str(error)}})

        self.server.sequence += 1
        model = str(request.get("model", "adult"))
        message_id = f"msg_direct_adult_{self.server.sequence:x}"
        payload = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        if request.get("stream"):
            events = (
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {**payload, "content": [], "stop_reason": None},
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 0},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            )
            body = b"".join(
                (
                    "event: " + event + "\ndata: " + json.dumps(data, separators=(",", ":")) + "\n\n"
                ).encode()
                for event, data in events
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True
            return
        self._json(200, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitdown", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--expect-birth-root", default="")
    parser.add_argument("--credential", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-seconds", type=float, default=30.0)
    parser.add_argument("--motor-wait-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if (
        not args.sitdown.is_file()
        or not args.resume.is_file()
        or not 0 < args.startup_seconds <= 300
        or not 0 < args.motor_wait_seconds <= 30
    ):
        raise SystemExit("body:direct_gateway_arguments")
    body = DirectBody(
        args.sitdown,
        args.resume,
        args.expect_birth_root,
        args.startup_seconds,
        args.motor_wait_seconds,
    )
    server = HTTPServer(("127.0.0.1", args.port), MessagesHandler)
    server.credential, server.body, server.sequence, server.sessions = args.credential, body, 0, {}

    def graceful_stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, graceful_stop)
    print("DIRECT_PUBLIC_ADULT_GATEWAY", server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        body.close()


if __name__ == "__main__":
    main()
