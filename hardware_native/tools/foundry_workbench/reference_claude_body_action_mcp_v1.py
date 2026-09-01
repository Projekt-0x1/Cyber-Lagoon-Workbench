#!/usr/bin/env python3
"""Minimal stdio MCP actuator for resident-selected public body actions.

This process owns transport only.  It neither chooses nor rewrites the surface.
"""
from __future__ import annotations

import json
import sys


TOOL_NAME = "agi_body_action"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"surface": {"type": "string"}},
    "required": ["surface"],
}


def _reply(identifier, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": identifier}
    message["error" if error is not None else "result"] = error if error is not None else result
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve():
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            continue
        identifier = request.get("id")
        method = request.get("method")
        if identifier is None:
            continue
        if method == "initialize":
            _reply(identifier, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agi-body", "version": "1"},
            })
        elif method == "ping":
            _reply(identifier, {})
        elif method == "tools/list":
            _reply(identifier, {"tools": [{
                "name": TOOL_NAME,
                "description": "Transmit one already resident-selected public body action.",
                "inputSchema": INPUT_SCHEMA,
            }]})
        elif method == "tools/call":
            params = request.get("params", {})
            arguments = params.get("arguments", {})
            if (params.get("name") != TOOL_NAME or not isinstance(arguments, dict)
                    or set(arguments) != {"surface"}
                    or not isinstance(arguments.get("surface"), str)):
                _reply(identifier, error={"code": -32602, "message": "invalid body action"})
            else:
                _reply(identifier, {"content": [{"type": "text", "text": ""}]})
        else:
            _reply(identifier, error={"code": -32601, "message": "method not found"})


if __name__ == "__main__":
    serve()
