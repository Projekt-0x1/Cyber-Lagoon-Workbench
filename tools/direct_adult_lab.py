#!/usr/bin/env python3
"""Run a bounded experiment on a disposable clone of a Direct Adult checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_bounded(command: list[str], body: bytes, timeout: float) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(body, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise TimeoutError(f"experiment exceeded {timeout:g}s")
    return process.returncode, stdout, stderr


def stopped_metrics(stderr: bytes) -> dict[str, str]:
    prefix = "DIRECT_ADULT_SITDOWN status=STOPPED "
    lines = stderr.decode("utf-8", errors="replace").splitlines()
    stopped = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    if len(stopped) != 1:
        raise RuntimeError(f"expected one Direct Adult STOPPED receipt, saw {len(stopped)}")
    return dict(field.split("=", 1) for field in stopped[0].split() if "=" in field)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Clone a retained Direct Adult checkpoint, run direct_adult_sitdown on "
            "the clone, and emit a non-GREEN experiment receipt."
        )
    )
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument(
        "--artifact",
        type=Path,
        default=Path("hardware_native/build_cuda/direct_adult_sitdown"),
        help="GPU-lock-wrapped direct_adult_sitdown artifact",
    )
    result.add_argument("--input", type=Path, help="body bytes; stdin when omitted")
    result.add_argument("--framed", action="store_true", help="forward framed body protocol")
    result.add_argument(
        "--expect-birth-root",
        help="require this 64-hex birth root from the resumed Adult",
    )
    result.add_argument("--timeout", type=float, default=60.0)
    result.add_argument(
        "--output-checkpoint",
        type=Path,
        help="publish the successful experimental branch; must not already exist",
    )
    result.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "atomically publish the branch checkpoint, exact input, motor output, "
            "sitdown log, and receipt; must not already exist"
        ),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = args.checkpoint.resolve()
    artifact = args.artifact.resolve()
    output = args.output_checkpoint.resolve() if args.output_checkpoint else None
    output_directory = (
        args.output_directory.resolve() if args.output_directory else None
    )
    real_artifact = artifact.with_name(artifact.name + ".real")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not artifact.is_file() or not os.access(artifact, os.X_OK):
        raise FileNotFoundError(f"executable Direct Adult artifact not found: {artifact}")
    if artifact.name.endswith(".real"):
        raise ValueError("pass the GPU-lock wrapper, not its unlocked .real binary")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.expect_birth_root is not None:
        root = args.expect_birth_root.lower()
        if len(root) != 64 or any(ch not in "0123456789abcdef" for ch in root):
            raise ValueError("--expect-birth-root requires 64 hexadecimal digits")
        args.expect_birth_root = root
    if output and output_directory:
        raise ValueError("choose --output-checkpoint or --output-directory, not both")
    if output and (output == source or output.exists()):
        raise FileExistsError(f"output checkpoint must be a new path: {output}")
    if output_directory and output_directory.exists():
        raise FileExistsError(
            f"output directory must be a new path: {output_directory}"
        )
    if output_directory and not output_directory.parent.is_dir():
        raise FileNotFoundError(output_directory.parent)
    if output_directory and not real_artifact.is_file():
        raise FileNotFoundError(
            f"unlocked Direct Adult artifact not found beside wrapper: {real_artifact}"
        )

    body = args.input.read_bytes() if args.input else sys.stdin.buffer.read()
    source_before = sha256(source)
    receipt: dict[str, object] = {
        "schema": "direct-adult-rapid-experiment-v1",
        "status": "RED",
        "claim_scope": "experiment_only",
        "green_eligible": False,
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "artifact_real_sha256": sha256(real_artifact) if real_artifact.is_file() else None,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_before,
        "input_sha256": hashlib.sha256(body).hexdigest(),
        "input_bytes": len(body),
        "framed": args.framed,
        "expected_birth_root": args.expect_birth_root,
    }

    temp_parent = (
        output_directory.parent
        if output_directory
        else output.parent if output else None
    )
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="direct-adult-lab-", dir=temp_parent) as directory:
            branch = Path(directory) / "adult.checkpoint"
            shutil.copyfile(source, branch)
            command = [str(artifact), "--resume", str(branch)]
            if args.expect_birth_root:
                command.extend(("--expect-birth-root", args.expect_birth_root))
            if args.framed:
                command.append("--framed")
            returncode, stdout, stderr = run_bounded(command, body, args.timeout)
            receipt.update(
                exit_code=returncode,
                stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                stdout_bytes=len(stdout),
                stderr_sha256=hashlib.sha256(stderr).hexdigest(),
                stderr_bytes=len(stderr),
            )
            if returncode != 0:
                raise RuntimeError(f"direct_adult_sitdown exited {returncode}")
            receipt["sitdown"] = stopped_metrics(stderr)
            if (
                args.expect_birth_root
                and receipt["sitdown"].get("birth_root") != args.expect_birth_root
            ):
                raise RuntimeError("Direct Adult birth root did not match expectation")
            receipt["branch_checkpoint_sha256"] = sha256(branch)
            if output:
                os.replace(branch, output)
                receipt["output_checkpoint"] = str(output)
            receipt["status"] = "EXPERIMENT_PASS"
            if output_directory:
                receipt.update(
                    output_directory=str(output_directory),
                    output_checkpoint=str(output_directory / "adult.checkpoint"),
                    input_artifact=str(output_directory / "input.bin"),
                    motor_artifact=str(output_directory / "motor_output.bin"),
                    sitdown_log=str(output_directory / "sitdown.stderr"),
                    receipt_artifact=str(output_directory / "receipt.json"),
                )
                receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
                receipt["source_checkpoint_unchanged"] = sha256(source) == source_before
                if not receipt["source_checkpoint_unchanged"]:
                    raise RuntimeError("source checkpoint changed during experiment")
                (Path(directory) / "input.bin").write_bytes(body)
                (Path(directory) / "motor_output.bin").write_bytes(stdout)
                (Path(directory) / "sitdown.stderr").write_bytes(stderr)
                (Path(directory) / "receipt.json").write_text(
                    json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(directory, output_directory)
    except Exception as error:
        receipt["error"] = str(error)
    finally:
        receipt.setdefault("elapsed_seconds", round(time.monotonic() - started, 6))
        receipt.setdefault(
            "source_checkpoint_unchanged", sha256(source) == source_before
        )
        if not receipt["source_checkpoint_unchanged"]:
            receipt["status"] = "RED"
            receipt["error"] = "source checkpoint changed during experiment"

    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "EXPERIMENT_PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(json.dumps({"status": "RED", "error": str(error)}, sort_keys=True))
        raise SystemExit(2)
