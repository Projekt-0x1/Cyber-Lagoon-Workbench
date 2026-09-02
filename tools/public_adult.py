#!/usr/bin/env python3
"""Start the one public Adult through a terminal or Claude Code body."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import secrets
import selectors
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".state"
REFERENCE_CHECKPOINT = STATE / "adult.json"
DIRECT_CHECKPOINT = STATE / "direct-adult.xcb"
DIRECT_SITDOWN = ROOT / ".build" / "direct" / "direct_adult_sitdown"
REFERENCE_GATEWAY = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_claude_gateway_v1.py"
REFERENCE_TERMINAL = ROOT / "hardware_native" / "tools" / "foundry_workbench" / "reference_language_mastery_terminal_v1.py"
DIRECT_GATEWAY = ROOT / "tools" / "public_direct_adult_gateway.py"
BODY_ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "COLORTERM")
GATEWAY_ENV_ALLOW = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
)
GATEWAY_STARTUP_SECONDS = 10.0
GATEWAY_BOOTSTRAP = """\
import runpy
import sys
from pathlib import Path
credential = sys.stdin.readline().rstrip("\\n")
if not credential:
    raise SystemExit("body:empty-gateway-credential")
script, credential_flag, *args = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
sys.argv = [script, *args, credential_flag, credential]
runpy.run_path(script, run_name="__main__")
"""


def symlink_free(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return False
    current = ROOT
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def regular_file(path: Path, executable: bool = False) -> bool:
    return (
        symlink_free(path)
        and path.is_file()
        and not path.is_symlink()
        and (not executable or os.access(path, os.X_OK))
    )


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"public-adult:symlinked-private-directory:{path.name}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise RuntimeError(f"public-adult:private-directory-unavailable:{path.name}")
    os.chmod(path, 0o700)


@contextmanager
def state_lock():
    lock_path = STATE / "adult.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise RuntimeError("public-adult:state-lock-unavailable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("public-adult:state-lock-invalid")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("public-adult:state-busy") from error
        except OSError as error:
            raise RuntimeError("public-adult:state-lock-unavailable") from error
        yield descriptor
    finally:
        os.close(descriptor)


def backend_available(name: str) -> bool:
    if name == "direct":
        return regular_file(DIRECT_CHECKPOINT) and regular_file(DIRECT_SITDOWN, executable=True)
    return regular_file(REFERENCE_CHECKPOINT)


def choose_backend(requested: str) -> str:
    if requested == "auto":
        # The continuing Claude dialogue path is reference-backed until Direct
        # partner/world consequence settlement is proved end to end. Real Direct
        # remains available explicitly and through ./bench without recompilation.
        requested = "reference"
    if not backend_available(requested):
        raise RuntimeError(f"public-adult:{requested}-backend-unavailable; run ./build")
    return requested


def gateway_command(backend: str) -> list[str]:
    if backend == "direct":
        script = DIRECT_GATEWAY
        credential_flag = "--credential"
        args = [
            "--sitdown",
            str(DIRECT_SITDOWN),
            "--resume",
            str(DIRECT_CHECKPOINT),
            "--port",
            "0",
        ]
    else:
        script = REFERENCE_GATEWAY
        credential_flag = "--auth-token"
        args = ["--resume", str(REFERENCE_CHECKPOINT), "--port", "0"]
    if not regular_file(script):
        raise RuntimeError("public-adult:gateway-unavailable")
    return [
        sys.executable,
        "-I",
        "-c",
        GATEWAY_BOOTSTRAP,
        str(script),
        credential_flag,
        *args,
    ]


def selected_environment(names: tuple[str, ...]) -> dict[str, str]:
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def gateway_environment() -> dict[str, str]:
    return selected_environment(GATEWAY_ENV_ALLOW)


def isolated_body_environment(port: int, credential: str, body_dir: Path) -> dict[str, str]:
    env = selected_environment(BODY_ENV_ALLOW)
    path_entries = []
    for raw in env.get("PATH", "").split(os.pathsep):
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if candidate == ROOT or ROOT in candidate.parents:
            continue
        path_entries.append(str(candidate))
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    else:
        env.pop("PATH", None)
    isolated = str(body_dir)
    env["HOME"] = isolated
    env["PWD"] = isolated
    env["TMPDIR"] = isolated
    env["CLAUDE_CONFIG_DIR"] = isolated
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["ANTHROPIC_" + "AUTH_TOKEN"] = credential
    env["ANTHROPIC_" + "API_KEY"] = credential
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def gateway_ready(gateway: subprocess.Popen[str]) -> int:
    assert gateway.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        selector.register(gateway.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=GATEWAY_STARTUP_SECONDS):
            raise RuntimeError("public-adult:gateway-startup-timeout")
    finally:
        selector.close()
    ready = gateway.stdout.readline().strip().split()
    if len(ready) < 2 or not ready[-1].isdigit():
        raise RuntimeError("public-adult:gateway-startup-refused")
    return int(ready[-1])


def run_claude(backend: str, model: str, prompt: str | None) -> int:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("public-adult:claude-code-not-found; use ./adult --terminal")
    claude_path = Path(claude).resolve()
    if claude_path == ROOT or ROOT in claude_path.parents:
        raise RuntimeError("public-adult:repo-local-claude-refused")

    credential = "local-" + secrets.token_hex(32)
    gateway = subprocess.Popen(
        gateway_command(backend),
        cwd=ROOT,
        env=gateway_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert gateway.stdin is not None
        try:
            gateway.stdin.write(credential + "\n")
            gateway.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError("public-adult:gateway-credential-refused") from error
        finally:
            gateway.stdin.close()

        port = gateway_ready(gateway)
        with tempfile.TemporaryDirectory(prefix=".claude-body.", dir=STATE) as temporary:
            body_dir = Path(temporary)
            private_directory(body_dir)
            env = isolated_body_environment(port, credential, body_dir)

            print(f"CYBER_LAGOON_ADULT backend={backend} body=claude-code", flush=True)
            command = [
                str(claude_path),
                "--bare",
                "--restricted",
                "--no-chrome",
                "--model",
                model,
            ]
            if prompt is None:
                return subprocess.run(command, cwd=body_dir, env=env, check=False).returncode
            command += [
                "-p",
                "--no-session-persistence",
                "--max-turns",
                "1",
                "--output-format",
                "json",
            ]
            return subprocess.run(
                command,
                cwd=body_dir,
                env=env,
                input=prompt,
                text=True,
                check=False,
            ).returncode
    finally:
        if gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "reference", "direct"), default="auto")
    parser.add_argument("--terminal", action="store_true", help="use the raw reference terminal body")
    parser.add_argument("--model", default="sonnet", help="Claude Code client label; cognition still comes from the Adult gateway")
    parser.add_argument("--print-backend", action="store_true")
    parser.add_argument("--prompt", help="send one normal-sentence contact through Claude Code and exit")
    args = parser.parse_args()

    if not STATE.exists() or not STATE.is_dir() or STATE.is_symlink():
        raise RuntimeError("public-adult:state-directory-unavailable; run ./build")
    private_directory(STATE)
    if args.print_backend:
        print(choose_backend(args.backend))
        return 0
    with state_lock() as lock_descriptor:
        backend = choose_backend(args.backend)
        if args.terminal:
            if backend != "reference":
                raise RuntimeError("public-adult:raw-terminal-currently-reference-only")
            if not regular_file(REFERENCE_TERMINAL):
                raise RuntimeError("public-adult:terminal-unavailable")
            os.set_inheritable(lock_descriptor, True)
            os.execv(
                sys.executable,
                [sys.executable, str(REFERENCE_TERMINAL), "--resume", str(REFERENCE_CHECKPOINT)],
            )
        if backend == "direct" and args.prompt is None:
            raise RuntimeError(
                "public-adult:direct-claude-is-one-shot-until-world-consequence-settlement-is-proved"
            )
        return run_claude(backend, args.model, args.prompt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
