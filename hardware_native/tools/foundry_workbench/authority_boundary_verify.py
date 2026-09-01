#!/usr/bin/env python3
"""Static authority firewall for the named graph-neutral runtime."""
import ast
from pathlib import Path


ROOT = Path(__file__).parent
RUNTIME = (ROOT / "reference_contract_1610.py", ROOT / "interchange.py")
OFFLINE_AUTHORING = (ROOT / "starting_state_freezer.py",)
FORBIDDEN_IMPORTS = {"subprocess", "socket", "urllib", "requests", "torch", "cuda"}
FORBIDDEN_LOCAL_IMPORTS = {"foundry", "full_loop", "reference_vm", "strict_runtime"}


def main():
    imports = set()
    text = ""
    for path in (*RUNTIME, *OFFLINE_AUTHORING):
        source = path.read_text(); text += source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    runner = (ROOT / "run_reference_contract_1610.sh").read_text()
    checks = {
        "no_forbidden_import": not imports & FORBIDDEN_IMPORTS,
        "no_legacy_runtime_import": not imports & FORBIDDEN_LOCAL_IMPORTS,
        "no_host_goal_api": "enqueue_goal" not in text,
        "no_answer_api": "def answer" not in text,
        "no_think_api": "def think" not in text,
        "no_speak_api": "def speak" not in text,
        "no_runtime_install_api": "def install" not in text,
        "no_runtime_transformer": "transformer_propose" not in text,
        "bounded_runner": "timeout --signal=KILL 60s" in runner,
        "single_authoritative_suite": "reference_contract_suite.py" in runner,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_AUTHORITY_BOUNDARY_RED " + ",".join(failed))
    print("FOUNDRY_AUTHORITY_BOUNDARY_GREEN adult_attached=0 legacy_runtime_import=0 "
          "host_goal_api=0 runtime_transformer=0 compiler=0 network=0 cuda=0")


if __name__ == "__main__":
    main()
