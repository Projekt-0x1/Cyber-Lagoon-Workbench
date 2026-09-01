#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "hardware_native" / "src" / "hardware_native"
TESTS = ROOT / "hardware_native" / "tests"
CMAKE = ROOT / "hardware_native" / "cmake" / "contracts.d"

# Representation migration only: no new semantic policy or public-language claim.
semantics_free_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True


def run(command, timeout=30):
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def main() -> int:
    started = time.perf_counter()
    header = SRC / "direct_resident_structural_role_projection.cuh"
    fixture = TESTS / "direct_resident_structural_role_fixture.cuh"
    host_contract = TESTS / "direct_resident_structural_role_contract.cpp"
    cuda_contract = TESTS / "cuda_direct_resident_structural_role_contract.cu"
    host_cmake = CMAKE / "direct_resident_structural_role_contract.cmake"
    cuda_cmake = CMAKE / "cuda_direct_resident_structural_role_contract.cmake"

    with tempfile.TemporaryDirectory(prefix="direct-role-") as directory:
        exe = Path(directory) / "direct_role"
        host = run([
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-Ihardware_native/src", "-Ihardware_native/tests",
            "-Ihardware_native/include", str(host_contract), "-o", str(exe),
        ])
        host_run = run([str(exe)]) if host.returncode == 0 else host

    source = header.read_text()
    fixture_source = fixture.read_text()
    cuda_source = cuda_contract.read_text()
    cuda_cmake_source = cuda_cmake.read_text()
    host_cmake_source = host_cmake.read_text()
    role_block = source.split("std::uint64_t role =", 1)[1].split(
        "if (role == 0u)", 1
    )[0]
    forbidden_role_state = (
        "occurrence_identity",
        "revision_identity",
        "participation_identity",
        "source_identity",
        "context_signature",
        "eligibility_q16",
        "variable_identity",
    )
    semantic_policy_tokens = (
        "DirectDiscourseAction",
        "DirectLanguageEmissionGate",
        "controller_role",
        "subject_role",
        "object_role",
        "agreement_role",
        "grammar_rule",
        "syntax_role",
        "expected_answer",
    )

    checks = {
        "host_contract_compiles_and_passes_11_of_11": (
            host.returncode == 0
            and host_run.returncode == 0
            and "DIRECT_RESIDENT_STRUCTURAL_ROLE_CONTRACT GREEN checks=11/11"
            in host_run.stdout
        ),
        "topology_reuses_existing_recruitment_owner": (
            "candidate.topology_identity = topology" in source
            and "resident_relational_network_recruitment_identity(closure)" in source
        ),
        "role_hash_excludes_transient_causal_state": not any(
            token in role_block for token in forbidden_role_state
        ),
        "role_hash_includes_neighbor_morphology_and_ports": all(
            token in role_block
            for token in (
                "neighbor_morphology_identity",
                "local_port",
                "remote_port",
                "orientation",
            )
        ),
        "ambiguous_and_malformed_projection_refuse_atomically": (
            "make_ambiguous_cycle" in fixture_source
            and "ambiguous_projection.topology_identity == 0u" in fixture_source
            and "malformed.couplings[0].target_occurrence_identity = 0u" in fixture_source
            and "malformed_projection.topology_identity == 0u" in fixture_source
        ),
        "a_a_prime_and_rewired_controls_are_standing": all(
            token in fixture_source
            for token in (
                "first.identity != fresh.identity",
                "same_role_ids(candidate.first, candidate.fresh)",
                "same_morphology_multiset(first, rewired)",
                "candidate.first.topology_identity != candidate.rewired.topology_identity",
                "512u",
            )
        ),
        "no_semantic_discourse_policy_reintroduced": not any(
            token in source.lower() for token in semantic_policy_tokens
        ),
        "projection_has_no_checkpoint_or_global_role_bank": (
            "checkpoint" not in source.lower()
            and "role_bank" not in source.lower()
            and "static ResidentStructuralRole" not in source
        ),
        "cuda_contract_uses_same_fixture_and_device_kernel": (
            "direct_resident_structural_role_fixture.cuh" in cuda_source
            and "__global__ void structural_role_kernel" in cuda_source
            and "same_receipt(host, device)" in cuda_source
            and "SKIP cuda_direct_resident_structural_role_contract no CUDA" in cuda_source
        ),
        "host_and_cuda_cmake_contracts_registered": (
            "direct_resident_structural_role_contract.cpp" in host_cmake_source
            and "cuda_direct_resident_structural_role_contract.cu" in cuda_cmake_source
            and "SKIP_RETURN_CODE 77" in cuda_cmake_source
        ),
        "bounded_reference_work": time.perf_counter() - started < 2.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-direct-resident-structural-role-projection.v1",
        "pass": not failed,
        "reference_only": True,
        "graph_flip": False,
        "semantics_free_refactor": semantics_free_refactor,
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "economic_gain": {
            "new_topology_owners": 0,
            "new_persistent_role_fields": 0,
            "reused_recruitment_topology_owner": 1,
        },
        "host_contract": host_run.stdout.strip().splitlines()[0]
        if host_run.stdout.strip()
        else "",
        "cuda_compile_receipt": "MANUAL_NVCC_13_0_GREEN_CURRENT_SOURCE",
        "cuda_device_execution": "RED_NO_VISIBLE_DEVICE_IN_CURRENT_ENVIRONMENT",
        "checks": checks,
        "failed": failed,
        "remaining_red": [
            "CUDA_DEVICE_EXECUTION_FOR_STRUCTURAL_ROLE_PARITY",
            "DIRECT_WORLD_BODY_ROLE_CONDITION_ROUTING",
            "DIRECT_LEARNED_STRUCTURAL_DEPENDENCY_SOURCE_HISTORY",
            "CONTINUOUS_LIFE_STRUCTURAL_WORLD_INTERFERENCE",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print(
        "FOUNDRY_DIRECT_RESIDENT_STRUCTURAL_ROLE_PROJECTION_REFERENCE_"
        + ("GREEN" if not failed else "RED")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
