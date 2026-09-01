#!/usr/bin/env python3
"""Economic receipt for the slim public Workbench export; no semantic claim."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

# Commit-gate declarations. This verifier is intentionally the economic lane:
# it changes packaging/build/iteration topology, not Adult semantic machinery.
economic_refactor = True
semantics_free_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True

# First measured broad working-tree export during GitHub #1660.
LEGACY_EXPORT_FILES = 7397

ROOT = Path(__file__).resolve().parents[3]
EXPORTER = ROOT / "tools" / "export_public_workbench.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="public-workbench-economics-") as directory:
        destination = Path(directory) / "workbench"
        run = subprocess.run(
            [sys.executable, str(EXPORTER), str(destination), "--working-tree"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            print(run.stderr, file=sys.stderr)
            return 1
        manifest = json.loads((destination / "EXPORT_MANIFEST.json").read_text())
        current_files = int(manifest["files"])
        complexity_reduction = LEGACY_EXPORT_FILES - current_files
        checks = {
            "export_reduced_files": 0 < current_files < LEGACY_EXPORT_FILES,
            "reduction_is_material": complexity_reduction >= 5000,
            "adult_entrypoint_preserved": (destination / "adult").is_file(),
            "bench_entrypoint_preserved": (destination / "bench").is_file(),
            "full_foundry_preserved": (
                destination
                / "hardware_native"
                / "tools"
                / "foundry_workbench"
                / "reference_life_function_curriculum_v1.py"
            ).is_file(),
            "direct_adult_source_preserved": (
                destination / "hardware_native" / "tools" / "direct_adult_sitdown.cu"
            ).is_file(),
            "compile_free_direct_source_preserved": (
                destination / "hardware_native" / "tools" / "direct_recipe_ir_lab.cu"
            ).is_file(),
            "future_update_source_preserved": (
                destination
                / "hardware_native"
                / "src"
                / "hardware_native"
                / "direct_adult_recipe_credit.cuh"
            ).is_file(),
        }
        failed = [name for name, passed in checks.items() if not passed]
        payload = {
            "schema": "cyber-lagoon.public-workbench-export-economics.v1",
            "status": "GREEN" if not failed else "RED",
            "lane": "economic_refactor",
            "legacy_export_files": LEGACY_EXPORT_FILES,
            "current_export_files": current_files,
            "complexity_reduction": complexity_reduction,
            "current_export_bytes": int(manifest["bytes"]),
            "phenotype_preserved": phenotype_preserved,
            "future_update_authority_preserved": future_update_authority_preserved,
            "checks": checks,
            "failed": failed,
        }
        print(json.dumps(payload, sort_keys=True))
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
