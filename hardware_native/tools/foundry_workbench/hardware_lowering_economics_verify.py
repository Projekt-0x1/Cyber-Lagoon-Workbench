#!/usr/bin/env python3
"""Compile and run the lowering assay without configuring the full project."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("FOUNDRY_HARDWARE_LOWERING_ECONOMICS SKIP nvcc=0 direct_parity=NOT_RUN")
        return 77

    source = Path(__file__).with_name("hardware_lowering_economics_probe.cu")
    with tempfile.TemporaryDirectory(prefix="foundry-lowering-") as temporary:
        binary = Path(temporary) / "lowering_probe"
        compile_result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                "-O3",
                "-arch=sm_89",
                "--expt-relaxed-constexpr",
                str(source),
                "-o",
                str(binary),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            sys.stderr.write(compile_result.stdout + compile_result.stderr)
            print("FOUNDRY_HARDWARE_LOWERING_ECONOMICS RED compile=0")
            return 1

        run = subprocess.run(
            [str(binary)], text=True, capture_output=True, timeout=30
        )
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if "SPATIAL_BVH_WORKLOAD equivalence=1" not in run.stdout:
            print("FOUNDRY_HARDWARE_LOWERING_ECONOMICS RED spatial_receipt=0")
            return 1
        if run.returncode == 77:
            print(
                "FOUNDRY_HARDWARE_LOWERING_ECONOMICS SKIP "
                "compile=1 spatial=GREEN gpu=0 direct_parity=NOT_RUN"
            )
            return 77
        if run.returncode != 0 or "LOWERING_ECONOMICS GREEN" not in run.stdout:
            print("FOUNDRY_HARDWARE_LOWERING_ECONOMICS RED gpu_receipt=0")
            return 1
        tensor_rows = [
            row
            for row in run.stdout.splitlines()
            if row.startswith("TENSOR_WMMA_WORKLOAD ")
        ]
        if len(tensor_rows) != 9 or any(
            " equivalence=1 " not in row for row in tensor_rows
        ):
            print("FOUNDRY_HARDWARE_LOWERING_ECONOMICS RED differential_receipt=0")
            return 1
        print(
            "FOUNDRY_HARDWARE_LOWERING_ECONOMICS GREEN "
            "compile=1 tensor_cases=9 spatial=GREEN rt_hardware_proof=0"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
