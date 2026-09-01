#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"

# The timeout is part of the contract. This runner performs no build, network,
# CUDA, subprocess transformer, or production-Adult operation.
exec timeout --signal=KILL 60s \
  python3 hardware_native/tools/foundry_workbench/reference_contract_suite.py
