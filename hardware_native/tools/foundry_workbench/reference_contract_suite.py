#!/usr/bin/env python3
"""Single-process, bounded authoritative reference contract suite."""
import time

import authority_boundary_verify
import reference_contract_1610_verify
import starting_state_freezer_verify
import strict_causal_language_closure_verify


def main():
    started = time.monotonic()
    authority_boundary_verify.main()
    starting_state_freezer_verify.main()
    reference_contract_1610_verify.main()
    strict_causal_language_closure_verify.main()
    elapsed = time.monotonic() - started
    if elapsed >= 60:
        raise SystemExit("FOUNDRY_REFERENCE_SUITE_RED deadline")
    print(f"FOUNDRY_REFERENCE_SUITE_GREEN elapsed_ms={elapsed * 1000:.3f} "
          "strict_authority=1 authored_vehicle_freezer=1 resident_contract=1 "
          "causal_language_closure=1 "
          "semantic_fixture_interchange=0 legacy_interchange=0")


if __name__ == "__main__":
    main()
