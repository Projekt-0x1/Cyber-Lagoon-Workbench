#!/usr/bin/env python3
"""Thin executable receipt for the organism-owned frontier curriculum.

All developmental logic lives in ReferenceOrganismV2.  This file only instantiates one
Adult, runs the built-in continuing curriculum once, prints the receipt, and exits RED/GREEN.
"""
from __future__ import annotations
import json
from reference_organism_v2 import ReferenceOrganismV2


def main():
    organism = ReferenceOrganismV2()
    receipt = organism.run_frontier_multilingual_curriculum()
    print("FOUNDRY_REFERENCE_ORGANISM_FRONTIER_CURRICULUM_" + receipt["status"])
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if receipt["status"] == "GREEN" else 1)


if __name__ == "__main__":
    main()
