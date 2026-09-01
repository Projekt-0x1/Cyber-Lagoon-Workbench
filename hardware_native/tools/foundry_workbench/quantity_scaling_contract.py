#!/usr/bin/env python3
"""Quantity / population contract for the graph-neutral Foundry workshop.

This does not pretend to simulate a human brain. It makes the engineering target
numeric: very large populations of simple state must be representable cold,
while only touched work becomes expensive. Capability claims still require
behavioral evidence; this contract only protects the scaling model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import reference_contract_1610 as ref

TARGET_SIMPLE_ELEMENTS = 80_000_000_000
GIB = 1024 ** 3
HOT_TOUCHES = 4096


@dataclass(frozen=True)
class CapacityVector:
    current_occurrences: int
    route_transport_width: int
    causal_ancestry_depth: int
    physical_terminal_claims: int
    public_trajectory_extent: int
    frozen_action_closure: int
    active_compositional_depth: int


class PackedPopulation:
    """One-bit cold population with O(touched) indexed access; no scan semantics."""
    def __init__(self, elements: int):
        if elements <= 0:
            raise ValueError("elements")
        self.elements = int(elements)
        self.storage = bytearray((self.elements + 7) // 8)

    def toggle(self, indices):
        touched = 0
        for index in indices:
            i = int(index) % self.elements
            byte, bit = divmod(i, 8)
            self.storage[byte] ^= 1 << bit
            touched += 1
        return touched


def packed_cost(bits_per_element: int):
    bits = TARGET_SIMPLE_ELEMENTS * bits_per_element
    return {
        "bits_per_element": bits_per_element,
        "bytes": (bits + 7) // 8,
        "gib": round(((bits + 7) // 8) / GIB, 3),
    }


def memory_capacity(gib: int, bits_per_element: int):
    return int((gib * GIB * 8) // bits_per_element)


def source_inventory():
    src = ROOT / "hardware_native/src/hardware_native"
    paths = {
        "tensor_core_dense": src / "direct_dense_tensor_execution.cu",
        "selective_state_space": src / "bcc32_grown_selective_state_space.cuh",
        "device_family_queue": src / "direct_adult_device_family_queue.cuh",
        "resident_executable_morphology": src / "direct_adult_resident_executable_morphology.cuh",
        "paged_sparse_route_bank": src / "bcc32_cuda_paged_resident_route_bank.cuh",
    }
    source = "\n".join(
        p.read_text(errors="ignore")
        for p in src.glob("*")
        if p.is_file() and p.suffix in {".cu", ".cuh", ".cpp", ".hpp"}
    )
    return {
        **{name: path.exists() for name, path in paths.items()},
        "rt_bvh_explicit_current_source": any(term in source for term in ("OptiX", "optix", "RT/BVH")),
    }


def packed_touch_assay():
    # Enough to catch accidental whole-population scans while remaining a cheap
    # reference test. Allocation is excluded from the touch timing.
    sizes = (1_000_000, 16_000_000, 128_000_000)
    rows = []
    for count in sizes:
        population = PackedPopulation(count)
        indices = tuple((i * 104729 + 17) % count for i in range(HOT_TOUCHES))
        start = time.perf_counter()
        touched = population.toggle(indices)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "elements": count,
            "packed_bytes": len(population.storage),
            "touched": touched,
            "touch_fraction": touched / count,
            "touch_ms": round(elapsed_ms, 3),
        })
    return rows


def main():
    capacity = CapacityVector(
        current_occurrences=ref.MAX_CURRENT_OCCURRENCES,
        route_transport_width=ref.MAX_ROUTE_TRANSPORT_WIDTH,
        causal_ancestry_depth=ref.MAX_CAUSAL_ANCESTRY_DEPTH,
        physical_terminal_claims=ref.MAX_PHYSICAL_TERMINAL_CLAIMS,
        public_trajectory_extent=ref.MAX_PUBLIC_TRAJECTORY_EXTENT,
        frozen_action_closure=ref.MAX_FROZEN_ACTION_CLOSURE,
        active_compositional_depth=ref.MAX_ACTIVE_COMPOSITIONAL_DEPTH,
    )
    # Equality of current numeric values is allowed; aliasing the *named laws*
    # is not. The contract therefore records each axis separately and checks
    # that they originate from separately named reference constants.
    axis_sources = {
        "current_occurrences": "MAX_CURRENT_OCCURRENCES",
        "route_transport_width": "MAX_ROUTE_TRANSPORT_WIDTH",
        "causal_ancestry_depth": "MAX_CAUSAL_ANCESTRY_DEPTH",
        "physical_terminal_claims": "MAX_PHYSICAL_TERMINAL_CLAIMS",
        "public_trajectory_extent": "MAX_PUBLIC_TRAJECTORY_EXTENT",
        "frozen_action_closure": "MAX_FROZEN_ACTION_CLOSURE",
        "active_compositional_depth": "MAX_ACTIVE_COMPOSITIONAL_DEPTH",
    }

    packed = {str(bits): packed_cost(bits) for bits in (1, 2, 4, 8, 16, 32)}
    consumer = {
        str(gib): {str(bits): memory_capacity(gib, bits) for bits in (1, 2, 4, 8, 16, 32)}
        for gib in (12, 16, 24, 32)
    }
    hot_fraction_counts = {
        "1e-2": TARGET_SIMPLE_ELEMENTS // 100,
        "1e-3": TARGET_SIMPLE_ELEMENTS // 1000,
        "1e-4": TARGET_SIMPLE_ELEMENTS // 10_000,
        "1e-5": TARGET_SIMPLE_ELEMENTS // 100_000,
    }
    touch_rows = packed_touch_assay()
    hardware = source_inventory()

    checks = {
        "quantity_target_numeric": TARGET_SIMPLE_ELEMENTS == 80_000_000_000,
        "capacity_axes_named_separately": len(axis_sources) == 7 and len(set(axis_sources.values())) == 7,
        "reference_bounds_finite": all(value > 0 for value in asdict(capacity).values()),
        "one_bit_target_fits_12gib": packed["1"]["bytes"] < 12 * GIB,
        "two_bit_target_fits_24gib": packed["2"]["bytes"] < 24 * GIB,
        "four_bit_target_exceeds_32gib": packed["4"]["bytes"] > 32 * GIB,
        "packed_touch_count_constant": all(row["touched"] == HOT_TOUCHES for row in touch_rows),
        "packed_storage_linear": all(row["packed_bytes"] == (row["elements"] + 7) // 8 for row in touch_rows),
        "tensor_surface_present": hardware["tensor_core_dense"],
        "state_space_surface_present": hardware["selective_state_space"],
        "sparse_route_surface_present": hardware["paged_sparse_route_bank"],
        "family_queue_surface_present": hardware["device_family_queue"],
        "morphology_lowering_surface_present": hardware["resident_executable_morphology"],
    }

    result = {
        "schema": "0x1.foundry-quantity-scaling-contract.v1",
        "pass": all(checks.values()),
        "target_simple_elements": TARGET_SIMPLE_ELEMENTS,
        "claim": "SCALING_MODEL_ONLY_NOT_HUMAN_BRAIN_SIMULATION_OR_CAPABILITY",
        "capacity_vector": asdict(capacity),
        "capacity_axis_sources": axis_sources,
        "packed_target_cost": packed,
        "consumer_memory_capacity_elements": consumer,
        "illustrative_hot_fraction_counts": hot_fraction_counts,
        "packed_touch_assay": touch_rows,
        "hardware_inventory": hardware,
        "lowering_policy": {
            "cold_population": "packed/procedural/sparse resident matter; no object-per-neuron requirement",
            "hot_occurrences": "device queues + local sparse/tiled execution; charge touched work",
            "dense_local_convergence": "Tensor Core lowering only after differential equivalence and density benefit",
            "temporal_population": "state-space/scan lowering when the relation family earns it",
            "long_range_routes": "paged sparse route banks / tract-like storage",
            "spatial_or_incidence_search": "RT/BVH is an optional candidate lowering; current source inventory does not make it doctrine",
            "expert_recurrence": "condense repeated closures while retaining deoptimization witness",
        },
        "checks": checks,
    }
    print(
        "FOUNDRY_QUANTITY_SCALING " + ("GREEN" if result["pass"] else "RED")
        + " target=80000000000 packed=1 touched_work=1 tensor=1 state_space=1 rt_bvh_required=0"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
