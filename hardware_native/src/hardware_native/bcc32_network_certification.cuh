// Patch 0006 of the FULL CUDA network-recipe patch program: CUDA-only
// t0-t6 certification. This landing implements and genuinely measures t0
// (construction fronts) only; t1-t6 are stated as honest, structurally-
// blocked RED with their exact causal reason rather than faked or silently
// skipped -- see bcc32_network_certification.cu's header comment and the
// diary entry for the specific mechanism gaps in patches 0004/0005 that
// block each later stage. This is deliberate: certification measures what
// the mechanism patches actually built; it does not itself build the
// missing mechanism (branching-after-first-commit re-visitation, fusion
// connector installation, multi-basin morphology) that t1+ require.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_CERTIFICATION_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_CERTIFICATION_CUH

#include <cstdint>

#include "hardware_native/bcc32_network_life_function.cuh"

namespace substrate::bcc32::network_recipe {

// t0 -- construction fronts (patch program section "t0 -- construction
// fronts"). Requires: at least two independent genesis lineages; exact
// predicted-vs-measured front geometry; no host-written mature edges;
// matter closure; allocation permutation invariance.
struct T0CertificationReport {
  std::uint32_t distinct_lineages_constructed;
  std::uint32_t nodes_constructed;
  bool matter_closure_held;
  bool front_geometry_exact;
  bool allocation_permutation_invariant;
  bool certified;  // true iff every one of the above requirements holds
};

// Runs a real, bounded t0 construction scenario twice -- once with
// block_size=32, once with block_size=256 -- over a genome with 3
// independent genesis lineages and no coordinate collisions (unlike patch
// 0004's own contract, which deliberately engineers a collision; t0 is
// about front geometry and matter closure, not conflict resolution, which
// patch 0004 already certified on its own). Compares the *set* of
// committed (coordinate, lineage, chemistry) triples between the two runs
// -- not their raw node-pool indices, which are allowed to differ -- for
// allocation-permutation invariance, and independently recomputes each
// committed node's page-directory placement from its known seed geometry
// to verify exact front geometry.
T0CertificationReport run_t0_construction_front_certification();

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_CERTIFICATION_CUH
