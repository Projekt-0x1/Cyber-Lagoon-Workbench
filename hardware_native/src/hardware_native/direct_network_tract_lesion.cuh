#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_TRACT_LESION_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_TRACT_LESION_CUH

// github #1268 -- ADULT DISCONNECTIONS.
//
// #1268's "Adult disconnections" section requires cutting a named tract while
// PRESERVING its endpoint territories, and states the constraint that governs
// every probe built on top of it:
//
//   "Territory lesion and tract disconnection must not be conflated."
//
// Nothing in the tree could cut a tract before this module. The only ablation
// that existed (direct_network_basin_probe's `ablate_inhibition_kernel`)
// selects tissue by a FLAG the genome grew -- the inhibitory sign -- which
// cannot address a corridor, because a corridor is not a flag but a relation
// between two territories.
//
// WHAT MAKES THIS A DISCONNECTION RATHER THAN A NEW MECHANISM. The organism
// already retracts a route by clearing `kRouteFlagActive`
// (direct_network_resident_development.cu:496), and the adult executor already
// refuses to propagate over a route without it (direct_adult_core.cu:373,
// `if (!(route.flags & kRouteFlagActive)) continue;`). This module selects
// WHICH routes an experimenter clears; it introduces no new causal path and no
// new flag. A cut here is the same event the organism can perform on itself.
//
// THE CONFLATION THIS MODULE EXISTS TO MAKE MEASURABLE. A readout that only
// counts corridor routes CANNOT distinguish the two interventions #1268 says
// must never be conflated: cutting the tract drives that count to zero, and
// lesioning either endpoint territory drives the same count to zero. The
// endpoint fields on `TractCensus` below are what separates them -- they
// survive a tract cut and do not survive a territory lesion. Any D-series arm
// that reports only `corridor_routes` is reporting a quantity that is blind to
// the distinction its own issue requires.

#include <cstdint>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network::tract_lesion {

// Matches any territory. As a corridor endpoint this widens the selection
// (every long tract OUT of the source, whatever it reaches); it never widens
// what a lesion touches beyond routes.
inline constexpr std::uint32_t kAnyTerritory = 0xffffffffu;

// One corridor's tissue, counted at BOTH grains an adult disconnection can be
// read at. Every count is MEASURED off the grown routes; `declared_active_routes`
// is the brain header's own number, recorded beside the measurement and never
// in place of it, so a stale header cannot pass as a census.
struct TractCensus {
  // The endpoints' matter. A tract cut must leave both of these untouched --
  // that is the whole content of "preserve endpoint territories".
  std::uint32_t source_nodes = 0u;
  std::uint32_t target_nodes = 0u;

  // The corridor itself: active long tracts from source territory to target.
  // Driven to zero by a tract cut AND by a lesion of either endpoint, so this
  // number alone cannot tell the two apart.
  std::uint32_t corridor_routes = 0u;

  // The three quantities that DO tell them apart. Each survives a tract cut
  // and dies under the corresponding territory lesion.
  std::uint32_t source_other_routes = 0u;      // active routes out of source, not this corridor
  std::uint32_t target_incoming_other = 0u;    // active routes into target, not this corridor
  std::uint32_t target_outgoing_routes = 0u;   // active routes out of target

  std::uint32_t active_routes_total = 0u;
  std::uint32_t declared_active_routes = 0u;

  // WHAT A SHAM CAN AND CANNOT BE MATCHED ON. A sham that removes the same
  // NUMBER of routes is dose-matched on count; whether it can also be matched
  // on KIND depends on whether the organism holds any long tract, or any
  // between-territory route at all, outside the corridor being cut. These two
  // totals answer that from the tissue instead of leaving a reader to assume
  // it: when `corridor_routes` equals `inter_territory_routes`, the corridor IS
  // the organism's entire between-territory connectivity, and no kind-matched
  // sham exists to build -- a limit of the SUBJECT, which a receipt must state
  // rather than let a count-matched sham imply it was controlled for.
  std::uint32_t long_tract_routes = 0u;
  std::uint32_t inter_territory_routes = 0u;

  // github #1276 rung 3: mean route.conductance_q16 over the corridor routes
  // above, i.e. the corridor's measured GAIN rather than merely its presence.
  // Zero when corridor_routes is zero -- a caller comparing this across two
  // organisms must gate on corridor_routes > 0 first, the same discipline
  // corridor_routes itself already requires of a lesion/sham comparison.
  std::int32_t mean_corridor_conductance_q16 = 0;

  // False when the tissue could not be read back. A caller must not read a
  // failed census as a negative result about the organism.
  bool valid = false;
};

TractCensus census_tract(const DirectBrain& brain,
                         std::uint32_t source_territory,
                         std::uint32_t target_territory);

struct LesionCounts {
  std::uint32_t routes_deactivated = 0u, source_node = kInvalidIndex, source_active_before = 0u, source_active_after = 0u;
  // The legacy disconnection operators leave this zero; the accounted focal
  // failure reports its source/target incidence writes here.
  std::uint32_t nodes_written = 0u, history_records = 0u; std::uint64_t matter_returned = 0u; std::uint16_t source_territory = 0u, reserved = 0u;
  bool applied = false;
};

// Clear `kRouteFlagActive` on every active LONG-TRACT route running from
// `source_territory` into `target_territory`. Writes no node and no route
// outside that set.
LesionCounts disconnect_tract(DirectBrain& brain,
                              std::uint32_t source_territory,
                              std::uint32_t target_territory);

// The intervention at the OTHER grain, and the reason this module reports
// endpoint counts at all: remove a territory's matter from the causal graph by
// deactivating every active route incident on its nodes, incoming and outgoing.
// This is what a tract cut must be distinguishable FROM.
LesionCounts lesion_territory(DirectBrain& brain, std::uint32_t territory); LesionCounts fail_source_routes_accounted(DirectBrain& brain, std::uint32_t source_node, std::uint32_t viability_floor = 1u, const std::uint32_t* causal_pin_bits = nullptr);

// Dose-matched sham: deactivate exactly `budget` active routes that are NOT in
// the named corridor, chosen in route-slot order so the selection is
// deterministic across runs.
//
// `long_tract_only` decides what the dose is matched ON, and the two answers
// support different claims. With it false the sham removes the same NUMBER of
// routes, which in practice means local recurrence -- enough to show an outcome
// was not caused by writing route flags, not enough to show it was caused by
// THIS tract rather than by losing a tract. With it true the sham removes the
// same number of routes AND the same KIND, other long tracts, which is the
// control that isolates the corridor's identity. Use the strict one wherever
// the organism holds a non-corridor long tract to spend; `TractCensus::
// long_tract_routes` says whether it does.
LesionCounts sham_disconnect(DirectBrain& brain,
                             std::uint32_t source_territory,
                             std::uint32_t target_territory,
                             std::uint32_t budget,
                             bool long_tract_only = false);

}  // namespace substrate::direct_network::tract_lesion

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_TRACT_LESION_CUH
