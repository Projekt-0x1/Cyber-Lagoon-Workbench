#ifndef HARDWARE_NATIVE_DIRECT_SCALE_FRONTIER_METERS_CUH
#define HARDWARE_NATIVE_DIRECT_SCALE_FRONTIER_METERS_CUH

#include <cstdint>

#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_adult {

// Observer-only #1193 categories plus #1313 silicon surfaces. These numbers
// are instruments for SCALE.frontier_not_corpus. They are not a SCALE verdict,
// not a metabolic currency, and not a scheduler.
struct DirectScaleFrontierMeters {
  std::uint64_t s1_persistent_recipe_units;
  std::uint64_t s2_physical_backing_units;
  std::uint64_t s3_live_unresolved_units;
  std::uint64_t s4_active_closures;
  std::uint64_t frontier_work;
  std::uint64_t vram_charged_bytes;
  std::uint64_t live_occurrences;
};

enum class DirectScaleCategory : std::uint8_t {
  none = 0,
  s1_persistent_recipe = 1,
  s2_physical_backing = 2,
  s3_live_unresolved = 3,
};

inline DirectScaleCategory scale_category_of(DirectResourcePoolKind kind) {
  switch (kind) {
    case DirectResourcePoolKind::node_state:
    case DirectResourcePoolKind::derivation_record:
    case DirectResourcePoolKind::derivation_parent_edge:
    case DirectResourcePoolKind::representation_source_state:
    case DirectResourcePoolKind::representation_state_owner:
      return DirectScaleCategory::s1_persistent_recipe;
    case DirectResourcePoolKind::explicit_interaction:
    case DirectResourcePoolKind::implicit_exception:
    case DirectResourcePoolKind::packed_panel:
    case DirectResourcePoolKind::dense_tile:
    case DirectResourcePoolKind::tract_packet:
    case DirectResourcePoolKind::delayed_packet:
    case DirectResourcePoolKind::checkpoint_future_state:
    case DirectResourcePoolKind::boundary_port:
      return DirectScaleCategory::s2_physical_backing;
    case DirectResourcePoolKind::eligibility_record:
    case DirectResourcePoolKind::context_record:
    case DirectResourcePoolKind::pending_consequence_ticket:
      return DirectScaleCategory::s3_live_unresolved;
    default:
      return DirectScaleCategory::none;
  }
}

// Reads the ecology ledger (17 pool rows + work counters) and one live
// Occurrence count from the actual-frontier surface. Does not walk nodes,
// routes, or the Recipe body.
inline DirectScaleFrontierMeters observe_scale_frontier_meters(
    const DirectResourceEcologyState& ecology, std::uint64_t live_occurrences) {
  DirectScaleFrontierMeters meters{};
  const std::uint32_t pool_count =
      static_cast<std::uint32_t>(DirectResourcePoolKind::count);
  for (std::uint32_t i = 0; i < pool_count; ++i) {
    const DirectResourcePoolState& pool = ecology.pools[i];
    switch (scale_category_of(static_cast<DirectResourcePoolKind>(i))) {
      case DirectScaleCategory::s1_persistent_recipe:
        meters.s1_persistent_recipe_units += pool.charged_units;
        break;
      case DirectScaleCategory::s2_physical_backing:
        meters.s2_physical_backing_units += pool.charged_units;
        break;
      case DirectScaleCategory::s3_live_unresolved:
        meters.s3_live_unresolved_units += pool.live_units;
        break;
      default:
        break;
    }
  }
  meters.s4_active_closures = ecology.work.frontier_events;
  meters.frontier_work = ecology.work.interaction_evaluations;
  meters.vram_charged_bytes = ecology.global_charged_bytes;
  meters.live_occurrences = live_occurrences;
  return meters;
}

inline bool scale_frontier_meters_collapsed(const DirectScaleFrontierMeters& meters) {
  return meters.s1_persistent_recipe_units == meters.s2_physical_backing_units &&
         meters.s2_physical_backing_units == meters.s3_live_unresolved_units &&
         meters.s3_live_unresolved_units == meters.s4_active_closures &&
         meters.s4_active_closures == meters.frontier_work &&
         meters.frontier_work == meters.vram_charged_bytes &&
         meters.vram_charged_bytes == meters.live_occurrences;
}

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_SCALE_FRONTIER_METERS_CUH
