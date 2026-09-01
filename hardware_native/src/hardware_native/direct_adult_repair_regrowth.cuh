#ifndef HARDWARE_NATIVE_DIRECT_ADULT_REPAIR_REGROWTH_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_REPAIR_REGROWTH_CUH

// e.repair (#1526): autonomous structural regrowth and route restoration
// replacing damaged or degraded local network regions.
//
// The focal repair path (direct_network_repair_regrowth.inl) nominates only
// when a loss source retains fewer than two active routes. A DEGRADED region
// -- members still carrying traffic through surviving routes -- is exactly
// what both existing nomination sites ignore, so regional damage that stops
// short of isolating its members never regrows. This module closes that gap
// with one device-owned nomination pass over committed retraction history;
// everything downstream stays the adult's own machinery (front proposal,
// atomic target arbitration, budget admission, exact-history growth commit).
//
// Law anchors (Revision 12):
//   * exact participating authority: every recruited front binds the full
//     64-bit committed loss identity with route, target and post-lesion
//     incarnation -- regrowth is bound to what physically happened;
//   * homeostasis: recruitment is gated on device-owned repair-field
//     influence at the loss territory, never on a host instruction;
//   * finite matter: bounded recruits per pass plus a constructor-reserve
//     gate; the L2 transaction retains final admission authority;
//   * no semantic branches: every decision reads resident data (field
//     strength, recipe opcodes, history kinds, route counters).

#include <cstdint>

#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_network_resident_development.cuh"

namespace substrate::direct_network {

// Committed records scanned per pass. Bounded work: nomination cost is
// independent of adult size and total history depth.
inline constexpr std::uint32_t kRegionRepairLookbackRecords = 64u;
inline constexpr std::uint32_t kRegionRepairFrontsPerPass = 4u;

struct RepairRegionCensus {
  std::uint32_t records_scanned;
  std::uint32_t loss_records;
  std::uint32_t degraded_sources;
  std::uint32_t fronts_recruited;
  std::uint32_t refused_owned;
  std::uint32_t refused_unlawful;
  std::uint32_t refused_capacity;
};

namespace detail {

__device__ inline std::int32_t region_abs(std::int32_t value) {
  return value < 0 ? -value : value;
}

__device__ inline std::int32_t region_clamp_i32(std::int32_t value,
                                                std::int32_t lo,
                                                std::int32_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

__device__ inline bool region_front_owns_or_saw_loss(const DirectBrain& brain,
                                                     std::uint32_t source,
                                                     std::uint64_t identity) {
  const std::uint32_t count =
      min(*brain.construction_front_count, brain.construction_front_capacity);
  const std::uint64_t watermark =
      brain.construction_front_generation_by_node[source];
  for (std::uint32_t i = 0u; i < count; ++i) {
    const ResidentConstructionFront& front = brain.construction_fronts[i];
    if (construction_front_has_loss_identity(front, identity)) return true;
    if (front.state == kConstructionFrontLive && front.source_node == source &&
        front.generation == watermark)
      return true;
  }
  return false;
}

// Resident repair-field influence at the loss node, mirroring the executor's
// own bound-field law (window, chemotype guard, linear distance falloff).
__device__ inline std::int32_t region_repair_field_influence_q16(
    const DirectBrain& brain, const DirectNode& node, std::uint32_t age) {
  if (node.territory_index >= brain.resident_field_range_count) return 0;
  const ResidentFieldRange range =
      brain.resident_field_ranges[node.territory_index];
  if (static_cast<std::uint64_t>(range.index_offset) + range.index_count >
      brain.resident_field_index_count)
    return 0;
  const std::uint64_t logical_tick =
      static_cast<std::uint64_t>(brain.development->birth_handoff_tick) + age;
  std::int64_t sum = 0;
  for (std::uint32_t i = 0u; i < range.index_count; ++i) {
    const std::uint16_t field_index =
        brain.resident_field_indices[range.index_offset + i];
    if (field_index >= brain.resident_field_count) continue;
    const ResidentDevelopmentField& field = brain.resident_fields[field_index];
    if (field.kind != DevelopmentFieldKind::repair) continue;
    if (logical_tick < field.begin_tick) continue;
    if (field.end_tick != 0u && logical_tick >= field.end_tick) continue;
    if ((node.chemotype & field.require_mask) != field.require_value) continue;
    std::int64_t distance = 0;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis)
      distance += detail::region_abs(node.coordinate[axis] - field.center[axis]);
    const std::uint64_t radius = static_cast<std::uint64_t>(field.radius) * 3ull;
    if (radius == 0ull ||
        static_cast<std::uint64_t>(distance) > radius)
      continue;
    const std::int64_t falloff_q16 =
        ((static_cast<std::int64_t>(radius) - distance) << 16) /
        static_cast<std::int64_t>(radius);
    sum += (static_cast<std::int64_t>(resident_field_decayed_strength_q16(
                field, logical_tick)) *
            falloff_q16) >>
           16;
  }
  return region_clamp_i32(static_cast<std::int32_t>(sum), -(16 << 16),
                          16 << 16);
}

__device__ inline std::uint32_t region_repair_recipe_cell(
    const DirectBrain& brain, const DirectNode& node) {
  if (node.territory_index >= brain.recipe_range_count) return kInvalidIndex;
  const ResidentRecipeRange range = brain.recipe_ranges[node.territory_index];
  for (std::uint32_t local = 0u; local < range.index_count; ++local) {
    const std::uint32_t index = range.index_offset + local;
    if (index >= brain.recipe_index_count) break;
    const std::uint32_t cell_index = brain.recipe_indices[index];
    if (cell_index >= brain.recipe_cell_count) continue;
    const ResidentRecipeCell& cell = brain.recipe_cells[cell_index];
    if (cell.rule_index < brain.resident_rule_count &&
        brain.resident_rules[cell.rule_index].opcode == RuleOpcode::repair &&
        (brain.resident_rules[cell.rule_index].flags & kRuleFlagPostBirthResident) != 0u)
      return cell_index;
  }
  return kInvalidIndex;
}

}  // namespace detail

__device__ inline void nominate_degraded_region_fronts_impl(
    DirectBrain brain, ResidentDevelopmentCounters* counters,
    RepairRegionCensus* census) {
  if (census == nullptr || brain.development == nullptr ||
      brain.construction_front_count == nullptr || brain.nodes == nullptr)
    return;
  *census = RepairRegionCensus{};
  const DirectExactHistoryHotPage& history = brain.development->exact_history;
  const std::uint32_t total = history.committed_slots;
  const std::uint32_t begin =
      total > kRegionRepairLookbackRecords ? total - kRegionRepairLookbackRecords : 0u;
  std::uint32_t recruited = 0u;
  for (std::uint32_t ordinal = begin; ordinal < total; ++ordinal) {
    ++census->records_scanned;
    if (recruited >= kRegionRepairFrontsPerPass) return;
    const DirectExactHistoryRecord& record = history.records[ordinal];
    if (record.kind != DirectExactHistoryKind::topology_retraction ||
        record.identity == 0u)
      continue;
    ++census->loss_records;
    const std::uint32_t source = record.source;
    if (source >= brain.node_count) continue;
    const DirectNode& node = brain.nodes[source];
    // Below two surviving routes the focal path owns the response.
    if (node.active_route_count < 2u) continue;
    ++census->degraded_sources;
    if (detail::region_front_owns_or_saw_loss(brain, source, record.identity)) {
      ++census->refused_owned;
      continue;
    }
    const std::int32_t repair_pressure = detail::region_repair_field_influence_q16(
        brain, node, brain.development->age_tick);
    const std::uint32_t cell_index =
        repair_pressure <= (1 << 14)
            ? kInvalidIndex
            : detail::region_repair_recipe_cell(brain, node);
    if (cell_index == kInvalidIndex ||
        brain.development->constructor_reserve == 0ull) {
      ++census->refused_unlawful;
      continue;
    }
    if (*brain.construction_front_count >= brain.construction_front_capacity) {
      ++census->refused_capacity;
      return;
    }
    const ResidentRecipeCell& cell = brain.recipe_cells[cell_index];
    const std::uint64_t generation = next_construction_front_generation(
        brain.construction_front_generation_by_node[source]);
    if (generation == 0u) {
      ++census->refused_unlawful;
      continue;
    }
    brain.construction_fronts[*brain.construction_front_count] =
        ResidentConstructionFront{
            source, cell_index, cell.rule_index, kInvalidIndex, generation,
            0u, 0u, static_cast<std::uint16_t>(node.territory_index),
            kConstructionFrontLive, record.subject, record.identity,
            record.incarnation_after, record.value, source};
    brain.construction_front_generation_by_node[source] = generation;
    *brain.construction_front_count += 1u;
    if (counters != nullptr) ++counters->construction_fronts_recruited;
    ++census->fronts_recruited;
    ++recruited;
  }
}

__global__ void nominate_degraded_region_fronts_kernel(
    DirectBrain brain, ResidentDevelopmentCounters* counters,
    RepairRegionCensus* census) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    nominate_degraded_region_fronts_impl(brain, counters, census);
}

}  // namespace substrate::direct_network

#endif
