#pragma once

// Resident developmental regulation for the persistent adult.
//
// This is not a host objective, named cognitive module, or observer-selected
// mode switch.  It is a bounded device-owned ecology that couples fast
// prediction error, slow consolidation, recurrent connectivity, resource
// pressure, structural turnover, repair, field history, and motor
// reafference.  The eight exported context components are physical projections
// consumed by another resident region; they are never semantic labels.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::adaptive_ecology {

inline constexpr std::uint32_t kMaxCells = 256u;
inline constexpr std::uint32_t kRouteCount = 4u;
inline constexpr std::uint32_t kEdgeSlots = 2u;
inline constexpr std::uint32_t kContextWidth = 8u;
inline constexpr std::uint16_t kInitialStructuralMatterQ8 = 64u;
inline constexpr std::uint16_t kEdgeStructuralDoseQ8 = 16u;
inline constexpr std::uint32_t kLocalProposalRadius = 8u;
inline constexpr std::uint16_t kNoCell = 0xffffu;
inline constexpr std::int32_t kSignalLimit = 4096;
inline constexpr std::int32_t kWeightLimit = 4096;

struct PlasticEdge {
  std::uint16_t target = kNoCell;
  std::int16_t weight_q8 = 0;
  std::int16_t eligibility_q8 = 0;
  std::uint16_t utility_q8 = 0u;
  std::uint16_t age = 0u;
  std::uint16_t structural_matter_q8 = 0u;
  std::uint8_t live = 0u;
  std::uint8_t reserved = 0u;
};

struct Cell {
  std::int32_t slow_trace_q12 = 0;
  std::uint32_t contact_credit_q12 = 0u;
  std::int32_t activity_target_q8 = 64;
  std::uint16_t plasticity_q8 = 128u;
  std::uint16_t damage_q8 = 0u;
  std::uint16_t repair_reserve_q8 = 0u;
  std::uint16_t quiet_age = 0u;
  std::uint16_t free_structural_matter_q8 = 0u;
  std::int16_t consolidated_weight[kRouteCount]{};
  std::uint16_t route_utility_q8[kRouteCount]{};
  std::uint16_t route_residual_ema[kRouteCount]{};
  PlasticEdge edges[kEdgeSlots]{};
};

struct State {
  Cell cells[kMaxCells]{};
  std::int32_t fast_context_q12[kContextWidth]{};
  std::int32_t slow_context_q12[kContextWidth]{};
  std::int32_t context_q12[kContextWidth]{};

  std::int64_t epoch_signed_residual = 0;
  std::uint64_t epoch_residual_l1 = 0u;
  std::uint32_t epoch_observations = 0u;

  std::uint32_t plasticity_q8 = 128u;
  std::uint32_t consolidation_q8 = 0u;
  std::uint32_t growth_q8 = 0u;
  std::uint32_t turnover_q8 = 0u;
  std::uint32_t repair_q8 = 0u;
  std::uint32_t replay_q8 = 0u;

  std::uint32_t live_edges = 0u;
  std::uint32_t consolidated_routes = 0u;
  std::uint32_t context_anchor = 0u;
  std::uint64_t last_anchor_contact_revision = 0u;
  std::uint32_t context_l1 = 0u;
  std::uint32_t damage_q8 = 0u;

  std::uint64_t revision = 0u;
  std::uint64_t structural_revision = 0u;
  std::uint64_t last_contact_credit_revision = 0u;
  std::uint64_t recruited_edges = 0u;
  std::uint64_t pruned_edges = 0u;
  std::uint64_t repaired_cells = 0u;
  std::uint64_t free_structural_matter_q8 = 0u;
  std::uint64_t edge_structural_matter_q8 = 0u;
  std::uint64_t structural_debit_q8 = 0u;
  std::uint64_t structural_refund_q8 = 0u;
  std::uint64_t local_proposal_activity = 0u;
};

__device__ __forceinline__ std::int32_t abs_i32(std::int32_t value) {
  return value < 0 ? static_cast<std::int32_t>(-
                         static_cast<std::int64_t>(value))
                   : value;
}

__device__ __forceinline__ std::int32_t clamp_signal(std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kSignalLimit
          ? -kSignalLimit
          : value > kSignalLimit ? kSignalLimit : value);
}

__device__ __forceinline__ std::int16_t clamp_weight(std::int64_t value) {
  return static_cast<std::int16_t>(
      value < -kWeightLimit
          ? -kWeightLimit
          : value > kWeightLimit ? kWeightLimit : value);
}

__device__ __forceinline__ std::uint32_t clamp_q8(std::int64_t value) {
  return static_cast<std::uint32_t>(value < 0 ? 0 : value > 256 ? 256
                                                                    : value);
}

__device__ __forceinline__ std::uint16_t clamp_u16(std::int64_t value) {
  return static_cast<std::uint16_t>(
      value < 0 ? 0 : value > 65535 ? 65535 : value);
}

__device__ inline void refresh_structural_matter(State* adaptive,
                                                 std::uint32_t cell_count) {
  std::uint64_t free_matter = 0u;
  std::uint64_t edge_matter = 0u;
  for (std::uint32_t source = 0u; source < cell_count; ++source) {
    const Cell& cell = adaptive->cells[source];
    free_matter += cell.free_structural_matter_q8;
    for (std::uint32_t slot = 0u; slot < kEdgeSlots; ++slot) {
      const PlasticEdge& edge = cell.edges[slot];
      if (edge.live != 0u) edge_matter += edge.structural_matter_q8;
    }
  }
  adaptive->free_structural_matter_q8 = free_matter;
  adaptive->edge_structural_matter_q8 = edge_matter;
}

__device__ __forceinline__ void clear_edge(PlasticEdge* edge) {
  *edge = {};
  edge->target = kNoCell;
}

__device__ __forceinline__ std::int32_t contact_signal(
    const std::uint32_t* contact, std::uint32_t contact_count,
    std::uint64_t founder, std::uint32_t index) {
  if (contact_count == 0u) return 0;
  const std::uint32_t word = contact[index % contact_count];
  const std::uint32_t shift = (index & 15u) + 1u;
  const std::uint32_t rotated = (word << shift) | (word >> (32u - shift));
  const std::uint32_t founder_byte = static_cast<std::uint32_t>(
      (founder >> ((index & 7u) * 8u)) & 0xffu);
  const std::uint32_t mixed = rotated ^ founder_byte ^
                              (0x9e3779b9u * (index + 1u));
  const std::int32_t odd = static_cast<std::int32_t>(
      __popc(mixed & 0x55555555u));
  const std::int32_t even = static_cast<std::int32_t>(
      __popc(mixed & 0xaaaaaaaau));
  const std::uint32_t nibble_shift = (index & 3u) * 4u;
  return (odd - even) * 32 +
         static_cast<std::int32_t>((mixed >> nibble_shift) & 15u) - 7;
}

__device__ __forceinline__ std::int32_t signed_step(std::int32_t target,
                                                     std::int32_t current,
                                                     std::int32_t divisor) {
  if (target == current) return 0;
  std::int32_t delta = (target - current) / divisor;
  if (delta == 0) delta = target > current ? 1 : -1;
  return delta;
}

__device__ inline void record_contact_credit(
    State* adaptive, const std::uint32_t* contact,
    std::uint32_t contact_count, std::uint64_t founder,
    std::uint32_t cell_count, std::uint64_t contact_revision) {
  if (cell_count == 0u || contact_count == 0u ||
      contact_revision == adaptive->last_contact_credit_revision)
    return;
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    const std::uint32_t magnitude = static_cast<std::uint32_t>(abs_i32(
        contact_signal(contact, contact_count, founder, index)));
    const std::uint32_t event_credit =
        magnitude > 512u ? 4096u : magnitude * 8u;
    Cell& cell = adaptive->cells[index];
    cell.contact_credit_q12 = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(cell.contact_credit_q12) * 31u +
         event_credit) /
        32u);
  }
  adaptive->last_contact_credit_revision = contact_revision;
}

__device__ inline void initialize(State* state, std::uint64_t founder,
                                  std::uint32_t cell_count) {
  *state = {};
  if (cell_count == 0u) return;
  const std::uint32_t anchor_span =
      cell_count > kContextWidth ? cell_count - kContextWidth : 1u;
  state->context_anchor = static_cast<std::uint32_t>(founder % anchor_span);
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    Cell& cell = state->cells[index];
    const std::uint32_t lane = static_cast<std::uint32_t>(
        (founder >> ((index & 7u) * 8u)) & 0xffu);
    cell.activity_target_q8 = 48 + static_cast<std::int32_t>(lane & 63u);
    cell.plasticity_q8 = static_cast<std::uint16_t>(96u + (lane & 63u));
    cell.free_structural_matter_q8 = kInitialStructuralMatterQ8;
    for (std::uint32_t edge = 0u; edge < kEdgeSlots; ++edge)
      cell.edges[edge].target = kNoCell;
  }
  refresh_structural_matter(state, cell_count);
}

__device__ __forceinline__ void begin_epoch(State* state) {
  state->epoch_signed_residual = 0;
  state->epoch_residual_l1 = 0u;
  state->epoch_observations = 0u;
}

__device__ inline void observe_prediction(State* state, std::uint32_t index,
                                          std::uint32_t route,
                                          std::int32_t observed,
                                          std::int32_t predicted,
                                          std::int32_t resource) {
  if (index >= kMaxCells || route >= kRouteCount) return;
  const std::int32_t residual = observed - predicted;
  const std::uint32_t magnitude = static_cast<std::uint32_t>(
      abs_i32(residual) > 65535 ? 65535 : abs_i32(residual));
  Cell& cell = state->cells[index];
  const std::uint32_t previous = cell.route_residual_ema[route];
  const std::uint32_t next = (previous * 7u + magnitude) / 8u;
  cell.route_residual_ema[route] = static_cast<std::uint16_t>(next);
  if (previous != 0u && magnitude <= previous + 8u) {
    const std::uint32_t utility = cell.route_utility_q8[route] + 4u;
    cell.route_utility_q8[route] = static_cast<std::uint16_t>(
        utility > 256u ? 256u : utility);
  } else if (cell.route_utility_q8[route] != 0u) {
    --cell.route_utility_q8[route];
  }
  const std::int64_t local_plasticity =
      48 + static_cast<std::int64_t>(next) / 8 + resource / 8 -
      cell.damage_q8 / 2;
  cell.plasticity_q8 = static_cast<std::uint16_t>(clamp_q8(local_plasticity));
  state->epoch_signed_residual += residual;
  state->epoch_residual_l1 += magnitude;
  ++state->epoch_observations;
}

__device__ __forceinline__ std::int32_t consolidated_bias(
    const State* state, std::uint32_t index, std::uint32_t route) {
  if (index >= kMaxCells || route >= kRouteCount) return 0;
  return state->cells[index].consolidated_weight[route];
}

__device__ __forceinline__ std::uint32_t eligibility_lifetime(
    const State* state) {
  return 4u + state->plasticity_q8 / 64u + state->replay_q8 / 64u;
}

__device__ inline void finish_epoch(
    State* adaptive, const std::int32_t* activity, const std::int32_t* trace,
    const std::int32_t* resource, const std::int32_t* reafference,
    const std::uint32_t* tissue_matter_q8, std::uint32_t cell_count,
    std::int32_t field_response_l1, std::int32_t field_residual_l1,
    std::int32_t field_packet_braid_sum, std::uint32_t field_damage_q8,
    std::uint32_t contact_count, std::uint64_t contact_revision) {
  if (cell_count == 0u) return;
  std::int64_t activity_sum = 0;
  std::int64_t trace_sum = 0;
  std::int64_t slow_sum = 0;
  std::int64_t resource_sum = 0;
  std::int64_t reafference_sum = 0;
  std::uint64_t utility_sum = 0u;
  std::uint32_t damaged = field_damage_q8;
  std::uint32_t live_edges = 0u;
  std::uint32_t best_anchor = adaptive->context_anchor % cell_count;
  std::uint64_t best_score = 0u;
  std::uint64_t current_anchor_score = 0u;

  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    Cell& cell = adaptive->cells[index];
    cell.slow_trace_q12 = clamp_signal(
        (static_cast<std::int64_t>(cell.slow_trace_q12) * 63 + trace[index]) /
        64);
    const std::int32_t magnitude = abs_i32(activity[index]);
    cell.activity_target_q8 = clamp_signal(
        (static_cast<std::int64_t>(cell.activity_target_q8) * 31 +
         magnitude) /
        32);
    if (contact_count == 0u) {
      if (cell.quiet_age != 0xffffu) ++cell.quiet_age;
    } else {
      cell.quiet_age = 0u;
    }
    activity_sum += activity[index];
    trace_sum += trace[index];
    slow_sum += cell.slow_trace_q12;
    resource_sum += resource[index];
    reafference_sum += reafference[index];
    damaged += cell.damage_q8;
    const std::uint64_t score = cell.contact_credit_q12;
    for (std::uint32_t route = 0u; route < kRouteCount; ++route)
      utility_sum += cell.route_utility_q8[route];
    for (std::uint32_t edge = 0u; edge < kEdgeSlots; ++edge) {
      if (cell.edges[edge].live == 0u) continue;
      ++live_edges;
    }
    if (index == adaptive->context_anchor % cell_count)
      current_anchor_score = score;
    const bool complete_context_locus =
        cell_count <= kContextWidth || index + kContextWidth < cell_count;
    if (complete_context_locus && score > best_score) {
      best_score = score;
      best_anchor = index;
    }
  }

  const std::int64_t divisor = static_cast<std::int64_t>(cell_count);
  const std::int64_t residual_mean =
      adaptive->epoch_observations == 0u
          ? 0
          : static_cast<std::int64_t>(adaptive->epoch_residual_l1 /
                                      adaptive->epoch_observations);
  const std::int64_t sample[kContextWidth]{
      activity_sum * 8 / divisor,
      (trace_sum - slow_sum) * 8 / divisor,
      residual_mean * 8,
      (resource_sum / divisor - 512) * 4,
      (static_cast<std::int64_t>(field_response_l1) +
       field_residual_l1) /
          divisor,
      static_cast<std::int64_t>(field_packet_braid_sum) / 2,
      reafference_sum * 8 / divisor,
      static_cast<std::int64_t>(damaged) * 8 / divisor};

  adaptive->context_l1 = 0u;
  for (std::uint32_t component = 0u; component < kContextWidth; ++component) {
    adaptive->fast_context_q12[component] = clamp_signal(
        (static_cast<std::int64_t>(adaptive->fast_context_q12[component]) *
             7 +
         sample[component]) /
        8);
    adaptive->slow_context_q12[component] = clamp_signal(
        (static_cast<std::int64_t>(adaptive->slow_context_q12[component]) *
             63 +
         adaptive->fast_context_q12[component]) /
        64);
    adaptive->context_q12[component] = clamp_signal(
        (static_cast<std::int64_t>(adaptive->fast_context_q12[component]) +
         adaptive->slow_context_q12[component]) /
        2);
    adaptive->context_l1 += static_cast<std::uint32_t>(
        abs_i32(adaptive->context_q12[component]));
  }

  const std::uint32_t occupancy_q8 = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(live_edges) * 256u /
      (static_cast<std::uint64_t>(cell_count) * kEdgeSlots));
  const std::uint32_t mean_utility_q8 = static_cast<std::uint32_t>(
      utility_sum /
      (static_cast<std::uint64_t>(cell_count) * kRouteCount + 1u));
  const std::uint32_t novelty_q8 = clamp_q8(
      static_cast<std::int64_t>(abs_i32(adaptive->context_q12[1])) / 16 +
      abs_i32(adaptive->context_q12[2]) / 16);
  const std::uint32_t resource_margin_q8 = clamp_q8(
      128 + static_cast<std::int64_t>(adaptive->context_q12[3]) / 16);
  const std::uint32_t damage_pressure_q8 = clamp_q8(
      static_cast<std::int64_t>(damaged) * 256 /
      (static_cast<std::int64_t>(cell_count) * 256 + 1));

  adaptive->plasticity_q8 = clamp_q8(
      48 + novelty_q8 +
      static_cast<std::int64_t>(abs_i32(adaptive->context_q12[6])) / 32 -
      damage_pressure_q8 / 2);
  adaptive->consolidation_q8 = clamp_q8(
      (contact_count == 0u ? 96 : 24) + mean_utility_q8 +
      (256u - adaptive->plasticity_q8) / 4u);
  adaptive->growth_q8 = clamp_q8(
      novelty_q8 + resource_margin_q8 / 2u +
      static_cast<std::int64_t>(abs_i32(adaptive->context_q12[4])) / 32 -
      occupancy_q8 / 2u - damage_pressure_q8 / 2u);
  adaptive->turnover_q8 = clamp_q8(
      occupancy_q8 + (contact_count == 0u ? 48u : 0u) +
      (256u - mean_utility_q8) / 3u);
  adaptive->repair_q8 = clamp_q8(
      damage_pressure_q8 + resource_margin_q8 / 2u +
      static_cast<std::int64_t>(field_damage_q8) / (cell_count + 1u));
  adaptive->replay_q8 =
      contact_count == 0u
          ? clamp_q8(adaptive->consolidation_q8 / 2u +
                     static_cast<std::int64_t>(
                         abs_i32(adaptive->context_q12[6])) /
                         16)
          : 0u;
  adaptive->live_edges = live_edges;
  // Context authority is a slow specialization, not an instantaneous argmax.
  // Relocation requires a new raw event, an intact incumbent, and a
  // substantially stronger resident competitor. Damage therefore causes a
  // real deficit that local repair/rewiring must resolve instead of an
  // observer-like instant remap.
  const std::uint32_t incumbent = adaptive->context_anchor % cell_count;
  const bool incumbent_intact = tissue_matter_q8[incumbent] != 0u;
  const bool new_contact_revision =
      contact_revision != adaptive->last_anchor_contact_revision;
  // Re-anchoring is a resident response to a new raw event. Idle autonomous
  // epochs still update ecology, but cannot create scheduler-dependent
  // relocation opportunities for identical contact histories.
  if (new_contact_revision && incumbent_intact &&
      best_score > current_anchor_score + current_anchor_score / 2u + 64u &&
      best_score != 0u)
    adaptive->context_anchor = best_anchor;
  if (new_contact_revision)
    adaptive->last_anchor_contact_revision = contact_revision;
  adaptive->damage_q8 = damaged;
  ++adaptive->revision;
}

__device__ inline void adapt_routes(State* adaptive, std::int32_t* prediction,
                                    std::int32_t* credit_weight,
                                    const std::uint32_t* eligibility_age,
                                    std::uint32_t cell_count,
                                    bool quiet_contact) {
  adaptive->consolidated_routes = 0u;
  const std::uint32_t lifetime = eligibility_lifetime(adaptive);
  for (std::uint32_t route = 0u; route < kRouteCount; ++route) {
    for (std::uint32_t index = 0u; index < cell_count; ++index) {
      const std::size_t offset =
          static_cast<std::size_t>(route) * kMaxCells + index;
      Cell& cell = adaptive->cells[index];
      std::int32_t consolidated = cell.consolidated_weight[route];
      std::int32_t fast = credit_weight[offset];
      const std::uint32_t utility = cell.route_utility_q8[route];
      if (utility >= 16u && adaptive->consolidation_q8 >= 48u) {
        consolidated += signed_step(fast, consolidated, 32);
      }
      if (quiet_contact) {
        fast += signed_step(consolidated, fast, 32);
        if ((adaptive->revision & 7u) == 0u &&
            cell.route_utility_q8[route] != 0u)
          --cell.route_utility_q8[route];
      }
      if (adaptive->turnover_q8 > 176u && utility < 4u &&
          eligibility_age[offset] > lifetime) {
        consolidated += signed_step(0, consolidated, 16);
        fast += signed_step(0, fast, 16);
        prediction[offset] += signed_step(0, prediction[offset], 16);
      }
      cell.consolidated_weight[route] = clamp_weight(consolidated);
      credit_weight[offset] = clamp_signal(fast);
      if (cell.consolidated_weight[route] != 0) ++adaptive->consolidated_routes;
    }
  }
}

__device__ inline std::uint32_t best_local_target(
    State* adaptive, const std::int32_t* activity,
    const std::int32_t* resource, const std::uint32_t* tissue_matter_q8,
    std::uint32_t source, std::uint32_t cell_count) {
  ++adaptive->local_proposal_activity;
  std::uint32_t best = cell_count;
  std::uint64_t best_score = 0u;
  const std::uint32_t begin =
      source > kLocalProposalRadius ? source - kLocalProposalRadius : 0u;
  const std::uint32_t unbounded_end = source + kLocalProposalRadius + 1u;
  const std::uint32_t end =
      unbounded_end < cell_count ? unbounded_end : cell_count;
  for (std::uint32_t candidate = begin; candidate < end; ++candidate) {
    if (candidate == source || tissue_matter_q8[candidate] == 0u) continue;
    const Cell& candidate_cell = adaptive->cells[candidate];
    std::uint64_t history_score = candidate_cell.contact_credit_q12;
    for (std::uint32_t route = 0u; route < kRouteCount; ++route)
      history_score += candidate_cell.route_residual_ema[route];
    const std::uint64_t resource_score = static_cast<std::uint64_t>(
        resource[candidate] > 0 ? resource[candidate] : 0);
    // The aperture is authored geometry; the winner is resident state.  Raw
    // activity, metabolic resource, surviving tissue, and accumulated local
    // history all contribute.  No population-wide argmax or fixed neighbor
    // can select the target.
    const std::uint64_t score =
        static_cast<std::uint64_t>(abs_i32(activity[candidate])) * 8u +
        static_cast<std::uint64_t>(
            abs_i32(candidate_cell.slow_trace_q12)) * 2u +
        resource_score +
        static_cast<std::uint64_t>(tissue_matter_q8[candidate]) * 4u +
        history_score + 1u;
    if (score > best_score) {
      best_score = score;
      best = candidate;
    }
  }
  return best;
}

__device__ inline void adapt_connectivity(
    State* adaptive, const std::int32_t* activity, const std::int32_t* trace,
    std::int32_t* resource, const std::uint32_t* tissue_matter_q8,
    const std::uint32_t* tissue_coupling_q8, std::uint32_t cell_count,
    bool new_contact) {
  std::uint32_t live_edges = 0u;
  const std::int32_t mean_residual =
      adaptive->epoch_observations == 0u
          ? 0
          : static_cast<std::int32_t>(adaptive->epoch_signed_residual /
                                      adaptive->epoch_observations);
  for (std::uint32_t source = 0u; source < cell_count; ++source) {
    Cell& cell = adaptive->cells[source];
    for (std::uint32_t slot = 0u; slot < kEdgeSlots; ++slot) {
      PlasticEdge& edge = cell.edges[slot];
      if (edge.live == 0u) continue;
      if (edge.structural_matter_q8 == 0u) {
        clear_edge(&edge);
        ++adaptive->structural_revision;
        continue;
      }
      if (edge.target >= cell_count ||
          tissue_matter_q8[edge.target] == 0u) {
        if (adaptive->repair_q8 > 96u) {
          const std::uint32_t replacement = best_local_target(
              adaptive, activity, resource, tissue_matter_q8, source,
              cell_count);
          if (replacement < cell_count) {
            edge.target = static_cast<std::uint16_t>(replacement);
            // A recruited target is new physical connectivity. Mature credit
            // from the destroyed target cannot be transferred onto it; lived
            // coactivity and residuals must earn its authority again.
            edge.weight_q8 = 0;
            edge.eligibility_q8 = 0;
            edge.utility_q8 = 8u;
            edge.age = 0u;
            ++adaptive->structural_revision;
          }
        }
        if (edge.target >= cell_count ||
            tissue_matter_q8[edge.target] == 0u)
          continue;
      }
      const std::int64_t coactivity =
          static_cast<std::int64_t>(trace[source]) * trace[edge.target] / 4096;
      edge.eligibility_q8 = clamp_weight(
          (static_cast<std::int64_t>(edge.eligibility_q8) * 7 + coactivity) /
          8);
      if (new_contact && mean_residual != 0) {
        const std::int64_t delta =
            static_cast<std::int64_t>(edge.eligibility_q8) * mean_residual *
            adaptive->plasticity_q8 / (1ll << 20);
        if (delta != 0) {
          edge.weight_q8 = clamp_weight(
              static_cast<std::int64_t>(edge.weight_q8) + delta);
          edge.utility_q8 = clamp_u16(
              static_cast<std::int64_t>(edge.utility_q8) + abs_i32(
                  static_cast<std::int32_t>(delta)) + 1);
        }
      }
      if (edge.age != 0xffffu) ++edge.age;
      if ((adaptive->revision & 15u) == 0u && edge.utility_q8 != 0u)
        --edge.utility_q8;
      if (!new_contact && edge.utility_q8 < 16u &&
          (adaptive->revision & 7u) == 0u) {
        edge.weight_q8 = clamp_weight(
            static_cast<std::int64_t>(edge.weight_q8) +
            signed_step(0, edge.weight_q8, 16));
      }
      if (adaptive->turnover_q8 > 128u && edge.age > 96u &&
          edge.utility_q8 < 8u) {
        const std::uint16_t refund = edge.structural_matter_q8;
        cell.free_structural_matter_q8 = static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(cell.free_structural_matter_q8) +
            refund);
        adaptive->structural_refund_q8 += refund;
        clear_edge(&edge);
        ++adaptive->pruned_edges;
        ++adaptive->structural_revision;
        continue;
      }
      const std::uint32_t source_gain = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(tissue_matter_q8[source]) *
          tissue_coupling_q8[source] / 256u);
      if (source_gain != 0u) ++live_edges;
    }
  }

  if (new_contact && adaptive->growth_q8 > 96u &&
      adaptive->epoch_residual_l1 != 0u) {
    std::uint32_t source = cell_count;
    std::uint32_t slot = kEdgeSlots;
    std::uint64_t best_score = 0u;
    for (std::uint32_t candidate = 0u; candidate < cell_count; ++candidate) {
      if (resource[candidate] < 128 || tissue_matter_q8[candidate] == 0u ||
          adaptive->cells[candidate].free_structural_matter_q8 <
              kEdgeStructuralDoseQ8)
        continue;
      std::uint32_t empty = kEdgeSlots;
      for (std::uint32_t edge = 0u; edge < kEdgeSlots; ++edge)
        if (adaptive->cells[candidate].edges[edge].live == 0u) {
          empty = edge;
          break;
        }
      if (empty == kEdgeSlots) continue;
      std::uint64_t score = adaptive->cells[candidate].plasticity_q8;
      for (std::uint32_t route = 0u; route < kRouteCount; ++route)
        score += adaptive->cells[candidate].route_residual_ema[route];
      if (score > best_score) {
        best_score = score;
        source = candidate;
        slot = empty;
      }
    }
    if (source < cell_count && slot < kEdgeSlots) {
      const std::uint32_t target = best_local_target(
          adaptive, activity, resource, tissue_matter_q8, source, cell_count);
      if (target < cell_count) {
        Cell& source_cell = adaptive->cells[source];
        PlasticEdge& edge = source_cell.edges[slot];
        edge.target = static_cast<std::uint16_t>(target);
        edge.weight_q8 =
            trace[source] == 0 || trace[target] == 0 ||
                    ((trace[source] < 0) == (trace[target] < 0))
                ? 16
                : -16;
        edge.utility_q8 = 8u;
        edge.age = 0u;
        edge.structural_matter_q8 = kEdgeStructuralDoseQ8;
        source_cell.free_structural_matter_q8 =
            static_cast<std::uint16_t>(
                source_cell.free_structural_matter_q8 -
                kEdgeStructuralDoseQ8);
        adaptive->structural_debit_q8 += kEdgeStructuralDoseQ8;
        resource[source] = resource[source] > 16 ? resource[source] - 16 : 0;
        edge.live = 1u;
        ++adaptive->recruited_edges;
        ++adaptive->structural_revision;
        ++live_edges;
      }
    }
  }
  adaptive->live_edges = live_edges;
  refresh_structural_matter(adaptive, cell_count);
}

__device__ __forceinline__ std::int32_t edge_drive(
    const State* adaptive, const std::int32_t* activity,
    const std::uint32_t* tissue_matter_q8,
    const std::uint32_t* tissue_coupling_q8, std::uint32_t index,
    std::uint32_t cell_count) {
  if (index >= cell_count) return 0;
  std::int64_t drive = 0;
  for (std::uint32_t slot = 0u; slot < kEdgeSlots; ++slot) {
    const PlasticEdge& edge = adaptive->cells[index].edges[slot];
    if (edge.live == 0u || edge.structural_matter_q8 == 0u ||
        edge.target >= cell_count)
      continue;
    const std::uint32_t source_gain = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(tissue_matter_q8[index]) *
        tissue_coupling_q8[index] / 256u);
    const std::uint32_t target_gain = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(tissue_matter_q8[edge.target]) *
        tissue_coupling_q8[edge.target] / 256u);
    drive += static_cast<std::int64_t>(edge.weight_q8) *
             activity[edge.target] * source_gain * target_gain *
             edge.structural_matter_q8 /
             (256ll * 256ll * 256ll * kEdgeStructuralDoseQ8);
  }
  return clamp_signal(drive);
}

__device__ inline void apply_damage(State* adaptive, std::uint32_t center,
                                    std::uint32_t radius,
                                    std::uint32_t removed_q8,
                                    std::uint32_t repair_q8,
                                    std::uint32_t cell_count) {
  if (cell_count == 0u || removed_q8 == 0u) return;
  const std::uint32_t survival_q8 =
      removed_q8 < 256u ? 256u - removed_q8 : 0u;
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    const std::uint32_t distance = index > center ? index - center
                                                  : center - index;
    if (distance > radius) continue;
    Cell& cell = adaptive->cells[index];
    if (removed_q8 > cell.damage_q8)
      cell.damage_q8 = static_cast<std::uint16_t>(removed_q8);
    cell.repair_reserve_q8 = clamp_u16(
        static_cast<std::uint64_t>(cell.repair_reserve_q8) + repair_q8);
    cell.free_structural_matter_q8 = static_cast<std::uint16_t>(
        static_cast<std::uint32_t>(cell.free_structural_matter_q8) *
        survival_q8 / 256u);
    cell.slow_trace_q12 = clamp_signal(
        static_cast<std::int64_t>(cell.slow_trace_q12) *
        survival_q8 / 256u);
    for (std::uint32_t route = 0u; route < kRouteCount; ++route) {
      cell.consolidated_weight[route] = clamp_weight(
          static_cast<std::int64_t>(cell.consolidated_weight[route]) *
          survival_q8 / 256u);
      cell.route_utility_q8[route] = static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(cell.route_utility_q8[route]) *
          survival_q8 / 256u);
    }
    for (std::uint32_t slot = 0u; slot < kEdgeSlots; ++slot) {
      PlasticEdge& edge = cell.edges[slot];
      edge.weight_q8 = clamp_weight(
          static_cast<std::int64_t>(edge.weight_q8) * survival_q8 /
          256u);
      edge.utility_q8 = static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(edge.utility_q8) *
          survival_q8 / 256u);
      edge.structural_matter_q8 = static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(edge.structural_matter_q8) *
          survival_q8 / 256u);
      if (edge.live != 0u && edge.structural_matter_q8 == 0u) {
        clear_edge(&edge);
        ++adaptive->structural_revision;
      }
    }
  }
  refresh_structural_matter(adaptive, cell_count);
}

__device__ inline void repair_tissue(State* adaptive,
                                     std::uint32_t* tissue_matter_q8,
                                     std::int32_t* resource,
                                     std::uint32_t cell_count) {
  if (adaptive->repair_q8 < 64u) return;
  const std::uint32_t base_pace = 1u + adaptive->repair_q8 / 128u;
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    Cell& cell = adaptive->cells[index];
    if (cell.damage_q8 == 0u || tissue_matter_q8[index] >= 256u ||
        resource[index] < 64)
      continue;
    std::uint32_t repaired = base_pace + cell.repair_reserve_q8 / 64u;
    if (repaired > cell.damage_q8) repaired = cell.damage_q8;
    if (repaired > 256u - tissue_matter_q8[index])
      repaired = 256u - tissue_matter_q8[index];
    const std::uint32_t affordable =
        static_cast<std::uint32_t>(resource[index] / 8);
    if (repaired > affordable) repaired = affordable;
    if (repaired == 0u) continue;
    tissue_matter_q8[index] += repaired;
    cell.damage_q8 = static_cast<std::uint16_t>(cell.damage_q8 - repaired);
    resource[index] -= static_cast<std::int32_t>(repaired * 8u);
    const std::uint32_t reserve_spent =
        repaired * 8u < cell.repair_reserve_q8
            ? repaired * 8u
            : cell.repair_reserve_q8;
    cell.repair_reserve_q8 = static_cast<std::uint16_t>(
        cell.repair_reserve_q8 - reserve_spent);
    if (cell.damage_q8 == 0u) {
      ++adaptive->repaired_cells;
      ++adaptive->structural_revision;
    }
  }
}


__device__ inline std::uint32_t effective_context_l1(
    const State* adaptive, const std::uint32_t* tissue_matter_q8,
    const std::uint32_t* tissue_coupling_q8, std::uint32_t cell_count) {
  if (cell_count == 0u) return 0u;
  std::uint32_t total = 0u;
  for (std::uint32_t component = 0u; component < kContextWidth; ++component) {
    const std::uint32_t unbounded = adaptive->context_anchor + component;
    const std::uint32_t locus =
        unbounded < cell_count ? unbounded : cell_count - 1u;
    const std::uint32_t gain_q8 = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(tissue_matter_q8[locus]) *
        tissue_coupling_q8[locus] / 256u);
    total += static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(abs_i32(adaptive->context_q12[component])) *
        gain_q8 / 256u);
  }
  return total;
}

__device__ inline std::uint32_t context_matter_q8(
    const State* adaptive, const std::uint32_t* tissue_matter_q8,
    const std::uint32_t* tissue_coupling_q8, std::uint32_t cell_count) {
  if (cell_count == 0u) return 0u;
  std::uint32_t total = 0u;
  for (std::uint32_t component = 0u; component < kContextWidth; ++component) {
    const std::uint32_t unbounded = adaptive->context_anchor + component;
    const std::uint32_t locus =
        unbounded < cell_count ? unbounded : cell_count - 1u;
    total += static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(tissue_matter_q8[locus]) *
        tissue_coupling_q8[locus] / 256u);
  }
  return total / kContextWidth;
}

__device__ inline void write_language_context(
    const State* adaptive, float* destination,
    const std::uint32_t* tissue_matter_q8,
    const std::uint32_t* tissue_coupling_q8, std::uint32_t cell_count) {
  if (destination == nullptr || cell_count == 0u) return;
  for (std::uint32_t component = 0u; component < kContextWidth; ++component) {
    const std::uint32_t unbounded = adaptive->context_anchor + component;
    const std::uint32_t locus =
        unbounded < cell_count ? unbounded : cell_count - 1u;
    const std::uint32_t gain_q8 = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(tissue_matter_q8[locus]) *
        tissue_coupling_q8[locus] / 256u);
    float value = static_cast<float>(adaptive->context_q12[component]) /
                  static_cast<float>(kSignalLimit);
    value *= static_cast<float>(gain_q8) / 256.0f;
    if (value < -1.0f) value = -1.0f;
    if (value > 1.0f) value = 1.0f;
    destination[component] = value;
  }
}

}  // namespace substrate::bcc32::adaptive_ecology
