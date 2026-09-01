#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_OPS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_OPS_CUH
// Thin always-on surface: types already in core.cuh, plus hot leaf primitives.
// Mechanism headers compile only in the device_ops owner (or a named caller).
#include "direct_adult_core.cuh"
#include "direct_adult_eligibility_claim_identity.cuh"
#include <cuda_runtime.h>
#include "direct_network_tube_chemistry.cuh"
#include "direct_adult_bounded_fanout.cuh"
#if defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
#define DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER __device__
#else
#define DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER __device__ inline
#endif
#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
#include "direct_adult_runtime_frontiers.cuh"
#include "direct_adult_delayed_recontact.cuh"
#include "direct_adult_recipe_credit.cuh"
#include "direct_adult_inverse_transformation.cuh"
#include "direct_adult_motor_expression.cuh"
#include "direct_retention_policy.cuh"
#include "direct_adult_local_coordinate_atlas.cuh"
#endif
namespace substrate::direct_adult {
struct DirectRetentionState;
}
namespace substrate::direct_network {
__device__ bool materialize_raw_contact_sparse_credit(
    DirectBrain brain, const DirectExactHistoryRecord& witness,
    const ResidentRawContactKey& contact,
    ResidentDevelopmentCounters* counters);
}
namespace substrate::direct_adult_core {
#include "direct_adult_contact_epoch_identity.cuh"
#ifndef HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
#define HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
#ifndef HARDWARE_NATIVE_BOUNDED_ROUTE_SCAN_COUNT_DEFINED
#define HARDWARE_NATIVE_BOUNDED_ROUTE_SCAN_COUNT_DEFINED
DIRECT_ADULT_HD inline std::uint32_t bounded_route_scan_count(
    const DirectNode& node) {
  return node.route_capacity < kDirectAdultFanoutCeiling
             ? node.route_capacity
             : kDirectAdultFanoutCeiling;
}
#endif
__device__ bool preserve_displaced_action(
    direct_network::ResidentDevelopmentState* development,
    const AsynchronousTicket& ticket, const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity);
DIRECT_ADULT_HD inline std::int32_t abs_q16(std::int32_t v) {
  return v < 0 ? -v : v; }
DIRECT_ADULT_HD inline std::int32_t max_q16(std::int32_t a, std::int32_t b) {
  return a > b ? a : b; }
DIRECT_ADULT_HD inline std::int32_t clamp_q16(std::int32_t value, std::int32_t min_val, std::int32_t max_val) {
  return value < min_val ? min_val : (value > max_val ? max_val : value);
}
DIRECT_ADULT_HD inline std::int32_t mul_q16(std::int32_t a, std::int32_t b) {
  return static_cast<std::int32_t>((static_cast<std::int64_t>(a) * b) >> 16); }
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
__device__ inline void decay_resident_transient_trace_q16(
    std::int32_t* trace_q16, std::int32_t decay_q16) {
  if (trace_q16 == nullptr) return;
  const std::int32_t decayed = mul_q16(*trace_q16, decay_q16);
  const std::int32_t magnitude = decayed < 0 ? -decayed : decayed;
  *trace_q16 = magnitude <= 1 ? 0 : decayed;
}
__device__ inline bool maturation_field_active(
    const DirectBrain& brain, const DirectNode& node,
    std::uint32_t field_index, std::uint32_t age) {
  if (field_index == direct_network::kInvalidIndex) return true;
  if (field_index >= brain.resident_field_count || brain.resident_fields == nullptr ||
      brain.development == nullptr) return false;
  const auto& field = brain.resident_fields[field_index];
  const std::uint64_t tick =
      static_cast<std::uint64_t>(brain.development->birth_handoff_tick) + age;
  if (tick < field.begin_tick || (field.end_tick != 0u && tick >= field.end_tick) ||
      (node.chemotype & field.require_mask) != field.require_value ||
      direct_network::resident_field_decayed_strength_q16(field, tick) == 0)
    return false;
  const std::uint64_t distance =
      static_cast<std::uint64_t>(abs_q16(node.coordinate[0] - field.center[0])) +
      abs_q16(node.coordinate[1] - field.center[1]) +
      abs_q16(node.coordinate[2] - field.center[2]);
  return field.radius != 0u && distance <= 3ull * field.radius;
}
__device__ inline std::int32_t maturation_plasticity_gain_q16(
    const DirectBrain& brain, std::uint32_t node_index, std::uint32_t age) {
  if (node_index >= brain.node_count || brain.nodes == nullptr) return kQ16One;
  const DirectNode& node = brain.nodes[node_index];
  constexpr std::int32_t floor = kQ16One / 16;
  const std::int32_t remaining = node.maturation_q16 >= static_cast<std::uint32_t>(kQ16One)
      ? 0 : kQ16One - static_cast<std::int32_t>(node.maturation_q16);
  const std::int32_t ordinary = floor + mul_q16(remaining, kQ16One - floor);
  if (node.territory_index >= brain.recipe_range_count || brain.recipe_ranges == nullptr ||
      brain.recipe_indices == nullptr || brain.recipe_cells == nullptr ||
      brain.resident_rules == nullptr) return ordinary;
  const auto range = brain.recipe_ranges[node.territory_index];
  for (std::uint32_t local = 0u; local < range.index_count; ++local) {
    const std::uint32_t index = range.index_offset + local;
    if (index >= brain.recipe_index_count) break;
    const std::uint32_t cell_index = brain.recipe_indices[index];
    if (cell_index >= brain.recipe_cell_count) continue;
    const auto& cell = brain.recipe_cells[cell_index];
    if (cell.rule_index >= brain.resident_rule_count) continue;
    const auto& rule = brain.resident_rules[cell.rule_index];
    if (rule.opcode == direct_network::RuleOpcode::mature &&
        age >= rule.begin_age && age < rule.end_age &&
        age >= rule.critical_begin_age && age <= rule.critical_end_age &&
        (node.chemotype & rule.require_mask) == rule.require_value &&
        maturation_field_active(brain, node, rule.field_index, age))
      return kQ16One;
  }
  return ordinary;
}
// #1310: shared signed sparse-delivery law for stepped and persistent executors.
DIRECT_ADULT_HD inline std::int32_t competition_output_gain_q16(const DirectNode& source) {
  return 2 * static_cast<std::int32_t>(direct_network::decode_competition_strength_q16(source.competition_strength_code_q16));
}
// #1294: both executors pass the tube's endpoint chemistry gain so one
// contextual chemistry law drives transmission everywhere; the default keeps
// unauthored tubes bitwise neutral.
DIRECT_ADULT_HD inline std::int32_t signed_sparse_route_delivery_q16(
    const DirectNode& source, const DirectRoute& route, std::int32_t competition_gain_q16,
    bool honor_inhibitory_sign = true,
    std::int32_t tube_conductance_gain_q16 =
        direct_network::kTubeChemistryNeutralQ16) {
  std::int32_t magnitude = mul_q16(route.conductance_q16, source.activation_q16);
  magnitude = mul_q16(magnitude, tube_conductance_gain_q16);
  const bool inhibitory = (route.flags & direct_network::kRouteFlagInhibitory) != 0u ||
                          (source.flags & direct_network::kNodeFlagInhibitory) != 0u;
  return honor_inhibitory_sign && inhibitory ? -mul_q16(magnitude, competition_gain_q16) : magnitude;
}
DIRECT_ADULT_HD inline bool motor_competition_score_precedes(
    const DirectNode& left, const DirectBoundaryPort& left_port, std::uint32_t left_index,
    const DirectNode& right, const DirectBoundaryPort& right_port, std::uint32_t right_index) {
  if (left.activation_q16 != right.activation_q16)
    return left.activation_q16 > right.activation_q16;
  if (left.activity_ema_q16 != right.activity_ema_q16)
    return left.activity_ema_q16 > right.activity_ema_q16;
  if (left.credit_ema_q16 != right.credit_ema_q16)
    return left.credit_ema_q16 > right.credit_ema_q16;
  if (left_port.physical_route != right_port.physical_route)
    return left_port.physical_route < right_port.physical_route;
  if (left_port.channel != right_port.channel)
    return left_port.channel < right_port.channel;
  if (left_port.node != right_port.node) return left_port.node < right_port.node;
  return left_index < right_index;
}
DIRECT_ADULT_HD inline bool motor_candidate_wins_local_competition(
    const DirectNode* nodes, const DirectBoundaryPort* ports, std::uint32_t port_count,
    std::uint32_t candidate_index) {
  const DirectBoundaryPort candidate_port = ports[candidate_index];
  const DirectNode candidate = nodes[candidate_port.node];
  if ((candidate.flags & direct_network::kNodeFlagCompetitive) == 0u) return true;
  for (std::uint32_t other_index = 0u; other_index < port_count; ++other_index) {
    if (other_index == candidate_index) continue;
    const DirectBoundaryPort other_port = ports[other_index];
    if ((other_port.role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::motor)) == 0u ||
        other_port.parent_route != candidate_port.parent_route) continue;
    const DirectNode other = nodes[other_port.node];
    if ((other.flags & direct_network::kNodeFlagCompetitive) == 0u ||
        other.territory_index != candidate.territory_index ||
        other.activation_q16 <= (kQ16One / 4)) continue;
    if (motor_competition_score_precedes(
            other, other_port, other_index, candidate, candidate_port, candidate_index))
      return false;
  }
  return true;
}
__device__ __forceinline__ void atomic_clamp_add_q16(std::int32_t* addr, std::int32_t delta,
                                                      std::int32_t min_val, std::int32_t max_val) {
  int* iaddr = reinterpret_cast<int*>(addr);
  int old_val = *iaddr;
  int assumed;
  do {
    assumed = old_val;
    const int new_val = clamp_q16(assumed + delta, min_val, max_val);
    old_val = atomicCAS(iaddr, assumed, new_val);
  } while (assumed != old_val);
}
__device__ __forceinline__ void atomic_ema_update_q16(std::int32_t* addr, std::int32_t delta) {
  int* iaddr = reinterpret_cast<int*>(addr);
  int old_val = *iaddr;
  int assumed;
  do {
    assumed = old_val;
    const int new_val = (assumed * 31 + delta) / 32;
    old_val = atomicCAS(iaddr, assumed, new_val);
  } while (assumed != old_val);
}
DIRECT_ADULT_HD inline std::uint32_t raw_contact_signature(
    std::uint32_t node, std::uint32_t channel, Word word) {
  std::uint32_t signature = 2166136261u;
  signature = (signature ^ node) * 16777619u;
  signature = (signature ^ channel) * 16777619u;
  signature = (signature ^ word) * 16777619u;
  return signature | 1u;
}
__device__ inline bool reserve_bounded_counter(std::uint32_t* counter,
                                               std::uint32_t* prior_out) {
  if (counter == nullptr || prior_out == nullptr) return false;
  std::uint32_t observed = atomicAdd(counter, 0u);
  while (observed != ~0u) {
    const std::uint32_t prior = atomicCAS(counter, observed, observed + 1u);
    if (prior == observed) {
      *prior_out = observed;
      return true;
    }
    observed = prior;
  }
  return false;
}
__device__ inline std::uint32_t mint_claim_incarnation(
    std::uint32_t* counter) {
  std::uint32_t prior = 0u;
  if (reserve_bounded_counter(counter, &prior)) return prior + 1u;
  return 0u;
}
__device__ inline NodeCausalParticipation read_participation_slot(
    const NodeCausalParticipation* slot) {
  NodeCausalParticipation snapshot{};
  auto* generation = const_cast<std::uint32_t*>(&slot->commit_generation);
  for (;;) {
    const std::uint32_t before = atomicAdd(generation, 0u);
    if ((before & 1u) != 0u) continue;
    snapshot.ticket_id = slot->ticket_id;
    snapshot.expiry_tick = slot->expiry_tick;
    snapshot.last_refresh_tick = slot->last_refresh_tick;
    snapshot.authority = slot->authority;
    snapshot.authority_incarnation = slot->authority_incarnation;
    snapshot.claim_incarnation = slot->claim_incarnation;
    snapshot.current_drive = slot->current_drive;
    snapshot.reserved0 = slot->reserved0;
    snapshot.reserved1 = slot->reserved1;
    snapshot.reserved2 = slot->reserved2;
    __threadfence();
    const std::uint32_t after = atomicAdd(generation, 0u);
    if (before == after && (after & 1u) == 0u) {
      snapshot.commit_generation = after;
      return snapshot;
    }
  }
}
DIRECT_ADULT_HD inline constexpr std::uint64_t participation_eligibility_tail(
    const NodeCausalParticipation& value) {
  return (static_cast<std::uint64_t>(value.reserved1) << 32u) |
         value.reserved0;
}
DIRECT_ADULT_HD inline void set_participation_eligibility_tail(
    NodeCausalParticipation* value, std::uint64_t tail) {
  if (value == nullptr) return;
  value->reserved0 = static_cast<std::uint32_t>(tail);
  value->reserved1 = static_cast<std::uint32_t>(tail >> 32u);
}
DIRECT_ADULT_HD inline constexpr bool participation_slot_reusable(
    const NodeCausalParticipation& current, std::uint32_t current_tick) {
  return current.ticket_id == 0ull || current.expiry_tick < current_tick;
}
DIRECT_ADULT_HD inline constexpr bool participation_slot_exact_match(
    const NodeCausalParticipation& current,
    const DirectParticipationDescriptor& descriptor, std::uint64_t tail,
    std::uint32_t current_tick) {
  return !participation_slot_reusable(current, current_tick) &&
         current.ticket_id == descriptor.ticket_id &&
         current.claim_incarnation == descriptor.claim_incarnation &&
         current.authority == descriptor.authority &&
         current.authority_incarnation == descriptor.authority_incarnation &&
         participation_eligibility_tail(current) == tail;
}
DIRECT_ADULT_HD inline std::uint64_t eligibility_record_ref(
    std::uint32_t slot, std::uint32_t generation) {
  return generation == 0u || slot >= kMaxLiveEligibilityRecords
             ? 0u
             : (static_cast<std::uint64_t>(generation) << 32u) |
                   (static_cast<std::uint64_t>(slot) + 1u);
}
__device__ inline std::uint32_t publish_participation_slot(
    NodeCausalParticipation* slot, const NodeCausalParticipation& value) {
  std::uint32_t stable = 0u;
  for (;;) {
    stable = atomicAdd(&slot->commit_generation, 0u);
    if ((stable & 1u) == 0u &&
        atomicCAS(&slot->commit_generation, stable, stable + 1u) == stable)
      break;
  }
  slot->ticket_id = value.ticket_id;
  slot->expiry_tick = value.expiry_tick;
  slot->last_refresh_tick = value.last_refresh_tick;
  slot->authority = value.authority;
  slot->authority_incarnation = value.authority_incarnation;
  slot->claim_incarnation = value.claim_incarnation;
  slot->current_drive = value.current_drive;
  slot->reserved0 = value.reserved0;
  slot->reserved1 = value.reserved1;
  slot->reserved2 = value.reserved2;
  __threadfence();
  const std::uint32_t committed = stable + 2u;
  atomicExch(&slot->commit_generation, committed);
  return committed;
}
__device__ inline bool lock_word(std::uint32_t* lock) {
  if (lock == nullptr) return false;
  for (std::uint32_t attempt = 0u; attempt < 256u; ++attempt) {
    if (atomicCAS(lock, 0u, 1u) == 0u) {
      __threadfence();
      return true;
    }
    __nanosleep(64u);
  }
  return false;
}
__device__ inline bool expire_pending_action_occurrence(
    DirectActionOccurrence* action, std::uint32_t current_tick,
    AdultCoreMetrics* metrics) {
  if (action == nullptr || current_tick <= action->expiry_tick ||
      atomicCAS(&action->state, kActionOccurrencePending,
                kActionOccurrenceExpired) != kActionOccurrencePending)
    return false;
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->action_occurrences_expired), 1ULL);
  return true;
}
__device__ inline bool lock_ticket_slot_for_motor(
    std::uint32_t* ticket_locks,
    const AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity,
    direct_network::ResidentDevelopmentState* development,
    std::uint32_t slot, std::uint32_t current_tick,
    std::uint32_t horizon_ticks,
    AdultCoreMetrics* metrics) {
  if (ticket_locks == nullptr || ticket_table == nullptr ||
      !lock_word(ticket_locks + slot)) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ticket_collision_rejects), 1ULL);
    return false;
  }
  const AsynchronousTicket existing = ticket_table[slot];
  DirectActionOccurrence* action = action_occurrences != nullptr
                                       ? action_occurrences + slot
                                       : nullptr;
  std::uint32_t action_state = action != nullptr
                                   ? atomicAdd(&action->state, 0u)
                                   : kActionOccurrenceFree;
  if (action != nullptr && action_state == kActionOccurrencePending &&
      current_tick > action->expiry_tick) {
    if (!preserve_displaced_action(development, existing, *action, action_links,
            kMaxAsynchronousTickets * participant_capacity) ||
        !expire_pending_action_occurrence(action, current_tick, metrics) ||
        atomicAdd(&action->state, 0u) != kActionOccurrenceExpired) {
      atomicExch(ticket_locks + slot, 0u);
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->action_ticket_collision_rejects), 1ULL);
      return false;
    }
    action_state = kActionOccurrenceExpired;
  }
  const bool action_state_known =
      action_state == kActionOccurrenceFree ||
      action_state == kActionOccurrencePending ||
      action_state == kActionOccurrenceSettled ||
      action_state == kActionOccurrenceExpired;
  const bool action_live = action != nullptr &&
                           action_state == kActionOccurrencePending;
  const bool ticket_is_action = action != nullptr &&
      action_state != kActionOccurrenceFree &&
      action->action_ticket_id == existing.ticket_id;
  const bool ticket_live = !ticket_is_action && existing.ticket_id != 0u &&
      existing.settled == 0u &&
      (current_tick < existing.emission_tick ||
       current_tick - existing.emission_tick <= horizon_ticks);
  if (action_state_known && !action_live && !ticket_live) return true;
  atomicExch(ticket_locks + slot, 0u);
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->action_ticket_collision_rejects), 1ULL);
  return false;
}
__device__ inline void unlock_ticket_slot(std::uint32_t* ticket_locks,
                                          std::uint32_t slot) {
  __threadfence();
  atomicExch(ticket_locks + slot, 0u);
}
__device__ inline bool lock_participation_node(std::uint32_t* locks,
                                               std::uint32_t node) {
  return locks == nullptr || lock_word(locks + node);
}
__device__ inline void unlock_participation_node(std::uint32_t* locks,
                                                 std::uint32_t node) {
  if (locks == nullptr) return;
  __threadfence();
  atomicExch(locks + node, 0u);
}
__device__ inline bool insert_active_participation(
    NodeCausalParticipation* participation,
    std::uint32_t* node_locks,
    std::uint32_t* claim_incarnation_counter,
    std::uint32_t node,
    std::uint64_t ticket,
    std::uint32_t expiry_tick,
    std::uint32_t current_tick,
    std::uint64_t* refused_out = nullptr,
    DirectParticipationAuthority authority = DirectParticipationAuthority::none,
    std::uint32_t authority_incarnation = 0u,
    std::uint64_t* authorized_refreshes_out = nullptr,
    std::uint64_t* authority_collisions_out = nullptr,
    bool* authority_collision_out = nullptr,
    std::uint32_t claim_incarnation = 0u,
    NodeCausalParticipation* committed_out = nullptr,
    std::uint32_t* committed_slot_out = nullptr,
    std::uint64_t ancestry_tail_ref = 0u) {
  if (authority_collision_out != nullptr) *authority_collision_out = false;
  if (committed_out != nullptr) *committed_out = NodeCausalParticipation{};
  if (committed_slot_out != nullptr) *committed_slot_out = kInvalidIndex;
  if (ticket == 0ULL || ticket == kInvalidTicket || expiry_tick < current_tick)
    return false;
  if (!lock_participation_node(node_locks, node)) {
    if (refused_out != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(refused_out), 1ULL);
    return false;
  }
  NodeCausalParticipation* base =
      participation + node * kNodeParticipationAperture;
  for (std::uint32_t s = 0u; s < kNodeParticipationAperture; ++s) {
    NodeCausalParticipation value = read_participation_slot(base + s);
    if (value.ticket_id == 0ull) continue;
    if (value.ticket_id != ticket) continue;
    if (claim_incarnation != 0u && value.claim_incarnation != claim_incarnation) continue;
    if (value.expiry_tick < current_tick) {
      if (refused_out != nullptr) atomicAdd(reinterpret_cast<unsigned long long*>(refused_out), 1ULL);
      unlock_participation_node(node_locks, node); return false;
    }
    if (value.authority != authority || value.authority_incarnation != authority_incarnation) {
      if (authority_collisions_out != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(authority_collisions_out), 1ULL);
      if (authority_collision_out != nullptr) *authority_collision_out = true;
      unlock_participation_node(node_locks, node);
      return false;
    }
    if (participation_eligibility_tail(value) != ancestry_tail_ref) continue;
    value.expiry_tick = expiry_tick;
    value.last_refresh_tick = current_tick;
    value.current_drive = 1u;
    set_participation_eligibility_tail(&value, ancestry_tail_ref);
    value.commit_generation = publish_participation_slot(base + s, value);
    if (committed_out != nullptr) *committed_out = value;
    if (committed_slot_out != nullptr) *committed_slot_out = s;
    if (authorized_refreshes_out != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(authorized_refreshes_out), 1ULL);
    unlock_participation_node(node_locks, node);
    return true;
  }
  std::uint32_t admission_slot = kInvalidIndex;
  std::uint32_t inactive_slot = kInvalidIndex;
  std::uint32_t inactive_refresh_tick = 0xffffffffu;
  for (std::uint32_t s = 0u; s < kNodeParticipationAperture; ++s) {
    const NodeCausalParticipation incumbent = read_participation_slot(base + s);
    if (incumbent.ticket_id == 0ULL || incumbent.expiry_tick < current_tick) {
      admission_slot = s;
      break;
    }
    // A new independent contact is actual boundary evidence competing for a
    // tiny transient participation bank. It may displace the oldest dormant
    // trace, but never a trace carrying current drive. Durable eligibility and
    // exact history remain resident; this only bounds active working matter.
    if (authority == DirectParticipationAuthority::independent_external &&
        incumbent.current_drive == 0u &&
        (inactive_slot == kInvalidIndex ||
         incumbent.last_refresh_tick < inactive_refresh_tick)) {
      inactive_slot = s;
      inactive_refresh_tick = incumbent.last_refresh_tick;
    }
  }
  if (admission_slot == kInvalidIndex) admission_slot = inactive_slot;
  if (admission_slot != kInvalidIndex) {
    NodeCausalParticipation value{};
    value.ticket_id = ticket;
    value.expiry_tick = expiry_tick;
    value.last_refresh_tick = current_tick;
    value.authority = authority;
    value.authority_incarnation = authority_incarnation;
    value.claim_incarnation = claim_incarnation != 0u
                                  ? claim_incarnation
                                  : mint_claim_incarnation(claim_incarnation_counter);
    if (value.claim_incarnation == 0u) {
      unlock_participation_node(node_locks, node);
      return false;
    }
    value.current_drive = 1u;
    set_participation_eligibility_tail(&value, ancestry_tail_ref);
    value.commit_generation =
        publish_participation_slot(base + admission_slot, value);
    if (committed_out != nullptr) *committed_out = value;
    if (committed_slot_out != nullptr) *committed_slot_out = admission_slot;
    unlock_participation_node(node_locks, node);
    return true;
  }

  if (refused_out != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(refused_out), 1ULL);
  unlock_participation_node(node_locks, node);
  return false;
}
struct DirectActionBindingResult {
  std::uint32_t admitted;
  std::uint32_t participant_offset;
  std::uint32_t participant_count;
  std::uint32_t ancestry_incomplete;
  std::uint64_t diagnostic_upstream_ticket;
};

__device__ inline void stage_current_contribution(
    const DirectParticipationDescriptor& descriptor,
    DirectParticipationDescriptor* staging,
    std::uint32_t* staging_count,
    std::uint32_t staging_capacity,
    AdultCoreMetrics* metrics) {
  if (staging_count == nullptr) return;
  const std::uint32_t cursor = atomicAdd(staging_count, 1u);
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->participation_descriptors_staged), 1ULL);
  if (staging != nullptr && cursor < staging_capacity) {
    staging[cursor] = descriptor;
  } else if (metrics != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->participation_staging_overflow), 1ULL);
  }
}

__device__ inline void stage_incomplete_contribution(
    std::uint32_t source_node,
    std::uint32_t target_node,
    std::uint32_t route_index,
    std::uint32_t context_signature,
    std::uint32_t current_tick,
    DirectParticipationDescriptor* staging,
    std::uint32_t* staging_count,
    std::uint32_t staging_capacity,
    AdultCoreMetrics* metrics) {
  DirectParticipationDescriptor descriptor{};
  descriptor.ticket_id = kInvalidTicket;
  descriptor.source_node = source_node;
  descriptor.target_node = target_node;
  descriptor.route_index = route_index;
  descriptor.context_signature = context_signature;
  descriptor.expiry_tick = current_tick;
  descriptor.contribution_kind = DirectContributionKind::ancestry_incomplete;
  stage_current_contribution(
      descriptor, staging, staging_count, staging_capacity, metrics);
}

__device__ inline bool action_claim_less(
    const DirectParticipationDescriptor& left,
    const DirectParticipationDescriptor& right) {
  if (left.ticket_id != right.ticket_id) return left.ticket_id < right.ticket_id;
  if (left.source_node != right.source_node) return left.source_node < right.source_node;
  if (left.target_node != right.target_node) return left.target_node < right.target_node;
  if (left.route_incarnation != right.route_incarnation)
    return left.route_incarnation < right.route_incarnation;
  if (left.context_signature != right.context_signature)
    return left.context_signature < right.context_signature;
  if (left.claim_incarnation != right.claim_incarnation)
    return left.claim_incarnation < right.claim_incarnation;
  if (left.authority != right.authority)
    return static_cast<std::uint32_t>(left.authority) <
           static_cast<std::uint32_t>(right.authority);
  if (left.authority_incarnation != right.authority_incarnation)
    return left.authority_incarnation < right.authority_incarnation;
  if (left.contribution_kind != right.contribution_kind)
    return static_cast<std::uint32_t>(left.contribution_kind) <
           static_cast<std::uint32_t>(right.contribution_kind);
  if (left.route_index != right.route_index)
    return left.route_index < right.route_index;
  if (left.parent_eligibility_ref != right.parent_eligibility_ref)
    return left.parent_eligibility_ref < right.parent_eligibility_ref;
  return left.ancestry_depth < right.ancestry_depth;
}

__device__ inline bool same_action_claim(
    const DirectParticipationDescriptor& left,
    const DirectParticipationDescriptor& right) {
  return left.ticket_id == right.ticket_id &&
         left.source_node == right.source_node &&
         left.target_node == right.target_node &&
         left.route_incarnation == right.route_incarnation &&
         left.context_signature == right.context_signature &&
         left.claim_incarnation == right.claim_incarnation &&
         left.authority == right.authority &&
         left.authority_incarnation == right.authority_incarnation &&
         left.contribution_kind == right.contribution_kind &&
         left.route_index == right.route_index &&
         left.parent_eligibility_ref == right.parent_eligibility_ref &&
         left.ancestry_depth == right.ancestry_depth;
}

__device__ inline bool same_action_participation(
    const NodeCausalParticipation& left,
    const NodeCausalParticipation& right) {
  return left.ticket_id == right.ticket_id &&
         left.expiry_tick == right.expiry_tick &&
         left.last_refresh_tick == right.last_refresh_tick &&
         left.authority == right.authority &&
         left.authority_incarnation == right.authority_incarnation &&
         left.claim_incarnation == right.claim_incarnation &&
         left.current_drive == right.current_drive &&
         left.commit_generation == right.commit_generation &&
         left.reserved0 == right.reserved0 &&
         left.reserved1 == right.reserved1 &&
         left.reserved2 == right.reserved2;
}

__device__ inline void merge_semantic_claim(
    const DirectParticipationDescriptor& value,
    DirectParticipationDescriptor* claims,
    std::uint32_t* count) {
  for (std::uint32_t i = 0u; i < *count; ++i) {
    if (!same_action_claim(value, claims[i])) continue;
    if (value.expiry_tick > claims[i].expiry_tick)
      claims[i].expiry_tick = value.expiry_tick;
    if (value.lineage_expiry_tick > claims[i].lineage_expiry_tick)
      claims[i].lineage_expiry_tick = value.lineage_expiry_tick;
    return;
  }
  if (*count < kMaxProvenanceSlotsPerNode) claims[(*count)++] = value;
}

__device__ bool has_unpublished_resident_actual_contact(
    const ResidentActualFrontier* frontier);
__device__ bool publish_admitted_resident_actual_contacts(
    DirectBrain brain, ResidentActualFrontier* frontier,
    NodeCausalParticipation* participation, std::uint32_t* participation_locks,
    EligibilityRecord* eligibility_table, std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging, std::uint32_t* staging_count,
    std::uint32_t staging_capacity, std::int32_t* incoming_excitation,
    AdultCoreMetrics* metrics, std::uint32_t current_tick);
__device__ std::uint32_t commit_resident_mismatch_credit_receipts(
    DirectBrain brain, ResidentMismatchOmissionFrontier* frontier,
    std::uint32_t current_tick, ResidentActualFrontier* actual_frontier);
__device__ Word resident_bound_action_motor_word(
    DirectBrain brain, const DirectNode& node, const DirectRoute* routes, std::uint32_t route_capacity, std::uint32_t route_scan_count,
    const ResidentActualFrontier* actual_frontier, DirectActionOccurrence* actions, const DirectActionParticipationLink* links,
    const DirectActionBindingResult& binding, std::uint32_t action_slot, std::uint32_t participant_capacity, std::uint64_t ticket,
    std::uint32_t motor_node, std::uint32_t motor_channel, std::uint32_t context_signature, std::uint32_t publication_tick);

#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
#include "direct_adult_delayed_sparse_schedule.cuh"
#include "direct_adult_delayed_sparse_delivery.cuh"
__device__ inline bool publish_resident_actual_occurrence_contribution(
    DirectBrain brain, ResidentActualFrontier& frontier,
    const ResidentContactEpochReceipt& credential, const ActivityEvent& event,
    NodeCausalParticipation* participation, std::uint32_t* participation_locks,
    EligibilityRecord* eligibility_table, std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging, std::uint32_t* staging_count,
    std::uint32_t staging_capacity, std::int32_t* incoming_excitation,
    AdultCoreMetrics* metrics, std::uint32_t current_tick) {
  DirectParticipationDescriptor descriptor{};
  std::int32_t drive_q16 = 0;
  if (staging == nullptr || staging_count == nullptr || incoming_excitation == nullptr ||
      *staging_count >= staging_capacity ||
      !resident_actual_occurrence_contribution(
          brain, frontier, credential.identity, credential.ingress_sequence,
          resident_contact_authority_incarnation(credential, event),
          current_tick, &descriptor, &drive_q16) ||
      !resident_actual_occurrence_participation_capacity(
          participation, descriptor.source_node, descriptor, current_tick) ||
      !resident_actual_occurrence_participation_capacity(
          participation, descriptor.target_node, descriptor, current_tick))
    return false;
  ResidentActualFrontierEntry* occurrence_entry = nullptr;
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity;
       ++slot) {
    auto& candidate = frontier.entries[slot];
    if (candidate.contact_identity != credential.identity ||
        candidate.ingress_sequence != credential.ingress_sequence ||
        candidate.state != ResidentActualFrontierState::live ||
        candidate.occurrence.state != kResidentRecipeOccurrenceLive) continue;
    if (occurrence_entry != nullptr) return false;
    occurrence_entry = &candidate;
  }
  if (occurrence_entry == nullptr) return false;
  const EligibilityAdmission eligibility = admit_ingress_eligibility_claim(
      descriptor, abs_q16(drive_q16), eligibility_table,
      live_eligibility_count, eligibility_claim_directory,
      eligibility_record_generations, eligibility_free_slots,
      eligibility_free_state, current_tick, metrics);
  if (eligibility.slot == kInvalidIndex) return false;
  descriptor.eligibility_slot = eligibility.slot;
  descriptor.eligibility_generation = eligibility.generation;
  descriptor.frozen_eligibility_q16 = eligibility.canonical_eligibility_q16;
  const std::uint64_t child_ref =
      eligibility_record_ref(eligibility.slot, eligibility.generation);
  auto* drops = metrics != nullptr
      ? &metrics->provenance_no_evictable_slot_drops : nullptr;
  auto* refreshes = metrics != nullptr
      ? &metrics->authorized_participation_refreshes : nullptr;
  auto* collisions = metrics != nullptr
      ? &metrics->participation_authority_collision_rejects : nullptr;
  if (!insert_active_participation(
          participation, participation_locks, nullptr, descriptor.source_node,
          descriptor.ticket_id, descriptor.expiry_tick, current_tick, drops,
          descriptor.authority, descriptor.authority_incarnation, refreshes,
          collisions, nullptr, descriptor.claim_incarnation) ||
      !insert_active_participation(
          participation, participation_locks, nullptr, descriptor.target_node,
          descriptor.ticket_id, descriptor.expiry_tick, current_tick, drops,
          descriptor.authority, descriptor.authority_incarnation, refreshes,
          collisions, nullptr, descriptor.claim_incarnation, nullptr, nullptr,
          child_ref))
    return false;
  brain.routes[descriptor.route_index].eligibility_q16 =
      eligibility.canonical_eligibility_q16;
  occurrence_entry->occurrence.eligibility_ref = child_ref;
  stage_current_contribution(
      descriptor, staging, staging_count, staging_capacity, metrics);
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->actual_frontier_causal.occurrence_contributions_staged),
              1ULL);
  atomicAdd(incoming_excitation + descriptor.target_node, drive_q16);
  return true;
}
__device__ bool publish_admitted_resident_actual_contacts(
    DirectBrain brain, ResidentActualFrontier* frontier,
    NodeCausalParticipation* participation, std::uint32_t* participation_locks,
    EligibilityRecord* eligibility_table, std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging, std::uint32_t* staging_count,
    std::uint32_t staging_capacity, std::int32_t* incoming_excitation,
    AdultCoreMetrics* metrics, std::uint32_t current_tick);
DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER bool
has_unpublished_resident_actual_contact(
    const ResidentActualFrontier* frontier) {
  if (frontier == nullptr) return false;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    if (frontier->pending_contacts[i].state ==
        ResidentPendingActualContactState::admitted)
      return true;
  return false;
}
__device__ inline bool pending_actual_contact_bootstrap_current(
    const ResidentActualFrontier* frontier,
    const DirectParticipationDescriptor& participant,
    std::uint32_t current_tick) {
  if (frontier == nullptr ||
      participant.authority != DirectParticipationAuthority::independent_external)
    return false;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const auto& pending = frontier->pending_contacts[i];
    if (pending.state == ResidentPendingActualContactState::pending &&
        pending.event.ticket_id == participant.ticket_id &&
        pending.claim_incarnation == participant.claim_incarnation &&
        pending.authority_incarnation == participant.authority_incarnation &&
        pending.expiry_tick >= current_tick &&
        pending.receipt.selection == ResidentContactSelection::resident_owned &&
        pending.receipt.integration == ResidentContactIntegration::canonical &&
        pending.receipt.source_available == 1u)
      return true;
  }
  return false;
}
#endif

__device__ inline bool current_eligibility_record(
    std::uint64_t ref, const EligibilityRecord* table,
    const std::uint32_t* generations, std::uint32_t current_tick,
    EligibilityRecord* out) {
  if (ref == 0u || table == nullptr || generations == nullptr || out == nullptr)
    return false;
  const std::uint32_t encoded_slot = static_cast<std::uint32_t>(ref);
  const std::uint32_t generation = static_cast<std::uint32_t>(ref >> 32);
  if (encoded_slot == 0u || generation == 0u) return false;
  const std::uint32_t slot = encoded_slot - 1u;
  if (slot >= kMaxLiveEligibilityRecords ||
      atomicAdd(const_cast<std::uint32_t*>(generations + slot), 0u) != generation)
    return false;
  const EligibilityRecord record = table[slot];
  __threadfence();
  if (atomicAdd(const_cast<std::uint32_t*>(generations + slot), 0u) != generation ||
      record.live == 0u || record.expiry_tick < current_tick ||
      record.lineage_expiry_tick < current_tick || record.ancestry_depth == 0u ||
      record.ancestry_depth > kMaxProvenanceSlotsPerNode)
    return false;
  *out = record;
  return true;
}

__device__ inline bool verified_resident_sensory_root(
    DirectBrain brain, const EligibilityRecord& root,
    std::uint32_t current_tick) {
  using namespace direct_network;
  if (brain.postbirth_constructor == nullptr ||
      root.parent_eligibility_ref != 0u || root.ancestry_depth != 1u ||
      root.lineage_expiry_tick < current_tick)
    return false;
  const ResidentRawContactBinding binding = resident_raw_contact_authority(
      *brain.postbirth_constructor, root.ticket_id);
  if (resident_raw_contact_key_empty(binding.key) ||
      binding.key.node != root.source_node ||
      root.authority != DirectParticipationAuthority::independent_external ||
      binding.authority.authority_incarnation != root.authority_incarnation ||
      binding.authority.claim_incarnation != root.claim_incarnation ||
      !resident_sensory_authority_receipt_valid(
          root.ticket_id, binding.key, binding.authority,
          static_cast<std::uint32_t>(CausalOrigin::external_contact)))
    return false;
  return true;
}

__device__ inline bool freeze_eligibility_ancestry(
    const DirectParticipationDescriptor& terminal, DirectBrain brain,
    const EligibilityRecord* table, const std::uint32_t* generations,
    std::uint32_t current_tick, DirectParticipationDescriptor* closure,
    std::uint32_t* closure_count) {
  if (closure == nullptr || closure_count == nullptr ||
      terminal.eligibility_slot == kInvalidIndex ||
      terminal.eligibility_generation == 0u)
    return false;
  const std::uint64_t terminal_ref = eligibility_record_ref(
      terminal.eligibility_slot, terminal.eligibility_generation);
  DirectParticipationDescriptor reverse[kMaxProvenanceSlotsPerNode]{};
  std::uint64_t visited[kMaxProvenanceSlotsPerNode]{};
  std::uint32_t count = 0u;
  std::uint64_t ref = terminal_ref;
  std::uint32_t expected_target = terminal.target_node;
  for (;;) {
    if (count == kMaxProvenanceSlotsPerNode) return false;
    for (std::uint32_t i = 0u; i < count; ++i)
      if (visited[i] == ref) return false;
    EligibilityRecord record{};
    if (!current_eligibility_record(ref, table, generations, current_tick, &record) ||
        record.target_node != expected_target ||
        record.ticket_id != terminal.ticket_id ||
        record.claim_incarnation != terminal.claim_incarnation ||
        record.authority != terminal.authority ||
        record.authority_incarnation != terminal.authority_incarnation ||
        record.route_index >= brain.route_capacity || brain.routes == nullptr ||
        brain.route_incarnations == nullptr)
      return false;
    const DirectRoute route = brain.routes[record.route_index];
    if (!route_is_active(route) || route.source != record.source_node ||
        route.target != record.target_node ||
        brain.route_incarnations[record.route_index] != record.route_incarnation)
      return false;
    if (count == 0u && !eligibility_claim_matches(terminal, record)) return false;
    DirectParticipationDescriptor descriptor{};
    descriptor.ticket_id = record.ticket_id;
    descriptor.source_node = record.source_node;
    descriptor.target_node = record.target_node;
    descriptor.route_index = record.route_index;
    descriptor.context_signature = record.context_signature;
    descriptor.expiry_tick = record.lineage_expiry_tick;
    descriptor.claim_incarnation = record.claim_incarnation;
    descriptor.route_incarnation = record.route_incarnation;
    descriptor.authority = record.authority;
    descriptor.authority_incarnation = record.authority_incarnation;
    descriptor.contribution_kind = DirectContributionKind::sparse_route;
    descriptor.eligibility_slot = static_cast<std::uint32_t>(ref) - 1u;
    descriptor.eligibility_generation = static_cast<std::uint32_t>(ref >> 32);
    descriptor.frozen_eligibility_q16 = record.eligibility_q16;
    descriptor.parent_eligibility_ref = record.parent_eligibility_ref;
    descriptor.lineage_expiry_tick = record.lineage_expiry_tick;
    descriptor.ancestry_depth = record.ancestry_depth;
    reverse[count] = descriptor;
    visited[count++] = ref;
    if (record.parent_eligibility_ref == 0u) {
      if (record.ancestry_depth != 1u ||
          !verified_resident_sensory_root(brain, record, current_tick))
        return false;
      break;
    }
    EligibilityRecord parent{};
    if (!current_eligibility_record(record.parent_eligibility_ref, table,
                                    generations, current_tick, &parent) ||
        parent.target_node != record.source_node ||
        parent.ancestry_depth + 1u != record.ancestry_depth ||
        parent.lineage_expiry_tick < record.lineage_expiry_tick)
      return false;
    expected_target = record.source_node;
    ref = record.parent_eligibility_ref;
  }
  if (count != reverse[0].ancestry_depth) return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    closure[i] = reverse[count - 1u - i];
  *closure_count = count;
  return true;
}
__device__ DirectActionBindingResult bind_action_occurrence(
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const ResidentActualFrontier* actual_frontier,
    std::uint32_t require_exact_occurrence_identity,
    std::uint32_t motor_node,
    std::uint32_t current_tick,
    std::uint64_t action_ticket_id,
    std::uint32_t action_context,
    std::uint32_t motor_channel,
    std::uint32_t action_expiry_tick,
    std::uint32_t action_slot,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity,
    AdultCoreMetrics* metrics,
    const ResidentActivationSoaPlane* activation_plane = nullptr,
    DirectBrain ancestry_brain = {},
    const EligibilityRecord* eligibility_table = nullptr,
    const std::uint32_t* eligibility_record_generations = nullptr);
__device__ inline bool frozen_sparse_action_link_current(
    const DirectActionParticipationLink& link, const DirectNode* nodes, std::uint32_t node_count, const DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity);
__device__ inline bool frozen_action_occurrence_identity_current(
    const DirectActionParticipationLink& link, DirectBrain brain,
    const ResidentActualFrontier* actual_frontier);
#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
// Frozen action locators select; current RecipeRevision morphology expresses.
DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER Word
resident_bound_action_motor_word(
    DirectBrain brain, const DirectNode& node, const DirectRoute* routes, std::uint32_t route_capacity, std::uint32_t route_scan_count,
    const ResidentActualFrontier* actual_frontier, DirectActionOccurrence* actions, const DirectActionParticipationLink* links,
    const DirectActionBindingResult& binding, std::uint32_t action_slot, std::uint32_t participant_capacity, std::uint64_t ticket,
    std::uint32_t motor_node, std::uint32_t motor_channel, std::uint32_t context_signature, std::uint32_t publication_tick) {
  const Word fallback = resident_motor_word(node, routes, route_capacity, route_scan_count);
  // Exact represented content remains usable; the caller keeps broader
  // unrepresented context explicit on MotorEvent::reserved.
  if (binding.admitted == 0u || actions == nullptr || links == nullptr ||
      action_slot >= kMaxAsynchronousTickets || participant_capacity == 0u || participant_capacity > kMaxActionParticipationLinks ||
      ticket == 0u || ticket == kInvalidTicket || action_slot != ticket % kMaxAsynchronousTickets) return fallback;
  DirectActionOccurrence& action = actions[action_slot];
  if (atomicAdd(&action.state, 0u) != kActionOccurrencePending || action.action_ticket_id != ticket ||
      action.motor_node != motor_node || action.motor_channel != motor_channel || action.context_signature != context_signature ||
      action.emission_tick != publication_tick || action.expiry_tick < publication_tick || action.participant_offset == kInvalidIndex ||
      action.participant_count == 0u || action.participant_offset != action_slot * participant_capacity ||
      action.participant_offset != binding.participant_offset || action.participant_count != binding.participant_count ||
      action.participant_count > participant_capacity || action.participant_count > kMaxActionParticipationLinks ||
      action.occurrence_identity_required == 0u || action.occurrence_identity_complete == 0u) return fallback;
  std::uint32_t recipe_links = 0u;
  for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
    const DirectActionParticipationLink& link = links[action.participant_offset + i];
    if (link.contribution_kind == DirectContributionKind::sparse_route && !frozen_sparse_action_link_current(link, brain.nodes, brain.node_count, brain.routes, brain.route_incarnations, brain.route_capacity)) return fallback;
    const bool has_recipe_identity = link.logical_recipe_id != 0u ||
        link.revision_identity != 0u || link.occurrence_identity != 0u ||
        link.participation_identity != 0u || link.occurrence_route_incarnation != 0u;
    if (!has_recipe_identity) continue;
    if (link.logical_recipe_id == 0u || link.revision_identity == 0u ||
        link.occurrence_identity == 0u || link.participation_identity == 0u ||
        link.occurrence_route_incarnation == 0u ||
        !frozen_action_occurrence_identity_current(link, brain, actual_frontier)) return fallback;
    ++recipe_links;
  }
  return recipe_links == 0u ? fallback : resident_motor_word(brain, node, routes, route_capacity, route_scan_count, actual_frontier, links, action.participant_offset, action.participant_count);
}
#endif
__device__ std::uint32_t record_dense_shatter_transaction(DirectDenseBlock*, std::uint32_t, const DirectActionParticipationLink*, std::uint32_t, const std::uint32_t*, std::uint32_t, direct_network::DirectExactHistoryRecord*, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t);
__device__ void apply_dense_shatter_transaction(DirectDenseBlock*, std::uint32_t, const DirectActionParticipationLink*, std::uint32_t, const std::uint32_t*, std::uint32_t, AdultCoreMetrics*);
__device__ inline void apply_recorded_sparse_action_link(
    const DirectActionParticipationLink& link, const ResolvedConsequenceContext& ctx,
    DirectNode* nodes, DirectRoute* routes, AdultCoreMetrics* metrics,
    std::int32_t applied_delta_q16);
__device__ inline void bind_raw_contact_recipe_incidence(
    DirectBrain brain, const DirectActionParticipationLink& link,
    const direct_network::DirectExactHistoryRecord& witness) {
  auto* state = brain.postbirth_constructor;
  if (state == nullptr || brain.development == nullptr ||
      brain.postbirth_derivations == nullptr ||
      brain.recipe_cells == nullptr || brain.routes == nullptr ||
      link.route_index >= brain.route_capacity)
    return;
  const auto contact = direct_network::resident_raw_contact_binding(
      *state, link.participant_ticket_id);
  if (direct_network::resident_raw_contact_key_empty(contact) ||
      link.source_node != contact.node)
    return;
  const std::uint32_t builder = direct_network::decode_route_recipe_builder(
      brain.routes[link.route_index].flags);
  if (builder < brain.development->recipe_cell_count)
    for (std::uint32_t d = 0u; d < state->derivation_count; ++d) {
      const auto& derivation = brain.postbirth_derivations[d];
      if (derivation.recipe_cell != builder ||
          derivation.route_index != link.route_index ||
          derivation.source_node != contact.node ||
          derivation.port_count == 0u ||
          derivation.ports[0].node != contact.node ||
          derivation.route_incarnations[0] != link.route_incarnation ||
          derivation.revision_identity !=
              brain.recipe_cells[builder].revision_identity)
        continue;
      direct_network::record_resident_recipe_incidence(
          state, d, derivation, contact);
      return;
    }
  (void)direct_network::materialize_raw_contact_sparse_credit(
      brain, witness, contact, nullptr);
}
__device__ void resolve_consequence_ticket_device(
    AsynchronousTicket* ticket_table, DirectActionOccurrence* action_occurrences, DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity, std::uint64_t target_ticket_id,
    Word returned_word, CausalOrigin origin, std::uint32_t admission_tick,
    direct_network::DirectExactHistoryHotPage* exact_history,
    std::uint32_t transport_cursor, DirectBrain brain,
    ResidentActualFrontier* actual_frontier,
    const ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    DirectNode* nodes, std::uint32_t node_count, DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    substrate::direct_adult::DirectRetentionState* retention_bank, direct_network::ResidentRecipeCell* recipe_cells,
    std::uint32_t recipe_cell_count, DirectDenseBlock* dense_blocks, std::uint32_t dense_block_count,
    std::int32_t learning_rate_q16, std::int32_t shatter_threshold_q16,
    ResolvedConsequenceContext* out_ctx, AdultCoreMetrics* metrics);
#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
__device__ inline bool frozen_action_occurrence_identity_current(
    const DirectActionParticipationLink& link, DirectBrain brain,
    const ResidentActualFrontier* actual_frontier) {
  if (actual_frontier == nullptr || brain.postbirth_constructor == nullptr ||
      link.logical_recipe_id == 0u || link.revision_identity == 0u ||
      link.occurrence_identity == 0u || link.participation_identity == 0u ||
      link.occurrence_route_incarnation == 0u ||
      actual_frontier->live_count > kResidentActualFrontierCapacity)
    return false;
  const auto& state = *brain.postbirth_constructor;
  const ResidentActualFrontierEntry* match = nullptr;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const ResidentActualFrontierEntry& entry = actual_frontier->entries[i];
    const ResidentRecipeOccurrence& occurrence = entry.occurrence;
    if (entry.state != ResidentActualFrontierState::live ||
        occurrence.occurrence_identity != link.occurrence_identity ||
        occurrence.logical_recipe_id != link.logical_recipe_id ||
        occurrence.revision_identity != link.revision_identity ||
        occurrence.participation_identity != link.participation_identity ||
        occurrence.context_signature != link.occurrence_context_signature ||
        occurrence.route_incarnation != link.occurrence_route_incarnation ||
        occurrence.source_incarnation != link.claim_incarnation ||
        occurrence.authority != link.authority)
      continue;
    if (match != nullptr) return false;
    match = &entry;
  }
  return match != nullptr &&
      resident_actual_frontier_entry_current(brain, state, *match);
}
__device__ inline bool frozen_sparse_action_link_current(
    const DirectActionParticipationLink& link, const DirectNode* nodes, std::uint32_t node_count, const DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity) {
  const bool route_valid = link.contribution_kind == DirectContributionKind::sparse_route && nodes && routes &&
      route_incarnations && link.source_node < node_count && link.target_node < node_count &&
      link.route_index < route_capacity && link.frozen_eligibility_q16 > 0;
  if (!route_valid) return false;
  const DirectRoute& route = routes[link.route_index];
  return route_incarnations[link.route_index] == link.route_incarnation && route.source == link.source_node &&
      route.target == link.target_node;
}
__device__ inline void apply_recorded_sparse_action_link(
    const DirectActionParticipationLink& link, const ResolvedConsequenceContext& ctx,
    DirectNode* nodes, DirectRoute* routes, AdultCoreMetrics* metrics,
    std::int32_t applied_delta_q16) {
  DirectRoute& route = routes[link.route_index];
  route.conductance_q16 += applied_delta_q16; route.last_credit_q16 = applied_delta_q16;
  route.last_credit_ticket = ctx.target_ticket_id;
  atomic_ema_update_q16(&nodes[link.source_node].credit_ema_q16, applied_delta_q16);
  atomic_ema_update_q16(&nodes[link.target_node].credit_ema_q16, applied_delta_q16);
  if (metrics) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->credit_updates_committed), 1ULL);
    if (applied_delta_q16 > 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->total_positive_credit_q16),
                static_cast<unsigned long long>(applied_delta_q16));
    } else {
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->total_negative_credit_q16),
                static_cast<unsigned long long>(-applied_delta_q16));
    }
  }
}
#endif
__device__ void device_process_single_sensory_event(
    DirectNode* nodes,
    const DirectBoundaryPort* ports,
    std::uint32_t port_count,
    std::uint32_t node_count,
    const ActivityEvent& event,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    NodeCausalParticipation* node_active_participation,
    std::uint32_t* node_active_participation_locks,
    std::uint32_t* claim_incarnation_counter,
    std::uint32_t* ticket_table_locks,
    AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectParticipationDescriptor* current_contributions,
    std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick,
    std::uint32_t horizon_ticks,
    direct_network::DirectExactHistoryRecord* history_record = nullptr,
    bool membrane_authenticated = false,
    std::uint32_t membrane_authority_incarnation = 0u);
#if !defined(DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY) || \
    defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
#if defined(DIRECT_ADULT_DEVICE_OPS_DEFINE_OWNER)
#define DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER __device__
#else
#define DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER __device__ inline
#endif
#include "hardware_native/direct_adult_device_ops_heavy.inl"
#undef DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER
#endif

}  // namespace substrate::direct_adult_core

#undef DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_DEVICE_OPS_CUH
