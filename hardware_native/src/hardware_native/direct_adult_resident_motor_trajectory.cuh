#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_CUH

#include <cstdint>
#include <type_traits>

inline constexpr std::uint32_t kResidentPublicMotorTrajectoryIdle = 0u;
inline constexpr std::uint32_t kResidentPublicMotorTrajectoryLive = 1u;

// One finite resident continuation. It freezes causal identities, never output
// bytes: each public Word is re-derived from current morphology at its tick.
struct alignas(8) ResidentPublicMotorTrajectory {
  std::uint64_t trajectory_identity;
  std::uint64_t causal_integrity_root;
  std::uint64_t boundary_session_epoch;
  std::uint32_t state;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_port_index;
  std::uint32_t motor_physical_route;
  std::uint32_t motor_parent_route;
  std::uint32_t origin_tick;
  std::uint32_t cursor;
  std::uint32_t extent;
  std::uint32_t contributor_count;
  std::uint32_t reserved;
  std::uint32_t composition_depth;
  DirectActionParticipationLink contributors[kMaxProvenanceSlotsPerNode];
  std::uint64_t contact_identities[kMaxProvenanceSlotsPerNode];
  std::uint64_t ingress_sequences[kMaxProvenanceSlotsPerNode];
  std::uint32_t occurrence_timestamps[kMaxProvenanceSlotsPerNode];
};
static_assert(std::is_standard_layout_v<ResidentPublicMotorTrajectory> &&
              std::is_trivial_v<ResidentPublicMotorTrajectory> &&
              std::has_unique_object_representations_v<ResidentPublicMotorTrajectory>);

// Immutable metadata attached to the already-public MotorEvent. No API accepts
// this record as input, so observer edits cannot affect resident execution.
struct alignas(8) ResidentPublicMotorTrajectoryArtifact {
  std::uint64_t trajectory_identity;
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t occurrence_identity;
  std::uint64_t participation_identity;
  std::uint64_t occurrence_route_incarnation;
  std::uint64_t participant_ticket_id;
  std::uint64_t contact_identity;
  std::uint64_t ingress_sequence;
  std::uint64_t route_incarnation;
  std::uint64_t boundary_session_epoch;
  std::uint64_t executed_revision_identity;
  std::uint64_t occurrence_source_identity;
  std::uint64_t causal_integrity_root;
  std::uint32_t cursor;
  std::uint32_t extent;
  std::uint32_t contributor_count;
  std::uint32_t occurrence_timestamp;
  std::uint32_t claim_incarnation;
  std::uint32_t authority_incarnation;
  std::uint32_t source_node;
  std::uint32_t target_node;
  std::uint32_t route_index;
  std::uint32_t motor_port_index;
  std::uint32_t motor_physical_route;
  std::uint32_t motor_parent_route;
  std::uint32_t origin_tick;
  std::uint32_t link_context_signature;
  std::uint32_t occurrence_context_signature;
  std::uint32_t composition_depth;
  std::uint32_t link_expiry_tick;
  std::uint32_t executed_expiry_tick;
  std::uint32_t authority;
  std::uint32_t contribution_kind;
  std::uint32_t occurrence_authority;
  std::uint32_t occurrence_lineage_kind;
  std::uint32_t occurrence_source_incarnation;
  std::int32_t frozen_eligibility_q16;
  std::uint32_t eligibility_slot;
  std::uint32_t eligibility_generation;
  std::uint32_t reserved;
  std::uint32_t result_composition_depth;
  std::uint32_t reserved3;
  std::uint32_t reserved4;
};
static_assert(sizeof(ResidentPublicMotorTrajectoryArtifact) == 232u &&
              std::is_standard_layout_v<ResidentPublicMotorTrajectoryArtifact> &&
              std::is_trivial_v<ResidentPublicMotorTrajectoryArtifact> &&
              std::has_unique_object_representations_v<ResidentPublicMotorTrajectoryArtifact>);

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_CUH

#if defined(HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_OPS) && \
    !defined(HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_OPS_CUH)
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_OPS_CUH

namespace substrate::direct_adult_core {

__device__ inline void resident_motor_trajectory_hash_bytes(
    std::uint64_t* root, const void* value, std::uint32_t bytes) {
  const auto* data = static_cast<const std::uint8_t*>(value);
  for (std::uint32_t i = 0u; i < bytes; ++i) {
    *root ^= data[i];
    *root *= 1099511628211ull;
  }
}

__device__ inline std::uint64_t resident_motor_trajectory_integrity_root(
    const ResidentPublicMotorTrajectory& trajectory) {
  std::uint64_t root = 1469598103934665603ull;
  resident_motor_trajectory_hash_bytes(
      &root, &trajectory.trajectory_identity,
      sizeof(trajectory.trajectory_identity));
  resident_motor_trajectory_hash_bytes(
      &root, &trajectory.boundary_session_epoch,
      sizeof(trajectory.boundary_session_epoch));
  resident_motor_trajectory_hash_bytes(
      &root, &trajectory.motor_node,
      sizeof(trajectory.motor_node) * 6u);
  resident_motor_trajectory_hash_bytes(
      &root, &trajectory.extent,
      sizeof(trajectory.extent) * 2u);
  resident_motor_trajectory_hash_bytes(
      &root, &trajectory.reserved, sizeof(trajectory.reserved) * 2u);
  for (std::uint32_t i = 0u; i < trajectory.extent; ++i) {
    resident_motor_trajectory_hash_bytes(
        &root, &trajectory.contributors[i], sizeof(trajectory.contributors[i]));
    resident_motor_trajectory_hash_bytes(
        &root, &trajectory.contact_identities[i],
        sizeof(trajectory.contact_identities[i]));
    resident_motor_trajectory_hash_bytes(
        &root, &trajectory.ingress_sequences[i],
        sizeof(trajectory.ingress_sequences[i]));
    resident_motor_trajectory_hash_bytes(
        &root, &trajectory.occurrence_timestamps[i],
        sizeof(trajectory.occurrence_timestamps[i]));
  }
  return root;
}

__device__ inline bool resident_motor_trajectory_link_current(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionParticipationLink& link, std::uint32_t current_tick,
    Word* word = nullptr) {
  const auto* entry =
      resident_current_lived_expression_entry(brain, frontier, link);
  if (entry == nullptr || current_tick > link.expiry_tick ||
      current_tick > entry->occurrence.expiry_tick ||
      link.route_index >= brain.route_capacity || brain.routes == nullptr ||
      brain.route_incarnations == nullptr ||
      !direct_network::route_is_active(brain.routes[link.route_index]) ||
      brain.routes[link.route_index].source != link.source_node ||
      brain.routes[link.route_index].target != link.target_node ||
      brain.route_incarnations[link.route_index] != link.route_incarnation)
    return false;
  return word == nullptr ||
      resident_lived_expression_entry(brain, *entry, word);
}

__device__ inline void refuse_resident_motor_trajectory(
    ResidentPublicMotorTrajectory* trajectory, AdultCoreMetrics* metrics) {
  if (trajectory != nullptr) *trajectory = ResidentPublicMotorTrajectory{};
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->motor_trajectory_stale_refusals), 1ULL);
}

__device__ inline std::uint32_t resident_motor_trajectory_candidate_count(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t current_tick) {
  if (action_links == nullptr || action.participant_offset == kInvalidIndex ||
      action.participant_count == 0u ||
      action.participant_count > kMaxActionParticipationLinks ||
      action.occurrence_identity_required == 0u ||
      action.occurrence_identity_complete == 0u)
    return 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
    const auto& link = action_links[action.participant_offset + i];
    if (link.logical_recipe_id == 0u) continue;
    if (!resident_motor_trajectory_link_current(
            brain, frontier, link, current_tick))
      return kInvalidIndex;
    bool duplicate = false;
    for (std::uint32_t j = 0u; j < i; ++j) {
      const auto& prior = action_links[action.participant_offset + j];
      duplicate |= prior.logical_recipe_id == link.logical_recipe_id &&
          prior.occurrence_identity == link.occurrence_identity &&
          prior.participation_identity == link.participation_identity;
    }
    if (duplicate) continue;
    ++count;
  }
  return count;
}

__device__ inline bool resident_motor_trajectory_required(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t current_tick) {
  const std::uint32_t count = resident_motor_trajectory_candidate_count(
      brain, frontier, action, action_links, current_tick);
  return count != kInvalidIndex && count > 1u;
}

__device__ inline bool begin_resident_motor_trajectory(
    ResidentPublicMotorTrajectory* trajectory, const DirectBrain& brain,
    const ResidentActualFrontier* frontier,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    const DirectBoundaryPort& motor_port, std::uint32_t motor_port_index,
    std::uint32_t capacity, std::uint64_t trajectory_identity,
    std::uint32_t ancestry_incomplete, AdultCoreMetrics* metrics) {
  if (trajectory == nullptr || frontier == nullptr || action_links == nullptr ||
      trajectory->state == kResidentPublicMotorTrajectoryLive ||
      action.participant_offset == kInvalidIndex ||
      action.participant_count == 0u ||
      capacity > kMaxProvenanceSlotsPerNode ||
      action.occurrence_identity_required == 0u ||
      action.occurrence_identity_complete == 0u) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->motor_trajectory_capacity_refusals), 1ULL);
    return false;
  }
  ResidentPublicMotorTrajectory next{};
  next.trajectory_identity = trajectory_identity;
  next.boundary_session_epoch = frontier->boundary_session_epoch;
  next.state = kResidentPublicMotorTrajectoryLive;
  next.motor_node = action.motor_node;
  next.motor_channel = action.motor_channel;
  next.motor_port_index = motor_port_index;
  next.motor_physical_route = motor_port.physical_route;
  next.motor_parent_route = motor_port.parent_route;
  next.origin_tick = action.emission_tick;
  next.extent = resident_motor_trajectory_candidate_count(
      brain, frontier, action, action_links, action.emission_tick);
  next.contributor_count = next.extent;
  next.reserved = ancestry_incomplete != 0u ? 1u : 0u;
  if (next.extent == kInvalidIndex || next.extent < 2u ||
      next.extent > capacity) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->motor_trajectory_capacity_refusals), 1ULL);
    return false;
  }
  for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
    const auto& link = action_links[action.participant_offset + i];
    if (link.composition_depth > next.composition_depth)
      next.composition_depth = link.composition_depth;
  }
  ++next.composition_depth;
  if (next.composition_depth > kMaxActiveCompositionDepth) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->motor_trajectory_capacity_refusals), 1ULL);
    return false;
  }
  std::uint32_t contributor = 0u;
  for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
    const auto link = action_links[action.participant_offset + i];
    if (resident_motor_trajectory_link_current(
            brain, frontier, link, action.emission_tick)) {
      bool duplicate = false;
      for (std::uint32_t j = 0u; j < contributor; ++j)
        duplicate |= next.contributors[j].logical_recipe_id ==
                         link.logical_recipe_id &&
            next.contributors[j].occurrence_identity ==
                link.occurrence_identity &&
            next.contributors[j].participation_identity ==
                link.participation_identity;
      if (duplicate) continue;
      const auto* entry =
          resident_current_lived_expression_entry(brain, frontier, link);
      next.contributors[contributor] = link;
      next.contact_identities[contributor] = entry->contact_identity;
      next.ingress_sequences[contributor] = entry->ingress_sequence;
      next.occurrence_timestamps[contributor] = entry->occurrence.timestamp;
      ++contributor;
    }
  }
  // Canonicalize contributor order by lived-expression Word, not injection
  // ingress. Ingress-first made coordinated EF vs FE emit different first
  // motors (cursor 0 tracked contact order). Strong A1 (#1610) needs one
  // shared motor for the same unordered constituent set: sort by the
  // experience-authorized expression of each contributor, then identity.
  Word contributor_words[kMaxProvenanceSlotsPerNode]{};
  for (std::uint32_t i = 0u; i < next.extent; ++i) {
    if (!resident_motor_trajectory_link_current(
            brain, frontier, next.contributors[i], action.emission_tick,
            &contributor_words[i])) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->motor_trajectory_stale_refusals), 1ULL);
      return false;
    }
  }
  for (std::uint32_t i = 1u; i < next.extent; ++i) {
    const DirectActionParticipationLink link = next.contributors[i];
    const std::uint64_t contact = next.contact_identities[i];
    const std::uint64_t ingress = next.ingress_sequences[i];
    const std::uint32_t timestamp = next.occurrence_timestamps[i];
    const Word expression = contributor_words[i];
    std::uint32_t j = i;
    while (j != 0u) {
      const bool before =
          expression < contributor_words[j - 1u] ||
          (expression == contributor_words[j - 1u] &&
           (link.occurrence_identity <
                next.contributors[j - 1u].occurrence_identity ||
            (link.occurrence_identity ==
                 next.contributors[j - 1u].occurrence_identity &&
             (ingress < next.ingress_sequences[j - 1u] ||
              (ingress == next.ingress_sequences[j - 1u] &&
               (timestamp < next.occurrence_timestamps[j - 1u] ||
                (timestamp == next.occurrence_timestamps[j - 1u] &&
                 contact < next.contact_identities[j - 1u])))))));
      if (!before) break;
      next.contributors[j] = next.contributors[j - 1u];
      next.contact_identities[j] = next.contact_identities[j - 1u];
      next.ingress_sequences[j] = next.ingress_sequences[j - 1u];
      next.occurrence_timestamps[j] = next.occurrence_timestamps[j - 1u];
      contributor_words[j] = contributor_words[j - 1u];
      --j;
    }
    next.contributors[j] = link;
    next.contact_identities[j] = contact;
    next.ingress_sequences[j] = ingress;
    next.occurrence_timestamps[j] = timestamp;
    contributor_words[j] = expression;
  }
  next.causal_integrity_root =
      resident_motor_trajectory_integrity_root(next);
  *trajectory = next;
  if (metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->motor_trajectory_origins), 1ULL);
  return true;
}

__device__ inline bool resident_motor_trajectory_word(
    const ResidentPublicMotorTrajectory& trajectory, const DirectBrain& brain,
    const ResidentActualFrontier* frontier, std::uint32_t current_tick,
    Word* word) {
  if (trajectory.state != kResidentPublicMotorTrajectoryLive ||
      trajectory.extent == 0u ||
      trajectory.extent > kMaxProvenanceSlotsPerNode ||
      trajectory.cursor >= trajectory.extent ||
      trajectory.contributor_count != trajectory.extent ||
      frontier == nullptr ||
      frontier->boundary_session_epoch != trajectory.boundary_session_epoch ||
      trajectory.causal_integrity_root !=
          resident_motor_trajectory_integrity_root(trajectory))
    return false;
  for (std::uint32_t i = 0u; i < trajectory.extent; ++i) {
    const auto* entry = resident_current_lived_expression_entry(
        brain, frontier, trajectory.contributors[i]);
    if (!resident_motor_trajectory_link_current(
            brain, frontier, trajectory.contributors[i], current_tick) ||
        entry == nullptr ||
        entry->contact_identity != trajectory.contact_identities[i] ||
        entry->ingress_sequence != trajectory.ingress_sequences[i] ||
        entry->occurrence.timestamp != trajectory.occurrence_timestamps[i])
      return false;
  }
  return resident_motor_trajectory_link_current(
      brain, frontier, trajectory.contributors[trajectory.cursor],
      current_tick, word);
}

__device__ inline ResidentPublicMotorTrajectoryArtifact
resident_motor_trajectory_artifact(
    const ResidentPublicMotorTrajectory& trajectory, const DirectBrain& brain,
    const ResidentActualFrontier* frontier) {
  ResidentPublicMotorTrajectoryArtifact artifact{};
  if (trajectory.state != kResidentPublicMotorTrajectoryLive ||
      trajectory.cursor >= trajectory.extent)
    return artifact;
  const auto& link = trajectory.contributors[trajectory.cursor];
  const auto* entry =
      resident_current_lived_expression_entry(brain, frontier, link);
  if (entry == nullptr) return artifact;
  artifact.trajectory_identity = trajectory.trajectory_identity;
  artifact.logical_recipe_id = link.logical_recipe_id;
  artifact.revision_identity = link.revision_identity;
  artifact.occurrence_identity = link.occurrence_identity;
  artifact.participation_identity = link.participation_identity;
  artifact.occurrence_route_incarnation =
      link.occurrence_route_incarnation;
  artifact.participant_ticket_id = link.participant_ticket_id;
  artifact.contact_identity =
      trajectory.contact_identities[trajectory.cursor];
  artifact.ingress_sequence =
      trajectory.ingress_sequences[trajectory.cursor];
  artifact.route_incarnation = link.route_incarnation;
  artifact.boundary_session_epoch = trajectory.boundary_session_epoch;
  artifact.executed_revision_identity = entry->occurrence.revision_identity;
  artifact.occurrence_source_identity = entry->occurrence.source_identity;
  artifact.causal_integrity_root = trajectory.causal_integrity_root;
  artifact.cursor = trajectory.cursor;
  artifact.extent = trajectory.extent;
  artifact.contributor_count = trajectory.extent;
  artifact.occurrence_timestamp =
      trajectory.occurrence_timestamps[trajectory.cursor];
  artifact.claim_incarnation = link.claim_incarnation;
  artifact.authority_incarnation = link.authority_incarnation;
  artifact.source_node = link.source_node;
  artifact.target_node = link.target_node;
  artifact.route_index = link.route_index;
  artifact.motor_port_index = trajectory.motor_port_index;
  artifact.motor_physical_route = trajectory.motor_physical_route;
  artifact.motor_parent_route = trajectory.motor_parent_route;
  artifact.origin_tick = trajectory.origin_tick;
  artifact.link_context_signature = link.context_signature;
  artifact.occurrence_context_signature = link.occurrence_context_signature;
  artifact.composition_depth = link.composition_depth;
  artifact.link_expiry_tick = link.expiry_tick;
  artifact.executed_expiry_tick = entry->occurrence.expiry_tick;
  artifact.authority = static_cast<std::uint32_t>(link.authority);
  artifact.contribution_kind =
      static_cast<std::uint32_t>(link.contribution_kind);
  artifact.occurrence_authority =
      static_cast<std::uint32_t>(entry->occurrence.authority);
  artifact.occurrence_lineage_kind =
      static_cast<std::uint32_t>(entry->occurrence.lineage_kind);
  artifact.occurrence_source_incarnation =
      entry->occurrence.source_incarnation;
  artifact.frozen_eligibility_q16 = link.frozen_eligibility_q16;
  artifact.eligibility_slot = link.eligibility_slot;
  artifact.eligibility_generation = link.eligibility_generation;
  artifact.reserved = trajectory.reserved;
  artifact.result_composition_depth = trajectory.composition_depth;
  return artifact;
}

__device__ inline void advance_resident_motor_trajectory(
    ResidentPublicMotorTrajectory* trajectory, AdultCoreMetrics* metrics,
    bool continuation) {
  if (trajectory == nullptr ||
      trajectory->state != kResidentPublicMotorTrajectoryLive)
    return;
  if (continuation && metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->motor_trajectory_continuations), 1ULL);
  ++trajectory->cursor;
  if (trajectory->cursor == trajectory->extent) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->motor_trajectory_completions), 1ULL);
    *trajectory = ResidentPublicMotorTrajectory{};
  }
}

__device__ inline bool same_resident_motor_trajectory_contributor(
    const DirectActionParticipationLink& frozen,
    const DirectActionParticipationLink& rebound, const DirectBrain& brain,
    const ResidentActualFrontier* frontier) {
  const auto* current =
      resident_current_lived_expression_entry(brain, frontier, frozen);
  const bool revision_current = current != nullptr &&
      rebound.revision_identity == current->occurrence.revision_identity;
  return frozen.participant_ticket_id == rebound.participant_ticket_id &&
      frozen.logical_recipe_id == rebound.logical_recipe_id &&
      (frozen.revision_identity == rebound.revision_identity ||
       revision_current) &&
      frozen.occurrence_identity == rebound.occurrence_identity &&
      frozen.participation_identity == rebound.participation_identity &&
      frozen.occurrence_route_incarnation ==
          rebound.occurrence_route_incarnation &&
      frozen.source_node == rebound.source_node &&
      frozen.target_node == rebound.target_node &&
      frozen.route_index == rebound.route_index &&
      frozen.context_signature == rebound.context_signature &&
      frozen.occurrence_context_signature ==
          rebound.occurrence_context_signature &&
      frozen.composition_depth == rebound.composition_depth &&
      frozen.expiry_tick == rebound.expiry_tick &&
      frozen.route_incarnation == rebound.route_incarnation &&
      frozen.claim_incarnation == rebound.claim_incarnation &&
      frozen.authority_incarnation == rebound.authority_incarnation &&
      frozen.authority == rebound.authority &&
      frozen.contribution_kind == rebound.contribution_kind &&
      frozen.frozen_eligibility_q16 == rebound.frozen_eligibility_q16 &&
      frozen.eligibility_slot == rebound.eligibility_slot &&
      frozen.eligibility_generation == rebound.eligibility_generation;
}

__device__ inline bool continue_resident_motor_trajectory(
    ResidentPublicMotorTrajectory* trajectory, const DirectBrain& brain,
    const DirectBoundaryPort* ports, std::uint32_t port_count,
    const ResidentActualFrontier* frontier, std::uint32_t current_tick,
    std::uint32_t action_horizon_ticks, AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count, std::uint32_t* ticket_table_locks,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity, AdultCoreMetrics* metrics,
    ResidentActivationSoaPlane* activation_plane,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations, MotorEvent* event) {
  if (trajectory == nullptr ||
      trajectory->state != kResidentPublicMotorTrajectoryLive || event == nullptr)
    return false;
  const bool physical_port_current =
      trajectory->motor_port_index < port_count &&
      (ports[trajectory->motor_port_index].role_mask &
       static_cast<std::uint32_t>(direct_network::BoundaryRole::motor)) != 0u &&
      ports[trajectory->motor_port_index].node == trajectory->motor_node &&
      ports[trajectory->motor_port_index].channel == trajectory->motor_channel &&
      ports[trajectory->motor_port_index].physical_route ==
          trajectory->motor_physical_route &&
      ports[trajectory->motor_port_index].parent_route ==
          trajectory->motor_parent_route;
  Word word = 0u;
  if (!physical_port_current ||
      !resident_motor_trajectory_word(
          *trajectory, brain, frontier, current_tick, &word)) {
    refuse_resident_motor_trajectory(trajectory, metrics);
    return false;
  }
  std::uint32_t ticket_slot = kInvalidIndex;
  for (std::uint32_t attempt = 0u; attempt < kMaxAsynchronousTickets; ++attempt) {
    const std::uint32_t candidate =
        atomicAdd(ticket_count, 1u) % kMaxAsynchronousTickets;
    if (lock_ticket_slot_for_motor(
            ticket_table_locks, ticket_table, action_occurrences, action_links,
            action_participant_capacity, brain.development, candidate,
            current_tick, action_horizon_ticks, metrics)) {
      ticket_slot = candidate;
      break;
    }
  }
  if (ticket_slot == kInvalidIndex) return false;
  const std::uint64_t ticket_id =
      (static_cast<std::uint64_t>(current_tick) << 32) |
      (static_cast<std::uint64_t>(trajectory->motor_channel) << 16) |
      ticket_slot;
  DirectParticipationDescriptor descriptors[kMaxProvenanceSlotsPerNode]{};
  for (std::uint32_t i = 0u; i < trajectory->extent; ++i) {
    const auto& link = trajectory->contributors[i];
    auto& descriptor = descriptors[i];
    descriptor.ticket_id = link.participant_ticket_id;
    descriptor.source_node = link.source_node;
    descriptor.target_node = link.target_node;
    descriptor.route_index = link.route_index;
    descriptor.context_signature = link.context_signature;
    descriptor.expiry_tick = link.expiry_tick;
    descriptor.claim_incarnation = link.claim_incarnation;
    descriptor.route_incarnation = link.route_incarnation;
    descriptor.authority = link.authority;
    descriptor.authority_incarnation = link.authority_incarnation;
    descriptor.contribution_kind = link.contribution_kind;
    descriptor.eligibility_slot = link.eligibility_slot;
    descriptor.eligibility_generation = link.eligibility_generation;
    descriptor.frozen_eligibility_q16 = link.frozen_eligibility_q16;
    EligibilityRecord terminal{};
    if (!current_eligibility_record(
            eligibility_record_ref(link.eligibility_slot,
                                   link.eligibility_generation),
            eligibility_table, eligibility_record_generations, current_tick,
            &terminal)) {
      unlock_ticket_slot(ticket_table_locks, ticket_slot);
      refuse_resident_motor_trajectory(trajectory, metrics);
      return false;
    }
    descriptor.parent_eligibility_ref = terminal.parent_eligibility_ref;
    descriptor.lineage_expiry_tick = terminal.lineage_expiry_tick;
    descriptor.ancestry_depth = terminal.ancestry_depth;
    if (!eligibility_claim_matches(descriptor, terminal) ||
        terminal.eligibility_q16 != link.frozen_eligibility_q16) {
      unlock_ticket_slot(ticket_table_locks, ticket_slot);
      refuse_resident_motor_trajectory(trajectory, metrics);
      return false;
    }
  }
  const std::uint32_t context =
      (trajectory->motor_node * 1103515245U) ^ current_tick;
  DirectActionBindingResult binding = bind_action_occurrence(
      descriptors, &trajectory->contributor_count, trajectory->extent,
      frontier, 1u,
      trajectory->motor_node, current_tick, ticket_id, context,
      trajectory->motor_channel, current_tick + action_horizon_ticks,
      ticket_slot, action_occurrences, action_links,
      action_participant_capacity, metrics, activation_plane, brain,
      eligibility_table, eligibility_record_generations);
  if (binding.admitted == 0u ||
      binding.participant_count < trajectory->extent ||
      binding.participant_count > action_participant_capacity ||
      binding.participant_offset == kInvalidIndex) {
    unlock_ticket_slot(ticket_table_locks, ticket_slot);
    refuse_resident_motor_trajectory(trajectory, metrics);
    return false;
  }
  bool rebound_matched[kMaxActionParticipationLinks]{};
  bool exact_bijection = true;
  for (std::uint32_t frozen_index = 0u;
       frozen_index < trajectory->extent; ++frozen_index) {
    std::uint32_t match = kInvalidIndex;
    for (std::uint32_t rebound_index = 0u;
         rebound_index < binding.participant_count; ++rebound_index) {
      if (rebound_matched[rebound_index] ||
          !same_resident_motor_trajectory_contributor(
              trajectory->contributors[frozen_index],
              action_links[binding.participant_offset + rebound_index], brain,
              frontier))
        continue;
      if (match != kInvalidIndex) {
        exact_bijection = false;
        break;
      }
      match = rebound_index;
    }
    if (!exact_bijection || match == kInvalidIndex) {
      exact_bijection = false;
      break;
    }
    rebound_matched[match] = true;
  }
  if (!exact_bijection) {
    atomicExch(&action_occurrences[ticket_slot].state,
               kActionOccurrenceExpired);
    unlock_ticket_slot(ticket_table_locks, ticket_slot);
    refuse_resident_motor_trajectory(trajectory, metrics);
    return false;
  }
  if (trajectory->reserved != 0u && binding.ancestry_incomplete == 0u) {
    binding.ancestry_incomplete = 1u;
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_incomplete),
                1ULL);
  }
  AsynchronousTicket ticket{};
  ticket.ticket_id = ticket_id;
  ticket.upstream_ticket_id = binding.diagnostic_upstream_ticket;
  ticket.motor_node = trajectory->motor_node;
  ticket.motor_channel = trajectory->motor_channel;
  ticket.motor_word = word;
  ticket.context_signature = context;
  ticket.emission_tick = current_tick;
  ticket_table[ticket_slot] = ticket;
  unlock_ticket_slot(ticket_table_locks, ticket_slot);
  event->ticket_id = ticket_id;
  event->upstream_ticket_id = binding.diagnostic_upstream_ticket;
  event->node = trajectory->motor_node;
  event->channel = trajectory->motor_channel;
  event->word = word;
  event->context = context;
  event->timestamp = current_tick;
  event->reserved = binding.ancestry_incomplete;
  event->trajectory =
      resident_motor_trajectory_artifact(*trajectory, brain, frontier);
  advance_resident_motor_trajectory(trajectory, metrics, true);
  return true;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_OPS_CUH

#if defined(HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL) && \
    !defined(HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL_CUH)
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL_CUH

__global__ void emit_motor_events_kernel(
    DirectBrain brain,
    const DirectNode* nodes,
    const DirectRoute* routes,
    const std::uint64_t* route_incarnations,
    std::uint32_t route_capacity,
    const DirectBoundaryPort* ports,
    const ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    std::uint32_t require_exact_occurrence_identity,
    std::uint32_t port_count,
    MotorEvent* egress_queue,
    const std::uint32_t* egress_head,
    std::uint32_t* egress_tail,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count,
    std::uint32_t* ticket_table_locks,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    ResidentPublicMotorTrajectory* resident_motor_trajectory,
    std::uint32_t resident_motor_trajectory_capacity,
    direct_network::DirectAffectBodyState* affect_body_state,
    std::uint32_t action_horizon_ticks,
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const std::uint32_t* phase_ancestry_incomplete,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations,
    ResidentDevelopmentState* development,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick,
    DirectEfferenceCopy* efference_ring,
    std::uint32_t* efference_head,
    std::uint32_t* efference_tail,
    std::int32_t* node_incoming_excitation,
    std::uint32_t route_efference_copies) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  MotorRootChannelMemo root_memo;
  direct_network::DirectExactHistoryHotPage* history =
      development != nullptr ? &development->exact_history : nullptr;
  if (affect_body_state != nullptr && history != nullptr)
    direct_network::affect_derive_from_ledgers(
        affect_body_state, ticket_table, kMaxAsynchronousTickets,
        history->records, history->committed_slots, 0u, kInvalidIndex);
  if (history != nullptr &&
      !direct_network::begin_exact_history_phase(
          history, direct_network::DirectExactHistoryKind::motor_output,
          port_count, current_tick)) return;
  const std::uint32_t tail_base = *egress_tail;
  const std::uint32_t head = egress_head != nullptr ? *egress_head : tail_base;
  std::uint32_t emitted = 0u;
  std::uint32_t copies_routed = 0u;
  if (resident_motor_trajectory != nullptr &&
      resident_motor_trajectory->state ==
          kResidentPublicMotorTrajectoryLive &&
      tail_base - head < kMaxEgressQueueSize) {
    std::uint32_t shadow_target = direct_network::kInvalidIndex;
    bool efference_ready = true;
    if (route_efference_copies != 0u) {
      efference_ready =
          efference_ring != nullptr && efference_head != nullptr &&
          efference_tail != nullptr && node_incoming_excitation != nullptr &&
          *efference_tail - *efference_head < kMaxEfferenceRingSize;
      if (efference_ready) {
        shadow_target = direct_network::efference_shadow_target(
            nodes, routes, nodes[resident_motor_trajectory->motor_node]);
        efference_ready = shadow_target != direct_network::kInvalidIndex;
      }
    }
    MotorEvent continuation{};
    if (efference_ready && continue_resident_motor_trajectory(
            resident_motor_trajectory, brain, ports, port_count,
            actual_frontier, current_tick, action_horizon_ticks, ticket_table,
            ticket_count, ticket_table_locks, action_occurrences,
            action_participation_links, action_participant_capacity, metrics,
            activation_plane, eligibility_table,
            eligibility_record_generations, &continuation)) {
      direct_network::stage_motor_history_record(
          history != nullptr ? &history->records[history->phase_base] : nullptr,
          continuation.ticket_id, continuation.upstream_ticket_id, current_tick,
          continuation.node, continuation.channel, continuation.word,
          continuation.context, continuation.reserved);
      if (brain.postbirth_constructor != nullptr &&
          brain.postbirth_derivations != nullptr) {
        direct_network::resident_register_motor_ground_surfaces(
            brain.postbirth_constructor, brain.boundary_ports,
            brain.boundary_port_count, brain.postbirth_derivations,
            brain.postbirth_constructor->derivation_count, continuation.word,
            continuation.trajectory.result_composition_depth);
      }
      egress_queue[tail_base % kMaxEgressQueueSize] = continuation;
      if (route_efference_copies != 0u) {
        DirectEfferenceCopy copy{};
        copy.ticket_id = continuation.ticket_id;
        copy.acting_node = continuation.node;
        copy.shadow_node = shadow_target;
        copy.channel = continuation.channel;
        copy.word = continuation.word;
        copy.timestamp = current_tick;
        efference_ring[*efference_tail % kMaxEfferenceRingSize] = copy;
        atomicAdd(&node_incoming_excitation[shadow_target], kQ16One / 8);
        ++copies_routed;
      }
      emitted = 1u;
    }
    if (emitted != 0u) {
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->motor_events_emitted), 1ULL);
      *egress_tail = tail_base +
          (history != nullptr
               ? direct_network::finish_exact_history_phase(history)
               : 1u);
      if (efference_tail != nullptr) *efference_tail += copies_routed;
      return;
    }
    // A live continuation owns this ordinary tick even when ticket/efference
    // capacity temporarily refuses publication. Do not originate another
    // action behind it or reset its future causal selection.
    if (resident_motor_trajectory->state ==
        kResidentPublicMotorTrajectoryLive) {
      *egress_tail = tail_base +
          (history != nullptr
               ? direct_network::finish_exact_history_phase(history)
               : 0u);
      return;
    }
  }
  for (std::uint32_t p = 0u; p < port_count; ++p) {
    const DirectBoundaryPort port = ports[p];
    if (!(port.role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::motor))) continue;
    const DirectNode node = nodes[port.node];
    if (node.activation_q16 <= 0 ||
        tail_base + emitted - head >= kMaxEgressQueueSize) continue;
    if (!affect_motor_candidate_wins_local_competition(
            nodes, ports, port_count, p, root_memo, affect_body_state,
            routes, route_incarnations, route_capacity,
            history != nullptr ? history->records : nullptr,
            history != nullptr ? history->committed_slots : 0u,
            current_contributions, current_contribution_count,
            current_contribution_capacity, current_tick)) continue;
    const auto wanting_liking_gate =
        direct_network::resident_wanting_liking_motor_gate(
            current_contributions,
            current_contribution_count != nullptr ? *current_contribution_count : 0u,
            port.node, affect_body_state);
    std::uint32_t affect_root = kInvalidIndex;
    (void)resolve_motor_root_channel(
        root_memo, history != nullptr ? history->records : nullptr,
        history != nullptr ? history->committed_slots : 0u,
        current_contributions, current_contribution_count,
        current_contribution_capacity, port.node, current_tick, &affect_root);
    direct_network::DirectAffectCandidate affect_candidate{};
    affect_candidate.root_channel = affect_root;
    affect_candidate.base_valuation_q16 =
        clamp_q16(node.activation_q16, 0, kQ16One);
    const direct_network::DirectAffectValuationReceipt affect_receipt =
        direct_network::affect_modulate_candidate(affect_body_state,
                                                  affect_candidate);
    const auto uncertainty = resident_motor_outcome_uncertainty(
        ticket_table, kMaxAsynchronousTickets, port.node);
    const std::int32_t affect_base_threshold_q16 =
        affect_preservation_base_threshold_q16(
            wanting_liking_gate.activation_threshold_q16,
            affect_receipt.bias_q16, wanting_liking_gate.live_pursuit);
    const std::int32_t pursuit_threshold_q16 =
        resident_uncertainty_pursuit_threshold_q16(
            affect_base_threshold_q16, uncertainty.signal_q16);
    if (affect_receipt.modulated_valuation_q16 <= pursuit_threshold_q16) continue;
    std::uint32_t shadow_target = direct_network::kInvalidIndex;
    if (route_efference_copies != 0u) {
      if (efference_ring == nullptr || efference_head == nullptr ||
          efference_tail == nullptr || node_incoming_excitation == nullptr ||
          *efference_tail + copies_routed - *efference_head >=
              kMaxEfferenceRingSize)
        continue;
      shadow_target =
          direct_network::efference_shadow_target(nodes, routes, node);
      if (shadow_target == direct_network::kInvalidIndex) continue;
    }
    std::uint32_t ticket_idx = kInvalidIndex;
    for (std::uint32_t attempt = 0u; attempt < kMaxAsynchronousTickets; ++attempt) {
      const std::uint32_t candidate =
          atomicAdd(ticket_count, 1u) % kMaxAsynchronousTickets;
      if (lock_ticket_slot_for_motor(
              ticket_table_locks, ticket_table, action_occurrences,
              action_participation_links, action_participant_capacity,
              development, candidate, current_tick, action_horizon_ticks,
              metrics)) {
        ticket_idx = candidate;
        break;
      }
    }
    if (ticket_idx == kInvalidIndex) continue;
    const std::uint64_t tid = (static_cast<std::uint64_t>(current_tick) << 32) |
                              (port.channel << 16) | ticket_idx;

    AsynchronousTicket ticket{};
    ticket.ticket_id = tid;
    ticket.upstream_ticket_id = kInvalidTicket;
    ticket.motor_node = port.node;
    ticket.motor_channel = port.channel;
    ticket.context_signature = (port.node * 1103515245U) ^ current_tick;
    ticket.emission_tick = current_tick;
    ticket.settled = 0u;
    ticket.settled_reward_q16 = 0;
    ticket.mismatch_bits = 0u;
    DirectActionBindingResult ancestry = bind_action_occurrence(
        current_contributions, current_contribution_count,
        current_contribution_capacity, actual_frontier,
        require_exact_occurrence_identity, port.node, current_tick, tid,
        ticket.context_signature, port.channel,
        current_tick + action_horizon_ticks, ticket_idx,
        action_occurrences, action_participation_links,
        action_participant_capacity, metrics, nullptr, brain,
        eligibility_table, eligibility_record_generations);
    if (ancestry.admitted == 0u) {
      unlock_ticket_slot(ticket_table_locks, ticket_idx);
      continue;
    }
    if (phase_ancestry_incomplete != nullptr &&
        phase_ancestry_incomplete[port.node] != 0u &&
        ancestry.ancestry_incomplete == 0u) {
      ancestry.ancestry_incomplete = 1u;
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->action_ancestry_incomplete),
                  1ULL);
    }
    ticket.upstream_ticket_id = ancestry.diagnostic_upstream_ticket;
    const bool trajectory_required =
        require_exact_occurrence_identity != 0u &&
        resident_motor_trajectory_required(
            brain, actual_frontier, action_occurrences[ticket_idx],
            action_participation_links, current_tick);
    ResidentPublicMotorTrajectoryArtifact trajectory_artifact{};
    if (trajectory_required) {
      if (!begin_resident_motor_trajectory(
              resident_motor_trajectory, brain, actual_frontier,
              action_occurrences[ticket_idx], action_participation_links,
              port, p,
              resident_motor_trajectory != nullptr
                  ? resident_motor_trajectory_capacity
                  : 0u,
              tid, ancestry.ancestry_incomplete, metrics) ||
          !resident_motor_trajectory_word(
              *resident_motor_trajectory, brain, actual_frontier, current_tick,
              &ticket.motor_word)) {
        atomicExch(&action_occurrences[ticket_idx].state,
                   kActionOccurrenceExpired);
        unlock_ticket_slot(ticket_table_locks, ticket_idx);
        continue;
      }
      trajectory_artifact =
          resident_motor_trajectory_artifact(
              *resident_motor_trajectory, brain, actual_frontier);
    } else {
      // Public expression is a current actual occurrence, never history replay.
      ticket.motor_word = resident_bound_action_motor_word(
          brain, node, routes, route_capacity, bounded_route_scan_count(node),
          actual_frontier, action_occurrences, action_participation_links,
          ancestry, ticket_idx, action_participant_capacity, tid, port.node,
          port.channel, ticket.context_signature, current_tick);
    }
    ticket_table[ticket_idx] = ticket;
    unlock_ticket_slot(ticket_table_locks, ticket_idx);

    const std::uint32_t slot = (tail_base + emitted) % kMaxEgressQueueSize;
    MotorEvent mevt{};
    mevt.ticket_id = tid;
    mevt.upstream_ticket_id = ancestry.diagnostic_upstream_ticket;
    mevt.node = port.node;
    mevt.channel = port.channel;
    mevt.word = ticket.motor_word;
    mevt.context = ticket.context_signature;
    mevt.timestamp = current_tick;
    mevt.reserved = ancestry.ancestry_incomplete;
    mevt.trajectory = trajectory_artifact;
    direct_network::stage_motor_history_record(
        history != nullptr ? &history->records[history->phase_base + p] : nullptr,
        mevt.ticket_id, mevt.upstream_ticket_id, current_tick, mevt.node,
        mevt.channel, mevt.word, mevt.context, mevt.reserved);
    if (brain.postbirth_constructor != nullptr &&
        brain.postbirth_derivations != nullptr) {
      direct_network::resident_register_motor_ground_surfaces(
          brain.postbirth_constructor, brain.boundary_ports,
          brain.boundary_port_count, brain.postbirth_derivations,
          brain.postbirth_constructor->derivation_count, ticket.motor_word,
          trajectory_artifact.result_composition_depth);
    }
    egress_queue[slot] = mevt;
    if (route_efference_copies != 0u) {
      DirectEfferenceCopy copy{};
      copy.ticket_id = tid;
      copy.acting_node = port.node;
      copy.shadow_node = shadow_target;
      copy.channel = port.channel;
      copy.word = ticket.motor_word;
      copy.timestamp = current_tick;
      efference_ring[(*efference_tail + copies_routed) %
                     kMaxEfferenceRingSize] = copy;
      atomicAdd(&node_incoming_excitation[shadow_target], kQ16One / 8);
      ++copies_routed;
    }
    ++emitted;
    if (trajectory_required) {
      advance_resident_motor_trajectory(
          resident_motor_trajectory, metrics, false);
      if (resident_motor_trajectory->state ==
          kResidentPublicMotorTrajectoryLive)
        break;
    }
  }
  if (emitted != 0u)
    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->motor_events_emitted),
              static_cast<unsigned long long>(emitted));
  *egress_tail = tail_base +
      (history != nullptr ? direct_network::finish_exact_history_phase(history) : emitted);
  if (efference_tail != nullptr) *efference_tail += copies_routed;
}
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_MOTOR_TRAJECTORY_STEPPED_KERNEL_CUH
