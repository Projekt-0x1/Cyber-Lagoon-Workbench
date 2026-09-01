#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CORE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CORE_CUH
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_network_resident_development.cuh"
#include "hardware_native/direct_adult_core_constants.cuh"
#include "hardware_native/direct_adult_contact_receipt.cuh"
struct CUstream_st;  // gh #1305: global scope keeps this header CUDA-free.
namespace substrate::direct_network {
struct DirectAffectBodyState;
struct DirectCausalWorldModel;
struct DirectResidentLanguageRuntimeBlock;
struct ResidentWantingLikingProfileV1;
}
namespace substrate::direct_adult_core {
struct DirectAdultActionControlRuntimeBlock;
using direct_network::DirectBrain;
using direct_network::DirectNode;
using direct_network::DirectRoute;
using direct_network::DirectDenseBlock;
using direct_network::DirectBoundaryPort;
using direct_network::DirectEfferenceCopy;
using direct_network::ResidentDevelopmentState;
using direct_network::ResidentDevelopmentWorkspace;
using direct_network::Root256;
inline constexpr std::uint32_t kEligibilityClaimDirectoryCapacity =
    2u * kMaxLiveEligibilityRecords;
inline constexpr std::uint32_t kEligibilityClaimLockCount = 4096u;
inline constexpr std::uint32_t kEligibilityClaimMaxProbes = 64u;
inline constexpr std::uint32_t kMaxEgressQueueSize = 4096u;
inline constexpr std::uint32_t kMaxEfferenceRingSize = 4096u;
inline constexpr std::uint32_t kMaxDelayedSparsePackets = 32768u;
inline constexpr std::uint32_t kMaxPhysicalRouteDelayTicks = 64u;
inline constexpr std::uint32_t kDelayedPacketIdentityCapacity =
    2u * kMaxDelayedSparsePackets;
inline constexpr std::uint32_t kDelayedPacketIdentityMaxProbes = 64u;
inline constexpr std::uint32_t kRouteTransportOwnerScratchCapacity =
    kEligibilityClaimDirectoryCapacity > kMaxDelayedSparsePackets
        ? kEligibilityClaimDirectoryCapacity
        : kMaxDelayedSparsePackets;
inline constexpr std::uint32_t kMaxProvenanceSlotsPerNode = 4u;
// #1610 / graph law: node-local participation residency aperture is distinct
// from provenance / producer-width matter. N=256 was capacity-falsified
// (uncapped demand peak 323); production must configure this aperture at 512.
// Do not implement by widening kMaxProvenanceSlotsPerNode.
inline constexpr std::uint32_t kNodeParticipationAperture = 512u;
inline constexpr std::uint32_t kRouteTransportProducerWidth =
    kMaxProvenanceSlotsPerNode;
inline constexpr std::uint32_t kMaxActionParticipationLinks =
    direct_network::kMaxResidentActionParticipationLinks;
inline constexpr std::uint32_t kMaxBasins = 8u;
enum class CausalOrigin : std::uint32_t {
  external_contact = 0u,
  world_return = 1u,
  endogenous_prediction = 2u,
  motor_reafference = 3u,
};
enum class AdultExecutionAuthority : std::uint32_t {
  host_stepped = 0u,
  starting = 1u,
  persistent = 2u,
  stopping = 3u,
};
enum class DirectParticipationAuthority : std::uint32_t {
  none = 0u, independent_external = 1u, resident_external_descendant = 2u};
DIRECT_ADULT_HD inline constexpr bool resident_occurrence_accepts_causal_authority(
    DirectParticipationAuthority occurrence_authority,
    DirectParticipationAuthority causal_authority) {
  return occurrence_authority == causal_authority ||
         (occurrence_authority ==
              DirectParticipationAuthority::resident_external_descendant &&
          causal_authority ==
              DirectParticipationAuthority::independent_external);
}
enum class DirectContributionKind : std::uint32_t {
  none = 0u, direct_ingress = 1u, sparse_route = 2u,
  ancestry_incomplete = 3u};

DIRECT_ADULT_HD inline bool is_external_evidence(CausalOrigin origin) {
  return origin == CausalOrigin::external_contact || origin == CausalOrigin::world_return ||
         origin == CausalOrigin::motor_reafference;
}

struct alignas(16) NodeCausalParticipation {
  std::uint64_t ticket_id;
  std::uint32_t expiry_tick;
  std::uint32_t last_refresh_tick;
  DirectParticipationAuthority authority;
  std::uint32_t commit_generation;
  std::uint32_t authority_incarnation;
  std::uint32_t claim_incarnation;
  std::uint32_t current_drive;
  std::uint32_t reserved0, reserved1, reserved2;
};
static_assert(sizeof(NodeCausalParticipation) == 48 && std::is_standard_layout_v<NodeCausalParticipation> && std::is_trivial_v<NodeCausalParticipation>);
static_assert(std::has_unique_object_representations_v<NodeCausalParticipation>);
struct alignas(8) DirectParticipationDescriptor {
  std::uint64_t ticket_id;
  std::uint32_t source_node, target_node, route_index, context_signature;
  std::uint32_t expiry_tick, claim_incarnation;
  std::uint64_t route_incarnation;
  DirectParticipationAuthority authority;
  std::uint32_t authority_incarnation;
  DirectContributionKind contribution_kind;
  std::uint32_t eligibility_slot, eligibility_generation;
  std::int32_t frozen_eligibility_q16;
  std::uint64_t parent_eligibility_ref;
  std::uint32_t lineage_expiry_tick;
  std::uint32_t ancestry_depth;
};
static_assert(sizeof(DirectParticipationDescriptor) == 80 && std::is_standard_layout_v<DirectParticipationDescriptor> && std::is_trivial_v<DirectParticipationDescriptor>);
static_assert(std::has_unique_object_representations_v<DirectParticipationDescriptor>);

// Resolve the one verified sensory root shared by the exact current
// participants of a prospective motor candidate. Incomplete, expired,
// over-capacity, or mixed-root participation has no modulatory authority.
DIRECT_ADULT_HD inline bool candidate_sensory_root_channel(
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count,
    const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count,
    std::uint32_t contribution_capacity, std::uint32_t motor_node,
    std::uint32_t current_tick, std::uint32_t* root_channel) {
  if (records == nullptr || contributions == nullptr ||
      contribution_count == nullptr || root_channel == nullptr)
    return false;
  const std::uint32_t staged = *contribution_count;
  if (staged > contribution_capacity) return false;
  std::uint64_t participant_tickets[kMaxProvenanceSlotsPerNode]{};
  std::uint32_t participant_count = 0u;
  bool saw_root = false;
  std::uint32_t resolved_root = 0u;
  for (std::uint32_t i = 0u; i < staged; ++i) {
    const DirectParticipationDescriptor& value = contributions[i];
    if (value.target_node != motor_node) continue;
    if (value.contribution_kind == DirectContributionKind::ancestry_incomplete ||
        (value.contribution_kind != DirectContributionKind::direct_ingress &&
         value.contribution_kind != DirectContributionKind::sparse_route) ||
        value.ticket_id == 0u || value.ticket_id == kInvalidTicket ||
        value.expiry_tick < current_tick)
      return false;
    bool duplicate = false;
    for (std::uint32_t j = 0u; j < participant_count; ++j)
      duplicate |= participant_tickets[j] == value.ticket_id;
    if (duplicate) continue;
    if (participant_count == kMaxProvenanceSlotsPerNode) return false;
    participant_tickets[participant_count++] = value.ticket_id;

    std::uint64_t pending[1u + kMaxProvenanceSlotsPerNode]{};
    std::uint32_t pending_count = 1u;
    std::uint32_t cursor = 0u;
    pending[0] = value.ticket_id;
    bool participant_root_seen = false;
    std::uint32_t participant_root = 0u;
    while (cursor < pending_count) {
      const std::uint64_t sought = pending[cursor++];
      for (std::uint32_t r = 0u; r < record_count; ++r) {
        const direct_network::DirectExactHistoryRecord& record = records[r];
        if (record.identity != sought ||
            record.kind == direct_network::DirectExactHistoryKind::empty)
          continue;
        if (record.kind == direct_network::DirectExactHistoryKind::sensory_contact) {
          if ((record.flags & direct_network::kDirectHistoryVerifiedObservation) == 0u)
            return false;
          if (!participant_root_seen) {
            participant_root = record.subject;
            participant_root_seen = true;
          } else if (participant_root != record.subject) {
            return false;
          }
          continue;
        }
        const std::uint64_t parent = record.parent_identity;
        if (parent == 0u || parent == kInvalidTicket) continue;
        bool known = false;
        for (std::uint32_t j = 0u; j < pending_count; ++j)
          known |= pending[j] == parent;
        if (known) continue;
        if (pending_count == 1u + kMaxProvenanceSlotsPerNode) return false;
        pending[pending_count++] = parent;
      }
    }
    if (!participant_root_seen) return false;
    if (!saw_root) {
      resolved_root = participant_root;
      saw_root = true;
    } else if (resolved_root != participant_root) {
      return false;
    }
  }
  if (!saw_root) return false;
  *root_channel = resolved_root;
  return true;
}

#include "hardware_native/direct_adult_motor_competition.cuh"

enum class DelayedPacketSlotState : std::uint32_t {
  free = 0u, reserved = 1u, published = 2u
};

// Packet slots use `reserved` while their bytes are being written. Identity
// entries need a distinct in-progress state so contention never becomes an
// allocator-capacity result.
enum class DelayedPacketIdentityState : std::uint32_t {
  free = 0u, claiming = 1u, published = 2u
};

// A delayed route is physical in-flight matter, not a host-side callback. The
// packet freezes the source-side causal dose needed at the due tick, including
// expired resident claims. Target coactivity is sampled only at arrival; `live`
// is the explicit FREE/RESERVED/PUBLISHED slot state.
struct alignas(8) DelayedSparsePacket {
  std::uint32_t live;
  std::uint32_t due_tick;
  std::uint32_t source_node;
  std::uint32_t target_node;
  std::uint32_t route_index;
  std::uint32_t context_signature;
  std::int32_t source_activation_q16;
  std::int32_t drive_q16;
  std::int32_t reserved_arrival_eligibility_q16;
  std::uint32_t source_ancestry_incomplete;
  std::uint32_t claim_count;
  std::uint32_t reserved0;
  std::uint64_t route_incarnation;
  DirectParticipationDescriptor source_claims[kMaxProvenanceSlotsPerNode];
};
static_assert(sizeof(DelayedSparsePacket) == 376 &&
              std::is_standard_layout_v<DelayedSparsePacket> &&
              std::is_trivial_v<DelayedSparsePacket>);
static_assert(std::has_unique_object_representations_v<DelayedSparsePacket>);
struct alignas(8) DelayedPacketIdentity {
  std::uint32_t state;
  std::uint32_t packet_index_plus_one;
  std::uint32_t route_index;
  std::uint32_t due_tick;
  std::uint64_t route_incarnation;
};
static_assert(sizeof(DelayedPacketIdentity) == 24 &&
              std::is_standard_layout_v<DelayedPacketIdentity> &&
              std::is_trivial_v<DelayedPacketIdentity> &&
              std::has_unique_object_representations_v<DelayedPacketIdentity>);
// Transient route-owned proposal and producer-bound claim work. These are
// rebuilt every epoch; accepted packet/eligibility bytes remain the durable
// checkpoint owners.
struct alignas(8) RouteTransportProposal {
  DelayedSparsePacket packet;
  std::uint32_t valid;
  std::uint32_t reserved;
};
static_assert(sizeof(RouteTransportProposal) == 384 &&
              std::is_standard_layout_v<RouteTransportProposal> &&
              std::is_trivial_v<RouteTransportProposal> &&
              std::has_unique_object_representations_v<RouteTransportProposal>);
struct alignas(8) EligibilityBatchClaim {
  DirectParticipationDescriptor descriptor;
  std::int32_t eligibility_q16;
  std::uint32_t valid;
  std::uint32_t directory_slot;
  std::uint32_t canonical_producer;
  std::uint32_t eligibility_slot;
  std::uint32_t eligibility_generation;
  std::uint32_t admitted;
  std::uint32_t chronology_tick;
  std::uint32_t physical_route_index;
};
struct RouteTransportPhaseView {
  RouteTransportProposal* proposals;
  EligibilityBatchClaim* claims;
  std::uint32_t claim_capacity;
  std::uint32_t* scan_a;
  std::uint32_t* scan_b;
  std::uint32_t scan_capacity;
  std::uint32_t* free_slots;
  std::uint32_t* eligibility_owners;
  std::uint32_t* node_candidate_owners;
  std::uint32_t* cursor;
};
inline constexpr std::uint32_t kActionOccurrenceFree = 0u;
inline constexpr std::uint32_t kActionOccurrencePending = 1u;
inline constexpr std::uint32_t kActionOccurrenceSettled = 2u;
inline constexpr std::uint32_t kActionOccurrenceExpired = 3u;
struct alignas(8) DirectActionParticipationLink {
  std::uint64_t participant_ticket_id, logical_recipe_id, revision_identity, occurrence_identity, participation_identity, occurrence_route_incarnation;
  std::uint32_t source_node, target_node, route_index, context_signature, occurrence_context_signature, composition_depth, expiry_tick, claim_incarnation;
  std::uint64_t route_incarnation;
  std::uint32_t authority_incarnation;
  DirectParticipationAuthority authority;
  DirectContributionKind contribution_kind;
  std::int32_t frozen_eligibility_q16;
  std::uint32_t eligibility_slot, eligibility_generation;
};
static_assert(sizeof(DirectActionParticipationLink) == 112 && std::is_standard_layout_v<DirectActionParticipationLink> && std::is_trivial_v<DirectActionParticipationLink>);
static_assert(std::has_unique_object_representations_v<DirectActionParticipationLink>);
struct alignas(8) DirectActionOccurrence {
  std::uint64_t action_ticket_id;
  // Freeze both the exact active closure and its compact reusable morphology.
  // Eligibility is unresolved action evidence, not persistent recruitment law.
  std::uint64_t network_identity, recruitment_identity;
  std::int64_t network_eligibility_signed_q16;
  std::uint64_t network_eligibility_l1_q16;
  std::uint32_t participant_offset, participant_count;
  std::uint32_t emission_tick, context_signature, motor_node, motor_channel;
  std::uint32_t state, expiry_tick, occurrence_identity_required, occurrence_identity_complete;
};
static_assert(sizeof(DirectActionOccurrence) == 80 && std::is_standard_layout_v<DirectActionOccurrence> && std::is_trivial_v<DirectActionOccurrence>);
static_assert(std::has_unique_object_representations_v<DirectActionOccurrence>);
#include "hardware_native/direct_adult_resident_motor_trajectory.cuh"
inline constexpr std::uint16_t kResidentRecipeOccurrenceFree = 0u;
inline constexpr std::uint16_t kResidentRecipeOccurrenceLive = 1u;
inline constexpr std::uint16_t kResidentRecipeOccurrenceSettled = 2u;
enum class ResidentOccurrenceLineageKind : std::uint16_t {
  none = 0u,
  actual = 1u,
  endogenous = 2u,
  replay = 3u,
};
struct ResidentOccurrencePortBinding {
  std::uint32_t variable_identity;
  std::uint16_t formal_port_index;
  std::uint16_t reserved;
};
struct alignas(8) ResidentRecipeOccurrence {
  std::uint64_t occurrence_identity;
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t participation_identity;
  std::uint64_t source_identity;
  std::uint64_t route_incarnation;
  // Sparse exact locator for the live causal eligibility that admitted this
  // Occurrence. The referenced record remains executor-owned and rebuildable;
  // action unfolding never scans the dormant eligibility population.
  std::uint64_t eligibility_ref;
  std::uint32_t context_signature;
  std::uint32_t timestamp;
  std::uint32_t expiry_tick;
  std::uint32_t source_incarnation;
  std::uint32_t last_solver_tick;
  std::uint32_t solver_steps;
  std::int32_t eligibility_q16;
  std::int32_t activation_q16;
  std::uint16_t binding_count;
  std::uint16_t state;
  ResidentOccurrenceLineageKind lineage_kind;
  std::uint16_t reserved;
  DirectParticipationAuthority authority;
  std::uint32_t reserved2;
  ResidentOccurrencePortBinding bindings[direct_network::kResidentDerivationWidth];
};
static_assert(std::is_standard_layout_v<ResidentRecipeOccurrence> &&
              std::is_trivial_v<ResidentRecipeOccurrence> &&
              std::has_unique_object_representations_v<ResidentRecipeOccurrence>);
DIRECT_ADULT_HD inline bool bind_resident_recipe_occurrence(
    const direct_network::ResidentRecipeCell& recipe,
    const direct_network::ResidentRecipeDerivation& derivation,
    const std::uint32_t* variable_identities, std::uint32_t binding_count,
    std::uint64_t occurrence_identity, std::uint64_t participation_identity,
    std::uint64_t source_identity, std::uint32_t source_incarnation,
    ResidentOccurrenceLineageKind lineage_kind,
    DirectParticipationAuthority authority,
    std::uint32_t context_signature, std::uint32_t timestamp,
    std::uint32_t expiry_tick, std::int32_t eligibility_q16,
    ResidentRecipeOccurrence* out) {
  if (out == nullptr || variable_identities == nullptr ||
      occurrence_identity == 0u || participation_identity == 0u ||
      source_identity == 0u || source_incarnation == 0u ||
      lineage_kind == ResidentOccurrenceLineageKind::none ||
      recipe.logical_recipe_id == 0u || recipe.revision_identity == 0u ||
      recipe.logical_recipe_id != derivation.logical_recipe_id ||
      binding_count == 0u || binding_count != derivation.port_count ||
      binding_count > direct_network::kResidentDerivationWidth ||
      expiry_tick < timestamp)
    return false;
  const bool actual = lineage_kind == ResidentOccurrenceLineageKind::actual;
  if ((actual && authority == DirectParticipationAuthority::none) ||
      (!actual && authority != DirectParticipationAuthority::none))
    return false;
  ResidentRecipeOccurrence occurrence{};
  occurrence.occurrence_identity = occurrence_identity;
  occurrence.logical_recipe_id = recipe.logical_recipe_id;
  occurrence.revision_identity = recipe.revision_identity;
  occurrence.participation_identity = participation_identity;
  occurrence.source_identity = source_identity;
  occurrence.context_signature = context_signature;
  occurrence.timestamp = timestamp;
  occurrence.expiry_tick = expiry_tick;
  occurrence.source_incarnation = source_incarnation;
  occurrence.last_solver_tick = timestamp;
  occurrence.eligibility_q16 = eligibility_q16;
  occurrence.binding_count = static_cast<std::uint16_t>(binding_count);
  occurrence.state = kResidentRecipeOccurrenceLive;
  occurrence.lineage_kind = lineage_kind;
  occurrence.authority = authority;
  for (std::uint32_t i = 0u; i < binding_count; ++i) {
    if (variable_identities[i] == 0u) return false;
    occurrence.bindings[i] = ResidentOccurrencePortBinding{
        variable_identities[i], static_cast<std::uint16_t>(i), 0u};
  }
  *out = occurrence;
  return true;
}
DIRECT_ADULT_HD inline bool apply_resident_occurrence_bound_activation(
    ResidentRecipeOccurrence* occurrence,
    std::uint64_t expected_occurrence_identity,
    std::uint32_t expected_context_signature,
    const std::uint32_t* expected_variable_identities,
    std::uint32_t expected_binding_count, std::uint32_t tick,
    std::int32_t activation_q16) {
  if (occurrence == nullptr || expected_variable_identities == nullptr ||
      expected_occurrence_identity == 0u ||
      occurrence->occurrence_identity != expected_occurrence_identity ||
      occurrence->context_signature != expected_context_signature ||
      occurrence->state != kResidentRecipeOccurrenceLive ||
      expected_binding_count == 0u ||
      expected_binding_count != occurrence->binding_count ||
      tick < occurrence->timestamp || tick > occurrence->expiry_tick ||
      tick < occurrence->last_solver_tick)
    return false;
  for (std::uint32_t i = 0u; i < expected_binding_count; ++i)
    if (occurrence->bindings[i].formal_port_index != i ||
        occurrence->bindings[i].variable_identity !=
            expected_variable_identities[i])
      return false;
  occurrence->activation_q16 = activation_q16;
  occurrence->last_solver_tick = tick;
  ++occurrence->solver_steps;
  return true;
}
struct alignas(8) ResidentOccurrenceCoupling {
  std::uint64_t source_occurrence_identity;
  std::uint64_t target_occurrence_identity;
  std::uint64_t source_revision_identity;
  std::uint64_t target_revision_identity;
  std::uint64_t source_derivation_rank;
  std::uint64_t target_derivation_rank;
  std::uint64_t source_identity;
  std::uint64_t target_identity;
  std::uint64_t source_route_incarnation;
  std::uint64_t target_route_incarnation;
  std::uint32_t variable_identity;
  std::uint32_t source_incarnation;
  std::uint32_t target_incarnation;
  std::uint16_t source_port_index;
  std::uint16_t target_port_index;
};
static_assert(std::is_standard_layout_v<ResidentOccurrenceCoupling> &&
              std::is_trivial_v<ResidentOccurrenceCoupling> &&
              std::has_unique_object_representations_v<ResidentOccurrenceCoupling>);
DIRECT_ADULT_HD inline bool bind_resident_occurrence_coupling(
    const ResidentRecipeOccurrence& source,
    const direct_network::ResidentRecipeDerivation& source_derivation,
    std::uint16_t source_port_index,
    const ResidentRecipeOccurrence& target,
    const direct_network::ResidentRecipeDerivation& target_derivation,
    std::uint16_t target_port_index, ResidentOccurrenceCoupling* out) {
  if (out == nullptr || source.state != kResidentRecipeOccurrenceLive ||
      target.state != kResidentRecipeOccurrenceLive ||
      source.occurrence_identity == 0u || target.occurrence_identity == 0u ||
      source.occurrence_identity == target.occurrence_identity ||
      source.logical_recipe_id != source_derivation.logical_recipe_id ||
      source.revision_identity != source_derivation.revision_identity ||
      target.logical_recipe_id != target_derivation.logical_recipe_id ||
      target.revision_identity != target_derivation.revision_identity ||
      source_port_index >= source.binding_count ||
      source_port_index >= source_derivation.port_count ||
      target_port_index >= target.binding_count ||
      target_port_index >= target_derivation.port_count)
    return false;
  const ResidentOccurrencePortBinding& source_binding =
      source.bindings[source_port_index];
  const ResidentOccurrencePortBinding& target_binding =
      target.bindings[target_port_index];
  if (source_binding.formal_port_index != source_port_index ||
      target_binding.formal_port_index != target_port_index ||
      source_binding.variable_identity == 0u ||
      source_binding.variable_identity != target_binding.variable_identity ||
      !direct_network::resident_recipe_ports_compatible(
          source_derivation.ports[source_port_index],
          target_derivation.ports[target_port_index]))
    return false;
  *out = ResidentOccurrenceCoupling{
      source.occurrence_identity, target.occurrence_identity,
      source.revision_identity, target.revision_identity,
      source_derivation.generation, target_derivation.generation,
      source.source_identity, target.source_identity,
      source.route_incarnation, target.route_incarnation,
      source_binding.variable_identity, source.source_incarnation,
      target.source_incarnation, source_port_index, target_port_index};
  return true;
}
struct alignas(16) IngressRingControl {
  std::uint32_t consumed_head;
  std::uint32_t published_tail;
  std::uint32_t capacity;
  std::uint32_t fault;
};
static_assert(sizeof(IngressRingControl) == 16 && std::is_standard_layout_v<IngressRingControl> && std::is_trivial_v<IngressRingControl>);
static_assert(std::has_unique_object_representations_v<IngressRingControl>);
struct alignas(8) ConsequenceIngressEvent {
  std::uint64_t ticket_id;
  Word returned_word;
  std::uint32_t admission_tick;
  CausalOrigin origin;
  std::uint32_t reserved;
};
static_assert(
    sizeof(ConsequenceIngressEvent) == 24 &&
    std::is_standard_layout_v<ConsequenceIngressEvent> &&
    std::is_trivial_v<ConsequenceIngressEvent> &&
    std::has_unique_object_representations_v<ConsequenceIngressEvent>);
static_assert(
    alignof(ConsequenceIngressEvent) == 8);
struct alignas(8) ActivityEvent {
  std::uint64_t ticket_id;
  std::uint32_t node;
  std::uint32_t channel;
  Word word;
  CausalOrigin origin;
  std::uint32_t context;
  std::uint32_t timestamp;
};
static_assert(sizeof(ActivityEvent) == kAdultPacketResourceBytesPerUnit &&
              std::is_standard_layout_v<ActivityEvent> &&
              std::is_trivial_v<ActivityEvent>);
static_assert(std::has_unique_object_representations_v<ActivityEvent>);
struct ResidentActualFrontier;
struct ResidentActivationSoaPlane;
struct ResidentMultiHorizonPredictionFrontier;
struct ResidentMismatchOmissionRuntime;
struct ResidentMismatchOmissionFrontier;
struct alignas(8) MotorEvent {
  std::uint64_t ticket_id;
  std::uint64_t upstream_ticket_id;
  std::uint32_t node;
  std::uint32_t channel;
  Word word;
  std::uint32_t context;
  std::uint32_t timestamp;
  std::uint32_t reserved;
  ResidentPublicMotorTrajectoryArtifact trajectory;
};
static_assert(sizeof(MotorEvent) == 272 && std::is_standard_layout_v<MotorEvent> && std::is_trivial_v<MotorEvent>);
static_assert(std::has_unique_object_representations_v<MotorEvent>);
struct alignas(8) EligibilityRecord {
  std::uint64_t ticket_id;
  std::uint32_t source_node, target_node;
  std::uint32_t route_index, context_signature;
  std::int32_t eligibility_q16;
  std::uint32_t expiry_tick, live, claim_incarnation;
  std::uint64_t route_incarnation;
  DirectParticipationAuthority authority;
  std::uint32_t authority_incarnation;
  std::uint64_t parent_eligibility_ref;
  std::uint32_t lineage_expiry_tick;
  std::uint32_t ancestry_depth;
};
static_assert(sizeof(EligibilityRecord) == kAdultEligibilityResourceBytesPerUnit &&
              std::is_standard_layout_v<EligibilityRecord> &&
              std::is_trivial_v<EligibilityRecord>);
static_assert(std::has_unique_object_representations_v<EligibilityRecord>);
struct alignas(8) AsynchronousTicket {
  std::uint64_t ticket_id, upstream_ticket_id;
  std::uint32_t motor_node, motor_channel;
  Word motor_word;
  std::uint32_t context_signature, emission_tick, settled;
  std::int32_t settled_reward_q16;
  std::uint32_t mismatch_bits;
};
static_assert(sizeof(AsynchronousTicket) == kAdultTicketResourceBytesPerUnit &&
              std::is_standard_layout_v<AsynchronousTicket> &&
              std::is_trivial_v<AsynchronousTicket>);
static_assert(std::has_unique_object_representations_v<AsynchronousTicket>);
struct alignas(8) ResolvedConsequenceContext {
  std::uint64_t target_ticket_id, upstream_ticket_id;
  std::int32_t effective_reward_q16;
  std::uint32_t mismatch_bits, valid, action_bound;
  std::uint32_t action_participant_offset, action_participant_count;
  std::uint32_t ancestry_incomplete, history_refused;
};
static_assert(sizeof(ResolvedConsequenceContext) == 48 && std::is_standard_layout_v<ResolvedConsequenceContext> && std::is_trivial_v<ResolvedConsequenceContext>);
static_assert(std::has_unique_object_representations_v<ResolvedConsequenceContext>);
struct AttractorBasinState {  // Legacy observer; never feeds activation or learning.
  std::int32_t basin_energy_q16[kMaxBasins];
  std::int32_t basin_stability_q16[kMaxBasins];
  std::uint32_t active_coalition_nodes[kMaxBasins];
  std::uint32_t basin_phase[kMaxBasins];
};
static_assert(sizeof(AttractorBasinState) == 128 && std::is_standard_layout_v<AttractorBasinState> && std::is_trivial_v<AttractorBasinState>);
static_assert(std::has_unique_object_representations_v<AttractorBasinState>);
struct alignas(8) DirectActualFrontierCausalCounters {
  std::uint64_t deferred_contacts;
  std::uint64_t valid_exact_candidates;
  std::uint64_t valid_wildcard_candidates;
  std::uint64_t exact_claim_matches;
  std::uint64_t pending_admissions;
  std::uint64_t ambiguity_rejects;
  std::uint64_t stale_currentness_rejects;
  std::uint64_t occurrence_contributions_staged;
};
static_assert(sizeof(DirectActualFrontierCausalCounters) == 64 &&
              std::is_standard_layout_v<DirectActualFrontierCausalCounters> &&
              std::is_trivial_v<DirectActualFrontierCausalCounters> &&
              std::has_unique_object_representations_v<DirectActualFrontierCausalCounters>);
struct AdultCoreMetrics {
  std::uint64_t tick_count, sensory_events_ingested, sensory_boundary_rejects;
  std::uint64_t ingress_overflow_drops, provenance_capacity_drops, motor_events_emitted;
  std::uint64_t predictive_shadows_evaluated, consequences_assimilated;
  std::uint64_t consequence_unknown_ticket_rejects, consequence_duplicate_ticket_rejects;
  std::uint64_t credit_updates_committed, eligibility_capacity_rejects;
  std::uint64_t dense_wmma_tiles_executed, dense_scalar_tiles_executed;
  std::uint64_t maintenance_energy_debited, routes_retracted_scarcity;
  std::uint64_t dense_crystallizations, dense_shatters;
  std::int64_t total_positive_credit_q16, total_negative_credit_q16;
  std::uint64_t queue_wraparounds;
  // #1178 rung 13. Retractions the sweep DECLINED because a live eligibility
  // record still named the route, and distinct routes marked pinned. Both
  // accumulate across sweeps like every other counter here, so on a 16-sweep run
  // that protects 32776 routes both read 524416, not 32776 -- they are work
  // counters, not matter counts. The matter side is `deferred_units` on the
  // explicit_interaction pool, which is what "postponed rather than performed or
  // refused" means in the ABI.
  std::uint64_t retractions_deferred_causal_pin, causally_pinned_routes;
  // #1178 rung 14. The exact directory may reclaim an expired record but must
  // never overwrite a live unexpired one. Keep the two outcomes separate: the
  // first is lawful reuse, the second is causal loss and must remain zero.
  // `eligibility_capacity_rejects` records explicit finite-matter pressure.
  std::uint64_t eligibility_expired_records_reclaimed, eligibility_live_records_evicted;
  // #1178 rung 15. Incumbent participations displaced to make room for an
  // arriving ticket, and arrivals refused because every slot was refreshed on
  // the current tick. Kept separate from
  // `provenance_capacity_drops` for the same reason the two eligibility counters
  // are separate: a displaced incumbent and a refused newcomer are opposite
  // outcomes and must never share a number.
  std::uint64_t provenance_incumbents_evicted, provenance_no_evictable_slot_drops;
  // gh #1343. Global producer-chain instrumentation for the D0-D5 discriminator
  // ladder ("does a fresh ticket ever mint an eligibility record, and if not,
  // where does it disappear?"). Deliberately global counters, not filtered to
  // a "ticket of interest" -- resident cognition never sees an observer tag.
  // A fixture that injects exactly one ticket into an otherwise idle adult
  // dose-matches these totals to that ticket's own path without adding any
  // semantic input to the kernels.
  std::uint64_t sensory_participation_admitted;   // D0: insert_active_participation succeeded at ingest
  std::uint64_t active_source_nodes_visited;      // D1: propagate thread's src_node.activation_q16 > 0
  std::uint64_t active_routes_visited;            // D2: route passed kRouteFlagActive
  std::uint64_t participation_slots_live_seen;    // D3: unexpired ticket found in source's provenance slot
  std::uint64_t eligibility_threshold_rejects;    // D4: new_elig <= kQ16One/8
  std::uint64_t eligibility_record_write_attempts; // D5: new_elig > kQ16One/8, exact admission attempted
  std::uint64_t eligibility_records_written;      // D5: record actually stored (attempt not capacity-rejected)
  DirectActualFrontierCausalCounters actual_frontier_causal; // D6-D11 observer-only actual-work ladder
  std::uint64_t participation_descriptors_staged, participation_staging_overflow; // #1337 per-tick scratch
  std::uint64_t action_occurrences_committed, action_participant_links_committed;
  std::uint64_t action_ancestry_backpressure;
  std::uint64_t authorized_participation_refreshes;
  std::uint64_t participation_authority_collision_rejects;
  std::uint64_t action_ticket_collision_rejects;
  std::uint64_t action_expired_rejects, action_occurrences_expired;
  std::uint64_t action_ancestry_incomplete;
  std::uint64_t stale_route_claim_rejects, raw_reafferent_authority_rejects;
  std::uint64_t delayed_packets_scheduled, delayed_packets_delivered;
  std::uint64_t delayed_packets_stale_rejected, delayed_packets_overflow_rejections;
  std::uint64_t delayed_packets_pending_peak, delayed_packets_duplicate_rejections;
  std::uint64_t motor_trajectory_origins, motor_trajectory_continuations;
  std::uint64_t motor_trajectory_completions, motor_trajectory_capacity_refusals;
  std::uint64_t motor_trajectory_stale_refusals;
  std::uint64_t active_composition_depth_peak, composition_depth_refusals;
  // #1610 named N=512 participation envelope (host-readable; observer-only for
  // uncapped demand). Distinct from provenance_no_evictable_slot_drops /
  // delayed_packets_pending_peak proxies.
  std::uint64_t node_participation_uncapped_demand_peak;
  std::uint64_t node_participation_capped_resident_peak;
  std::uint64_t node_participation_no_free_drops;
};
static_assert(sizeof(AdultCoreMetrics) == 568 && std::is_standard_layout_v<AdultCoreMetrics> && std::is_trivial_v<AdultCoreMetrics>);
static_assert(std::has_unique_object_representations_v<AdultCoreMetrics>);

struct AdultCoreObserverSnapshot {
  AdultCoreMetrics metrics{};
  std::uint32_t resident_tick = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t request_id = 0u;
  std::uint64_t completion_id = 0u;
};

struct AdultExecutionConfig {
  std::int32_t learning_rate_q16 = 1 << 13, eligibility_decay_q16 = 62259;
  std::uint32_t eligibility_horizon_ticks = 32u;
  std::int32_t maintenance_cost_per_route_q16 = 4, maintenance_cost_per_dense_tile_q16 = 64;
  std::int32_t vitality_budget_per_epoch_q16 = 1 << 20, attractor_coupling_gain_q16 = 1 << 14;
  std::uint32_t refractory_period_ticks = 2u;
  // Exact bounded ancestry links reserved per emitted action occurrence. Four
  // physical terminal claims may each contribute a four-edge rooted closure.
  std::uint32_t action_participant_capacity = kMaxActionParticipationLinks;
  std::int32_t dense_shatter_threshold_q16 = -(1 << 14);
  std::uint32_t block_dim = 256u;
  // Host transport may page completed exact-history hot pages, but it gets no
  // authority over their contents or resident timing. The capacity is a real,
  // finite cold-store bound; zero disables paging and therefore fails closed.
  std::uint64_t exact_history_archive_capacity_bytes = 1ull << 30;

  // ── TWO NAMED LAW CEILINGS ────────────────────────────────────────────────
  //
  // Both were hard-coded constants introduced as repairs, and a repair whose
  // effect nobody can turn off cannot be attributed -- the same reason
  // `refractory_period_ticks` is a parameter rather than a literal 2. Their
  // defaults ARE the current behaviour; raising one until it stops binding is
  // the ablation, exactly as `refractory_period_ticks = 0` ablates the gate.
  //
  // persistent_bias_ceiling_q16: the node-local support and slow-context sum
  // may not exceed this. At
  // kQ16One/16 -- the activation threshold -- a node whose only drive is the
  // bias sits exactly at the threshold and cannot fire, so only transient
  // evidence arriving this tick can fire a node.
  std::int32_t persistent_bias_ceiling_q16 = (1 << 16) / 16;
  // slow_context_ceiling_q16: node_slow_context_q16 accumulates by atomicAdd
  // from lived contact with no decay anywhere, so without a bound it is a
  // monotone counter that eventually exceeds any threshold and then overflows.
  std::int32_t slow_context_ceiling_q16 = 1 << 16;

  // ── CAUSAL PIN (#1178 rung 13 / gh #1294) ─────────────────────────────────
  //
  // honor_causal_pin: the autopoietic sweep judges a route by how quiet its
  // SOURCE NODE is, and by nothing else. That reading cannot tell "obsolete"
  // from "quiet right now", which is the property #1294 protects:
  //
  //     quiet != obsolete;  low traffic != causally irrelevant
  //
  // A route that a live eligibility record still names is one some episode's
  // exact causal ancestry runs through. Reclaiming it destroys the credit path
  // before the consequence returns -- and consequence return is asynchronous
  // here, so "the node has been quiet for a while" is the NORMAL state of a
  // route awaiting one. The V01 ecology has known this since
  // `is_route_hard_pinned_by_eligibility_exact`; the Direct lane, which is the
  // lane the canon picks, never got it.
  //
  // Default 1 IS the new law. Setting it to 0 is the property-destroying mutant
  // arm required by #1290 / #1294 §7: it must recover the collapse signature,
  // and if it does not, the attribution is invalid. It exists as a parameter
  // for exactly the reason the two ceilings above do -- a repair nobody can
  // switch off cannot be attributed to.
  std::uint32_t honor_causal_pin = 1u;

  std::uint32_t honor_inhibitory_sign = 1u;

  // f.efference (#1470) paired-dispatch law. Default 1 routes a same-tick
  // efference copy beside every emitted motor command; 0 ablates the routing
  // entirely -- the property-destroying arm that attributes any claimed
  // shadow consequence to the copies themselves.
  std::uint32_t route_efference_copies = 1u;
  // Physical resident contributor capacity; never content or cadence.
  std::uint32_t resident_motor_trajectory_capacity =
      kMaxProvenanceSlotsPerNode;
};
struct alignas(32) ResidentAdultEpochSnapshot {
  std::uint32_t stop;
  std::uint32_t tick;
  std::uint32_t ingress_head;
  std::uint32_t ingress_tail;
  std::uint32_t consequence_head;
  std::uint32_t consequence_tail;
  std::uint32_t ingress_fault;
  std::uint32_t consequence_fault;
};
static_assert(sizeof(ResidentAdultEpochSnapshot) == 32 && std::is_standard_layout_v<ResidentAdultEpochSnapshot> && std::is_trivial_v<ResidentAdultEpochSnapshot>);
static_assert(std::has_unique_object_representations_v<ResidentAdultEpochSnapshot>);

struct ResidentCueSalienceTable;
// Device-side adult execution mesh runtime.
struct DirectAdultRuntime {
  DirectBrain* brain;
  ResidentDevelopmentWorkspace resident_development;
  std::uint64_t resident_development_epochs;
  ActivityEvent* ingress_queue;
  ResidentContactEpochReceipt* ingress_contact_credentials;
  ResidentActualFrontier* actual_frontier;
  ResidentActivationSoaPlane* activation_plane;
  ResidentMultiHorizonPredictionFrontier* causal_credit_predictions;
  ResidentMismatchOmissionRuntime* mismatch_omission;
  // gh #1369 / #1208. Device-owned consumed head cursor, updated after all
  // reader blocks in ingest_sensory_events_kernel have completed (alias into ingress_control).
  std::uint32_t* ingress_consumed_head;
  // Host transport-owned published tail cursor (alias into ingress_control).
  std::uint32_t* ingress_published_tail;
  // Host mirror of `ingress_queue`, kMaxIngressQueueSize entries, page-locked,
  // owned by the runtime. Admission writes HERE and performs no CUDA call at all;
  // flush_sensory_ingress ships the pending range in bulk copies on the step stream.
  ActivityEvent* host_ingress_staging;
  ResidentContactEpochReceipt* host_ingress_contact_staging;
  DirectBoundaryPort* host_boundary_ports;
  MotorEvent* egress_queue;
  std::uint32_t* egress_head;
  std::uint32_t* egress_tail;
  DirectEfferenceCopy* efference_ring;
  std::uint32_t* efference_head;
  std::uint32_t* efference_tail;
  EligibilityRecord* eligibility_table;
  std::uint32_t* live_eligibility_count;
  std::uint64_t* eligibility_claim_directory;
  std::uint32_t* eligibility_claim_locks;
  std::uint32_t* eligibility_record_generations;
  AsynchronousTicket* ticket_table;
  std::uint32_t* ticket_count;
  std::uint32_t* ticket_table_locks;
  std::uint32_t* claim_incarnation_counter;
  DirectActionOccurrence* action_occurrences;
  DirectActionParticipationLink* action_participation_links;
  ResidentPublicMotorTrajectory* resident_motor_trajectory;
  // Resident, per-Adult body-consequence state. Host transport never derives
  // or selects its contents; exact device consequence ancestry does.
  direct_network::DirectAffectBodyState* affect_body_state;
  // Resident incentive-salience / hedonic-impact profile. The resident step
  // loop writes it from exact participation and settled ledgers only; the
  // host may read it but never compose or select its contents.
  direct_network::ResidentWantingLikingProfileV1* wanting_liking_state;
  // Consequence-earned resident action-to-world relation projection.
  // Checkpointed with the contiguous consequence-state allocation.
  direct_network::DirectCausalWorldModel* causal_world_model;
  // Continuing Adult-owned language-plasticity factor. This is ordinary
  // future-causal resident state, not host transcript/context authority and not
  // an inherited language ontology. Its complete type stays behind a thin ABI.
  direct_network::DirectResidentLanguageRuntimeBlock* resident_language;
  // Domain-general prospective/action-control factors owned by the same Adult.
  DirectAdultActionControlRuntimeBlock* action_control_runtime;
  ResidentCueSalienceTable* cue_salience_table;
  ResolvedConsequenceContext* resolved_consequence_ctx;
  AttractorBasinState* attractor_state;
  std::int32_t* node_incoming_excitation;
  std::int32_t* node_slow_context_q16;
  DelayedSparsePacket* delayed_packets;
  std::uint32_t* delayed_packet_live_count;
  std::uint32_t* delayed_packet_free_head;
  std::uint32_t* delayed_packet_next_free;
  DelayedPacketIdentity* delayed_packet_identities;
  RouteTransportProposal* route_transport_proposals;
  EligibilityBatchClaim* eligibility_batch_claims;
  std::uint32_t eligibility_batch_capacity;
  std::uint32_t* route_transport_scan_a;
  std::uint32_t* route_transport_scan_b;
  std::uint32_t route_transport_scan_capacity;
  std::uint32_t* route_transport_free_slots;
  std::uint32_t* eligibility_batch_owners;
  std::uint32_t* node_participation_candidate_owners;
  std::uint32_t* route_transport_cursor;
  // #1178 exact reclaim opportunity. Canonical born brains own this in their
  // arena; legacy/manual fixtures may use an explicitly charged runtime fallback.
  std::uint64_t* route_opportunity_incarnations;
  bool owns_route_opportunity_incarnations;
  // One causal-pin bit per route slot, rebuilt on each maintenance tick.
  // A per-route O(routes * records) scan -- 136k * 65k on the fixture -- is avoided; the pin is
  // materialised once per sweep at O(records + route_words) and read as one bit.
  // Exact, not sampled: there is no bounded-scan `unknown` state to fold into
  // "pinned", because nothing is skipped.
  std::uint32_t* route_causal_pin_bits;
  NodeCausalParticipation *node_active_participation, *node_next_participation;
  std::uint32_t *node_active_participation_locks, *node_next_participation_locks;
  std::uint32_t *node_active_ancestry_incomplete, *node_next_ancestry_incomplete;
  DirectParticipationDescriptor* participation_staging;
  std::uint32_t* participation_staging_count;
  std::uint32_t participation_staging_capacity;
  AdultCoreMetrics* device_metrics;
  AdultExecutionConfig config;
  // gh #1369 / #1208. Distinct host ingress frontiers:
  //   host_ingress_write_tail: CPU has staged through here
  //   host_ingress_publish_tail: transport has published/enqueued through here
  //   host_ingress_observed_head: conservative device-consumed/reclaim frontier
  std::uint32_t host_ingress_write_tail;
  std::uint32_t host_ingress_publish_tail;
  std::uint32_t host_ingress_observed_head;
  std::uint32_t host_ingress_dispatched_tail;
  std::uint32_t* host_ingress_head_snapshot;  // pinned page-locked 4B for pressure-driven D2H
  cudaEvent_t ingress_consumed_event;        // recorded after host-stepped consumed-head commit
  std::uint32_t* host_ingress_publish_slot_pinned; // pinned 4B for async publication

  std::uint64_t host_ingress_overflow_drops;
  std::uint64_t host_ingress_protocol_faults;
  std::uint32_t current_tick;
  // gh #1305/#1208. MUST be a blocking stream (plain cudaStreamCreate): ordering
  // against the still-legacy lifecycle launches and every blocking cudaMemcpy
  // here rests on legacy-default-stream implicit sync, which
  // cudaStreamNonBlocking switches off. nullptr IS the legacy stream, so a
  // caller that never reaches create_direct_adult_runtime is unaffected.
  CUstream_st* stream;
  // gh #1208 persistent transport stream and explicit control ABI
  IngressRingControl* ingress_control;
  CUstream_st* transport_stream;

  // gh #1208 / Patch 3 consequence transport ring
  ConsequenceIngressEvent* host_consequence_staging;
  ConsequenceIngressEvent* consequence_queue;
  IngressRingControl* consequence_control;
  std::uint32_t host_consequence_write_tail;
  std::uint32_t host_consequence_publish_tail;
  std::uint32_t host_consequence_observed_head;
  std::uint32_t* host_consequence_head_snapshot;  // pinned page-locked 4B
  cudaEvent_t consequence_consumed_event;
  std::uint32_t* host_consequence_publish_slot_pinned; // pinned 4B
  std::uint64_t host_consequence_overflow_drops;
  std::uint64_t host_consequence_protocol_faults;

  // Global epoch snapshot for cooperative grid
  ResidentAdultEpochSnapshot* device_epoch_snapshot;

  // gh #1208 persistent execution control & authority model
  std::uint32_t* device_stop_flag;
  std::uint32_t* device_resident_tick;
  std::uint32_t* device_epoch_limit;
  AdultExecutionAuthority execution_authority;
  bool is_persistent_running;
  CUstream_st* persistent_stream;
  cudaEvent_t transport_ingress_event;
  cudaEvent_t transport_consequence_event;
  char exact_history_archive_directory[512]{};
  std::uint64_t persistent_bootstrap_launches;
  AdultCoreObserverSnapshot* host_observer_snapshot;
  AdultCoreObserverSnapshot* device_observer_snapshot;
  bool observer_snapshot_pending;
  std::uint64_t observer_snapshot_requests;
  std::uint64_t observer_snapshot_completions;
};

// Host lifecycle interface
DirectAdultRuntime* create_direct_adult_runtime(DirectBrain* brain, const AdultExecutionConfig& config = {});
void destroy_direct_adult_runtime(DirectAdultRuntime* runtime);
direct_network::ResidentDevelopmentCounters observe_direct_adult_resident_development(
    DirectAdultRuntime* runtime);

// Persistent adult lifecycle & asynchronous transport interface (gh #1208)
bool start_persistent_direct_adult(DirectAdultRuntime* runtime);
bool run_persistent_direct_adult_epochs(DirectAdultRuntime* runtime, std::uint32_t epoch_count);
void stop_persistent_direct_adult(DirectAdultRuntime* runtime);
bool configure_direct_exact_history_archive(
    DirectAdultRuntime* runtime, const char* directory,
    std::uint64_t capacity_bytes);
bool publish_ingress_transport(DirectAdultRuntime* runtime);
bool publish_consequence_transport(DirectAdultRuntime* runtime);
std::uint32_t get_persistent_adult_resident_tick(const DirectAdultRuntime* runtime);
AdultExecutionAuthority get_direct_adult_execution_authority(const DirectAdultRuntime* runtime);

// Ingress & Egress operations (non-blocking, backpressure protected)
bool inject_sensory_event(DirectAdultRuntime* runtime, const ActivityEvent& event);
std::uint32_t inject_sensory_events(DirectAdultRuntime* runtime, const ActivityEvent* events, std::uint32_t count);
// Actual participation requires an immutable body-bound credential. Ordinary
// sensory transport deliberately carries a zero credential and cannot mint an
// actual Occurrence merely by existing.
bool inject_actual_sensory_contact(
    DirectAdultRuntime* runtime, const ActivityEvent& event,
    const ResidentContactEpochReceipt& receipt);
// Flushes pending staged sensory events to the device ingress queue via stream-ordered
// asynchronous transfers on runtime->stream. Returns the number of events published.
std::uint32_t flush_sensory_ingress(DirectAdultRuntime* runtime);
// Native raw-return boundary: callers provide only the pending ticket identity
// and the opaque returned word.  The device compares that word with the
// resident ticket prediction and derives the signed consequence itself; no
// caller-supplied reward, correctness label, or expected value crosses here.
bool inject_raw_reafferent_contact(DirectAdultRuntime* runtime,
                                   std::uint64_t ticket_id, Word returned_word);
// Independent physical/world return for a pending resident action. Like raw
// reafference, the caller supplies no reward or answer; only this independent
// causal origin may revise compact recruited-Network preference.
bool inject_raw_world_return(DirectAdultRuntime* runtime,
                             std::uint64_t ticket_id, Word returned_word);
std::uint32_t read_motor_events(DirectAdultRuntime* runtime, MotorEvent* out_buffer, std::uint32_t max_count);
std::uint32_t read_efference_copies(DirectAdultRuntime* runtime, DirectEfferenceCopy* out_buffer, std::uint32_t max_count);

// Multi-step GPU execution loop.
void step_direct_adult_epochs(DirectAdultRuntime* runtime, std::uint32_t epoch_count);
// Observer-only fixed-morphology execution. Uses the same ingress, recurrent,
// inhibition, delay, motor and learning kernels as step_direct_adult_epochs,
// but deliberately skips resident structural-development and maintenance
// transactions so a post-hoc perturbation compares responses of one completed
// morphology rather than a different morphology per probe arm.
void step_direct_adult_fixed_morphology_epochs(
    DirectAdultRuntime* runtime, std::uint32_t epoch_count);
// Host-stepped and persistent executors share the same finite exact-history
// paging boundary. Returns false rather than dropping a sealed resident page.
bool page_completed_direct_exact_history(DirectAdultRuntime* runtime);

// Metrics inspection
AdultCoreMetrics get_adult_core_metrics(const DirectAdultRuntime* runtime);
void reset_adult_core_metrics(DirectAdultRuntime* runtime);
bool request_adult_core_observer_snapshot(DirectAdultRuntime* runtime);
bool query_adult_core_observer_snapshot(DirectAdultRuntime* runtime,
                                        AdultCoreObserverSnapshot* out);

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CORE_CUH
