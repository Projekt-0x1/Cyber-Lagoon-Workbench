#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PARTICIPATION_DESCRIPTOR_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PARTICIPATION_DESCRIPTOR_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"

// Mid-include splice for substrate::direct_adult_core. Do not open a namespace.

enum class DirectContributionKind : std::uint32_t {
  none = 0u, direct_ingress = 1u, sparse_route = 2u,
  ancestry_incomplete = 3u};

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
static_assert(sizeof(DirectParticipationDescriptor) == 80 &&
              std::is_standard_layout_v<DirectParticipationDescriptor> &&
              std::is_trivial_v<DirectParticipationDescriptor>);
static_assert(std::has_unique_object_representations_v<DirectParticipationDescriptor>);

#endif
