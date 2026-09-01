#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PSEUDO_KINSHIP_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PSEUDO_KINSHIP_CUH

// h.pseudo_kinship_categorical_binding (#1614). In-group / out-group
// categories grow from shared symbolic marker evidence under device-owned
// partner identities. The binding is malleable by construction: when the
// self marker changes, every classification is recomputed against the new
// evidence and each flip is a counted rebinding. Payload bytes other than
// the marker decide nothing, and unknown partners fail closed.

#include <cstdint>

namespace substrate::direct_network {

inline constexpr std::uint32_t kKinshipPartnerCapacity = 16u;

struct DirectKinshipStanding {
  std::uint64_t partner_identity;
  std::uint32_t observed_marker;
  std::uint32_t observations;
  std::uint32_t in_group;
};

struct DirectKinshipBinding {
  DirectKinshipStanding partners[kKinshipPartnerCapacity] = {};
  std::uint32_t partner_count;
  std::uint32_t rebindings;
  std::uint32_t self_marker;
  std::uint64_t binding_identity;
};

static_assert(std::is_trivially_copyable_v<DirectKinshipBinding>);

__host__ __device__ inline std::uint64_t kinship_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

__host__ __device__ inline std::int32_t kinship_find_partner(
    const DirectKinshipBinding& binding, std::uint64_t partner_identity) {
  for (std::uint32_t i = 0u; i < binding.partner_count; ++i)
    if (binding.partners[i].partner_identity == partner_identity)
      return static_cast<std::int32_t>(i);
  return -1;
}

__host__ __device__ inline void kinship_reclassify(
    DirectKinshipBinding* binding, DirectKinshipStanding& standing,
    bool flip_counts_as_rebinding) {
  const std::uint32_t was_in_group = standing.in_group;
  standing.in_group =
      standing.observed_marker == binding->self_marker ? 1u : 0u;
  // A partner's first sighting classifies without rebinding anything:
  // there was no prior binding to move. Later flips are rebindings, whether
  // they come from new evidence or from moving our own marker.
  if (flip_counts_as_rebinding && was_in_group != standing.in_group)
    ++binding->rebindings;
}

// One marker observation of one partner. First sightings mint standing and
// classify immediately against the current self marker.
__host__ __device__ inline bool observe_partner_marker(
    DirectKinshipBinding* binding, std::uint64_t partner_identity,
    std::uint32_t observed_marker) {
  if (binding == nullptr || partner_identity == 0u) return false;
  std::int32_t index = kinship_find_partner(*binding, partner_identity);
  if (index < 0) {
    if (binding->partner_count >= kKinshipPartnerCapacity) return false;
    index = static_cast<std::int32_t>(binding->partner_count);
    DirectKinshipStanding fresh{};
    fresh.partner_identity = partner_identity;
    binding->partners[index] = fresh;
    ++binding->partner_count;
  }
  DirectKinshipStanding& standing = binding->partners[index];
  const bool prior_sightings = standing.observations > 0u;
  standing.observed_marker = observed_marker;
  ++standing.observations;
  kinship_reclassify(binding, standing, prior_sightings);
  binding->binding_identity =
      kinship_fold(kinship_fold(binding->binding_identity, partner_identity),
                   static_cast<std::uint64_t>(observed_marker));
  return true;
}

// Malleability: moving our own symbolic marker rebinds every classification.
__host__ __device__ inline bool set_kinship_self_marker(
    DirectKinshipBinding* binding, std::uint32_t self_marker) {
  if (binding == nullptr) return false;
  binding->self_marker = self_marker;
  for (std::uint32_t i = 0u; i < binding->partner_count; ++i)
    kinship_reclassify(binding, binding->partners[i], true);
  binding->binding_identity =
      kinship_fold(binding->binding_identity,
                   static_cast<std::uint64_t>(self_marker) << 32u);
  return true;
}

__host__ __device__ inline bool partner_in_group(
    const DirectKinshipBinding& binding, std::uint64_t partner_identity,
    bool* known) {
  if (known != nullptr) *known = false;
  const std::int32_t index = kinship_find_partner(binding, partner_identity);
  if (index < 0) return false;
  if (known != nullptr) *known = true;
  return binding.partners[index].in_group != 0u;
}

}  // namespace substrate::direct_network

#endif
