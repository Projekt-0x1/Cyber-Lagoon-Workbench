#ifndef HARDWARE_NATIVE_DIRECT_ADULT_GREEN_BEARD_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_GREEN_BEARD_CUH

// h.green_beard_honest_signaling (#1615). Cooperation composes two grown
// evidence surfaces: the symbolic-marker kinship binding (#1614) and the
// reciprocity standing ledger (#1612). A marker alone protects nobody --
// a signaling partner who defects across the trust threshold loses the gate
// and earns a counted sanction. Unknown partners fail closed.

#include <cstdint>

#include "hardware_native/direct_adult_partner_reciprocity.cuh"
#include "hardware_native/direct_adult_pseudo_kinship.cuh"

namespace substrate::direct_network {

struct DirectGreenBeardGate {
  std::uint32_t in_group;
  std::uint32_t trusted;
  std::uint32_t cooperating;  // composed gate decision
};

// One interaction with one partner: their signal joins the kinship binding
// and their settled outcome joins the reciprocity ledger, then both surfaces
// vote on cooperation.
__host__ __device__ inline DirectGreenBeardGate green_beard_interact(
    DirectKinshipBinding* kinship, DirectReciprocityLedger* reciprocity,
    std::uint64_t partner_identity, std::uint32_t observed_marker,
    std::int32_t settled_reward_q16) {
  DirectGreenBeardGate gate{};
  if (kinship == nullptr || reciprocity == nullptr ||
      !observe_partner_marker(kinship, partner_identity, observed_marker) ||
      !observe_partner_outcome(reciprocity, partner_identity,
                               settled_reward_q16))
    return gate;
  bool known = false;
  bool trusted = false;
  const bool in_group =
      partner_in_group(*kinship, partner_identity, &known);
  (void)known;
  trusted = partner_trusted(*reciprocity, partner_identity);
  gate.in_group = in_group ? 1u : 0u;
  gate.trusted = trusted ? 1u : 0u;
  gate.cooperating = (gate.in_group != 0u && gate.trusted != 0u) ? 1u : 0u;
  return gate;
}

}  // namespace substrate::direct_network

#endif
