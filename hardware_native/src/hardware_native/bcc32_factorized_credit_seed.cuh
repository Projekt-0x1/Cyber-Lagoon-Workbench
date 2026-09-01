#pragma once

// Current-law factorized-credit composition probe.
//
// This header intentionally contains founder matter only.  It does not expose
// a CUDA-side counter, a host-selected route mask, or an update kernel: any
// durable credit must be a consequence of ordinary F acting on these seeds.
// The companion contract checks whether the existing parts supply the required
// local collision before it is allowed to claim factorized temporal credit.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_port_translator_seed.cuh"
#include "bcc32_processive_prediction_comparator_seed.cuh"
#include "bcc32_processive_weight_region_seed.cuh"
#include "bcc32_synaptic_route_activity_bridge_seed.cuh"

namespace substrate::bcc32 {

// Keep the auxiliary credit-port and weight bodies disjoint from the resident
// comparator/bridge.  These offsets are founder geometry, not region labels or
// a routing table.
inline constexpr Int3 kFactorizedCreditTranslatorOffset{48, -48, -48};
inline constexpr Int3 kFactorizedCreditWeightOffset{-48, 48, 48};
inline constexpr ProcessiveWeightRegionSeedHash kFactorizedCreditWeightHash =
    make_processive_weight_region_seed_hash(1u, 0u, 1u, 2u,
                                            kProcessiveWeightGene);
inline constexpr std::size_t kFactorizedCreditWeightSeedSiteCount = 4u;
inline constexpr std::size_t kFactorizedCreditTranslatorSeedSiteCount =
    kCreditPortB2SeedSiteCount;

template <std::size_t N>
constexpr std::array<DevelopmentalSeedSite, N>
factorized_credit_translate_seed(const std::array<DevelopmentalSeedSite, N>& seed,
                                 Int3 offset) {
  std::array<DevelopmentalSeedSite, N> result{};
  for (std::size_t index = 0u; index < N; ++index) {
    result[index] = {
        static_cast<std::int8_t>(static_cast<std::int32_t>(seed[index].x) + offset.x),
        static_cast<std::int8_t>(static_cast<std::int32_t>(seed[index].y) + offset.y),
        static_cast<std::int8_t>(static_cast<std::int32_t>(seed[index].z) + offset.z),
        seed[index].word,
    };
  }
  return result;
}

constexpr DevelopmentalSeedSite factorized_credit_translate_site(
    DevelopmentalSeedSite site, Int3 offset) {
  return {
      static_cast<std::int8_t>(static_cast<std::int32_t>(site.x) + offset.x),
      static_cast<std::int8_t>(static_cast<std::int32_t>(site.y) + offset.y),
      static_cast<std::int8_t>(static_cast<std::int32_t>(site.z) + offset.z),
      site.word,
  };
}

inline constexpr auto factorized_credit_translator_seed() {
  // The translator's final slot is an intentional vacant output port (`kQ`).
  // Founder attachment accepts only matter, so retain the occupied parent
  // sites and leave that port vacant in the resident field.
  std::array<DevelopmentalSeedSite, kFactorizedCreditTranslatorSeedSiteCount>
      result{};
  const auto source = credit_port_translator_seed(kCreditPortTranslatorSeedHash);
  for (std::size_t index = 0u; index < result.size(); ++index)
    result[index] = factorized_credit_translate_site(
        source[index], kFactorizedCreditTranslatorOffset);
  return result;
}

inline constexpr auto factorized_credit_weight_seed() {
  std::array<DevelopmentalSeedSite, kFactorizedCreditWeightSeedSiteCount> result{};
  const auto source = processive_weight_region_seed(kFactorizedCreditWeightHash);
  for (std::size_t index = 0u; index < result.size(); ++index)
    result[index] = factorized_credit_translate_site(source[index],
                                                     kFactorizedCreditWeightOffset);
  return result;
}

constexpr Int3 factorized_credit_weight_body() {
  return kFactorizedCreditWeightOffset;
}

constexpr Int3 factorized_credit_translator_input_port() {
  return kFactorizedCreditTranslatorOffset + Int3{0, 0, -2};
}

constexpr Int3 factorized_credit_bridge_e1() {
  return synaptic_route_activity_bridge_hub(false) + Int3{0, -1, 0};
}

constexpr Int3 factorized_credit_bridge_e3() {
  return synaptic_route_activity_bridge_hub(false) + Int3{1, 1, 1};
}

constexpr bool factorized_credit_same_site(Int3 left, Int3 right) {
  return left == right;
}

// This is deliberately only a *seeded local-collision* predicate.  It says
// nothing about a hypothetical later law epoch; it prevents the current-law
// contract from relabelling separated ports as a three-factor reaction.
constexpr bool factorized_credit_has_seeded_local_collision() {
  const Int3 positive_error = processive_prediction_comparator_error_port(
      ProcessivePredictionComparatorArm::positive);
  const Int3 negative_error = processive_prediction_comparator_error_port(
      ProcessivePredictionComparatorArm::negative);
  return factorized_credit_same_site(positive_error, factorized_credit_bridge_e1()) ||
         factorized_credit_same_site(positive_error, factorized_credit_bridge_e3()) ||
         factorized_credit_same_site(negative_error, factorized_credit_bridge_e1()) ||
         factorized_credit_same_site(negative_error, factorized_credit_bridge_e3()) ||
         factorized_credit_same_site(positive_error,
                                     factorized_credit_translator_input_port()) ||
         factorized_credit_same_site(negative_error,
                                     factorized_credit_translator_input_port()) ||
         factorized_credit_same_site(factorized_credit_bridge_e1(),
                                     factorized_credit_weight_body()) ||
         factorized_credit_same_site(factorized_credit_bridge_e3(),
                                     factorized_credit_weight_body());
}

static_assert(valid_processive_weight_region_hash(kFactorizedCreditWeightHash));
static_assert(processive_weight_length(kFactorizedCreditWeightHash) == 1u);
static_assert(!factorized_credit_has_seeded_local_collision());

}  // namespace substrate::bcc32
