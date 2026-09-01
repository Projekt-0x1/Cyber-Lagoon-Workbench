#pragma once

// Value-neutral resident anatomy for the frozen M2 raw-contact task.
//
// This composition deliberately contains only checked-in developmental matter.
// It does not schedule contacts, inspect byte values, select a route, or update
// a weight. Ordinary F must connect the raw sensory surface to these bodies.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "bcc32_developmental_adult.cuh"
#include "bcc32_factorized_credit_seed.cuh"
#include "bcc32_processive_prediction_projection_seed.cuh"

namespace substrate::bcc32 {

inline constexpr Int3 kM2ResidentPredictivePopulationOffset{0, 16, 0};

template <std::size_t N>
inline void append_m2_resident_seed(
    const developmental_adult::GrownAdult& adult,
    const std::array<DevelopmentalSeedSite, N>& seed,
    std::vector<developmental_adult::StateEntry>* entries) {
  for (const DevelopmentalSeedSite& site : seed) {
    if (site.word == kQ) continue;
    const std::uint64_t target = adult.physical_slot(
        {site.x + kM2ResidentPredictivePopulationOffset.x,
         site.y + kM2ResidentPredictivePopulationOffset.y,
         site.z + kM2ResidentPredictivePopulationOffset.z});
    const auto existing = std::find_if(
        entries->begin(), entries->end(),
        [target](const developmental_adult::StateEntry& entry) {
          return entry.slot == target;
        });
    if (existing != entries->end()) {
      if (existing->word != site.word) {
        throw std::logic_error(
            "M2 resident predictive founder families overlap");
      }
      continue;
    }
    entries->push_back({target, site.word});
  }
}

inline std::vector<developmental_adult::StateEntry>
m2_resident_predictive_population_seed(
    const developmental_adult::GrownAdult& adult) {
  std::vector<developmental_adult::StateEntry> entries;
  append_m2_resident_seed(adult, processive_prediction_projection_seed(),
                          &entries);
  append_m2_resident_seed(
      adult,
      processive_prediction_comparator_seed(
          kProcessivePredictionComparatorSeedHash),
      &entries);
  append_m2_resident_seed(
      adult,
      synaptic_route_activity_bridge_seed(
          kSynapticRouteActivityBridgeSeedHash, false),
      &entries);
  append_m2_resident_seed(adult, factorized_credit_translator_seed(),
                          &entries);
  append_m2_resident_seed(adult, factorized_credit_weight_seed(), &entries);
  return entries;
}

struct M2ResidentFactorCoordinates {
  Int3 prediction_source{};
  Int3 prediction_zero_output{};
  Int3 prediction_one_output{};
  Int3 positive_error{};
  Int3 negative_error{};
  Int3 eligibility_e1{};
  Int3 eligibility_e3{};
  Int3 translator_input{};
  Int3 weight{};
};

constexpr M2ResidentFactorCoordinates m2_resident_factor_coordinates() {
  const auto shifted = [](Int3 value) constexpr {
    return value + kM2ResidentPredictivePopulationOffset;
  };
  return {
      shifted(processive_prediction_projection_body(
          kProcessivePredictionProjectionThresholdIndex)),
      shifted(processive_prediction_projection_output(false)),
      shifted(processive_prediction_projection_output(true)),
      shifted(processive_prediction_comparator_error_port(
          ProcessivePredictionComparatorArm::positive)),
      shifted(processive_prediction_comparator_error_port(
          ProcessivePredictionComparatorArm::negative)),
      shifted(factorized_credit_bridge_e1()),
      shifted(factorized_credit_bridge_e3()),
      shifted(factorized_credit_translator_input_port()),
      shifted(factorized_credit_weight_body()),
  };
}

static_assert(!factorized_credit_has_seeded_local_collision());

}  // namespace substrate::bcc32
