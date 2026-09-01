#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bcc32_cuda_resident_synthesis {

inline constexpr std::uint32_t kResidentSynthesisPolicyVariants = 5u;
inline constexpr std::uint32_t kResidentSynthesisPolicyWindow = 64u;
inline constexpr std::uint32_t kResidentSynthesisPolicyMask =
    (1u << kResidentSynthesisPolicyVariants) - 1u;

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
resident_synthesis_policy_weight(std::uint32_t variant) {
  return variant < kResidentSynthesisPolicyVariants ? (1u << variant) : 0u;
}

// One CUDA lane owns one state and advances it in predict/observe order.
// Candidate values are opaque raw-unit identities. Success populations and
// history are device-resident evidence, not host-selected scores.
struct ResidentSynthesisPolicyState {
  std::uint32_t success_population[kResidentSynthesisPolicyVariants]{};
  std::uint32_t success_history[kResidentSynthesisPolicyWindow]{};
  std::uint32_t pending_raw_units[kResidentSynthesisPolicyVariants]{};
  std::uint32_t history_cursor = 0u;
  std::uint32_t history_size = 0u;
  std::uint32_t pending_selected_variant = 0u;
  std::uint32_t pending_selected_raw_unit = 0u;
  std::uint32_t pending_valid = 0u;
  std::uint32_t lesion_count = 0u;
  unsigned long long observations_since_lesion = 0ull;
  unsigned long long lifetime_observations = 0ull;
};

struct ResidentSynthesisPolicyDecision {
  std::uint32_t ready = 0u;
  std::uint32_t variant = 0u;
  std::uint32_t weight = 0u;
  std::uint32_t raw_unit = 0u;
  std::uint32_t evidence_events = 0u;
};

struct ResidentSynthesisPolicyObservation {
  std::uint32_t accepted = 0u;
  std::uint32_t selected_variant = 0u;
  std::uint32_t selected_hit = 0u;
  std::uint32_t success_mask = 0u;
  std::uint32_t evicted_mask = 0u;
  std::uint32_t success_mass_before = 0u;
  std::uint32_t success_mass_after = 0u;
};

[[nodiscard]] __device__ inline std::uint32_t
resident_synthesis_policy_popcount(std::uint32_t value) {
  return static_cast<std::uint32_t>(__popc(value));
}

[[nodiscard]] __device__ inline std::uint32_t
resident_synthesis_policy_success_mass(const ResidentSynthesisPolicyState& state) {
  std::uint32_t mass = 0u;
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    mass += state.success_population[variant];
  }
  return mass;
}

[[nodiscard]] __device__ inline bool resident_synthesis_policy_accounting_valid(
    const ResidentSynthesisPolicyState& state) {
  if (state.history_size > kResidentSynthesisPolicyWindow ||
      state.history_cursor >= kResidentSynthesisPolicyWindow ||
      state.pending_valid > 1u) {
    return false;
  }
  if (state.history_size < kResidentSynthesisPolicyWindow &&
      state.history_cursor != state.history_size) {
    return false;
  }

  std::uint32_t history_mass = 0u;
  std::uint32_t reconstructed[kResidentSynthesisPolicyVariants]{};
  for (std::uint32_t slot = 0u; slot < kResidentSynthesisPolicyWindow; ++slot) {
    const std::uint32_t mask = state.success_history[slot];
    if ((mask & ~kResidentSynthesisPolicyMask) != 0u) return false;
    history_mass += resident_synthesis_policy_popcount(mask);
    for (std::uint32_t variant = 0u;
         variant < kResidentSynthesisPolicyVariants; ++variant) {
      reconstructed[variant] += (mask >> variant) & 1u;
    }
  }

  std::uint32_t population_mass = 0u;
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    if (state.success_population[variant] != reconstructed[variant] ||
        state.success_population[variant] > state.history_size) {
      return false;
    }
    population_mass += state.success_population[variant];
  }
  if (population_mass != history_mass) return false;

  if (state.pending_valid != 0u) {
    if (state.pending_selected_variant >= kResidentSynthesisPolicyVariants ||
        state.pending_selected_raw_unit !=
            state.pending_raw_units[state.pending_selected_variant]) {
      return false;
    }
  }
  return true;
}

__device__ inline void resident_synthesis_policy_reset(
    ResidentSynthesisPolicyState* state) {
  if (state == nullptr) return;
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    state->success_population[variant] = 0u;
    state->pending_raw_units[variant] = 0u;
  }
  for (std::uint32_t slot = 0u; slot < kResidentSynthesisPolicyWindow; ++slot) {
    state->success_history[slot] = 0u;
  }
  state->history_cursor = 0u;
  state->history_size = 0u;
  state->pending_selected_variant = 0u;
  state->pending_selected_raw_unit = 0u;
  state->pending_valid = 0u;
  state->lesion_count = 0u;
  state->observations_since_lesion = 0ull;
  state->lifetime_observations = 0ull;
}

// A lesion is an explicit intervention boundary. It clears only policy
// evidence and a pending choice while preserving lifetime accounting.
__device__ inline void resident_synthesis_policy_lesion(
    ResidentSynthesisPolicyState* state) {
  if (state == nullptr) return;
  const unsigned long long lifetime = state->lifetime_observations;
  const std::uint32_t lesions = state->lesion_count + 1u;
  resident_synthesis_policy_reset(state);
  state->lifetime_observations = lifetime;
  state->lesion_count = lesions;
}

[[nodiscard]] __device__ inline std::uint32_t resident_synthesis_policy_select(
    const ResidentSynthesisPolicyState& state) {
  std::uint32_t selected = 0u;
  for (std::uint32_t variant = 1u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    if (state.success_population[variant] >
        state.success_population[selected]) {
      selected = variant;
    }
  }
  return selected;
}

// This operation cannot see the next observed raw unit. It commits one
// selection and all five candidate predictions before contact arrives.
[[nodiscard]] __device__ inline ResidentSynthesisPolicyDecision
resident_synthesis_policy_predict(
    ResidentSynthesisPolicyState* state,
    const std::uint32_t* candidate_raw_units) {
  ResidentSynthesisPolicyDecision decision{};
  if (state == nullptr || candidate_raw_units == nullptr ||
      state->pending_valid != 0u) {
    return decision;
  }

  const std::uint32_t selected = resident_synthesis_policy_select(*state);
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    state->pending_raw_units[variant] = candidate_raw_units[variant];
  }
  state->pending_selected_variant = selected;
  state->pending_selected_raw_unit = candidate_raw_units[selected];
  state->pending_valid = 1u;

  decision.ready = 1u;
  decision.variant = selected;
  decision.weight = resident_synthesis_policy_weight(selected);
  decision.raw_unit = candidate_raw_units[selected];
  decision.evidence_events = state->history_size;
  return decision;
}

// Pure post-diction: this is the sole normal update of success populations.
// The exact accounting identity is
//   before + popcount(success) = after + popcount(evicted).
[[nodiscard]] __device__ inline ResidentSynthesisPolicyObservation
resident_synthesis_policy_observe(ResidentSynthesisPolicyState* state,
                                  std::uint32_t observed_raw_unit) {
  ResidentSynthesisPolicyObservation observation{};
  if (state == nullptr || state->pending_valid == 0u) return observation;

  const std::uint32_t selected = state->pending_selected_variant;
  std::uint32_t success_mask = 0u;
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    if (state->pending_raw_units[variant] == observed_raw_unit) {
      success_mask |= 1u << variant;
    }
  }

  const std::uint32_t evicted_mask =
      state->history_size == kResidentSynthesisPolicyWindow
          ? state->success_history[state->history_cursor]
          : 0u;
  const std::uint32_t mass_before =
      resident_synthesis_policy_success_mass(*state);
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    const std::uint32_t removed = (evicted_mask >> variant) & 1u;
    const std::uint32_t added = (success_mask >> variant) & 1u;
    state->success_population[variant] =
        state->success_population[variant] - removed + added;
  }
  state->success_history[state->history_cursor] = success_mask;
  state->history_cursor =
      (state->history_cursor + 1u) % kResidentSynthesisPolicyWindow;
  if (state->history_size < kResidentSynthesisPolicyWindow) {
    ++state->history_size;
  }
  ++state->observations_since_lesion;
  ++state->lifetime_observations;

  state->pending_selected_variant = 0u;
  state->pending_selected_raw_unit = 0u;
  state->pending_valid = 0u;
  for (std::uint32_t variant = 0u;
       variant < kResidentSynthesisPolicyVariants; ++variant) {
    state->pending_raw_units[variant] = 0u;
  }

  observation.accepted = 1u;
  observation.selected_variant = selected;
  observation.selected_hit = (success_mask >> selected) & 1u;
  observation.success_mask = success_mask;
  observation.evicted_mask = evicted_mask;
  observation.success_mass_before = mass_before;
  observation.success_mass_after =
      resident_synthesis_policy_success_mass(*state);
  return observation;
}

}  // namespace bcc32_cuda_resident_synthesis
