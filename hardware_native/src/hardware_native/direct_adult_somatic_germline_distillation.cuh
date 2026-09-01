#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SOMATIC_GERMLINE_DISTILLATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SOMATIC_GERMLINE_DISTILLATION_CUH

// f.somatic_germline_distillation (#1607). Repeated cross-life structural
// evidence distills into one heritable DirectGenomeV1 developmental-rule
// delta. The germline loop (04 §21) strips every donor-runtime byte: the
// candidate is constructed exclusively from the base genome plus scalar
// adaptation masses, so inheritance carries the cause and never the weights.
// No single life may promote, and a variant that fails to dominate the
// baseline in every observed life refuses outright.

#include <cstdint>

#include "hardware_native/direct_adult_core_constants.cuh"
#include "hardware_native/direct_network_genome.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDistillationMinimumLives = 2u;

struct DirectSomaticLifeEvidence {
  std::uint64_t life_identity;       // observer tag only; never enters the genome
  std::uint32_t adaptation_mass_q16; // structural work expressed by this life
};

struct DirectGermlineDeltaCandidate {
  DirectGenomeV1 juvenile_genome;
  std::uint32_t fused_rule_index;
  std::uint32_t observed_lives;
  std::int32_t min_margin_q16;
  std::uint64_t cause_identity;
};

static_assert(std::is_trivially_copyable_v<DirectGermlineDeltaCandidate>);

__host__ __device__ inline std::uint64_t distillation_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

// One cause per delta: every observed life must cite the same causal rule
// index, dominate the baseline mass, and there must be at least
// kDistillationMinimumLives of them. The mutation amplifies the developmental
// integration (fuse) rule's group width -- a developmental-cause change that
// fresh juveniles express from birth -- never a copy of any lived structure.
__host__ __device__ inline bool distill_somatic_germline_delta(
    const DirectGenomeV1& base_genome,
    const DirectSomaticLifeEvidence* lives, std::uint32_t life_count,
    std::uint32_t baseline_mass_q16,
    DirectGermlineDeltaCandidate* out) {
  if (out == nullptr || lives == nullptr ||
      life_count < kDistillationMinimumLives)
    return false;
  if (base_genome.header.rule_count == 0u ||
      base_genome.header.rule_count >
          sizeof(base_genome.rules) / sizeof(base_genome.rules[0]))
    return false;
  std::int32_t min_margin_q16 = direct_adult_core::kQ16One;
  std::uint64_t cause = 0x6765726d6c696e65ULL;
  for (std::uint32_t i = 0u; i < life_count; ++i) {
    if (lives[i].adaptation_mass_q16 <= baseline_mass_q16) return false;
    const std::int32_t margin =
        static_cast<std::int32_t>(lives[i].adaptation_mass_q16 -
                                  baseline_mass_q16);
    if (margin < min_margin_q16) min_margin_q16 = margin;
    cause = distillation_fold(distillation_fold(cause, lives[i].life_identity),
                              lives[i].adaptation_mass_q16);
  }
  std::int32_t fused_index = -1;
  for (std::uint32_t r = 0u; r < base_genome.header.rule_count; ++r) {
    if (base_genome.rules[r].opcode == DirectRuleOpcodeV1::fuse) {
      fused_index = static_cast<std::int32_t>(r);
      break;
    }
  }
  if (fused_index < 0) return false;
  DirectGermlineDeltaCandidate candidate{};
  candidate.juvenile_genome = base_genome;
  const std::uint32_t node_span =
      candidate.juvenile_genome.rules[fused_index].branch_count * 2u;
  if (node_span == 0u || node_span > 64u) return false;
  candidate.juvenile_genome.rules[fused_index].branch_count = node_span;
  candidate.fused_rule_index = static_cast<std::uint32_t>(fused_index);
  candidate.observed_lives = life_count;
  candidate.min_margin_q16 = min_margin_q16;
  candidate.cause_identity =
      distillation_fold(cause, candidate.juvenile_genome.header.development_seed);
  *out = candidate;
  return true;
}

}  // namespace substrate::direct_network

#endif
