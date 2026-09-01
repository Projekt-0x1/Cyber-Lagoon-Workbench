#ifndef HARDWARE_NATIVE_DIRECT_ADULT_INCENTIVE_SALIENCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_INCENTIVE_SALIENCE_CUH

#include <type_traits>

#include "direct_adult_core_constants.cuh"
#include "direct_exact_history.cuh"

namespace substrate::direct_adult_core {

// Berridge's evolutionary split, resident: a context signature that preceded
// a REWARDED pursuit earns incentive salience -- pursuit readiness under that
// context rises -- while every hedonic store stays untouched by attribution
// alone. Salience decays without renewal, clamps inside its band, and refuses
// contexts with no reward history fail-closed.
inline constexpr std::uint32_t kResidentCueSalienceCapacity = 16u;
inline constexpr std::uint32_t kResidentSalienceStepQ16 = 1u << 14;
inline constexpr std::uint32_t kResidentSalienceMaxQ16 = 1u << 15;
inline constexpr std::uint64_t kResidentSalienceSeed = 0x73616c69656e6365ull;

struct ResidentCueSalienceTable {
  std::uint32_t context_signature[kResidentCueSalienceCapacity];
  std::uint32_t salience_q16[kResidentCueSalienceCapacity];
  std::uint64_t table_identity;
};
static_assert(std::is_trivial_v<ResidentCueSalienceTable> &&
              std::is_standard_layout_v<ResidentCueSalienceTable>);

__device__ inline void refresh_resident_salience_identity(
    ResidentCueSalienceTable* table) {
  if (table == nullptr) return;
  std::uint64_t identity =
      direct_network::exact_history_fold_word(kResidentSalienceSeed,
                                              kResidentCueSalienceCapacity);
  for (std::uint32_t i = 0u; i < kResidentCueSalienceCapacity; ++i) {
    identity = direct_network::exact_history_fold_word(
        identity, static_cast<std::uint64_t>(table->context_signature[i]));
    identity = direct_network::exact_history_fold_word(
        identity, static_cast<std::uint64_t>(table->salience_q16[i]));
  }
  table->table_identity = identity == 0u ? 1u : identity | (1ull << 63);
}

__device__ inline std::int32_t resident_cue_slot(
    const ResidentCueSalienceTable* table, std::uint32_t context_signature) {
  if (table == nullptr) return -1;
  for (std::uint32_t i = 0u; i < kResidentCueSalienceCapacity; ++i)
    if (table->context_signature[i] == context_signature) {
      return static_cast<std::int32_t>(i);
    }
  return -1;
}

// Attributes one step of salience to `context_signature` because it preceded
// a rewarded pursuit (settled_reward_q16 > 0). Unrewarded or unknown contexts
// refuse fail-closed. Hedonic stores are the caller's business and stay
// untouched here by construction: this function receives none.
__device__ inline bool attribute_resident_incentive_salience(
    ResidentCueSalienceTable* table, std::uint32_t context_signature,
    std::int32_t settled_reward_q16) {
  if (table == nullptr || settled_reward_q16 <= 0) return false;
  std::int32_t slot = resident_cue_slot(table, context_signature);
  if (slot < 0) {
    // Fresh cue: claim an empty slot, refusing when the table is full.
    for (std::uint32_t i = 0u; i < kResidentCueSalienceCapacity; ++i) {
      if (table->context_signature[i] == 0u &&
          table->salience_q16[i] == 0u) {
        table->context_signature[i] = context_signature;
        slot = static_cast<std::int32_t>(i);
        break;
      }
    }
    if (slot < 0) return false;
  }
  std::uint32_t updated =
      table->salience_q16[slot] + kResidentSalienceStepQ16;
  if (updated > kResidentSalienceMaxQ16) updated = kResidentSalienceMaxQ16;
  table->salience_q16[slot] = updated;
  refresh_resident_salience_identity(table);
  return true;
}

// Decay without renewal: every entry loses one step toward zero.
__device__ inline void decay_resident_cue_salience(
    ResidentCueSalienceTable* table) {
  if (table == nullptr) return;
  for (std::uint32_t i = 0u; i < kResidentCueSalienceCapacity; ++i) {
    if (table->salience_q16[i] > kResidentSalienceStepQ16)
      table->salience_q16[i] -= kResidentSalienceStepQ16;
    else {
      table->salience_q16[i] = 0u;
      table->context_signature[i] = 0u;
    }
  }
  refresh_resident_salience_identity(table);
}

// One epoch of renewal-less decay, launched by both executors next to their
// other per-epoch maintenance passes. Defined once in direct_adult_core.cu.
__global__ void decay_cue_salience_epoch_kernel(
    ResidentCueSalienceTable* table);

}  // namespace substrate::direct_adult_core

#endif
