#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DISTRIBUTED_MODULATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DISTRIBUTED_MODULATION_CUH

#include <type_traits>

#include "direct_adult_core_constants.cuh"
#include "direct_exact_history.cuh"

namespace substrate::direct_adult_core {

// Global diffuse neuromodulatory signals: one resident level per class, no
// ticket or target, whose sole authority is scaling existing gate thresholds
// broadly. Dopaminergic gain lowers evidence/plasticity gates (incentive
// gain), serotonergic level counter-scales plasticity readiness back up,
// noradrenergic gain lowers the volitional veto's alerting threshold. One
// level move shifts every gate of its class together and never touches
// another class's gates.
enum class ResidentModulatorKind : std::uint32_t {
  dopaminergic = 0u,
  serotonergic = 1u,
  noradrenergic = 2u,
};
inline constexpr std::uint32_t kResidentModulatorClassCount = 3u;
inline constexpr std::uint64_t kResidentModulationSeed = 0x6d6f64756c617465ull;

struct ResidentDiffuseModulationState {
  std::uint32_t levels_q16[kResidentModulatorClassCount];
  std::uint64_t state_identity;
};
static_assert(std::is_trivial_v<ResidentDiffuseModulationState> &&
              std::is_standard_layout_v<ResidentDiffuseModulationState>);

__device__ inline void refresh_resident_modulation_identity(
    ResidentDiffuseModulationState* state) {
  if (state == nullptr) return;
  std::uint64_t identity =
      direct_network::exact_history_fold_word(kResidentModulationSeed,
                                              kResidentModulatorClassCount);
  for (std::uint32_t i = 0u; i < kResidentModulatorClassCount; ++i)
    identity = direct_network::exact_history_fold_word(
        identity, static_cast<std::uint64_t>(state->levels_q16[i]));
  state->state_identity =
      identity == 0u ? 1u : identity | (1ull << 63);
}

// Sets one class level clamped into [0, kQ16One]. Zero restores exact
// neutrality for that class.
__device__ inline bool set_resident_modulation_level(
    ResidentDiffuseModulationState* state, ResidentModulatorKind kind,
    std::uint32_t level_q16) {
  if (state == nullptr ||
      static_cast<std::uint32_t>(kind) >= kResidentModulatorClassCount)
    return false;
  constexpr std::uint32_t kOneQ16 = substrate::direct_adult_core::kQ16One;
  state->levels_q16[static_cast<std::uint32_t>(kind)] =
      level_q16 > kOneQ16 ? kOneQ16 : level_q16;
  refresh_resident_modulation_identity(state);
  return true;
}

// The broad scaling relation. Dopaminergic gain multiplies the base
// threshold DOWN (plasticity admits easier); serotonergic multiplies it UP;
// noradrenergic leaves plasticity gates untouched. Result stays bounded to
// [0, base] for gains and never exceeds the plain base for neutral level.
__device__ inline std::uint32_t modulate_resident_plasticity_threshold_q16(
    const ResidentDiffuseModulationState& state, std::uint32_t base_q16) {
  const std::int64_t dopaminergic =
      state.levels_q16[static_cast<std::uint32_t>(
          ResidentModulatorKind::dopaminergic)];
  const std::int64_t serotonergic =
      state.levels_q16[static_cast<std::uint32_t>(
          ResidentModulatorKind::serotonergic)];
  // Effective gain in [0, 2]: dopamine pulls down from 1, serotonin pushes up.
  std::int64_t numerator = static_cast<std::int64_t>(kQ16One) - dopaminergic +
                           serotonergic;
  if (numerator < 0) numerator = 0;
  const std::int64_t scaled = (static_cast<std::int64_t>(base_q16) *
                               numerator) /
                              static_cast<std::int64_t>(kQ16One);
  return static_cast<std::uint32_t>(scaled);
}

__device__ inline std::uint32_t modulate_resident_alerting_threshold_q16(
    const ResidentDiffuseModulationState& state, std::uint32_t base_q16) {
  const std::int64_t noradrenergic =
      state.levels_q16[static_cast<std::uint32_t>(
          ResidentModulatorKind::noradrenergic)];
  const std::int64_t serotonergic =
      state.levels_q16[static_cast<std::uint32_t>(
          ResidentModulatorKind::serotonergic)];
  std::int64_t numerator =
      static_cast<std::int64_t>(kQ16One) - noradrenergic + serotonergic / 2;
  if (numerator < 0) numerator = 0;
  const std::int64_t scaled = (static_cast<std::int64_t>(base_q16) *
                               numerator) /
                              static_cast<std::int64_t>(kQ16One);
  return static_cast<std::uint32_t>(scaled);
}

}  // namespace substrate::direct_adult_core

#endif
