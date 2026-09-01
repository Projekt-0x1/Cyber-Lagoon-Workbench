#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_CUH

#include <cstdint>
#include <type_traits>

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_CAUSAL_PROGRAM_HD __host__ __device__
#else
#define DIRECT_CAUSAL_PROGRAM_HD
#endif

inline constexpr std::int32_t kQ16One = 1 << 16;
inline constexpr std::uint32_t kMaxProgramSteps = 16u;
inline constexpr std::uint32_t kMaxProgramContexts = 8u;
inline constexpr std::uint32_t kProgramBankCapacity = 16u;
inline constexpr std::uint32_t kControlSuccessQuorum = 2u;
inline constexpr std::int32_t kControlHistoryStepQ16 = kQ16One / 8;

struct ContextOutcome {
  std::uint32_t context;
  std::uint32_t samples;
  std::int32_t outcome_mean_q16;
  std::int32_t somatic_mean_q16;
};

struct PredictiveProfile {
  std::uint64_t structure_identity;
  std::uint32_t exposures;
  std::uint32_t outcome_samples;
  std::uint32_t control_attempts;
  std::uint32_t control_successes;
  std::uint32_t background_attempts;
  std::uint32_t background_successes;
  std::int32_t accessibility_q16;
  std::int32_t duration_mean_q16;
  std::int32_t effort_mean_q16;
  std::int32_t outcome_mean_q16;
  std::int32_t somatic_mean_q16;
  std::int32_t controllability_q16;
  std::int32_t control_history_q16;
  ContextOutcome contexts[kMaxProgramContexts];
  std::uint32_t context_count;
};
static_assert(std::is_standard_layout_v<PredictiveProfile> &&
              std::is_trivial_v<PredictiveProfile>);

struct CurrentState {
  std::int32_t urgency_q16;
  std::int32_t resource_pressure_q16;
  std::int32_t social_relief_q16;
  std::uint32_t social_relief_authenticated;
};

struct ProgramStep {
  std::uint32_t node;
  std::uint32_t channel;
  std::uint32_t due_offset;
};

struct Program {
  std::uint64_t identity;
  std::uint64_t initiation_participation_identity;
  std::uint64_t initiation_parent_eligibility_ref;
  std::uint32_t initiation_expiry_tick;
  std::uint32_t depth;
  ProgramStep steps[kMaxProgramSteps];
  std::uint32_t step_count;
};
static_assert(std::is_standard_layout_v<Program> && std::is_trivial_v<Program>);

struct ProgramBankEntry {
  Program program;
  PredictiveProfile profile;
  std::uint32_t occupied;
};

struct ProgramBank {
  ProgramBankEntry entries[kProgramBankCapacity];
  std::uint32_t count;
  std::uint32_t evictions;
};
static_assert(std::is_standard_layout_v<ProgramBank> && std::is_trivial_v<ProgramBank>);

DIRECT_CAUSAL_PROGRAM_HD inline bool control_ready(const PredictiveProfile& profile);
DIRECT_CAUSAL_PROGRAM_HD inline bool control_supported(const PredictiveProfile& profile);

DIRECT_CAUSAL_PROGRAM_HD inline bool retention_less(
    const PredictiveProfile& left, const PredictiveProfile& right) {
  const std::uint32_t left_consequence = left.outcome_samples != 0u ? 1u : 0u;
  const std::uint32_t right_consequence = right.outcome_samples != 0u ? 1u : 0u;
  if (left_consequence != right_consequence) return left_consequence < right_consequence;
  const std::uint32_t left_control = control_supported(left) ? 1u : 0u;
  const std::uint32_t right_control = control_supported(right) ? 1u : 0u;
  if (left_control != right_control) return left_control < right_control;
  if (left.control_history_q16 != right.control_history_q16)
    return left.control_history_q16 < right.control_history_q16;
  if (left.outcome_samples != right.outcome_samples) return left.outcome_samples < right.outcome_samples;
  if (left.control_successes != right.control_successes) return left.control_successes < right.control_successes;
  if (left.exposures != right.exposures) return left.exposures < right.exposures;
  return left.structure_identity > right.structure_identity;
}

DIRECT_CAUSAL_PROGRAM_HD inline ProgramBankEntry* find_program(
    ProgramBank* bank, std::uint64_t identity) {
  if (bank == nullptr || identity == 0u) return nullptr;
  for (std::uint32_t i = 0; i < kProgramBankCapacity; ++i)
    if (bank->entries[i].occupied != 0u && bank->entries[i].program.identity == identity)
      return &bank->entries[i];
  return nullptr;
}

DIRECT_CAUSAL_PROGRAM_HD inline ProgramBankEntry* admit_program(
    ProgramBank* bank, const Program& program, std::uint64_t* evicted_identity) {
  if (evicted_identity != nullptr) *evicted_identity = 0u;
  if (bank == nullptr || program.identity == 0u) return nullptr;
  if (auto* existing = find_program(bank, program.identity)) return existing;
  for (std::uint32_t i = 0; i < kProgramBankCapacity; ++i)
    if (bank->entries[i].occupied == 0u) {
      bank->entries[i] = ProgramBankEntry{};
      bank->entries[i].program = program;
      bank->entries[i].profile.structure_identity = program.identity;
      bank->entries[i].occupied = 1u;
      ++bank->count;
      return &bank->entries[i];
    }
  std::uint32_t victim = 0u;
  for (std::uint32_t i = 1u; i < kProgramBankCapacity; ++i)
    if (retention_less(bank->entries[i].profile, bank->entries[victim].profile)) victim = i;
  if (evicted_identity != nullptr) *evicted_identity = bank->entries[victim].program.identity;
  bank->entries[victim] = ProgramBankEntry{};
  bank->entries[victim].program = program;
  bank->entries[victim].profile.structure_identity = program.identity;
  bank->entries[victim].occupied = 1u;
  ++bank->evictions;
  return &bank->entries[victim];
}

DIRECT_CAUSAL_PROGRAM_HD inline constexpr std::int32_t clamp_q16(std::int32_t value) {
  return value < 0 ? 0 : (value > kQ16One ? kQ16One : value);
}

DIRECT_CAUSAL_PROGRAM_HD inline std::int32_t ema(std::int32_t old_value, std::int32_t sample,
                        std::uint32_t count) {
  return count <= 1u ? sample
                     : old_value + (sample - old_value) /
                                       static_cast<std::int32_t>(count);
}

DIRECT_CAUSAL_PROGRAM_HD inline ContextOutcome* context_slot(PredictiveProfile* profile,
                                    std::uint32_t context) {
  if (profile == nullptr) return nullptr;
  for (std::uint32_t i = 0; i < profile->context_count; ++i)
    if (profile->contexts[i].context == context) return &profile->contexts[i];
  if (profile->context_count >= kMaxProgramContexts) return nullptr;
  auto* out = &profile->contexts[profile->context_count++];
  *out = ContextOutcome{};
  out->context = context;
  return out;
}

DIRECT_CAUSAL_PROGRAM_HD inline void observe_use(PredictiveProfile* profile, std::uint32_t duration_ticks,
                        std::int32_t effort_q16) {
  if (profile == nullptr) return;
  ++profile->exposures;
  profile->accessibility_q16 = clamp_q16(profile->accessibility_q16 + kQ16One / 8);
  profile->duration_mean_q16 = ema(profile->duration_mean_q16,
                                   static_cast<std::int32_t>(duration_ticks) * kQ16One,
                                   profile->exposures);
  profile->effort_mean_q16 = ema(profile->effort_mean_q16, effort_q16,
                                 profile->exposures);
}

DIRECT_CAUSAL_PROGRAM_HD inline void observe_return(PredictiveProfile* profile, std::uint32_t context,
                           std::int32_t outcome_q16,
                           std::int32_t somatic_q16,
                           bool independent) {
  if (profile == nullptr || !independent) return;
  ++profile->outcome_samples;
  profile->outcome_mean_q16 = ema(profile->outcome_mean_q16, outcome_q16,
                                  profile->outcome_samples);
  profile->somatic_mean_q16 = ema(profile->somatic_mean_q16, somatic_q16,
                                  profile->outcome_samples);
  if (auto* slot = context_slot(profile, context)) {
    ++slot->samples;
    slot->outcome_mean_q16 = ema(slot->outcome_mean_q16, outcome_q16, slot->samples);
    slot->somatic_mean_q16 = ema(slot->somatic_mean_q16, somatic_q16, slot->samples);
  }
}

DIRECT_CAUSAL_PROGRAM_HD inline bool control_ready(const PredictiveProfile& profile) {
  return profile.control_successes >= kControlSuccessQuorum &&
         profile.controllability_q16 >= kQ16One / 2;
}

DIRECT_CAUSAL_PROGRAM_HD inline bool control_supported(const PredictiveProfile& profile) {
  return control_ready(profile) || profile.control_history_q16 >= kQ16One / 2;
}

DIRECT_CAUSAL_PROGRAM_HD inline void observe_control(PredictiveProfile* profile, bool public_action,
                            bool independent_return) {
  if (profile == nullptr) return;
  const bool was_ready = control_ready(*profile);
  if (public_action) {
    ++profile->control_attempts;
    if (independent_return) ++profile->control_successes;
  } else {
    ++profile->background_attempts;
    if (independent_return) ++profile->background_successes;
  }
  const std::int32_t action_rate = profile->control_attempts == 0u ? 0 :
      static_cast<std::int32_t>((static_cast<std::uint64_t>(profile->control_successes) * kQ16One) /
                                profile->control_attempts);
  const std::int32_t background_rate = profile->background_attempts == 0u ? 0 :
      static_cast<std::int32_t>((static_cast<std::uint64_t>(profile->background_successes) * kQ16One) /
                                profile->background_attempts);
  profile->controllability_q16 = action_rate > background_rate ? action_rate - background_rate : 0;
  const bool now_ready = control_ready(*profile);
  if (now_ready && !was_ready) {
    profile->control_history_q16 = kQ16One;
  } else if (now_ready && public_action && independent_return) {
    profile->control_history_q16 = clamp_q16(profile->control_history_q16 + kControlHistoryStepQ16);
  } else if (public_action != independent_return) {
    profile->control_history_q16 = profile->control_history_q16 > kControlHistoryStepQ16
        ? profile->control_history_q16 - kControlHistoryStepQ16 : 0;
  }
}

DIRECT_CAUSAL_PROGRAM_HD inline std::int32_t context_outcome(const PredictiveProfile& profile,
                                    std::uint32_t context) {
  for (std::uint32_t i = 0; i < profile.context_count; ++i)
    if (profile.contexts[i].context == context && profile.contexts[i].samples != 0u)
      return profile.contexts[i].outcome_mean_q16;
  return profile.outcome_mean_q16;
}

DIRECT_CAUSAL_PROGRAM_HD inline std::int32_t context_somatic(const PredictiveProfile& profile,
                                    std::uint32_t context) {
  for (std::uint32_t i = 0; i < profile.context_count; ++i)
    if (profile.contexts[i].context == context && profile.contexts[i].samples != 0u)
      return profile.contexts[i].somatic_mean_q16;
  return profile.somatic_mean_q16;
}

DIRECT_CAUSAL_PROGRAM_HD inline std::int64_t prospective_score(const PredictiveProfile& profile,
                                      std::uint32_t context,
                                      const CurrentState& state) {
  if (!control_supported(profile)) return -(1ll << 60);
  std::int64_t score = context_outcome(profile, context) +
                       context_somatic(profile, context) +
                       profile.accessibility_q16 / 4 -
                       profile.effort_mean_q16 / 8;
  const std::int32_t urgency = clamp_q16(state.urgency_q16);
  const std::int32_t pressure = clamp_q16(state.resource_pressure_q16);
  const std::int32_t relief = state.social_relief_authenticated != 0u
      ? clamp_q16(state.social_relief_q16)
      : 0;
  const std::int32_t effective_pressure = pressure > relief ? pressure - relief : 0;
  score -= (static_cast<std::int64_t>(profile.duration_mean_q16) * urgency) /
           (4ll * kQ16One);
  // Ordinary effort has an ordinary opportunity cost.  Endogenous load adds a
  // nonlinear surcharge as remaining operating margin collapses; at zero load
  // this term is exactly zero. Authenticated relief may widen that margin.
  if (effective_pressure != 0) {
    const std::int32_t capacity_margin =
        effective_pressure >= 15 * (kQ16One / 16)
            ? kQ16One / 16
            : kQ16One - effective_pressure;
    score -= (static_cast<std::int64_t>(profile.effort_mean_q16) *
              effective_pressure) /
             capacity_margin;
  }
  return score;
}

DIRECT_CAUSAL_PROGRAM_HD inline bool initiation_current(const Program& program,
                               std::uint64_t participation_identity,
                               std::uint32_t current_tick) {
  return program.identity != 0u &&
         program.initiation_participation_identity != 0u &&
         program.initiation_parent_eligibility_ref != 0u &&
         participation_identity == program.initiation_participation_identity &&
         current_tick <= program.initiation_expiry_tick;
}

#undef DIRECT_CAUSAL_PROGRAM_HD

}  // namespace substrate::direct_causal_program

#endif
