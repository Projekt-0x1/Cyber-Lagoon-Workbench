#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_COMPETITION_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_COMPETITION_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_causal_program.cuh"

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_PROGRAM_COMPETITION_HD __host__ __device__
#else
#define DIRECT_PROGRAM_COMPETITION_HD
#endif

inline constexpr std::int32_t kProgramCompetitionBaseMarginQ16 = kQ16One / 8;

enum class ProgramCompetitionDecision : std::uint32_t {
  none = 0u,
  unique_leader = 1u,
  unresolved = 2u,
};

struct ProgramCompetitionReceipt {
  ProgramCompetitionDecision decision;
  std::uint64_t leader_identity;
  std::uint64_t runner_up_identity;
  std::int64_t leader_drive_q16;
  std::int64_t runner_up_drive_q16;
  std::int64_t required_margin_q16;
  std::uint32_t considered;
  std::uint32_t unsupported;
  std::uint32_t stale;
};
static_assert(std::is_standard_layout_v<ProgramCompetitionReceipt> &&
              std::is_trivial_v<ProgramCompetitionReceipt>);

DIRECT_PROGRAM_COMPETITION_HD inline std::uint32_t program_context_samples(
    const PredictiveProfile& profile, std::uint32_t context) {
  for (std::uint32_t i = 0u; i < profile.context_count; ++i)
    if (profile.contexts[i].context == context) return profile.contexts[i].samples;
  return 0u;
}

DIRECT_PROGRAM_COMPETITION_HD inline std::uint64_t program_prediction_fold(
    std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
  return h == 0u ? 1u : h;
}

DIRECT_PROGRAM_COMPETITION_HD inline std::uint64_t program_prediction_identity(
    const ProgramBankEntry& entry, std::uint32_t context) {
  if (entry.occupied == 0u || entry.program.identity == 0u ||
      entry.profile.structure_identity != entry.program.identity ||
      entry.profile.outcome_samples == 0u)
    return 0u;
  std::uint64_t h = program_prediction_fold(0x70726f6770726564ull, entry.program.identity);
  h = program_prediction_fold(h, context);
  h = program_prediction_fold(h, program_context_samples(entry.profile, context));
  h = program_prediction_fold(h, entry.profile.outcome_samples);
  h = program_prediction_fold(h, static_cast<std::uint32_t>(context_outcome(entry.profile, context)));
  h = program_prediction_fold(h, static_cast<std::uint32_t>(context_somatic(entry.profile, context)));
  return h;
}

DIRECT_PROGRAM_COMPETITION_HD inline std::int64_t program_competition_margin_q16(
    std::int32_t urgency_q16) {
  const std::int32_t urgency = clamp_q16(urgency_q16);
  return (static_cast<std::int64_t>(kProgramCompetitionBaseMarginQ16) *
          (kQ16One - urgency)) / kQ16One;
}

DIRECT_PROGRAM_COMPETITION_HD inline ProgramCompetitionReceipt arbitrate_program_bank(
    const ProgramBank* bank, std::uint32_t context,
    std::uint64_t current_participation_identity, std::uint32_t current_tick,
    const CurrentState& state) {
  ProgramCompetitionReceipt out{};
  out.decision = ProgramCompetitionDecision::none;
  out.required_margin_q16 = program_competition_margin_q16(state.urgency_q16);
  if (bank == nullptr || current_participation_identity == 0u) return out;

  bool have_leader = false;
  std::int64_t leader_drive = 0;
  std::int64_t runner_drive = 0;
  std::uint64_t leader_identity = 0u;
  std::uint64_t runner_identity = 0u;

  for (std::uint32_t i = 0u; i < kProgramBankCapacity; ++i) {
    const ProgramBankEntry& entry = bank->entries[i];
    if (entry.occupied == 0u || entry.program.identity == 0u) continue;
    if (!initiation_current(entry.program, current_participation_identity, current_tick)) {
      ++out.stale;
      continue;
    }
    if (!control_supported(entry.profile)) {
      ++out.unsupported;
      continue;
    }
    const std::int64_t drive = prospective_score(entry.profile, context, state);
    ++out.considered;
    if (!have_leader || drive > leader_drive) {
      runner_drive = leader_drive;
      runner_identity = leader_identity;
      leader_drive = drive;
      leader_identity = entry.program.identity;
      have_leader = true;
    } else if (runner_identity == 0u || drive > runner_drive) {
      runner_drive = drive;
      runner_identity = entry.program.identity;
    }
  }

  if (!have_leader) return out;
  out.leader_drive_q16 = leader_drive;
  out.leader_identity = leader_identity;
  if (runner_identity == 0u) {
    out.decision = ProgramCompetitionDecision::unique_leader;
    return out;
  }
  out.runner_up_drive_q16 = runner_drive;
  out.runner_up_identity = runner_identity;
  const std::int64_t separation = leader_drive - runner_drive;
  if (separation <= 0 || separation < out.required_margin_q16) {
    out.decision = ProgramCompetitionDecision::unresolved;
    out.leader_identity = 0u;
    return out;
  }
  out.decision = ProgramCompetitionDecision::unique_leader;
  return out;
}

#undef DIRECT_PROGRAM_COMPETITION_HD
}  // namespace substrate::direct_causal_program
#endif
