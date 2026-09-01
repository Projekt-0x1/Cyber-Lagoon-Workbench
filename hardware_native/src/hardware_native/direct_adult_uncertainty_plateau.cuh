#ifndef HARDWARE_NATIVE_DIRECT_ADULT_UNCERTAINTY_PLATEAU_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_UNCERTAINTY_PLATEAU_CUH

#include "direct_adult_core.cuh"

namespace substrate::direct_adult_core {

// Dopamine search signal over outcome uncertainty: for a binary prospect
// with reward probability p = numerator/denominator, the resident signal is
// 4*p*(1-p) in Q16 -- exactly kQ16One at the p=0.5 plateau, exactly zero at
// both certainty poles, symmetric under p <-> 1-p, monotone on each side.
// The signal scales PROSPECTIVE EXPLORATION DRIVE only; hedonic stores are
// structurally absent from this relation (wanting/liking dissociation).
inline constexpr std::uint32_t kResidentUncertaintyOneQ16 =
    substrate::direct_adult_core::kQ16One;

__device__ inline std::uint32_t resident_outcome_uncertainty_q16(
    std::uint64_t reward_outcomes, std::uint64_t total_outcomes) {
  if (total_outcomes == 0u || reward_outcomes > total_outcomes) return 0u;
  const std::int64_t one = static_cast<std::int64_t>(kResidentUncertaintyOneQ16);
  const std::int64_t p = static_cast<std::int64_t>(
      reward_outcomes * static_cast<std::uint64_t>(one) / total_outcomes);
  const std::int64_t anti = one - p;
  const std::int64_t signal = (4 * p * anti) / one;
  return static_cast<std::uint32_t>(signal > 0 ? signal : 0);
}

// Scales prospective exploration drive by the uncertainty signal, bounded
// above by kQ16One. Consumption-side stores never appear here.
__device__ inline std::uint32_t scale_resident_exploration_drive_q16(
    std::uint32_t base_drive_q16, std::uint32_t uncertainty_signal_q16) {
  const std::int64_t drive = static_cast<std::int64_t>(base_drive_q16);
  const std::int64_t lift =
      (drive * static_cast<std::int64_t>(uncertainty_signal_q16)) /
      static_cast<std::int64_t>(kResidentUncertaintyOneQ16);
  std::int64_t scaled = drive + lift;
  const std::int64_t one = static_cast<std::int64_t>(kResidentUncertaintyOneQ16);
  if (scaled > one) scaled = one;
  return static_cast<std::uint32_t>(scaled);
}

struct ResidentMotorOutcomeUncertaintyV1 {
  std::uint32_t reward_outcomes;
  std::uint32_t total_outcomes;
  std::uint32_t signal_q16;
};

// Read only already-settled public action outcomes for this motor node.
// This is bounded resident evidence, not host-authored probability. Unknown
// or still-pending tickets contribute nothing. Positive settled consequence
// is the rewarded branch; zero/negative settled consequence is the other
// observed branch.
__device__ inline ResidentMotorOutcomeUncertaintyV1
resident_motor_outcome_uncertainty(
    const AsynchronousTicket* tickets, std::uint32_t ticket_capacity,
    std::uint32_t motor_node) {
  ResidentMotorOutcomeUncertaintyV1 result{};
  if (tickets == nullptr) return result;
  for (std::uint32_t i = 0u; i < ticket_capacity; ++i) {
    const AsynchronousTicket& ticket = tickets[i];
    if (ticket.ticket_id == 0u || ticket.settled == 0u ||
        ticket.motor_node != motor_node)
      continue;
    ++result.total_outcomes;
    result.reward_outcomes += ticket.settled_reward_q16 > 0 ? 1u : 0u;
  }
  result.signal_q16 = resident_outcome_uncertainty_q16(
      result.reward_outcomes, result.total_outcomes);
  return result;
}

// Uncertainty changes only the prospective action threshold. It cannot write
// participation, causal credit, affect/liking stores, or settled tickets. At
// maximal p=0.5 uncertainty it contributes the same bounded 1/8-Q16 pursuit
// lift used by the resident wanting gate; certainty contributes zero.
__device__ inline std::int32_t resident_uncertainty_pursuit_threshold_q16(
    std::int32_t base_threshold_q16, std::uint32_t uncertainty_signal_q16) {
  const std::int32_t lift = static_cast<std::int32_t>(
      uncertainty_signal_q16 > kResidentUncertaintyOneQ16
          ? kResidentUncertaintyOneQ16 / 8u
          : uncertainty_signal_q16 / 8u);
  return base_threshold_q16 > lift ? base_threshold_q16 - lift : 0;
}

}  // namespace substrate::direct_adult_core

#endif
