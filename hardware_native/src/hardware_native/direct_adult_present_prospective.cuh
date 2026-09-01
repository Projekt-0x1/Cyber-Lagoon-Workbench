#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PRESENT_PROSPECTIVE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PRESENT_PROSPECTIVE_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"

#include "hardware_native/direct_adult_alternative_futures.cuh"

namespace substrate::direct_adult_core {

// THE DECLARED TOLERANCE. A branch whose projected state sits within this
// many q16 units of the parent's present output intersects the present; a
// wider gap is divergence. Device constant -- no host path supplies it.
inline constexpr std::int64_t kPresentProspectiveToleranceQ16 = 1u << 9;

struct alignas(8) ResidentPresentProspectiveReading {
  std::uint64_t branch_identity;
  std::int64_t delta_q16;  // projected minus present
  std::uint32_t branch_ordinal;
  std::uint32_t aligned;
};
static_assert(
    std::is_standard_layout_v<ResidentPresentProspectiveReading> &&
    std::is_trivial_v<ResidentPresentProspectiveReading>);

struct alignas(8) ResidentPresentProspectiveReceipt {
  std::uint64_t parent_occurrence_identity;
  std::int64_t present_q16;
  ResidentPresentProspectiveReading
      readings[kResidentAlternativeBankCapacity];
  std::uint32_t reading_count;
  std::uint32_t aligned_count;
  std::uint32_t diverged_count;
  std::uint32_t refusals;
};
static_assert(
    std::is_standard_layout_v<ResidentPresentProspectiveReceipt> &&
    std::is_trivial_v<ResidentPresentProspectiveReceipt>);

// Aligns one generated branch set against the parent's PRESENT output. Reads
// only endogenous prediction state and the live actual Occurrence; writes
// only the receipt. The classification follows the delta's magnitude against
// the declared tolerance, never its sign alone, and the same branches
// re-classify against a moved present without regeneration.
DIRECT_ADULT_HD inline bool align_resident_present_prospective(
    const ResidentActualFrontier& actual_frontier,
    std::uint32_t parent_frontier_slot,
    const ResidentAlternativeFuturesFrontier& branches,
    ResidentPresentProspectiveReceipt* receipt) {
  if (receipt == nullptr ||
      parent_frontier_slot >= kResidentActualFrontierCapacity) {
    if (receipt != nullptr) ++receipt->refusals;
    return false;
  }
  const ResidentActualFrontierEntry& parent =
      actual_frontier.entries[parent_frontier_slot];
  if (parent.state != ResidentActualFrontierState::live ||
      parent.occurrence.state != kResidentRecipeOccurrenceLive ||
      parent.occurrence.occurrence_identity == 0u ||
      branches.live_count == 0u ||
      branches.parent_occurrence_identity !=
          parent.occurrence.occurrence_identity) {
    ++receipt->refusals;
    return false;
  }

  ResidentPresentProspectiveReceipt out{};
  out.parent_occurrence_identity = parent.occurrence.occurrence_identity;
  out.present_q16 = parent.output_q16;
  for (std::uint32_t i = 0u; i < kResidentAlternativeBankCapacity; ++i) {
    const auto& branch = branches.branches[i];
    if (branch.state != ResidentAlternativeState::live) continue;
    ResidentPresentProspectiveReading reading{};
    reading.branch_identity = branch.branch_identity;
    reading.branch_ordinal = branch.branch_ordinal;
    reading.delta_q16 = static_cast<std::int64_t>(branch.projected_state_q16) -
                        static_cast<std::int64_t>(parent.output_q16);
    const std::uint64_t magnitude =
        resident_prediction_magnitude(reading.delta_q16);
    reading.aligned = magnitude <= kPresentProspectiveToleranceQ16 ? 1u : 0u;
    out.readings[out.reading_count++] = reading;
    if (reading.aligned)
      ++out.aligned_count;
    else
      ++out.diverged_count;
  }
  if (out.reading_count == 0u) {
    ++receipt->refusals;
    return false;
  }
  *receipt = out;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_PRESENT_PROSPECTIVE_CUH
