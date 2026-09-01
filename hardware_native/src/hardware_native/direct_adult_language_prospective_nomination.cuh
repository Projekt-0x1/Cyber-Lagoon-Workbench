#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_PROSPECTIVE_NOMINATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_PROSPECTIVE_NOMINATION_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_volitional_veto.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_adult_core {

#if defined(__CUDACC__)
#define DIRECT_LANGUAGE_PROSPECTIVE_HD __host__ __device__
#else
#define DIRECT_LANGUAGE_PROSPECTIVE_HD
#endif

// Convert a lawful endogenous language Recipe occurrence into ordinary
// prospective work. This function cannot commit or publish an action. It binds
// the candidate to one current actual parent plus prediction/resource state so
// the normal volitional/arbitration machinery remains the only commit path.
DIRECT_LANGUAGE_PROSPECTIVE_HD inline bool
resident_language_nomination_to_prospective(
    const ResidentRecipeOccurrence& nomination,
    const ResidentRecipeOccurrence& actual_parent,
    std::uint64_t prediction_identity,
    std::uint32_t current_tick,
    direct_network::ResidentProspectiveTrajectory* out) {
  using namespace direct_network;
  if (out == nullptr || prediction_identity == 0u || current_tick == 0u ||
      nomination.state != kResidentRecipeOccurrenceLive ||
      nomination.lineage_kind != ResidentOccurrenceLineageKind::endogenous ||
      nomination.authority != DirectParticipationAuthority::none ||
      nomination.eligibility_q16 != 0 ||
      nomination.occurrence_identity == 0u || nomination.logical_recipe_id == 0u ||
      nomination.revision_identity == 0u || nomination.participation_identity == 0u ||
      actual_parent.state != kResidentRecipeOccurrenceLive ||
      actual_parent.lineage_kind != ResidentOccurrenceLineageKind::actual ||
      actual_parent.authority == DirectParticipationAuthority::none ||
      actual_parent.occurrence_identity == 0u || actual_parent.revision_identity == 0u ||
      actual_parent.participation_identity == 0u ||
      nomination.source_identity == 0u ||
      nomination.context_signature != actual_parent.context_signature ||
      current_tick < nomination.timestamp)
    return false;

  std::uint64_t trajectory_identity = exact_history_fold_word(
      nomination.occurrence_identity, prediction_identity);
  trajectory_identity = exact_history_fold_word(
      trajectory_identity, actual_parent.occurrence_identity);
  trajectory_identity = exact_history_fold_word(
      trajectory_identity, nomination.context_signature);
  if (trajectory_identity == 0u) trajectory_identity = nomination.occurrence_identity;

  ResidentProspectiveTrajectory candidate{};
  candidate.trajectory_identity = trajectory_identity;
  candidate.prediction_identity = prediction_identity;
  candidate.parent_occurrence_identity = actual_parent.occurrence_identity;
  candidate.parent_revision_identity = actual_parent.revision_identity;
  // Current participation authority comes only from the current actual parent.
  // The nomination's source/participation remain historical nomination provenance.
  candidate.participation_identity = actual_parent.participation_identity;
  candidate.context_signature = nomination.context_signature;
  candidate.generation_tick = current_tick;
  *out = candidate;
  return true;
}

#undef DIRECT_LANGUAGE_PROSPECTIVE_HD

}  // namespace substrate::direct_adult_core

#endif
