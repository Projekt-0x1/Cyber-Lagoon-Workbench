#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SELF_DECEPTION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SELF_DECEPTION_CUH

#include "direct_adult_volitional_veto.cuh"

namespace substrate::direct_adult_core {

// Prospective evaluation suppression of internal conflict traces: a
// trajectory that carries conflict but still passes the volitional gate
// commits in ONE decision pass, and the persistent commitment structurally
// omits every conflict field -- downstream observers of the commitment
// cannot distinguish it from an unconflicted one. The full evidence remains
// wherever it was observed (exact history is never touched). Suppression has
// a hard boundary: above-threshold conflict still vetoes, so self-deception
// can never convert a refusal into a commit.
struct ResidentSelfDeceptionReceipt {
  direct_network::ResidentVolitionalVetoReceipt gate;
  std::uint32_t suppressed_conflict_q16;
  std::uint32_t reevaluation_rounds_saved;
};
static_assert(std::is_trivial_v<ResidentSelfDeceptionReceipt> &&
              std::is_standard_layout_v<ResidentSelfDeceptionReceipt>);

__device__ inline bool strategic_resident_self_deception_commit(
    const direct_network::ResidentProspectiveTrajectory& prospective,
    std::uint64_t expected_prediction_identity,
    std::uint64_t expected_parent_occurrence_identity,
    std::uint64_t expected_parent_revision_identity,
    std::uint64_t expected_participation_identity,
    std::uint32_t expected_context_signature,
    const direct_network::ResidentVolitionalBrakeEvidence* brake_evidence,
    std::uint32_t mismatch_brake_threshold_q16,
    std::uint32_t decision_tick,
    direct_network::ResidentPersistentCommitment* persistent,
    ResidentSelfDeceptionReceipt* out) {
  if (out == nullptr) return false;
  *out = ResidentSelfDeceptionReceipt{};
  direct_network::ResidentPersistentCommitment staged{};
  if (!direct_network::resident_volitional_precommit_gate(
          prospective, expected_prediction_identity,
          expected_parent_occurrence_identity, expected_parent_revision_identity,
          expected_participation_identity, expected_context_signature,
          brake_evidence, mismatch_brake_threshold_q16,
          decision_tick, &staged, &out->gate))
    return false;
  if (out->gate.decision ==
      direct_network::ResidentVolitionalDecision::commit) {
    // Single-pass commit: the conflict internals were evaluated and held out
    // of the committed record entirely.
    out->suppressed_conflict_q16 = brake_evidence == nullptr
        ? 0u : brake_evidence->mismatch_magnitude_q16;
    out->reevaluation_rounds_saved = 1u;
    *persistent = staged;
  }
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
