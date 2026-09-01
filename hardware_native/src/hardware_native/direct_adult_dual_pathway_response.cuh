#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DUAL_PATHWAY_RESPONSE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DUAL_PATHWAY_RESPONSE_CUH

#include "direct_adult_core_constants.cuh"
#include "direct_adult_volitional_veto.cuh"

namespace substrate::direct_adult_core {

// Dual-pathway response to one threat stimulus, both regimes veto-respecting.
// The REACTIVE regime answers high affect-ledger pressure: it commits at the
// generation tick when the action passes the standard gate. The PROACTIVE
// regime spends bounded deliberation ticks before the same gate. Pressure
// selects the regime; only the gate commits the action.
enum class ResidentResponsePathway : std::uint32_t {
  none = 0u,
  reactive = 1u,
  proactive = 2u,
};

struct ResidentDualPathwayDecision {
  direct_network::ResidentVolitionalVetoReceipt gate;
  ResidentResponsePathway pathway;
  std::uint32_t pressure_q16;
  std::uint32_t deliberation_ticks;
};

__device__ inline std::uint32_t resident_threat_pressure_q16(
    std::int32_t damage_q16, std::int32_t stress_q16) {
  const std::int64_t combined =
      static_cast<std::int64_t>(damage_q16) + stress_q16;
  if (combined <= 0) return 0u;
  return combined > kQ16One ? static_cast<std::uint32_t>(kQ16One)
                            : static_cast<std::uint32_t>(combined);
}

// Selects the pathway from affect-ledger pressure and runs the standard
// precommit gate under that regime's latency. Reactive commits at
// generation_tick; proactive at generation_tick + deliberation_ticks.
__device__ inline bool decide_resident_dual_pathway(
    const direct_network::ResidentProspectiveTrajectory& prospective,
    std::uint32_t expected_prediction_identity,
    std::uint64_t expected_parent_occurrence_identity,
    std::uint64_t expected_parent_revision_identity,
    std::uint64_t expected_participation_identity,
    std::uint32_t expected_context_signature,
    std::int32_t damage_q16, std::int32_t stress_q16,
    std::uint32_t pressure_threshold_q16,
    const direct_network::ResidentVolitionalBrakeEvidence* brake_evidence,
    std::uint32_t mismatch_brake_threshold_q16,
    std::uint32_t deliberation_ticks,
    direct_network::ResidentPersistentCommitment* persistent,
    ResidentDualPathwayDecision* out) {
  if (out == nullptr || persistent == nullptr || deliberation_ticks == 0u)
    return false;
  *out = ResidentDualPathwayDecision{};
  out->pressure_q16 =
      resident_threat_pressure_q16(damage_q16, stress_q16);
  out->pathway = out->pressure_q16 >= pressure_threshold_q16
      ? ResidentResponsePathway::reactive
      : ResidentResponsePathway::proactive;
  const std::uint32_t decision_tick =
      prospective.generation_tick +
      (out->pathway == ResidentResponsePathway::proactive
           ? deliberation_ticks
           : 0u);
  if (!direct_network::resident_volitional_precommit_gate(
          prospective, expected_prediction_identity,
          expected_parent_occurrence_identity,
          expected_parent_revision_identity,
          expected_participation_identity, expected_context_signature,
          brake_evidence, mismatch_brake_threshold_q16, decision_tick, persistent,
          &out->gate))
    return false;
  if (out->gate.decision !=
      direct_network::ResidentVolitionalDecision::commit) {
    *out = ResidentDualPathwayDecision{};
    out->pressure_q16 = resident_threat_pressure_q16(damage_q16, stress_q16);
    out->pathway = out->pressure_q16 >= pressure_threshold_q16
        ? ResidentResponsePathway::reactive
        : ResidentResponsePathway::proactive;
    out->gate.decision = direct_network::ResidentVolitionalDecision::refused;
    return true;  // selection made; the gate refused the action either way
  }
  out->deliberation_ticks =
      out->pathway == ResidentResponsePathway::proactive ? deliberation_ticks
                                                         : 0u;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
