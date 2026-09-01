#ifndef HARDWARE_NATIVE_DIRECT_ADULT_VOLITIONAL_VETO_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_VOLITIONAL_VETO_CUH

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define DIRECT_VOLITION_HD __host__ __device__
#else
#define DIRECT_VOLITION_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kVolitionalQ16One = 1u << 16u;

enum class ResidentVolitionalDecision : std::uint32_t {
  refused = 0u,
  commit = 1u,
  veto_prediction_mismatch = 2u,
};

// A proposal is already resident work. The gate cannot invent an action: it
// can only bind an exact prospective identity and stop its persistent commit.
struct alignas(8) ResidentProspectiveTrajectory {
  std::uint64_t trajectory_identity;
  std::uint64_t prediction_identity;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t parent_revision_identity;
  std::uint64_t participation_identity;
  std::uint32_t context_signature;
  std::uint32_t generation_tick;
};

struct alignas(8) ResidentVolitionalBrakeEvidence {
  std::uint64_t evidence_identity;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t participation_identity;
  std::uint32_t context_signature;
  std::uint32_t resident_tick;
  std::uint32_t mismatch_magnitude_q16;
  std::uint32_t mismatch_kind;
};

struct alignas(8) ResidentPersistentCommitment {
  std::uint64_t trajectory_identity;
  std::uint64_t prediction_identity;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t parent_revision_identity;
  std::uint64_t participation_identity;
  std::uint64_t generation;
  std::uint32_t context_signature;
  std::uint32_t commit_tick;
};

struct alignas(8) ResidentVolitionalVetoReceipt {
  std::uint64_t trajectory_identity;
  std::uint64_t prediction_identity;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t participation_identity;
  std::uint64_t persistent_generation_before;
  std::uint64_t persistent_generation_after;
  std::uint64_t brake_evidence_identity;
  std::uint32_t brake_magnitude_q16;
  std::uint32_t brake_kind;
  ResidentVolitionalDecision decision;
  std::uint32_t decision_tick;
};

static_assert(std::is_trivial_v<ResidentProspectiveTrajectory> &&
              std::is_standard_layout_v<ResidentProspectiveTrajectory> &&
              std::has_unique_object_representations_v<
                  ResidentProspectiveTrajectory>);
static_assert(std::is_trivial_v<ResidentVolitionalBrakeEvidence> &&
              std::is_standard_layout_v<ResidentVolitionalBrakeEvidence> &&
              std::has_unique_object_representations_v<ResidentVolitionalBrakeEvidence>);
static_assert(std::is_trivial_v<ResidentPersistentCommitment> &&
              std::is_standard_layout_v<ResidentPersistentCommitment> &&
              std::has_unique_object_representations_v<
                  ResidentPersistentCommitment>);
static_assert(std::is_trivial_v<ResidentVolitionalVetoReceipt> &&
              std::is_standard_layout_v<ResidentVolitionalVetoReceipt> &&
              std::has_unique_object_representations_v<
                  ResidentVolitionalVetoReceipt>);

DIRECT_VOLITION_HD inline bool resident_volitional_precommit_gate(
    const ResidentProspectiveTrajectory& prospective,
    std::uint64_t expected_prediction_identity,
    std::uint64_t expected_parent_occurrence_identity,
    std::uint64_t expected_parent_revision_identity,
    std::uint64_t expected_participation_identity,
    std::uint32_t expected_context_signature,
    const ResidentVolitionalBrakeEvidence* brake_evidence,
    std::uint32_t mismatch_brake_threshold_q16,
    std::uint32_t decision_tick,
    ResidentPersistentCommitment* persistent,
    ResidentVolitionalVetoReceipt* receipt) {
  if (persistent == nullptr || receipt == nullptr ||
      prospective.trajectory_identity == 0u || prospective.prediction_identity == 0u ||
      prospective.parent_occurrence_identity == 0u || prospective.parent_revision_identity == 0u ||
      prospective.participation_identity == 0u ||
      prospective.prediction_identity != expected_prediction_identity ||
      prospective.parent_occurrence_identity != expected_parent_occurrence_identity ||
      prospective.parent_revision_identity != expected_parent_revision_identity ||
      prospective.participation_identity != expected_participation_identity ||
      prospective.context_signature != expected_context_signature ||
      mismatch_brake_threshold_q16 == 0u || mismatch_brake_threshold_q16 > kVolitionalQ16One ||
      decision_tick < prospective.generation_tick)
    return false;
  ResidentVolitionalVetoReceipt candidate{};
  candidate.trajectory_identity = prospective.trajectory_identity;
  candidate.prediction_identity = prospective.prediction_identity;
  candidate.parent_occurrence_identity = prospective.parent_occurrence_identity;
  candidate.participation_identity = prospective.participation_identity;
  candidate.persistent_generation_before = persistent->generation;
  candidate.decision_tick = decision_tick;
  if (brake_evidence != nullptr && brake_evidence->evidence_identity != 0u) {
    if (brake_evidence->parent_occurrence_identity != expected_parent_occurrence_identity ||
        brake_evidence->participation_identity != expected_participation_identity ||
        brake_evidence->context_signature != expected_context_signature ||
        brake_evidence->resident_tick > decision_tick ||
        brake_evidence->mismatch_magnitude_q16 > kVolitionalQ16One ||
        brake_evidence->mismatch_kind == 0u)
      return false;
    candidate.brake_evidence_identity = brake_evidence->evidence_identity;
    candidate.brake_magnitude_q16 = brake_evidence->mismatch_magnitude_q16;
    candidate.brake_kind = brake_evidence->mismatch_kind;
  }
  if (candidate.brake_evidence_identity != 0u &&
             candidate.brake_magnitude_q16 >= mismatch_brake_threshold_q16) {
    candidate.decision = ResidentVolitionalDecision::veto_prediction_mismatch;
  } else {
    ResidentPersistentCommitment commitment{};
    commitment.trajectory_identity = prospective.trajectory_identity;
    commitment.prediction_identity = prospective.prediction_identity;
    commitment.parent_occurrence_identity = prospective.parent_occurrence_identity;
    commitment.parent_revision_identity = prospective.parent_revision_identity;
    commitment.participation_identity = prospective.participation_identity;
    commitment.generation = persistent->generation + 1u;
    commitment.context_signature = prospective.context_signature;
    commitment.commit_tick = decision_tick;
    *persistent = commitment;
    candidate.decision = ResidentVolitionalDecision::commit;
  }
  candidate.persistent_generation_after = persistent->generation;
  *receipt = candidate;
  return true;
}
}  // namespace substrate::direct_network

#undef DIRECT_VOLITION_HD

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_VOLITIONAL_VETO_CUH
