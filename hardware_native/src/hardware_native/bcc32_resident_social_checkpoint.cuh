#pragma once

#include "bcc32_resident_relation_boundary.cuh"
#include "bcc32_resident_response_inhibition.cuh"
#include "bcc32_resident_source_calibration.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32::social_checkpoint {

inline constexpr std::uint64_t kSchema = 0x534f4349414c3031ull;

struct Identity {
  std::uint64_t lineage_hash = 0u;
  std::uint64_t checkpoint_hash = 0u;
  std::uint64_t instance_id = 0u;
  std::uint64_t boot_epoch = 0u;
  std::uint64_t online_learning_epoch = 0u;
};

struct AdultState {
  Identity identity{};
  response_inhibition::State inhibition{};
  source_calibration::State calibration{};
  relation_boundary::State boundary{};
};

struct Snapshot {
  std::uint64_t schema = kSchema;
  std::uint64_t lineage_hash = 0u;
  std::uint64_t source_instance_id = 0u;
  std::uint64_t source_boot_epoch = 0u;
  std::uint64_t online_learning_epoch = 0u;
  std::uint64_t organization_digest = 0u;
  response_inhibition::State inhibition{};
  source_calibration::State calibration{};
  relation_boundary::State boundary{};
  std::uint64_t integrity_digest = 0u;
};

__host__ __device__ inline void mix(std::uint64_t* digest,
                                    std::uint64_t value) {
  *digest ^= value + 0x9e3779b97f4a7c15ull + (*digest << 6u) +
             (*digest >> 2u);
  *digest *= 0x100000001b3ull;
}

__host__ __device__ inline void hash_inhibition(
    std::uint64_t* digest, const response_inhibition::State& state) {
  mix(digest, state.next_decision_sequence);
  for (const response_inhibition::SpeakerState& speaker : state.speaker) {
    mix(digest, speaker.identity);
    mix(digest, speaker.last_payload_digest);
    mix(digest, speaker.last_ingress_sequence);
    mix(digest, speaker.pending_decision_sequence);
    mix(digest, speaker.settled_decision_sequence);
    mix(digest, speaker.repetition_run);
    mix(digest, speaker.learned_inhibition_q8);
    mix(digest, speaker.missed_consequence_q8);
    mix(digest, speaker.conserved_resource_q8);
    mix(digest, speaker.revision);
    mix(digest, speaker.pending_payload_digest);
    mix(digest, static_cast<std::uint32_t>(speaker.pending_disposition));
  }
  const response_inhibition::Receipt& receipt = state.receipt;
  mix(digest, receipt.decisions);
  mix(digest, receipt.attended);
  mix(digest, receipt.clarifications);
  mix(digest, receipt.deferred);
  mix(digest, receipt.ignored);
  mix(digest, receipt.settled);
  mix(digest, receipt.rejected_contacts);
  mix(digest, receipt.rejected_consequences);
  mix(digest, receipt.last_speaker_identity);
  mix(digest, receipt.last_payload_digest);
  mix(digest, receipt.last_decision_sequence);
  mix(digest, static_cast<std::uint32_t>(receipt.last_disposition));
}

__host__ __device__ inline void hash_calibration(
    std::uint64_t* digest, const source_calibration::State& state) {
  mix(digest, state.next_decision_sequence);
  for (const source_calibration::SourceState& source : state.source) {
    mix(digest, source.identity);
    mix(digest, source.last_episode_sequence);
    mix(digest, source.pending_decision_sequence);
    mix(digest, source.settled_decision_sequence);
    mix(digest, source.pending_claim_digest);
    mix(digest, source.pending_observation_route_identity);
    mix(digest, source.support_q8);
    mix(digest, source.contradiction_q8);
    mix(digest, source.settled_observations);
    mix(digest, source.revision);
  }
  const source_calibration::Receipt& receipt = state.receipt;
  mix(digest, receipt.testimony_contacts);
  mix(digest, receipt.settled_observations);
  mix(digest, receipt.matched_observations);
  mix(digest, receipt.contradicted_observations);
  mix(digest, receipt.rejected_testimony);
  mix(digest, receipt.rejected_observations);
  mix(digest, receipt.conflict_queries);
  mix(digest, receipt.unresolved_conflicts);
}

__host__ __device__ inline void hash_boundary(
    std::uint64_t* digest, const relation_boundary::State& state) {
  mix(digest, state.next_decision_sequence);
  for (const relation_boundary::RelationState& relation : state.relation) {
    mix(digest, relation.speaker_identity);
    mix(digest, relation.last_interaction_sequence);
    mix(digest, relation.pending_decision_sequence);
    mix(digest, relation.settled_decision_sequence);
    mix(digest, relation.boundary_pressure_q8);
    mix(digest, relation.accumulated_damage_q8);
    mix(digest, relation.accumulated_repair_q8);
    mix(digest, relation.settled_interactions);
    mix(digest, relation.revision);
  }
  const relation_boundary::Receipt& receipt = state.receipt;
  mix(digest, receipt.issued);
  mix(digest, receipt.settled);
  mix(digest, receipt.rejected);
  mix(digest, receipt.damaging);
  mix(digest, receipt.repairing);
  mix(digest, receipt.neutral);
  mix(digest, receipt.engage_decisions);
  mix(digest, receipt.defer_decisions);
  mix(digest, receipt.refuse_decisions);
}

__host__ __device__ inline std::uint64_t organization_digest(
    const response_inhibition::State& inhibition,
    const source_calibration::State& calibration,
    const relation_boundary::State& boundary) {
  std::uint64_t digest = 0xcbf29ce484222325ull;
  hash_inhibition(&digest, inhibition);
  hash_calibration(&digest, calibration);
  hash_boundary(&digest, boundary);
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint64_t snapshot_integrity(
    const Snapshot& snapshot) {
  std::uint64_t digest = 0xcbf29ce484222325ull;
  mix(&digest, snapshot.schema);
  mix(&digest, snapshot.lineage_hash);
  mix(&digest, snapshot.source_instance_id);
  mix(&digest, snapshot.source_boot_epoch);
  mix(&digest, snapshot.online_learning_epoch);
  mix(&digest, snapshot.organization_digest);
  hash_inhibition(&digest, snapshot.inhibition);
  hash_calibration(&digest, snapshot.calibration);
  hash_boundary(&digest, snapshot.boundary);
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline bool make_snapshot(const AdultState& adult,
                                              Snapshot* snapshot) {
  if (snapshot == nullptr || adult.identity.lineage_hash == 0u ||
      adult.identity.instance_id == 0u) return false;
  Snapshot result{};
  result.lineage_hash = adult.identity.lineage_hash;
  result.source_instance_id = adult.identity.instance_id;
  result.source_boot_epoch = adult.identity.boot_epoch;
  result.online_learning_epoch = adult.identity.online_learning_epoch;
  result.inhibition = adult.inhibition;
  result.calibration = adult.calibration;
  result.boundary = adult.boundary;
  result.organization_digest = organization_digest(
      result.inhibition, result.calibration, result.boundary);
  result.integrity_digest = snapshot_integrity(result);
  *snapshot = result;
  return true;
}

__host__ __device__ inline bool restore_snapshot(const Snapshot& snapshot,
                                                 std::uint64_t new_instance_id,
                                                 AdultState* adult) {
  if (adult == nullptr || snapshot.schema != kSchema ||
      snapshot.lineage_hash == 0u || snapshot.source_instance_id == 0u ||
      new_instance_id == 0u || new_instance_id == snapshot.source_instance_id ||
      snapshot.organization_digest != organization_digest(
          snapshot.inhibition, snapshot.calibration, snapshot.boundary) ||
      snapshot.integrity_digest != snapshot_integrity(snapshot))
    return false;
  AdultState restored{};
  restored.identity.lineage_hash = snapshot.lineage_hash;
  restored.identity.checkpoint_hash = snapshot.organization_digest;
  restored.identity.instance_id = new_instance_id;
  restored.identity.boot_epoch = snapshot.source_boot_epoch + 1u;
  restored.identity.online_learning_epoch = snapshot.online_learning_epoch;
  restored.inhibition = snapshot.inhibition;
  restored.calibration = snapshot.calibration;
  restored.boundary = snapshot.boundary;
  *adult = restored;
  return true;
}

}  // namespace substrate::bcc32::social_checkpoint
