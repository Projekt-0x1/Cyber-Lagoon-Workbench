#pragma once

#include <cstdint>

#include "causal_rewrite_universe.cuh"

#if defined(__CUDACC__)
#define BCC32_CAUSAL_SOURCE_WITNESS_HD __host__ __device__
#else
#define BCC32_CAUSAL_SOURCE_WITNESS_HD
#endif

// A source witness is ordinary resident provenance matter.  It is made only
// while a complete external trajectory is still present and stores no raw
// words, answer, relation, or executable route.  One record is enough: the
// 64-bit digest is split across the canonical lane/reserved fields, avoiding a
// two-record stack/capacity tax at the physical END seam.
// This header is included after the canonical rewrite namespace has been
// opened. Keeping the witness namespace relative makes it available to every
// translation unit that includes `causal_rewrite_universe.cuh` without trying
// to reopen `substrate::bcc32` from inside that namespace.
namespace resident_causal_relation_source_witness {

namespace rewrite = substrate::bcc32::causal_rewrite;

inline constexpr std::uint32_t kWitnessMarker = 0xe71a4c01u;
inline constexpr std::uint32_t kWitnessReservedMarker = 0xe71a4c02u;
inline constexpr std::uint32_t kReafferentWitnessMarker = 0xe71a4c03u;

BCC32_CAUSAL_SOURCE_WITNESS_HD inline std::uint64_t mix(
    std::uint64_t value, std::uint64_t input) {
  value ^= input + 0x9e3779b97f4a7c15ull + (value << 6u) + (value >> 2u);
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  return value;
}

BCC32_CAUSAL_SOURCE_WITNESS_HD inline bool is_witness(
    const rewrite::Record& record, std::uint32_t owner) {
  return record.matter_q8 != 0u &&
         record.lane[0] == rewrite::kFormTransformationWitness &&
         record.lane[1] == owner && record.lane[4] == owner &&
         record.lane[2] >= 3u && record.lane[3] != 0u &&
         record.lane[6] != 0u && record.lane[6] != rewrite::kInvalid &&
         (record.lane[7] == kWitnessMarker ||
          record.lane[7] == kReafferentWitnessMarker) &&
         record.reserved[0] != 0u && record.reserved[1] == kWitnessReservedMarker;
}

BCC32_CAUSAL_SOURCE_WITNESS_HD inline bool is_reafferent_witness(
    const rewrite::ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == rewrite::kInvalid)
    return false;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (is_witness(record, owner) && record.lane[7] == kReafferentWitnessMarker)
      return true;
  }
  return false;
}

BCC32_CAUSAL_SOURCE_WITNESS_HD inline bool witness_valid(
    const rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint64_t* digest, std::uint32_t* external_leaves) {
  if (state == nullptr || digest == nullptr || external_leaves == nullptr ||
      owner == 0u || owner == rewrite::kInvalid)
    return false;
  const rewrite::Record* witness = nullptr;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(state); ++slot) {
    const rewrite::Record& candidate = state->records[slot];
    if (!is_witness(candidate, owner)) continue;
    if (witness != nullptr) return false;
    witness = &candidate;
  }
  if (witness == nullptr) return false;
  *digest = (static_cast<std::uint64_t>(witness->reserved[0]) << 32u) |
            witness->lane[3];
  *external_leaves = witness->lane[2];
  return true;
}

// Commit a compact witness after the source has been consumed by the
// resumable physical-end stage. The witness carries only exact source
// provenance; it never stores a raw word, answer, relation, or output route.
BCC32_CAUSAL_SOURCE_WITNESS_HD inline bool retain_external_source_digest(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t source_revision, std::uint64_t digest,
    std::uint32_t external_leaves,
    std::uint32_t marker = kWitnessMarker) {
  if (state == nullptr || owner == 0u || owner == rewrite::kInvalid ||
      source_revision == 0u || digest == 0u || external_leaves < 3u ||
      (marker != kWitnessMarker && marker != kReafferentWitnessMarker))
    return false;
  std::uint64_t existing_digest = 0u;
  std::uint32_t existing_leaves = 0u;
  if (witness_valid(state, owner, &existing_digest, &existing_leaves))
    return existing_leaves == external_leaves && existing_digest == digest;
  if (rewrite::free_record_count(state) == 0u) return false;
  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;
  rewrite::Record& witness = state->records[slot];
  const std::uint32_t matter = witness.matter_q8;
  const std::uint32_t revision = witness.revision + 1u;
  witness = rewrite::Record{};
  witness.lane[0] = rewrite::kFormTransformationWitness;
  witness.lane[1] = owner;
  witness.lane[2] = external_leaves;
  witness.lane[3] = static_cast<std::uint32_t>(digest);
  witness.lane[4] = owner;
  witness.lane[6] = source_revision;
  witness.lane[7] = marker;
  witness.reserved[0] = static_cast<std::uint32_t>(digest >> 32u);
  witness.reserved[1] = kWitnessReservedMarker;
  witness.revision = revision;
  witness.matter_q8 = matter;
  ++state->revision;
  return true;
}

// Compatibility helper for focused donor contracts. Production RWR0 uses the
// resumable digest path above, so a long source never becomes one synchronous
// END call or a fixed-size local buffer.
BCC32_CAUSAL_SOURCE_WITNESS_HD inline bool retain_external_source(
    rewrite::ResidentRewriteState* state, const rewrite::Record& source) {
  if (state == nullptr || source.matter_q8 == 0u ||
      source.lane[0] != rewrite::kFormTrajectory || source.lane[1] == 0u ||
      source.lane[1] == rewrite::kInvalid || source.lane[2] < 3u ||
      source.lane[3] != 0u || source.lane[4] != 0u ||
      source.lane[5] != rewrite::kInvalid || source.lane[7] != 0u)
    return false;
  const std::uint32_t owner = source.lane[1];
  std::uint64_t digest = mix(0x13198a2e03707344ull, owner);
  digest = mix(digest, source.revision);
  for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
    const std::uint32_t provenance_slot = rewrite::find_owned_block(
        state, rewrite::kFormTrajectoryProvenance, owner, index / 2u);
    if (provenance_slot == rewrite::kInvalid) return false;
    const rewrite::Record& provenance = state->records[provenance_slot];
    const std::uint32_t local = index % 2u;
    const std::uint32_t origin_lane = 3u + local * 2u;
    const std::uint32_t producer_lane = origin_lane + 1u;
    std::uint32_t word = 0u;
    if ((provenance.lane[rewrite::kProvenanceValidityLane] & (1u << local)) == 0u ||
        provenance.lane[origin_lane] != rewrite::kProvenanceExternalOrigin ||
        provenance.lane[producer_lane] != rewrite::kInvalid ||
        !rewrite::trajectory_word_at(state, owner, index, &word))
      return false;
    digest = mix(digest, index);
    digest = mix(digest, word);
    digest = mix(digest, provenance.revision);
    digest = mix(digest, provenance.matter_q8);
  }
  return retain_external_source_digest(
      state, owner, source.revision, digest, source.lane[2]);
}

}  // namespace resident_causal_relation_source_witness

#undef BCC32_CAUSAL_SOURCE_WITNESS_HD
