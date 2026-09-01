#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_MIXED_PROVENANCE_HD __host__ __device__
#else
#define BCC32_MIXED_PROVENANCE_HD
#endif

namespace substrate::bcc32::causal_rewrite::open_inquiry {
BCC32_MIXED_PROVENANCE_HD bool inquiry_constructor_authoritative(
    const ResidentRewriteState* state, std::uint32_t constructor_slot);
}

namespace substrate::bcc32::causal_rewrite::mixed_provenance {

// Preserve the established namespace spelling while the physical form owner
// lives in the canonical rewrite namespace shared with the one-shot converter.
using ::substrate::bcc32::causal_rewrite::kFormTrajectoryProvenance;

// Provenance is ordinary Record matter, not spare bits hidden in trajectory
// terms. Each owner-bound block describes two adjacent events. Lanes 3/4 and
// 5/6 hold origin/producer pairs; lane 7 says which pairs are physically
// present. The distinction is generic: it records only external contact versus
// a producing resident Program locus, never a semantic role or expected value.
inline constexpr std::uint32_t kProvenanceLayoutFault = 0x4d505633u;
inline constexpr std::uint32_t kMalformedIngressFault = 0x4d505634u;
inline constexpr std::uint32_t kExternalStampFault = 0x4d505635u;
inline constexpr std::uint32_t kUntaggedAdvanceFault = 0x4d505636u;
inline constexpr std::uint32_t kGeneratedStampFault = 0x4d505637u;
inline constexpr std::uint32_t kTrajectoryQualifiedMixed = 4u;

static_assert(kLaneCount == 8u,
              "mixed provenance requires the current eight-lane Record");

enum class Origin : std::uint32_t {
  external = 0u,
  generated = 1u,
  // A generated event whose public closure was rederived from distributed
  // participation.  It has no producer locus: the closure receipt below is
  // provenance only and cannot become a singleton semantic authority.
  distributed = 2u,
};

inline constexpr std::uint32_t kProvenanceDistributedOrigin =
    static_cast<std::uint32_t>(Origin::distributed);

BCC32_MIXED_PROVENANCE_HD inline std::uint32_t origin_lane(
    std::uint32_t local) {
  return 3u + local * 2u;
}

BCC32_MIXED_PROVENANCE_HD inline std::uint32_t producer_lane(
    std::uint32_t local) {
  return 4u + local * 2u;
}

BCC32_MIXED_PROVENANCE_HD inline std::uint32_t valid_bit(
    std::uint32_t local) {
  return 1u << local;
}

BCC32_MIXED_PROVENANCE_HD inline bool live_episodic_diagnostic_locus(
    const ResidentRewriteState* state, std::uint32_t locus);

BCC32_MIXED_PROVENANCE_HD inline bool grounded_producer(
    const ResidentRewriteState* state, std::uint32_t locus) {
  return resident_program_authoritative(state, locus) ||
         open_inquiry::inquiry_constructor_authoritative(state, locus) ||
         live_episodic_diagnostic_locus(state, locus) ||
         resident_revision_participation_reader_authoritative(state, locus);
}

BCC32_MIXED_PROVENANCE_HD inline void clear_provenance(
    ResidentRewriteState* state, std::uint32_t owner) {
  clear_owned_records(state, kFormTrajectoryProvenance, owner);
  clear_owned_records(state, kFormTrajectoryParentRoute, owner);
}

BCC32_MIXED_PROVENANCE_HD inline void clear_orphaned_provenance(
    ResidentRewriteState* state) {
  if (state == nullptr) return;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& provenance = state->records[slot];
    if (provenance.matter_q8 == 0u ||
        (provenance.lane[0] != kFormTrajectoryProvenance &&
         provenance.lane[0] != kFormTrajectoryParentRoute))
      continue;
    if (find_header(state, kFormTrajectory, provenance.lane[1]) == kInvalid)
      clear_record(&provenance);
  }
}

BCC32_MIXED_PROVENANCE_HD inline void clear_trajectory_and_provenance(
    ResidentRewriteState* state, std::uint32_t trajectory_slot) {
  if (state == nullptr || trajectory_slot == kInvalid) return;
  const Record& trajectory = state->records[trajectory_slot];
  if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory)
    return;
  const std::uint32_t owner = trajectory.lane[1];
  clear_provenance(state, owner);
  clear_trajectory(state, trajectory_slot);
}

BCC32_MIXED_PROVENANCE_HD inline Record* ensure_provenance_block(
    ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t ordinal) {
  std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryProvenance, owner, ordinal);
  if (slot != kInvalid) return &state->records[slot];
  slot = allocate_record(state);
  if (slot == kInvalid) return nullptr;
  Record& record = state->records[slot];
  record.lane[0] = kFormTrajectoryProvenance;
  record.lane[1] = owner;
  record.lane[2] = ordinal;
  record.lane[3] = 0u;
  record.lane[4] = kInvalid;
  record.lane[5] = 0u;
  record.lane[6] = kInvalid;
  record.lane[7] = 0u;
  record.reserved[0] = kInvalid;
  record.reserved[1] = kInvalid;
  ++record.revision;
  return &record;
}

BCC32_MIXED_PROVENANCE_HD inline Record* ensure_parent_route_block(
    ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t ordinal) {
  std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryParentRoute, owner, ordinal);
  if (slot != kInvalid) return &state->records[slot];
  slot = allocate_record(state);
  if (slot == kInvalid) return nullptr;
  Record& record = state->records[slot];
  record.lane[0] = kFormTrajectoryParentRoute;
  record.lane[1] = owner;
  record.lane[2] = ordinal;
  record.lane[3] = 0u;
  record.lane[4] = 0u;
  record.lane[5] = 0u;
  record.lane[6] = 0u;
  record.lane[7] = 0u;
  record.reserved[0] = kInvalid;
  record.reserved[1] = kInvalid;
  ++record.revision;
  return &record;
}

BCC32_MIXED_PROVENANCE_HD inline bool mark_last(
    ResidentRewriteState* state, Origin origin, std::uint32_t producer,
    std::uint32_t physical_route = kInvalid,
    std::uint32_t parent_route = kInvalid) {
  const std::uint32_t header_slot = find_current_trajectory(state);
  if (header_slot == kInvalid) return false;
  const Record& header = state->records[header_slot];
  if (header.lane[2] == 0u) return false;
  if (origin == Origin::generated) {
    if (!grounded_producer(state, producer)) return false;
  } else if (producer != kInvalid) {
    return false;
  }
  const std::uint32_t index = header.lane[2] - 1u;
  const std::uint32_t local = index % 2u;
  Record* provenance =
      ensure_provenance_block(state, header.lane[1], index / 2u);
  if (provenance == nullptr ||
      (provenance->lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
      (provenance->lane[kProvenanceValidityLane] & valid_bit(local)) != 0u)
    return false;
  Record* parent = nullptr;
  if (origin == Origin::external && parent_route != kInvalid) {
    parent = ensure_parent_route_block(state, header.lane[1], index / 2u);
    if (parent == nullptr ||
        (parent->lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
        (parent->lane[kProvenanceValidityLane] & valid_bit(local)) != 0u)
      return false;
  }
  if (origin == Origin::generated) {
    provenance->lane[origin_lane(local)] =
        static_cast<std::uint32_t>(Origin::generated);
    provenance->lane[producer_lane(local)] = producer;
  } else {
    provenance->lane[origin_lane(local)] =
        static_cast<std::uint32_t>(Origin::external);
    provenance->lane[producer_lane(local)] = kInvalid;
    provenance->reserved[local] = physical_route;
    if (parent != nullptr) {
      parent->reserved[local] = parent_route;
      parent->lane[kProvenanceValidityLane] |= valid_bit(local);
      ++parent->revision;
    }
  }
  provenance->lane[kProvenanceValidityLane] |= valid_bit(local);
  ++provenance->revision;
  return true;
}

BCC32_MIXED_PROVENANCE_HD inline bool mark_last_distributed(
    ResidentRewriteState* state) {
  if (state == nullptr || state->generated_word_valid == 0u ||
      state->generated_locus != kInvalid ||
      state->causal_relation_generated_events == 0u ||
      state->causal_relation_component_digest == 0u ||
      state->causal_relation_component_revision_digest == 0u ||
      state->causal_relation_external_provenance_digest == 0u ||
      state->causal_relation_external_leaves == 0u)
    return false;
  const std::uint32_t header_slot = find_current_trajectory(state);
  if (header_slot == kInvalid) return false;
  const Record& header = state->records[header_slot];
  if (header.lane[2] == 0u) return false;
  const std::uint32_t index = header.lane[2] - 1u;
  const std::uint32_t local = index % 2u;
  Record* provenance =
      ensure_provenance_block(state, header.lane[1], index / 2u);
  if (provenance == nullptr ||
      (provenance->lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
      (provenance->lane[kProvenanceValidityLane] & valid_bit(local)) != 0u)
    return false;
  provenance->lane[origin_lane(local)] = kProvenanceDistributedOrigin;
  provenance->lane[producer_lane(local)] = kInvalid;
  provenance->lane[kProvenanceValidityLane] |= valid_bit(local);
  ++provenance->revision;
  return true;
}

// A distributed closure gets only an opaque public-lineage handle. There is
// deliberately no physical receipt Record: a Record-shaped witness would
// become an accidental singleton authority for action-return qualification.
BCC32_MIXED_PROVENANCE_HD inline bool install_distributed_generation_receipt(
    ResidentRewriteState* state) {
  if (state == nullptr || state->generated_word_valid == 0u ||
      state->generated_locus != kInvalid ||
      state->causal_relation_component_digest == 0u ||
      state->causal_relation_component_revision_digest == 0u ||
      state->causal_relation_external_provenance_digest == 0u)
    return false;
  state->generated_receipt_owner = static_cast<std::uint32_t>(
      state->causal_relation_component_digest);
  if (state->generated_receipt_owner == 0u ||
      state->generated_receipt_owner == kInvalid)
    state->generated_receipt_owner = 3u;
  state->generated_receipt_valid = 1u;
  state->generated_receipt_topology_digest =
      state->causal_relation_component_digest;
  return true;
}

BCC32_MIXED_PROVENANCE_HD inline bool
distributed_generation_receipt_authoritative(
    const ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot == kInvalid || slot == 0u ||
      state->generated_receipt_valid == 0u ||
      state->generated_receipt_owner != slot ||
      state->causal_relation_component_digest == 0u ||
      state->causal_relation_component_revision_digest == 0u ||
      state->causal_relation_external_provenance_digest == 0u)
    return false;
  return state->generated_receipt_topology_digest ==
         state->causal_relation_component_digest;
}

BCC32_MIXED_PROVENANCE_HD inline bool origin_at(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index, Origin* origin, std::uint32_t* producer) {
  if (state == nullptr || origin == nullptr || producer == nullptr ||
      index >= trajectory.lane[2])
    return false;
  const std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryProvenance, trajectory.lane[1], index / 2u);
  if (slot == kInvalid) return false;
  const Record& provenance = state->records[slot];
  const std::uint32_t local = index % 2u;
  if (provenance.matter_q8 == 0u ||
      provenance.lane[0] != kFormTrajectoryProvenance ||
      (provenance.lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
      (provenance.lane[kProvenanceValidityLane] & valid_bit(local)) == 0u)
    return false;
  const std::uint32_t encoded_origin = provenance.lane[origin_lane(local)];
  const std::uint32_t locus = provenance.lane[producer_lane(local)];
  if (encoded_origin == static_cast<std::uint32_t>(Origin::external)) {
    if (locus != kInvalid) return false;
    *origin = Origin::external;
    *producer = kInvalid;
    return true;
  }
  if (encoded_origin != static_cast<std::uint32_t>(Origin::generated))
    if (encoded_origin != kProvenanceDistributedOrigin) return false;
  if (encoded_origin == kProvenanceDistributedOrigin) {
    if (locus != kInvalid) return false;
    *origin = Origin::distributed;
    *producer = kInvalid;
    return true;
  }
  // Authority is checked when the event is created. Afterwards this locus is
  // historical physical provenance: damaging the producing Program must not
  // relabel its already committed output as external evidence or fault the
  // adult. Lesioning the provenance Record itself still removes the witness.
  if (locus == kInvalid || locus >= live_record_capacity(state)) return false;
  *origin = Origin::generated;
  *producer = locus;
  return true;
}

BCC32_MIXED_PROVENANCE_HD inline std::uint32_t external_route_at(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index) {
  Origin origin = Origin::generated;
  std::uint32_t producer = kInvalid;
  if (!origin_at(state, trajectory, index, &origin, &producer) ||
      origin != Origin::external)
    return kInvalid;
  const std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryProvenance, trajectory.lane[1], index / 2u);
  if (slot == kInvalid) return kInvalid;
  return state->records[slot].reserved[index % 2u];
}

BCC32_MIXED_PROVENANCE_HD inline std::uint32_t external_parent_route_at(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index) {
  Origin origin = Origin::generated;
  std::uint32_t producer = kInvalid;
  if (!origin_at(state, trajectory, index, &origin, &producer) ||
      origin != Origin::external)
    return kInvalid;
  const std::uint32_t slot = find_owned_block(
      state, kFormTrajectoryParentRoute, trajectory.lane[1], index / 2u);
  if (slot == kInvalid) return kInvalid;
  const Record& parent = state->records[slot];
  if ((parent.lane[kProvenanceValidityLane] & valid_bit(index % 2u)) == 0u)
    return kInvalid;
  return parent.reserved[index % 2u];
}

BCC32_MIXED_PROVENANCE_HD inline bool external_route_seen_before(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index, std::uint32_t route) {
  if (route == kInvalid) return false;
  for (std::uint32_t prior = 0u; prior < index; ++prior) {
    if (external_route_at(state, trajectory, prior) == route) return true;
  }
  return false;
}

BCC32_MIXED_PROVENANCE_HD inline bool external_parent_route_seen_before(
    const ResidentRewriteState* state, const Record& trajectory,
    std::uint32_t index, std::uint32_t parent_route) {
  if (parent_route == kInvalid) return false;
  for (std::uint32_t prior = 0u; prior < index; ++prior) {
    if (external_parent_route_at(state, trajectory, prior) == parent_route)
      return true;
  }
  return false;
}

// Passive ancestry census for falsifiers and microscope receipts.  This is
// deliberately not an authority score: it only counts the already committed
// origin tags on one resident trajectory.  In particular, generated and
// distributed events can be arbitrarily numerous without increasing the
// external count or changing the external-only digest computed by a caller.
struct OriginCensus {
  std::uint32_t external = 0u;
  std::uint32_t routed_external = 0u;
  // These are route-token multiplicities, not independent-source or trust
  // counts. A repeated token is observable copied/replayed-route evidence; a
  // distinct token is only a necessary physical distinction for later
  // causal-independence assays, never sufficient authority by itself.
  std::uint32_t distinct_external_routes = 0u;
  std::uint32_t repeated_external_routes = 0u;
  std::uint32_t parented_external = 0u;
  std::uint32_t distinct_parent_routes = 0u;
  std::uint32_t repeated_parent_routes = 0u;
  std::uint32_t generated = 0u;
  std::uint32_t distributed = 0u;
  std::uint32_t invalid = 0u;
  std::uint64_t external_route_digest = 0u;
  std::uint64_t external_parent_route_digest = 0u;
};

BCC32_MIXED_PROVENANCE_HD inline bool census_trajectory_origins(
    const ResidentRewriteState* state, const Record& trajectory,
    OriginCensus* census) {
  if (state == nullptr || census == nullptr ||
      trajectory.lane[0] != kFormTrajectory) {
    return false;
  }
  *census = OriginCensus{};
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    Origin origin = Origin::external;
    std::uint32_t producer = kInvalid;
    if (!origin_at(state, trajectory, index, &origin, &producer)) {
      ++census->invalid;
      continue;
    }
    if (origin == Origin::external) {
      ++census->external;
      const std::uint32_t route = external_route_at(state, trajectory, index);
      if (route != kInvalid) {
        ++census->routed_external;
        if (external_route_seen_before(state, trajectory, index, route))
          ++census->repeated_external_routes;
        else
          ++census->distinct_external_routes;
        census->external_route_digest =
            (census->external_route_digest ^
             (static_cast<std::uint64_t>(route) + index + 0x9e3779b9u)) *
            1099511628211ull;
      }
      const std::uint32_t parent_route =
          external_parent_route_at(state, trajectory, index);
      if (parent_route != kInvalid) {
        ++census->parented_external;
        if (external_parent_route_seen_before(
                state, trajectory, index, parent_route))
          ++census->repeated_parent_routes;
        else
          ++census->distinct_parent_routes;
        census->external_parent_route_digest =
            (census->external_parent_route_digest ^
             (static_cast<std::uint64_t>(parent_route) + index +
              0x517cc1b7u)) *
            1099511628211ull;
      }
    } else if (origin == Origin::generated) {
      ++census->generated;
    } else if (origin == Origin::distributed) {
      ++census->distributed;
    } else {
      ++census->invalid;
    }
  }
  return census->invalid == 0u;
}

BCC32_MIXED_PROVENANCE_HD inline bool tagged_history(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (trajectory.lane[2] == 0u) return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    Origin origin = Origin::external;
    std::uint32_t producer = kInvalid;
    if (!origin_at(state, trajectory, index, &origin, &producer)) return false;
  }
  return true;
}

// The migration-only exact-episode fallback identifies its physical donor by
// the retained source TrajectoryTerm that supplied the next byte. It is not an
// authoritative Program and can never pay learning support, but it is enough
// to distinguish an intentionally zero-credit generated history from malformed
// ingress. When generation follows a closed physical cue, the fresh current
// trajectory has no external prefix and its first-generated offset is zero.
BCC32_MIXED_PROVENANCE_HD inline bool live_episodic_diagnostic_locus(
    const ResidentRewriteState* state, std::uint32_t locus) {
  if (state == nullptr || locus == kInvalid || locus >= live_record_capacity(state))
    return false;
  const Record& term = state->records[locus];
  if (term.matter_q8 == 0u || term.lane[0] != kFormTrajectoryTerm)
    return false;
  const std::uint32_t source_slot =
      find_header(state, kFormTrajectory, term.lane[1]);
  if (source_slot == kInvalid) return false;
  const Record& source = state->records[source_slot];
  return source.lane[3] != 0u && source.lane[7] == 0u &&
         term.lane[2] < (source.lane[2] + 1u) / 2u;
}

// Admit only the exact fallback transition shape: an optional fully tagged
// external prefix followed by one or more contiguous untagged generated events
// beginning at the core's first-generated offset. A pure generated history is
// valid only while its live retained episodic donor term is still present.
// Ordinary ingress or END then retires the trajectory through the existing
// zero-credit path.
BCC32_MIXED_PROVENANCE_HD inline bool legacy_generated_suffix(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.lane[2] == 0u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u ||
      trajectory.lane[5] == kInvalid ||
      trajectory.lane[5] >= trajectory.lane[2])
    return false;
  if (trajectory.lane[5] == 0u &&
      !live_episodic_diagnostic_locus(state, state->generated_locus))
    return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[5]; ++index) {
    Origin origin = Origin::external;
    std::uint32_t producer = kInvalid;
    if (!origin_at(state, trajectory, index, &origin, &producer)) return false;
  }
  for (std::uint32_t index = trajectory.lane[5];
       index < trajectory.lane[2]; ++index) {
    const std::uint32_t slot = find_owned_block(
        state, kFormTrajectoryProvenance, trajectory.lane[1], index / 2u);
    if (slot == kInvalid) continue;
    const Record& provenance = state->records[slot];
    const std::uint32_t local = index % 2u;
    if (provenance.matter_q8 == 0u ||
        provenance.lane[0] != kFormTrajectoryProvenance ||
        (provenance.lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
        (provenance.lane[kProvenanceValidityLane] & valid_bit(local)) != 0u)
      return false;
  }
  return true;
}

BCC32_MIXED_PROVENANCE_HD inline bool complete_external_generated_external(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (trajectory.lane[2] < 3u ||
      (trajectory.lane[7] & kTrajectoryHasGenerated) == 0u)
    return false;
  bool saw_generated = false;
  Origin last = Origin::generated;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    Origin origin = Origin::external;
    std::uint32_t producer = kInvalid;
    if (!origin_at(state, trajectory, index, &origin, &producer)) return false;
    if (index == 0u && origin != Origin::external) return false;
    saw_generated |= origin == Origin::generated ||
                     origin == Origin::distributed;
    last = origin;
  }
  return saw_generated && last == Origin::external;
}

BCC32_MIXED_PROVENANCE_HD inline bool fail_closed(
    ResidentRewriteState* state,
    std::uint32_t fault = kProvenanceLayoutFault) {
  if (state == nullptr) return false;
  if (state->fault == 0u) state->fault = fault;
  const std::uint32_t current = find_current_trajectory(state);
  if (current != kInvalid) clear_trajectory_and_provenance(state, current);
  clear_generated_word(state);
  refresh_receipt(state);
  return false;
}

// External ingress remains the current core ingress.  Only a tagged generated
// owner gets its yielded bit lowered before a later external event, preventing
// the core's normal generated-history retirement from destroying the evidence.
BCC32_MIXED_PROVENANCE_HD inline bool consume_external_event(
    ResidentRewriteEngine engine, RawRewriteEvent event,
    bool update_receipt = true) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  if (event.valid != 0u && event.reserved == kEventFrameNone) {
    const std::uint32_t current = find_current_trajectory(state);
    // An ordinary external-prefix Carry (no generated bit) is genuinely
    // zero-authority scratch and skips provenance stamping below. A
    // generated-execution Carry (a distributed/generated bridge retained as
    // context across END) is different: the new external word arriving now
    // is exactly the fresh connective the relation reader's query-provenance
    // digest requires a stamp for, so it must NOT be skipped.
    const bool ordinary_prefix_carry =
        current != kInvalid &&
        (state->records[current].lane[7] & kTrajectoryHasCarry) != 0u &&
        (state->records[current].lane[7] & kTrajectoryHasGenerated) == 0u;
    std::uint32_t retiring_owner = kInvalid;
    if (current != kInvalid &&
        (state->records[current].lane[7] & kTrajectoryWasYielded) != 0u &&
        (state->records[current].lane[7] & kTrajectoryHasGenerated) == 0u) {
      // Core append intentionally clears the yielded source header before
      // allocating the fresh external owner. Capture and retire its
      // provenance first so repeated Pause -> quiet -> raw contact cannot
      // strand source-only blocks under an unreachable owner.
      const std::uint32_t yielded_owner = state->records[current].lane[1];
      clear_provenance(state, yielded_owner);
    }
    if (current != kInvalid &&
        (state->records[current].lane[7] & kTrajectoryHasGenerated) != 0u) {
      if (tagged_history(state, state->records[current])) {
        state->records[current].lane[4] = 0u;
        ++state->records[current].revision;
      } else {
        if (!legacy_generated_suffix(state, state->records[current]))
          return fail_closed(state, kMalformedIngressFault);
        retiring_owner = state->records[current].lane[1];
      }
    }
    consume_rewrite_event(engine, event, false);
    if (retiring_owner != kInvalid) clear_provenance(state, retiring_owner);
    if (ordinary_prefix_carry) {
      if (update_receipt) refresh_receipt(state);
      return state->fault == 0u;
    }
    const std::uint32_t physical_route =
        event.physical_route != kInvalid ? event.physical_route
                                         : event.value >> 24u;
    if (!mark_last(state, Origin::external, kInvalid, physical_route,
                   event.parent_route))
      return fail_closed(state, kExternalStampFault);
    if (update_receipt) refresh_receipt(state);
    return true;
  }
  consume_rewrite_event(engine, event, update_receipt);
  return state->fault == 0u;
}

// Admit one bounded external packet through the provenance-aware path and
// expose the same exact completion boundary as core batch ingress. Counting
// follows the resident's admitted-event clock, so an event that reached core
// ingress before a provenance fault remains visible while trailing events are
// never reported as consumed.
BCC32_MIXED_PROVENANCE_HD inline RewriteBatchReceipt consume_external_events(
    ResidentRewriteEngine engine, const RawRewriteEvent* events,
    std::uint32_t count) {
  RewriteBatchReceipt receipt{};
  receipt.requested = count;
  ResidentRewriteState* state = engine.state;
  if (state == nullptr) return receipt;
  receipt.fault = state->fault;
  if (count == 0u) {
    receipt.completed = state->fault == 0u ? 1u : 0u;
    receipt.observer_settled =
        state->cross_context_factor_pending == 0u ? 1u : 0u;
    return receipt;
  }
  if (events == nullptr || state->fault != 0u) return receipt;

  for (std::uint32_t index = 0u; index < count && state->fault == 0u;
       ++index) {
    const std::uint64_t admitted_before = state->admitted_events;
    const bool accepted = consume_external_event(engine, events[index], false);
    if (state->admitted_events != admitted_before) ++receipt.consumed;
    if (!accepted) break;
  }
  refresh_receipt(state);
  receipt.fault = state->fault;
  receipt.observer_settled =
      state->cross_context_factor_pending == 0u ? 1u : 0u;
  receipt.completed =
      receipt.consumed == count && receipt.fault == 0u ? 1u : 0u;
  return receipt;
}

// Reorganization remains on the same persistent adult clock, but starts only
// after ingress helpers have returned so CUDA never carries both the contact
// admission frame and recursive structural validation frame at once.
BCC32_MIXED_PROVENANCE_HD inline bool settle_cross_context_factor(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u ||
      state->cross_context_factor_pending == 0u)
    return false;
  state->cross_context_factor_pending = 0u;
  const bool changed =
      cross_context::cross_context_factor_all_mature_programs(state);
  refresh_receipt(state);
  return changed;
}

// The executor remains the sole source of generated consequences.  This
// wrapper only stamps the newly appended ordinary trajectory term.
BCC32_MIXED_PROVENANCE_HD inline bool advance_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  (void)settle_cross_context_factor(state);
  const std::uint32_t current = state == nullptr ? kInvalid :
      find_current_trajectory(state);
  // No current history is ordinary quiet time. The exact episodic bridge may
  // also leave one deliberately untagged generated suffix; it has no Program
  // authority and is retired by the next external contact. Any other partial
  // provenance is malformed resident matter and fails closed.
  if (current == kInvalid) return false;
  // See consume_external_event above: an ordinary external-prefix Carry (no
  // generated bit) is zero-authority scratch and skips the tagged-history
  // requirement and later stamping below. A generated-execution Carry (a
  // distributed/generated bridge retained across END) is genuinely tagged
  // resident history and must keep both.
  const bool ordinary_prefix_carry =
      (state->records[current].lane[7] & kTrajectoryHasCarry) != 0u &&
      (state->records[current].lane[7] & kTrajectoryHasGenerated) == 0u;
  if (!ordinary_prefix_carry &&
      !tagged_history(state, state->records[current])) {
    if (legacy_generated_suffix(state, state->records[current])) return false;
    return fail_closed(state, kUntaggedAdvanceFault);
  }
  const std::uint32_t owner = state->records[current].lane[1];
  Record& trajectory = state->records[current];
  if (trajectory.lane[4] == 0u) return false;
  // Participation-backed revised readers own their matching cue even when a
  // close distractor makes their direct executor abstain. Never route around
  // that abstention through the generic Program collector.
  if (revision_participation_reader_engaged(state, trajectory)) return false;

  // The canonical executor performs one complete consensus pass, including
  // fixed, VersionSpace, SpanProgram, and raw-span candidates. It is told to
  // retain a yielded grounded history when no continuation wins so a later
  // physical consequence can still join it. Candidate discovery must not be
  // duplicated here: a partial mirror had already drifted behind the core and
  // could reject a valid raw-span continuation before execution saw it.
  if (!causal_rewrite::advance_resident_program_once(engine, true)) {
    if (find_header(state, kFormTrajectory, owner) == kInvalid)
      clear_provenance(state, owner);
    return false;
  }
  if (!ordinary_prefix_carry) {
    const bool distributed =
        state->generated_locus == kInvalid &&
        state->causal_relation_generated_events != 0u;
    if (distributed ? !mark_last_distributed(state)
                   : !mark_last(state, Origin::generated,
                                state->generated_locus))
      return fail_closed(state, kGeneratedStampFault);
  }
  refresh_receipt(state);
  return true;
}

// Qualified settlement is the only path that exposes a mixed history to the
// existing ordinary fixed-program support/anti-unification primitives. The
// coarse generated header bit is masked only while those existing primitives
// inspect the trajectory. A first independent witness is retained under a
// distinct non-authoritative header bit, so ordinary closure cannot mistake it
// for all-external evidence. No term origin or producer locus changes.
BCC32_MIXED_PROVENANCE_HD inline bool qualify_current_history(
    ResidentRewriteState* state) {
  if (state == nullptr) return false;
  const std::uint32_t current = find_current_trajectory(state);
  if (current == kInvalid)
    return false;
  // A promoted generated-execution Carry (a distributed/generated bridge
  // retained as zero-authority execution context across a physical END) must
  // never be reinterpreted as ordinary external/generated/external teaching
  // matter -- that would launder resident context into source authority.
  if ((state->records[current].lane[7] & kTrajectoryHasCarry) != 0u)
    return false;
  if (!complete_external_generated_external(state, state->records[current]))
    return false;
  Record& trajectory = state->records[current];
  const std::uint32_t current_owner = trajectory.lane[1];
  trajectory.lane[7] = 0u;
  const bool supported = support_existing_programs(state, current, true, true);
  if (find_current_trajectory(state) != current) {
    clear_provenance(state, current_owner);
    refresh_receipt(state);
    return supported;
  }

  std::uint32_t left = kInvalid;
  std::uint32_t identity = kInvalid;
  bool conflict = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormTrajectory ||
        candidate.lane[3] == 0u ||
        candidate.lane[7] != kTrajectoryQualifiedMixed)
      continue;
    const std::uint32_t candidate_provenance = candidate.lane[7];
    candidate.lane[7] = 0u;
    std::uint32_t candidate_identity = kInvalid;
    const bool compatible =
        induce_program(state, slot, current, &candidate_identity, false);
    candidate.lane[7] = candidate_provenance;
    if (!compatible)
      continue;
    if (left == kInvalid) {
      left = slot;
      identity = candidate_identity;
    } else if (identity != candidate_identity) {
      conflict = true;
    }
  }
  if (left != kInvalid && !conflict) {
    Record& witness = state->records[left];
    const std::uint32_t witness_owner = witness.lane[1];
    const std::uint32_t witness_provenance = witness.lane[7];
    witness.lane[7] = 0u;
    if (induce_program(state, left, current)) {
      clear_provenance(state, witness_owner);
      clear_provenance(state, current_owner);
      refresh_receipt(state);
      return true;
    }
    if (left < live_record_capacity(state) && witness.matter_q8 != 0u)
      witness.lane[7] = witness_provenance;
  }
  if (current < live_record_capacity(state) && state->records[current].matter_q8 != 0u &&
      state->records[current].lane[0] == kFormTrajectory) {
    Record& retained = state->records[current];
    retained.lane[3] = 1u;
    retained.lane[4] = 0u;
    retained.lane[7] = kTrajectoryQualifiedMixed;
    ++retained.revision;
  }
  refresh_receipt(state);
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite::mixed_provenance

#undef BCC32_MIXED_PROVENANCE_HD

// Keep the authority extension visible in every translation unit that uses
// generated provenance.  The reciprocal include is safe under #pragma once:
// open inquiry consumes the provenance API, while provenance verifies the
// complete Constructor witness chain rather than trusting a form bit.
#include "bcc32_resident_open_inquiry.cuh"
