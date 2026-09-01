#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PARALLEL_LANGUAGE_STREAMS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PARALLEL_LANGUAGE_STREAMS_CUH

// f.parallel_language_grounding_expression_streams (#1572).  Two parallel,
// physically distinct resident pathways grown from one born adult's exact
// history.  The grounded mapping stream binds coherent multi-channel contact
// into sensory-to-meaning sites only where an actual settled world return
// authorizes the meaning side.  The articulation stream grows
// word-transition edges from motor-sequential execution alone.  Nothing here
// knows what a word means: surface bytes, channel numbers and node indices
// are opaque, and which stream does what follows only from record kind,
// chronology and capacity -- never from a semantic branch.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_multimodal_grounding.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kStreamsGroundedSiteCapacity = 256u;
inline constexpr std::uint32_t kStreamsEdgeCapacity = 256u;
inline constexpr std::uint32_t kStreamsEmitterCapacity = 128u;
inline constexpr std::uint32_t kStreamsTraceCapacity = 12u;
inline constexpr std::uint32_t kStreamsContextChannelCapacity = 8u;
// Successive motor outputs farther apart than this are separate utterances,
// not a sequence; the articulation stream refuses to chain them.
inline constexpr std::uint32_t kStreamsSequenceWindowTicks = 8u;
// A grounded selection needs this many reactivated meaning-bearing sites
// across at least two context channels before it may seed expression,
// keeping the mapping pathway distributed rather than a single-row lookup.
inline constexpr std::uint32_t kStreamsQuorum = 3u;

__host__ __device__ inline bool streams_mass_bound(std::int32_t mass) {
  return mass >= kGroundingBoundThresholdQ16;
}

struct DirectGroundedMeaningSite {
  std::uint32_t surface_channel;
  std::uint32_t surface_value;
  std::uint32_t context_channel;
  std::uint32_t context_value;
  std::uint32_t action_node;
  std::int32_t bind_mass_q16;
  std::uint32_t reactivations;
  std::uint32_t active;
};

struct DirectArticulationEdge {
  std::uint32_t from_word;
  std::uint32_t to_word;
  std::uint32_t growths;
  std::uint32_t traversals;
  std::uint32_t last_tick;
  std::uint32_t active;
};

struct DirectStreamEmitter {
  std::uint32_t word;
  std::uint32_t node;
};

struct DirectExpressionWord {
  std::uint32_t word;
  std::uint32_t edge_index;
};

struct DirectStreamsDivergence {
  std::uint32_t grounded_sites;
  std::uint32_t grounded_meaning_sites;
  std::uint32_t grounded_surface_channels[kStreamsContextChannelCapacity];
  std::uint32_t grounded_context_channels[kStreamsContextChannelCapacity];
  std::uint32_t grounded_channel_kinds;
  std::int32_t grounded_mass_total_q16;
  std::int32_t grounded_mass_max_q16;
  std::uint32_t articulation_edges;
  std::uint32_t articulation_distinct_words;
  std::uint32_t articulation_growth_total;
  std::uint32_t articulation_traversal_total;
  std::uint32_t matter_intersection;
  std::uint32_t divergence_reserved;
  std::uint64_t grounding_topology_hash;
  std::uint64_t articulation_topology_hash;
};

struct DirectGroundedSelectionReceipt {
  std::uint32_t entry_word;
  std::uint32_t action_node;
  std::uint32_t supporting_sites;
  std::uint32_t distinct_context_channels;
  std::uint32_t admitted;
  // Fold of the resident slot indices that reactivated, proving which
  // distributed meaning matter the context actually addressed.
  std::uint32_t reactivated_hash_lo;
  std::uint32_t reactivated_hash_hi;
};

struct DirectParallelLanguageStreamsState {
  DirectGroundedMeaningSite grounded_sites[kStreamsGroundedSiteCapacity];
  std::uint32_t grounded_site_count;
  DirectArticulationEdge edges[kStreamsEdgeCapacity];
  std::uint32_t edge_count;
  DirectStreamEmitter emitters[kStreamsEmitterCapacity];
  std::uint32_t emitter_count;
  DirectExpressionWord trace[kStreamsTraceCapacity];
  std::uint32_t trace_length;
  std::uint32_t trace_novel_composition;
  std::uint32_t trace_history_occurrences;
  std::uint32_t grounded_events;
  std::uint32_t grounded_deposits;
  std::uint32_t grounded_reactivations;
  std::uint32_t grounded_refusals;
  std::uint32_t grounded_capacity_refusals;
  std::uint32_t grounded_alignment_pad;
  std::uint32_t articulation_events;
  std::uint32_t articulation_growths;
  std::uint32_t articulation_traversals;
  std::uint32_t articulation_refusals;
  std::uint32_t attribution_refusals;
  std::uint32_t unattributed_trace_words;
  std::uint32_t emitter_capacity_refusals;
  std::uint32_t emitter_alignment_pad;
  std::uint32_t lesion_grounding_events;
  std::uint32_t lesion_articulation_events;
  std::uint32_t q_contacts;
  std::uint32_t motor_outputs;
  std::uint32_t world_returns;
  std::uint32_t assimilation_rotations;
  // 26 u32 scalars precede the u64 digests so no implicit padding byte
  // enters the state's object representation.
  std::uint64_t revision_identity;
  std::uint64_t raw_contact_hash;
  std::uint64_t control_hash;
  std::uint64_t global_cursor;
};

static_assert(std::is_trivially_copyable_v<DirectParallelLanguageStreamsState>);
static_assert(
    std::has_unique_object_representations_v<DirectParallelLanguageStreamsState>);
static_assert(std::has_unique_object_representations_v<DirectGroundedMeaningSite>);
static_assert(std::has_unique_object_representations_v<DirectArticulationEdge>);

__device__ inline std::uint64_t streams_fold(std::uint64_t h,
                                             std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
  return h;
}

__device__ inline std::int32_t streams_find_site(
    const DirectParallelLanguageStreamsState& state,
    std::uint32_t surface_channel, std::uint32_t surface_value,
    std::uint32_t context_channel, std::uint32_t context_value) {
  for (std::uint32_t i = 0u; i < state.grounded_site_count; ++i) {
    const DirectGroundedMeaningSite& site = state.grounded_sites[i];
    if (site.active != 0u && site.surface_channel == surface_channel &&
        site.surface_value == surface_value &&
        site.context_channel == context_channel &&
        site.context_value == context_value)
      return static_cast<std::int32_t>(i);
  }
  return -1;
}

__device__ inline void streams_deposit_grounded(
    DirectParallelLanguageStreamsState* state,
    const DirectCrossModalContact& surface,
    const DirectCrossModalContact& context) {
  if (surface.channel == context.channel) return;
  const std::int32_t slot = streams_find_site(*state, surface.channel,
                                              surface.value, context.channel,
                                              context.value);
  if (slot >= 0) {
    DirectGroundedMeaningSite& site = state->grounded_sites[slot];
    const bool was_bound = streams_mass_bound(site.bind_mass_q16);
    site.bind_mass_q16 = grounding_rise_q16(site.bind_mass_q16);
    // The mapping stream's work units are binding events: an association is
    // minted when a site is created or crosses the bound threshold.  The
    // sub-threshold mass nudges between those crossings are the slow
    // distributed integration the law describes, not stream events.
    if (!was_bound && streams_mass_bound(site.bind_mass_q16)) {
      ++state->grounded_events;
      ++state->grounded_deposits;
    }
    return;
  }
  // A lesioned association revives only through renewed coherent contact,
  // restarting its integration from nothing; until then its tombstone
  // reserves the matter it once held.
  for (std::uint32_t i = 0u; i < state->grounded_site_count; ++i) {
    DirectGroundedMeaningSite& dormant = state->grounded_sites[i];
    if (dormant.active != 0u || dormant.surface_channel != surface.channel ||
        dormant.surface_value != surface.value ||
        dormant.context_channel != context.channel ||
        dormant.context_value != context.value)
      continue;
    dormant.bind_mass_q16 = grounding_rise_q16(0);
    dormant.reactivations = 0u;
    dormant.action_node = kInvalidIndex;
    dormant.active = 1u;
    ++state->grounded_events;
    ++state->grounded_deposits;
    return;
  }
  if (state->grounded_site_count >= kStreamsGroundedSiteCapacity) {
    ++state->grounded_capacity_refusals;
    return;
  }
  DirectGroundedMeaningSite fresh{};
  fresh.surface_channel = surface.channel;
  fresh.surface_value = surface.value;
  fresh.context_channel = context.channel;
  fresh.context_value = context.value;
  fresh.action_node = kInvalidIndex;
  fresh.bind_mass_q16 = grounding_rise_q16(0);
  fresh.active = 1u;
  state->grounded_sites[state->grounded_site_count++] = fresh;
  ++state->grounded_events;
  ++state->grounded_deposits;
}

__device__ inline bool streams_has_world_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t identity) {
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::world_return &&
        records[i].identity == identity)
      return true;
  return false;
}

__device__ inline void streams_note_emitter(
    DirectParallelLanguageStreamsState* state, std::uint32_t word,
    std::uint32_t node) {
  for (std::uint32_t i = 0u; i < state->emitter_count; ++i) {
    if (state->emitters[i].word == word) {
      state->emitters[i].node = node;
      return;
    }
  }
  if (state->emitter_count >= kStreamsEmitterCapacity) {
    ++state->emitter_capacity_refusals;
    return;
  }
  state->emitters[state->emitter_count++] = DirectStreamEmitter{word, node};
}

__device__ inline std::int32_t streams_find_edge(
    const DirectParallelLanguageStreamsState& state, std::uint32_t from_word,
    std::uint32_t to_word) {
  for (std::uint32_t i = 0u; i < state.edge_count; ++i) {
    const DirectArticulationEdge& edge = state.edges[i];
    if (edge.active != 0u && edge.from_word == from_word &&
        edge.to_word == to_word)
      return static_cast<std::int32_t>(i);
  }
  return -1;
}

__device__ inline void streams_deposit_edge(
    DirectParallelLanguageStreamsState* state, std::uint32_t from_word,
    std::uint32_t to_word, std::uint32_t tick) {
  if (from_word == to_word) return;
  const std::int32_t slot = streams_find_edge(*state, from_word, to_word);
  if (slot >= 0) {
    state->edges[slot].growths += 1u;
    state->edges[slot].last_tick = tick;
    ++state->articulation_events;
    ++state->articulation_growths;
    return;
  }
  if (state->edge_count >= kStreamsEdgeCapacity) {
    ++state->articulation_refusals;
    return;
  }
  DirectArticulationEdge fresh{};
  fresh.from_word = from_word;
  fresh.to_word = to_word;
  fresh.growths = 1u;
  fresh.traversals = 0u;
  fresh.last_tick = tick;
  fresh.active = 1u;
  state->edges[state->edge_count++] = fresh;
  ++state->articulation_events;
  ++state->articulation_growths;
}

// One chronological pass over the new exact-history suffix.  Coherent
// distinct-channel contact inside the grounding window deposits grounded
// mapping mass; successive motor outputs inside the sequence window grow
// articulation transition edges regardless of any grounded state.  Meaning
// attachment runs afterwards: a strong site earns its acting-node anchor
// only from an emission whose own identity earned an actual settled world
// return, and only where the site's own legs participated inside the
// consequence window before that emission.
__device__ inline void streams_attach_meaning(
    DirectParallelLanguageStreamsState* state,
    const DirectExactHistoryRecord* records, std::uint64_t archived,
    std::uint64_t begin_global, std::uint64_t end_global) {
  for (std::uint64_t r = begin_global; r < end_global; ++r) {
    const DirectExactHistoryRecord& record = records[r - archived];
    if (record.kind != DirectExactHistoryKind::world_return) continue;
    std::uint32_t emission_tick = 0u;
    std::uint32_t emission_node = kInvalidIndex;
    bool found_emission = false;
    for (std::uint64_t j = archived; j < end_global && !found_emission; ++j) {
      if (records[j - archived].kind ==
              DirectExactHistoryKind::motor_output &&
          records[j - archived].identity == record.identity) {
        emission_tick = records[j - archived].resident_tick;
        emission_node = records[j - archived].source;
        found_emission = true;
      }
    }
    if (!found_emission || emission_node == kInvalidIndex ||
        emission_tick < kGroundingConsequenceWindowTicks)
      continue;
    for (std::uint64_t j = begin_global; j < end_global; ++j) {
      const DirectExactHistoryRecord& contact = records[j - archived];
      if (contact.kind != DirectExactHistoryKind::sensory_contact ||
          contact.resident_tick > emission_tick ||
          emission_tick - contact.resident_tick >
              kGroundingConsequenceWindowTicks)
        continue;
      for (std::uint32_t s = 0u; s < state->grounded_site_count; ++s) {
        DirectGroundedMeaningSite& site = state->grounded_sites[s];
        if (site.active == 0u || site.action_node != kInvalidIndex ||
            !streams_mass_bound(site.bind_mass_q16))
          continue;
        const bool surface_leg = contact.subject == site.surface_channel &&
                                 contact.value == site.surface_value;
        const bool context_leg = contact.subject == site.context_channel &&
                                 contact.value == site.context_value;
        if (surface_leg || context_leg) site.action_node = emission_node;
      }
    }
  }
}

__device__ inline void streams_assimilate(
    DirectParallelLanguageStreamsState* state,
    const DirectExactHistoryRecord* records, std::uint64_t archived,
    std::uint32_t committed) {
  const std::uint64_t end_global =
      static_cast<std::uint64_t>(archived) + committed;
  if (state == nullptr || records == nullptr ||
      end_global <= state->global_cursor)
    return;
  std::uint64_t begin_global = state->global_cursor;
  state->global_cursor = end_global;
  if (begin_global < archived) {
    // The hot page rotated under us: the superseded span is sealed into the
    // archive chain, so assimilation resumes at the resident page base.
    ++state->assimilation_rotations;
    begin_global = archived;
  }

  DirectCrossModalContact recent[kGroundingRecentCapacity];
  std::uint32_t recent_count = 0u;
  std::uint32_t pending_motor_word = 0u;
  std::uint32_t pending_motor_tick = 0u;
  std::uint64_t pending_motor_identity = 0u;
  for (std::uint64_t i = begin_global; i < end_global; ++i) {
    const DirectExactHistoryRecord& record = records[i - archived];
    if (record.kind == DirectExactHistoryKind::sensory_contact) {
      ++state->q_contacts;
      state->raw_contact_hash =
          streams_fold(streams_fold(state->raw_contact_hash, record.subject),
                       record.value);
      state->revision_identity =
          streams_fold(state->revision_identity, record.identity);
      const DirectCrossModalContact contact{record.subject, record.value,
                                            record.resident_tick};
      for (std::uint32_t j = 0u; j < recent_count; ++j) {
        const std::uint32_t age =
            contact.resident_tick >= recent[j].resident_tick
                ? contact.resident_tick - recent[j].resident_tick
                : 0u;
        if (age > kGroundingCoherenceWindowTicks) continue;
        streams_deposit_grounded(state, recent[j], contact);
        streams_deposit_grounded(state, contact, recent[j]);
      }
      std::uint32_t kept = 0u;
      for (std::uint32_t j = 0u; j < recent_count; ++j) {
        const std::uint32_t age =
            contact.resident_tick >= recent[j].resident_tick
                ? contact.resident_tick - recent[j].resident_tick
                : 0u;
        if (age <= kGroundingCoherenceWindowTicks &&
            kept < kGroundingRecentCapacity)
          recent[kept++] = recent[j];
      }
      if (kept < kGroundingRecentCapacity) recent[kept++] = contact;
      recent_count = kept;
    } else if (record.kind == DirectExactHistoryKind::motor_output) {
      ++state->motor_outputs;
      state->revision_identity =
          streams_fold(state->revision_identity, record.identity);
      streams_note_emitter(state, record.value, record.source);
      if (pending_motor_identity != 0u &&
          record.resident_tick >= pending_motor_tick &&
          record.resident_tick - pending_motor_tick <=
              kStreamsSequenceWindowTicks)
        streams_deposit_edge(state, pending_motor_word, record.value,
                             record.resident_tick);
      pending_motor_word = record.value;
      pending_motor_tick = record.resident_tick;
      pending_motor_identity = record.identity;
    } else if (record.kind == DirectExactHistoryKind::world_return) {
      ++state->world_returns;
      state->revision_identity =
          streams_fold(state->revision_identity, record.identity);
    }
  }
  streams_attach_meaning(state, records, archived, begin_global, end_global);
}

__device__ inline std::int32_t streams_emitter_node(
    const DirectParallelLanguageStreamsState& state, std::uint32_t word) {
  for (std::uint32_t i = 0u; i < state.emitter_count; ++i)
    if (state.emitters[i].word == word)
      return static_cast<std::int32_t>(state.emitters[i].node);
  return -1;
}

// Among the words a meaning anchor actually emitted, the one with the most
// historically grown outgoing transitions (smallest word breaks ties) is
// where articulation begins.  The mapping pathway chooses WHAT speaks; the
// transition graph decides WHERE the utterance goes from there.
__device__ inline std::uint32_t streams_best_connected_word(
    const DirectParallelLanguageStreamsState& state, std::uint32_t node) {
  std::uint32_t best_word = 0u;
  std::uint32_t best_growth = 0u;
  bool any_emitted = false;
  std::uint32_t first_emitted = 0u;
  for (std::uint32_t e = 0u; e < state.emitter_count; ++e) {
    if (state.emitters[e].node != node) continue;
    const std::uint32_t word = state.emitters[e].word;
    if (!any_emitted || word < first_emitted) first_emitted = word;
    any_emitted = true;
    std::uint32_t growth = 0u;
    for (std::uint32_t i = 0u; i < state.edge_count; ++i) {
      const DirectArticulationEdge& edge = state.edges[i];
      if (edge.active != 0u && edge.from_word == word) growth += edge.growths;
    }
    if (growth > best_growth ||
        (growth == best_growth && growth != 0u && word < best_word)) {
      best_word = word;
      best_growth = growth;
    }
  }
  return best_growth != 0u ? best_word : (any_emitted ? first_emitted : 0u);
}

// Grounded selection: recent surface AND context contact reactivates the
// strong meaning-bearing sites carrying those exact legs; each reactivated
// site votes for the latest word its own authorized anchor actually emitted.
// Without a quorum nothing is admitted -- the mapping pathway gates
// articulation content yet owns none of its matter.
__device__ inline DirectGroundedSelectionReceipt streams_grounded_select(
    DirectParallelLanguageStreamsState* state,
    const DirectExactHistoryRecord* records, std::uint64_t archived,
    std::uint64_t begin_global, std::uint32_t committed) {
  DirectGroundedSelectionReceipt receipt{};
  const std::uint64_t end_global =
      static_cast<std::uint64_t>(archived) + committed;
  const std::uint64_t begin =
      begin_global < archived ? archived : begin_global;
  if (records == nullptr || begin >= end_global) {
    ++state->grounded_refusals;
    return receipt;
  }
  std::uint32_t newest_tick = 0u;
  for (std::uint64_t i = begin; i < end_global; ++i)
    if (records[i - archived].kind ==
            DirectExactHistoryKind::sensory_contact &&
        records[i - archived].resident_tick > newest_tick)
      newest_tick = records[i - archived].resident_tick;

  std::uint32_t votes[kStreamsEmitterCapacity]{};
  std::uint32_t nodes[kStreamsEmitterCapacity]{};
  std::uint32_t node_count = 0u;
  std::uint32_t context_channels[kStreamsContextChannelCapacity]{};
  std::uint32_t channel_count = 0u;
  for (std::uint32_t i = 0u; i < state->grounded_site_count; ++i) {
    DirectGroundedMeaningSite& site = state->grounded_sites[i];
    if (site.active == 0u || site.action_node == kInvalidIndex ||
        !streams_mass_bound(site.bind_mass_q16))
      continue;
    bool surface_recent = false;
    bool context_recent = false;
    for (std::uint64_t j = begin;
         j < end_global && (!surface_recent || !context_recent); ++j) {
      const DirectExactHistoryRecord& record = records[j - archived];
      if (record.kind != DirectExactHistoryKind::sensory_contact ||
          record.resident_tick > newest_tick ||
          newest_tick - record.resident_tick > kGroundingCoherenceWindowTicks)
        continue;
      surface_recent = surface_recent ||
                       (record.subject == site.surface_channel &&
                        record.value == site.surface_value);
      context_recent = context_recent ||
                       (record.subject == site.context_channel &&
                        record.value == site.context_value);
    }
    if (!surface_recent || !context_recent) continue;
    {
      const std::uint64_t slot_fold =
          streams_fold(streams_fold(0x15721572ULL, i), site.action_node);
      receipt.reactivated_hash_lo ^= static_cast<std::uint32_t>(slot_fold);
      receipt.reactivated_hash_hi ^=
          static_cast<std::uint32_t>(slot_fold >> 32u);
    }
    ++site.reactivations;
    ++state->grounded_events;
    ++state->grounded_reactivations;
    ++receipt.supporting_sites;
    bool seen_channel = false;
    for (std::uint32_t c = 0u; c < channel_count; ++c)
      seen_channel = seen_channel || context_channels[c] == site.context_channel;
    if (!seen_channel && channel_count < kStreamsContextChannelCapacity)
      context_channels[channel_count++] = site.context_channel;

    std::uint32_t slot = node_count;
    for (std::uint32_t n = 0u; n < node_count; ++n)
      if (nodes[n] == site.action_node) slot = n;
    if (slot == node_count && node_count < kStreamsEmitterCapacity)
      nodes[node_count++] = site.action_node;
    if (slot < kStreamsEmitterCapacity) ++votes[slot];
  }
  receipt.distinct_context_channels = channel_count;
  std::uint32_t winning_votes = 0u;
  for (std::uint32_t n = 0u; n < node_count; ++n)
    if (votes[n] > winning_votes ||
        (votes[n] == winning_votes &&
         (winning_votes == 0u || nodes[n] < receipt.action_node))) {
      receipt.action_node = nodes[n];
      winning_votes = votes[n];
    }
  receipt.admitted = receipt.supporting_sites >= kStreamsQuorum &&
                     receipt.distinct_context_channels >= 2u &&
                     winning_votes != 0u;
  if (receipt.admitted == 0u) {
    ++state->grounded_refusals;
    receipt.action_node = kInvalidIndex;
  } else {
    receipt.entry_word =
        streams_best_connected_word(*state, receipt.action_node);
    if (receipt.entry_word == 0u) {
      receipt.admitted = 0u;
      ++state->grounded_refusals;
    }
  }
  return receipt;
}


__device__ inline void streams_trace_seed(
    DirectParallelLanguageStreamsState* state,
    const DirectGroundedSelectionReceipt& selection) {
  if (selection.admitted == 0u) return;
  state->trace_length = 0u;
  state->trace_novel_composition = 0u;
  state->trace_history_occurrences = 0u;
  state->trace[0] = DirectExpressionWord{selection.entry_word, kInvalidIndex};
  state->trace_length = 1u;
}

// One articulation step: follow the most historically grown outgoing edge
// (smallest successor breaks ties), attribute the successor word to the node
// that actually emitted it, and bias that node through the ordinary brain
// state so the adult may express the step publicly.  No learned continuation
// ends the trajectory; an unattributable word mints nothing.
__device__ inline std::uint32_t streams_articulate_step(
    DirectParallelLanguageStreamsState* state, DirectBrain brain) {
  if (state->trace_length == 0u || state->trace_length >= kStreamsTraceCapacity) {
    ++state->articulation_refusals;
    return 0u;
  }
  const std::uint32_t current_word = state->trace[state->trace_length - 1u].word;
  // Expression first walks transitions it has never executed, so learned
  // material recomposes into trajectories rather than replaying one fluent
  // basin; only exhausted material falls back to its most-grown ordering.
  std::int32_t best = -1;
  std::int32_t fallback = -1;
  for (std::uint32_t i = 0u; i < state->edge_count; ++i) {
    const DirectArticulationEdge& edge = state->edges[i];
    if (edge.active == 0u || edge.from_word != current_word) continue;
    auto better = [](const DirectArticulationEdge& candidate,
                     const DirectArticulationEdge& champion) {
      return candidate.growths > champion.growths ||
             (candidate.growths == champion.growths &&
              candidate.to_word < champion.to_word);
    };
    if (edge.traversals == 0u) {
      if (best < 0 || better(edge, state->edges[best]))
        best = static_cast<std::int32_t>(i);
    } else if (fallback < 0 || better(edge, state->edges[fallback])) {
      fallback = static_cast<std::int32_t>(i);
    }
  }
  if (best < 0) best = fallback;
  if (best < 0) return 0u;
  const std::uint32_t next_word = state->edges[best].to_word;
  const std::int32_t node = streams_emitter_node(*state, next_word);
  if (node < 0 || static_cast<std::uint32_t>(node) >= brain.node_count) {
    ++state->attribution_refusals;
    ++state->unattributed_trace_words;
    return 0u;
  }
  state->edges[best].traversals += 1u;
  ++state->articulation_events;
  ++state->articulation_traversals;
  state->trace[state->trace_length++] =
      DirectExpressionWord{next_word, static_cast<std::uint32_t>(best)};
  atomicAdd(&brain.nodes[node].activation_q16, kQ16One);
  atomicAdd(&brain.nodes[node].credit_ema_q16, kQ16One / 8);
  return next_word;
}

// Composition verdict over the pre-expression history prefix [0, horizon):
// every adjacent pair must be a previously grown edge, and the full chain
// must never have occurred verbatim as consecutive motor output.  A chain
// that fails either test stays unproven rather than being graded loosely.
__device__ inline void streams_verify_composition(
    DirectParallelLanguageStreamsState* state,
    const DirectExactHistoryRecord* records, std::uint64_t archived,
    std::uint64_t horizon_global, std::uint32_t committed) {
  state->trace_novel_composition = 0u;
  state->trace_history_occurrences = 0u;
  if (state->trace_length < 3u) return;
  for (std::uint32_t i = 1u; i < state->trace_length; ++i) {
    const std::int32_t edge = streams_find_edge(*state, state->trace[i - 1u].word,
                                                state->trace[i].word);
    if (edge < 0 || state->edges[edge].growths == 0u) return;
  }
  const std::uint64_t page_end =
      static_cast<std::uint64_t>(archived) + committed;
  const std::uint64_t horizon =
      horizon_global < page_end ? horizon_global : page_end;
  if (horizon < static_cast<std::uint64_t>(archived) + state->trace_length)
    return;
  std::uint32_t occurrences = 0u;
  for (std::uint64_t i = archived; i + state->trace_length <= horizon; ++i) {
    std::uint32_t cursor = 0u;
    for (std::uint64_t j = i; j < horizon && cursor < state->trace_length; ++j) {
      const DirectExactHistoryRecord& record = records[j - archived];
      if (record.kind != DirectExactHistoryKind::motor_output) continue;
      if (record.value != state->trace[cursor].word) break;
      ++cursor;
    }
    if (cursor == state->trace_length) ++occurrences;
  }
  state->trace_history_occurrences = occurrences;
  state->trace_novel_composition = occurrences == 0u ? 1u : 0u;
}

__device__ inline void streams_divergence_receipt(
    const DirectParallelLanguageStreamsState& state,
    DirectStreamsDivergence* receipt) {
  DirectStreamsDivergence out{};
  out.grounded_sites = state.grounded_site_count;
  for (std::uint32_t i = 0u; i < state.grounded_site_count; ++i) {
    const DirectGroundedMeaningSite& site = state.grounded_sites[i];
    if (site.active == 0u) continue;
    if (site.action_node != kInvalidIndex) ++out.grounded_meaning_sites;
    out.grounded_mass_total_q16 += site.bind_mass_q16;
    if (site.bind_mass_q16 > out.grounded_mass_max_q16)
      out.grounded_mass_max_q16 = site.bind_mass_q16;
    bool surface_seen = false;
    bool context_seen = false;
    for (std::uint32_t c = 0u; c < kStreamsContextChannelCapacity; ++c) {
      surface_seen = surface_seen ||
                     out.grounded_surface_channels[c] == site.surface_channel;
      context_seen = context_seen ||
                     out.grounded_context_channels[c] == site.context_channel;
    }
    if (!surface_seen) {
      for (std::uint32_t c = 0u; c < kStreamsContextChannelCapacity; ++c)
        if (out.grounded_surface_channels[c] == 0u) {
          out.grounded_surface_channels[c] = site.surface_channel;
          break;
        }
    }
    if (!context_seen) {
      for (std::uint32_t c = 0u; c < kStreamsContextChannelCapacity; ++c)
        if (out.grounded_context_channels[c] == 0u) {
          out.grounded_context_channels[c] = site.context_channel;
          break;
        }
    }
    out.grounding_topology_hash = streams_fold(
        streams_fold(streams_fold(streams_fold(out.grounding_topology_hash,
                                               site.surface_channel),
                                  site.surface_value),
                     site.context_channel),
        site.context_value);
  }
  for (std::uint32_t c = 0u; c < kStreamsContextChannelCapacity; ++c)
    out.grounded_channel_kinds +=
        (out.grounded_surface_channels[c] != 0u ? 1u : 0u) +
        (out.grounded_context_channels[c] != 0u ? 1u : 0u);
  std::uint32_t distinct_words[kStreamsEmitterCapacity]{};
  std::uint32_t distinct_count = 0u;
  for (std::uint32_t i = 0u; i < state.edge_count; ++i) {
    const DirectArticulationEdge& edge = state.edges[i];
    if (edge.active == 0u) continue;
    out.articulation_growth_total += edge.growths;
    out.articulation_traversal_total += edge.traversals;
    out.articulation_topology_hash =
        streams_fold(streams_fold(out.articulation_topology_hash, edge.from_word),
                     edge.to_word);
    for (std::uint32_t w = 0u; w < 2u; ++w) {
      const std::uint32_t word = w == 0u ? edge.from_word : edge.to_word;
      bool seen = false;
      for (std::uint32_t d = 0u; d < distinct_count && !seen; ++d)
        seen = distinct_words[d] == word;
      if (!seen && distinct_count < kStreamsEmitterCapacity)
        distinct_words[distinct_count++] = word;
    }
    for (std::uint32_t s = 0u; s < state.grounded_site_count; ++s) {
      const DirectGroundedMeaningSite& site = state.grounded_sites[s];
      if (site.active == 0u) continue;
      out.matter_intersection +=
          (edge.from_word == site.surface_value ||
           edge.from_word == site.context_value ||
           edge.to_word == site.surface_value ||
           edge.to_word == site.context_value)
              ? 1u
              : 0u;
    }
  }
  out.articulation_distinct_words = distinct_count;
  *receipt = out;
}

__device__ inline std::uint32_t streams_lesion_grounding(
    DirectParallelLanguageStreamsState* state) {
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < state->grounded_site_count; ++i) {
    DirectGroundedMeaningSite& site = state->grounded_sites[i];
    if (site.active != 0u) {
      site.active = 0u;
      ++removed;
    }
  }
  ++state->lesion_grounding_events;
  state->revision_identity = streams_fold(state->revision_identity, removed);
  return removed;
}

__device__ inline std::uint32_t streams_lesion_articulation(
    DirectParallelLanguageStreamsState* state) {
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < state->edge_count; ++i) {
    DirectArticulationEdge& edge = state->edges[i];
    if (edge.active != 0u) {
      edge.active = 0u;
      ++removed;
    }
  }
  ++state->lesion_articulation_events;
  state->revision_identity = streams_fold(state->revision_identity, removed);
  return removed;
}

// Dose-matched sham: equal bytes of reserve matter touched, no acquired
// stream matter changed.
__device__ inline std::uint32_t streams_remote_sham(
    DirectParallelLanguageStreamsState* state, std::uint32_t matter) {
  state->control_hash = streams_fold(state->control_hash, matter);
  return matter <= kStreamsGroundedSiteCapacity ? matter
                                                : kStreamsGroundedSiteCapacity;
}

}  // namespace substrate::direct_network

#endif
