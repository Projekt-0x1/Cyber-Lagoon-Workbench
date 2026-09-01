#ifndef HARDWARE_NATIVE_DIRECT_ADULT_EMBODIED_SYMBOL_TRANSFER_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_EMBODIED_SYMBOL_TRANSFER_CUH

// f.language_body_skill (#1553).  This is an opaque contact-to-action
// organization, not a tokenizer, language router, or string generator.
// Repeated cross-channel contact recruits many individually insufficient
// resident sites.  Only a quorum reactivated by current exact-history contact
// may bias a grown motor node, and only actual world returns can teach that
// bias.  Observer names for the physical sources never enter this state.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_multimodal_grounding.cuh"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kEmbodiedSymbolSiteCapacity = 192u;
inline constexpr std::uint32_t kEmbodiedSymbolActionCapacity = 32u;
inline constexpr std::uint32_t kEmbodiedSymbolQuorum = 6u;

struct DirectEmbodiedSymbolSite {
  std::uint32_t channel_a;
  std::uint32_t value_a;
  std::uint32_t channel_b;
  std::uint32_t value_b;
  std::uint32_t action_node;
  std::uint32_t matter_identity;
  std::int32_t strength_q16;
  std::uint32_t active;
};

struct DirectEmbodiedActionSupport {
  std::uint32_t node;
  std::uint32_t settled_samples;
};

struct DirectEmbodiedSymbolTransferState {
  DirectMultimodalGroundingTable grounding;
  DirectEmbodiedSymbolSite sites[kEmbodiedSymbolSiteCapacity];
  DirectEmbodiedActionSupport actions[kEmbodiedSymbolActionCapacity];
  std::uint32_t site_count;
  std::uint32_t action_count;
  std::uint32_t cursor;
  std::uint32_t q_contacts;
  std::uint32_t resident_work;
  std::uint32_t world_returns;
  std::uint32_t lesion_events;
  std::uint32_t lesion_matter;
  std::uint32_t reacquired_sites;
  std::uint32_t fence_refusals;
  std::uint64_t raw_contact_hash;
  std::uint64_t control_hash;
  std::uint64_t revision_identity;
};

struct DirectEmbodiedActionReceipt {
  std::uint32_t action_node;
  std::uint32_t supporting_sites;
  std::uint32_t distinct_channels;
  std::uint32_t resident_matter;
  std::uint32_t resident_work;
  std::uint32_t concentration_q16;
  std::uint64_t revision_identity;
  std::uint32_t admitted;
};

static_assert(std::is_trivially_copyable_v<DirectEmbodiedSymbolTransferState>);
static_assert(std::has_unique_object_representations_v<DirectEmbodiedSymbolSite>);

__device__ inline std::uint64_t embodied_fold(std::uint64_t h,
                                              std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
  return h;
}

__device__ inline bool embodied_has_world_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t identity) {
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::world_return &&
        records[i].identity == identity)
      return true;
  return false;
}

__device__ inline void embodied_note_action(
    DirectEmbodiedSymbolTransferState* state, std::uint32_t node) {
  for (std::uint32_t i = 0u; i < state->action_count; ++i) {
    if (state->actions[i].node == node) {
      ++state->actions[i].settled_samples;
      return;
    }
  }
  if (state->action_count >= kEmbodiedSymbolActionCapacity) {
    ++state->fence_refusals;
    return;
  }
  state->actions[state->action_count++] = {node, 1u};
}

__device__ inline std::uint32_t embodied_preferred_action(
    const DirectEmbodiedSymbolTransferState& state) {
  std::uint32_t node = kInvalidIndex;
  std::uint32_t samples = 0u;
  for (std::uint32_t i = 0u; i < state.action_count; ++i) {
    const DirectEmbodiedActionSupport& candidate = state.actions[i];
    if (candidate.settled_samples > samples ||
        (candidate.settled_samples == samples && candidate.node < node)) {
      node = candidate.node;
      samples = candidate.settled_samples;
    }
  }
  return samples == 0u ? kInvalidIndex : node;
}

__device__ inline bool embodied_site_exists(
    const DirectEmbodiedSymbolTransferState& state,
    const DirectCrossModalPair& pair) {
  for (std::uint32_t i = 0u; i < state.site_count; ++i) {
    const DirectEmbodiedSymbolSite& site = state.sites[i];
    if (site.active != 0u && site.channel_a == pair.channel_a &&
        site.value_a == pair.value_a && site.channel_b == pair.channel_b &&
        site.value_b == pair.value_b)
      return true;
  }
  return false;
}

// Incremental production assimilation must preserve the same cross-modal
// coincidence law as a batched exact-history reconstruction.  The generic
// grounding helper owns a transient recent-contact window, so calling it on
// only [cursor,count) would forget a contact that arrived on the previous
// Adult tick.  Reconstruct only the bounded pre-cursor tail still inside the
// coherence window, then deposit pairs involving each NEW sensory contact.
// Old-old pairs are never replayed, so pair mass remains identical to one
// chronological pass over the complete history.
__device__ inline void embodied_grounding_ingest_incremental(
    DirectMultimodalGroundingTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count) {
  if (table == nullptr || records == nullptr || begin > count) return;
  DirectCrossModalContact recent[kGroundingRecentCapacity]{};
  std::uint32_t recent_count = 0u;

  std::uint32_t first_new_tick = 0u;
  for (std::uint32_t i = begin; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::sensory_contact) {
      first_new_tick = records[i].resident_tick;
      break;
    }

  if (first_new_tick != 0u) {
    for (std::uint32_t cursor = begin; cursor > 0u; --cursor) {
      const DirectExactHistoryRecord& record = records[cursor - 1u];
      if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
      if (first_new_tick < record.resident_tick ||
          first_new_tick - record.resident_tick > kGroundingCoherenceWindowTicks)
        break;
      if (recent_count < kGroundingRecentCapacity)
        recent[recent_count++] =
            DirectCrossModalContact{record.subject, record.value,
                                    record.resident_tick};
    }
    for (std::uint32_t i = 0u; i < recent_count / 2u; ++i) {
      const DirectCrossModalContact tmp = recent[i];
      recent[i] = recent[recent_count - 1u - i];
      recent[recent_count - 1u - i] = tmp;
    }
  }

  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    const DirectCrossModalContact contact{record.subject, record.value,
                                          record.resident_tick};
    grounding_note_exposure(table, contact.channel);
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick - recent[j].resident_tick
                                    : 0u;
      if (age <= kGroundingCoherenceWindowTicks)
        grounding_deposit_pair(table, recent[j], contact);
    }
    std::uint32_t kept = 0u;
    for (std::uint32_t j = 0u; j < recent_count; ++j) {
      const std::uint32_t age = contact.resident_tick >= recent[j].resident_tick
                                    ? contact.resident_tick - recent[j].resident_tick
                                    : 0u;
      if (age <= kGroundingCoherenceWindowTicks &&
          kept < kGroundingRecentCapacity)
        recent[kept++] = recent[j];
    }
    if (kept < kGroundingRecentCapacity) recent[kept++] = contact;
    recent_count = kept;
  }
}

// Assimilate only the new suffix [state.cursor, count).  The suffix is always
// read from the born adult's device exact history.  Repeated coherent contacts
// may regrow sites after a lesion; merely calling this twice without new
// history cannot change the organization.
__device__ inline void embodied_symbol_assimilate(
    DirectEmbodiedSymbolTransferState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr) return;
  if (count < state->cursor) state->cursor = 0u;
  const std::uint32_t begin = state->cursor;
  embodied_grounding_ingest_incremental(&state->grounding, records, begin, count);
  grounding_extract_objects(&state->grounding);

  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind == DirectExactHistoryKind::sensory_contact) {
      ++state->q_contacts;
      ++state->resident_work;
      state->raw_contact_hash = embodied_fold(
          embodied_fold(state->raw_contact_hash, record.subject), record.value);
      state->revision_identity = embodied_fold(state->revision_identity,
                                                record.identity);
    } else if (record.kind == DirectExactHistoryKind::world_return) {
      ++state->world_returns;
      ++state->resident_work;
      state->revision_identity = embodied_fold(state->revision_identity,
                                                record.identity);
    }
  }
  // World return, not first sight of the motor output, is the moment an action
  // becomes consequential evidence. Production normally observes those on
  // different ticks, so resolve each NEW return back to its already-recorded
  // motor exactly once. This is equivalent to the batched "motor has return"
  // predicate without requiring the motor to re-enter the incremental suffix.
  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& consequence = records[i];
    if (consequence.kind != DirectExactHistoryKind::world_return ||
        (consequence.flags & kDirectHistoryVerifiedObservation) == 0u)
      continue;
    for (std::uint32_t j = 0u; j < count; ++j) {
      const DirectExactHistoryRecord& motor = records[j];
      if (motor.kind != DirectExactHistoryKind::motor_output) continue;
      const bool same_ticket = consequence.identity == motor.identity ||
                               consequence.parent_identity == motor.identity;
      if (same_ticket && consequence.source == motor.source &&
          consequence.subject == motor.subject &&
          consequence.value == motor.value) {
        embodied_note_action(state, motor.source);
        break;
      }
    }
  }

  const std::uint32_t action = embodied_preferred_action(*state);
  if (action != kInvalidIndex) {
    for (std::uint32_t i = 0u; i < state->grounding.pair_count; ++i) {
      const DirectCrossModalPair& pair = state->grounding.pairs[i];
      if (!grounding_pair_strong(pair) || embodied_site_exists(*state, pair))
        continue;
      std::uint32_t slot = state->site_count;
      for (std::uint32_t j = 0u; j < state->site_count; ++j)
        if (state->sites[j].active == 0u) {
          slot = j;
          break;
        }
      if (slot >= kEmbodiedSymbolSiteCapacity) {
        ++state->fence_refusals;
        break;
      }
      if (slot == state->site_count) ++state->site_count;
      const bool regrowth = state->lesion_events != 0u;
      state->sites[slot] = {pair.channel_a,
                            pair.value_a,
                            pair.channel_b,
                            pair.value_b,
                            action,
                            slot + 1u,
                            pair.bind_mass_q16,
                            1u};
      if (regrowth) ++state->reacquired_sites;
      ++state->resident_work;
    }
  }
  state->cursor = count;
}

__device__ inline bool embodied_recent_leg(
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count, std::uint32_t channel, std::uint32_t value,
    std::uint32_t newest_tick) {
  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind == DirectExactHistoryKind::sensory_contact &&
        record.subject == channel && record.value == value &&
        newest_tick >= record.resident_tick &&
        newest_tick - record.resident_tick <= kGroundingCoherenceWindowTicks)
      return true;
  }
  return false;
}

__device__ inline DirectEmbodiedActionReceipt embodied_symbol_select(
    const DirectEmbodiedSymbolTransferState& state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count) {
  DirectEmbodiedActionReceipt receipt{};
  receipt.action_node = kInvalidIndex;
  receipt.resident_matter = state.site_count;
  receipt.resident_work = state.resident_work;
  receipt.revision_identity = state.revision_identity;
  if (records == nullptr || begin >= count) return receipt;
  std::uint32_t newest_tick = 0u;
  for (std::uint32_t i = begin; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::sensory_contact &&
        records[i].resident_tick > newest_tick)
      newest_tick = records[i].resident_tick;

  std::uint32_t nodes[kEmbodiedSymbolActionCapacity]{};
  std::uint32_t votes[kEmbodiedSymbolActionCapacity]{};
  std::uint32_t node_count = 0u;
  std::uint32_t channels[kGroundingExposureCapacity]{};
  std::uint32_t channel_count = 0u;
  for (std::uint32_t i = 0u; i < state.site_count; ++i) {
    const DirectEmbodiedSymbolSite& site = state.sites[i];
    if (site.active == 0u ||
        !embodied_recent_leg(records, begin, count, site.channel_a,
                             site.value_a, newest_tick) ||
        !embodied_recent_leg(records, begin, count, site.channel_b,
                             site.value_b, newest_tick))
      continue;
    ++receipt.supporting_sites;
    const std::uint32_t pair_channels[2] = {site.channel_a, site.channel_b};
    for (std::uint32_t c = 0u; c < 2u; ++c) {
      bool seen = false;
      for (std::uint32_t j = 0u; j < channel_count; ++j)
        seen = seen || channels[j] == pair_channels[c];
      if (!seen && channel_count < kGroundingExposureCapacity)
        channels[channel_count++] = pair_channels[c];
    }
    std::uint32_t slot = node_count;
    for (std::uint32_t j = 0u; j < node_count; ++j)
      if (nodes[j] == site.action_node) slot = j;
    if (slot == node_count && node_count < kEmbodiedSymbolActionCapacity) {
      nodes[node_count] = site.action_node;
      ++node_count;
    }
    if (slot < kEmbodiedSymbolActionCapacity) ++votes[slot];
  }
  receipt.distinct_channels = channel_count;
  std::uint32_t winning_votes = 0u;
  for (std::uint32_t i = 0u; i < node_count; ++i)
    if (votes[i] > winning_votes ||
        (votes[i] == winning_votes && nodes[i] < receipt.action_node)) {
      receipt.action_node = nodes[i];
      winning_votes = votes[i];
    }
  receipt.concentration_q16 =
      receipt.supporting_sites == 0u
          ? 0u
          : static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(1u << 16) /
                 receipt.supporting_sites));
  receipt.admitted = receipt.supporting_sites >= kEmbodiedSymbolQuorum &&
                     receipt.distinct_channels >= 2u &&
                     winning_votes == receipt.supporting_sites;
  if (receipt.admitted == 0u) receipt.action_node = kInvalidIndex;
  return receipt;
}

__device__ inline DirectEmbodiedActionReceipt embodied_symbol_express(
    const DirectEmbodiedSymbolTransferState& state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count, DirectBrain brain) {
  DirectEmbodiedActionReceipt receipt =
      embodied_symbol_select(state, records, begin, count);
  if (receipt.admitted == 0u || receipt.action_node >= brain.node_count) return receipt;
  atomicAdd(&brain.nodes[receipt.action_node].activation_q16, kQ16One);
  atomicAdd(&brain.nodes[receipt.action_node].credit_ema_q16, kQ16One / 8);
  return receipt;
}

__device__ inline std::uint32_t embodied_symbol_focal_lesion(
    DirectEmbodiedSymbolTransferState* state) {
  if (state == nullptr) return 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < state->site_count; ++i)
    if (state->sites[i].active != 0u) {
      state->sites[i].active = 0u;
      ++removed;
    }
  ++state->lesion_events;
  state->lesion_matter += removed;
  state->revision_identity = embodied_fold(state->revision_identity, removed);
  return removed;
}

// Equal bytes of remote reserve matter are touched, but no acquired site is
// changed.  This is the dose-matched sham, not a second lesion disguised as a
// control.
__device__ inline std::uint32_t embodied_symbol_remote_sham(
    DirectEmbodiedSymbolTransferState* state, std::uint32_t matter) {
  if (state == nullptr) return 0u;
  state->control_hash = embodied_fold(state->control_hash, matter);
  return matter <= kEmbodiedSymbolSiteCapacity ? matter
                                               : kEmbodiedSymbolSiteCapacity;
}

}  // namespace substrate::direct_network

#endif
