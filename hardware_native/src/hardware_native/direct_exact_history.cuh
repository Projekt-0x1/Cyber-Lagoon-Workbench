#ifndef HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_CUH
#define HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_network_recipe.hpp"
#if defined(__CUDACC__)
#define DIRECT_EXACT_HISTORY_HD __host__ __device__
#else
#define DIRECT_EXACT_HISTORY_HD
#endif
namespace substrate::direct_network {
inline constexpr std::uint32_t kDirectExactHistoryHotPageCapacity = 16384u;
inline constexpr std::uint32_t kDirectRecentSensoryCapacity = 64u;
inline constexpr std::uint32_t kDirectTopologyHistoryResident = 0u, kDirectTopologyHistoryMaintenance = 1u;

enum class DirectExactHistoryKind : std::uint32_t {
  empty = 0u,
  sensory_contact = 1u,
  motor_output = 2u,
  topology_growth = 3u,
  topology_retraction = 4u, world_return = 5u, sparse_credit = 6u, dense_shatter = 7u, homeostatic_weight = 8u, recipe_commit = 9u, recipe_revision = 10u,
  unexpected_presence = 11u, omitted_consequence = 12u, source_assertion = 13u,
  network_credit = 14u,
};

// Provenance is an attribution, never an inference from payload resemblance.
// Only membrane-authenticated contact and a settled world return may carry the
// verified bit. All other branches remain explicitly prospective.
inline constexpr std::uint32_t kDirectHistoryVerifiedObservation = 1u << 31;
inline constexpr std::uint32_t kDirectHistoryPayloadFlags =
    ~kDirectHistoryVerifiedObservation;

enum class DirectSpeculativeProvenance : std::uint32_t {
  unresolved = 0u,
  endogenous_simulation = 1u,
  verified_world_observation = 2u,
};

struct alignas(8) DirectExactHistoryRecord {
  std::uint64_t sequence;
  std::uint64_t identity;
  std::uint64_t parent_identity;
  std::uint32_t resident_tick;
  std::uint32_t event_tick;
  DirectExactHistoryKind kind;
  std::uint32_t source;
  std::uint32_t subject;
  std::uint32_t value;
  std::uint32_t context;
  std::uint32_t flags;
  std::uint64_t incarnation_before;
  std::uint64_t incarnation_after;
  std::int64_t resource_delta;
};

// Shared bounded current-perception state. This is not archival history and
// carries no semantic interpretation: it is only the recent authenticated
// sensory surface needed by ordinary resident consumers across hot-page
// rotation. Consumers apply their own resident-time horizon at read time.
struct alignas(8) DirectRecentSensoryContact {
  std::uint64_t sequence;
  std::uint64_t identity;
  std::uint32_t resident_tick;
  std::uint32_t event_tick;
  std::uint32_t source;
  std::uint32_t subject;
  std::uint32_t value;
  std::uint32_t context;
  std::uint32_t flags;
  std::uint32_t reserved;
};
struct alignas(8) DirectRecentSensoryState {
  DirectRecentSensoryContact contacts[kDirectRecentSensoryCapacity];
  std::uint32_t begin;
  std::uint32_t count;
  std::uint64_t writes;
  std::uint64_t evictions;
};
static_assert(std::is_standard_layout_v<DirectRecentSensoryContact> &&
              std::is_trivial_v<DirectRecentSensoryContact> &&
              std::has_unique_object_representations_v<DirectRecentSensoryContact>);
static_assert(std::is_standard_layout_v<DirectRecentSensoryState> &&
              std::is_trivial_v<DirectRecentSensoryState> &&
              std::has_unique_object_representations_v<DirectRecentSensoryState>);
static_assert(std::is_standard_layout_v<DirectExactHistoryRecord> &&
              std::is_trivial_v<DirectExactHistoryRecord> &&
              std::has_unique_object_representations_v<DirectExactHistoryRecord>);

DIRECT_EXACT_HISTORY_HD inline void stage_mismatch_credit_history_record(
    DirectExactHistoryRecord* record, DirectExactHistoryKind kind,
    std::uint64_t identity, std::uint64_t parent_identity,
    std::uint32_t resident_tick, std::uint32_t horizon_tick,
    std::uint32_t route_index, std::uint32_t context_signature,
    std::uint32_t source_incarnation, std::uint64_t route_incarnation,
    std::int32_t credit_delta_q16, bool verified_actual) {
  if (record == nullptr) return;
  *record = {};
  record->identity = identity;
  record->parent_identity = parent_identity;
  record->resident_tick = resident_tick;
  record->event_tick = horizon_tick;
  record->kind = kind;
  record->source = route_index;
  record->subject = context_signature;
  record->value = source_incarnation;
  record->flags = verified_actual ? kDirectHistoryVerifiedObservation : 0u;
  record->incarnation_before = route_incarnation;
  record->incarnation_after = route_incarnation;
  record->resource_delta = credit_delta_q16;
}

inline constexpr std::uint32_t kDirectExactHistoryTierIndexCapacity = 8u;
inline constexpr std::uint32_t kDirectExactHistoryPromotionAccesses = 2u;
inline constexpr std::uint64_t kDirectExactHistoryWarmObjectCapacity =
    256u + sizeof(DirectExactHistoryRecord) * kDirectExactHistoryHotPageCapacity;

struct DirectExactHistoryTierIndexEntry {
  recipe::Root256 address;
  std::uint32_t access_count;
  std::uint32_t cold_fetch_count;
  std::uint32_t last_access_tick;
  std::uint32_t valid;
};
static_assert(std::is_standard_layout_v<DirectExactHistoryTierIndexEntry> &&
              std::is_trivial_v<DirectExactHistoryTierIndexEntry> &&
              std::has_unique_object_representations_v<DirectExactHistoryTierIndexEntry>);

struct alignas(8) DirectExactHistoryWarmPage {
  recipe::Root256 address;
  std::uint64_t object_bytes;
  std::uint32_t valid;
  std::uint32_t reserved;
  alignas(8) unsigned char bytes[kDirectExactHistoryWarmObjectCapacity];
};
static_assert(std::is_standard_layout_v<DirectExactHistoryWarmPage> &&
              std::is_trivial_v<DirectExactHistoryWarmPage> &&
              std::has_unique_object_representations_v<DirectExactHistoryWarmPage>);

// Resident, semantics-blind placement state.  The host may transport an exact
// object requested here, but cannot choose which address becomes warm.
struct alignas(8) DirectExactHistoryTierState {
  DirectExactHistoryTierIndexEntry entries[kDirectExactHistoryTierIndexCapacity];
  std::uint32_t entry_count;
  std::uint32_t promotion_threshold;
  std::uint32_t promotion_index;
  std::uint32_t promotion_pending;
  std::uint64_t warm_hits;
  std::uint64_t cold_fetches;
  DirectExactHistoryWarmPage warm;
};
static_assert(std::is_standard_layout_v<DirectExactHistoryTierState> &&
              std::is_trivial_v<DirectExactHistoryTierState> &&
              std::has_unique_object_representations_v<DirectExactHistoryTierState>);

struct DirectExactHistoryHotPage {
  std::uint64_t prefix_root, page_prefix_root, next_sequence;
  std::uint64_t archived_record_count, archived_bytes, archive_capacity_bytes;
  recipe::Root256 archive_chain_head; std::uint32_t committed_slots, archived_pages;
  std::uint32_t sealed, overflow_refusals;
  std::uint32_t phase_base;
  std::uint32_t phase_width;
  DirectExactHistoryKind phase_kind;
  std::uint32_t phase_admitted;
  std::uint32_t phase_tick;
  std::uint32_t last_phase_records;
  std::uint32_t reserved, archive_reserved;
  DirectRecentSensoryState recent_sensory;
  DirectExactHistoryRecord records[kDirectExactHistoryHotPageCapacity];
};
static_assert(std::is_standard_layout_v<DirectExactHistoryHotPage> &&
              std::is_trivial_v<DirectExactHistoryHotPage> &&
              std::has_unique_object_representations_v<DirectExactHistoryHotPage>);
DIRECT_EXACT_HISTORY_HD inline std::uint64_t exact_history_fold_word(
    std::uint64_t root, std::uint64_t value) {
  return (root ^ value) * 0x100000001b3ull;
}
DIRECT_EXACT_HISTORY_HD inline void exact_history_note_recent_sensory(
    DirectRecentSensoryState* state, const DirectExactHistoryRecord& record) {
  if (state == nullptr || record.kind != DirectExactHistoryKind::sensory_contact ||
      (record.flags & kDirectHistoryVerifiedObservation) == 0u)
    return;
  if (state->count == kDirectRecentSensoryCapacity) {
    state->begin = (state->begin + 1u) % kDirectRecentSensoryCapacity;
    --state->count;
    ++state->evictions;
  }
  const std::uint32_t slot =
      (state->begin + state->count) % kDirectRecentSensoryCapacity;
  state->contacts[slot] = DirectRecentSensoryContact{
      record.sequence, record.identity, record.resident_tick, record.event_tick,
      record.source, record.subject, record.value, record.context, record.flags, 0u};
  ++state->count;
  ++state->writes;
}

DIRECT_EXACT_HISTORY_HD inline std::uint32_t exact_history_recent_sensory_records(
    const DirectRecentSensoryState& recent, std::uint32_t current_tick,
    std::uint32_t horizon_ticks, DirectExactHistoryRecord* out,
    std::uint32_t out_capacity = kDirectRecentSensoryCapacity) {
  if (out == nullptr || out_capacity == 0u) return 0u;
  std::uint32_t written = 0u;
  for (std::uint32_t i = 0u; i < recent.count && written < out_capacity; ++i) {
    const auto& contact =
        recent.contacts[(recent.begin + i) % kDirectRecentSensoryCapacity];
    if (current_tick < contact.resident_tick ||
        current_tick - contact.resident_tick > horizon_ticks)
      continue;
    auto& record = out[written++];
    record = {};
    record.sequence = contact.sequence;
    record.identity = contact.identity;
    record.resident_tick = contact.resident_tick;
    record.event_tick = contact.event_tick;
    record.kind = DirectExactHistoryKind::sensory_contact;
    record.source = contact.source;
    record.subject = contact.subject;
    record.value = contact.value;
    record.context = contact.context;
    record.flags = contact.flags;
  }
  return written;
}

DIRECT_EXACT_HISTORY_HD inline std::uint64_t exact_history_fold_record(
    std::uint64_t root, const DirectExactHistoryRecord& record) {
  root = exact_history_fold_word(root, record.sequence);
  root = exact_history_fold_word(root, record.identity);
  root = exact_history_fold_word(root, record.parent_identity);
  root = exact_history_fold_word(root, record.resident_tick);
  root = exact_history_fold_word(root, record.event_tick);
  root = exact_history_fold_word(root, static_cast<std::uint32_t>(record.kind));
  root = exact_history_fold_word(root, record.source);
  root = exact_history_fold_word(root, record.subject);
  root = exact_history_fold_word(root, record.value);
  root = exact_history_fold_word(root, record.context);
  root = exact_history_fold_word(root, record.flags);
  root = exact_history_fold_word(root, record.incarnation_before);
  root = exact_history_fold_word(root, record.incarnation_after);
  return exact_history_fold_word(root, static_cast<std::uint64_t>(record.resource_delta));
}
DIRECT_EXACT_HISTORY_HD inline bool begin_exact_history_phase(
    DirectExactHistoryHotPage* history, DirectExactHistoryKind kind,
    std::uint32_t width, std::uint32_t tick) {
  if (history == nullptr) return true;
  if (history->sealed != 0u ||
      history->phase_kind != DirectExactHistoryKind::empty ||
      width > kDirectExactHistoryHotPageCapacity - history->committed_slots) {
    history->sealed = 1u;
    history->phase_admitted = 0u;
    history->last_phase_records = 0u;
    history->overflow_refusals += width == 0u ? 1u : width;
    return false;
  }
  history->phase_base = history->committed_slots;
  history->phase_width = width;
  history->phase_kind = kind;
  history->phase_admitted = 1u;
  history->phase_tick = tick;
  history->last_phase_records = 0u;
  for (std::uint32_t i = 0u; i < width; ++i)
    history->records[history->phase_base + i] = DirectExactHistoryRecord{};
  return true;
}

DIRECT_EXACT_HISTORY_HD inline std::uint32_t finish_exact_history_phase(
    DirectExactHistoryHotPage* history) {
  if (history == nullptr || history->phase_admitted == 0u) return 0u;
  std::uint32_t written = 0u;
  for (std::uint32_t i = 0u; i < history->phase_width; ++i) {
    DirectExactHistoryRecord record = history->records[history->phase_base + i];
    if (record.kind == DirectExactHistoryKind::empty) continue;
    record.sequence = history->next_sequence + written + 1u;
    history->records[history->phase_base + written] = record;
    history->prefix_root = exact_history_fold_record(history->prefix_root, record);
    exact_history_note_recent_sensory(&history->recent_sensory, record);
    ++written;
  }
  for (std::uint32_t i = written; i < history->phase_width; ++i)
    history->records[history->phase_base + i] = DirectExactHistoryRecord{};
  history->committed_slots += written;
  history->next_sequence += written;
  history->last_phase_records = written;
  history->sealed = history->committed_slots == kDirectExactHistoryHotPageCapacity;
  history->phase_base = 0u;
  history->phase_width = 0u;
  history->phase_kind = DirectExactHistoryKind::empty;
  history->phase_admitted = 0u;
  history->phase_tick = 0u;
  return written;
}

DIRECT_EXACT_HISTORY_HD inline void stage_sensory_history_record(
    DirectExactHistoryRecord* record, std::uint64_t ticket,
    std::uint32_t resident_tick, std::uint32_t event_tick, std::uint32_t node,
    std::uint32_t channel, std::uint32_t word, std::uint32_t context,
    std::uint32_t origin, bool membrane_authenticated = false) {
  if (record == nullptr) return;
  record->identity = ticket; record->resident_tick = resident_tick;
  record->event_tick = event_tick; record->kind = DirectExactHistoryKind::sensory_contact;
  record->source = node; record->subject = channel; record->value = word;
  record->context = context;
  record->flags = (origin & kDirectHistoryPayloadFlags) |
      (membrane_authenticated ? kDirectHistoryVerifiedObservation : 0u);
}

DIRECT_EXACT_HISTORY_HD inline void stage_motor_history_record(
    DirectExactHistoryRecord* record, std::uint64_t ticket, std::uint64_t parent_ticket,
    std::uint32_t tick, std::uint32_t node, std::uint32_t channel, std::uint32_t word,
    std::uint32_t context, std::uint32_t flags) {
  if (record == nullptr) return;
  record->identity = ticket; record->parent_identity = parent_ticket;
  record->resident_tick = tick; record->event_tick = tick;
  record->kind = DirectExactHistoryKind::motor_output;
  record->source = node; record->subject = channel; record->value = word;
  record->context = context; record->flags = flags;
}
DIRECT_EXACT_HISTORY_HD inline void stage_world_return_history_record(DirectExactHistoryRecord* record,
    std::uint64_t ticket, std::uint64_t parent_ticket, std::uint32_t resident_tick, std::uint32_t emission_tick, std::uint32_t node,
    std::uint32_t channel, std::uint32_t word, std::uint32_t context,
    std::uint32_t mismatch, std::uint32_t transport_cursor,
    std::int32_t reward_q16, std::uint32_t causal_origin = 1u) {
  if (record == nullptr) return;
  record->identity = ticket; record->parent_identity = parent_ticket;
  record->resident_tick = resident_tick; record->event_tick = emission_tick; record->kind = DirectExactHistoryKind::world_return;
  record->source = node; record->subject = channel; record->value = word;
  record->context = context;
  record->flags = (mismatch & kDirectHistoryPayloadFlags) |
      kDirectHistoryVerifiedObservation;
  record->incarnation_before = transport_cursor;
  record->incarnation_after = causal_origin;
  record->resource_delta = reward_q16;
}
DIRECT_EXACT_HISTORY_HD inline void stage_sparse_credit_history_record(DirectExactHistoryRecord* record,
    std::uint64_t action_ticket, std::uint64_t participant_ticket, std::uint32_t commit_tick,
    std::uint32_t emission_tick, std::uint32_t source, std::uint32_t route_index, std::uint32_t target, std::uint32_t context,
    std::int32_t prior_q16, std::uint64_t route_incarnation,
    std::uint32_t claim_incarnation, std::int32_t applied_delta_q16) {
  if (record == nullptr) return;
  record->identity = action_ticket; record->parent_identity = participant_ticket;
  record->resident_tick = commit_tick; record->event_tick = emission_tick; record->kind = DirectExactHistoryKind::sparse_credit;
  record->source = source; record->subject = route_index; record->value = target;
  record->context = context; record->flags = static_cast<std::uint32_t>(prior_q16); record->incarnation_before = route_incarnation;
  record->incarnation_after = claim_incarnation; record->resource_delta = applied_delta_q16;
}
DIRECT_EXACT_HISTORY_HD inline void stage_network_credit_history_record(
    DirectExactHistoryRecord* record, std::uint64_t action_ticket,
    std::uint64_t active_network_identity, std::uint64_t recruitment_identity,
    std::uint32_t commit_tick, std::uint32_t emission_tick,
    std::int64_t prior_credit_q16, std::int64_t next_credit_q16,
    std::int64_t applied_delta_q16) {
  if (record == nullptr) return;
  *record = {};
  record->identity = action_ticket;
  record->parent_identity = active_network_identity;
  record->resident_tick = commit_tick;
  record->event_tick = emission_tick;
  record->kind = DirectExactHistoryKind::network_credit;
  record->source = static_cast<std::uint32_t>(recruitment_identity);
  record->subject = static_cast<std::uint32_t>(recruitment_identity >> 32u);
  record->incarnation_before = static_cast<std::uint64_t>(prior_credit_q16);
  record->incarnation_after = static_cast<std::uint64_t>(next_credit_q16);
  record->resource_delta = applied_delta_q16;
}
DIRECT_EXACT_HISTORY_HD inline void stage_topology_history_record(DirectExactHistoryRecord* record,
    DirectExactHistoryKind kind, std::uint32_t tick, std::uint32_t source, std::uint32_t route_index, std::uint32_t target,
    std::uint32_t flags, std::uint64_t incarnation_before, std::uint64_t incarnation_after,
    std::int64_t resource_delta, std::uint32_t context = kDirectTopologyHistoryResident) {
  if (record == nullptr) return;
  record->resident_tick = tick; record->event_tick = tick; record->kind = kind;
  record->source = source; record->subject = route_index; record->value = target;
  record->context = context; record->flags = flags;
  record->incarnation_before = incarnation_before; record->incarnation_after = incarnation_after;
  record->resource_delta = resource_delta;
}

inline constexpr std::uint32_t kDirectPredecessorShadowCapacity = 8u;
inline constexpr std::uint32_t kDirectPredecessorShadowScanLimit = 128u;

struct DirectPredecessorShadowEntry {
  std::uint64_t identity;
  std::uint64_t parent_identity;
  std::uint64_t sequence;
  DirectExactHistoryKind kind;
  std::uint32_t source;
  std::uint32_t subject;
  std::uint32_t resident_tick;
  std::uint32_t event_tick;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectPredecessorShadowEntry> &&
              std::is_trivial_v<DirectPredecessorShadowEntry> &&
              std::has_unique_object_representations_v<DirectPredecessorShadowEntry>);

struct DirectPredecessorShadowTrace {
  DirectPredecessorShadowEntry entries[kDirectPredecessorShadowCapacity];
  std::uint64_t target_identity;
  std::uint32_t entry_count;
  std::uint32_t evidence_boundaries;
  std::uint32_t verified_observation_boundaries;
  std::uint32_t unverified_observation_boundaries;
  std::uint32_t scan_work;
  std::uint32_t unresolved_identities;
  std::uint32_t complete;
  std::uint32_t overflow;
  DirectSpeculativeProvenance provenance;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectPredecessorShadowTrace> &&
              std::is_trivial_v<DirectPredecessorShadowTrace> &&
              std::has_unique_object_representations_v<DirectPredecessorShadowTrace>);

// Trace recent exact parent links backwards without reclassifying actual
// contact as an endogenous prediction. Work is bounded independently of the
// hot-page size; a chain outside the window is reported incomplete rather than
// guessed. Multiple records may share an action identity, so every matching
// sparse contributor is retained until the finite shadow frontier fills.
DIRECT_EXACT_HISTORY_HD inline bool trace_direct_predecessor_shadows(
    const DirectExactHistoryHotPage& history, std::uint64_t target_identity,
    DirectPredecessorShadowTrace* out) {
  if (out == nullptr || target_identity == 0u) return false;
  DirectPredecessorShadowTrace trace{};
  trace.target_identity = target_identity;
  std::uint64_t pending[kDirectPredecessorShadowCapacity]{};
  std::uint32_t pending_count = 1u, cursor = 0u;
  pending[0] = target_identity;
  while (cursor < pending_count && trace.overflow == 0u) {
    const std::uint64_t sought = pending[cursor++];
    bool found = false;
    const std::uint32_t committed = history.committed_slots <
            kDirectExactHistoryHotPageCapacity
        ? history.committed_slots
        : kDirectExactHistoryHotPageCapacity;
    std::uint32_t scanned = 0u;
    for (std::uint32_t reverse = committed;
         reverse != 0u && scanned < kDirectPredecessorShadowScanLimit;
         --reverse, ++scanned) {
      const DirectExactHistoryRecord& record = history.records[reverse - 1u];
      ++trace.scan_work;
      if (record.identity != sought || record.kind == DirectExactHistoryKind::empty)
        continue;
      found = true;
      if (record.kind == DirectExactHistoryKind::sensory_contact ||
          record.kind == DirectExactHistoryKind::world_return) {
        ++trace.evidence_boundaries;
        if ((record.flags & kDirectHistoryVerifiedObservation) != 0u)
          ++trace.verified_observation_boundaries;
        else
          ++trace.unverified_observation_boundaries;
        if (record.parent_identity != 0u &&
            record.parent_identity != 0xffffffffffffffffULL) {
          bool known = false;
          for (std::uint32_t i = 0u; i < pending_count; ++i)
            known |= pending[i] == record.parent_identity;
          if (!known) {
            if (pending_count == kDirectPredecessorShadowCapacity) {
              trace.overflow = 1u;
              break;
            }
            pending[pending_count++] = record.parent_identity;
          }
        }
        continue;
      }
      if (trace.entry_count == kDirectPredecessorShadowCapacity) {
        trace.overflow = 1u;
        break;
      }
      trace.entries[trace.entry_count++] = DirectPredecessorShadowEntry{
          record.identity, record.parent_identity, record.sequence, record.kind,
          record.source, record.subject, record.resident_tick, record.event_tick, 0u};
      if (record.parent_identity == 0u ||
          record.parent_identity == 0xffffffffffffffffULL)
        continue;
      bool known = false;
      for (std::uint32_t i = 0u; i < pending_count; ++i)
        known |= pending[i] == record.parent_identity;
      if (!known) {
        if (pending_count == kDirectPredecessorShadowCapacity) {
          trace.overflow = 1u;
          break;
        }
        pending[pending_count++] = record.parent_identity;
      }
    }
    if (!found) ++trace.unresolved_identities;
  }
  trace.complete = trace.overflow == 0u && trace.unresolved_identities == 0u;
  trace.provenance = trace.complete == 0u
      ? DirectSpeculativeProvenance::unresolved
      : trace.verified_observation_boundaries != 0u
          ? DirectSpeculativeProvenance::verified_world_observation
          : DirectSpeculativeProvenance::endogenous_simulation;
  *out = trace;
  return trace.complete != 0u;
}

}  // namespace substrate::direct_network

#undef DIRECT_EXACT_HISTORY_HD

#endif  // HARDWARE_NATIVE_DIRECT_EXACT_HISTORY_CUH
