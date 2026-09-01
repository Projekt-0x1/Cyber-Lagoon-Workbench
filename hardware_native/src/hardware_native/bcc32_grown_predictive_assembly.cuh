#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define BCC32_PA_HD __host__ __device__
#define BCC32_PA_DEVICE __device__
#define BCC32_PA_GLOBAL __global__
#else
#define BCC32_PA_HD
#define BCC32_PA_DEVICE
#define BCC32_PA_GLOBAL
#endif
namespace bcc32::grown_predictive_assembly {
// This is deliberately a small, fixed resident substrate.  A later adult
// integration may allocate several instances; none of the types below carries
// a language or task-specific identifier.
constexpr std::uint32_t kPopulationCount = 6u;
constexpr std::uint32_t kP0 = 0u;
constexpr std::uint32_t kP1 = 1u;
constexpr std::uint32_t kP2 = 2u;
constexpr std::uint32_t kP3 = 3u;
constexpr std::uint32_t kP4 = 4u;
constexpr std::uint32_t kP5 = 5u;
constexpr std::uint32_t kRawColumnCount = 257u;
constexpr std::uint32_t kContextCellsPerColumn = 4u;
constexpr std::uint32_t kCellsPerPopulation =
    kRawColumnCount * kContextCellsPerColumn;
constexpr std::uint32_t kSegmentsPerPopulation = 1024u;
constexpr std::uint32_t kSynapsesPerPopulation = 8192u;
constexpr std::uint32_t kEligibilityPerPopulation = 256u;
constexpr std::uint32_t kHistoryDepth = 32u;
constexpr std::uint32_t kJournalCapacity = 16384u;
constexpr std::uint32_t kMaxSegmentSynapses = 8u;
constexpr std::uint32_t kMaxContextCells = 4u;
constexpr std::uint32_t kReceiptWinners = 16u;
constexpr std::uint32_t kMotorTrajectoryCapacity = 128u;
constexpr std::uint32_t kPublicationCapacity = 32u;
constexpr std::uint32_t kContactMemberCapacity = 8u;
constexpr std::uint32_t kSeparatorCount = 256u;
constexpr std::uint32_t kErrorJournalFull = 1u << 0u;
constexpr std::uint32_t kErrorAllocatorExhausted = 1u << 1u;
constexpr std::uint32_t kErrorForbiddenEdge = 1u << 2u;
constexpr std::uint32_t kErrorMixedSource = 1u << 3u;
constexpr std::uint32_t kErrorInvalidDelay = 1u << 4u;
constexpr std::uint32_t kErrorExpiredEligibility = 1u << 5u;
constexpr std::uint32_t kErrorInvariant = 1u << 6u;
constexpr std::uint32_t kErrorCommand = 1u << 7u;
enum class RunMode : std::uint8_t { kLearn = 0u, kFreeze = 1u, kProbe = 2u };
struct Event {
  std::uint8_t value = 0u;
  std::uint8_t contact_boundary = 0u;
  std::uint8_t channel = 0u;
};
struct RunCommand {
  Event event{};
  RunMode mode = RunMode::kLearn;
  std::uint8_t has_event = 1u;
  std::uint8_t rollback_on_error = 1u;
  std::uint8_t learn_population_mask =
      static_cast<std::uint8_t>((1u << kPopulationCount) - 1u);
  std::uint32_t tick = 0u;
};
struct PopulationActivityFrame {
  std::uint8_t active[kPopulationCount][kCellsPerPopulation]{};
  std::uint32_t tick = 0u;
};
struct GrowthContract {
  std::uint8_t target_population = 0u;
  std::uint8_t source_population = 0u;
  std::uint16_t target_cell = 0u;
  std::uint8_t synapse_count = 0u;
  std::uint16_t threshold = 0u;
  std::uint16_t source_cells[kMaxSegmentSynapses]{};
  std::uint8_t delays[kMaxSegmentSynapses]{};
  std::int8_t weights[kMaxSegmentSynapses]{};
};
// A cell is only a local state carrier.  It has no symbolic identity.
struct Cell {
  std::int32_t activation = 0;
  std::int32_t utility = 0;
  std::uint32_t generation = 0u;
  std::uint8_t live = 0u;
  std::uint8_t frozen = 0u;
  std::uint16_t reserved = 0u;
};
// Segment fields are intentionally limited to the resident structural contract.
struct Segment {
  std::uint16_t target_cell = 0u;
  std::uint8_t source_population = 0u;
  std::uint8_t synapse_count = 0u;
  std::uint16_t synapse_begin = 0u;
  std::int16_t utility = 0;
  std::int16_t strength = 0;
  std::uint16_t threshold = 0u;
  std::uint8_t live = 0u;
  std::uint8_t frozen = 0u;
  std::uint8_t route_class = 0u;
  std::uint8_t reserved = 0u;
  std::uint32_t target_generation = 0u;
  std::uint32_t generation = 0u;
};
struct Synapse {
  std::uint16_t source_cell = 0u;
  std::uint8_t delay = 0u;
  std::int8_t weight = 0;
  std::int16_t utility = 0;
  std::uint8_t live = 0u;
  std::uint8_t frozen = 0u;
  std::uint16_t reserved = 0u;
  std::uint32_t generation = 0u;
};
// The exact segment and active synapses are retained until the delayed residual.
struct Eligibility {
  std::uint8_t target_population = 0u;
  std::uint8_t source_population = 0u;
  std::uint16_t segment_index = 0u;
  std::uint8_t active_synapse_count = 0u;
  std::uint8_t live = 0u;
  std::int16_t trace = 0;
  std::uint16_t active_synapses[kMaxSegmentSynapses]{};
  std::uint32_t segment_generation = 0u;
  std::uint32_t target_generation = 0u;
  std::uint32_t active_synapse_generations[kMaxSegmentSynapses]{};
  std::uint8_t active_synapse_delays[kMaxSegmentSynapses]{};
  std::uint32_t publication_tick = 0u;
  std::uint32_t due_tick = 0u;
  std::uint16_t credit_group = 0u;
  std::uint16_t publication_index = 0u;
};
struct HistoryEntry {
  Event event{};
  std::uint8_t active_count[kPopulationCount]{};
  std::uint16_t active_cells[kPopulationCount][kReceiptWinners]{};
  std::uint32_t tick = 0u;
};
enum class JournalKind : std::uint8_t {
  kCell = 0u,
  kSegment = 1u,
  kSynapse = 2u,
  kEligibility = 3u,
  kHistory = 4u,
  kMetadata = 5u,
  kAllocator = 6u,
};
struct Publication {
  std::uint8_t live = 0u;
  std::uint8_t target_population = 0u;
  std::uint16_t eligibility_index = 0u;
  std::uint32_t due_tick = 0u;
  std::uint16_t credit_group = 0u;
  std::uint16_t reserved = 0u;
};
struct ContactState {
  std::uint8_t active = 0u;
  std::uint8_t pending_p0_count = 0u;
  std::uint8_t pending_p1_count = 0u;
  std::uint8_t recalled_form_count = 0u;
  std::uint8_t replay_start_count = 0u;
  std::uint8_t separator_count = 0u;
  std::uint8_t has_recalled = 0u;
  std::uint8_t recalled_contact_count = 0u;
  std::uint8_t last_value = 0u;
  std::uint8_t has_last_value = 0u;
  std::uint16_t current_contact_cell = 0u;
  std::uint16_t recalled_contact_cell = 0u;
  std::uint16_t recalled_contact_cells[kReceiptWinners]{};
  // Event-local resident form publications.  These are cells, not surface
  // values; consumers must observe the learned P1 dispositions directly.
  std::uint16_t event_closed_content_p1_cell = 0u;
  std::uint16_t event_separator_p1_cell = 0u;
  std::uint8_t event_closed_content_p1_valid = 0u;
  std::uint8_t event_separator_p1_valid = 0u;
  std::uint32_t start_tick = 0u;
  std::uint32_t contact_count = 0u;
  std::uint16_t pending_p0[kContactMemberCapacity]{};
  std::uint16_t pending_p1[kContactMemberCapacity]{};
  std::uint16_t recalled_p1[kContactMemberCapacity]{};
  std::uint32_t boundary_evidence[kSeparatorCount]{};
  std::uint32_t interior_evidence[kSeparatorCount]{};
  std::uint32_t transition_degree[kSeparatorCount]{};
  // A compact learned successor relation.  The row is the preceding raw
  // event and the bits are the distinct successors observed after it.
  std::uint32_t successor_bits[kSeparatorCount][8u]{};
  // Its resident transpose keeps bidirectional boundary scoring local and
  // bounded instead of rescanning the complete transition table per event.
  std::uint32_t predecessor_bits[kSeparatorCount][8u]{};
};
struct MetadataSnapshot {
  std::uint32_t tick = 0u;
  std::uint32_t predicted_mask = 0u;
  std::uint32_t error_bits = 0u;
  std::uint32_t transaction_mark = 0u;
  std::uint16_t predicted_value = 0u;
  std::uint8_t prediction_active = 0u;
  std::uint8_t predicted_population = 0u;
  std::uint16_t predicted_segment = 0u;
  std::uint16_t predicted_target = 0u;
  std::uint16_t publication_count = 0u;
  std::uint16_t frozen_edge_class_mask = 0u;
  std::uint64_t represented_matter = 0u;
  std::uint64_t free_matter = 0u;
  std::uint64_t internal_growth_calls = 0u;
  std::uint64_t external_growth_calls = 0u;
  std::uint64_t canonical_snapshot_hash = 0u;
  std::uint32_t canonical_snapshot_tick = 0u;
  std::uint8_t canonical_snapshot_valid = 0u;
};
// One entry restores the complete old value of the mutated resident state.
// Allocator stacks are derived caches rebuilt from live bits during rollback.
struct JournalEntry {
  JournalKind kind = JournalKind::kCell;
  std::uint8_t population = 0u;
  std::uint16_t index = 0u;
  Cell old_cell{};
  Segment old_segment{};
  Synapse old_synapse{};
  Eligibility old_eligibility{};
  HistoryEntry old_history{};
  MetadataSnapshot old_metadata{};
  std::uint32_t old_u32[8]{};
  std::uint64_t old_u64[4]{};
};
struct Population {
  Cell cells[kCellsPerPopulation]{};
  Segment segments[kSegmentsPerPopulation]{};
  Synapse synapses[kSynapsesPerPopulation]{};
  Eligibility eligibilities[kEligibilityPerPopulation]{};

  // These stacks are derived from the live flags, but are finite resident
  // allocator state exposed for later growth organs.
  std::uint16_t free_segments[kSegmentsPerPopulation]{};
  std::uint16_t free_synapses[kSynapsesPerPopulation]{};
  std::uint32_t free_segment_count = 0u;
  std::uint32_t free_synapse_count = 0u;
  std::uint64_t represented_matter = 0u;
  std::uint64_t free_matter = 0u;
};
struct DeviceReceipt {
  std::uint32_t tick = 0u;
  std::uint32_t prediction_count = 0u;
  std::uint32_t correct_count = 0u;
  std::uint32_t false_prediction_count = 0u;
  std::uint32_t expired_eligibility_count = 0u;
  std::uint32_t learned_segment_count = 0u;
  std::uint32_t reclaimed_segment_count = 0u;
  std::uint32_t journal_entries = 0u;
  std::uint32_t errors = 0u;
  std::uint32_t output_winner_count = 0u;
  std::uint16_t output_winner_columns[kReceiptWinners]{};
  std::uint64_t prediction_hash = 0u;
  std::uint64_t output_hash = 0u;
  std::uint64_t state_hash_before = 0u;
  std::uint64_t state_hash_after = 0u;
  std::uint64_t represented_matter = 0u;
  std::uint64_t free_matter = 0u;
  std::uint32_t learned_separator_count = 0u;
  std::uint32_t form_assembly_count = 0u;
  std::uint32_t contact_assembly_count = 0u;
  std::uint16_t recalled_contact_cell = 0u;
  std::uint32_t recalled_contact_count = 0u;
  std::uint32_t recalled_form_count = 0u;
  std::uint32_t replay_start_count = 0u;
  std::uint64_t external_growth_calls = 0u;
  std::uint64_t internal_growth_calls = 0u;
};
struct MotorTrajectory {
  std::uint8_t bytes[kMotorTrajectoryCapacity]{};
  std::uint32_t length = 0u;
  std::uint8_t complete = 0u;
  std::uint8_t overflow = 0u;
};
struct DeviceState {
  Population populations[kPopulationCount]{};
  HistoryEntry history[kHistoryDepth]{};
  JournalEntry journal[kJournalCapacity]{};
  std::uint32_t journal_count = 0u;
  std::uint32_t transaction_mark = 0u;
  std::uint32_t tick = 0u;
  std::uint32_t predicted_mask = 0u;
  std::uint16_t predicted_value = 0u;
  std::uint8_t prediction_active = 0u;
  std::uint8_t predicted_population = 0u;
  std::uint16_t predicted_segment = 0u;
  std::uint16_t predicted_target = 0u;
  Publication publications[kPublicationCapacity]{};
  std::uint16_t publication_count = 0u;
  std::uint16_t frozen_edge_class_mask = 0u;
  ContactState contact{};
  std::uint64_t internal_growth_calls = 0u;
  std::uint64_t external_growth_calls = 0u;
  std::uint64_t canonical_snapshot_hash = 0u;
  std::uint32_t canonical_snapshot_tick = 0u;
  std::uint8_t canonical_snapshot_valid = 0u;
  std::uint8_t metadata_snapshot_active = 0u;
  ContactState transaction_contact_before{};
  Publication transaction_publications_before[kPublicationCapacity]{};
  std::uint32_t error_bits = 0u;
  std::uint64_t initial_matter = 0u;
  std::uint64_t represented_matter = 0u;
  std::uint64_t free_matter = 0u;
  std::uint8_t initialized = 0u;
};

static_assert(std::is_integral_v<decltype(Cell::activation)>);
static_assert(std::is_integral_v<decltype(Segment::target_cell)>);
static_assert(std::is_integral_v<decltype(Synapse::weight)>);
static_assert(std::is_integral_v<decltype(Eligibility::trace)>);
static_assert(std::is_integral_v<decltype(Event::value)>);

BCC32_PA_HD constexpr bool edge_allowed(std::uint32_t source, std::uint32_t target) {
  return (source == kP0 && target == kP0) ||
         (source == kP0 && target == kP1) ||
         (source == kP0 && target == kP3) ||
         (source == kP1 && target == kP0) ||
         (source == kP1 && target == kP1) ||
         (source == kP1 && target == kP2) ||
         (source == kP1 && target == kP3) ||
         (source == kP2 && target == kP1) ||
         (source == kP2 && target == kP2) ||
         (source == kP2 && target == kP3) ||
         (source == kP3 && target == kP3) ||
         (source == kP3 && target == kP4) ||
         (source == kP4 && target == kP1) ||
         (source == kP1 && target == kP4) ||
         (source == kP4 && target == kP5) ||
         (source == kP3 && target == kP5) ||
         (source == kP0 && target == kP5);
}

BCC32_PA_HD constexpr std::uint32_t edge_max_synapses(std::uint32_t source, std::uint32_t target) {
  return ((source == kP0 && target == kP0) ||
          (source == kP1 && target == kP0) ||
          (source == kP1 && target == kP1) ||
          (source == kP2 && target == kP1) ||
          (source == kP2 && target == kP2) ||
          (source == kP4 && target == kP1))
             ? 2u
             : 8u;
}

BCC32_PA_HD constexpr bool edge_delay_allowed(std::uint32_t source, std::uint32_t target,
                                               std::uint32_t delay) {
  if (source == kP0 && target == kP1)
    return delay < kContactMemberCapacity;
  if (source == kP1 && target == kP2)
    return delay < kContactMemberCapacity;
  if (source == kP1 && target == kP0)
    return delay < kContactMemberCapacity;
  if ((source == kP0 && target == kP0) ||
      (source == kP1 && target == kP1) ||
      (source == kP2 && target == kP1) ||
      (source == kP2 && target == kP2) ||
      (source == kP4 && target == kP1)) {
    return delay == 1u || delay == 2u;
  }
  if (source == kP3 && target == kP3) {
    return delay > 0u && delay <= kHistoryDepth;
  }
  return delay == 0u;
}

static_assert(edge_allowed(kP0, kP0) && edge_allowed(kP0, kP1));
static_assert(edge_allowed(kP0, kP3) && edge_allowed(kP1, kP1));
static_assert(edge_allowed(kP1, kP0) && edge_allowed(kP1, kP2));
static_assert(edge_allowed(kP2, kP1) && edge_allowed(kP2, kP2));
static_assert(edge_allowed(kP4, kP1) && edge_allowed(kP1, kP4));
static_assert(edge_allowed(kP1, kP3) && edge_allowed(kP3, kP3));
static_assert(edge_allowed(kP2, kP3));
static_assert(edge_allowed(kP3, kP4) && edge_allowed(kP3, kP5));
static_assert(edge_allowed(kP4, kP5) && edge_allowed(kP0, kP5));
static_assert(!edge_allowed(kP2, kP4) && !edge_allowed(kP2, kP5));
static_assert(edge_max_synapses(kP0, kP0) == 2u);
static_assert(edge_max_synapses(kP2, kP2) == 2u);
static_assert(edge_delay_allowed(kP0, kP0, 1u) && !edge_delay_allowed(kP0, kP0, 0u));
static_assert(edge_delay_allowed(kP1, kP0, 0u) &&
              edge_delay_allowed(kP1, kP0, kContactMemberCapacity - 1u) &&
              edge_delay_allowed(kP2, kP1, 1u));
static_assert(edge_delay_allowed(kP2, kP2, 2u) && edge_delay_allowed(kP4, kP1, 1u));
static_assert(edge_delay_allowed(kP1, kP1, 2u) &&
              edge_delay_allowed(kP3, kP3, kHistoryDepth));
static_assert(edge_delay_allowed(kP1, kP2, 0u));
static_assert(sizeof(DeviceState) < 16u * 1024u * 1024u);
BCC32_PA_HD inline std::uint32_t mix(std::uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}
BCC32_PA_HD inline std::uint16_t clamp_cell(std::uint32_t cell) {
  return static_cast<std::uint16_t>(cell % kCellsPerPopulation);
}
BCC32_PA_HD constexpr std::uint8_t route_class(std::uint32_t source,
                                               std::uint32_t target) {
  return static_cast<std::uint8_t>((source * kPopulationCount + target) & 15u);
}
BCC32_PA_HD inline std::int32_t abs_i32(std::int32_t value) {
  return value < 0 ? -value : value;
}
#include "bcc32_grown_predictive_assembly_transaction_journal.inl"
BCC32_PA_DEVICE inline bool edge_class_frozen(const DeviceState& state,
                                              std::uint32_t source_population,
                                              std::uint32_t target_population) {
  return (state.frozen_edge_class_mask &
          (1u << route_class(source_population, target_population))) != 0u;
}
BCC32_PA_DEVICE inline bool segment_has_live_eligibility(const DeviceState& state,
                                                         std::uint32_t target_population,
                                                         std::uint32_t segment_index) {
  if (target_population >= kPopulationCount || segment_index >= kSegmentsPerPopulation)
    return true;
  const Population& pool = state.populations[target_population];
  for (std::uint32_t i = 0u; i < kEligibilityPerPopulation; ++i) {
    const Eligibility& eligibility = pool.eligibilities[i];
    if (eligibility.live && eligibility.segment_index == segment_index &&
        eligibility.target_population == target_population)
      return true;
  }
  return false;
}
BCC32_PA_DEVICE inline bool segment_is_locked(const DeviceState& state,
                                              std::uint32_t target_population,
                                              const Segment& segment) {
  return segment.frozen != 0u ||
         edge_class_frozen(state, segment.source_population, target_population) ||
         segment_has_live_eligibility(state, target_population,
                                      static_cast<std::uint32_t>(&segment -
                                      state.populations[target_population].segments));
}
BCC32_PA_DEVICE inline bool validate_segment(const DeviceState& state,
                                             std::uint32_t target_population,
                                             std::uint32_t segment_index) {
  if (target_population >= kPopulationCount || segment_index >= kSegmentsPerPopulation)
    return false;
  const Population& target = state.populations[target_population];
  const Segment& segment = target.segments[segment_index];
  if (!segment.live || segment.source_population >= kPopulationCount ||
      segment.target_cell >= kCellsPerPopulation ||
      segment.synapse_count > kMaxSegmentSynapses ||
      segment.synapse_begin + segment.synapse_count > kSynapsesPerPopulation ||
      !edge_allowed(segment.source_population, target_population) ||
      segment.synapse_count > edge_max_synapses(segment.source_population, target_population))
    return false;
  const Population& source = state.populations[segment.source_population];
  for (std::uint32_t i = 0u; i < segment.synapse_count; ++i) {
    const Synapse& synapse = source.synapses[segment.synapse_begin + i];
    if (!synapse.live || !edge_delay_allowed(segment.source_population, target_population,
                                              synapse.delay))
      return false;
  }
  return true;
}
BCC32_PA_DEVICE inline bool allocate_synapse_range(DeviceState* state, std::uint32_t population,
                                                   std::uint32_t count,
                                                   std::uint16_t* begin) {
  if (population >= kPopulationCount || count == 0u || count > kMaxSegmentSynapses) return false;
  Population& pool = state->populations[population];
  if (pool.free_synapse_count < count) {
    set_error(state, kErrorAllocatorExhausted);
    return false;
  }
  for (std::uint32_t start = 0u; start + count <= kSynapsesPerPopulation; ++start) {
    bool free_range = true;
    for (std::uint32_t i = 0u; i < count; ++i)
      free_range = free_range && !pool.synapses[start + i].live;
    if (!free_range) continue;
    for (std::uint32_t i = 0u; i < count; ++i) {
      if (!journal_synapse(state, population, start + i)) return false;
    }
    for (std::uint32_t i = 0u; i < count; ++i) pool.synapses[start + i].live = 1u;
    *begin = static_cast<std::uint16_t>(start);
    return true;
  }
  set_error(state, kErrorAllocatorExhausted);
  return false;
}
BCC32_PA_DEVICE inline bool allocate_segment(DeviceState* state, std::uint32_t target_population,
                                             std::uint32_t source_population,
                                             std::uint32_t target_cell, std::uint32_t count,
                                             std::uint32_t threshold,
                                             std::uint16_t* segment_index) {
  if (target_population >= kPopulationCount || source_population >= kPopulationCount ||
      !edge_allowed(source_population, target_population) ||
      count == 0u || count > edge_max_synapses(source_population, target_population) ||
      target_cell >= kCellsPerPopulation) {
    set_error(state, kErrorForbiddenEdge);
    return false;
  }
  if (!journal_metadata(state)) return false;
  Population& target = state->populations[target_population];
  if (target.free_segment_count == 0u) {
    std::uint32_t victim = kSegmentsPerPopulation;
    std::int32_t weakest = 0x7fffffff;
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
      const Segment& trial = target.segments[i];
      if (trial.live && !segment_is_locked(*state, target_population, trial) &&
          (trial.utility < weakest ||
           (trial.utility == weakest && i < victim))) {
        weakest = trial.utility;
        victim = i;
      }
    }
    if (victim == kSegmentsPerPopulation) {
      set_error(state, kErrorAllocatorExhausted);
      return false;
    }
    const std::uint32_t old_source_population = target.segments[victim].source_population;
    const std::uint32_t old_synapse_count = target.segments[victim].synapse_count;
    if (!journal_allocator(state, target_population) ||
        (old_source_population != target_population &&
         !journal_allocator(state, old_source_population)) ||
        !journal_segment(state, target_population, victim))
      return false;
    Segment& old = target.segments[victim];
    Population& old_source = state->populations[old_source_population];
    for (std::uint32_t i = 0u; i < old_synapse_count; ++i) {
      if (!journal_synapse(state, old_source_population, old.synapse_begin + i)) return false;
      old_source.synapses[old.synapse_begin + i].live = 0u;
    }
    old = Segment{};
    rebuild_allocator(&target);
    if (old_source_population != target_population) rebuild_allocator(&old_source);
    if (old_source_population == target_population) {
      target.free_matter += old_synapse_count + 1u;
      target.represented_matter -= old_synapse_count + 1u;
    } else {
      ++target.free_matter;
      old_source.free_matter += old_synapse_count;
      --target.represented_matter;
      old_source.represented_matter -= old_synapse_count;
    }
    state->free_matter += static_cast<std::uint64_t>(old_synapse_count + 1u);
    state->represented_matter -= static_cast<std::uint64_t>(old_synapse_count + 1u);
  }
  if (!journal_allocator(state, target_population)) return false;
  if (source_population != target_population && !journal_allocator(state, source_population)) return false;
  std::uint16_t begin = 0u;
  if (!allocate_synapse_range(state, source_population, count, &begin)) return false;
  std::uint32_t index = target.free_segments[0u];
  if (!journal_segment(state, target_population, index)) return false;
  target.segments[index] = Segment{};
  target.segments[index].target_cell = static_cast<std::uint16_t>(target_cell);
  target.segments[index].source_population = static_cast<std::uint8_t>(source_population);
  target.segments[index].synapse_begin = begin;
  target.segments[index].synapse_count = static_cast<std::uint8_t>(count);
    target.segments[index].threshold = static_cast<std::uint16_t>(threshold);
    target.segments[index].live = 1u;
    target.segments[index].route_class = route_class(source_population, target_population);
    target.segments[index].target_generation =
        target.cells[target_cell].generation;
    target.segments[index].generation = state->tick;
  target.free_segment_count = target.free_segment_count > 0u ? target.free_segment_count - 1u : 0u;
  Population& source = state->populations[source_population];
  source.free_synapse_count -= count;
  target.represented_matter += 1u;
  source.represented_matter += count;
  target.free_matter -= 1u;
  source.free_matter -= count;
  state->represented_matter += static_cast<std::uint64_t>(count + 1u);
  state->free_matter -= static_cast<std::uint64_t>(count + 1u);
  rebuild_allocator(&target);
  if (source_population != target_population) rebuild_allocator(&source);
  *segment_index = static_cast<std::uint16_t>(index);
  return true;
}
BCC32_PA_DEVICE inline bool configure_synapse(DeviceState* state, std::uint32_t target_population,
                                              std::uint32_t segment_index, std::uint32_t offset,
                                              std::uint32_t source_cell, std::uint32_t delay,
                                              std::int32_t weight) {
  if (target_population >= kPopulationCount || segment_index >= kSegmentsPerPopulation) return false;
  Segment& segment = state->populations[target_population].segments[segment_index];
  if (!segment.live || offset >= segment.synapse_count || segment.source_population >= kPopulationCount ||
      source_cell >= kCellsPerPopulation || !edge_delay_allowed(segment.source_population,
                                                                target_population, delay)) {
    set_error(state, kErrorInvalidDelay);
    return false;
  }
  if (segment_is_locked(*state, target_population, segment)) {
    set_error(state, kErrorInvariant);
    return false;
  }
  Population& source = state->populations[segment.source_population];
  const std::uint32_t index = segment.synapse_begin + offset;
  if (!journal_synapse(state, segment.source_population, index)) return false;
  source.synapses[index].source_cell = static_cast<std::uint16_t>(source_cell);
  source.synapses[index].delay = static_cast<std::uint8_t>(delay);
  source.synapses[index].weight = static_cast<std::int8_t>(weight < -127 ? -127 : weight > 127 ? 127 : weight);
  source.synapses[index].utility = static_cast<std::int16_t>(weight);
  source.synapses[index].live = 1u;
  source.synapses[index].generation = state->tick;
  return true;
}
BCC32_PA_DEVICE inline bool grow_segment_contract(DeviceState* state, const GrowthContract& contract,
                                                  std::uint16_t* segment_index) {
  if (state != nullptr) ++state->external_growth_calls;
  if (contract.synapse_count == 0u || contract.synapse_count > kMaxSegmentSynapses ||
      contract.synapse_count > edge_max_synapses(contract.source_population,
                                                  contract.target_population)) {
    set_error(state, kErrorForbiddenEdge);
    return false;
  }
  if (!allocate_segment(state, contract.target_population, contract.source_population,
                        contract.target_cell, contract.synapse_count, contract.threshold,
                        segment_index)) return false;
  for (std::uint32_t i = 0u; i < contract.synapse_count; ++i) {
    if (!configure_synapse(state, contract.target_population, *segment_index, i,
                           contract.source_cells[i], contract.delays[i], contract.weights[i]))
      return false;
  }
  return true;
}
BCC32_PA_DEVICE inline bool grow_segment_internal(
    DeviceState* state, std::uint32_t target_population, std::uint32_t source_population,
    std::uint32_t target_cell, const std::uint16_t* source_cells,
    const std::uint8_t* delays, const std::int8_t* weights, std::uint32_t count,
    std::uint32_t threshold, std::uint16_t* segment_index) {
  if (state == nullptr || source_cells == nullptr || delays == nullptr ||
      weights == nullptr || count == 0u || count > kMaxSegmentSynapses) {
    if (state != nullptr) set_error(state, kErrorInvariant);
    return false;
  }
  if (!allocate_segment(state, target_population, source_population, target_cell,
                        count, threshold, segment_index))
    return false;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (!configure_synapse(state, target_population, *segment_index, i,
                           source_cells[i], delays[i], weights[i]))
      return false;
  }
  ++state->internal_growth_calls;
  return true;
}
BCC32_PA_DEVICE inline std::int32_t active_segment_support(const DeviceState& state,
                                                           std::uint32_t target_population,
                                                           std::uint32_t segment_index,
                                                           const std::uint8_t* active,
                                                           std::uint32_t history_tick) {
  if (!validate_segment(state, target_population, segment_index)) return 0;
  const Segment& segment = state.populations[target_population].segments[segment_index];
  const Population& source = state.populations[segment.source_population];
  std::int32_t support = 0;
  for (std::uint32_t i = 0u; i < segment.synapse_count; ++i) {
    const Synapse& synapse = source.synapses[segment.synapse_begin + i];
    const std::uint32_t tick = history_tick >= synapse.delay ? history_tick - synapse.delay : 0u;
    const std::uint32_t slot = tick % kHistoryDepth;
    bool on = synapse.delay == 0u && active != nullptr &&
              active[synapse.source_cell] != 0u;
    const HistoryEntry& entry = state.history[slot];
    if (synapse.delay != 0u && entry.tick == tick) {
      for (std::uint32_t c = 0u;
           c < entry.active_count[segment.source_population]; ++c) {
        on = on ||
             entry.active_cells[segment.source_population][c] ==
                 synapse.source_cell;
      }
    }
    if (on) support += synapse.weight;
  }
  return support;
}

// Generic distinct-source coincidence.  Each required source population gets
// its own strongest active segment; segments never mix source populations.
BCC32_PA_DEVICE inline std::int32_t settle_target(const DeviceState& state,
                                                  std::uint32_t target_population,
                                                  std::uint32_t target_cell,
                                                  std::uint32_t required_source_mask,
                                                  std::uint32_t history_tick,
                                                  std::int32_t inhibition,
                                                  const std::uint8_t* active_by_population = nullptr) {
  if (target_population >= kPopulationCount || target_cell >= kCellsPerPopulation) return 0;
  std::int32_t minimum = 0x7fffffff;
  bool found = false;
  for (std::uint32_t source_population = 0u; source_population < kPopulationCount;
       ++source_population) {
    if ((required_source_mask & (1u << source_population)) == 0u) continue;
    std::int32_t strongest = 0;
    bool source_found = false;
    const Population& target = state.populations[target_population];
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
      const Segment& segment = target.segments[i];
      if (!segment.live || segment.target_cell != target_cell ||
          segment.source_population != source_population) continue;
      const std::uint8_t* active = active_by_population == nullptr
                                       ? nullptr
                                       : active_by_population + source_population * kCellsPerPopulation;
      const std::int32_t support = active_segment_support(state, target_population, i, active,
                                                           history_tick);
      if (support >= static_cast<std::int32_t>(segment.threshold)) {
        if (!source_found || support > strongest) strongest = support;
        source_found = true;
      }
    }
    if (!source_found) return 0;
    if (strongest < minimum) minimum = strongest;
    found = true;
  }
  return found ? minimum - inhibition : 0;
}

BCC32_PA_DEVICE inline bool settle_population_winner(
    const DeviceState& state, std::uint32_t target_population,
    std::uint32_t required_source_mask, std::uint32_t history_tick,
    std::int32_t inhibition, const std::uint8_t* active_by_population,
    std::uint16_t* winner_cell, std::int32_t* winner_support) {
  if (target_population >= kPopulationCount || winner_cell == nullptr ||
      winner_support == nullptr)
    return false;
  bool found = false;
  std::int32_t best = 0;
  std::uint16_t best_cell = 0u;
  for (std::uint32_t cell = 0u; cell < kCellsPerPopulation; ++cell) {
    const std::int32_t support =
        settle_target(state, target_population, cell, required_source_mask,
                      history_tick, inhibition, active_by_population);
    if (support > 0 &&
        (!found || support > best ||
         (support == best && cell < static_cast<std::uint32_t>(best_cell)))) {
      found = true;
      best = support;
      best_cell = static_cast<std::uint16_t>(cell);
    }
  }
  *winner_cell = best_cell;
  *winner_support = best;
  return found;
}

// Proximal identity is a physical raw-byte column. Temporal context is selected
// among equivalent siblings by learned distal support, never by hashing the
// clock or the complete predecessor history into an address.
BCC32_PA_HD inline std::uint32_t event_column(const Event& event) {
  return event.contact_boundary != 0u ? 256u
                                      : static_cast<std::uint32_t>(event.value);
}

BCC32_PA_HD inline std::uint32_t context_cell_column(std::uint32_t cell) {
  return cell / kContextCellsPerColumn;
}

BCC32_PA_DEVICE inline bool append_motor_winner(MotorTrajectory* trajectory,
                                                std::uint16_t winner_cell) {
  if (trajectory == nullptr || winner_cell >= kCellsPerPopulation ||
      trajectory->complete != 0u)
    return false;
  const std::uint32_t column = context_cell_column(winner_cell);
  if (column == kRawColumnCount - 1u) {
    trajectory->complete = 1u;
    return true;
  }
  if (trajectory->length >= kMotorTrajectoryCapacity) {
    trajectory->overflow = 1u;
    return false;
  }
  trajectory->bytes[trajectory->length++] = static_cast<std::uint8_t>(column);
  return true;
}

BCC32_PA_DEVICE inline std::uint16_t select_context_cell(
    const DeviceState& state, const Event& event) {
  const std::uint32_t column = event_column(event);
  const std::uint32_t base = column * kContextCellsPerColumn;
  std::uint32_t winner = base;
  for (std::uint32_t sibling = 1u; sibling < kContextCellsPerColumn;
       ++sibling) {
    const std::uint32_t trial = base + sibling;
    const Cell& left = state.populations[kP0].cells[trial];
    const Cell& right = state.populations[kP0].cells[winner];
    if (left.utility > right.utility ||
        (left.utility == right.utility && left.generation > right.generation)) {
      winner = trial;
    }
  }
  return static_cast<std::uint16_t>(winner);
}
BCC32_PA_DEVICE inline std::uint16_t choose_growth_cell(
    const DeviceState& state, std::uint32_t population) {
  std::uint32_t chosen = 0u;
  std::int32_t utility = 0x7fffffff;
  std::uint32_t generation = 0xffffffffu;
  if (population >= kPopulationCount) return 0u;
  for (std::uint32_t cell = 0u; cell < kCellsPerPopulation; ++cell) {
    const Cell& value = state.populations[population].cells[cell];
    if (!value.live || value.frozen != 0u) continue;
    if (value.utility < utility ||
        (value.utility == utility && value.generation < generation)) {
      chosen = cell;
      utility = value.utility;
      generation = value.generation;
    }
  }
  return static_cast<std::uint16_t>(chosen);
}
BCC32_PA_DEVICE inline bool grow_one_internal(
    DeviceState* state, std::uint32_t target_population, std::uint32_t source_population,
    std::uint32_t target_cell, std::uint32_t source_cell, std::uint32_t delay,
    std::int32_t weight, std::uint32_t threshold = 1u) {
  std::uint16_t source_cells[1] = {static_cast<std::uint16_t>(source_cell)};
  std::uint8_t delays[1] = {static_cast<std::uint8_t>(delay)};
  std::int8_t weights[1] = {static_cast<std::int8_t>(weight)};
  std::uint16_t segment_index = 0u;
  return grow_segment_internal(state, target_population, source_population, target_cell,
                               source_cells, delays, weights, 1u, threshold,
                               &segment_index);
}
BCC32_PA_DEVICE inline bool touch_growth_cell(DeviceState* state,
                                              std::uint32_t population,
                                              std::uint32_t cell) {
  if (population >= kPopulationCount || cell >= kCellsPerPopulation) return false;
  if (!journal_cell(state, population, cell)) return false;
  Cell& value = state->populations[population].cells[cell];
  if (value.utility < 0x7fffffff) ++value.utility;
  return true;
}
BCC32_PA_DEVICE inline bool has_one_synapse(const DeviceState& state,
                                            std::uint32_t target_population,
                                            std::uint32_t source_population,
                                            std::uint32_t target_cell,
                                            std::uint32_t source_cell,
                                            std::uint32_t required_delay = 0xffffffffu) {
  if (target_population >= kPopulationCount || source_population >= kPopulationCount)
    return false;
  const Population& target = state.populations[target_population];
  const Population& source = state.populations[source_population];
  for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
    const Segment& segment = target.segments[i];
    if (!segment.live || segment.source_population != source_population ||
        segment.target_cell != target_cell || segment.synapse_count != 1u)
      continue;
    const Synapse& synapse = source.synapses[segment.synapse_begin];
    if (synapse.live && synapse.source_cell == source_cell &&
        (required_delay == 0xffffffffu || synapse.delay == required_delay))
      return true;
  }
  return false;
}
BCC32_PA_DEVICE inline bool settle_form_cell(
    const DeviceState& state, const std::uint16_t* p0_cells,
    std::uint32_t count, std::uint16_t* form_cell) {
  if (p0_cells == nullptr || form_cell == nullptr || count == 0u ||
      count > kContactMemberCapacity)
    return false;
  const Population& forms = state.populations[kP1];
  const Population& surface = state.populations[kP0];
  for (std::uint32_t segment_index = 0u;
       segment_index < kSegmentsPerPopulation; ++segment_index) {
    const Segment& segment = forms.segments[segment_index];
    if (!segment.live || segment.source_population != kP0 ||
        segment.synapse_count != count || segment.threshold != count)
      continue;
    bool equal = true;
    for (std::uint32_t position = 0u; position < count; ++position) {
      const Synapse& synapse = surface.synapses[segment.synapse_begin + position];
      const std::uint32_t expected_delay = count - position - 1u;
      if (!synapse.live || synapse.source_cell != p0_cells[position] ||
          synapse.delay != expected_delay || synapse.weight != 1) {
        equal = false;
        break;
      }
    }
    if (!equal) continue;
    *form_cell = segment.target_cell;
    return true;
  }
  return false;
}
BCC32_PA_DEVICE inline bool remember_contact_cell(DeviceState* state,
                                                  std::uint16_t cell) {
  ContactState& contact = state->contact;
  if (contact.pending_p0_count >= kContactMemberCapacity) return true;
  contact.pending_p0[contact.pending_p0_count++] = cell;
  return true;
}
BCC32_PA_DEVICE inline std::uint32_t successor_branching(
    const ContactState& contact, std::uint32_t value) {
  if (value >= kSeparatorCount) return 0u;
  std::uint32_t branching = 0u;
  for (std::uint32_t word = 0u; word < 8u; ++word) {
    std::uint32_t bits = contact.successor_bits[value][word];
    while (bits != 0u) {
      bits &= bits - 1u;
      ++branching;
    }
  }
  return branching;
}
BCC32_PA_DEVICE inline std::uint32_t predecessor_branching(
    const ContactState& contact, std::uint32_t value) {
  if (value >= kSeparatorCount) return 0u;
  std::uint32_t branching = 0u;
  for (std::uint32_t word = 0u; word < 8u; ++word) {
    std::uint32_t bits = contact.predecessor_bits[value][word];
    while (bits != 0u) {
      bits &= bits - 1u;
      ++branching;
    }
  }
  return branching;
}
BCC32_PA_DEVICE inline std::uint64_t separator_score(
    const ContactState& contact, std::uint32_t value) {
  const std::uint32_t successors = successor_branching(contact, value);
  const std::uint32_t predecessors = predecessor_branching(contact, value);
  const std::uint32_t interior = contact.interior_evidence[value];
  if (successors <= 2u || predecessors <= 2u || interior <= 2u) return 0u;
  return static_cast<std::uint64_t>(successors) * predecessors;
}
BCC32_PA_DEVICE inline bool learned_separator(const ContactState& contact,
                                              std::uint32_t value) {
  if (value >= kSeparatorCount || separator_score(contact, value) == 0u)
    return false;
  std::uint64_t best_score = 0u;
  std::uint32_t best_value = kSeparatorCount;
  std::uint32_t best_count = 0u;
  for (std::uint32_t candidate = 0u; candidate < kSeparatorCount;
       ++candidate) {
    const std::uint64_t score = separator_score(contact, candidate);
    if (score > best_score) {
      best_score = score;
      best_value = candidate;
      best_count = 1u;
    } else if (score != 0u && score == best_score) {
      ++best_count;
    }
  }
  return best_count == 1u && best_value == value;
}
BCC32_PA_DEVICE inline bool remember_successor(ContactState* contact,
                                                std::uint32_t predecessor,
                                                std::uint32_t successor) {
  if (contact == nullptr || predecessor >= kSeparatorCount ||
      successor >= kSeparatorCount)
    return false;
  contact->successor_bits[predecessor][successor >> 5u] |=
      1u << (successor & 31u);
  contact->predecessor_bits[successor][predecessor >> 5u] |=
      1u << (predecessor & 31u);
  return true;
}
BCC32_PA_DEVICE inline bool materialize_form_from_p0(
    DeviceState* state, const std::uint16_t* p0_cells, std::uint32_t p0_count,
    std::uint16_t* out_form_cell, DeviceReceipt* receipt) {
  if (state == nullptr || p0_cells == nullptr || out_form_cell == nullptr ||
      p0_count == 0u || p0_count > kContactMemberCapacity) {
    if (state != nullptr) set_error(state, kErrorInvariant);
    return false;
  }
  std::uint16_t resolved_form_cell = 0u;
  const bool existing =
      settle_form_cell(*state, p0_cells, p0_count, &resolved_form_cell);
  if (!existing) resolved_form_cell = choose_growth_cell(*state, kP1);
  if (!touch_growth_cell(state, kP1, resolved_form_cell)) return false;
  if (!existing) {
    std::uint8_t delays[kContactMemberCapacity]{};
    std::int8_t weights[kContactMemberCapacity]{};
    for (std::uint32_t position = 0u; position < p0_count; ++position) {
      delays[position] = static_cast<std::uint8_t>(
          p0_count - position - 1u);
      weights[position] = 1;
    }
    std::uint16_t segment_index = 0u;
    if (!grow_segment_internal(state, kP1, kP0, resolved_form_cell,
                               p0_cells, delays, weights, p0_count, p0_count,
                               &segment_index))
      return false;
  }
  for (std::uint32_t i = 0u; i < p0_count; ++i) {
    if (!has_one_synapse(*state, kP0, kP1, p0_cells[i], resolved_form_cell, i) &&
        !grow_one_internal(state, kP0, kP1, p0_cells[i], resolved_form_cell, i,
                           1))
      return false;
  }
  if (!has_one_synapse(*state, kP1, kP1, resolved_form_cell, resolved_form_cell) &&
      !grow_one_internal(state, kP1, kP1, resolved_form_cell, resolved_form_cell,
                         1u, 1))
    return false;
  if (receipt != nullptr) ++receipt->form_assembly_count;
  *out_form_cell = resolved_form_cell;
  return true;
}
BCC32_PA_DEVICE inline bool close_pending_form(DeviceState* state,
                                               DeviceReceipt* receipt) {
  ContactState& contact = state->contact;
  if (contact.pending_p0_count == 0u) return true;
  if (contact.pending_p1_count >= kContactMemberCapacity) {
    set_error(state, kErrorInvariant);
    return false;
  }
  std::uint16_t form_cell = 0u;
  if (!materialize_form_from_p0(state, contact.pending_p0,
                                contact.pending_p0_count, &form_cell,
                                receipt))
    return false;
  contact.pending_p1[contact.pending_p1_count++] = form_cell;
  contact.event_closed_content_p1_cell = form_cell;
  contact.event_closed_content_p1_valid = 1u;
  contact.pending_p0_count = 0u;
  return true;
}
BCC32_PA_DEVICE inline bool recall_contact_forms(DeviceState* state,
                                                 std::uint16_t contact_cell,
                                                 DeviceReceipt* receipt) {
  ContactState& contact = state->contact;
  contact.recalled_form_count = 0u;
  contact.replay_start_count = 0u;
  for (std::uint32_t position = 0u;
       position < kContactMemberCapacity &&
       contact.recalled_form_count < kContactMemberCapacity; ++position) {
    std::uint16_t form = 0u;
    bool present = false;
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
      const Segment& segment = state->populations[kP2].segments[i];
      if (!segment.live || segment.source_population != kP1 ||
          segment.target_cell != contact_cell || segment.synapse_count != 1u)
        continue;
      const Synapse& synapse = state->populations[kP1].synapses[segment.synapse_begin];
      if (synapse.live && synapse.delay == position) {
        form = synapse.source_cell;
        present = true;
        break;
      }
    }
    if (!present) continue;
    contact.recalled_p1[contact.recalled_form_count++] = form;
    std::int32_t best_strength = -0x7fffffff;
    std::uint16_t best_p0 = 0u;
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
      const Segment& segment = state->populations[kP0].segments[i];
      if (!segment.live || segment.source_population != kP1 ||
          segment.synapse_count != 1u) continue;
      const Synapse& synapse = state->populations[kP1].synapses[segment.synapse_begin];
      if (!synapse.live || synapse.source_cell != form) continue;
      if (segment.strength > best_strength ||
          (segment.strength == best_strength && segment.target_cell < best_p0)) {
        best_strength = segment.strength;
        best_p0 = segment.target_cell;
      }
    }
    if (best_strength > -0x7fffffff) {
      contact.pending_p0[contact.replay_start_count < kContactMemberCapacity
                             ? contact.replay_start_count
                             : kContactMemberCapacity - 1u] = best_p0;
      if (contact.replay_start_count < kContactMemberCapacity)
        ++contact.replay_start_count;
    }
  }
  if (receipt != nullptr) {
    receipt->recalled_form_count = contact.recalled_form_count;
    receipt->replay_start_count = contact.replay_start_count;
  }
  return true;
}
BCC32_PA_DEVICE inline bool recall_contact_state(DeviceState* state,
                                                 DeviceReceipt* receipt) {
  const ContactState& contact = state->contact;
  if (contact.pending_p1_count == 0u) return true;
  std::uint16_t top_cells[kReceiptWinners]{};
  std::uint16_t top_scores[kReceiptWinners]{};
  std::uint32_t top_count = 0u;
  const std::uint32_t threshold = contact.pending_p1_count > 1u ? 2u : 1u;
  for (std::uint32_t cell = 0u; cell < kCellsPerPopulation; ++cell) {
    std::uint32_t score = 0u;
    for (std::uint32_t position = 0u;
         position < contact.pending_p1_count; ++position) {
      for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
        const Segment& segment = state->populations[kP2].segments[i];
        if (!segment.live || segment.source_population != kP1 ||
            segment.target_cell != cell || segment.synapse_count != 1u)
          continue;
        const Synapse& synapse = state->populations[kP1].synapses[segment.synapse_begin];
        if (synapse.live && synapse.source_cell == contact.pending_p1[position] &&
            synapse.delay == position) {
          ++score;
          break;
        }
      }
    }
    if (score < threshold) continue;
    std::uint32_t insert = top_count;
    if (insert < kReceiptWinners) ++top_count;
    else if (score <= top_scores[kReceiptWinners - 1u]) continue;
    while (insert > 0u &&
           (top_scores[insert - 1u] < score ||
            (top_scores[insert - 1u] == score &&
             top_cells[insert - 1u] > cell))) {
      if (insert < kReceiptWinners) {
        top_scores[insert] = top_scores[insert - 1u];
        top_cells[insert] = top_cells[insert - 1u];
      }
      --insert;
    }
    if (insert < kReceiptWinners) {
      top_scores[insert] = static_cast<std::uint16_t>(score);
      top_cells[insert] = static_cast<std::uint16_t>(cell);
    }
  }
  if (top_count == 0u) return true;
  if (!journal_metadata(state)) return false;
  state->contact.has_recalled = 1u;
  state->contact.recalled_contact_cell = top_cells[0u];
  state->contact.recalled_contact_count = 0u;
  for (std::uint32_t index = 0u; index < top_count; ++index) {
    state->contact.recalled_contact_cells[state->contact.recalled_contact_count++] =
        top_cells[index];
  }
  if (!recall_contact_forms(state, top_cells[0u], receipt)) return false;
  if (receipt != nullptr) {
    receipt->recalled_contact_cell = top_cells[0u];
    receipt->recalled_contact_count = state->contact.recalled_contact_count;
  }
  return true;
}
BCC32_PA_DEVICE inline bool close_contact_state(DeviceState* state,
                                                DeviceReceipt* receipt) {
  ContactState& contact = state->contact;
  if (!close_pending_form(state, receipt) ||
      !recall_contact_state(state, receipt))
    return false;
  const std::uint16_t contact_cell = choose_growth_cell(*state, kP2);
  if (!touch_growth_cell(state, kP2, contact_cell)) return false;
  for (std::uint32_t i = 0u; i < contact.pending_p1_count; ++i) {
    if (!grow_one_internal(state, kP2, kP1, contact_cell,
                           contact.pending_p1[i], i, 1) ||
        !grow_one_internal(state, kP1, kP2, contact.pending_p1[i],
                           contact_cell, 1u, 1))
      return false;
  }
  if (!grow_one_internal(state, kP2, kP2, contact_cell, contact_cell, 1u, 1))
    return false;
  contact.current_contact_cell = contact_cell;
  contact.contact_count += 1u;
  contact.active = 0u;
  contact.separator_count = 0u;
  for (std::uint32_t value = 0u; value < kSeparatorCount; ++value) {
    if (learned_separator(contact, value))
      ++contact.separator_count;
  }
  if (receipt != nullptr) {
    receipt->contact_assembly_count += 1u;
    receipt->learned_separator_count = contact.separator_count;
    receipt->recalled_contact_cell = contact.recalled_contact_cell;
  }
  return true;
}
BCC32_PA_DEVICE inline bool ingest_contact_event(DeviceState* state,
                                                 const Event& event,
                                                 std::uint16_t p0_cell,
                                                 RunMode mode,
                                                 DeviceReceipt* receipt) {
  ContactState& contact = state->contact;
  contact.event_closed_content_p1_cell = 0u;
  contact.event_separator_p1_cell = 0u;
  contact.event_closed_content_p1_valid = 0u;
  contact.event_separator_p1_valid = 0u;
  if (mode != RunMode::kLearn) return true;
  if (contact.active == 0u) {
    contact.active = 1u;
    contact.start_tick = state->tick;
    contact.pending_p0_count = 0u;
    contact.pending_p1_count = 0u;
    contact.recalled_form_count = 0u;
    contact.replay_start_count = 0u;
    contact.has_recalled = 0u;
    contact.recalled_contact_count = 0u;
    contact.recalled_contact_cell = 0u;
  }
  if (event.contact_boundary != 0u) {
    if (!close_contact_state(state, receipt)) return false;
    contact.last_value = event.value;
    contact.has_last_value = 0u;
    return true;
  }
  if (contact.has_last_value != 0u && contact.last_value != event.value) {
    ++contact.transition_degree[contact.last_value];
    if (!remember_successor(&contact, contact.last_value, event.value))
      return false;
  }
  ++contact.interior_evidence[event.value];
  const bool separator = learned_separator(contact, event.value);
  if (separator) {
    if (!close_pending_form(state, receipt)) return false;
    if (contact.event_closed_content_p1_valid != 0u) {
      const std::uint16_t separator_p0 = p0_cell;
      std::uint16_t separator_p1 = 0u;
      if (!materialize_form_from_p0(state, &separator_p0, 1u, &separator_p1,
                                    receipt))
        return false;
      contact.event_separator_p1_cell = separator_p1;
      contact.event_separator_p1_valid = 1u;
    }
  } else if (!remember_contact_cell(state, p0_cell)) {
    return false;
  }
  contact.last_value = event.value;
  contact.has_last_value = 1u;
  return true;
}
BCC32_PA_DEVICE inline std::uint64_t hash_event(const Event& event, std::uint32_t tick) {
  const std::uint32_t event_bits =
      static_cast<std::uint32_t>(event.value) |
      (static_cast<std::uint32_t>(event.contact_boundary) << 8u);
  return static_cast<std::uint64_t>(mix(event_bits ^ tick));
}

BCC32_PA_DEVICE inline std::uint64_t hash_state(const DeviceState& state) {
  std::uint64_t hash = 0xcbf29ce484222325ull;
  for (std::uint32_t population = 0u; population < kPopulationCount; ++population) {
    const Population& pool = state.populations[population];
    hash ^= pool.represented_matter + 0x9e3779b97f4a7c15ull * (population + 1u);
    hash *= 0x100000001b3ull;
    hash ^= pool.free_matter + 0x517cc1b727220a95ull * (population + 1u);
    hash *= 0x100000001b3ull;
    for (std::uint32_t i = 0u; i < kCellsPerPopulation; ++i) {
      const Cell& cell = pool.cells[i];
      hash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.activation)) |
              (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.utility)) << 32u);
      hash ^= static_cast<std::uint64_t>(cell.generation) << 17u;
      hash ^= static_cast<std::uint64_t>(cell.live) << 7u;
      hash *= 0x100000001b3ull;
    }
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
      const Segment& segment = pool.segments[i];
      if (!segment.live) continue;
      hash ^= static_cast<std::uint64_t>(segment.target_cell + 257u * segment.source_population +
                                         65537u * static_cast<std::uint32_t>(segment.strength));
      hash ^= static_cast<std::uint64_t>(segment.generation) << 11u;
      hash ^= static_cast<std::uint64_t>(segment.target_generation) << 23u;
      hash *= 0x100000001b3ull;
    }
    for (std::uint32_t i = 0u; i < kSynapsesPerPopulation; ++i) {
      const Synapse& synapse = pool.synapses[i];
      if (!synapse.live) continue;
      hash ^= static_cast<std::uint64_t>(synapse.source_cell + 131u * synapse.delay +
                                         65537u * static_cast<std::uint8_t>(synapse.weight));
      hash ^= static_cast<std::uint64_t>(synapse.generation) << 19u;
      hash *= 0x100000001b3ull;
    }
    for (std::uint32_t i = 0u; i < kEligibilityPerPopulation; ++i) {
      const Eligibility& eligibility = pool.eligibilities[i];
      if (!eligibility.live) continue;
      hash ^= static_cast<std::uint64_t>(eligibility.segment_generation) |
              (static_cast<std::uint64_t>(eligibility.due_tick) << 32u);
      hash ^= static_cast<std::uint64_t>(eligibility.credit_group) << 13u;
      hash *= 0x100000001b3ull;
    }
  }
  hash ^= static_cast<std::uint64_t>(state.tick) |
          (static_cast<std::uint64_t>(state.error_bits) << 32u);
  hash ^= state.represented_matter + state.free_matter;
  hash ^= state.external_growth_calls + 0x9e3779b97f4a7c15ull;
  hash ^= state.internal_growth_calls + 0x517cc1b727220a95ull;
  hash ^= state.contact.contact_count +
          (static_cast<std::uint64_t>(state.contact.recalled_contact_cell) << 32u);
  for (std::uint32_t value = 0u; value < kSeparatorCount; ++value) {
    hash ^= state.contact.boundary_evidence[value] +
            0x100000001b3ull * state.contact.interior_evidence[value];
    hash *= 0x100000001b3ull;
  }
  for (std::uint32_t i = 0u; i < kHistoryDepth; ++i) {
    const HistoryEntry& entry = state.history[i];
    hash ^= static_cast<std::uint64_t>(entry.tick) |
            (static_cast<std::uint64_t>(entry.event.value) << 32u) |
            (static_cast<std::uint64_t>(entry.event.contact_boundary) << 40u);
    for (std::uint32_t population = 0u; population < kPopulationCount; ++population) {
      hash ^= entry.active_count[population] << (population & 7u);
      for (std::uint32_t cell = 0u; cell < entry.active_count[population]; ++cell)
        hash ^= static_cast<std::uint64_t>(entry.active_cells[population][cell]) <<
                ((cell + population) & 31u);
    }
    hash *= 0x100000001b3ull;
  }
  for (std::uint32_t i = 0u; i < kPublicationCapacity; ++i) {
    const Publication& publication = state.publications[i];
    hash ^= static_cast<std::uint64_t>(publication.live) |
            (static_cast<std::uint64_t>(publication.due_tick) << 8u) |
            (static_cast<std::uint64_t>(publication.credit_group) << 40u);
    hash *= 0x100000001b3ull;
  }
  return hash;
}
BCC32_PA_DEVICE inline void initialize_device_state(DeviceState* state) {
  for (std::uint32_t population = 0u; population < kPopulationCount; ++population) {
    Population& pool = state->populations[population];
    for (std::uint32_t i = 0u; i < kCellsPerPopulation; ++i) pool.cells[i].live = 1u;
    for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i)
      pool.free_segments[i] = static_cast<std::uint16_t>(i);
    for (std::uint32_t i = 0u; i < kSynapsesPerPopulation; ++i)
      pool.free_synapses[i] = static_cast<std::uint16_t>(i);
    pool.free_segment_count = kSegmentsPerPopulation;
    pool.free_synapse_count = kSynapsesPerPopulation;
    pool.free_matter = kSegmentsPerPopulation + kSynapsesPerPopulation;
    state->free_matter += pool.free_matter;
  }
  state->initial_matter = state->free_matter;
  state->initialized = 1u;
}

BCC32_PA_DEVICE inline bool publish_segment_eligibility(
    DeviceState* state, std::uint32_t target_population,
    std::uint32_t segment_index, const std::uint8_t* active_source,
    std::uint32_t publication_tick, std::int32_t trace,
    std::uint16_t* published_index = nullptr,
    std::uint32_t delay_filter = 0xffffffffu) {
  if (!validate_segment(*state, target_population, segment_index)) {
    set_error(state, kErrorInvariant);
    return false;
  }
  Population& target = state->populations[target_population];
  std::uint32_t eligibility_index = kEligibilityPerPopulation;
  for (std::uint32_t index = 0u; index < kEligibilityPerPopulation; ++index) {
    if (!target.eligibilities[index].live) {
      eligibility_index = index;
      break;
    }
  }
  if (eligibility_index == kEligibilityPerPopulation) {
    set_error(state, kErrorAllocatorExhausted);
    return false;
  }
  if (!journal_eligibility(state, target_population, eligibility_index))
    return false;
  Eligibility& eligibility = target.eligibilities[eligibility_index];
  eligibility = Eligibility{};
  const Segment& segment = target.segments[segment_index];
  eligibility.target_population = static_cast<std::uint8_t>(target_population);
  eligibility.source_population = segment.source_population;
  eligibility.segment_index = static_cast<std::uint16_t>(segment_index);
  eligibility.publication_tick = publication_tick;
  eligibility.segment_generation = segment.generation;
  eligibility.target_generation =
      state->populations[target_population].cells[segment.target_cell].generation;
  eligibility.trace = static_cast<std::int16_t>(
      trace < -32768 ? -32768 : trace > 32767 ? 32767 : trace);
  eligibility.live = 1u;
  const Population& source = state->populations[segment.source_population];
  for (std::uint32_t offset = 0u; offset < segment.synapse_count; ++offset) {
    const Synapse& synapse = source.synapses[segment.synapse_begin + offset];
    if (delay_filter != 0xffffffffu && synapse.delay != delay_filter) continue;
    bool active = synapse.delay == 0u && active_source != nullptr &&
                  active_source[synapse.source_cell] != 0u;
    if (synapse.delay != 0u && publication_tick >= synapse.delay) {
      const std::uint32_t source_tick = publication_tick - synapse.delay;
      const HistoryEntry& history = state->history[source_tick % kHistoryDepth];
      if (history.tick == source_tick) {
        for (std::uint32_t cell = 0u;
             cell < history.active_count[segment.source_population]; ++cell) {
          active = active ||
                   history.active_cells[segment.source_population][cell] ==
                       synapse.source_cell;
        }
      }
    }
    if (active && eligibility.active_synapse_count < kMaxSegmentSynapses) {
      const std::uint32_t active_index = eligibility.active_synapse_count++;
      eligibility.active_synapses[active_index] =
          static_cast<std::uint16_t>(segment.synapse_begin + offset);
      eligibility.active_synapse_generations[active_index] = synapse.generation;
      eligibility.active_synapse_delays[active_index] = synapse.delay;
    }
  }
  // The residual arrives when this tick's observation is available.  The
  // source delay identifies which prior history entry earned the prediction;
  // it is not an additional wait after the observation.
  eligibility.due_tick = publication_tick;
  eligibility.credit_group = static_cast<std::uint16_t>(publication_tick & 0xffffu);
  bool published = false;
  for (std::uint32_t publication = 0u; publication < kPublicationCapacity;
       ++publication) {
    if (state->publications[publication].live != 0u) continue;
    state->publications[publication].live = 1u;
    state->publications[publication].target_population =
        static_cast<std::uint8_t>(target_population);
    state->publications[publication].eligibility_index =
        static_cast<std::uint16_t>(eligibility_index);
    state->publications[publication].due_tick = eligibility.due_tick;
    state->publications[publication].credit_group = eligibility.credit_group;
    eligibility.publication_index = static_cast<std::uint16_t>(publication);
    ++state->publication_count;
    published = true;
    break;
  }
  if (!published) {
    eligibility.live = 0u;
    set_error(state, kErrorAllocatorExhausted);
    return false;
  }
  if (published_index != nullptr)
    *published_index = static_cast<std::uint16_t>(eligibility_index);
  return true;
}

BCC32_PA_DEVICE inline bool apply_segment_residual(
    DeviceState* state, std::uint32_t target_population,
    std::uint32_t eligibility_index, std::int32_t signed_residual,
    RunMode mode,
    std::uint32_t learn_population_mask = (1u << kPopulationCount) - 1u) {
  if (target_population >= kPopulationCount ||
      eligibility_index >= kEligibilityPerPopulation ||
      signed_residual < -1 || signed_residual > 1) {
    set_error(state, kErrorInvariant);
    return false;
  }
  Eligibility& eligibility =
      state->populations[target_population].eligibilities[eligibility_index];
  if (!eligibility.live ||
      eligibility.target_population != target_population ||
      !validate_segment(*state, target_population,
                        eligibility.segment_index) ||
      state->tick != eligibility.due_tick) {
    if (eligibility.live && state->tick != eligibility.due_tick)
      set_error(state, kErrorExpiredEligibility);
    set_error(state, kErrorInvariant);
    return false;
  }
  const Segment& checked_segment =
      state->populations[target_population].segments[eligibility.segment_index];
  if (checked_segment.generation != eligibility.segment_generation ||
      checked_segment.target_generation != eligibility.target_generation ||
      checked_segment.frozen != 0u ||
      edge_class_frozen(*state, checked_segment.source_population, target_population)) {
    set_error(state, kErrorExpiredEligibility);
    return false;
  }
  if (mode == RunMode::kLearn && signed_residual != 0 &&
      (learn_population_mask & (1u << target_population)) != 0u) {
    const std::uint32_t segment_index = eligibility.segment_index;
    if (!journal_segment(state, target_population, segment_index)) return false;
    Segment& segment =
        state->populations[target_population].segments[segment_index];
    const std::int32_t next_strength =
        static_cast<std::int32_t>(segment.strength) + signed_residual;
    const std::int32_t next_utility =
        static_cast<std::int32_t>(segment.utility) + signed_residual;
    segment.strength = static_cast<std::int16_t>(
        next_strength < -32768 ? -32768
                               : next_strength > 32767 ? 32767 : next_strength);
    segment.utility = static_cast<std::int16_t>(
        next_utility < -32768 ? -32768
                              : next_utility > 32767 ? 32767 : next_utility);
    Population& source =
        state->populations[eligibility.source_population];
    for (std::uint32_t offset = 0u;
         offset < eligibility.active_synapse_count; ++offset) {
      const std::uint32_t synapse_index = eligibility.active_synapses[offset];
      if (!journal_synapse(state, eligibility.source_population,
                           synapse_index))
        return false;
      Synapse& synapse = source.synapses[synapse_index];
      if (!synapse.live || synapse.generation != eligibility.active_synapse_generations[offset] ||
          synapse.delay != eligibility.active_synapse_delays[offset] || synapse.frozen != 0u) {
        set_error(state, kErrorExpiredEligibility);
        return false;
      }
      const std::int32_t next_weight =
          static_cast<std::int32_t>(synapse.weight) + signed_residual;
      const std::int32_t next_synapse_utility =
          static_cast<std::int32_t>(synapse.utility) + signed_residual;
      synapse.weight = static_cast<std::int8_t>(
          next_weight < -127 ? -127 : next_weight > 127 ? 127 : next_weight);
      synapse.utility = static_cast<std::int16_t>(
          next_synapse_utility < -32768
              ? -32768
              : next_synapse_utility > 32767 ? 32767
                                             : next_synapse_utility);
    }
  }
  if (!journal_eligibility(state, target_population, eligibility_index))
    return false;
  eligibility.live = 0u;
  if (eligibility.publication_index < kPublicationCapacity &&
      state->publications[eligibility.publication_index].live != 0u) {
    state->publications[eligibility.publication_index].live = 0u;
    if (state->publication_count > 0u) --state->publication_count;
  }
  return true;
}

#include "bcc32_grown_predictive_assembly_eligibility_lifecycle.inl"

BCC32_PA_DEVICE inline bool learn_p0_order(DeviceState* state, const Event& event,
                                           std::uint16_t current_cell,
                                           std::uint32_t tick, RunMode mode,
                                           std::uint32_t learn_population_mask,
                                           DeviceReceipt* receipt) {
  const HistoryEntry& previous = state->history[(tick - 1u) % kHistoryDepth];
  const HistoryEntry& previous_two = state->history[(tick - 2u) % kHistoryDepth];
  if (tick < 2u || previous.tick + 1u != tick || previous.active_count[kP0] == 0u ||
      previous.event.contact_boundary != 0u || event.contact_boundary != 0u ||
      previous_two.tick + 2u != tick || previous_two.active_count[kP0] == 0u ||
      mode != RunMode::kLearn ||
      (learn_population_mask & (1u << kP0)) == 0u)
    return true;
  const std::uint32_t target = current_cell;
  std::uint16_t segment_index = 0u;
  bool found = false;
  for (std::uint32_t i = 0u; i < kSegmentsPerPopulation; ++i) {
    const Segment& segment = state->populations[kP0].segments[i];
    if (!segment.live || segment.source_population != kP0 ||
        segment.target_cell != target) {
      continue;
    }
    const Population& source = state->populations[kP0];
    const Synapse& first = source.synapses[segment.synapse_begin];
    if (first.source_cell == previous.active_cells[kP0][0u]) {
      segment_index = static_cast<std::uint16_t>(i);
      found = true;
      break;
    }
  }
  if (!found) {
    if (!allocate_segment(state, kP0, kP0, target, 2u, 1u, &segment_index)) return false;
    if (!configure_synapse(state, kP0, segment_index, 0u,
                           previous.active_cells[kP0][0u], 1u, 1) ||
        !configure_synapse(state, kP0, segment_index, 1u,
                           previous_two.active_cells[kP0][0u],
                           2u, 1))
      return false;
    ++receipt->learned_segment_count;
  }
  if (!journal_cell(state, kP0, current_cell)) return false;
  Cell& winner = state->populations[kP0].cells[current_cell];
  if (winner.utility < 0x7fffffff) ++winner.utility;
  return true;
}

BCC32_PA_DEVICE inline bool record_history(DeviceState* state, const Event& event,
                                           std::uint16_t observed_cell,
                                           std::uint32_t tick) {
  const std::uint32_t slot = tick % kHistoryDepth;
  if (!journal_history(state, slot)) return false;
  HistoryEntry& entry = state->history[slot];
  entry = HistoryEntry{};
  entry.event = event;
  entry.active_count[kP0] = 1u;
  entry.active_cells[kP0][0u] = observed_cell;
  entry.tick = tick;
  return true;
}

BCC32_PA_DEVICE inline bool record_population_activity(
    DeviceState* state, std::uint32_t population, const std::uint16_t* cells,
    std::uint32_t count, std::uint32_t tick) {
  if (population >= kPopulationCount || count > kReceiptWinners ||
      (count != 0u && cells == nullptr)) {
    set_error(state, kErrorInvariant);
    return false;
  }
  const std::uint32_t slot = tick % kHistoryDepth;
  HistoryEntry& entry = state->history[slot];
  if (entry.tick != tick || !journal_history(state, slot)) return false;
  entry.active_count[population] = static_cast<std::uint8_t>(count);
  for (std::uint32_t index = 0u; index < count; ++index) {
    if (cells[index] >= kCellsPerPopulation) {
      set_error(state, kErrorInvariant);
      return false;
    }
    entry.active_cells[population][index] = cells[index];
  }
  return true;
}

BCC32_PA_DEVICE inline bool run_tick(DeviceState* state, const RunCommand& command,
                                     DeviceReceipt* receipt) {
  if (!state || !receipt || !command.has_event || command.mode > RunMode::kProbe) {
    if (state) set_error(state, kErrorCommand);
    return false;
  }
  if (!state->initialized) initialize_device_state(state);
  *receipt = DeviceReceipt{};
  const bool read_only = command.mode != RunMode::kLearn;
  begin_transaction(state);
  const std::uint32_t mark = state->transaction_mark;
  receipt->state_hash_before = hash_state(*state);
  if (!journal_metadata(state)) return false;
  const std::uint32_t tick = command.tick != 0u ? command.tick : state->tick + 1u;
  state->tick = tick;
  receipt->tick = tick;
  if (!publish_prediction(state, receipt, tick)) {
    if (command.rollback_on_error) inverse_to(state, mark);
    receipt->errors = state->error_bits;
    return false;
  }
  const std::uint16_t observed_cell = select_context_cell(*state, command.event);
  if (!apply_prediction_residual(state, receipt, observed_cell,
                                 event_column(command.event), tick,
                                 command.mode,
                                 command.learn_population_mask) ||
      !learn_p0_order(state, command.event, observed_cell, tick, command.mode,
                      command.learn_population_mask,
                      receipt) ||
      !ingest_contact_event(state, command.event, observed_cell, command.mode,
                            receipt) ||
      !record_history(state, command.event, observed_cell, tick)) {
    if (command.rollback_on_error) inverse_to(state, mark);
    receipt->errors = state->error_bits;
    return false;
  }
  const HistoryEntry& latest = state->history[tick % kHistoryDepth];
  receipt->output_winner_count = latest.active_count[kP0];
  for (std::uint32_t i = 0u; i < receipt->output_winner_count; ++i)
    receipt->output_winner_columns[i] = latest.active_cells[kP0][i];
  receipt->output_hash = hash_event(command.event, tick);
  if (command.event.contact_boundary != 0u &&
      state->contact.pending_p1_count != 0u) {
    const std::uint32_t p2_count = state->contact.has_recalled != 0u
                                       ? state->contact.recalled_contact_count
                                       : 0u;
    if (!record_population_activity(state, kP1, state->contact.pending_p1,
                                    state->contact.pending_p1_count, tick) ||
        !record_population_activity(state, kP2,
                                    state->contact.recalled_contact_cells,
                                    p2_count, tick)) {
      if (command.rollback_on_error) inverse_to(state, mark);
      receipt->errors = state->error_bits;
      return false;
    }
  }
  receipt->journal_entries = state->journal_count - mark;
  receipt->learned_separator_count = state->contact.separator_count;
  receipt->recalled_contact_cell = state->contact.recalled_contact_cell;
  receipt->recalled_contact_count = state->contact.recalled_contact_count;
  receipt->recalled_form_count = state->contact.recalled_form_count;
  receipt->replay_start_count = state->contact.replay_start_count;
  receipt->external_growth_calls = state->external_growth_calls;
  receipt->internal_growth_calls = state->internal_growth_calls;
  for (std::uint32_t population = 0u; population < kPopulationCount; ++population) {
    receipt->represented_matter += state->populations[population].represented_matter;
    receipt->free_matter += state->populations[population].free_matter;
  }
  receipt->represented_matter = state->represented_matter;
  receipt->free_matter = state->free_matter;
  receipt->errors = state->error_bits;
  if (read_only) {
    inverse_to(state, mark);
  }
  // Transaction markers are execution scratch, not persistent organism state.
  // Canonicalize them after either commit or rollback so a read-only tick can
  // recover the complete byte image, not merely its semantic hash.
  state->transaction_mark = 0u;
  state->metadata_snapshot_active = 0u;
  state->transaction_contact_before = ContactState{};
  for (std::uint32_t i = 0u; i < kPublicationCapacity; ++i)
    state->transaction_publications_before[i] = Publication{};
  receipt->state_hash_after = hash_state(*state);
  return true;
}

#include "bcc32_grown_predictive_assembly_runtime.inl"

}  // namespace bcc32::grown_predictive_assembly

#undef BCC32_PA_HD
#undef BCC32_PA_DEVICE
#undef BCC32_PA_GLOBAL
