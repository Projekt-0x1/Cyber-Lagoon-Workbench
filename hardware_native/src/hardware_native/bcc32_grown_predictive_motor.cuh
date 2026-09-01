#pragma once

#include "bcc32_grown_predictive_assembly.cuh"

#if defined(__CUDACC__)
#define BCC32_PM_HD __host__ __device__
#define BCC32_PM_DEVICE __device__
#else
#define BCC32_PM_HD
#define BCC32_PM_DEVICE
#endif

namespace bcc32::grown_predictive_assembly {

// Timed motor replay is intentionally separate from DeviceState until the
// adult integration owns its persistent fields.  P2 history drives transient
// P1 pulses, P1 position-delay edges drive P0 pulses, and only the shared
// structural feature rails are credit-bearing.  No form bytes are copied into
// this state and no source contact is read as an answer.
constexpr std::uint32_t kMotorLaneCapacity = 32u;
constexpr std::uint32_t kMotorRecalledPositionCount = kContactMemberCapacity;
constexpr std::uint32_t kMotorCurrentAbsentPosition = kContactMemberCapacity;
constexpr std::uint32_t kMotorCurrentPositionCount = kContactMemberCapacity + 1u;
constexpr std::uint32_t kMotorFeatureCount =
    kMotorRecalledPositionCount * kMotorCurrentPositionCount;
constexpr std::uint32_t kMotorEligibilityCapacity = kHistoryDepth;
constexpr std::int32_t kMotorWeightLimit = 127;
static_assert(kMotorFeatureCount <= 0xffffu);

struct MotorFeatureRail {
  std::uint16_t p3_cell = 0u;
  std::uint16_t p4_cell = 0u;
  std::uint16_t p5_cell = 0u;
  std::uint16_t p3_segment = 0u;
  std::uint16_t p4_segment = 0u;
  std::uint16_t p5_segment = 0u;
  std::uint32_t p3_generation = 0u;
  std::uint32_t p4_generation = 0u;
  std::uint32_t p5_generation = 0u;
  std::uint8_t initialized = 0u;
  std::uint8_t reserved[3]{};
};

struct MotorCandidateLane {
  // This is a transient active P1 pulse, not a lexical record.
  std::uint16_t p1_cell = 0u;
  std::uint16_t feature = 0u;
  std::uint8_t active = 0u;
  std::uint8_t resolved = 0u;
  std::uint8_t failed = 0u;
  std::uint8_t predicted_valid = 0u;
  std::uint8_t predicted_tie = 0u;
  std::uint8_t reserved = 0u;
  std::uint32_t launch_tick = 0u;
  std::uint32_t prediction_tick = 0u;
  std::uint16_t predicted_p0_cell = 0u;
  std::uint16_t predicted_column = 0u;
  std::uint16_t output_steps = 0u;
  std::uint16_t position = 0u;
};

struct MotorEligibility {
  std::uint8_t live = 0u;
  std::uint16_t feature = 0u;
  std::int8_t signed_residual = 0;
  std::uint8_t reserved = 0u;
  std::uint32_t publication_tick = 0u;
  std::uint32_t due_tick = 0u;
  // These are shared feature routes: P3->P4, P4->P5, and P3->P5.
  std::uint16_t p3_segment = 0u;
  std::uint16_t p4_segment = 0u;
  std::uint16_t p5_segment = 0u;
  std::uint32_t p3_generation = 0u;
  std::uint32_t p4_generation = 0u;
  std::uint32_t p5_generation = 0u;
};

struct MotorSelection {
  std::uint8_t found = 0u;
  std::uint8_t unique = 0u;
  std::uint8_t lane = 0u;
  std::uint16_t feature = 0u;
  std::uint16_t p1_cell = 0u;
  std::int16_t score = 0;
  std::int16_t runner_up = 0;
};

struct MotorSettledSupport {
  std::int32_t p3 = 0;
  std::int32_t p4 = 0;
  std::int32_t p5 = 0;
  std::uint8_t p4_active = 0u;
  std::uint8_t p5_active = 0u;
  std::uint8_t reserved[2]{};
};

struct MotorLearnerReceipt {
  std::uint32_t tick = 0u;
  std::uint32_t p1_pulses = 0u;
  std::uint32_t p0_predictions = 0u;
  std::uint32_t raw_matches = 0u;
  std::uint32_t raw_misses = 0u;
  std::uint32_t positive_updates = 0u;
  std::uint32_t negative_updates = 0u;
  std::uint32_t expired_updates = 0u;
  std::uint32_t abstentions = 0u;
  std::uint32_t errors = 0u;
  std::uint64_t state_hash_before = 0u;
  std::uint64_t state_hash_after = 0u;
};

struct MotorLearnerState {
  MotorFeatureRail features[kMotorFeatureCount]{};
  MotorCandidateLane lanes[kMotorLaneCapacity]{};
  MotorEligibility eligibility[kMotorEligibilityCapacity]{};
  std::uint32_t tick = 0u;
  std::uint32_t replay_count = 0u;
  std::uint32_t expired_eligibility = 0u;
  std::uint32_t dropped_eligibility = 0u;
  std::uint32_t error_bits = 0u;
  std::uint8_t initialized = 0u;
  std::uint8_t selection_consumed = 0u;
  std::uint8_t committed_valid = 0u;
  std::uint8_t committed_lane = 0u;
  std::uint16_t separator_form_cell = 0u;
  std::uint16_t separator_p0_cell = 0u;
  std::uint16_t separator_route_segment = 0u;
  std::uint32_t separator_route_generation = 0u;
  std::uint16_t separator_form_segment = 0u;
  std::uint32_t separator_form_generation = 0u;
  std::uint8_t separator_ready = 0u;
  std::uint8_t reserved_separator[3]{};
};

static_assert(std::is_integral_v<decltype(MotorCandidateLane::p1_cell)>);
static_assert(std::is_same_v<decltype(MotorCandidateLane::feature),
                             std::uint16_t>);
static_assert(std::is_same_v<decltype(MotorEligibility::feature),
                             std::uint16_t>);
static_assert(std::is_same_v<decltype(MotorSelection::feature),
                             std::uint16_t>);
static_assert(std::is_integral_v<decltype(MotorEligibility::signed_residual)>);
static_assert(sizeof(MotorLearnerState) < 64u * 1024u);

BCC32_PM_HD inline std::int16_t motor_clamp_score(std::int32_t value) {
  return static_cast<std::int16_t>(value < -32768 ? -32768
                                                   : value > 32767 ? 32767
                                                                   : value);
}

BCC32_PM_HD inline std::int8_t motor_clamp_weight(std::int32_t value) {
  return static_cast<std::int8_t>(value < -kMotorWeightLimit
                                      ? -kMotorWeightLimit
                                      : value > kMotorWeightLimit ? kMotorWeightLimit : value);
}

BCC32_PM_DEVICE inline void initialize_motor_learner(MotorLearnerState* state) {
  if (state == nullptr) return;
  *state = MotorLearnerState{};
  state->initialized = 1u;
}

BCC32_PM_DEVICE inline void reset_motor_lanes(MotorLearnerState* state) {
  if (state == nullptr) return;
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index)
    state->lanes[index] = MotorCandidateLane{};
  state->selection_consumed = 0u;
  state->committed_valid = 0u;
  state->committed_lane = 0u;
}

BCC32_PM_DEVICE inline void reset_motor_learner(MotorLearnerState* state) {
  initialize_motor_learner(state);
}

BCC32_PM_DEVICE inline void snapshot_motor_learner(
    const MotorLearnerState& state, MotorLearnerState* snapshot) {
  if (snapshot != nullptr) *snapshot = state;
}

BCC32_PM_DEVICE inline void inverse_motor_learner(
    const MotorLearnerState& snapshot, MotorLearnerState* state) {
  if (state != nullptr) *state = snapshot;
}

BCC32_PM_DEVICE inline std::uint64_t hash_motor_learner(
    const MotorLearnerState& state) {
  std::uint64_t hash = 0xcbf29ce484222325ull;
  for (std::uint32_t index = 0u; index < kMotorFeatureCount; ++index) {
    const MotorFeatureRail& rail = state.features[index];
    hash ^= static_cast<std::uint64_t>(rail.p3_cell) |
            (static_cast<std::uint64_t>(rail.p4_cell) << 16u) |
            (static_cast<std::uint64_t>(rail.p5_cell) << 32u) |
            (static_cast<std::uint64_t>(rail.initialized) << 48u);
    hash *= 0x100000001b3ull;
  }
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index) {
    const MotorCandidateLane& lane = state.lanes[index];
    hash ^= static_cast<std::uint64_t>(lane.p1_cell) |
            (static_cast<std::uint64_t>(lane.feature) << 16u) |
            (static_cast<std::uint64_t>(lane.active) << 24u) |
            (static_cast<std::uint64_t>(lane.launch_tick) << 32u);
    hash *= 0x100000001b3ull;
  }
  for (std::uint32_t index = 0u; index < kMotorEligibilityCapacity; ++index) {
    const MotorEligibility& entry = state.eligibility[index];
    if (entry.live == 0u) continue;
    hash ^= static_cast<std::uint64_t>(entry.feature) |
            (static_cast<std::uint64_t>(static_cast<std::uint8_t>(
                 entry.signed_residual)) << 8u) |
            (static_cast<std::uint64_t>(entry.due_tick) << 16u);
    hash *= 0x100000001b3ull;
  }
  hash ^= static_cast<std::uint64_t>(state.separator_form_cell) |
          (static_cast<std::uint64_t>(state.separator_p0_cell) << 16u) |
          (static_cast<std::uint64_t>(state.separator_route_segment) << 32u) |
          (static_cast<std::uint64_t>(state.separator_ready) << 48u);
  hash *= 0x100000001b3ull;
  return hash ^ static_cast<std::uint64_t>(state.tick) ^
         (static_cast<std::uint64_t>(state.error_bits) << 32u) ^
         (static_cast<std::uint64_t>(state.committed_valid) << 48u) ^
         (static_cast<std::uint64_t>(state.committed_lane) << 56u);
}

BCC32_PM_DEVICE inline bool motor_find_route(
    const DeviceState& state, std::uint32_t target_population,
    std::uint32_t source_population, std::uint32_t target_cell,
    std::uint32_t source_cell, std::uint32_t delay,
    std::uint16_t* segment_index) {
  if (segment_index == nullptr || target_population >= kPopulationCount ||
      source_population >= kPopulationCount || target_cell >= kCellsPerPopulation ||
      source_cell >= kCellsPerPopulation)
    return false;
  const Population& target = state.populations[target_population];
  const Population& source = state.populations[source_population];
  for (std::uint32_t index = 0u; index < kSegmentsPerPopulation; ++index) {
    const Segment& segment = target.segments[index];
    if (!segment.live || segment.source_population != source_population ||
        segment.target_cell != target_cell || segment.synapse_count != 1u)
      continue;
    const Synapse& synapse = source.synapses[segment.synapse_begin];
    if (synapse.live && synapse.source_cell == source_cell && synapse.delay == delay) {
      *segment_index = static_cast<std::uint16_t>(index);
      return true;
    }
  }
  return false;
}

BCC32_PM_DEVICE inline bool motor_ensure_route(
    DeviceState* state, std::uint32_t target_population,
    std::uint32_t source_population, std::uint32_t target_cell,
    std::uint32_t source_cell, std::uint32_t delay, std::int32_t weight) {
  if (state == nullptr || !edge_allowed(source_population, target_population) ||
      !edge_delay_allowed(source_population, target_population, delay))
    return false;
  std::uint16_t segment = 0u;
  if (motor_find_route(*state, target_population, source_population, target_cell,
                      source_cell, delay, &segment))
    return true;
  return grow_one_internal(state, target_population, source_population, target_cell,
                           source_cell, delay, weight);
}

BCC32_PM_DEVICE inline bool capture_learned_separator_disposition(
    DeviceState* state, MotorLearnerState* motor, const Event& event,
    std::uint32_t tick) {
  if (state == nullptr || motor == nullptr || !motor->initialized ||
      event.contact_boundary != 0u ||
      state->contact.event_separator_p1_valid == 0u)
    return false;
  const HistoryEntry& history = state->history[tick % kHistoryDepth];
  if (history.tick != tick || history.active_count[kP0] != 1u)
    return false;
  const std::uint16_t p0_cell = history.active_cells[kP0][0u];
  const std::uint16_t form_cell = state->contact.event_separator_p1_cell;
  if (p0_cell >= kCellsPerPopulation || form_cell >= kCellsPerPopulation)
    return false;
  // Assembly owns the separator P1 disposition. The motor may only consume
  // its existing P0->P1 route and grow the reverse action route.
  std::uint16_t assembly_form_segment = 0u;
  if (!motor_find_route(*state, kP1, kP0, form_cell, p0_cell, 0u,
                        &assembly_form_segment) ||
      !validate_segment(*state, kP1, assembly_form_segment))
    return false;
  const Segment& assembly_form =
      state->populations[kP1].segments[assembly_form_segment];
  if (assembly_form.source_population != kP0 ||
      assembly_form.target_cell != form_cell ||
      assembly_form.synapse_count != 1u || assembly_form.threshold == 0u)
    return false;
  const Synapse& assembly_synapse =
      state->populations[kP0].synapses[assembly_form.synapse_begin];
  if (!assembly_synapse.live || assembly_synapse.source_cell != p0_cell ||
      assembly_synapse.delay != 0u || assembly_synapse.weight <= 0)
    return false;
  if (!motor_ensure_route(state, kP0, kP1, p0_cell, form_cell, 0u, 1))
    return false;
  std::uint16_t route_segment = 0u;
  if (!motor_find_route(*state, kP0, kP1, p0_cell, form_cell, 0u,
                        &route_segment) ||
      !validate_segment(*state, kP0, route_segment))
    return false;
  const Segment& route = state->populations[kP0].segments[route_segment];
  if (route.source_population != kP1 || route.target_cell != p0_cell ||
      route.synapse_count != 1u || route.threshold == 0u)
    return false;
  const Synapse& route_synapse =
      state->populations[kP1].synapses[route.synapse_begin];
  if (!route_synapse.live || route_synapse.source_cell != form_cell ||
      route_synapse.delay != 0u || route_synapse.weight <= 0)
    return false;
  if (motor->separator_ready != 0u &&
      (motor->separator_p0_cell != p0_cell ||
       motor->separator_form_cell != form_cell ||
       motor->separator_route_segment != route_segment ||
       motor->separator_form_segment != assembly_form_segment ||
       !validate_segment(*state, kP0, motor->separator_route_segment) ||
       !validate_segment(*state, kP1, motor->separator_form_segment) ||
       state->populations[kP0].segments[motor->separator_route_segment].generation !=
           motor->separator_route_generation ||
       state->populations[kP1].segments[motor->separator_form_segment].generation !=
           motor->separator_form_generation))
    return false;
  motor->separator_form_cell = form_cell;
  motor->separator_p0_cell = p0_cell;
  motor->separator_route_segment = route_segment;
  motor->separator_route_generation =
      state->populations[kP0].segments[route_segment].generation;
  motor->separator_form_segment = assembly_form_segment;
  motor->separator_form_generation =
      state->populations[kP1].segments[assembly_form_segment].generation;
  motor->separator_ready = 1u;
  return true;
}

BCC32_PM_DEVICE inline bool unfold_learned_separator_disposition(
    const DeviceState& state, const MotorLearnerState& motor,
    MotorTrajectory* output) {
  if (output == nullptr || motor.separator_ready == 0u ||
      motor.separator_form_cell >= kCellsPerPopulation ||
      motor.separator_p0_cell >= kCellsPerPopulation ||
      !validate_segment(state, kP0, motor.separator_route_segment) ||
      !validate_segment(state, kP1, motor.separator_form_segment))
    return false;
  const Segment& route =
      state.populations[kP0].segments[motor.separator_route_segment];
  if (route.generation != motor.separator_route_generation ||
      route.source_population != kP1 ||
      route.target_cell != motor.separator_p0_cell || route.synapse_count != 1u ||
      route.threshold == 0u)
    return false;
  const Segment& form_route =
      state.populations[kP1].segments[motor.separator_form_segment];
  if (form_route.generation != motor.separator_form_generation ||
      form_route.source_population != kP0 ||
      form_route.target_cell != motor.separator_form_cell ||
      form_route.synapse_count != 1u || form_route.threshold == 0u)
    return false;
  const Synapse& synapse =
      state.populations[kP1].synapses[route.synapse_begin];
  const Synapse& form_synapse =
      state.populations[kP0].synapses[form_route.synapse_begin];
  if (!synapse.live || synapse.source_cell != motor.separator_form_cell ||
      synapse.delay != 0u || synapse.weight <= 0 || !form_synapse.live ||
      form_synapse.source_cell != motor.separator_p0_cell ||
      form_synapse.delay != 0u || form_synapse.weight <= 0)
    return false;
  std::uint8_t active[kCellsPerPopulation]{};
  active[motor.separator_form_cell] = 1u;
  const std::int32_t support =
      active_segment_support(state, kP0, motor.separator_route_segment, active, 0u);
  if (support < static_cast<std::int32_t>(route.threshold)) return false;
  return append_motor_winner(output, motor.separator_p0_cell);
}

BCC32_PM_DEVICE inline std::uint32_t motor_current_position(
    std::uint16_t p1_cell, const std::uint16_t* current_forms,
    std::uint32_t current_count) {
  if (current_forms == nullptr && current_count != 0u)
    return kMotorCurrentAbsentPosition;
  for (std::uint32_t index = 0u; index < current_count; ++index)
    if (current_forms[index] == p1_cell)
      return index < kMotorCurrentAbsentPosition ? index : kMotorCurrentAbsentPosition;
  return kMotorCurrentAbsentPosition;
}

BCC32_PM_HD inline std::uint32_t motor_feature_id(
    std::uint32_t recalled_position, std::uint32_t current_position) {
  const std::uint32_t recalled = recalled_position < kMotorRecalledPositionCount
                                      ? recalled_position
                                      : kMotorRecalledPositionCount - 1u;
  const std::uint32_t current = current_position < kMotorCurrentPositionCount
                                    ? current_position
                                    : kMotorCurrentAbsentPosition;
  return recalled * kMotorCurrentPositionCount + current;
}

BCC32_PM_DEVICE inline bool motor_p1_position(
    const DeviceState& state, std::uint16_t p1_cell, const HistoryEntry& p2_history,
    std::uint32_t* position, std::int32_t* support) {
  if (position == nullptr || support == nullptr) return false;
  std::uint32_t best_position = 0xffffffffu;
  std::int32_t best_support = 0;
  const Population& p1 = state.populations[kP1];
  const Population& p2 = state.populations[kP2];
  for (std::uint32_t index = 0u; index < kSegmentsPerPopulation; ++index) {
    const Segment& segment = p2.segments[index];
    if (!segment.live || segment.source_population != kP1 ||
        segment.synapse_count != 1u)
      continue;
    bool target_active = false;
    for (std::uint32_t target = 0u; target < p2_history.active_count[kP2]; ++target)
      target_active = target_active ||
                      p2_history.active_cells[kP2][target] == segment.target_cell;
    if (!target_active) continue;
    const Synapse& synapse = p1.synapses[segment.synapse_begin];
    if (!synapse.live || synapse.source_cell != p1_cell ||
        synapse.delay >= kMotorRecalledPositionCount)
      continue;
    const std::int32_t trial = synapse.weight;
    if (best_position == 0xffffffffu || trial > best_support ||
        (trial == best_support && synapse.delay < best_position)) {
      best_position = synapse.delay;
      best_support = trial;
    }
  }
  if (best_position == 0xffffffffu) return false;
  *position = best_position;
  *support = best_support;
  return true;
}

BCC32_PM_DEVICE inline bool motor_ensure_feature_rail(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t feature) {
  if (state == nullptr || motor == nullptr || feature >= kMotorFeatureCount)
    return false;
  MotorFeatureRail& rail = motor->features[feature];
  if (rail.initialized == 0u) {
    rail.p3_cell = choose_growth_cell(*state, kP3);
    rail.p4_cell = choose_growth_cell(*state, kP4);
    rail.p5_cell = choose_growth_cell(*state, kP5);
    if (!touch_growth_cell(state, kP3, rail.p3_cell) ||
        !touch_growth_cell(state, kP4, rail.p4_cell) ||
        !touch_growth_cell(state, kP5, rail.p5_cell) ||
        !motor_ensure_route(state, kP3, kP3, rail.p3_cell, rail.p3_cell, 1u, 0) ||
        !motor_ensure_route(state, kP4, kP3, rail.p4_cell, rail.p3_cell, 0u, 0) ||
        !motor_ensure_route(state, kP5, kP4, rail.p5_cell, rail.p4_cell, 0u, 0))
      return false;
    if (!motor_find_route(*state, kP3, kP3, rail.p3_cell, rail.p3_cell, 1u,
                          &rail.p3_segment) ||
        !motor_find_route(*state, kP4, kP3, rail.p4_cell, rail.p3_cell, 0u,
                          &rail.p4_segment) ||
        !motor_find_route(*state, kP5, kP4, rail.p5_cell, rail.p4_cell, 0u,
                          &rail.p5_segment))
      return false;
    rail.p3_generation = state->populations[kP3].segments[rail.p3_segment].generation;
    rail.p4_generation = state->populations[kP4].segments[rail.p4_segment].generation;
    rail.p5_generation = state->populations[kP5].segments[rail.p5_segment].generation;
    rail.initialized = 1u;
  }
  return true;
}

BCC32_PM_DEVICE inline bool motor_feature_segments_valid(
    const DeviceState& state, const MotorFeatureRail& rail) {
  return rail.initialized != 0u &&
         validate_segment(state, kP3, rail.p3_segment) &&
         validate_segment(state, kP4, rail.p4_segment) &&
         validate_segment(state, kP5, rail.p5_segment) &&
         state.populations[kP3].segments[rail.p3_segment].generation == rail.p3_generation &&
         state.populations[kP4].segments[rail.p4_segment].generation == rail.p4_generation &&
         state.populations[kP5].segments[rail.p5_segment].generation == rail.p5_generation;
}

// Publish one candidate P3 pulse, settle its learned P3->P4 route, then
// publish only the resulting P4 pulse before settling P4->P5. Production
// observes this local activity frame, never cached Segment::strength.
BCC32_PM_DEVICE inline bool settle_motor_candidate_support(
    const DeviceState& state, const MotorLearnerState& motor,
    const MotorCandidateLane& lane, MotorSettledSupport* result) {
  if (result == nullptr || lane.feature >= kMotorFeatureCount ||
      lane.active == 0u || lane.failed != 0u ||
      !motor_feature_segments_valid(state, motor.features[lane.feature]))
    return false;
  const MotorFeatureRail& rail = motor.features[lane.feature];
  const Segment& p4_segment = state.populations[kP4].segments[rail.p4_segment];
  const Segment& p5_segment = state.populations[kP5].segments[rail.p5_segment];
  if (p4_segment.source_population != kP3 || p4_segment.target_cell != rail.p4_cell ||
      p4_segment.synapse_count != 1u || p5_segment.source_population != kP4 ||
      p5_segment.target_cell != rail.p5_cell || p5_segment.synapse_count != 1u)
    return false;
  const Synapse& p4_synapse =
      state.populations[kP3].synapses[p4_segment.synapse_begin];
  const Synapse& p5_synapse =
      state.populations[kP4].synapses[p5_segment.synapse_begin];
  if (!p4_synapse.live || p4_synapse.source_cell != rail.p3_cell ||
      p4_synapse.delay != 0u || !p5_synapse.live ||
      p5_synapse.source_cell != rail.p4_cell || p5_synapse.delay != 0u)
    return false;

  PopulationActivityFrame activity{};
  activity.active[kP3][rail.p3_cell] = 1u;
  result->p3 = activity.active[kP3][rail.p3_cell];
  result->p4 = active_segment_support(state, kP4, rail.p4_segment,
                                       activity.active[kP3], 0u);
  result->p4_active = static_cast<std::uint8_t>(
      result->p4 >= static_cast<std::int32_t>(p4_segment.threshold));
  if (result->p4_active != 0u)
    activity.active[kP4][rail.p4_cell] = 1u;
  result->p5 = active_segment_support(state, kP5, rail.p5_segment,
                                       activity.active[kP4], 0u);
  result->p5_active = static_cast<std::uint8_t>(
      result->p4_active != 0u &&
      result->p5 >= static_cast<std::int32_t>(p5_segment.threshold));
  return true;
}

// P2 history is the only source of replay candidates.  The P1 set is settled
// from delayed resident synapses; recalled_p1[] and any source-contact bytes
// are deliberately not consulted.
BCC32_PM_DEVICE inline bool begin_recalled_motor_replay(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t tick,
    MotorLearnerReceipt* receipt = nullptr) {
  if (state == nullptr || motor == nullptr || !motor->initialized)
    return false;
  if (motor->selection_consumed != 0u) reset_motor_lanes(motor);
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index)
    if (motor->lanes[index].active != 0u || motor->lanes[index].resolved != 0u)
      return false;
  const HistoryEntry& p2_history = state->history[(tick - 1u) % kHistoryDepth];
  if (tick == 0u || p2_history.tick + 1u != tick ||
      p2_history.active_count[kP2] == 0u)
    return false;
  const std::uint16_t* current_forms = p2_history.active_cells[kP1];
  const std::uint32_t current_count = p2_history.active_count[kP1];
  std::uint32_t registered = 0u;
  for (std::uint32_t p1_cell = 0u;
       p1_cell < kCellsPerPopulation && registered < kMotorLaneCapacity; ++p1_cell) {
    const std::int32_t support =
        settle_target(*state, kP1, p1_cell, 1u << kP2, tick, 0, nullptr);
    if (support <= 0) continue;
    std::uint32_t position = 0u;
    std::int32_t position_support = 0;
    if (!motor_p1_position(*state, static_cast<std::uint16_t>(p1_cell), p2_history,
                           &position, &position_support))
      continue;
    (void)position_support;
    const std::uint32_t feature = motor_feature_id(
        position, motor_current_position(static_cast<std::uint16_t>(p1_cell),
                                         current_forms, current_count));
    if (!motor_ensure_feature_rail(state, motor, feature)) {
      motor->error_bits |= 1u;
      continue;
    }
    MotorCandidateLane& lane = motor->lanes[registered++];
    lane = MotorCandidateLane{};
    lane.p1_cell = static_cast<std::uint16_t>(p1_cell);
    lane.feature = static_cast<std::uint16_t>(feature);
    lane.launch_tick = tick;
    lane.position = static_cast<std::uint16_t>(position);
    lane.active = 1u;
    if (receipt != nullptr) ++receipt->p1_pulses;
  }
  if (registered != 0u) ++motor->replay_count;
  motor->tick = tick;
  return registered != 0u;
}

BCC32_PM_DEVICE inline bool motor_settle_lane_p0(
    const DeviceState& state, MotorCandidateLane* lane, std::uint32_t tick) {
  if (lane == nullptr || lane->active == 0u || tick < lane->launch_tick) return false;
  lane->predicted_valid = 0u;
  lane->predicted_column = 0u;
  bool found = false;
  bool tied = false;
  std::int32_t best_support = 0;
  std::uint16_t best_cell = 0u;
  const Population& p0 = state.populations[kP0];
  const Population& p1 = state.populations[kP1];
  const std::uint32_t elapsed = tick - lane->launch_tick;
  for (std::uint32_t index = 0u; index < kSegmentsPerPopulation; ++index) {
    const Segment& segment = p0.segments[index];
    if (!segment.live || segment.source_population != kP1 ||
        segment.target_cell >= kCellsPerPopulation)
      continue;
    std::int32_t support = 0;
    for (std::uint32_t offset = 0u; offset < segment.synapse_count; ++offset) {
      const Synapse& synapse = p1.synapses[segment.synapse_begin + offset];
      if (synapse.live && synapse.source_cell == lane->p1_cell &&
          synapse.delay == elapsed)
        support += synapse.weight;
    }
    if (support < static_cast<std::int32_t>(segment.threshold)) continue;
    if (!found || support > best_support) {
      found = true;
      tied = false;
      best_support = support;
      best_cell = segment.target_cell;
    } else if (support == best_support && segment.target_cell != best_cell) {
      tied = true;
    }
  }
  lane->predicted_valid = static_cast<std::uint8_t>(found && !tied);
  lane->predicted_tie = static_cast<std::uint8_t>(found && tied);
  lane->predicted_p0_cell = best_cell;
  lane->predicted_column = static_cast<std::uint16_t>(context_cell_column(best_cell));
  lane->prediction_tick = tick;
  if (found && !tied) return true;
  return found;
}

BCC32_PM_DEVICE inline bool settle_all_motor_predictions(
    const DeviceState& state, MotorLearnerState* motor, std::uint32_t tick,
    MotorLearnerReceipt* receipt = nullptr) {
  if (motor == nullptr || !motor->initialized) return false;
  bool any = false;
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index) {
    MotorCandidateLane& lane = motor->lanes[index];
    if (lane.active == 0u) continue;
    any = true;
    if (motor_settle_lane_p0(state, &lane, tick) && lane.predicted_valid != 0u &&
        receipt != nullptr)
      ++receipt->p0_predictions;
  }
  motor->tick = tick;
  return any;
}

BCC32_PM_DEVICE inline bool apply_delayed_motor_credit(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t tick,
    MotorLearnerReceipt* receipt);

BCC32_PM_DEVICE inline bool observe_motor_raw_event(
    DeviceState* state, MotorLearnerState* motor, const Event& event,
    std::uint32_t tick, MotorLearnerReceipt* receipt = nullptr) {
  if (state == nullptr || motor == nullptr || !motor->initialized) return false;
  apply_delayed_motor_credit(state, motor, tick, receipt);
  settle_all_motor_predictions(*state, motor, tick, receipt);
  bool any = false;
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index) {
    MotorCandidateLane& lane = motor->lanes[index];
    if (lane.active == 0u) continue;
    any = true;
    const bool closed_content_match =
        state->contact.event_closed_content_p1_valid != 0u &&
        lane.p1_cell == state->contact.event_closed_content_p1_cell;
    // A separator closes only the exact published content form when no
    // unique P0 continuation remains. Other lanes still see the raw event.
    const bool match =
        closed_content_match
            ? lane.predicted_valid == 0u
            : event.contact_boundary != 0u
                  ? lane.predicted_valid == 0u
                  : lane.predicted_valid != 0u &&
                        lane.predicted_column == event_column(event);
    if (match) {
      if (receipt != nullptr) ++receipt->raw_matches;
    } else {
      lane.failed = 1u;
      if (receipt != nullptr) ++receipt->raw_misses;
    }
  }
  if (event.contact_boundary == 0u &&
      state->contact.event_separator_p1_valid != 0u &&
      !capture_learned_separator_disposition(state, motor, event, tick))
    motor->error_bits |= 1u << 3u;
  return any;
}

BCC32_PM_DEVICE inline bool motor_apply_segment_delta(
    DeviceState* state, std::uint32_t target_population, std::uint32_t segment_index,
    std::uint32_t expected_generation, std::int32_t residual) {
  if (state == nullptr || residual < -1 || residual > 1 || residual == 0 ||
      !validate_segment(*state, target_population, segment_index))
    return false;
  Segment& segment = state->populations[target_population].segments[segment_index];
  if (segment.generation != expected_generation || segment.frozen != 0u ||
      edge_class_frozen(*state, segment.source_population, target_population))
    return false;
  const Population& source_read = state->populations[segment.source_population];
  for (std::uint32_t offset = 0u; offset < segment.synapse_count; ++offset)
    if (!source_read.synapses[segment.synapse_begin + offset].live)
      return false;
  if (!journal_segment(state, target_population, segment_index)) return false;
  segment.strength = motor_clamp_score(static_cast<std::int32_t>(segment.strength) + residual);
  segment.utility = motor_clamp_score(static_cast<std::int32_t>(segment.utility) + residual);
  Population& source = state->populations[segment.source_population];
  for (std::uint32_t offset = 0u; offset < segment.synapse_count; ++offset) {
    const std::uint32_t synapse_index = segment.synapse_begin + offset;
    if (!journal_synapse(state, segment.source_population, synapse_index)) return false;
    Synapse& synapse = source.synapses[synapse_index];
    synapse.weight = motor_clamp_weight(static_cast<std::int32_t>(synapse.weight) + residual);
    synapse.utility = motor_clamp_score(static_cast<std::int32_t>(synapse.utility) + residual);
  }
  return true;
}

BCC32_PM_DEVICE inline bool motor_schedule_feature_credit(
    MotorLearnerState* motor, std::uint32_t feature, std::int32_t residual,
    std::uint32_t tick, const DeviceState& state) {
  if (motor == nullptr || feature >= kMotorFeatureCount || residual < -1 ||
      residual > 1 || residual == 0 ||
      !motor_feature_segments_valid(state, motor->features[feature]))
    return false;
  const MotorFeatureRail& rail = motor->features[feature];
  for (std::uint32_t index = 0u; index < kMotorEligibilityCapacity; ++index) {
    MotorEligibility& entry = motor->eligibility[index];
    if (entry.live != 0u) continue;
    entry = MotorEligibility{};
    entry.live = 1u;
    entry.feature = static_cast<std::uint16_t>(feature);
    entry.signed_residual = static_cast<std::int8_t>(residual);
    entry.publication_tick = tick;
    entry.due_tick = tick + 1u;
    entry.p3_segment = rail.p3_segment;
    entry.p4_segment = rail.p4_segment;
    entry.p5_segment = rail.p5_segment;
    entry.p3_generation = rail.p3_generation;
    entry.p4_generation = rail.p4_generation;
    entry.p5_generation = rail.p5_generation;
    return true;
  }
  ++motor->dropped_eligibility;
  return false;
}

BCC32_PM_DEVICE inline bool apply_delayed_motor_credit(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t tick,
    MotorLearnerReceipt* receipt = nullptr) {
  if (state == nullptr || motor == nullptr || !motor->initialized) return false;
  bool ok = true;
  for (std::uint32_t index = 0u; index < kMotorEligibilityCapacity; ++index) {
    MotorEligibility& entry = motor->eligibility[index];
    if (entry.live == 0u) continue;
    if (tick > entry.publication_tick + kHistoryDepth) {
      entry.live = 0u;
      ++motor->expired_eligibility;
      if (receipt != nullptr) ++receipt->expired_updates;
      continue;
    }
    if (tick != entry.due_tick) continue;
    const std::int32_t residual = entry.signed_residual;
    const bool valid = validate_segment(*state, kP3, entry.p3_segment) &&
                       validate_segment(*state, kP4, entry.p4_segment) &&
                       validate_segment(*state, kP5, entry.p5_segment) &&
                       state->populations[kP3].segments[entry.p3_segment].generation ==
                           entry.p3_generation &&
                       state->populations[kP4].segments[entry.p4_segment].generation ==
                           entry.p4_generation &&
                       state->populations[kP5].segments[entry.p5_segment].generation ==
                           entry.p5_generation;
    const bool updated = valid &&
        motor_apply_segment_delta(state, kP3, entry.p3_segment, entry.p3_generation, residual) &&
        motor_apply_segment_delta(state, kP4, entry.p4_segment, entry.p4_generation, residual) &&
        motor_apply_segment_delta(state, kP5, entry.p5_segment, entry.p5_generation, residual);
    if (!updated) {
      motor->error_bits |= 1u << 1u;
      ok = false;
    } else if (receipt != nullptr) {
      if (residual > 0) ++receipt->positive_updates;
      else ++receipt->negative_updates;
    }
    entry.live = 0u;
  }
  motor->tick = tick;
  return ok;
}

// The observer is called after every closed contact.  It resolves existing
// transient P1 lanes from physical membership, then leaves lane opening to
// begin_recalled_motor_replay(), which is driven by the next P2 history tick.
BCC32_PM_DEVICE inline bool observe_closed_contact(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t tick,
    MotorLearnerReceipt* receipt = nullptr) {
  if (state == nullptr || motor == nullptr || !motor->initialized)
    return false;
  const HistoryEntry& closed = state->history[tick % kHistoryDepth];
  if (closed.tick != tick || closed.event.contact_boundary == 0u) return false;
  if (receipt != nullptr) {
    receipt->tick = tick;
    receipt->state_hash_before = hash_motor_learner(*motor);
  }
  apply_delayed_motor_credit(state, motor, tick, receipt);
  bool any = false;
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index) {
    MotorCandidateLane& lane = motor->lanes[index];
    if (lane.active == 0u) continue;
    any = true;
    bool member = false;
    for (std::uint32_t form = 0u; form < closed.active_count[kP1]; ++form)
      member = member || closed.active_cells[kP1][form] == lane.p1_cell;
    const bool exact = member && lane.failed == 0u;
    if (!motor_schedule_feature_credit(motor, lane.feature, exact ? 1 : -1,
                                       tick, *state))
      motor->error_bits |= 1u << 2u;
    lane.active = 0u;
    lane.resolved = 1u;
  }
  ++motor->replay_count;
  if (receipt != nullptr) receipt->state_hash_after = hash_motor_learner(*motor);
  return any;
}

BCC32_PM_DEVICE inline MotorSelection select_unique_motor_candidate(
    const DeviceState& state, const MotorLearnerState& motor) {
  MotorSelection result{};
  std::int16_t best = -32768;
  std::int16_t second = -32768;
  std::uint32_t best_lane = kMotorLaneCapacity;
  for (std::uint32_t index = 0u; index < kMotorLaneCapacity; ++index) {
    const MotorCandidateLane& lane = motor.lanes[index];
    if ((lane.active == 0u && lane.resolved == 0u) || lane.failed != 0u) continue;
    MotorSettledSupport settled{};
    if (!settle_motor_candidate_support(state, motor, lane, &settled) ||
        settled.p5_active == 0u)
      continue;
    const std::int16_t content = lane.predicted_valid != 0u &&
                                         lane.predicted_tie == 0u
                                     ? 1
                                     : 0;
    const std::int16_t score = motor_clamp_score(
        settled.p5 * static_cast<std::int32_t>(content));
    if (score > best) {
      second = best;
      best = score;
      best_lane = index;
    } else if (score > second) {
      second = score;
    }
  }
  result.score = best;
  result.runner_up = second;
  if (best_lane == kMotorLaneCapacity || best <= 0) return result;
  result.found = 1u;
  if (best == second) return result;
  result.unique = 1u;
  result.lane = static_cast<std::uint8_t>(best_lane);
  result.feature = motor.lanes[best_lane].feature;
  result.p1_cell = motor.lanes[best_lane].p1_cell;
  return result;
}

// Settles one timed P1->P0 step and appends only the physically winning P0
// cell.  If no P0 pulse remains, positive shared P5 pressure permits the
// learned boundary/EOS route; there is no stored expected length.
BCC32_PM_DEVICE inline bool unfold_selected_motor_step(
    const DeviceState& state, MotorLearnerState* motor,
    const MotorSelection& selection, std::uint32_t tick,
    MotorTrajectory* output) {
  if (motor == nullptr || output == nullptr || selection.found == 0u ||
      selection.unique == 0u || selection.lane >= kMotorLaneCapacity)
    return false;
  MotorCandidateLane& lane = motor->lanes[selection.lane];
  if (lane.active == 0u && lane.resolved == 0u) return false;
  motor_settle_lane_p0(state, &lane, tick);
  if (lane.predicted_valid != 0u) {
    if (lane.predicted_column == kRawColumnCount - 1u)
      return append_motor_winner(output, lane.predicted_p0_cell);
    ++lane.output_steps;
    return append_motor_winner(output, lane.predicted_p0_cell);
  }
  MotorSettledSupport settled{};
  if (lane.output_steps == 0u ||
      !settle_motor_candidate_support(state, *motor, lane, &settled) ||
      settled.p5_active == 0u)
    return false;
  const std::uint16_t eos = static_cast<std::uint16_t>(
      (kRawColumnCount - 1u) * kContextCellsPerColumn);
  return append_motor_winner(output, eos);
}

BCC32_PM_DEVICE inline MotorSelection committed_motor_selection(
    const DeviceState& state, const MotorLearnerState& motor) {
  MotorSelection selection{};
  if (motor.committed_valid == 0u || motor.committed_lane >= kMotorLaneCapacity)
    return selection;
  const MotorCandidateLane& lane = motor.lanes[motor.committed_lane];
  MotorSettledSupport settled{};
  if (!settle_motor_candidate_support(state, motor, lane, &settled) ||
      settled.p5_active == 0u)
    return selection;
  selection.found = 1u;
  selection.unique = 1u;
  selection.lane = motor.committed_lane;
  selection.feature = lane.feature;
  selection.p1_cell = lane.p1_cell;
  selection.score = motor_clamp_score(settled.p5);
  return selection;
}

// Preferred inference entry: settle all transient lanes, apply due credit,
// select a unique positive shared feature/content product, and emit one timed
// motor step. Repeated calls advance the same P1 pulse through learned delays.
BCC32_PM_DEVICE inline bool select_and_unfold_motor_candidate(
    DeviceState* state, MotorLearnerState* motor, std::uint32_t tick,
    MotorSelection* selection, MotorTrajectory* output,
    MotorLearnerReceipt* receipt = nullptr) {
  if (state == nullptr || motor == nullptr || selection == nullptr || output == nullptr)
    return false;
  if (receipt != nullptr) receipt->state_hash_before = hash_motor_learner(*motor);
  apply_delayed_motor_credit(state, motor, tick, receipt);
  if (motor->committed_valid != 0u) {
    *selection = committed_motor_selection(*state, *motor);
  } else {
    settle_all_motor_predictions(*state, motor, tick, receipt);
    *selection = select_unique_motor_candidate(*state, *motor);
    if (selection->unique != 0u) {
      motor->committed_valid = 1u;
      motor->committed_lane = selection->lane;
    }
  }
  if (selection->unique == 0u) {
    if (receipt != nullptr) ++receipt->abstentions;
    return false;
  }
  const bool emitted = unfold_selected_motor_step(*state, motor, *selection,
                                                   tick, output);
  if (emitted && output->complete != 0u) {
    motor->lanes[selection->lane].active = 0u;
    motor->lanes[selection->lane].resolved = 1u;
    motor->committed_valid = 0u;
    motor->selection_consumed = 1u;
  }
  if (receipt != nullptr) receipt->state_hash_after = hash_motor_learner(*motor);
  return emitted;
}

}  // namespace bcc32::grown_predictive_assembly

#undef BCC32_PM_HD
#undef BCC32_PM_DEVICE
