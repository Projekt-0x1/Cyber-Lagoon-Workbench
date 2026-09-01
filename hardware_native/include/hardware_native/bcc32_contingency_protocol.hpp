#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace substrate::bcc32::contingency {

#if defined(__CUDACC__)
#define BCC32_CONTINGENCY_HD __host__ __device__
#else
#define BCC32_CONTINGENCY_HD
#endif

// These records are host manifest data. They are not BCC words and F never
// reads them.
struct ContactCoordinate {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;

  friend constexpr bool operator==(const ContactCoordinate&,
                                   const ContactCoordinate&) = default;
};

struct PhysicalSourceSlot {
  ContactCoordinate coordinate{};
  std::uint8_t channel = 0;
  std::uint32_t tick = 0;

  friend constexpr bool operator==(const PhysicalSourceSlot&,
                                   const PhysicalSourceSlot&) = default;
};

struct PhysicalContact {
  PhysicalSourceSlot source{};
  std::int32_t transfer = 0;
  std::int64_t work = 0;

  friend constexpr bool operator==(const PhysicalContact&,
                                   const PhysicalContact&) = default;
};

enum class HostContactRole : std::uint8_t {
  a,
  b,
  u,
  p,
  idle,
};

// Role and phase are host protocol fields. The physical contact is the complete
// occupied source channel and its declared transfer/work.
struct CommonEventSlot {
  std::uint32_t tick = 0;
  std::uint16_t phase = 0;
  HostContactRole role = HostContactRole::idle;
  PhysicalContact contact{};

  friend constexpr bool operator==(const CommonEventSlot&,
                                   const CommonEventSlot&) = default;
};

struct CommonEventSkeleton {
  std::uint32_t total_duration = 0;
  std::vector<CommonEventSlot> events;
};

enum class HostInitialRegime : std::uint8_t {
  r0 = 0,
  r1 = 1,
};

enum class HostStreamKind : std::uint8_t {
  contingent,
  unpaired,
  shuffled,
};

struct HostRegimePath {
  std::array<HostInitialRegime, 3> blocks{};
};

// Every nominal boundary must sometimes change and sometimes stay, and both
// change directions must occur. Otherwise an elapsed-block toggle can replace
// evidence-dependent adaptation.
[[nodiscard]] inline bool regime_family_blocks_clock(
    std::span<const HostRegimePath> paths) {
  if (paths.empty()) return false;
  for (std::size_t boundary = 0u; boundary + 1u < HostRegimePath{}.blocks.size();
       ++boundary) {
    bool unchanged = false;
    bool r0_to_r1 = false;
    bool r1_to_r0 = false;
    for (const HostRegimePath& path : paths) {
      const HostInitialRegime before = path.blocks[boundary];
      const HostInitialRegime after = path.blocks[boundary + 1u];
      unchanged = unchanged || before == after;
      r0_to_r1 =
          r0_to_r1 || (before == HostInitialRegime::r0 && after == HostInitialRegime::r1);
      r1_to_r0 =
          r1_to_r0 || (before == HostInitialRegime::r1 && after == HostInitialRegime::r0);
    }
    if (!unchanged || !r0_to_r1 || !r1_to_r0) return false;
  }
  return true;
}

struct CandidateContactTrace {
  std::uint32_t candidate = 0;
  std::vector<PhysicalSourceSlot> delivered_sources;
};

// Regime and stream labels belong only in this host-side manifest. The common
// skeleton is intentionally free of outcomes, labels, and BCC representation.
struct HostContingencyManifest {
  HostInitialRegime initial_regime = HostInitialRegime::r0;
  HostStreamKind stream_kind = HostStreamKind::contingent;
  CommonEventSkeleton skeleton;
  std::vector<CandidateContactTrace> candidate_contacts;
};

static_assert(std::is_trivially_copyable_v<ContactCoordinate>);
static_assert(std::is_trivially_copyable_v<PhysicalSourceSlot>);
static_assert(std::is_trivially_copyable_v<PhysicalContact>);
static_assert(std::is_trivially_copyable_v<CommonEventSlot>);

enum class ProtocolValidationError : std::uint8_t {
  none,
  zero_duration,
  source_tick_out_of_range,
  source_event_tick_mismatch,
  channel_out_of_range,
  duplicate_physical_source_slot,
  candidate_dependent_contact_skip,
  total_duration_mismatch,
  event_count_mismatch,
  contact_role_count_mismatch,
  non_u_schedule_mismatch,
  u_tick_mismatch,
  phase_histogram_mismatch,
  gap_multiset_mismatch,
  occupied_channel_transfer_work_mismatch,
};

struct ProtocolValidation {
  ProtocolValidationError error = ProtocolValidationError::none;
  std::uint32_t index = 0;
};

namespace detail {

[[nodiscard]] constexpr bool source_slot_less(const PhysicalSourceSlot& left,
                                               const PhysicalSourceSlot& right) {
  if (left.tick != right.tick) return left.tick < right.tick;
  if (left.coordinate.x != right.coordinate.x) return left.coordinate.x < right.coordinate.x;
  if (left.coordinate.y != right.coordinate.y) return left.coordinate.y < right.coordinate.y;
  if (left.coordinate.z != right.coordinate.z) return left.coordinate.z < right.coordinate.z;
  return left.channel < right.channel;
}

[[nodiscard]] inline std::vector<PhysicalSourceSlot> sorted_sources(
    std::span<const CommonEventSlot> events) {
  std::vector<PhysicalSourceSlot> result;
  result.reserve(events.size());
  for (const CommonEventSlot& event : events) result.push_back(event.contact.source);
  std::sort(result.begin(), result.end(), source_slot_less);
  return result;
}

[[nodiscard]] inline std::vector<std::uint32_t> sorted_u_ticks(
    std::span<const CommonEventSlot> events) {
  std::vector<std::uint32_t> result;
  for (const CommonEventSlot& event : events) {
    if (event.role == HostContactRole::u) result.push_back(event.tick);
  }
  std::sort(result.begin(), result.end());
  return result;
}

[[nodiscard]] inline std::vector<std::pair<std::uint8_t, std::uint16_t>> role_phase_multiset(
    std::span<const CommonEventSlot> events) {
  std::vector<std::pair<std::uint8_t, std::uint16_t>> result;
  result.reserve(events.size());
  for (const CommonEventSlot& event : events)
    result.push_back({static_cast<std::uint8_t>(event.role), event.phase});
  std::sort(result.begin(), result.end());
  return result;
}

[[nodiscard]] inline std::vector<std::uint8_t> sorted_roles(
    std::span<const CommonEventSlot> events) {
  std::vector<std::uint8_t> result;
  result.reserve(events.size());
  for (const CommonEventSlot& event : events)
    result.push_back(static_cast<std::uint8_t>(event.role));
  std::sort(result.begin(), result.end());
  return result;
}

[[nodiscard]] inline std::vector<CommonEventSlot> sorted_non_u_events(
    std::span<const CommonEventSlot> events) {
  std::vector<CommonEventSlot> result;
  for (const CommonEventSlot& event : events) {
    if (event.role != HostContactRole::u) result.push_back(event);
  }
  std::sort(result.begin(), result.end(), [](const CommonEventSlot& left,
                                             const CommonEventSlot& right) {
    return std::tie(left.tick, left.phase, left.role, left.contact.source.coordinate.x,
                    left.contact.source.coordinate.y, left.contact.source.coordinate.z,
                    left.contact.source.channel, left.contact.transfer, left.contact.work) <
           std::tie(right.tick, right.phase, right.role, right.contact.source.coordinate.x,
                    right.contact.source.coordinate.y, right.contact.source.coordinate.z,
                    right.contact.source.channel, right.contact.transfer, right.contact.work);
  });
  return result;
}

[[nodiscard]] inline std::vector<std::uint32_t> gap_multiset(
    const CommonEventSkeleton& skeleton) {
  std::vector<std::uint32_t> ticks;
  ticks.reserve(skeleton.events.size());
  for (const CommonEventSlot& event : skeleton.events) ticks.push_back(event.tick);
  std::sort(ticks.begin(), ticks.end());

  std::vector<std::uint32_t> gaps;
  gaps.reserve(ticks.size() + 1u);
  std::uint32_t previous = 0;
  for (const std::uint32_t tick : ticks) {
    gaps.push_back(tick - previous);
    previous = tick;
  }
  gaps.push_back(skeleton.total_duration - previous);
  std::sort(gaps.begin(), gaps.end());
  return gaps;
}

struct MatchedPhysicalFrame {
  HostContactRole role = HostContactRole::idle;
  ContactCoordinate coordinate{};
  std::uint8_t channel = 0;
  std::int32_t transfer = 0;
  std::int64_t work = 0;

  friend constexpr bool operator==(const MatchedPhysicalFrame&,
                                   const MatchedPhysicalFrame&) = default;
};

[[nodiscard]] inline std::vector<MatchedPhysicalFrame> sorted_physical_frames(
    std::span<const CommonEventSlot> events) {
  std::vector<MatchedPhysicalFrame> result;
  result.reserve(events.size());
  for (const CommonEventSlot& event : events) {
    result.push_back({event.role, event.contact.source.coordinate, event.contact.source.channel,
                      event.contact.transfer, event.contact.work});
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return std::tie(left.role, left.coordinate.x, left.coordinate.y, left.coordinate.z,
                    left.channel, left.transfer, left.work) <
           std::tie(right.role, right.coordinate.x, right.coordinate.y, right.coordinate.z,
                    right.channel, right.transfer, right.work);
  });
  return result;
}

[[nodiscard]] inline std::vector<PhysicalSourceSlot> sorted_trace_sources(
    const CandidateContactTrace& trace) {
  std::vector<PhysicalSourceSlot> result = trace.delivered_sources;
  std::sort(result.begin(), result.end(), source_slot_less);
  return result;
}

[[nodiscard]] constexpr std::uint64_t absolute_difference(std::int64_t left,
                                                           std::int64_t right) {
  const std::uint64_t encoded_left = static_cast<std::uint64_t>(left);
  const std::uint64_t encoded_right = static_cast<std::uint64_t>(right);
  return left >= right ? encoded_left - encoded_right : encoded_right - encoded_left;
}

}  // namespace detail

[[nodiscard]] inline ProtocolValidation validate_manifest(const HostContingencyManifest& manifest) {
  const CommonEventSkeleton& skeleton = manifest.skeleton;
  if (skeleton.total_duration == 0u) return {ProtocolValidationError::zero_duration, 0u};

  const std::vector<PhysicalSourceSlot> sources = detail::sorted_sources(skeleton.events);
  for (std::size_t index = 0; index < skeleton.events.size(); ++index) {
    const CommonEventSlot& event = skeleton.events[index];
    if (event.contact.source.tick >= skeleton.total_duration) {
      return {ProtocolValidationError::source_tick_out_of_range,
              static_cast<std::uint32_t>(index)};
    }
    if (event.contact.source.tick != event.tick) {
      return {ProtocolValidationError::source_event_tick_mismatch,
              static_cast<std::uint32_t>(index)};
    }
    if (event.contact.source.channel >= 8u) {
      return {ProtocolValidationError::channel_out_of_range,
              static_cast<std::uint32_t>(index)};
    }
  }
  for (std::size_t index = 1; index < sources.size(); ++index) {
    if (sources[index - 1u] == sources[index]) {
      return {ProtocolValidationError::duplicate_physical_source_slot,
              static_cast<std::uint32_t>(index)};
    }
  }
  for (std::size_t index = 0; index < manifest.candidate_contacts.size(); ++index) {
    if (detail::sorted_trace_sources(manifest.candidate_contacts[index]) != sources) {
      return {ProtocolValidationError::candidate_dependent_contact_skip,
              static_cast<std::uint32_t>(index)};
    }
  }
  return {};
}

// Compare only physical scheduling invariants. Initial R and stream kind are
// deliberately excluded: controls change those host labels while preserving the
// contact opportunity presented to every candidate.
[[nodiscard]] inline ProtocolValidation validate_matched_protocol(
    const HostContingencyManifest& left, const HostContingencyManifest& right) {
  if (const ProtocolValidation left_validation = validate_manifest(left);
      left_validation.error != ProtocolValidationError::none) {
    return left_validation;
  }
  if (const ProtocolValidation right_validation = validate_manifest(right);
      right_validation.error != ProtocolValidationError::none) {
    return right_validation;
  }

  const CommonEventSkeleton& left_skeleton = left.skeleton;
  const CommonEventSkeleton& right_skeleton = right.skeleton;
  if (left_skeleton.total_duration != right_skeleton.total_duration) {
    return {ProtocolValidationError::total_duration_mismatch, 0u};
  }
  if (left_skeleton.events.size() != right_skeleton.events.size()) {
    return {ProtocolValidationError::event_count_mismatch, 0u};
  }
  if (detail::sorted_roles(left_skeleton.events) != detail::sorted_roles(right_skeleton.events)) {
    return {ProtocolValidationError::contact_role_count_mismatch, 0u};
  }
  if (detail::sorted_non_u_events(left_skeleton.events) !=
      detail::sorted_non_u_events(right_skeleton.events)) {
    return {ProtocolValidationError::non_u_schedule_mismatch, 0u};
  }
  if (detail::sorted_u_ticks(left_skeleton.events) != detail::sorted_u_ticks(right_skeleton.events)) {
    return {ProtocolValidationError::u_tick_mismatch, 0u};
  }
  if (detail::role_phase_multiset(left_skeleton.events) !=
      detail::role_phase_multiset(right_skeleton.events)) {
    return {ProtocolValidationError::phase_histogram_mismatch, 0u};
  }
  if (detail::gap_multiset(left_skeleton) != detail::gap_multiset(right_skeleton)) {
    return {ProtocolValidationError::gap_multiset_mismatch, 0u};
  }
  if (detail::sorted_physical_frames(left_skeleton.events) !=
      detail::sorted_physical_frames(right_skeleton.events)) {
    return {ProtocolValidationError::occupied_channel_transfer_work_mismatch, 0u};
  }
  return {};
}

using IntegerEffect = std::int64_t;

struct IntegerEffectTrace {
  std::vector<IntegerEffect> samples;
};

[[nodiscard]] inline std::optional<std::int64_t> effect_distance(
    std::span<const IntegerEffect> left, std::span<const IntegerEffect> right) {
  if (left.size() != right.size()) return std::nullopt;

  std::int64_t total = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const std::uint64_t difference = detail::absolute_difference(left[index], right[index]);
    if (difference > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() - total)) {
      return std::nullopt;
    }
    total += static_cast<std::int64_t>(difference);
  }
  return total;
}

struct RegimeEffectTraces {
  IntegerEffectTrace ab;
  IntegerEffectTrace ba;
};

struct ContingencyEffectTraces {
  RegimeEffectTraces r0;
  RegimeEffectTraces r1;
};

// C = d(E0AB,E0BA) + d(E1BA,E1AB) - d(E0AB,E1BA) - d(E0BA,E1AB).
[[nodiscard]] inline std::optional<std::int64_t> contingency_contrast(
    const ContingencyEffectTraces& effects) {
  const auto e0_within = effect_distance(effects.r0.ab.samples, effects.r0.ba.samples);
  const auto e1_within = effect_distance(effects.r1.ba.samples, effects.r1.ab.samples);
  const auto ab_across = effect_distance(effects.r0.ab.samples, effects.r1.ba.samples);
  const auto ba_across = effect_distance(effects.r0.ba.samples, effects.r1.ab.samples);
  if (!e0_within || !e1_within || !ab_across || !ba_across) return std::nullopt;

  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  if (*e0_within > kMax - *e1_within || *ab_across > kMax - *ba_across) {
    return std::nullopt;
  }
  const std::int64_t within = *e0_within + *e1_within;
  const std::int64_t across = *ab_across + *ba_across;
  return within >= across ? within - across : -(across - within);
}

struct StageScore {
  std::int64_t control_margin = 0;
  std::int64_t relation_identity_margin = 0;
  std::int64_t contingency_contrast = 0;
};

struct RelationIdentityDistances {
  std::int64_t within_r0 = 0;
  std::int64_t within_r1 = 0;
  std::int64_t cross_ab = 0;
  std::int64_t cross_ba = 0;
};

[[nodiscard]] BCC32_CONTINGENCY_HD constexpr std::int64_t
relation_identity_gap(const RelationIdentityDistances& distances) {
  const std::int64_t largest_cross =
      distances.cross_ab > distances.cross_ba ? distances.cross_ab
                                               : distances.cross_ba;
  const std::int64_t r0 = distances.within_r0 - largest_cross;
  const std::int64_t r1 = distances.within_r1 - largest_cross;
  return r0 < r1 ? r0 : r1;
}

struct MasterLongitudinalTerms {
  std::int64_t acquisition_from_pretraining = 0;
  std::int64_t retention_from_acquisition = 0;
  std::int64_t retention_from_pretraining = 0;
  std::int64_t reversal_from_retention = 0;
  std::int64_t reacquisition_from_acquisition = 0;
  std::int64_t reacquisition_from_reversal = 0;
  std::int64_t omitted_change_drift = 0;
};

struct MasterThresholds {
  std::int64_t minimum_acquisition_change = 0;
  std::int64_t maximum_retention_drift = 0;
  std::int64_t minimum_retained_from_pretraining = 0;
  std::int64_t minimum_reversal_change = 0;
  std::int64_t maximum_reacquisition_drift = 0;
  std::int64_t minimum_reacquisition_from_reversal = 0;
  std::int64_t maximum_omitted_change_drift = 0;
};

struct MasterPassMargins {
  std::array<std::int64_t, 7> values{};

  [[nodiscard]] std::int64_t minimum() const {
    return *std::min_element(values.begin(), values.end());
  }
};

[[nodiscard]] inline std::optional<MasterPassMargins> longitudinal_margins(
    const MasterLongitudinalTerms& observed, const MasterThresholds& thresholds) {
  const std::array<std::int64_t, 13> inputs{
      observed.acquisition_from_pretraining,
      observed.retention_from_acquisition,
      observed.retention_from_pretraining,
      observed.reversal_from_retention,
      observed.reacquisition_from_acquisition,
      observed.reacquisition_from_reversal,
      observed.omitted_change_drift,
      thresholds.minimum_acquisition_change,
      thresholds.maximum_retention_drift,
      thresholds.minimum_retained_from_pretraining,
      thresholds.minimum_reversal_change,
      thresholds.maximum_reacquisition_drift,
      thresholds.minimum_reacquisition_from_reversal,
  };
  if (std::any_of(inputs.begin(), inputs.end(), [](std::int64_t value) { return value < 0; }) ||
      thresholds.maximum_omitted_change_drift < 0) {
    return std::nullopt;
  }
  return MasterPassMargins{{{
      observed.acquisition_from_pretraining - thresholds.minimum_acquisition_change,
      thresholds.maximum_retention_drift - observed.retention_from_acquisition,
      observed.retention_from_pretraining - thresholds.minimum_retained_from_pretraining,
      observed.reversal_from_retention - thresholds.minimum_reversal_change,
      thresholds.maximum_reacquisition_drift - observed.reacquisition_from_acquisition,
      observed.reacquisition_from_reversal -
          thresholds.minimum_reacquisition_from_reversal,
      thresholds.maximum_omitted_change_drift - observed.omitted_change_drift,
  }}};
}

struct RankedScoreTerms {
  std::uint32_t physical_invalidity = 0;
  std::uint32_t pretraining_leakage = 0;
  std::uint32_t hard_gate_failures = 0;
  std::int64_t minimum_longitudinal_margin = std::numeric_limits<std::int64_t>::min();
  std::int64_t minimum_relation_identity_margin =
      std::numeric_limits<std::int64_t>::min();
  std::int64_t minimum_acquisition_from_pretraining =
      std::numeric_limits<std::int64_t>::min();
  std::int64_t maximum_retention_from_acquisition =
      std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_retention_from_pretraining =
      std::numeric_limits<std::int64_t>::min();
  std::int64_t minimum_reversal_from_retention =
      std::numeric_limits<std::int64_t>::min();
  std::int64_t maximum_reacquisition_from_acquisition =
      std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_reacquisition_from_reversal =
      std::numeric_limits<std::int64_t>::min();
  std::int64_t maximum_omitted_change_drift =
      std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_control_margin = std::numeric_limits<std::int64_t>::min();
  std::int64_t minimum_contingency_contrast = std::numeric_limits<std::int64_t>::min();
  std::int64_t viability = 0;
  std::uint32_t genome_bytes = 0;
};

[[nodiscard]] inline RankedScoreTerms ranked_score_terms(
    std::uint32_t physical_invalidity, std::uint32_t pretraining_leakage,
    const MasterThresholds& thresholds, std::span<const MasterLongitudinalTerms> masters,
    std::span<const StageScore> stages, std::int64_t viability, std::uint32_t genome_bytes) {
  RankedScoreTerms result{
      .physical_invalidity = physical_invalidity,
      .pretraining_leakage = pretraining_leakage,
      .viability = viability,
      .genome_bytes = genome_bytes,
  };
  if (!masters.empty()) {
    result.minimum_longitudinal_margin = std::numeric_limits<std::int64_t>::max();
    result.minimum_acquisition_from_pretraining =
        masters.front().acquisition_from_pretraining;
    result.maximum_retention_from_acquisition =
        masters.front().retention_from_acquisition;
    result.minimum_retention_from_pretraining =
        masters.front().retention_from_pretraining;
    result.minimum_reversal_from_retention = masters.front().reversal_from_retention;
    result.maximum_reacquisition_from_acquisition =
        masters.front().reacquisition_from_acquisition;
    result.minimum_reacquisition_from_reversal =
        masters.front().reacquisition_from_reversal;
    result.maximum_omitted_change_drift = masters.front().omitted_change_drift;
    for (const MasterLongitudinalTerms& master : masters) {
      const auto margins = longitudinal_margins(master, thresholds);
      if (!margins) {
        result.minimum_longitudinal_margin = std::numeric_limits<std::int64_t>::min();
        ++result.hard_gate_failures;
      } else if (result.minimum_longitudinal_margin !=
                 std::numeric_limits<std::int64_t>::min()) {
        result.minimum_longitudinal_margin =
            std::min(result.minimum_longitudinal_margin, margins->minimum());
        result.hard_gate_failures += margins->minimum() < 0;
      }
    }
    for (const MasterLongitudinalTerms& master : masters.subspan(1u)) {
      result.minimum_acquisition_from_pretraining =
          std::min(result.minimum_acquisition_from_pretraining,
                   master.acquisition_from_pretraining);
      result.maximum_retention_from_acquisition =
          std::max(result.maximum_retention_from_acquisition,
                   master.retention_from_acquisition);
      result.minimum_retention_from_pretraining =
          std::min(result.minimum_retention_from_pretraining,
                   master.retention_from_pretraining);
      result.minimum_reversal_from_retention =
          std::min(result.minimum_reversal_from_retention, master.reversal_from_retention);
      result.maximum_reacquisition_from_acquisition =
          std::max(result.maximum_reacquisition_from_acquisition,
                   master.reacquisition_from_acquisition);
      result.minimum_reacquisition_from_reversal =
          std::min(result.minimum_reacquisition_from_reversal,
                   master.reacquisition_from_reversal);
      result.maximum_omitted_change_drift =
          std::max(result.maximum_omitted_change_drift, master.omitted_change_drift);
    }
  }
  if (stages.empty()) return result;
  result.minimum_control_margin = stages.front().control_margin;
  result.minimum_relation_identity_margin =
      stages.front().relation_identity_margin;
  result.minimum_contingency_contrast = stages.front().contingency_contrast;
  result.hard_gate_failures += stages.front().control_margin < 0;
  result.hard_gate_failures += stages.front().relation_identity_margin < 0;
  for (const StageScore& stage : stages.subspan(1u)) {
    result.minimum_control_margin = std::min(result.minimum_control_margin, stage.control_margin);
    result.minimum_relation_identity_margin =
        std::min(result.minimum_relation_identity_margin,
                 stage.relation_identity_margin);
    result.minimum_contingency_contrast =
        std::min(result.minimum_contingency_contrast, stage.contingency_contrast);
    result.hard_gate_failures += stage.control_margin < 0;
    result.hard_gate_failures += stage.relation_identity_margin < 0;
  }
  return result;
}

// The order is the complete public search order. Hard failures never trade
// against larger margins; there is intentionally no promotion bit.
[[nodiscard]] constexpr bool ranks_before(const RankedScoreTerms& left,
                                          const RankedScoreTerms& right) {
  if (left.physical_invalidity != right.physical_invalidity) {
    return left.physical_invalidity < right.physical_invalidity;
  }
  if (left.pretraining_leakage != right.pretraining_leakage) {
    return left.pretraining_leakage < right.pretraining_leakage;
  }
  if (left.hard_gate_failures != right.hard_gate_failures) {
    return left.hard_gate_failures < right.hard_gate_failures;
  }
  if (left.minimum_control_margin != right.minimum_control_margin) {
    return left.minimum_control_margin > right.minimum_control_margin;
  }
  if (left.minimum_relation_identity_margin !=
      right.minimum_relation_identity_margin) {
    return left.minimum_relation_identity_margin >
           right.minimum_relation_identity_margin;
  }
  if (left.minimum_longitudinal_margin != right.minimum_longitudinal_margin) {
    return left.minimum_longitudinal_margin > right.minimum_longitudinal_margin;
  }
  if (left.minimum_contingency_contrast != right.minimum_contingency_contrast) {
    return left.minimum_contingency_contrast > right.minimum_contingency_contrast;
  }
  if (left.viability != right.viability) return left.viability > right.viability;
  return left.genome_bytes < right.genome_bytes;
}

}  // namespace substrate::bcc32::contingency

#undef BCC32_CONTINGENCY_HD
