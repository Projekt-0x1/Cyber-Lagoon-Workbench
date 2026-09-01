#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LOCAL_COORDINATE_ATLAS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LOCAL_COORDINATE_ATLAS_CUH
#include "direct_adult_core_constants.cuh"
#include "direct_network_brain.cuh"

namespace substrate::direct_adult_core {
using direct_network::DirectLocalChartTraceV1;
using direct_network::DirectLocalChartTransitionV1;
using direct_network::DirectLocalCoordinateAtlasV1;
using direct_network::exact_history_fold_word;
using direct_network::kMaxLocalChartTransitions;
using direct_network::kMaxLocalCoordinateCharts;
inline constexpr std::uint32_t kDirectLocalCoordinateAtlasVersion = 1u;

DIRECT_ADULT_HD inline std::int64_t local_chart_map(
    const DirectLocalChartTransitionV1& transition, std::int32_t coordinate_q16) {
  return static_cast<std::int64_t>(transition.orientation) * coordinate_q16 +
      transition.offset_q16;
}
DIRECT_ADULT_HD inline std::uint64_t local_chart_transition_identity(
    const DirectLocalChartTransitionV1& transition) {
  std::uint64_t value = exact_history_fold_word(
      0x6c6f63616c74726eull, transition.source_chart);
  value = exact_history_fold_word(value, transition.target_chart);
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.source_lower_q16));
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.source_upper_q16));
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.target_lower_q16));
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.target_upper_q16));
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.offset_q16));
  value = exact_history_fold_word(
      value, static_cast<std::uint32_t>(transition.orientation));
  return value == 0u ? 1u : value;
}
DIRECT_ADULT_HD inline std::uint64_t local_coordinate_atlas_identity(
    const DirectLocalCoordinateAtlasV1& atlas) {
  std::uint64_t value = exact_history_fold_word(
      0x6c6f63616c61746cull, atlas.version);
  value = exact_history_fold_word(value, atlas.chart_count);
  value = exact_history_fold_word(value, atlas.transition_count);
  for (std::uint32_t i = 0u; i < atlas.chart_count &&
       i < kMaxLocalCoordinateCharts; ++i) {
    const auto& chart = atlas.charts[i];
    value = exact_history_fold_word(value, chart.frame_identity);
    value = exact_history_fold_word(
        value, static_cast<std::uint32_t>(chart.lower_q16));
    value = exact_history_fold_word(
        value, static_cast<std::uint32_t>(chart.upper_q16));
    value = exact_history_fold_word(value, chart.generation);
  }
  for (std::uint32_t i = 0u; i < atlas.transition_count &&
       i < kMaxLocalChartTransitions; ++i)
    value = exact_history_fold_word(
        value, local_chart_transition_identity(atlas.transitions[i]));
  return value == 0u ? 1u : value;
}
DIRECT_ADULT_HD inline bool local_coordinate_atlas_valid(
    const DirectLocalCoordinateAtlasV1& atlas) {
  if (atlas.version != kDirectLocalCoordinateAtlasVersion ||
      atlas.reserved != 0u || atlas.chart_count < 2u ||
      atlas.chart_count > kMaxLocalCoordinateCharts ||
      atlas.transition_count == 0u ||
      atlas.transition_count > kMaxLocalChartTransitions) return false;
  for (std::uint32_t i = 0u; i < atlas.chart_count; ++i) {
    const auto& chart = atlas.charts[i];
    if (chart.frame_identity == 0u || chart.lower_q16 > chart.upper_q16 ||
        chart.reserved != 0u) return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (atlas.charts[prior].frame_identity == chart.frame_identity)
        return false;
  }
  for (std::uint32_t i = 0u; i < atlas.transition_count; ++i) {
    const auto& transition = atlas.transitions[i];
    if (transition.source_chart >= atlas.chart_count ||
        transition.target_chart >= atlas.chart_count ||
        transition.source_chart == transition.target_chart ||
        (transition.orientation != 1 && transition.orientation != -1) ||
        transition.reserved[0] != 0u || transition.reserved[1] != 0u ||
        transition.source_lower_q16 > transition.source_upper_q16 ||
        transition.target_lower_q16 > transition.target_upper_q16 ||
        transition.source_lower_q16 <
            atlas.charts[transition.source_chart].lower_q16 ||
        transition.source_upper_q16 >
            atlas.charts[transition.source_chart].upper_q16 ||
        transition.target_lower_q16 <
            atlas.charts[transition.target_chart].lower_q16 ||
        transition.target_upper_q16 >
            atlas.charts[transition.target_chart].upper_q16 ||
        transition.transition_identity !=
            local_chart_transition_identity(transition)) return false;
    const std::int64_t lower =
        local_chart_map(transition, transition.source_lower_q16);
    const std::int64_t upper =
        local_chart_map(transition, transition.source_upper_q16);
    if (transition.orientation == 1 ?
        (lower != transition.target_lower_q16 ||
         upper != transition.target_upper_q16) :
        (lower != transition.target_upper_q16 ||
         upper != transition.target_lower_q16)) return false;
  }
  return atlas.atlas_identity == local_coordinate_atlas_identity(atlas);
}
DIRECT_ADULT_HD inline bool trace_local_coordinate_atlas(
    const DirectLocalCoordinateAtlasV1& atlas, std::uint32_t start_chart,
    std::int32_t coordinate_q16, DirectLocalChartTraceV1* trace) {
  if (trace == nullptr) return false;
  *trace = {};
  if (!local_coordinate_atlas_valid(atlas) || start_chart >= atlas.chart_count ||
      coordinate_q16 < atlas.charts[start_chart].lower_q16 ||
      coordinate_q16 > atlas.charts[start_chart].upper_q16) {
    trace->refused = 1u;
    return false;
  }
  std::uint32_t chart = start_chart;
  for (std::uint32_t hop = 0u; hop < atlas.chart_count; ++hop) {
    trace->frame_identities[hop] = atlas.charts[chart].frame_identity;
    trace->coordinates_q16[hop] = coordinate_q16;
    trace->count = hop + 1u;
    const DirectLocalChartTransitionV1* selected = nullptr;
    for (std::uint32_t i = 0u; i < atlas.transition_count; ++i) {
      const auto& transition = atlas.transitions[i];
      if (transition.source_chart == chart &&
          coordinate_q16 >= transition.source_lower_q16 &&
          coordinate_q16 <= transition.source_upper_q16) {
        selected = &transition;
        break;
      }
    }
    if (selected == nullptr) return true;
    const std::int64_t mapped = local_chart_map(*selected, coordinate_q16);
    if (mapped < INT32_MIN || mapped > INT32_MAX) return false;
    trace->constraint_residual_q16 = static_cast<std::int32_t>(
        mapped - local_chart_map(*selected, coordinate_q16));
    chart = selected->target_chart;
    coordinate_q16 = static_cast<std::int32_t>(mapped);
    for (std::uint32_t prior = 0u; prior < trace->count; ++prior)
      if (trace->frame_identities[prior] == atlas.charts[chart].frame_identity) {
        trace->refused = 1u;
        return false;
      }
  }
  return true;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_LOCAL_COORDINATE_ATLAS_CUH
