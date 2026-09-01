#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "bcc32_law.cuh"
#include "hardware_native/bcc32_contingency_promotion.hpp"

namespace substrate::bcc32::contingency::replay {

inline constexpr std::string_view kGenomeSchema = "bcc32-g1-online-contingency-genome-v1\n";
inline constexpr std::size_t kGenomeHelperCount = 48u;
inline constexpr std::size_t kCanonicalGenomeBytes =
    kGenomeSchema.size() + kGenomeHelperCount * sizeof(SiteWord);

struct Genome {
  std::array<SiteWord, kGenomeHelperCount> helpers{};

  friend bool operator==(const Genome&, const Genome&) = default;
  friend auto operator<=>(const Genome&, const Genome&) = default;
};

enum class GenomeValidationError : std::uint8_t {
  none = 0,
  byte_count,
  schema,
  word,
};

struct GenomeParseResult {
  Genome value{};
  GenomeValidationError error = GenomeValidationError::none;
  std::uint32_t helper_index = std::numeric_limits<std::uint32_t>::max();

  [[nodiscard]] constexpr explicit operator bool() const {
    return error == GenomeValidationError::none;
  }
};

[[nodiscard]] constexpr bool candidate_word(SiteWord word) {
  if (word == kQ)
    return true;
  return std::popcount(kQ & ~word) == 1 && std::popcount(word & ~kCarrierMask) == 1;
}

[[nodiscard]] inline std::array<std::byte, kCanonicalGenomeBytes> canonical_genome_bytes(
    const Genome& genome) {
  std::array<std::byte, kCanonicalGenomeBytes> result{};
  for (std::size_t index = 0; index < kGenomeSchema.size(); ++index)
    result[index] = static_cast<std::byte>(kGenomeSchema[index]);
  std::size_t offset = kGenomeSchema.size();
  for (const SiteWord word : genome.helpers) {
    for (std::uint32_t byte = 0; byte < sizeof(SiteWord); ++byte)
      result[offset++] = static_cast<std::byte>((word >> (8u * byte)) & 0xffu);
  }
  return result;
}

[[nodiscard]] inline GenomeParseResult parse_canonical_genome(std::span<const std::byte> bytes) {
  if (bytes.size() != kCanonicalGenomeBytes)
    return {.error = GenomeValidationError::byte_count};
  for (std::size_t index = 0; index < kGenomeSchema.size(); ++index) {
    if (bytes[index] != static_cast<std::byte>(kGenomeSchema[index]))
      return {.error = GenomeValidationError::schema};
  }
  Genome result{};
  std::size_t offset = kGenomeSchema.size();
  for (std::size_t index = 0; index < result.helpers.size(); ++index) {
    SiteWord word = 0u;
    for (std::uint32_t byte = 0; byte < sizeof(SiteWord); ++byte)
      word |= static_cast<SiteWord>(std::to_integer<std::uint8_t>(bytes[offset++])) << (8u * byte);
    if (!candidate_word(word)) {
      return {.error = GenomeValidationError::word,
              .helper_index = static_cast<std::uint32_t>(index)};
    }
    result.helpers[index] = word;
  }
  return {.value = result};
}

enum class ReplayKind : std::uint8_t {
  observed_master = 0,
  matched_control,
  randomized_null,
  named_null,
  target_lesion,
  matched_lesion,
  rescue,
  transplant,
  transplant_control,
};

enum class NamedNull : std::uint8_t {
  fresh_adult = 0,
  consequence_only,
  event_code_0_only,
  event_code_1_only,
  order_00,
  order_11,
  idle,
  probe_only,
  body_absent,
  wrong_channel,
  matched_lesion,
};

inline constexpr std::array<NamedNull, promotion::kNamedNullFamilies> kNamedNulls{
    NamedNull::fresh_adult,
    NamedNull::consequence_only,
    NamedNull::event_code_0_only,
    NamedNull::event_code_1_only,
    NamedNull::order_00,
    NamedNull::order_11,
    NamedNull::idle,
    NamedNull::probe_only,
    NamedNull::body_absent,
    NamedNull::wrong_channel,
    NamedNull::matched_lesion};

struct ReplayRequest {
  std::uint32_t case_index = 0;
  std::uint8_t master_index = 0;
  ReplayKind kind = ReplayKind::observed_master;
  std::uint16_t variant_index = 0;
  promotion::DerivedMasterSchedule schedule{};
  bool fresh_adult = false;
  bool probe_only = false;
  bool body_absent = false;
  bool rotate_contact_channels = false;
  // A nonzero mask marks an observer-side reduction over already executed
  // lesion-family forks. It does not authorize another physical trajectory.
  std::uint8_t aggregate_lesion_family_mask = 0u;
};

struct CaseReplayPlan {
  std::uint32_t case_index = 0;
  promotion::DerivedCaseSchedule derived{};
  std::vector<ReplayRequest> requests;
};

struct PromotionReplayPlan {
  std::vector<CaseReplayPlan> cases;
  std::uint64_t logical_requests = 0;
};

[[nodiscard]] inline promotion::DerivedMasterSchedule filtered_schedule(
    const promotion::DerivedMasterSchedule& source, NamedNull transformation) {
  promotion::DerivedMasterSchedule result;
  result.events.reserve(source.events.size());
  for (promotion::DerivedEvent event : source.events) {
    bool keep = true;
    switch (transformation) {
      case NamedNull::fresh_adult:
      case NamedNull::idle:
      case NamedNull::probe_only:
        keep = false;
        break;
      case NamedNull::consequence_only:
        keep = event.event_code == 2u;
        break;
      case NamedNull::event_code_0_only:
        keep = event.event_code == 0u;
        break;
      case NamedNull::event_code_1_only:
        keep = event.event_code == 1u;
        break;
      case NamedNull::order_00:
        if (event.event_code < 2u)
          event.event_code = 0u;
        break;
      case NamedNull::order_11:
        if (event.event_code < 2u)
          event.event_code = 1u;
        break;
      case NamedNull::body_absent:
      case NamedNull::wrong_channel:
      case NamedNull::matched_lesion:
        break;
    }
    if (keep)
      result.events.push_back(event);
  }
  return result;
}

[[nodiscard]] inline std::optional<promotion::DerivedMasterSchedule> randomized_null_schedule(
    const promotion::PromotionCase& definition, const promotion::DerivedCaseSchedule& derived,
    std::uint8_t master_index, std::uint32_t replicate_index) {
  const auto masks = promotion::randomized_null_masks(definition, master_index, replicate_index);
  if (!masks || master_index >= promotion::kMasterCount)
    return std::nullopt;
  promotion::DerivedMasterSchedule result = derived.masters[master_index];
  for (promotion::DerivedEvent& event : result.events) {
    if (event.event_code == 2u) {
      event.lane = ((*masks)[event.block] & (1u << event.trial)) != 0u ? 0u : 1u;
    }
  }
  return result;
}

[[nodiscard]] inline std::optional<CaseReplayPlan> derive_case_replay_plan(
    const promotion::PromotionCase& definition) {
  const auto derived = promotion::derive_case_schedule(definition);
  if (!derived)
    return std::nullopt;
  CaseReplayPlan result{.case_index = definition.index, .derived = *derived};
  constexpr std::size_t kRequestsPerCase =
      promotion::kMasterCount + promotion::kMatchedControlFamilies * promotion::kMasterCount +
      promotion::kRandomizedNullReplicates * promotion::kMasterCount +
      promotion::kNamedNullFamilies * promotion::kMasterCount +
      promotion::kLesionFamilies * promotion::kMasterCount * 5u;
  result.requests.reserve(kRequestsPerCase);

  for (std::uint8_t master = 0; master < promotion::kMasterCount; ++master) {
    result.requests.push_back({.case_index = definition.index,
                               .master_index = master,
                               .kind = ReplayKind::observed_master,
                               .schedule = derived->masters[master]});
  }
  for (std::uint16_t family = 0; family < promotion::kMatchedControlFamilies; ++family) {
    for (std::uint8_t master = 0; master < promotion::kMasterCount; ++master) {
      result.requests.push_back({.case_index = definition.index,
                                 .master_index = master,
                                 .kind = ReplayKind::matched_control,
                                 .variant_index = family,
                                 .schedule = derived->comparison_controls[family][master]});
    }
  }
  for (std::uint16_t replicate = 0; replicate < promotion::kRandomizedNullReplicates; ++replicate) {
    for (std::uint8_t master = 0; master < promotion::kMasterCount; ++master) {
      auto schedule = randomized_null_schedule(definition, *derived, master, replicate);
      if (!schedule)
        return std::nullopt;
      result.requests.push_back({.case_index = definition.index,
                                 .master_index = master,
                                 .kind = ReplayKind::randomized_null,
                                 .variant_index = replicate,
                                 .schedule = std::move(*schedule)});
    }
  }
  for (std::uint16_t family = 0; family < kNamedNulls.size(); ++family) {
    const NamedNull transformation = kNamedNulls[family];
    for (std::uint8_t master = 0; master < promotion::kMasterCount; ++master) {
      result.requests.push_back({
          .case_index = definition.index,
          .master_index = master,
          .kind = ReplayKind::named_null,
          .variant_index = family,
          .schedule = filtered_schedule(derived->masters[master], transformation),
          .fresh_adult = transformation == NamedNull::fresh_adult,
          .probe_only = transformation == NamedNull::probe_only,
          .body_absent = transformation == NamedNull::body_absent,
          .rotate_contact_channels = transformation == NamedNull::wrong_channel,
          .aggregate_lesion_family_mask = static_cast<std::uint8_t>(
              transformation == NamedNull::matched_lesion ? (1u << promotion::kLesionFamilies) - 1u
                                                          : 0u),
      });
    }
  }
  constexpr std::array<ReplayKind, 5> kLesionForks{
      ReplayKind::target_lesion, ReplayKind::matched_lesion, ReplayKind::rescue,
      ReplayKind::transplant, ReplayKind::transplant_control};
  for (std::uint16_t family = 0; family < promotion::kLesionFamilies; ++family) {
    for (std::uint8_t master = 0; master < promotion::kMasterCount; ++master) {
      for (const ReplayKind kind : kLesionForks) {
        result.requests.push_back({.case_index = definition.index,
                                   .master_index = master,
                                   .kind = kind,
                                   .variant_index = family,
                                   .schedule = derived->masters[master]});
      }
    }
  }
  if (result.requests.size() != kRequestsPerCase)
    return std::nullopt;
  return result;
}

[[nodiscard]] inline std::optional<PromotionReplayPlan> derive_replay_plan(
    const promotion::PromotionCaseSet& definitions) {
  if (definitions.cases.size() != promotion::kCaseCount)
    return std::nullopt;
  PromotionReplayPlan result;
  result.cases.reserve(definitions.cases.size());
  for (std::size_t index = 0; index < definitions.cases.size(); ++index) {
    if (definitions.cases[index].index != index)
      return std::nullopt;
    auto plan = derive_case_replay_plan(definitions.cases[index]);
    if (!plan)
      return std::nullopt;
    for (const ReplayRequest& request : plan->requests) {
      ++result.logical_requests;
    }
    result.cases.push_back(std::move(*plan));
  }
  return result;
}

struct ResidentBitTransition {
  promotion::LesionBitKey key{};
  bool pretraining_value = false;
  bool acquisition_value = false;
};

struct LesionPartition {
  std::array<std::vector<ResidentBitTransition>, promotion::kLesionFamilies> targets;
  std::array<std::vector<ResidentBitTransition>, promotion::kLesionFamilies> matched;
};

[[nodiscard]] inline std::optional<LesionPartition> derive_lesion_partition(
    std::span<const ResidentBitTransition> transitions) {
  LesionPartition result;
  std::vector<ResidentBitTransition> changed;
  changed.reserve(transitions.size());
  std::set<std::tuple<std::uint32_t, std::uint8_t, std::int64_t, std::int64_t, std::int64_t,
                      std::uint8_t>>
      keys;
  for (const ResidentBitTransition& transition : transitions) {
    if (transition.key.bit_index < 8u || transition.key.bit_index >= 32u)
      return std::nullopt;
    if (!keys.emplace(transition.key.case_index, transition.key.master_index, transition.key.x,
                      transition.key.y, transition.key.z, transition.key.bit_index)
             .second) {
      return std::nullopt;
    }
    if (transition.pretraining_value != transition.acquisition_value) {
      changed.push_back(transition);
      result.targets[promotion::lesion_family(transition.key)].push_back(transition);
    }
  }
  for (std::size_t family = 0; family < result.targets.size(); ++family) {
    if (result.targets[family].empty())
      return std::nullopt;
    std::vector<ResidentBitTransition> eligible;
    for (const ResidentBitTransition& transition : changed) {
      if (promotion::lesion_family(transition.key) != family)
        eligible.push_back(transition);
    }
    std::sort(eligible.begin(), eligible.end(),
              [](const ResidentBitTransition& left, const ResidentBitTransition& right) {
                return promotion::matched_lesion_less(left.key, right.key);
              });
    if (eligible.size() < result.targets[family].size())
      return std::nullopt;
    result.matched[family].assign(eligible.begin(),
                                  eligible.begin() + result.targets[family].size());
  }
  return result;
}

[[nodiscard]] inline std::uint64_t schedule_fingerprint(
    const promotion::DerivedMasterSchedule& schedule) {
  std::uint64_t hash = 14'695'981'039'346'656'037ull;
  auto append = [&](std::uint8_t value) {
    hash ^= value;
    hash *= 1'099'511'628'211ull;
  };
  for (const promotion::DerivedEvent& event : schedule.events) {
    for (std::uint32_t shift = 0; shift < 32u; shift += 8u)
      append(static_cast<std::uint8_t>(event.tick >> shift));
    append(event.lane);
    append(event.event_code);
    append(event.block);
    append(event.trial);
  }
  return hash;
}

}  // namespace substrate::bcc32::contingency::replay
