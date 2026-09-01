#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace substrate::bcc32::contingency::promotion {

inline constexpr std::string_view kCaseSchema = "bcc32-g1-online-contingency-promotion-cases-v1";
inline constexpr std::size_t kCaseCount = 24u;
inline constexpr std::size_t kMasterCount = 2u;
inline constexpr std::size_t kBlockCount = 3u;
inline constexpr std::size_t kTrialsPerBlock = 4u;
inline constexpr std::size_t kRequiredStages = 4u;
inline constexpr std::size_t kProbeClasses = 2u;
inline constexpr std::size_t kMatchedControlFamilies = 4u;
inline constexpr std::size_t kRelationIdentityComponents = kRequiredStages;
inline constexpr std::size_t kMatchedControlComponents =
    kRequiredStages * kProbeClasses * kMatchedControlFamilies;
inline constexpr std::size_t kNamedNullFamilies = 11u;
inline constexpr std::size_t kLesionFamilies = 4u;
inline constexpr std::size_t kRandomizedNullReplicates = 99u;
inline constexpr std::size_t kProbeSampleCount = 6u;
inline constexpr std::size_t kMaximumPlaintextBytes = 1'048'576u;
inline constexpr std::size_t kMaximumJsonNesting = 64u;
inline constexpr std::uint64_t kRandomizedNullSeed = 7'640'891'576'956'012'809ull;

struct CaseTiming {
  std::uint32_t pre_delay_ticks = 0;
  std::uint32_t cue_spacing_ticks = 0;
  std::uint32_t consequence_spacing_ticks = 0;
  std::uint32_t inter_trial_spacing_ticks = 0;
  std::uint32_t inter_block_rest_ticks = 0;
  std::uint32_t retention_rest_ticks = 0;
  std::uint32_t probe_spacing_ticks = 0;

  friend constexpr bool operator==(const CaseTiming&, const CaseTiming&) = default;
};

struct PromotionCase {
  std::uint32_t index = 0;
  CaseTiming timing{};
  std::uint8_t orientation_mask = 0;
  std::array<std::uint8_t, 2> control_masks{};
  std::array<std::array<std::uint8_t, kBlockCount>, kMasterCount> relation_paths{};
};

struct PromotionCaseSet {
  std::vector<PromotionCase> cases;
};

enum class CaseValidationError : std::uint8_t {
  none = 0,
  plaintext_size,
  json_syntax,
  duplicate_key,
  root_type,
  missing_field,
  unexpected_field,
  field_type,
  schema_mismatch,
  case_count,
  case_index,
  timing_range,
  timing_overflow,
  timing_family_not_variable,
  mask_range,
  mask_balance,
  control_mask_relation,
  relation_value,
  relation_complement,
  relation_path_degenerate,
  orientation_family_not_complement_closed,
  clock_family_not_blocked,
};

struct CaseValidation {
  CaseValidationError error = CaseValidationError::none;
  std::size_t byte_offset = 0;
  std::uint32_t case_index = std::numeric_limits<std::uint32_t>::max();

  [[nodiscard]] constexpr explicit operator bool() const {
    return error == CaseValidationError::none;
  }
};

struct CaseParseResult {
  PromotionCaseSet value{};
  CaseValidation validation{};

  [[nodiscard]] constexpr explicit operator bool() const { return static_cast<bool>(validation); }
};

// Event codes are deliberately numeric physical slots: 0 and 1 are the two
// contact-frame variants, and 2 is the consequence-frame variant.
struct DerivedEvent {
  std::uint32_t tick = 0;
  std::uint8_t lane = 0;
  std::uint8_t event_code = 0;
  std::uint8_t block = 0;
  std::uint8_t trial = 0;

  friend constexpr bool operator==(const DerivedEvent&, const DerivedEvent&) = default;
};

struct DerivedMasterSchedule {
  std::vector<DerivedEvent> events;
};

struct DerivedCaseSchedule {
  // pretraining, acquisition, retention, reversal, reacquisition
  std::array<std::uint32_t, 5> assay_ticks{};
  std::array<std::uint32_t, kBlockCount> block_start_ticks{};
  std::array<std::array<std::uint32_t, kTrialsPerBlock>, kBlockCount>
      acquisition_curve_sample_ticks{};
  std::array<std::uint32_t, 3> probe_contact_ticks{};
  std::array<std::uint32_t, kProbeSampleCount> probe_sample_ticks{};
  std::array<DerivedMasterSchedule, 6> masters{};
  // Four comparison families, each with both complement-path copies. Family
  // order is unpaired, derived yoked, no-consequence, shuffled.
  std::array<std::array<DerivedMasterSchedule, kMasterCount>, kMatchedControlFamilies>
      comparison_controls{};
  std::uint32_t terminal_tick = 0;
};

struct LesionBitKey {
  std::uint32_t case_index = 0;
  std::uint8_t master_index = 0;
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;
  std::uint8_t bit_index = 0;
};

namespace detail {

enum class JsonKind : std::uint8_t { null_value, boolean, unsigned_integer, string, array, object };

struct JsonValue {
  JsonKind kind = JsonKind::null_value;
  bool boolean = false;
  std::uint64_t unsigned_integer = 0;
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  [[nodiscard]] std::optional<JsonValue> parse() {
    skip_space();
    auto value = parse_value();
    if (!value)
      return std::nullopt;
    skip_space();
    if (position_ != input_.size()) {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    return value;
  }

  [[nodiscard]] CaseValidation validation() const { return validation_; }

 private:
  void skip_space() {
    while (position_ < input_.size()) {
      const char c = input_[position_];
      if (c != ' ' && c != '\n' && c != '\r' && c != '\t')
        break;
      ++position_;
    }
  }

  void fail(CaseValidationError error) {
    if (validation_.error == CaseValidationError::none) {
      validation_ = {.error = error, .byte_offset = position_};
    }
  }

  [[nodiscard]] bool consume(char expected) {
    skip_space();
    if (position_ >= input_.size() || input_[position_] != expected) {
      fail(CaseValidationError::json_syntax);
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] std::optional<std::string> parse_string() {
    skip_space();
    if (position_ >= input_.size() || input_[position_] != '"') {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    ++position_;
    std::string result;
    while (position_ < input_.size()) {
      const unsigned char c = static_cast<unsigned char>(input_[position_++]);
      if (c == '"')
        return result;
      // The committed v1 format has a bounded ASCII spelling. Escapes and
      // non-ASCII spellings are rejected before schema interpretation.
      if (c < 0x20u || c > 0x7eu || c == '\\') {
        fail(CaseValidationError::json_syntax);
        return std::nullopt;
      }
      result.push_back(static_cast<char>(c));
    }
    fail(CaseValidationError::json_syntax);
    return std::nullopt;
  }

  [[nodiscard]] std::optional<JsonValue> parse_unsigned_integer() {
    skip_space();
    const std::size_t start = position_;
    if (position_ >= input_.size() ||
        !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    if (input_[position_] == '0' && position_ + 1u < input_.size() &&
        std::isdigit(static_cast<unsigned char>(input_[position_ + 1u]))) {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    std::uint64_t value = 0;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      const std::uint64_t digit = static_cast<std::uint64_t>(input_[position_] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10u) {
        fail(CaseValidationError::json_syntax);
        return std::nullopt;
      }
      value = value * 10u + digit;
      ++position_;
    }
    if (position_ == start ||
        (position_ < input_.size() &&
         (input_[position_] == '.' || input_[position_] == 'e' || input_[position_] == 'E'))) {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    JsonValue result;
    result.kind = JsonKind::unsigned_integer;
    result.unsigned_integer = value;
    return result;
  }

  [[nodiscard]] std::optional<JsonValue> parse_array(std::size_t depth) {
    if (!consume('['))
      return std::nullopt;
    JsonValue result;
    result.kind = JsonKind::array;
    skip_space();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_;
      return result;
    }
    while (true) {
      auto value = parse_value(depth);
      if (!value)
        return std::nullopt;
      result.array.push_back(std::move(*value));
      skip_space();
      if (position_ < input_.size() && input_[position_] == ']') {
        ++position_;
        return result;
      }
      if (!consume(','))
        return std::nullopt;
    }
  }

  [[nodiscard]] std::optional<JsonValue> parse_object(std::size_t depth) {
    if (!consume('{'))
      return std::nullopt;
    JsonValue result;
    result.kind = JsonKind::object;
    skip_space();
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_;
      return result;
    }
    while (true) {
      auto key = parse_string();
      if (!key || !consume(':'))
        return std::nullopt;
      if (std::any_of(result.object.begin(), result.object.end(),
                      [&](const auto& member) { return member.first == *key; })) {
        fail(CaseValidationError::duplicate_key);
        return std::nullopt;
      }
      auto value = parse_value(depth);
      if (!value)
        return std::nullopt;
      result.object.emplace_back(std::move(*key), std::move(*value));
      skip_space();
      if (position_ < input_.size() && input_[position_] == '}') {
        ++position_;
        return result;
      }
      if (!consume(','))
        return std::nullopt;
    }
  }

  [[nodiscard]] bool consume_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      fail(CaseValidationError::json_syntax);
      return false;
    }
    position_ += literal.size();
    return true;
  }

  [[nodiscard]] std::optional<JsonValue> parse_value(std::size_t depth = 0u) {
    skip_space();
    if (position_ >= input_.size()) {
      fail(CaseValidationError::json_syntax);
      return std::nullopt;
    }
    switch (input_[position_]) {
      case '{':
        if (depth >= kMaximumJsonNesting) {
          fail(CaseValidationError::json_syntax);
          return std::nullopt;
        }
        return parse_object(depth + 1u);
      case '[':
        if (depth >= kMaximumJsonNesting) {
          fail(CaseValidationError::json_syntax);
          return std::nullopt;
        }
        return parse_array(depth + 1u);
      case '"': {
        auto value = parse_string();
        if (!value)
          return std::nullopt;
        JsonValue result;
        result.kind = JsonKind::string;
        result.string = std::move(*value);
        return result;
      }
      case 't': {
        if (!consume_literal("true"))
          return std::nullopt;
        JsonValue result;
        result.kind = JsonKind::boolean;
        result.boolean = true;
        return result;
      }
      case 'f': {
        if (!consume_literal("false"))
          return std::nullopt;
        JsonValue result;
        result.kind = JsonKind::boolean;
        return result;
      }
      case 'n': {
        if (!consume_literal("null"))
          return std::nullopt;
        return JsonValue{};
      }
      default:
        return parse_unsigned_integer();
    }
  }

  std::string_view input_;
  std::size_t position_ = 0;
  CaseValidation validation_{};
};

[[nodiscard]] inline const JsonValue* member(const JsonValue& object, std::string_view key) {
  if (object.kind != JsonKind::object)
    return nullptr;
  for (const auto& entry : object.object) {
    if (entry.first == key)
      return &entry.second;
  }
  return nullptr;
}

template <std::size_t N>
[[nodiscard]] inline CaseValidation exact_keys(
    const JsonValue& object, const std::array<std::string_view, N>& expected,
    std::uint32_t case_index = std::numeric_limits<std::uint32_t>::max()) {
  if (object.kind != JsonKind::object) {
    return {.error = CaseValidationError::field_type, .case_index = case_index};
  }
  for (const std::string_view key : expected) {
    if (member(object, key) == nullptr) {
      return {.error = CaseValidationError::missing_field, .case_index = case_index};
    }
  }
  for (const auto& entry : object.object) {
    if (std::find(expected.begin(), expected.end(), entry.first) == expected.end()) {
      return {.error = CaseValidationError::unexpected_field, .case_index = case_index};
    }
  }
  return {};
}

[[nodiscard]] inline std::optional<std::uint32_t> uint32_value(const JsonValue* value) {
  if (value == nullptr || value->kind != JsonKind::unsigned_integer ||
      value->unsigned_integer > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(value->unsigned_integer);
}

[[nodiscard]] constexpr bool checked_add(std::uint64_t left, std::uint64_t right,
                                         std::uint64_t* result) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right)
    return false;
  *result = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                              std::uint64_t* result) {
  if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

[[nodiscard]] inline bool timing_in_range(const CaseTiming& timing) {
  constexpr std::uint32_t kMaximumFieldTicks = 1'000'000u;
  return timing.pre_delay_ticks <= kMaximumFieldTicks && timing.cue_spacing_ticks >= 1u &&
         timing.cue_spacing_ticks <= kMaximumFieldTicks && timing.consequence_spacing_ticks >= 1u &&
         timing.consequence_spacing_ticks <= kMaximumFieldTicks &&
         timing.inter_trial_spacing_ticks >= 1u &&
         timing.inter_trial_spacing_ticks <= kMaximumFieldTicks &&
         timing.inter_block_rest_ticks >= 1u &&
         timing.inter_block_rest_ticks <= kMaximumFieldTicks && timing.retention_rest_ticks >= 1u &&
         timing.retention_rest_ticks <= kMaximumFieldTicks && timing.probe_spacing_ticks >= 1u &&
         timing.probe_spacing_ticks <= kMaximumFieldTicks;
}

[[nodiscard]] constexpr std::uint32_t unsigned_distance(std::uint32_t left, std::uint32_t right) {
  return left >= right ? left - right : right - left;
}

[[nodiscard]] constexpr std::uint64_t splitmix64_next(std::uint64_t* state) {
  *state += 0x9e3779b97f4a7c15ull;
  std::uint64_t value = *state;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

constexpr void fnv1a_byte(std::uint64_t* hash, std::uint8_t byte) {
  *hash ^= byte;
  *hash *= 1'099'511'628'211ull;
}

template <typename Unsigned>
constexpr void fnv1a_little_endian(std::uint64_t* hash, Unsigned value) {
  static_assert(std::is_unsigned_v<Unsigned>);
  for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
    fnv1a_byte(hash, static_cast<std::uint8_t>(value & 0xffu));
    value >>= 8u;
  }
}

[[nodiscard]] constexpr std::uint64_t lesion_hash(const LesionBitKey& key, bool matched_order) {
  std::uint64_t hash = 14'695'981'039'346'656'037ull;
  fnv1a_little_endian(&hash, key.case_index);
  fnv1a_byte(&hash, key.master_index);
  fnv1a_little_endian(&hash, static_cast<std::uint64_t>(key.x));
  fnv1a_little_endian(&hash, static_cast<std::uint64_t>(key.y));
  fnv1a_little_endian(&hash, static_cast<std::uint64_t>(key.z));
  fnv1a_byte(&hash, key.bit_index);
  if (matched_order)
    fnv1a_byte(&hash, 77u);
  return hash;
}

}  // namespace detail

[[nodiscard]] inline std::optional<std::array<std::uint8_t, kBlockCount>> randomized_null_masks(
    const PromotionCase& definition, std::uint8_t master_index, std::uint32_t replicate_index) {
  if (definition.index >= kCaseCount || definition.orientation_mask > 0x0fu ||
      std::popcount(definition.orientation_mask) != 2 || master_index >= kMasterCount ||
      replicate_index >= kRandomizedNullReplicates) {
    return std::nullopt;
  }
  constexpr std::array<std::uint8_t, 6> kBalancedMasks{3u, 5u, 6u, 9u, 10u, 12u};
  std::array<std::uint8_t, 4> eligible{};
  std::size_t eligible_count = 0;
  for (const std::uint8_t mask : kBalancedMasks) {
    if (std::popcount(static_cast<std::uint8_t>(mask ^ definition.orientation_mask)) == 2) {
      eligible[eligible_count++] = mask;
    }
  }
  if (eligible_count != eligible.size()) {
    return std::nullopt;
  }
  std::uint64_t state =
      kRandomizedNullSeed ^ (static_cast<std::uint64_t>(definition.index) * 0x9e3779b97f4a7c15ull) ^
      (static_cast<std::uint64_t>(master_index) * 0xbf58476d1ce4e5b9ull) ^ replicate_index;
  std::array<std::uint8_t, kBlockCount> result{};
  for (std::uint8_t& mask : result) {
    mask = eligible[detail::splitmix64_next(&state) % eligible.size()];
  }
  return result;
}

[[nodiscard]] constexpr std::uint8_t lesion_family(const LesionBitKey& key) {
  return static_cast<std::uint8_t>(detail::lesion_hash(key, false) & 3u);
}

[[nodiscard]] constexpr std::uint64_t matched_lesion_order(const LesionBitKey& key) {
  return detail::lesion_hash(key, true);
}

[[nodiscard]] constexpr bool matched_lesion_less(const LesionBitKey& left,
                                                 const LesionBitKey& right) {
  const std::uint64_t left_hash = matched_lesion_order(left);
  const std::uint64_t right_hash = matched_lesion_order(right);
  if (left_hash != right_hash)
    return left_hash < right_hash;
  return std::tie(left.x, left.y, left.z, left.bit_index, left.case_index, left.master_index) <
         std::tie(right.x, right.y, right.z, right.bit_index, right.case_index, right.master_index);
}

[[nodiscard]] inline std::optional<DerivedCaseSchedule> derive_case_schedule(
    const PromotionCase& definition) {
  if (definition.index >= kCaseCount || !detail::timing_in_range(definition.timing) ||
      definition.orientation_mask > 0x0fu || std::popcount(definition.orientation_mask) != 2 ||
      definition.control_masks[0] == definition.control_masks[1])
    return std::nullopt;
  for (const std::uint8_t mask : definition.control_masks) {
    if (mask > 0x0fu || std::popcount(mask) != 2 ||
        std::popcount(static_cast<std::uint8_t>(mask ^ definition.orientation_mask)) != 2)
      return std::nullopt;
  }
  bool path_changes = false;
  for (std::size_t block = 0; block < kBlockCount; ++block) {
    if (definition.relation_paths[0][block] > 1u ||
        definition.relation_paths[1][block] !=
            static_cast<std::uint8_t>(1u - definition.relation_paths[0][block]))
      return std::nullopt;
    if (block > 0u &&
        definition.relation_paths[0][block] != definition.relation_paths[0][block - 1u])
      path_changes = true;
  }
  if (!path_changes)
    return std::nullopt;

  const CaseTiming& timing = definition.timing;
  std::uint64_t trial_stride = 0;
  std::uint64_t block_duration = 0;
  if (!detail::checked_add(timing.cue_spacing_ticks, timing.consequence_spacing_ticks,
                           &trial_stride) ||
      !detail::checked_add(trial_stride, timing.inter_trial_spacing_ticks, &trial_stride) ||
      !detail::checked_multiply(trial_stride, kTrialsPerBlock, &block_duration)) {
    return std::nullopt;
  }

  std::array<std::uint64_t, kBlockCount> starts{};
  std::array<std::uint64_t, 5> assays{};
  starts[0] = timing.pre_delay_ticks;
  if (!detail::checked_add(starts[0], block_duration, &assays[1]) ||
      !detail::checked_add(assays[1], timing.retention_rest_ticks, &assays[2]) ||
      !detail::checked_add(assays[2], timing.inter_block_rest_ticks, &starts[1]) ||
      !detail::checked_add(starts[1], block_duration, &assays[3]) ||
      !detail::checked_add(assays[3], timing.inter_block_rest_ticks, &starts[2]) ||
      !detail::checked_add(starts[2], block_duration, &assays[4])) {
    return std::nullopt;
  }
  assays[0] = 0u;
  std::uint64_t terminal = 0;
  if (!detail::checked_add(assays[4], timing.inter_block_rest_ticks, &terminal) ||
      terminal > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }

  DerivedCaseSchedule result;
  result.terminal_tick = static_cast<std::uint32_t>(terminal);
  for (std::size_t index = 0; index < assays.size(); ++index) {
    if (assays[index] > std::numeric_limits<std::uint32_t>::max())
      return std::nullopt;
    result.assay_ticks[index] = static_cast<std::uint32_t>(assays[index]);
  }
  for (std::size_t block = 0; block < starts.size(); ++block) {
    if (starts[block] > std::numeric_limits<std::uint32_t>::max())
      return std::nullopt;
    result.block_start_ticks[block] = static_cast<std::uint32_t>(starts[block]);
    for (std::size_t trial = 0; trial < kTrialsPerBlock; ++trial) {
      const std::uint64_t first = starts[block] + trial_stride * trial;
      const std::uint64_t second = first + timing.cue_spacing_ticks;
      const std::uint64_t consequence = second + timing.consequence_spacing_ticks;
      const std::uint64_t sample = consequence + 1u;
      if (sample > std::numeric_limits<std::uint32_t>::max())
        return std::nullopt;
      result.acquisition_curve_sample_ticks[block][trial] = static_cast<std::uint32_t>(sample);
    }
  }

  const std::uint64_t probe_second = timing.probe_spacing_ticks;
  const std::uint64_t probe_third = probe_second * 2u;
  if (probe_third + kProbeSampleCount > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  result.probe_contact_ticks = {0u, static_cast<std::uint32_t>(probe_second),
                                static_cast<std::uint32_t>(probe_third)};
  for (std::size_t index = 0; index < kProbeSampleCount; ++index) {
    result.probe_sample_ticks[index] = static_cast<std::uint32_t>(probe_third + 1u + index);
  }

  for (std::size_t master = 0; master < result.masters.size(); ++master) {
    DerivedMasterSchedule& schedule = result.masters[master];
    schedule.events.reserve(kBlockCount * kTrialsPerBlock * 5u);
    const std::size_t path_index = master % kMasterCount;
    const std::size_t stream_index = master / kMasterCount;
    for (std::size_t block = 0; block < kBlockCount; ++block) {
      std::uint8_t consequence_mask = 0;
      if (stream_index == 0u) {
        consequence_mask = definition.orientation_mask;
        if (definition.relation_paths[path_index][block] == 1u)
          consequence_mask ^= 0x0fu;
      } else {
        consequence_mask = definition.control_masks[stream_index - 1u];
      }
      for (std::size_t trial = 0; trial < kTrialsPerBlock; ++trial) {
        const std::uint32_t first =
            static_cast<std::uint32_t>(starts[block] + trial_stride * trial);
        const std::uint32_t second = first + timing.cue_spacing_ticks;
        const std::uint32_t consequence = second + timing.consequence_spacing_ticks;
        const bool lane_zero_is_01 = (definition.orientation_mask & (1u << trial)) != 0u;
        schedule.events.push_back({first, 0u, static_cast<std::uint8_t>(lane_zero_is_01 ? 0u : 1u),
                                   static_cast<std::uint8_t>(block),
                                   static_cast<std::uint8_t>(trial)});
        schedule.events.push_back({first, 1u, static_cast<std::uint8_t>(lane_zero_is_01 ? 1u : 0u),
                                   static_cast<std::uint8_t>(block),
                                   static_cast<std::uint8_t>(trial)});
        schedule.events.push_back({second, 0u, static_cast<std::uint8_t>(lane_zero_is_01 ? 1u : 0u),
                                   static_cast<std::uint8_t>(block),
                                   static_cast<std::uint8_t>(trial)});
        schedule.events.push_back({second, 1u, static_cast<std::uint8_t>(lane_zero_is_01 ? 0u : 1u),
                                   static_cast<std::uint8_t>(block),
                                   static_cast<std::uint8_t>(trial)});
        const std::uint8_t consequence_lane = (consequence_mask & (1u << trial)) != 0u ? 0u : 1u;
        schedule.events.push_back({consequence, consequence_lane, 2u,
                                   static_cast<std::uint8_t>(block),
                                   static_cast<std::uint8_t>(trial)});
      }
    }
  }

  constexpr std::array<std::uint8_t, 6> kBalancedMasks{3u, 5u, 6u, 9u, 10u, 12u};
  std::uint8_t yoked_mask = 0xffu;
  for (const std::uint8_t mask : kBalancedMasks) {
    if (mask != definition.orientation_mask &&
        mask != static_cast<std::uint8_t>(definition.orientation_mask ^ 0x0fu) &&
        mask != definition.control_masks[0] && mask != definition.control_masks[1]) {
      yoked_mask = mask;
      break;
    }
  }
  if (yoked_mask == 0xffu)
    return std::nullopt;
  const std::array<std::optional<std::uint8_t>, kMatchedControlFamilies> comparison_masks{
      definition.control_masks[0], yoked_mask, std::nullopt, definition.control_masks[1]};
  for (std::size_t family = 0; family < comparison_masks.size(); ++family) {
    for (std::size_t path = 0; path < kMasterCount; ++path) {
      DerivedMasterSchedule& schedule = result.comparison_controls[family][path];
      schedule.events.reserve(kBlockCount * kTrialsPerBlock * (comparison_masks[family] ? 5u : 4u));
      for (std::size_t block = 0; block < kBlockCount; ++block) {
        for (std::size_t trial = 0; trial < kTrialsPerBlock; ++trial) {
          const std::uint32_t first =
              static_cast<std::uint32_t>(starts[block] + trial_stride * trial);
          const std::uint32_t second = first + timing.cue_spacing_ticks;
          const std::uint32_t consequence = second + timing.consequence_spacing_ticks;
          const bool lane_zero_is_01 = (definition.orientation_mask & (1u << trial)) != 0u;
          schedule.events.push_back(
              {first, 0u, static_cast<std::uint8_t>(lane_zero_is_01 ? 0u : 1u),
               static_cast<std::uint8_t>(block), static_cast<std::uint8_t>(trial)});
          schedule.events.push_back(
              {first, 1u, static_cast<std::uint8_t>(lane_zero_is_01 ? 1u : 0u),
               static_cast<std::uint8_t>(block), static_cast<std::uint8_t>(trial)});
          schedule.events.push_back(
              {second, 0u, static_cast<std::uint8_t>(lane_zero_is_01 ? 1u : 0u),
               static_cast<std::uint8_t>(block), static_cast<std::uint8_t>(trial)});
          schedule.events.push_back(
              {second, 1u, static_cast<std::uint8_t>(lane_zero_is_01 ? 0u : 1u),
               static_cast<std::uint8_t>(block), static_cast<std::uint8_t>(trial)});
          if (comparison_masks[family]) {
            const std::uint8_t lane = (*comparison_masks[family] & (1u << trial)) != 0u ? 0u : 1u;
            schedule.events.push_back({consequence, lane, 2u, static_cast<std::uint8_t>(block),
                                       static_cast<std::uint8_t>(trial)});
          }
        }
      }
    }
  }
  return result;
}

[[nodiscard]] inline CaseParseResult parse_case_set(std::string_view plaintext) {
  if (plaintext.size() > kMaximumPlaintextBytes) {
    return {.validation = {.error = CaseValidationError::plaintext_size}};
  }
  detail::JsonParser parser(plaintext);
  auto root = parser.parse();
  if (!root)
    return {.validation = parser.validation()};
  if (root->kind != detail::JsonKind::object) {
    return {.validation = {.error = CaseValidationError::root_type}};
  }
  if (const CaseValidation keys =
          detail::exact_keys(*root, std::array<std::string_view, 2>{"schema", "cases"});
      !keys) {
    return {.validation = keys};
  }
  const detail::JsonValue* schema = detail::member(*root, "schema");
  if (schema->kind != detail::JsonKind::string) {
    return {.validation = {.error = CaseValidationError::field_type}};
  }
  if (schema->string != kCaseSchema) {
    return {.validation = {.error = CaseValidationError::schema_mismatch}};
  }
  const detail::JsonValue* cases = detail::member(*root, "cases");
  if (cases->kind != detail::JsonKind::array) {
    return {.validation = {.error = CaseValidationError::field_type}};
  }
  if (cases->array.size() != kCaseCount) {
    return {.validation = {.error = CaseValidationError::case_count}};
  }

  PromotionCaseSet result;
  result.cases.reserve(kCaseCount);
  constexpr std::array<std::string_view, 5> kCaseKeys{"index", "timing", "orientation_mask",
                                                      "control_masks", "relation_paths"};
  constexpr std::array<std::string_view, 7> kTimingKeys{
      "pre_delay_ticks",           "cue_spacing_ticks",      "consequence_spacing_ticks",
      "inter_trial_spacing_ticks", "inter_block_rest_ticks", "retention_rest_ticks",
      "probe_spacing_ticks"};

  for (std::size_t position = 0; position < cases->array.size(); ++position) {
    const detail::JsonValue& encoded = cases->array[position];
    const std::uint32_t expected_index = static_cast<std::uint32_t>(position);
    if (const CaseValidation keys = detail::exact_keys(encoded, kCaseKeys, expected_index); !keys) {
      return {.validation = keys};
    }
    const auto index = detail::uint32_value(detail::member(encoded, "index"));
    if (!index) {
      return {
          .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
    }
    if (*index != expected_index) {
      return {
          .validation = {.error = CaseValidationError::case_index, .case_index = expected_index}};
    }

    const detail::JsonValue* timing_value = detail::member(encoded, "timing");
    if (const CaseValidation keys = detail::exact_keys(*timing_value, kTimingKeys, expected_index);
        !keys) {
      return {.validation = keys};
    }
    const auto timing_field = [&](std::string_view name) {
      return detail::uint32_value(detail::member(*timing_value, name));
    };
    const auto pre_delay = timing_field("pre_delay_ticks");
    const auto cue_spacing = timing_field("cue_spacing_ticks");
    const auto consequence_spacing = timing_field("consequence_spacing_ticks");
    const auto inter_trial = timing_field("inter_trial_spacing_ticks");
    const auto inter_block = timing_field("inter_block_rest_ticks");
    const auto retention = timing_field("retention_rest_ticks");
    const auto probe_spacing = timing_field("probe_spacing_ticks");
    if (!pre_delay || !cue_spacing || !consequence_spacing || !inter_trial || !inter_block ||
        !retention || !probe_spacing) {
      return {
          .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
    }

    PromotionCase decoded{
        .index = *index,
        .timing = {*pre_delay, *cue_spacing, *consequence_spacing, *inter_trial, *inter_block,
                   *retention, *probe_spacing},
    };
    if (!detail::timing_in_range(decoded.timing)) {
      return {
          .validation = {.error = CaseValidationError::timing_range, .case_index = expected_index}};
    }

    const auto orientation = detail::uint32_value(detail::member(encoded, "orientation_mask"));
    if (!orientation) {
      return {
          .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
    }
    if (*orientation > 0x0fu) {
      return {
          .validation = {.error = CaseValidationError::mask_range, .case_index = expected_index}};
    }
    decoded.orientation_mask = static_cast<std::uint8_t>(*orientation);
    if (std::popcount(decoded.orientation_mask) != 2) {
      return {
          .validation = {.error = CaseValidationError::mask_balance, .case_index = expected_index}};
    }

    const detail::JsonValue* controls = detail::member(encoded, "control_masks");
    if (controls->kind != detail::JsonKind::array || controls->array.size() != 2u) {
      return {
          .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
    }
    for (std::size_t control = 0; control < 2u; ++control) {
      const auto value = detail::uint32_value(&controls->array[control]);
      if (!value) {
        return {
            .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
      }
      if (*value > 0x0fu) {
        return {
            .validation = {.error = CaseValidationError::mask_range, .case_index = expected_index}};
      }
      decoded.control_masks[control] = static_cast<std::uint8_t>(*value);
      if (std::popcount(decoded.control_masks[control]) != 2) {
        return {.validation = {.error = CaseValidationError::mask_balance,
                               .case_index = expected_index}};
      }
      if (std::popcount(static_cast<std::uint8_t>(decoded.control_masks[control] ^
                                                  decoded.orientation_mask)) != 2) {
        return {.validation = {.error = CaseValidationError::control_mask_relation,
                               .case_index = expected_index}};
      }
    }
    if (decoded.control_masks[0] == decoded.control_masks[1]) {
      return {.validation = {.error = CaseValidationError::control_mask_relation,
                             .case_index = expected_index}};
    }

    const detail::JsonValue* paths = detail::member(encoded, "relation_paths");
    if (paths->kind != detail::JsonKind::array || paths->array.size() != kMasterCount) {
      return {
          .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
    }
    for (std::size_t master = 0; master < kMasterCount; ++master) {
      if (paths->array[master].kind != detail::JsonKind::array ||
          paths->array[master].array.size() != kBlockCount) {
        return {
            .validation = {.error = CaseValidationError::field_type, .case_index = expected_index}};
      }
      for (std::size_t block = 0; block < kBlockCount; ++block) {
        const auto relation = detail::uint32_value(&paths->array[master].array[block]);
        if (!relation) {
          return {.validation = {.error = CaseValidationError::field_type,
                                 .case_index = expected_index}};
        }
        if (*relation > 1u) {
          return {.validation = {.error = CaseValidationError::relation_value,
                                 .case_index = expected_index}};
        }
        decoded.relation_paths[master][block] = static_cast<std::uint8_t>(*relation);
      }
    }
    bool changed = false;
    for (std::size_t block = 0; block < kBlockCount; ++block) {
      if (decoded.relation_paths[1][block] !=
          static_cast<std::uint8_t>(1u - decoded.relation_paths[0][block])) {
        return {.validation = {.error = CaseValidationError::relation_complement,
                               .case_index = expected_index}};
      }
      if (block > 0u && decoded.relation_paths[0][block] != decoded.relation_paths[0][block - 1u]) {
        changed = true;
      }
    }
    if (!changed) {
      return {.validation = {.error = CaseValidationError::relation_path_degenerate,
                             .case_index = expected_index}};
    }
    if (!derive_case_schedule(decoded)) {
      return {.validation = {.error = CaseValidationError::timing_overflow,
                             .case_index = expected_index}};
    }
    result.cases.push_back(decoded);
  }

  std::array<std::uint32_t, 16> orientation_counts{};
  std::array<bool, kBlockCount - 1u> unchanged{};
  std::array<bool, kBlockCount - 1u> zero_to_one{};
  std::array<bool, kBlockCount - 1u> one_to_zero{};
  for (const PromotionCase& definition : result.cases) {
    ++orientation_counts[definition.orientation_mask];
    for (const auto& path : definition.relation_paths) {
      for (std::size_t boundary = 0; boundary + 1u < kBlockCount; ++boundary) {
        unchanged[boundary] = unchanged[boundary] || path[boundary] == path[boundary + 1u];
        zero_to_one[boundary] =
            zero_to_one[boundary] || (path[boundary] == 0u && path[boundary + 1u] == 1u);
        one_to_zero[boundary] =
            one_to_zero[boundary] || (path[boundary] == 1u && path[boundary + 1u] == 0u);
      }
    }
  }
  for (std::uint8_t mask = 0; mask < 16u; ++mask) {
    if (std::popcount(mask) == 2 &&
        (orientation_counts[mask] == 0u ||
         orientation_counts[mask] != orientation_counts[mask ^ 0x0fu])) {
      return {
          .validation = {.error = CaseValidationError::orientation_family_not_complement_closed}};
    }
  }
  for (std::size_t boundary = 0; boundary < unchanged.size(); ++boundary) {
    if (!unchanged[boundary] || !zero_to_one[boundary] || !one_to_zero[boundary]) {
      return {.validation = {.error = CaseValidationError::clock_family_not_blocked}};
    }
  }

  const auto varies = [&](auto member_pointer) {
    const std::uint32_t first = result.cases.front().timing.*member_pointer;
    return std::any_of(result.cases.begin() + 1u, result.cases.end(),
                       [&](const PromotionCase& definition) {
                         return definition.timing.*member_pointer != first;
                       });
  };
  if (!varies(&CaseTiming::pre_delay_ticks) || !varies(&CaseTiming::cue_spacing_ticks) ||
      !varies(&CaseTiming::consequence_spacing_ticks) ||
      !varies(&CaseTiming::inter_trial_spacing_ticks) ||
      !varies(&CaseTiming::inter_block_rest_ticks) || !varies(&CaseTiming::retention_rest_ticks) ||
      !varies(&CaseTiming::probe_spacing_ticks)) {
    return {.validation = {.error = CaseValidationError::timing_family_not_variable}};
  }
  return {.value = std::move(result), .validation = {}};
}

struct PromotionThresholds {
  std::uint32_t maximum_vector_bits = 1'000'000u;
  std::uint32_t pretraining_max_distance_bits = 0u;
  std::uint32_t relation_identity_min_bits = 16u;
  std::uint32_t acquisition_min_bits = 16u;
  std::uint32_t retention_max_drift_bits = 4u;
  std::uint32_t retained_from_pretraining_min_bits = 16u;
  std::uint32_t reversal_min_bits = 16u;
  std::uint32_t reacquisition_max_drift_bits = 4u;
  std::uint32_t reversal_separation_min_bits = 16u;
  std::uint32_t omitted_change_max_drift_bits = 4u;
  std::uint32_t matched_control_advantage_min_bits = 8u;
  std::uint32_t effect_size_min_bits = 16u;
  std::uint32_t effect_size_min_ppm = 10'000u;
  std::uint32_t curve_evidence_advantage_min = 1u;
  std::uint32_t named_null_max_bits = 4u;
  std::uint32_t null_alpha_numerator = 1u;
  std::uint32_t null_alpha_denominator = 100u;
  std::uint32_t lesion_loss_min_bits = 8u;
  std::uint32_t lesion_over_matched_min_bits = 4u;
  std::uint32_t rescue_max_drift_bits = 4u;
  std::uint32_t transplant_over_control_min_bits = 8u;
};

inline constexpr PromotionThresholds kFrozenThresholds{};

struct IntegrityEvidence {
  std::uint32_t contact_failures = 0;
  std::uint32_t viability_failures = 0;
  std::uint32_t conservation_errors = 0;
  std::uint32_t inverse_errors = 0;
  std::uint32_t reproducibility_errors = 0;
  std::uint32_t reference_disagreements = 0;
  std::uint32_t production_replay_disagreements = 0;
};

struct LongitudinalEvidence {
  std::uint32_t acquisition_from_pretraining = 0;
  std::uint32_t retention_from_acquisition = 0;
  std::uint32_t retention_from_pretraining = 0;
  std::uint32_t reversal_from_retention = 0;
  std::uint32_t reacquisition_from_acquisition = 0;
  std::uint32_t reacquisition_from_reversal = 0;
  std::uint32_t omitted_change_drift = 0;
};

struct RelationIdentityComponentEvidence {
  std::uint32_t within_relation_bits = 0;
  std::uint32_t cross_ab_bits = 0;
  std::uint32_t cross_ba_bits = 0;
  std::int64_t unpaired_relation_gap_bits = 0;
  std::int64_t shuffled_relation_gap_bits = 0;
};

struct LesionFamilyEvidence {
  std::uint32_t lesioned_bit_count = 0;
  std::uint32_t matched_bit_count = 0;
  std::uint32_t unlesioned_effect_bits = 0;
  std::uint32_t lesioned_effect_bits = 0;
  std::uint32_t matched_lesion_effect_bits = 0;
  std::uint32_t rescued_effect_bits = 0;
  std::uint32_t transplant_effect_bits = 0;
  std::uint32_t transplant_control_effect_bits = 0;
};

struct MasterPromotionEvidence {
  std::uint32_t vector_bits = 0;
  IntegrityEvidence integrity{};
  std::uint32_t pretraining_distance_bits = 0;
  std::array<RelationIdentityComponentEvidence, kRelationIdentityComponents> relation_identity{};
  LongitudinalEvidence longitudinal{};
  std::array<std::uint32_t, kTrialsPerBlock> acquisition_curve_bits{};
  std::array<std::uint32_t, kTrialsPerBlock> reacquisition_curve_bits{};
  std::array<std::uint32_t, kMatchedControlComponents> contingent_component_bits{};
  std::array<std::uint32_t, kMatchedControlComponents> control_component_bits{};
  std::array<std::uint32_t, kNamedNullFamilies> named_null_effect_bits{};
  std::vector<std::uint32_t> randomized_null_effect_bits;
  std::array<LesionFamilyEvidence, kLesionFamilies> lesions{};
};

struct CasePromotionEvidence {
  std::uint32_t case_index = 0;
  std::array<MasterPromotionEvidence, kMasterCount> masters{};
};

struct PromotionEvidence {
  std::vector<CasePromotionEvidence> cases;
};

enum class PromotionFailure : std::uint8_t {
  none = 0,
  evidence_shape,
  physical_integrity,
  pretraining_leakage,
  relation_identity,
  acquisition,
  retention,
  retained_cause,
  reversal,
  reacquisition,
  reversal_separation,
  omitted_change,
  acquisition_curve,
  matched_control,
  effect_size,
  named_null_control,
  null_confidence,
  lesion_family,
};

struct PromotionDecision {
  bool passed = false;
  PromotionFailure failure = PromotionFailure::evidence_shape;
  std::uint32_t case_index = std::numeric_limits<std::uint32_t>::max();
  std::uint8_t master_index = std::numeric_limits<std::uint8_t>::max();
  std::uint16_t component_index = std::numeric_limits<std::uint16_t>::max();
};

namespace detail {

[[nodiscard]] constexpr bool integrity_is_zero(const IntegrityEvidence& evidence) {
  return evidence.contact_failures == 0u && evidence.viability_failures == 0u &&
         evidence.conservation_errors == 0u && evidence.inverse_errors == 0u &&
         evidence.reproducibility_errors == 0u && evidence.reference_disagreements == 0u &&
         evidence.production_replay_disagreements == 0u;
}

[[nodiscard]] constexpr bool passes_effect_size(std::uint32_t effect_bits,
                                                std::uint32_t vector_bits,
                                                const PromotionThresholds& thresholds) {
  if (vector_bits == 0u || vector_bits > thresholds.maximum_vector_bits ||
      effect_bits > vector_bits || effect_bits < thresholds.effect_size_min_bits) {
    return false;
  }
  return static_cast<std::uint64_t>(effect_bits) * 1'000'000u >=
         static_cast<std::uint64_t>(vector_bits) * thresholds.effect_size_min_ppm;
}

[[nodiscard]] inline std::optional<std::uint32_t> first_curve_crossing(
    const std::array<std::uint32_t, kTrialsPerBlock>& curve, std::uint32_t vector_bits,
    const PromotionThresholds& thresholds) {
  for (std::size_t index = 0; index < curve.size(); ++index) {
    if (passes_effect_size(curve[index], vector_bits, thresholds)) {
      return static_cast<std::uint32_t>(index + 1u);
    }
  }
  return std::nullopt;
}

[[nodiscard]] constexpr std::uint32_t primary_effect_bits(const MasterPromotionEvidence& evidence) {
  std::uint32_t result = evidence.longitudinal.acquisition_from_pretraining;
  result = std::min(result, evidence.longitudinal.retention_from_pretraining);
  result = std::min(result, evidence.longitudinal.reversal_from_retention);
  result = std::min(result, evidence.longitudinal.reacquisition_from_reversal);
  for (const RelationIdentityComponentEvidence& component : evidence.relation_identity) {
    result = std::min(result, component.within_relation_bits);
  }
  return result;
}

[[nodiscard]] constexpr std::int64_t relation_identity_margin(
    const RelationIdentityComponentEvidence& component, const PromotionThresholds& thresholds) {
  const std::int64_t largest_cross =
      static_cast<std::int64_t>(std::max(component.cross_ab_bits, component.cross_ba_bits));
  const std::int64_t arm_gap =
      static_cast<std::int64_t>(component.within_relation_bits) - largest_cross;
  const std::int64_t minimum_identity =
      arm_gap - static_cast<std::int64_t>(thresholds.relation_identity_min_bits);
  const std::int64_t unpaired =
      arm_gap - component.unpaired_relation_gap_bits -
      static_cast<std::int64_t>(thresholds.matched_control_advantage_min_bits);
  const std::int64_t shuffled =
      arm_gap - component.shuffled_relation_gap_bits -
      static_cast<std::int64_t>(thresholds.matched_control_advantage_min_bits);
  return std::min({minimum_identity, unpaired, shuffled});
}

template <typename Function>
[[nodiscard]] inline std::optional<PromotionDecision> visit_masters(
    const PromotionEvidence& evidence, PromotionFailure failure, Function&& function) {
  for (const CasePromotionEvidence& case_evidence : evidence.cases) {
    for (std::size_t master = 0; master < case_evidence.masters.size(); ++master) {
      const std::optional<std::uint16_t> component = function(case_evidence.masters[master]);
      if (component) {
        return PromotionDecision{.passed = false,
                                 .failure = failure,
                                 .case_index = case_evidence.case_index,
                                 .master_index = static_cast<std::uint8_t>(master),
                                 .component_index = *component};
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] inline bool evidence_values_fit(const MasterPromotionEvidence& evidence,
                                              const PromotionThresholds& thresholds) {
  if (evidence.vector_bits == 0u || evidence.vector_bits > thresholds.maximum_vector_bits ||
      evidence.randomized_null_effect_bits.size() != kRandomizedNullReplicates) {
    return false;
  }
  const auto fits = [&](std::uint32_t value) { return value <= evidence.vector_bits; };
  if (!fits(evidence.pretraining_distance_bits))
    return false;
  const LongitudinalEvidence& longitudinal = evidence.longitudinal;
  const std::array<std::uint32_t, 7> longitudinal_values{
      longitudinal.acquisition_from_pretraining,
      longitudinal.retention_from_acquisition,
      longitudinal.retention_from_pretraining,
      longitudinal.reversal_from_retention,
      longitudinal.reacquisition_from_acquisition,
      longitudinal.reacquisition_from_reversal,
      longitudinal.omitted_change_drift};
  if (!std::all_of(longitudinal_values.begin(), longitudinal_values.end(), fits) ||
      !std::all_of(evidence.acquisition_curve_bits.begin(), evidence.acquisition_curve_bits.end(),
                   fits) ||
      !std::all_of(evidence.reacquisition_curve_bits.begin(),
                   evidence.reacquisition_curve_bits.end(), fits) ||
      !std::all_of(evidence.contingent_component_bits.begin(),
                   evidence.contingent_component_bits.end(), fits) ||
      !std::all_of(evidence.control_component_bits.begin(), evidence.control_component_bits.end(),
                   fits) ||
      !std::all_of(evidence.named_null_effect_bits.begin(), evidence.named_null_effect_bits.end(),
                   fits) ||
      !std::all_of(evidence.randomized_null_effect_bits.begin(),
                   evidence.randomized_null_effect_bits.end(), fits)) {
    return false;
  }
  for (const RelationIdentityComponentEvidence& component : evidence.relation_identity) {
    if (!fits(component.within_relation_bits) || !fits(component.cross_ab_bits) ||
        !fits(component.cross_ba_bits) ||
        component.unpaired_relation_gap_bits < -static_cast<std::int64_t>(evidence.vector_bits) ||
        component.unpaired_relation_gap_bits > static_cast<std::int64_t>(evidence.vector_bits) ||
        component.shuffled_relation_gap_bits < -static_cast<std::int64_t>(evidence.vector_bits) ||
        component.shuffled_relation_gap_bits > static_cast<std::int64_t>(evidence.vector_bits)) {
      return false;
    }
  }
  for (const LesionFamilyEvidence& lesion : evidence.lesions) {
    if (lesion.lesioned_bit_count == 0u || lesion.lesioned_bit_count != lesion.matched_bit_count ||
        !fits(lesion.lesioned_bit_count) || !fits(lesion.matched_bit_count) ||
        !fits(lesion.unlesioned_effect_bits) || !fits(lesion.lesioned_effect_bits) ||
        !fits(lesion.matched_lesion_effect_bits) || !fits(lesion.rescued_effect_bits) ||
        !fits(lesion.transplant_effect_bits) || !fits(lesion.transplant_control_effect_bits)) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

[[nodiscard]] inline PromotionDecision evaluate_promotion(const PromotionEvidence& evidence) {
  constexpr const PromotionThresholds& thresholds = kFrozenThresholds;
  if (evidence.cases.size() != kCaseCount) {
    return {.failure = PromotionFailure::evidence_shape};
  }
  for (std::size_t index = 0; index < evidence.cases.size(); ++index) {
    if (evidence.cases[index].case_index != index) {
      return {.failure = PromotionFailure::evidence_shape,
              .case_index = static_cast<std::uint32_t>(index)};
    }
    for (std::size_t master = 0; master < kMasterCount; ++master) {
      if (!detail::evidence_values_fit(evidence.cases[index].masters[master], thresholds)) {
        return {.failure = PromotionFailure::evidence_shape,
                .case_index = static_cast<std::uint32_t>(index),
                .master_index = static_cast<std::uint8_t>(master)};
      }
    }
    const MasterPromotionEvidence& first = evidence.cases[index].masters[0];
    const MasterPromotionEvidence& second = evidence.cases[index].masters[1];
    if (first.vector_bits != second.vector_bits) {
      return {.failure = PromotionFailure::evidence_shape,
              .case_index = static_cast<std::uint32_t>(index),
              .master_index = 1u};
    }
    for (std::size_t component = 0; component < kRelationIdentityComponents; ++component) {
      const RelationIdentityComponentEvidence& left = first.relation_identity[component];
      const RelationIdentityComponentEvidence& right = second.relation_identity[component];
      if (left.cross_ab_bits != right.cross_ab_bits || left.cross_ba_bits != right.cross_ba_bits ||
          left.unpaired_relation_gap_bits != right.unpaired_relation_gap_bits ||
          left.shuffled_relation_gap_bits != right.shuffled_relation_gap_bits) {
        return {.failure = PromotionFailure::evidence_shape,
                .case_index = static_cast<std::uint32_t>(index),
                .master_index = 1u,
                .component_index = static_cast<std::uint16_t>(component)};
      }
    }
  }

  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::physical_integrity,
          [](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            return detail::integrity_is_zero(master.integrity) ? std::nullopt
                                                               : std::optional<std::uint16_t>{0u};
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::pretraining_leakage,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            return master.pretraining_distance_bits <= thresholds.pretraining_max_distance_bits
                       ? std::nullopt
                       : std::optional<std::uint16_t>{0u};
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::relation_identity,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            for (std::size_t index = 0; index < master.relation_identity.size(); ++index) {
              if (detail::relation_identity_margin(master.relation_identity[index], thresholds) <
                  0) {
                return static_cast<std::uint16_t>(index);
              }
            }
            return std::nullopt;
          })) {
    return *failure;
  }
  const auto require_minimum = [&](PromotionFailure category, auto member,
                                   std::uint32_t minimum) -> std::optional<PromotionDecision> {
    return detail::visit_masters(
        evidence, category,
        [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
          return master.longitudinal.*member >= minimum ? std::nullopt
                                                        : std::optional<std::uint16_t>{0u};
        });
  };
  const auto require_maximum = [&](PromotionFailure category, auto member,
                                   std::uint32_t maximum) -> std::optional<PromotionDecision> {
    return detail::visit_masters(
        evidence, category,
        [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
          return master.longitudinal.*member <= maximum ? std::nullopt
                                                        : std::optional<std::uint16_t>{0u};
        });
  };
  if (auto failure = require_minimum(PromotionFailure::acquisition,
                                     &LongitudinalEvidence::acquisition_from_pretraining,
                                     thresholds.acquisition_min_bits)) {
    return *failure;
  }
  if (auto failure = require_maximum(PromotionFailure::retention,
                                     &LongitudinalEvidence::retention_from_acquisition,
                                     thresholds.retention_max_drift_bits)) {
    return *failure;
  }
  if (auto failure = require_minimum(PromotionFailure::retained_cause,
                                     &LongitudinalEvidence::retention_from_pretraining,
                                     thresholds.retained_from_pretraining_min_bits)) {
    return *failure;
  }
  if (auto failure = require_minimum(PromotionFailure::reversal,
                                     &LongitudinalEvidence::reversal_from_retention,
                                     thresholds.reversal_min_bits)) {
    return *failure;
  }
  if (auto failure = require_maximum(PromotionFailure::reacquisition,
                                     &LongitudinalEvidence::reacquisition_from_acquisition,
                                     thresholds.reacquisition_max_drift_bits)) {
    return *failure;
  }
  if (auto failure = require_minimum(PromotionFailure::reversal_separation,
                                     &LongitudinalEvidence::reacquisition_from_reversal,
                                     thresholds.reversal_separation_min_bits)) {
    return *failure;
  }
  if (auto failure = require_maximum(PromotionFailure::omitted_change,
                                     &LongitudinalEvidence::omitted_change_drift,
                                     thresholds.omitted_change_max_drift_bits)) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::acquisition_curve,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            const auto acquisition = detail::first_curve_crossing(master.acquisition_curve_bits,
                                                                  master.vector_bits, thresholds);
            const auto reacquisition = detail::first_curve_crossing(master.reacquisition_curve_bits,
                                                                    master.vector_bits, thresholds);
            if (!acquisition || !reacquisition ||
                static_cast<std::uint64_t>(*reacquisition) +
                        thresholds.curve_evidence_advantage_min >
                    *acquisition) {
              return 0u;
            }
            return std::nullopt;
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::matched_control,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            for (std::size_t index = 0; index < kMatchedControlComponents; ++index) {
              if (static_cast<std::uint64_t>(master.control_component_bits[index]) +
                      thresholds.matched_control_advantage_min_bits >
                  master.contingent_component_bits[index]) {
                return static_cast<std::uint16_t>(index);
              }
            }
            return std::nullopt;
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::effect_size,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            return detail::passes_effect_size(detail::primary_effect_bits(master),
                                              master.vector_bits, thresholds)
                       ? std::nullopt
                       : std::optional<std::uint16_t>{0u};
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::named_null_control,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            for (std::size_t index = 0; index < master.named_null_effect_bits.size(); ++index) {
              if (master.named_null_effect_bits[index] > thresholds.named_null_max_bits) {
                return static_cast<std::uint16_t>(index);
              }
            }
            return std::nullopt;
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::null_confidence,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            const std::uint32_t observed = detail::primary_effect_bits(master);
            const std::uint64_t equal_or_greater = static_cast<std::uint64_t>(
                std::count_if(master.randomized_null_effect_bits.begin(),
                              master.randomized_null_effect_bits.end(),
                              [&](std::uint32_t null_effect) { return null_effect >= observed; }));
            const std::uint64_t left = (equal_or_greater + 1u) * thresholds.null_alpha_denominator;
            const std::uint64_t right =
                (master.randomized_null_effect_bits.size() + 1u) * thresholds.null_alpha_numerator;
            return left <= right ? std::nullopt : std::optional<std::uint16_t>{0u};
          })) {
    return *failure;
  }
  if (auto failure = detail::visit_masters(
          evidence, PromotionFailure::lesion_family,
          [&](const MasterPromotionEvidence& master) -> std::optional<std::uint16_t> {
            for (std::size_t index = 0; index < master.lesions.size(); ++index) {
              const LesionFamilyEvidence& lesion = master.lesions[index];
              if (lesion.unlesioned_effect_bits < lesion.lesioned_effect_bits ||
                  lesion.unlesioned_effect_bits < lesion.matched_lesion_effect_bits) {
                return static_cast<std::uint16_t>(index);
              }
              const std::uint32_t target_loss =
                  lesion.unlesioned_effect_bits - lesion.lesioned_effect_bits;
              const std::uint32_t matched_loss =
                  lesion.unlesioned_effect_bits - lesion.matched_lesion_effect_bits;
              const std::uint32_t rescue_drift = detail::unsigned_distance(
                  lesion.unlesioned_effect_bits, lesion.rescued_effect_bits);
              if (target_loss < thresholds.lesion_loss_min_bits ||
                  static_cast<std::uint64_t>(matched_loss) +
                          thresholds.lesion_over_matched_min_bits >
                      target_loss ||
                  rescue_drift > thresholds.rescue_max_drift_bits ||
                  static_cast<std::uint64_t>(lesion.transplant_control_effect_bits) +
                          thresholds.transplant_over_control_min_bits >
                      lesion.transplant_effect_bits) {
                return static_cast<std::uint16_t>(index);
              }
            }
            return std::nullopt;
          })) {
    return *failure;
  }
  return {.passed = true, .failure = PromotionFailure::none};
}

[[nodiscard]] constexpr std::string_view failure_name(PromotionFailure failure) {
  switch (failure) {
    case PromotionFailure::none:
      return "none";
    case PromotionFailure::evidence_shape:
      return "evidence_shape";
    case PromotionFailure::physical_integrity:
      return "physical_integrity";
    case PromotionFailure::pretraining_leakage:
      return "pretraining_leakage";
    case PromotionFailure::relation_identity:
      return "relation_identity";
    case PromotionFailure::acquisition:
      return "acquisition";
    case PromotionFailure::retention:
      return "retention";
    case PromotionFailure::retained_cause:
      return "retained_cause";
    case PromotionFailure::reversal:
      return "reversal";
    case PromotionFailure::reacquisition:
      return "reacquisition";
    case PromotionFailure::reversal_separation:
      return "reversal_separation";
    case PromotionFailure::omitted_change:
      return "omitted_change";
    case PromotionFailure::acquisition_curve:
      return "acquisition_curve";
    case PromotionFailure::matched_control:
      return "matched_control";
    case PromotionFailure::effect_size:
      return "effect_size";
    case PromotionFailure::named_null_control:
      return "named_null_control";
    case PromotionFailure::null_confidence:
      return "null_confidence";
    case PromotionFailure::lesion_family:
      return "lesion_family";
  }
  return "unknown";
}

}  // namespace substrate::bcc32::contingency::promotion
