#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_grown_instance_basin.cuh"
#include "hardware_native/bcc32_law.cuh"

namespace substrate::bcc32::grown_sensorimotor_factor {

namespace instance = grown_instance_basin_factor;

inline constexpr std::uint32_t kActionCount = 2u;
inline constexpr std::uint32_t kMaxContactBytes = 192u;
inline constexpr std::uint32_t kNoAction = 0xffffffffu;
// Exact inverse history must cover one complete instance -> relation -> form
// interaction. This is proof storage, not an eligibility or learning horizon.
inline constexpr std::uint32_t kJournalDepth = 64u;
inline constexpr std::uint32_t kEligibilityMaxAge = 3u;
inline constexpr std::uint32_t kRelationSlotCount = 4u;
inline constexpr std::uint32_t kNoRelation = 0xffffffffu;
inline constexpr SiteWord kLayoutVersionValue = 5u;
inline constexpr SiteWord kFactorMarkerValue = 0x5e75a13du;

enum RelationField : std::uint32_t {
  kRelationOccupied,
  kRelationContext,
  kRelationCue,
  kRelationAction,
  kRelationPredictedReafference,
  kRelationPredictedInternal,
  kRelationPositive,
  kRelationNegative,
  kRelationEndpoint0,
  kRelationEndpoint1,
  kRelationChanged,
  kRelationUnchanged,
  kRelationFieldCount,
};

enum Rail : std::uint32_t {
  kCue0,
  kCue1,
  kPositive0,
  kPositive1,
  kNegative0,
  kNegative1,
  kPredictedReafference0,
  kPredictedReafference1,
  kPredictedInternal0,
  kPredictedInternal1,
  kRouteEnabled0,
  kRouteEnabled1,
  kSomaticPath0,
  kSomaticPath1,
  kReafferencePath,
  kExplorerPhase,
  kActiveCue,
  kEligibility,
  kEligibilityAge,
  kMotor,
  kActualReafference,
  kActualInternal,
  kRemoteControl,
  kWorldReafference0,
  kWorldReafference1,
  kWorldInternal0,
  kWorldInternal1,
  kActiveContext,
  kActiveRelationSlot,
  kRelationBase,
  kRelationRailEnd = kRelationBase + kRelationSlotCount * kRelationFieldCount,
  kTransformSupportBase = kRelationRailEnd,
  kTransformSupportEnd = kTransformSupportBase + 16u,
  kTransformInput = kTransformSupportEnd,
  kTransformOutput,
  kTransformValid,
  kJournalCount,
  kLayoutVersion,
  kFactorMarker,
  kRailCount
};

inline constexpr std::uint32_t kResidentPhysicalRailCount = kRailCount * 2u;
inline constexpr std::uint32_t kJournalPhysicalRailCount =
    kResidentPhysicalRailCount * kJournalDepth;
inline constexpr std::uint32_t kPhysicalRailCount =
    kResidentPhysicalRailCount + kJournalPhysicalRailCount;

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kPhysicalRailCount]{};
  grown_instance_basin_factor::DeviceLayout context{};
};

struct DeviceInputs {
  const std::uint8_t* cue = nullptr;
  const std::uint8_t* actual_reafference = nullptr;
  const std::uint8_t* actual_internal = nullptr;
  std::uint32_t cue_count = 0u;
  std::uint32_t actual_reafference_count = 0u;
  std::uint32_t actual_internal_count = 0u;
  std::uint32_t staged = 0u;
  std::uint32_t transform_mode = 0u;
  std::uint32_t transform_input = 0u;
  std::uint32_t transform_observed = 0u;
};

struct PredictionReceipt {
  std::uint32_t action = kNoAction;
  std::uint32_t relation_slot = kNoRelation;
  SiteWord context_signature = 0u;
  SiteWord cue_signature = 0u;
  SiteWord predicted_reafference = 0u;
  SiteWord predicted_internal = 0u;
  std::int32_t action0_score = 0;
  std::int32_t action1_score = 0;
  std::uint32_t abstained = 0u;
  std::uint32_t prediction_valid = 0u;
  std::uint32_t probation = 0u;
};

struct ConsequenceReceipt {
  std::uint32_t action = kNoAction;
  std::uint32_t relation_slot = kNoRelation;
  SiteWord context_signature = 0u;
  std::uint32_t processed = 0u;
  std::uint32_t revised = 0u;
  std::uint32_t matched = 0u;
  std::uint32_t reafference_connected = 0u;
  std::uint32_t somatic_connected = 0u;
  std::uint32_t positive_support = 0u;
  std::uint32_t negative_support = 0u;
  std::uint32_t world_transition = 0u;
  std::uint32_t prediction_valid = 0u;
  std::uint32_t probation = 0u;
  std::uint32_t expired = 0u;
};

struct LesionReceipt {
  std::uint32_t changed_bits = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
};

struct RelationCensus {
  std::uint32_t occupied[kRelationSlotCount]{};
  SiteWord context[kRelationSlotCount]{};
  SiteWord cue[kRelationSlotCount]{};
  std::uint32_t action[kRelationSlotCount]{};
  SiteWord predicted_reafference[kRelationSlotCount]{};
  SiteWord predicted_internal[kRelationSlotCount]{};
  SiteWord positive[kRelationSlotCount]{};
  SiteWord negative[kRelationSlotCount]{};
  SiteWord endpoint0[kRelationSlotCount]{};
  SiteWord endpoint1[kRelationSlotCount]{};
  SiteWord changed[kRelationSlotCount]{};
  SiteWord unchanged[kRelationSlotCount]{};
};

// A view keeps the learned basin masks separate from their currently live
// witnesses. The masks are resident relation state; the witnesses are the
// portion still present in the active instance field after lesions/expiry.
struct RelationEndpointView {
  SiteWord active_basin_mask = 0u;
  SiteWord eligible_basin_mask = 0u;
  SiteWord endpoint0[kRelationSlotCount]{};
  SiteWord endpoint1[kRelationSlotCount]{};
  SiteWord endpoint0_witness[kRelationSlotCount]{};
  SiteWord endpoint1_witness[kRelationSlotCount]{};
  SiteWord changed[kRelationSlotCount]{};
  SiteWord unchanged[kRelationSlotCount]{};
  SiteWord changed_witness[kRelationSlotCount]{};
  SiteWord unchanged_witness[kRelationSlotCount]{};
};

struct TransformReceipt {
  std::uint32_t input = 0u;
  std::uint32_t observed = 0u;
  std::uint32_t predicted = 0u;
  std::uint32_t valid_mask = 0u;
  std::uint32_t learned_lanes = 0u;
};

__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {72 + static_cast<std::int32_t>(index % 48u),
          -96 + static_cast<std::int32_t>((index / 48u) % 48u),
          48 + static_cast<std::int32_t>(index / (48u * 48u))};
}

__host__ __device__ inline std::uint32_t value_index(Rail rail) {
  return static_cast<std::uint32_t>(rail) * 2u;
}

__host__ __device__ inline std::uint32_t complement_index(Rail rail) {
  return value_index(rail) + 1u;
}

__host__ __device__ inline std::uint32_t journal_index(
    std::uint32_t event, std::uint32_t resident_physical_index) {
  return kResidentPhysicalRailCount +
         event * kResidentPhysicalRailCount + resident_physical_index;
}

__host__ __device__ inline Rail transform_support_rail(
    std::uint32_t lane, std::uint32_t relation) {
  return static_cast<Rail>(
      static_cast<std::uint32_t>(kTransformSupportBase) +
      lane * 2u + relation);
}

__host__ __device__ inline Rail relation_rail(std::uint32_t slot,
                                               RelationField field) {
  return static_cast<Rail>(static_cast<std::uint32_t>(kRelationBase) +
                           slot * kRelationFieldCount +
                           static_cast<std::uint32_t>(field));
}

__device__ inline SiteWord read_value(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      Rail rail) {
  return words[layout.rails[value_index(rail)]];
}

__device__ inline void write_value(SiteWord* words,
                                   const DeviceLayout& layout, Rail rail,
                                   SiteWord value) {
  words[layout.rails[value_index(rail)]] = value;
  words[layout.rails[complement_index(rail)]] = ~value;
}

__device__ inline void write_journal_pair(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t event,
    std::uint32_t resident_physical_index, SiteWord value) {
  words[layout.rails[journal_index(event, resident_physical_index)]] = value;
  words[layout.rails[journal_index(event, resident_physical_index + 1u)]] =
      ~value;
}

__device__ inline bool begin_journal(SiteWord* words,
                                     const DeviceLayout& layout) {
  const std::uint32_t count = read_value(words, layout, kJournalCount);
  if (count >= kJournalDepth) return false;
  for (std::uint32_t index = 0u; index < kResidentPhysicalRailCount;
       index += 2u) {
    write_journal_pair(words, layout, count, index,
                       words[layout.rails[index]]);
  }
  write_value(words, layout, kJournalCount, count + 1u);
  return true;
}

__device__ inline bool journal_available(const SiteWord* words,
                                         const DeviceLayout& layout) {
  return read_value(words, layout, kJournalCount) < kJournalDepth;
}

__device__ inline void restore_last_journal(SiteWord* words,
                                            const DeviceLayout& layout) {
  const std::uint32_t count = read_value(words, layout, kJournalCount);
  if (count == 0u || count > kJournalDepth) return;
  const std::uint32_t event = count - 1u;
  for (std::uint32_t index = 0u; index < kResidentPhysicalRailCount;
       index += 2u) {
    const SiteWord value =
        words[layout.rails[journal_index(event, index)]];
    words[layout.rails[index]] = value;
    words[layout.rails[index + 1u]] = ~value;
    write_journal_pair(words, layout, event, index, 0u);
  }
}

__host__ __device__ inline SiteWord contact_signature(
    const std::uint8_t* bytes, std::uint32_t count) {
  SiteWord signature = 0x9e3779b9u ^ count;
  for (std::uint32_t index = 0u; index < count; ++index) {
    signature = ((signature << 5u) | (signature >> 27u)) ^
                (static_cast<SiteWord>(bytes[index]) + 0x7f4a7c15u + index);
  }
  return signature == 0u ? 1u : signature;
}

struct ContextDescriptor {
  SiteWord appearance = 0u;
  std::int32_t relative_current = 0;
  std::int32_t velocity = 0;
  std::int32_t displacement = 0;
};

__device__ inline SiteWord rotate_left(SiteWord value, std::uint32_t amount) {
  amount &= 31u;
  return amount == 0u ? value : (value << amount) | (value >> (32u - amount));
}

__device__ inline bool context_descriptor_before(const ContextDescriptor& left,
                                                 const ContextDescriptor& right) {
  if (left.appearance != right.appearance) {
    return left.appearance < right.appearance;
  }
  if (left.relative_current != right.relative_current) {
    return left.relative_current < right.relative_current;
  }
  if (left.velocity != right.velocity) return left.velocity < right.velocity;
  return left.displacement < right.displacement;
}

// Context identity is deliberately restricted to geometry that remains stable
// under translation and exposure. Confidence/support are learning state, not
// identity, and therefore never enter this signature.
__device__ inline SiteWord active_context_signature(
    const SiteWord* words, const DeviceLayout& layout) {
  ContextDescriptor descriptors[instance::kBasinCount]{};
  std::uint32_t count = 0u;
  std::uint32_t minimum_site = instance::kSiteCount;
  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    const instance::DeviceLayout& context = layout.context;
    if (instance::read_field(words, context,
                             instance::basin_field(basin, instance::kActive)) ==
            0u ||
        instance::read_field(
            words, context,
            instance::basin_field(basin, instance::kMissingAge)) != 0u) {
      continue;
    }
    const std::uint32_t current =
        instance::read_field(words, context,
                             instance::basin_field(basin,
                                                   instance::kCurrentSite)) &
        0xffu;
    if (current < minimum_site) minimum_site = current;
  }
  if (minimum_site == instance::kSiteCount) return 0u;

  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    const instance::DeviceLayout& context = layout.context;
    if (instance::read_field(words, context,
                             instance::basin_field(basin, instance::kActive)) ==
            0u ||
        instance::read_field(
            words, context,
            instance::basin_field(basin, instance::kMissingAge)) != 0u) {
      continue;
    }
    const std::int32_t previous = static_cast<std::int32_t>(
        instance::read_field(words, context,
                             instance::basin_field(basin,
                                                   instance::kPreviousSite)) &
        0xffu);
    const std::int32_t current = static_cast<std::int32_t>(
        instance::read_field(words, context,
                             instance::basin_field(basin,
                                                   instance::kCurrentSite)) &
        0xffu);
    const std::int32_t predicted = static_cast<std::int32_t>(
        instance::read_field(words, context,
                             instance::basin_field(basin,
                                                   instance::kPredictedSite)) &
        0xffu);
    descriptors[count++] = {
        instance::read_field(words, context,
                             instance::basin_field(basin,
                                                   instance::kAppearance)),
        current - static_cast<std::int32_t>(minimum_site), current - previous,
        predicted - current};
  }
  for (std::uint32_t start = 1u; start < count; ++start) {
    const ContextDescriptor value = descriptors[start];
    std::uint32_t position = start;
    while (position > 0u && context_descriptor_before(
                                value, descriptors[position - 1u])) {
      descriptors[position] = descriptors[position - 1u];
      --position;
    }
    descriptors[position] = value;
  }

  SiteWord signature = 0x6d2b79f5u ^ count;
  for (std::uint32_t index = 0u; index < count; ++index) {
    const ContextDescriptor descriptor = descriptors[index];
    signature = rotate_left(signature, 5u) ^ descriptor.appearance;
    signature = rotate_left(signature, 7u) ^
                static_cast<SiteWord>(descriptor.relative_current);
    signature = rotate_left(signature, 11u) ^
                static_cast<SiteWord>(descriptor.velocity);
    signature = rotate_left(signature, 13u) ^
                static_cast<SiteWord>(descriptor.displacement);
  }
  return signature == 0u ? 1u : signature;
}

__device__ inline bool cue_matches(const SiteWord* words,
                                   const DeviceLayout& layout,
                                   std::uint32_t action, SiteWord cue) {
  const Rail cue_rail = action == 0u ? kCue0 : kCue1;
  const SiteWord learned_cue = read_value(words, layout, cue_rail);
  return learned_cue != 0u &&
         __popc(learned_cue & cue) > __popc(learned_cue ^ cue);
}

__device__ inline std::int32_t route_score(const SiteWord* words,
                                           const DeviceLayout& layout,
                                           std::uint32_t action,
                                           SiteWord cue) {
  const Rail cue_rail = action == 0u ? kCue0 : kCue1;
  const Rail positive = action == 0u ? kPositive0 : kPositive1;
  const Rail negative = action == 0u ? kNegative0 : kNegative1;
  const Rail enabled = action == 0u ? kRouteEnabled0 : kRouteEnabled1;
  if (read_value(words, layout, enabled) == 0u) return -0x3fffffff;
  const SiteWord learned_cue = read_value(words, layout, cue_rail);
  if (learned_cue == 0u) return 0;
  if (__popc(learned_cue & cue) <= __popc(learned_cue ^ cue)) return 0;
  const std::int32_t positive_mass =
      static_cast<std::int32_t>(__popc(read_value(words, layout, positive)));
  const std::int32_t negative_mass =
      static_cast<std::int32_t>(__popc(read_value(words, layout, negative)));
  // A newly observed route has a probationary resident prior. It is usable
  // for competition, but the first observation is never reported as correct.
  if (positive_mass == 0 && negative_mass == 0) return 1;
  return positive_mass - negative_mass;
}

__device__ inline void clear_relation(SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t slot) {
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    write_value(words, layout, relation_rail(slot,
                                             static_cast<RelationField>(field)),
                0u);
  }
}

__device__ inline bool relation_matches(const SiteWord* words,
                                        const DeviceLayout& layout,
                                        std::uint32_t slot, SiteWord context,
                                        SiteWord cue) {
  return read_value(words, layout,
                    relation_rail(slot, kRelationOccupied)) != 0u &&
         read_value(words, layout,
                    relation_rail(slot, kRelationContext)) == context &&
         read_value(words, layout, relation_rail(slot, kRelationCue)) == cue;
}

struct EndpointMasks {
  SiteWord first = 0u;
  SiteWord second = 0u;
};

// This is trajectory evidence, not a semantic role assignment. A valid
// directional contact has exactly one eligible basin whose previous site
// differs from its current site and exactly one eligible stable counterpart.
// Every other case is deliberately represented as zero evidence.
struct DirectionalEvidence {
  SiteWord changed = 0u;
  SiteWord unchanged = 0u;
};

__device__ inline SiteWord active_basin_mask(const SiteWord* words,
                                             const DeviceLayout& layout) {
  SiteWord mask = 0u;
  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    const bool active = instance::read_field(
                            words, layout.context,
                            instance::basin_field(basin, instance::kActive)) !=
                        0u;
    if (active) mask |= 1u << basin;
  }
  return mask;
}

__device__ inline SiteWord relation_eligible_basin_mask(
    const SiteWord* words, const DeviceLayout& layout) {
  SiteWord mask = 0u;
  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    const bool eligible =
        instance::read_field(words, layout.context,
                             instance::basin_field(basin, instance::kActive)) !=
            0u &&
        instance::read_field(
            words, layout.context,
            instance::basin_field(basin, instance::kMissingAge)) == 0u &&
        instance::read_field(
            words, layout.context,
            instance::basin_field(basin, instance::kEligibility)) != 0u;
    if (eligible) mask |= 1u << basin;
  }
  return mask;
}

__device__ inline DirectionalEvidence derive_directional_evidence(
    const SiteWord* words, const DeviceLayout& layout) {
  DirectionalEvidence result{};
  const SiteWord eligible = relation_eligible_basin_mask(words, layout);
  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    if ((eligible & (1u << basin)) == 0u) continue;
    const std::uint32_t previous = instance::read_field(
        words, layout.context,
        instance::basin_field(basin, instance::kPreviousSite)) & 0xffu;
    const std::uint32_t current = instance::read_field(
        words, layout.context,
        instance::basin_field(basin, instance::kCurrentSite)) & 0xffu;
    if (previous != current)
      result.changed |= 1u << basin;
    else
      result.unchanged |= 1u << basin;
  }
  if (__popc(result.changed) != 1 || __popc(result.unchanged) != 1)
    return {};
  return result;
}

// Endpoint identity is a physical ordering, not a semantic role. Only basins
// carrying live instance eligibility for this contact may participate. The
// first two eligible basins in current-site order are retained as one-hot
// masks; basin index only breaks physical ties.
__device__ inline EndpointMasks derive_endpoint_masks(
    const SiteWord* words, const DeviceLayout& layout) {
  EndpointMasks result{};
  const SiteWord eligible = relation_eligible_basin_mask(words, layout);
  std::uint32_t selected[2] = {instance::kBasinCount,
                               instance::kBasinCount};
  std::uint32_t sites[2] = {instance::kSiteCount, instance::kSiteCount};
  for (std::uint32_t basin = 0u; basin < instance::kBasinCount; ++basin) {
    if ((eligible & (1u << basin)) == 0u) continue;
    const std::uint32_t site =
        instance::read_field(words, layout.context,
                             instance::basin_field(basin,
                                                   instance::kCurrentSite)) &
        0xffu;
    for (std::uint32_t position = 0u; position < 2u; ++position) {
      if (site > sites[position] ||
          (site == sites[position] && basin >= selected[position])) {
        continue;
      }
      for (std::uint32_t shift = 1u; shift > position; --shift) {
        selected[shift] = selected[shift - 1u];
        sites[shift] = sites[shift - 1u];
      }
      selected[position] = basin;
      sites[position] = site;
      break;
    }
  }
  if (selected[0] < instance::kBasinCount) result.first = 1u << selected[0];
  if (selected[1] < instance::kBasinCount) result.second = 1u << selected[1];
  return result;
}

__device__ inline std::int32_t relation_score(const SiteWord* words,
                                              const DeviceLayout& layout,
                                              std::uint32_t slot) {
  const std::int32_t positive = static_cast<std::int32_t>(__popc(
      read_value(words, layout, relation_rail(slot, kRelationPositive))));
  const std::int32_t negative = static_cast<std::int32_t>(__popc(
      read_value(words, layout, relation_rail(slot, kRelationNegative))));
  return positive == 0 && negative == 0 ? 1 : positive - negative;
}

__device__ inline std::uint32_t free_relation_slot(
    const SiteWord* words, const DeviceLayout& layout) {
  for (std::uint32_t slot = 0u; slot < kRelationSlotCount; ++slot) {
    if (read_value(words, layout,
                   relation_rail(slot, kRelationOccupied)) == 0u) {
      return slot;
    }
  }
  return kNoRelation;
}

__device__ inline void prepare_device(SiteWord* words,
                                      const DeviceLayout& layout,
                                      DeviceInputs& inputs,
                                      PredictionReceipt* receipt) {
  PredictionReceipt local{};
  const SiteWord cue = contact_signature(inputs.cue, inputs.cue_count);
  local.cue_signature = cue;

  const SiteWord context = active_context_signature(words, layout);
  local.context_signature = context;
  write_value(words, layout, kActiveContext, context);
  write_value(words, layout, kActiveRelationSlot, kNoRelation);

  if (context != 0u) {
    std::int32_t scores[kActionCount] = {-0x3fffffff, -0x3fffffff};
    std::uint32_t best_slot[kActionCount] = {kNoRelation, kNoRelation};
    std::uint32_t matching[kActionCount] = {0u, 0u};
    std::uint32_t evidence[kActionCount] = {0u, 0u};
    for (std::uint32_t slot = 0u; slot < kRelationSlotCount; ++slot) {
      if (!relation_matches(words, layout, slot, context, cue)) continue;
      const std::uint32_t action = read_value(
          words, layout, relation_rail(slot, kRelationAction)) & 1u;
      const SiteWord positive =
          read_value(words, layout, relation_rail(slot, kRelationPositive));
      const SiteWord negative =
          read_value(words, layout, relation_rail(slot, kRelationNegative));
      const std::uint32_t mass = __popc(positive) + __popc(negative);
      const std::int32_t score = relation_score(words, layout, slot);
      ++matching[action];
      evidence[action] += mass;
      if (best_slot[action] == kNoRelation) {
        best_slot[action] = slot;
        scores[action] = score;
      } else {
        scores[action] += score;
        if (score > relation_score(words, layout, best_slot[action]) ||
            (score == relation_score(words, layout, best_slot[action]) &&
             slot < best_slot[action])) {
          best_slot[action] = slot;
        }
      }
    }
    if (matching[0] == 0u) scores[0] = 0;
    if (matching[1] == 0u) scores[1] = 0;
    local.action0_score = scores[0];
    local.action1_score = scores[1];

    const bool comparable = matching[0] != 0u && matching[1] != 0u &&
                            evidence[0] != 0u && evidence[1] != 0u &&
                            (scores[0] - scores[1] <= 1) &&
                            (scores[1] - scores[0] <= 1);
    std::uint32_t action = kNoAction;
    std::uint32_t slot = kNoRelation;
    if (comparable) {
      local.abstained = 1u;
    } else if (matching[0] != 0u || matching[1] != 0u) {
      if (matching[0] == 0u || matching[1] == 0u) {
        action = matching[0] != 0u ? 0u : 1u;
        slot = best_slot[action];
        const SiteWord endpoints =
            read_value(words, layout,
                       relation_rail(slot, kRelationEndpoint0)) |
            read_value(words, layout,
                       relation_rail(slot, kRelationEndpoint1));
        const std::uint32_t endpoint_mass = __popc(endpoints);
        const std::uint32_t alternative = action ^ 1u;
        const std::uint32_t free_slot = free_relation_slot(words, layout);
        if (endpoint_mass != 0u && evidence[action] >= endpoint_mass &&
            free_slot != kNoRelation) {
          action = alternative;
          slot = free_slot;
        }
      } else if (evidence[0] == 0u && evidence[1] == 0u) {
        const SiteWord phase = read_value(words, layout, kExplorerPhase);
        action = __popc(phase) & 1u;
        write_value(words, layout, kExplorerPhase,
                    phase ^ (1u << ((__popc(phase) + 3u) & 31u)));
        slot = best_slot[action];
      } else {
        // A new exact route carries a probationary +1 prior. Let that resident
        // prior compete with mature evidence; otherwise any counterevidenced
        // old route permanently suppresses the alternative merely because the
        // alternative has not yet had a second consequence contact.
        action = scores[1] > scores[0] ? 1u : 0u;
        slot = best_slot[action];
      }
    } else {
      slot = free_relation_slot(words, layout);
      if (slot != kNoRelation) {
        const SiteWord phase = read_value(words, layout, kExplorerPhase);
        action = __popc(phase) & 1u;
        write_value(words, layout, kExplorerPhase,
                    phase ^ (1u << ((__popc(phase) + 3u) & 31u)));
      }
    }
    local.action = action;
    local.relation_slot = slot;
    if (action != kNoAction && slot != kNoRelation) {
      const bool exact = relation_matches(words, layout, slot, context, cue);
      local.prediction_valid = exact ? 1u : 0u;
      local.probation = exact ? 0u : 1u;
      if (exact) {
        local.predicted_reafference = read_value(
            words, layout, relation_rail(slot, kRelationPredictedReafference));
        local.predicted_internal = read_value(
            words, layout, relation_rail(slot, kRelationPredictedInternal));
      }
    }
    write_value(words, layout, kActiveCue, cue);
    write_value(words, layout, kActiveRelationSlot, slot);
    write_value(words, layout, kEligibility,
                action == kNoAction ? 0u : (1u << action));
    write_value(words, layout, kEligibilityAge,
                action == kNoAction ? 0u : 1u);
    write_value(words, layout, kMotor,
                action == kNoAction ? 0u : (1u << action));
    inputs.staged = 0u;
    if (receipt != nullptr) *receipt = local;
    return;
  }

  local.action0_score = route_score(words, layout, 0u, cue);
  local.action1_score = route_score(words, layout, 1u, cue);
  const bool learned0 = read_value(words, layout, kCue0) != 0u;
  const bool learned1 = read_value(words, layout, kCue1) != 0u;

  std::uint32_t action = kNoAction;
  if (learned0 && learned1 && local.action0_score == local.action1_score) {
    local.abstained = 1u;
  } else if (!learned0 && !learned1) {
    const SiteWord phase = read_value(words, layout, kExplorerPhase);
    action = __popc(phase) & 1u;
    write_value(words, layout, kExplorerPhase,
                phase ^ (1u << ((__popc(phase) + 3u) & 31u)));
  } else if (!learned1 && local.action0_score <= 0) {
    action = 1u;
  } else if (!learned0 && local.action1_score <= 0) {
    action = 0u;
  } else {
    action = local.action1_score > local.action0_score ? 1u : 0u;
  }

  local.action = action;
  local.relation_slot = kNoRelation;
  if (action != kNoAction) {
    local.prediction_valid = cue_matches(words, layout, action, cue) ? 1u : 0u;
    local.probation = local.prediction_valid == 0u ? 1u : 0u;
  }
  write_value(words, layout, kActiveCue, cue);
  write_value(words, layout, kEligibility,
              action == kNoAction ? 0u : (1u << action));
  write_value(words, layout, kEligibilityAge,
              action == kNoAction ? 0u : 1u);
  write_value(words, layout, kMotor,
              action == kNoAction ? 0u : (1u << action));
  if (action != kNoAction) {
    local.predicted_reafference = read_value(
        words, layout,
        action == 0u ? kPredictedReafference0 : kPredictedReafference1);
    local.predicted_internal = read_value(
        words, layout,
        action == 0u ? kPredictedInternal0 : kPredictedInternal1);
  }
  inputs.staged = 0u;
  if (receipt != nullptr) *receipt = local;
}

__device__ inline SiteWord credit_bit(SiteWord cue, std::uint32_t action) {
  const SiteWord rotated = (cue << 11u) | (cue >> 21u);
  return 1u << ((cue ^ rotated ^ (action * 13u)) & 31u);
}

__device__ inline SiteWord increment_unary(SiteWord value) {
  const SiteWord empty = ~value;
  return empty == 0u ? value : value | (empty & (0u - empty));
}

__device__ inline SiteWord decrement_unary(SiteWord value) {
  return value == 0u ? 0u : value & (value - 1u);
}

__device__ inline void transform_device(
    SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
    TransformReceipt* receipt) {
  TransformReceipt local{};
  local.input = inputs->transform_input & 0xffu;
  local.observed = inputs->transform_observed & 0xffu;
  for (std::uint32_t lane = 0u; lane < 8u; ++lane) {
    const std::uint32_t input = (local.input >> lane) & 1u;
    if (inputs->transform_mode == 1u) {
      const std::uint32_t output = (local.observed >> lane) & 1u;
      const Rail selected = transform_support_rail(lane, input ^ output);
      write_value(words, layout, selected,
                  increment_unary(read_value(words, layout, selected)));
      ++local.learned_lanes;
    }
    const std::uint32_t same_mass =
        __popc(read_value(words, layout, transform_support_rail(lane, 0u)));
    const std::uint32_t flip_mass =
        __popc(read_value(words, layout, transform_support_rail(lane, 1u)));
    if (same_mass == flip_mass) continue;
    local.valid_mask |= 1u << lane;
    const std::uint32_t relation = flip_mass > same_mass ? 1u : 0u;
    if ((input ^ relation) != 0u) local.predicted |= 1u << lane;
  }
  write_value(words, layout, kTransformInput, local.input);
  write_value(words, layout, kTransformOutput, local.predicted);
  write_value(words, layout, kTransformValid, local.valid_mask);
  inputs->transform_mode = 0u;
  if (receipt != nullptr) *receipt = local;
}

__device__ inline void settle_device(SiteWord* words,
                                     const DeviceLayout& layout,
                                     DeviceInputs& inputs,
                                     ConsequenceReceipt* receipt) {
  ConsequenceReceipt local{};
  const SiteWord eligibility = read_value(words, layout, kEligibility);
  const SiteWord motor = read_value(words, layout, kMotor);
  if (eligibility == 0u || motor == 0u) {
    if (receipt != nullptr) *receipt = local;
    return;
  }

  const std::uint32_t action = (__ffs(motor) - 1u) & 1u;
  local.action = action;
  local.processed = 1u;
  const SiteWord prior_actual_reafference =
      read_value(words, layout, kActualReafference);
  const SiteWord prior_actual_internal =
      read_value(words, layout, kActualInternal);
  const SiteWord actual_reafference = contact_signature(
      inputs.actual_reafference, inputs.actual_reafference_count);
  const SiteWord actual_internal =
      contact_signature(inputs.actual_internal, inputs.actual_internal_count);
  write_value(words, layout, kActualReafference, actual_reafference);
  write_value(words, layout, kActualInternal, actual_internal);
  local.world_transition =
      (actual_reafference != prior_actual_reafference ||
       actual_internal != prior_actual_internal)
          ? 1u
          : 0u;
  local.context_signature = read_value(words, layout, kActiveContext);
  local.relation_slot = read_value(words, layout, kActiveRelationSlot);
  local.reafference_connected =
      read_value(words, layout, kReafferencePath) != 0u;
  const Rail somatic_path = action == 0u ? kSomaticPath0 : kSomaticPath1;
  local.somatic_connected = read_value(words, layout, somatic_path) != 0u;
  const Rail cue_rail = action == 0u ? kCue0 : kCue1;
  const Rail positive = action == 0u ? kPositive0 : kPositive1;
  const Rail negative = action == 0u ? kNegative0 : kNegative1;
  const Rail predicted_reafference =
      action == 0u ? kPredictedReafference0 : kPredictedReafference1;
  const Rail predicted_internal =
      action == 0u ? kPredictedInternal0 : kPredictedInternal1;
  const DirectionalEvidence direction =
      derive_directional_evidence(words, layout);

  SiteWord learned_before[kRelationFieldCount]{};
  Rail learned_rails[kRelationFieldCount]{};
  std::uint32_t learned_count = 0u;
  const bool relation_learning =
      local.reafference_connected != 0u && local.somatic_connected != 0u &&
      local.context_signature != 0u &&
      local.relation_slot < kRelationSlotCount;
  if (relation_learning) {
    for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
      learned_rails[learned_count] = relation_rail(
          local.relation_slot, static_cast<RelationField>(field));
      learned_before[learned_count] =
          read_value(words, layout, learned_rails[learned_count]);
      ++learned_count;
    }
  } else if (local.reafference_connected != 0u &&
             local.somatic_connected != 0u) {
    learned_rails[0] = cue_rail;
    learned_rails[1] = predicted_reafference;
    learned_rails[2] = predicted_internal;
    learned_rails[3] = positive;
    learned_rails[4] = negative;
    learned_count = 5u;
    for (std::uint32_t index = 0u; index < learned_count; ++index) {
      learned_before[index] = read_value(words, layout, learned_rails[index]);
    }
  }

  if (relation_learning) {
    const std::uint32_t slot = local.relation_slot;
    const SiteWord cue = read_value(words, layout, kActiveCue);
    const EndpointMasks endpoints = derive_endpoint_masks(words, layout);
    const bool occupied = read_value(
                              words, layout,
                              relation_rail(slot, kRelationOccupied)) != 0u;
    if (!occupied) {
      write_value(words, layout, relation_rail(slot, kRelationOccupied), 1u);
      write_value(words, layout, relation_rail(slot, kRelationContext),
                  local.context_signature);
      write_value(words, layout, relation_rail(slot, kRelationCue), cue);
      write_value(words, layout, relation_rail(slot, kRelationAction), action);
      write_value(words, layout,
                  relation_rail(slot, kRelationPredictedReafference),
                  actual_reafference);
      write_value(words, layout, relation_rail(slot, kRelationPredictedInternal),
                  actual_internal);
      write_value(words, layout, relation_rail(slot, kRelationPositive), 0u);
      write_value(words, layout, relation_rail(slot, kRelationNegative), 0u);
      write_value(words, layout, relation_rail(slot, kRelationEndpoint0),
                  endpoints.first);
      write_value(words, layout, relation_rail(slot, kRelationEndpoint1),
                  endpoints.second);
      write_value(words, layout, relation_rail(slot, kRelationChanged),
                  direction.changed);
      write_value(words, layout, relation_rail(slot, kRelationUnchanged),
                  direction.unchanged);
      local.probation = 1u;
      local.prediction_valid = 0u;
    } else {
      const SiteWord expected_reafference = read_value(
          words, layout, relation_rail(slot, kRelationPredictedReafference));
      const SiteWord expected_internal = read_value(
          words, layout, relation_rail(slot, kRelationPredictedInternal));
      const bool exact =
          read_value(words, layout, relation_rail(slot, kRelationContext)) ==
              local.context_signature &&
          read_value(words, layout, relation_rail(slot, kRelationCue)) == cue &&
          (read_value(words, layout, relation_rail(slot, kRelationAction)) &
           1u) == action;
      local.prediction_valid = exact ? 1u : 0u;
      local.matched = exact && expected_reafference == actual_reafference &&
                      expected_internal == actual_internal;
      SiteWord positive_value = read_value(
          words, layout, relation_rail(slot, kRelationPositive));
      SiteWord negative_value = read_value(
          words, layout, relation_rail(slot, kRelationNegative));
      const SiteWord endpoint_mask =
          read_value(words, layout,
                     relation_rail(slot, kRelationEndpoint0)) |
          read_value(words, layout,
                     relation_rail(slot, kRelationEndpoint1));
      const std::uint32_t support_limit = __popc(endpoint_mask);
      if (local.matched != 0u) {
        if (__popc(positive_value) < support_limit) {
          positive_value = increment_unary(positive_value);
        }
        negative_value = decrement_unary(negative_value);
      } else {
        positive_value = decrement_unary(positive_value);
        if (__popc(negative_value) < support_limit) {
          negative_value = increment_unary(negative_value);
        }
      }
      write_value(words, layout, relation_rail(slot, kRelationPositive),
                  positive_value);
      write_value(words, layout, relation_rail(slot, kRelationNegative),
                  negative_value);
      if (endpoints.first != 0u && endpoints.second != 0u) {
        write_value(words, layout, relation_rail(slot, kRelationEndpoint0),
                    endpoints.first);
        write_value(words, layout, relation_rail(slot, kRelationEndpoint1),
                    endpoints.second);
      }
      if (direction.changed != 0u && direction.unchanged != 0u) {
        write_value(words, layout, relation_rail(slot, kRelationChanged),
                    direction.changed);
        write_value(words, layout, relation_rail(slot, kRelationUnchanged),
                    direction.unchanged);
      }
      local.positive_support = __popc(positive_value);
      local.negative_support = __popc(negative_value);
    }
  } else if (local.reafference_connected != 0u &&
             local.somatic_connected != 0u) {
    const SiteWord cue = read_value(words, layout, kActiveCue);
    SiteWord expected_reafference =
        read_value(words, layout, predicted_reafference);
    SiteWord expected_internal = read_value(words, layout, predicted_internal);
    const bool first_contact = read_value(words, layout, cue_rail) == 0u;
    if (first_contact) {
      write_value(words, layout, cue_rail, cue);
      write_value(words, layout, predicted_reafference, actual_reafference);
      write_value(words, layout, predicted_internal, actual_internal);
      expected_reafference = actual_reafference;
      expected_internal = actual_internal;
      local.probation = 1u;
      local.prediction_valid = 0u;
    } else {
      local.probation = 0u;
      local.prediction_valid = 1u;
    }
    if (!first_contact) {
      local.matched = expected_reafference == actual_reafference &&
                      expected_internal == actual_internal;
      const SiteWord bit = credit_bit(cue, action);
      SiteWord positive_value = read_value(words, layout, positive);
      SiteWord negative_value = read_value(words, layout, negative);
      if (local.matched != 0u) {
        positive_value |= bit;
        negative_value &= ~bit;
      } else {
        positive_value &= ~bit;
        negative_value |= bit;
      }
      write_value(words, layout, positive, positive_value);
      write_value(words, layout, negative, negative_value);
      local.positive_support = __popc(positive_value);
      local.negative_support = __popc(negative_value);
    }
  }
  local.revised = 0u;
  for (std::uint32_t index = 0u; index < learned_count; ++index) {
    if (learned_before[index] !=
        read_value(words, layout, learned_rails[index])) {
      local.revised = 1u;
    }
  }
  write_value(words, layout, kEligibility, 0u);
  write_value(words, layout, kEligibilityAge, 0u);
  write_value(words, layout, kMotor, 0u);
  write_value(words, layout, kActiveContext, 0u);
  write_value(words, layout, kActiveRelationSlot, kNoRelation);
  inputs.staged = 0u;
  if (receipt != nullptr) *receipt = local;
}

__device__ inline void expire_eligibility(SiteWord* words,
                                          const DeviceLayout& layout,
                                          ConsequenceReceipt* receipt) {
  const SiteWord eligibility = read_value(words, layout, kEligibility);
  if (eligibility == 0u) return;
  const std::uint32_t age = read_value(words, layout, kEligibilityAge);
  if (age < kEligibilityMaxAge) {
    write_value(words, layout, kEligibilityAge, age + 1u);
    return;
  }
  ConsequenceReceipt local{};
  local.action = (__ffs(read_value(words, layout, kMotor)) - 1u) & 1u;
  local.expired = 1u;
  write_value(words, layout, kEligibility, 0u);
  write_value(words, layout, kEligibilityAge, 0u);
  write_value(words, layout, kMotor, 0u);
  write_value(words, layout, kActiveContext, 0u);
  write_value(words, layout, kActiveRelationSlot, kNoRelation);
  if (receipt != nullptr) *receipt = local;
}

__device__ inline bool step_device(SiteWord* words,
                                  const DeviceLayout& layout,
                                  DeviceInputs* inputs,
                                  PredictionReceipt* prediction,
                                  ConsequenceReceipt* consequence,
                                  TransformReceipt* transform) {
  if (!begin_journal(words, layout)) return false;
  if (inputs == nullptr) return true;
  if (inputs->transform_mode != 0u) {
    transform_device(words, layout, inputs, transform);
    return true;
  }
  if (inputs->staged == 0u) {
    expire_eligibility(words, layout, consequence);
    return true;
  }
  if (read_value(words, layout, kEligibility) == 0u) {
    prepare_device(words, layout, *inputs, prediction);
  } else {
    settle_device(words, layout, *inputs, consequence);
  }
  return true;
}

__device__ inline void inverse_step_device(SiteWord* words,
                                           const DeviceLayout& layout) {
  restore_last_journal(words, layout);
}

static __global__ void step_kernel(SiteWord* words,
                                   const DeviceLayout* layout,
                                   DeviceInputs* inputs,
                                   PredictionReceipt* prediction,
                                   ConsequenceReceipt* consequence,
                                   TransformReceipt* transform,
                                   std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *advanced = step_device(words, *layout, inputs, prediction, consequence,
                          transform)
                  ? 1u
                  : 0u;
}

static __global__ void inverse_step_kernel(SiteWord* words,
                                           const DeviceLayout* layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  inverse_step_device(words, *layout);
}

static __global__ void set_path_kernel(SiteWord* words,
                                       const DeviceLayout* layout,
                                       Rail rail, std::uint32_t enabled,
                                       LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  LesionReceipt local{};
  for (std::uint32_t index = 0u; index < kRailCount; ++index) {
    local.matter_before +=
        __popc(words[layout->rails[value_index(static_cast<Rail>(index))]]) +
        __popc(words[layout->rails[complement_index(static_cast<Rail>(index))]]);
  }
  const SiteWord before = read_value(words, *layout, rail);
  const SiteWord after = enabled != 0u ? 0xffffffffu : 0u;
  write_value(words, *layout, rail, after);
  local.changed_bits = __popc(before ^ after);
  for (std::uint32_t index = 0u; index < kRailCount; ++index) {
    local.matter_after +=
        __popc(words[layout->rails[value_index(static_cast<Rail>(index))]]) +
        __popc(words[layout->rails[complement_index(static_cast<Rail>(index))]]);
  }
  if (receipt != nullptr) *receipt = local;
}

static __global__ void census_kernel(const SiteWord* words,
                                     const DeviceLayout* layout,
                                     std::uint32_t* matter) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    total += __popc(words[layout->rails[index]]);
  }
  *matter = total;
}

static __global__ void census_with_context_kernel(const SiteWord* words,
                                                  const DeviceLayout* layout,
                                                  std::uint32_t* matter) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    total += __popc(words[layout->rails[index]]);
  }
  for (std::uint32_t index = 0u;
       index < instance::kPhysicalRailCount; ++index) {
    total += __popc(words[layout->context.rails[index]]);
  }
  *matter = total;
}

static __global__ void lesion_relation_kernel(SiteWord* words,
                                              const DeviceLayout* layout,
                                              std::uint32_t slot,
                                              LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  LesionReceipt local{};
  if (slot >= kRelationSlotCount) {
    if (receipt != nullptr) *receipt = local;
    return;
  }
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    local.matter_before += __popc(words[layout->rails[index]]);
  }
  SiteWord before_value[kRelationFieldCount]{};
  SiteWord before_complement[kRelationFieldCount]{};
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    const Rail rail = relation_rail(slot, static_cast<RelationField>(field));
    before_value[field] = read_value(words, *layout, rail);
    before_complement[field] =
        words[layout->rails[complement_index(rail)]];
  }
  clear_relation(words, *layout, slot);
  for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
    const Rail rail = relation_rail(slot, static_cast<RelationField>(field));
    local.changed_bits += __popc(before_value[field] ^
                                  read_value(words, *layout, rail));
    local.changed_bits += __popc(
        before_complement[field] ^
        words[layout->rails[complement_index(rail)]]);
  }
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    local.matter_after += __popc(words[layout->rails[index]]);
  }
  if (receipt != nullptr) *receipt = local;
}

static __global__ void matched_remote_perturbation_kernel(
    SiteWord* words, const DeviceLayout* layout,
    std::uint32_t physical_changed_bits, LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  LesionReceipt local{};
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    local.matter_before += __popc(words[layout->rails[index]]);
  }
  std::uint32_t logical_bits = physical_changed_bits / 2u;
  for (std::uint32_t index = 0u;
       index < 16u && logical_bits > 0u; ++index) {
    const Rail rail = static_cast<Rail>(
        static_cast<std::uint32_t>(kTransformSupportBase) + index);
    const std::uint32_t width = logical_bits < 32u ? logical_bits : 32u;
    const SiteWord mask =
        width == 32u ? 0xffffffffu : ((1u << width) - 1u);
    write_value(words, *layout, rail, read_value(words, *layout, rail) ^ mask);
    local.changed_bits += 2u * __popc(mask);
    logical_bits -= width;
  }
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    local.matter_after += __popc(words[layout->rails[index]]);
  }
  if (receipt != nullptr) *receipt = local;
}

static __global__ void relation_census_kernel(const SiteWord* words,
                                              const DeviceLayout* layout,
                                              RelationCensus* census) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || census == nullptr) return;
  RelationCensus local{};
  for (std::uint32_t slot = 0u; slot < kRelationSlotCount; ++slot) {
    local.occupied[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationOccupied));
    local.context[slot] =
        read_value(words, *layout, relation_rail(slot, kRelationContext));
    local.cue[slot] =
        read_value(words, *layout, relation_rail(slot, kRelationCue));
    local.action[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationAction));
    local.predicted_reafference[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationPredictedReafference));
    local.predicted_internal[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationPredictedInternal));
    local.positive[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationPositive));
    local.negative[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationNegative));
    local.endpoint0[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationEndpoint0));
    local.endpoint1[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationEndpoint1));
    local.changed[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationChanged));
    local.unchanged[slot] = read_value(
        words, *layout, relation_rail(slot, kRelationUnchanged));
  }
  *census = local;
}

__device__ inline void read_relation_endpoint_view_device(
    const SiteWord* words, const DeviceLayout& layout,
    RelationEndpointView* view) {
  if (view == nullptr) return;
  RelationEndpointView local{};
  local.active_basin_mask = active_basin_mask(words, layout);
  local.eligible_basin_mask = relation_eligible_basin_mask(words, layout);
  for (std::uint32_t slot = 0u; slot < kRelationSlotCount; ++slot) {
    local.endpoint0[slot] = read_value(
        words, layout, relation_rail(slot, kRelationEndpoint0));
    local.endpoint1[slot] = read_value(
        words, layout, relation_rail(slot, kRelationEndpoint1));
    local.changed[slot] = read_value(
        words, layout, relation_rail(slot, kRelationChanged));
    local.unchanged[slot] = read_value(
        words, layout, relation_rail(slot, kRelationUnchanged));
    local.endpoint0_witness[slot] =
        local.endpoint0[slot] & local.active_basin_mask;
    local.endpoint1_witness[slot] =
        local.endpoint1[slot] & local.active_basin_mask;
    local.changed_witness[slot] =
        local.changed[slot] & local.active_basin_mask;
    local.unchanged_witness[slot] =
        local.unchanged[slot] & local.active_basin_mask;
  }
  *view = local;
}

static __global__ void relation_endpoint_view_kernel(
    const SiteWord* words, const DeviceLayout* layout,
    RelationEndpointView* view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  read_relation_endpoint_view_device(words, *layout, view);
}

static __global__ void remote_probe_kernel(const SiteWord* words,
                                           const DeviceLayout* layout,
                                           std::uint32_t* active) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *active = read_value(words, *layout, kRemoteControl) != 0u ? 1u : 0u;
}

static __global__ void motor_probe_kernel(const SiteWord* words,
                                          const DeviceLayout* layout,
                                          std::uint32_t* action) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const SiteWord motor = read_value(words, *layout, kMotor);
  *action = __popc(motor) == 1 ? static_cast<std::uint32_t>(__ffs(motor) - 1)
                               : kNoAction;
}

}  // namespace substrate::bcc32::grown_sensorimotor_factor
