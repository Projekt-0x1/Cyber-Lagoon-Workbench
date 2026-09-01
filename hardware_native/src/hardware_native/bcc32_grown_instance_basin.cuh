#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_law.cuh"

namespace substrate::bcc32::grown_instance_basin_factor {

using substrate::bcc32::SiteWord;

inline constexpr std::uint32_t kBasinCount = 8u;
inline constexpr std::uint32_t kMaxPatches = 8u;
inline constexpr std::uint32_t kMaxPatchBytes = 64u;
inline constexpr std::uint32_t kSiteCount = 16u;
inline constexpr std::uint32_t kMaxJump = 3u;
inline constexpr std::uint32_t kFullMask = (1u << kBasinCount) - 1u;
inline constexpr std::uint32_t kJournalDepth = 16u;
inline constexpr SiteWord kFactorMarkerValue = 0x1b51a7e3u;
inline constexpr SiteWord kLayoutVersionValue = 0x1b510002u;

// This factor stores unsigned words as value/complement pairs, matching the
// resident field convention used by the other grown factors.
enum GlobalField : std::uint32_t {
  kFreeMask = 0u,
  kOccupiedMask,
  kTick,
  kFactorMarker,
  kLayoutVersion,
  kJournalCount,
  kGlobalFieldCount,
};

enum BasinField : std::uint32_t {
  kActive = 0u,
  kAppearance,
  kPreviousSite,
  kCurrentSite,
  kPredictedSite,
  kEligibility,
  kConfidence,
  kSupport,
  kMissingAge,
  kBasinFieldCount,
};

inline constexpr std::uint32_t kResidentFieldCount =
    kGlobalFieldCount + kBasinCount * kBasinFieldCount;
inline constexpr std::uint32_t kResidentPhysicalRailCount =
    kResidentFieldCount * 2u;
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
};

// A descriptor carries only physical sensor geometry and a byte span. It has
// no identity, object, body, or track field.
struct PatchDescriptor {
  std::uint32_t sensor_site = 0u;
  std::uint32_t byte_offset = 0u;
  std::uint32_t byte_count = 0u;
};

struct DeviceInputs {
  const PatchDescriptor* descriptors = nullptr;
  const std::uint8_t* bytes = nullptr;
  std::uint32_t descriptor_count = 0u;
  std::uint32_t byte_count = 0u;
  std::uint32_t staged = 0u;
};

struct PredictionReceipt {
  SiteWord active_mask = 0u;
  SiteWord valid_mask = 0u;
  std::uint8_t predicted_site[kBasinCount]{};
  std::uint8_t confidence[kBasinCount]{};
  std::uint64_t state_hash = 0u;
};

struct StepReceipt {
  std::uint64_t before_hash = 0u;
  std::uint64_t after_hash = 0u;
  SiteWord free_before = kFullMask;
  SiteWord free_after = kFullMask;
  SiteWord occupied_before = 0u;
  SiteWord occupied_after = 0u;
  SiteWord matched_mask = 0u;
  SiteWord recruited_mask = 0u;
  SiteWord occluded_mask = 0u;
  SiteWord expired_mask = 0u;
  SiteWord abstained_mask = 0u;
  SiteWord predicted_mask = 0u;
  std::uint32_t descriptor_count = 0u;
  std::uint32_t assigned_count = 0u;
  std::uint32_t generic_motion_prior = 1u;
  std::uint32_t conservation = 0u;
  std::uint32_t inverse_ready = 0u;
};

__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  // A separate physical aperture: x is outside the cloud factor, while y is
  // disjoint from the sensor, form, and readout apertures. y0=194 (previously
  // 210) keeps every site inside the kSpatialMacroClosureRadius=26 clearance
  // window on all three axes; the prior y0=210 let the top of the band reach
  // global y=489, 16 past the global y<=473 ceiling. Pure translation: shape,
  // size and stride are unchanged.
  return {40 + static_cast<std::int32_t>(index % 48u),
          194 + static_cast<std::int32_t>((index / 48u) % 30u),
          100 + static_cast<std::int32_t>(index / (48u * 30u))};
}

__host__ __device__ inline std::uint32_t resident_index(
    std::uint32_t field) {
  return field * 2u;
}

__host__ __device__ inline std::uint32_t journal_index(
    std::uint32_t event, std::uint32_t physical_index) {
  return kResidentPhysicalRailCount +
         event * kResidentPhysicalRailCount + physical_index;
}

__host__ __device__ inline std::uint32_t basin_field(
    std::uint32_t basin, std::uint32_t field) {
  return kGlobalFieldCount + basin * kBasinFieldCount + field;
}

__device__ inline SiteWord read_field(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t field) {
  return words[layout.rails[resident_index(field)]];
}

__device__ inline void write_field(SiteWord* words, const DeviceLayout& layout,
                                   std::uint32_t field, SiteWord value) {
  const std::uint32_t index = resident_index(field);
  words[layout.rails[index]] = value;
  words[layout.rails[index + 1u]] = ~value;
}

__device__ inline SiteWord read_journal_field(
    const SiteWord* words, const DeviceLayout& layout, std::uint32_t event,
    std::uint32_t field) {
  return words[layout.rails[journal_index(event, resident_index(field))]];
}

__device__ inline void write_journal_field(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t event,
    std::uint32_t field, SiteWord value) {
  const std::uint32_t index = journal_index(event, resident_index(field));
  words[layout.rails[index]] = value;
  words[layout.rails[index + 1u]] = ~value;
}

__device__ inline std::uint32_t clamp_site(std::int32_t site) {
  if (site < 0) return 0u;
  if (site >= static_cast<std::int32_t>(kSiteCount)) return kSiteCount - 1u;
  return static_cast<std::uint32_t>(site);
}

__device__ inline std::uint32_t site_distance(std::uint32_t left,
                                               std::uint32_t right) {
  return left > right ? left - right : right - left;
}

__device__ inline SiteWord rotate_left(SiteWord value, std::uint32_t amount) {
  amount &= 31u;
  return amount == 0u ? value : (value << amount) | (value >> (32u - amount));
}

__device__ inline SiteWord patch_signature(const std::uint8_t* bytes,
                                           std::uint32_t offset,
                                           std::uint32_t count,
                                           std::uint32_t byte_count) {
  if (bytes == nullptr || offset > byte_count ||
      count > byte_count - offset || count == 0u) {
    return 0u;
  }
  SiteWord value = 0x9e3779b9u ^ count;
  for (std::uint32_t index = 0u; index < count; ++index) {
    value ^= rotate_left(static_cast<SiteWord>(bytes[offset + index]),
                         index & 31u);
    value = value * 1664525u + 1013904223u;
  }
  return value;
}

__device__ inline std::uint32_t appearance_similarity(SiteWord left,
                                                       SiteWord right) {
  return 32u - static_cast<std::uint32_t>(__popc(left ^ right));
}

__device__ inline std::uint64_t mix_hash(std::uint64_t hash, SiteWord value) {
  hash ^= static_cast<std::uint64_t>(value);
  hash *= 1099511628211ull;
  return hash;
}

__device__ inline std::uint64_t state_hash(const SiteWord* words,
                                           const DeviceLayout& layout) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::uint32_t field = 0u; field < kResidentFieldCount; ++field) {
    hash = mix_hash(hash, read_field(words, layout, field));
  }
  return hash;
}

__device__ inline bool journal_available(const SiteWord* words,
                                         const DeviceLayout& layout) {
  return read_field(words, layout, kFactorMarker) == kFactorMarkerValue &&
         read_field(words, layout, kLayoutVersion) == kLayoutVersionValue &&
         read_field(words, layout, kJournalCount) < kJournalDepth;
}

__device__ inline void journal_state(SiteWord* words,
                                     const DeviceLayout& layout) {
  const std::uint32_t event =
      read_field(words, layout, kJournalCount);
  for (std::uint32_t field = 0u; field < kResidentFieldCount; ++field) {
    write_journal_field(words, layout, event, field,
                        read_field(words, layout, field));
  }
  write_field(words, layout, kJournalCount, event + 1u);
}

__device__ inline void restore_journal(SiteWord* words,
                                       const DeviceLayout& layout) {
  const std::uint32_t count =
      read_field(words, layout, kJournalCount);
  if (count == 0u || count > kJournalDepth) return;
  const std::uint32_t event = count - 1u;
  for (std::uint32_t field = 0u; field < kResidentFieldCount; ++field) {
    write_field(words, layout, field,
                read_journal_field(words, layout, event, field));
    write_journal_field(words, layout, event, field, 0u);
  }
}

struct PatchObservation {
  std::uint32_t site = 0u;
  SiteWord signature = 0u;
};

__device__ inline std::uint32_t next_site(std::uint32_t previous,
                                           std::uint32_t current) {
  // Generic constant-velocity prior. It is intentionally disclosed in every
  // receipt; this factor does not claim that motion was learned.
  const std::int32_t velocity = static_cast<std::int32_t>(current) -
                                static_cast<std::int32_t>(previous);
  return clamp_site(static_cast<std::int32_t>(current) + velocity);
}

__device__ inline bool pair_better(std::uint32_t score, std::uint32_t site,
                                   SiteWord signature, std::uint32_t basin,
                                   std::uint32_t best_score,
                                   std::uint32_t best_site,
                                   SiteWord best_signature,
                                   std::uint32_t best_basin) {
  if (score != best_score) return score > best_score;
  if (site != best_site) return site < best_site;
  if (signature != best_signature) return signature < best_signature;
  return basin < best_basin;
}

__device__ inline void predict_device(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      PredictionReceipt* receipt) {
  PredictionReceipt local{};
  local.state_hash = state_hash(words, layout);
  local.active_mask = read_field(words, layout, kOccupiedMask) & kFullMask;
  for (std::uint32_t basin = 0u; basin < kBasinCount; ++basin) {
    const std::uint32_t active = read_field(
        words, layout, basin_field(basin, kActive));
    local.predicted_site[basin] = static_cast<std::uint8_t>(
        read_field(words, layout, basin_field(basin, kPredictedSite)) & 0xffu);
    local.confidence[basin] = static_cast<std::uint8_t>(
        read_field(words, layout, basin_field(basin, kConfidence)) & 0xffu);
    if (active != 0u) local.valid_mask |= 1u << basin;
  }
  if (receipt != nullptr) *receipt = local;
}

__device__ inline void clear_basin(SiteWord* words, const DeviceLayout& layout,
                                   std::uint32_t basin) {
  for (std::uint32_t field = 0u; field < kBasinFieldCount; ++field) {
    write_field(words, layout, basin_field(basin, field), 0u);
  }
}

__device__ inline StepReceipt step_device(
    SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
    StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  local.free_before = read_field(words, layout, kFreeMask) & kFullMask;
  local.occupied_before =
      read_field(words, layout, kOccupiedMask) & kFullMask;
  if (inputs == nullptr || inputs->staged == 0u) {
    local.after_hash = local.before_hash;
    local.free_after = local.free_before;
    local.occupied_after = local.occupied_before;
    local.conservation = 1u;
    if (receipt != nullptr) *receipt = local;
    return local;
  }
  local.inverse_ready = 1u;
  journal_state(words, layout);

  PatchObservation observations[kMaxPatches]{};
  std::uint32_t observation_count = 0u;
  if (inputs != nullptr && inputs->staged != 0u &&
      inputs->descriptors != nullptr) {
    observation_count = inputs->descriptor_count > kMaxPatches
                            ? kMaxPatches
                            : inputs->descriptor_count;
    for (std::uint32_t index = 0u; index < observation_count; ++index) {
      const PatchDescriptor descriptor = inputs->descriptors[index];
      observations[index].site = descriptor.sensor_site >= kSiteCount
                                     ? kSiteCount - 1u
                                     : descriptor.sensor_site;
      observations[index].signature = patch_signature(
          inputs->bytes, descriptor.byte_offset, descriptor.byte_count,
          inputs->byte_count);
    }
  }
  // Canonicalize the device working set before any assignment or recruitment.
  // Equal site/signature observations are interchangeable, so their original
  // descriptor positions cannot affect resident state.
  for (std::uint32_t start = 0u; start < observation_count; ++start) {
    std::uint32_t best = start;
    for (std::uint32_t candidate = start + 1u;
         candidate < observation_count; ++candidate) {
      if (observations[candidate].site < observations[best].site ||
          (observations[candidate].site == observations[best].site &&
           observations[candidate].signature < observations[best].signature)) {
        best = candidate;
      }
    }
    if (best != start) {
      const PatchObservation swap = observations[start];
      observations[start] = observations[best];
      observations[best] = swap;
    }
  }
  local.descriptor_count = observation_count;

  bool observation_used[kMaxPatches]{};
  bool basin_used[kBasinCount]{};
  std::int32_t assigned[kBasinCount];
  for (std::uint32_t basin = 0u; basin < kBasinCount; ++basin) {
    assigned[basin] = -1;
  }

  // Repeatedly choose the globally best admissible pair. The comparison never
  // uses descriptor array position, so shuffling the input is observationally
  // irrelevant; duplicate equal observations are physically indistinguishable.
  for (std::uint32_t round = 0u; round < kBasinCount; ++round) {
    bool found = false;
    std::uint32_t best_score = 0u;
    std::uint32_t best_site = kSiteCount;
    SiteWord best_signature = 0xffffffffu;
    std::uint32_t best_basin = kBasinCount;
    std::uint32_t best_observation = kMaxPatches;
    for (std::uint32_t basin = 0u; basin < kBasinCount; ++basin) {
      if (basin_used[basin] ||
          read_field(words, layout, basin_field(basin, kActive)) == 0u) {
        continue;
      }
      const SiteWord appearance =
          read_field(words, layout, basin_field(basin, kAppearance));
      const std::uint32_t predicted =
          read_field(words, layout, basin_field(basin, kPredictedSite)) & 0xffu;
      const std::uint32_t eligibility =
          read_field(words, layout, basin_field(basin, kEligibility)) & 0xffu;
      for (std::uint32_t observation = 0u; observation < observation_count;
           ++observation) {
        if (observation_used[observation]) continue;
        const std::uint32_t distance =
            site_distance(predicted, observations[observation].site);
        if (distance > kMaxJump) continue;
        const std::uint32_t similarity = appearance_similarity(
            appearance, observations[observation].signature);
        if (similarity < 20u) continue;
        const std::uint32_t score = similarity * 16u +
                                    (kMaxJump - distance) * 32u + eligibility;
        if (!found || pair_better(
                          score, observations[observation].site,
                          observations[observation].signature, basin,
                          best_score, best_site, best_signature, best_basin)) {
          found = true;
          best_score = score;
          best_site = observations[observation].site;
          best_signature = observations[observation].signature;
          best_basin = basin;
          best_observation = observation;
        }
      }
    }
    if (!found) break;
    basin_used[best_basin] = true;
    observation_used[best_observation] = true;
    assigned[best_basin] = static_cast<std::int32_t>(best_observation);
    local.matched_mask |= 1u << best_basin;
    ++local.assigned_count;
  }

  for (std::uint32_t basin = 0u; basin < kBasinCount; ++basin) {
    const std::uint32_t active = read_field(
        words, layout, basin_field(basin, kActive));
    if (active == 0u) continue;
    const std::uint32_t current =
        read_field(words, layout, basin_field(basin, kCurrentSite)) & 0xffu;
    const std::uint32_t predicted =
        read_field(words, layout, basin_field(basin, kPredictedSite)) & 0xffu;
    const std::uint32_t confidence =
        read_field(words, layout, basin_field(basin, kConfidence)) & 0xffu;
    const std::uint32_t eligibility =
        read_field(words, layout, basin_field(basin, kEligibility)) & 0xffu;
    if (assigned[basin] >= 0) {
      const PatchObservation observation =
          observations[static_cast<std::uint32_t>(assigned[basin])];
      const std::uint32_t distance = site_distance(predicted, observation.site);
      write_field(words, layout, basin_field(basin, kPreviousSite), current);
      write_field(words, layout, basin_field(basin, kCurrentSite),
                  observation.site);
      write_field(words, layout, basin_field(basin, kPredictedSite),
                  next_site(current, observation.site));
      write_field(words, layout, basin_field(basin, kAppearance),
                  observation.signature);
      write_field(words, layout, basin_field(basin, kEligibility),
                  eligibility + 64u > 255u ? 255u : eligibility + 64u);
      write_field(words, layout, basin_field(basin, kConfidence),
                  distance <= 1u
                      ? (confidence + 32u > 255u ? 255u : confidence + 32u)
                      : (confidence > 32u ? confidence - 32u : 0u));
      const std::uint32_t support =
          read_field(words, layout, basin_field(basin, kSupport)) & 0xffu;
      write_field(words, layout, basin_field(basin, kSupport),
                  support == 255u ? 255u : support + 1u);
      write_field(words, layout, basin_field(basin, kMissingAge), 0u);
      continue;
    }

    const std::uint32_t missing =
        read_field(words, layout, basin_field(basin, kMissingAge)) & 0xffu;
    if (missing >= 1u) {
      local.expired_mask |= 1u << basin;
      clear_basin(words, layout, basin);
      continue;
    }
    local.occluded_mask |= 1u << basin;
    write_field(words, layout, basin_field(basin, kPreviousSite), current);
    write_field(words, layout, basin_field(basin, kCurrentSite), predicted);
    write_field(words, layout, basin_field(basin, kPredictedSite),
                next_site(current, predicted));
    write_field(words, layout, basin_field(basin, kEligibility), eligibility >> 1u);
    write_field(words, layout, basin_field(basin, kConfidence), confidence >> 1u);
    write_field(words, layout, basin_field(basin, kMissingAge), missing + 1u);
    local.predicted_mask |= 1u << basin;
  }

  SiteWord free_mask = read_field(words, layout, kFreeMask) & kFullMask;
  for (std::uint32_t observation = 0u; observation < observation_count;
       ++observation) {
    (void)observation;
    std::uint32_t selected_observation = kMaxPatches;
    for (std::uint32_t candidate = 0u; candidate < observation_count;
         ++candidate) {
      if (observation_used[candidate]) continue;
      if (selected_observation == kMaxPatches ||
          observations[candidate].site <
              observations[selected_observation].site ||
          (observations[candidate].site ==
               observations[selected_observation].site &&
           observations[candidate].signature <
               observations[selected_observation].signature)) {
        selected_observation = candidate;
      }
    }
    if (selected_observation == kMaxPatches) break;
    std::uint32_t basin = kBasinCount;
    for (std::uint32_t candidate = 0u; candidate < kBasinCount; ++candidate) {
      if ((free_mask & (1u << candidate)) != 0u) {
        basin = candidate;
        break;
      }
    }
    if (basin == kBasinCount) {
      local.abstained_mask |= 1u << (selected_observation & 31u);
      continue;
    }
    free_mask &= ~(1u << basin);
    observation_used[selected_observation] = true;
    write_field(words, layout, basin_field(basin, kActive), 1u);
    write_field(words, layout, basin_field(basin, kAppearance),
                observations[selected_observation].signature);
    write_field(words, layout, basin_field(basin, kPreviousSite),
                observations[selected_observation].site);
    write_field(words, layout, basin_field(basin, kCurrentSite),
                observations[selected_observation].site);
    write_field(words, layout, basin_field(basin, kPredictedSite),
                observations[selected_observation].site);
    write_field(words, layout, basin_field(basin, kEligibility), 128u);
    write_field(words, layout, basin_field(basin, kConfidence), 64u);
    write_field(words, layout, basin_field(basin, kSupport), 1u);
    write_field(words, layout, basin_field(basin, kMissingAge), 0u);
    local.recruited_mask |= 1u << basin;
    ++local.assigned_count;
  }

  SiteWord occupied = 0u;
  for (std::uint32_t basin = 0u; basin < kBasinCount; ++basin) {
    if (read_field(words, layout, basin_field(basin, kActive)) != 0u) {
      occupied |= 1u << basin;
    }
  }
  free_mask = kFullMask & ~occupied;
  write_field(words, layout, kOccupiedMask, occupied);
  write_field(words, layout, kFreeMask, free_mask);
  write_field(words, layout, kTick,
              read_field(words, layout, kTick) + 1u);
  local.free_after = free_mask;
  local.occupied_after = occupied;
  local.after_hash = state_hash(words, layout);
  local.conservation = ((free_mask | occupied) == kFullMask &&
                        (free_mask & occupied) == 0u)
                           ? 1u
                           : 0u;
  if (receipt != nullptr) *receipt = local;
  return local;
}

static __global__ void step_kernel(SiteWord* words, DeviceLayout layout,
                                   DeviceInputs* inputs,
                                   StepReceipt* receipt,
                                   std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (!journal_available(words, layout)) {
    *advanced = 0u;
    return;
  }
  (void)step_device(words, layout, inputs, receipt);
  *advanced = 1u;
}

static __global__ void predict_kernel(const SiteWord* words,
                                      DeviceLayout layout,
                                      PredictionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  predict_device(words, layout, receipt);
}

static __global__ void inverse_step_kernel(SiteWord* words,
                                           DeviceLayout layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  restore_journal(words, layout);
}

static __global__ void lesion_kernel(SiteWord* words, DeviceLayout layout,
                                     std::uint32_t basin) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || basin >= kBasinCount) return;
  clear_basin(words, layout, basin);
  SiteWord occupied = 0u;
  for (std::uint32_t index = 0u; index < kBasinCount; ++index) {
    if (read_field(words, layout, basin_field(index, kActive)) != 0u) {
      occupied |= 1u << index;
    }
  }
  write_field(words, layout, kOccupiedMask, occupied);
  write_field(words, layout, kFreeMask, kFullMask & ~occupied);
}

}  // namespace substrate::bcc32::grown_instance_basin_factor
