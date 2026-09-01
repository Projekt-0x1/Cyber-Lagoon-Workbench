#pragma once

// Compact recipe for the one phase-gated-mirror experiment derived directly
// from the BCC netlist.  The parent cavity is the measured 0x07 clock paired
// with the measured 0x04 leaker.  The hash chooses only their separation and
// whether the mirror's existing negative-face-4 collar gains face 5, 6, or 7.
//
// This is deliberately a developmental recipe, not a host timing program.  A
// caller may birth the ten ordinary sites once; only F determines whether the
// popcount-two collar remains a body and whether it releases a schedule.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23PhaseMirrorHash = std::uint8_t;

constexpr std::uint32_t kPeriod23PhaseMirrorDistanceMask = 0x1fu;
constexpr std::uint32_t kPeriod23PhaseMirrorVariantShift = 5u;
constexpr std::uint32_t kPeriod23PhaseMirrorVariantMask = 0x03u;
constexpr std::uint32_t kPeriod23PhaseMirrorControl = 0u;
constexpr std::uint32_t kPeriod23PhaseMirrorFirstVariant = 1u;
constexpr std::uint32_t kPeriod23PhaseMirrorLastVariant = 3u;
constexpr std::size_t kPeriod23PhaseMirrorSeedSiteCount = 2u * kPeriod23ClockSeedSiteCount;

constexpr Period23PhaseMirrorHash make_period23_phase_mirror_hash(std::uint32_t distance,
                                                                    std::uint32_t variant) {
  return static_cast<Period23PhaseMirrorHash>(
      (distance & kPeriod23PhaseMirrorDistanceMask) |
      ((variant & kPeriod23PhaseMirrorVariantMask) << kPeriod23PhaseMirrorVariantShift));
}

constexpr std::uint32_t period23_phase_mirror_distance(Period23PhaseMirrorHash hash) {
  return static_cast<std::uint32_t>(hash & kPeriod23PhaseMirrorDistanceMask);
}

constexpr std::uint32_t period23_phase_mirror_variant(Period23PhaseMirrorHash hash) {
  return static_cast<std::uint32_t>((hash >> kPeriod23PhaseMirrorVariantShift) &
                                    kPeriod23PhaseMirrorVariantMask);
}

constexpr SiteWord period23_phase_mirror_second_face(Period23PhaseMirrorHash hash) {
  const std::uint32_t variant = period23_phase_mirror_variant(hash);
  return variant == kPeriod23PhaseMirrorControl ? 0u : face_bit(variant + 4u);
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23PhaseMirrorSeedSiteCount>
period23_phase_mirror_seed(Period23PhaseMirrorHash hash, bool include_leaker = true) {
  std::array<DevelopmentalSeedSite, kPeriod23PhaseMirrorSeedSiteCount> result{};
  const auto clock = period23_clock_seed(kPeriod23ClockSeedHash);
  const auto leaker = period23_clock_seed(make_period23_clock_seed_hash(false, false, true, 0u));
  const std::int32_t distance = static_cast<std::int32_t>(period23_phase_mirror_distance(hash));
  for (std::size_t index = 0u; index < clock.size(); ++index)
    result[index] = clock[index];
  // The final clock site is the known (2,0,0) gate-7 mirror.  Face 4 is
  // present in the parent; a variant adds exactly one sibling negative face.
  result[kPeriod23ClockSeedSiteCount - 1u].word =
      static_cast<SiteWord>(result[kPeriod23ClockSeedSiteCount - 1u].word |
                            period23_phase_mirror_second_face(hash));
  for (std::size_t index = 0u; index < leaker.size(); ++index)
    result[kPeriod23ClockSeedSiteCount + index] = {
        static_cast<std::int8_t>(leaker[index].x + distance), leaker[index].y, leaker[index].z,
        include_leaker ? leaker[index].word : kQ};
  return result;
}

}  // namespace substrate::bcc32
