#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32::sparse_completion {

struct Entry {
  Z3Coordinate coordinate;
  SiteWord word;
};

inline const Z3Coordinate kSource{0, -1, 1};
inline const Z3Coordinate kTarget{1, 0, 1};
inline const Z3Coordinate kLesion{-1, 0, 1};
inline constexpr SiteWord kSignal = channel_bit(kConformationShift, 0u);
inline constexpr SiteWord kPrimaryCue = channel_bit(kConformationShift, 0u);
inline constexpr SiteWord kAlternateCue = channel_bit(kConformationShift, 2u);

// A language-blank production-F completion motif. C0 and C2 are distinct
// physical cue states, but either one completes C0 at kTarget after three F
// supersteps. Reversibility keeps their different microhistories elsewhere.
inline const std::array<Entry, 11> kCompletionMotif{{
    {{-1, -2, 0}, 0x000080ffu},
    {{-1, -1, 0}, 0x0000007fu},
    {{-1, -1, 1}, 0x000001ffu},
    {{-1, 0, 1}, 0x600000f9u},
    {{0, -2, 1}, 0x000000fdu},
    {{0, -1, 2}, 0x000040ffu},
    {{0, 0, 0}, 0x000030bfu},
    {{0, 0, 1}, 0x101008f4u},
    {{0, 1, 1}, 0x090070ffu},
    {{2, -1, 1}, 0xb00000f9u},
    {{2, 0, 1}, 0x0000905fu},
}};

inline Z3Coordinate add(const Z3Coordinate& left, const Z3Coordinate& right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline ReferenceLattice fixture() {
  ReferenceLattice result;
  for (const Entry& entry : kCompletionMotif)
    result.write(entry.coordinate, entry.word);
  result.write(kSource, kQ);
  return result;
}

inline bool merge_ordinary(ReferenceLattice* destination, const ReferenceLattice& source) {
  for (const ReferenceSite& site : source.support()) {
    const SiteWord current = destination->read(site.coordinate);
    const SiteWord combined = kQ ^ static_cast<SiteWord>((current ^ kQ) | (site.word ^ kQ));
    destination->write(site.coordinate, combined);
  }
  return true;
}

struct CompletionModule {
  ReferenceLattice blank;
  Z3Coordinate source;
  Z3Coordinate lesion;
  Z3Coordinate target;
};

inline CompletionModule module() {
  return {fixture(), kSource, kLesion, kTarget};
}

inline CompletionModule translated_module(const CompletionModule& source_module,
                                          const Z3Coordinate& displacement) {
  return {
      translated(source_module.blank, displacement),
      add(source_module.source, displacement),
      add(source_module.lesion, displacement),
      add(source_module.target, displacement),
  };
}

inline void set_cue(ReferenceLattice* world, const Z3Coordinate& coordinate, SiteWord cue) {
  world->write(coordinate, kQ | cue);
}

inline std::uint32_t active_count(const ReferenceLattice& world,
                                  const std::vector<Z3Coordinate>& coordinates) {
  std::uint32_t result = 0u;
  for (const Z3Coordinate& coordinate : coordinates)
    result += (world.read(coordinate) & kSignal) != 0u;
  return result;
}

}  // namespace substrate::bcc32::sparse_completion
