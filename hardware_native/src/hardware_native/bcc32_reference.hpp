#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "bcc32_coordinate.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

struct ReferenceSite {
  Z3Coordinate coordinate{};
  SiteWord word = kQuiescentWord;

  friend bool operator==(const ReferenceSite&, const ReferenceSite&) = default;
};

using BasisPermutation = std::array<std::uint32_t, 4>;

// The canonical (marker, path, waste) role frame: the three named bases in
// order, with the single remaining basis last.
//
// Built in ONE place because several factors derive their footprint through it
// -- processive rearm, processive release and their owner scans each rebuilt it
// inline -- and a divergent fourth entry would silently ROTATE a factor's
// geometry rather than fail. It is also what lets an external probe supply
// eligible matter for those factors by CALLING the mapping instead of
// restating it.
// The radius-one collar offsets a factor's neighbourhood is read through.
//
// Same reason as `role_basis_permutation` below: these were file-local to
// `bcc32_law.cpp`'s anonymous namespace, so a probe wanting to build a
// neighbourhood the law actually reads had to RESTATE the geometry, and a
// restatement that drifts rotates a factor's collar rather than failing.
[[nodiscard]] inline Z3Coordinate positive_offset(std::uint32_t basis) {
  const Int3 offset = direction_offset(static_cast<Direction>(basis));
  return {offset.x, offset.y, offset.z};
}

[[nodiscard]] inline Z3Coordinate negative_offset(std::uint32_t basis) {
  const Int3 offset = direction_offset(static_cast<Direction>(basis));
  return {-offset.x, -offset.y, -offset.z};
}

[[nodiscard]] constexpr BasisPermutation role_basis_permutation(
    std::uint32_t marker, std::uint32_t path, std::uint32_t waste) {
  BasisPermutation permutation{marker, path, waste, 0u};
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    if (basis != marker && basis != path && basis != waste)
      permutation[3u] = basis;
  return permutation;
}

// ---------------------------------------------------------------------------
// Native execution view.
//
// The law's cost is not the lookup -- it is building arbitrary-precision
// coordinates. Profiled inclusive: operator+ 17.19%, subtract 14.57%,
// ExactCoordinate ctor 14.44%, dtor 12.37%, against read 9.59%. A factor
// derives ~13 coordinates from ONE base plus small constant offsets, so the
// fix is to narrow the base ONCE and keep the whole derivation native.
//
// ⛔ An earlier attempt narrowed per read instead and LOST (17.02s -> 19.59s):
// the caller had already paid to construct the exact coordinate, so
// accelerating the last step in the chain recovered nothing. Do not add an
// overload of NativeReadView::read taking an ExactCoordinate; the absence of
// one is what keeps that mistake out of new call sites.
struct NativeCoordinate {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  friend bool operator==(const NativeCoordinate&, const NativeCoordinate&) = default;
};

struct NativeCoordinateHash {
  [[nodiscard]] std::size_t operator()(const NativeCoordinate& value) const noexcept {
    std::uint64_t mixed = 0xcbf29ce484222325ull;
    for (const std::int64_t part : {value.x, value.y, value.z}) {
      mixed = (mixed ^ static_cast<std::uint64_t>(part)) * 0x100000001b3ull;
    }
    return static_cast<std::size_t>(mixed);
  }
};

// Every offset a law factor adds to a narrowed base is a small constant. This
// guard band is the envelope proof: a base inside it plus any offset bounded by
// it cannot overflow, so the per-site additions are ordinary unchecked int64.
inline constexpr std::int64_t kNativeGuardBand = (std::numeric_limits<std::int64_t>::max)() / 4;

[[nodiscard]] inline constexpr NativeCoordinate native_offset(std::int64_t x, std::int64_t y,
                                                              std::int64_t z) {
  return NativeCoordinate{x, y, z};
}

[[nodiscard]] inline constexpr bool native_offset_in_guard_band(const NativeCoordinate& offset) {
  return offset.x >= -kNativeGuardBand && offset.x <= kNativeGuardBand &&
         offset.y >= -kNativeGuardBand && offset.y <= kNativeGuardBand &&
         offset.z >= -kNativeGuardBand && offset.z <= kNativeGuardBand;
}

// Unchecked BY CONSTRUCTION: legal only after try_narrow_guarded accepted the
// base and every offset passed native_offset_in_guard_band.
[[nodiscard]] inline constexpr NativeCoordinate add_unchecked(const NativeCoordinate& base,
                                                              const NativeCoordinate& offset) {
  return NativeCoordinate{base.x + offset.x, base.y + offset.y, base.z + offset.z};
}

// This is an exact finite-support observer for X_Q. Missing coordinates are
// complete ordinary Q matter, not absent cells. It has no extent, wrapping,
// active-set policy, or execution schedule.
class ReferenceLattice {
 public:
  ReferenceLattice() = default;
  explicit ReferenceLattice(std::span<const ReferenceSite> support);

  [[nodiscard]] SiteWord read(const Z3Coordinate& coordinate) const;
  void write(const Z3Coordinate& coordinate, SiteWord word);

  // Replaces finite non-Q support. Q entries compact away; duplicate
  // coordinates are rejected so an import cannot hide a write order.
  void replace_support(std::span<const ReferenceSite> support);

  [[nodiscard]] std::size_t support_size() const { return support_.size(); }
  [[nodiscard]] bool empty() const { return support_.empty(); }

  // A value snapshot permits law code to derive a phase from a stable finite
  // state rather than mutate while traversing storage.
  [[nodiscard]] std::vector<ReferenceSite> support() const;

  // Returns support plus every supplied relative coordinate, sorted and
  // deduplicated. This is geometric closure only; it selects no phase or
  // execution order.
  [[nodiscard]] std::vector<Z3Coordinate> causal_closure(
      std::span<const Z3Coordinate> relative_coordinates) const;

  // A native-keyed projection of THIS lattice, for the derive-many-from-one
  // pattern in the law. It is a snapshot, not a live view: build it from the
  // immutable `before` copy a factor already makes, never from the destination
  // being written. Named `build_` because construction is O(support).
  class NativeReadView {
   public:
    [[nodiscard]] SiteWord read(const NativeCoordinate& coordinate) const noexcept {
      const auto found = sites_.find(coordinate);
      return found == sites_.end() ? kQuiescentWord : found->second;
    }

    [[nodiscard]] std::size_t size() const noexcept { return sites_.size(); }

   private:
    friend class ReferenceLattice;

    std::unordered_map<NativeCoordinate, SiteWord, NativeCoordinateHash> sites_;
  };

  [[nodiscard]] NativeReadView build_native_read_view() const;

  // Succeeds only inside the guard band, so a later add_unchecked of any
  // in-band offset cannot overflow. A base outside it -- or one that is not a
  // machine integer at all -- yields nullopt, and the caller must then evaluate
  // that WHOLE candidate on the exact path. Never mix the two per site.
  [[nodiscard]] static std::optional<NativeCoordinate> try_narrow_guarded(
      const Z3Coordinate& coordinate);

 private:
  // Lookup is hashed, not ordered. The support is read on the order of
  // thousands of times per superstep by the law, and each ordered lookup cost
  // O(log n) comparisons of three arbitrary-precision integers.
  //
  // NO ORDER IS LOST. Ordering was never a property of this member -- it was a
  // property of what support() returns, and support() now sorts explicitly
  // with the same Z3CoordinateLess. same_lattice() compares those sorted
  // snapshots, and causal_closure() has always accumulated into an ordered
  // set. Nothing else iterates this container.
  using SupportMap =
      std::unordered_map<Z3Coordinate, SiteWord, Z3CoordinateHash>;

  SupportMap support_;
};

using DeltaNQ = boost::multiprecision::cpp_int;

[[nodiscard]] DeltaNQ delta_n_q(const ReferenceLattice& lattice);
[[nodiscard]] bool same_lattice(const ReferenceLattice& left, const ReferenceLattice& right);
[[nodiscard]] ReferenceLattice translated(const ReferenceLattice& lattice,
                                          const Z3Coordinate& displacement);
[[nodiscard]] bool is_basis_permutation(const BasisPermutation& permutation);
[[nodiscard]] Z3Coordinate transformed_coordinate(const Z3Coordinate& coordinate,
                                                  const BasisPermutation& permutation);
[[nodiscard]] SiteWord transformed_word(SiteWord word, const BasisPermutation& permutation);
[[nodiscard]] ReferenceLattice tetrahedral_transform(const ReferenceLattice& lattice,
                                                     const BasisPermutation& permutation);

}  // namespace substrate::bcc32
