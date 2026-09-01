#include "bcc32_reference.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace substrate::bcc32 {

ReferenceLattice::ReferenceLattice(std::span<const ReferenceSite> support) {
  replace_support(support);
}

SiteWord ReferenceLattice::read(const Z3Coordinate& coordinate) const {
  const auto found = support_.find(coordinate);
  return found == support_.end() ? kQuiescentWord : found->second;
}

void ReferenceLattice::write(const Z3Coordinate& coordinate, SiteWord word) {
  if (word == kQuiescentWord) {
    support_.erase(coordinate);
    return;
  }
  support_.insert_or_assign(coordinate, word);
}

void ReferenceLattice::replace_support(std::span<const ReferenceSite> support) {
  SupportMap replacement;
  for (const ReferenceSite& site : support) {
    if (!replacement.emplace(site.coordinate, site.word).second) {
      throw std::invalid_argument("BCC-32 reference support contains a duplicate coordinate");
    }
  }
  for (auto site = replacement.begin(); site != replacement.end();) {
    if (site->second == kQuiescentWord) {
      site = replacement.erase(site);
    } else {
      ++site;
    }
  }
  support_ = std::move(replacement);
}

std::vector<ReferenceSite> ReferenceLattice::support() const {
  std::vector<ReferenceSite> result;
  result.reserve(support_.size());
  for (const auto& [coordinate, word] : support_) {
    result.push_back({coordinate, word});
  }
  // The snapshot is the ONLY place this class exposes an order, and callers
  // depend on it: same_lattice() compares two snapshots elementwise, so an
  // unordered traversal would report two identical lattices as different.
  // Storage is hashed for lookup speed; the order is restored here with the
  // same comparator the ordered container used to supply implicitly.
  std::sort(result.begin(), result.end(),
            [](const ReferenceSite& left, const ReferenceSite& right) {
              return Z3CoordinateLess{}(left.coordinate, right.coordinate);
            });
  return result;
}

std::optional<NativeCoordinate> ReferenceLattice::try_narrow_guarded(
    const Z3Coordinate& coordinate) {
  static const CoordinateComponent low{-kNativeGuardBand};
  static const CoordinateComponent high{kNativeGuardBand};
  if (coordinate.x < low || coordinate.x > high) return std::nullopt;
  if (coordinate.y < low || coordinate.y > high) return std::nullopt;
  if (coordinate.z < low || coordinate.z > high) return std::nullopt;
  return NativeCoordinate{coordinate.x.convert_to<std::int64_t>(),
                          coordinate.y.convert_to<std::int64_t>(),
                          coordinate.z.convert_to<std::int64_t>()};
}

ReferenceLattice::NativeReadView ReferenceLattice::build_native_read_view() const {
  NativeReadView view;
  view.sites_.reserve(support_.size());
  for (const auto& [coordinate, word] : support_) {
    // Sites outside the guard band are deliberately absent. That is safe
    // because a candidate whose base and offset envelope are both in band can
    // never address one, and such a candidate goes down the exact path whole.
    if (const std::optional<NativeCoordinate> native = try_narrow_guarded(coordinate)) {
      view.sites_.emplace(*native, word);
    }
  }
  return view;
}

std::vector<Z3Coordinate> ReferenceLattice::causal_closure(
    std::span<const Z3Coordinate> relative_coordinates) const {
  std::set<Z3Coordinate, Z3CoordinateLess> closure;
  for (const auto& [coordinate, word] : support_) {
    (void)word;
    closure.emplace(coordinate);
    for (const Z3Coordinate& relative : relative_coordinates) {
      closure.emplace(coordinate + relative);
    }
  }
  return {closure.begin(), closure.end()};
}

DeltaNQ delta_n_q(const ReferenceLattice& lattice) {
  DeltaNQ total = 0;
  for (const ReferenceSite& site : lattice.support()) {
    total += static_cast<std::int64_t>(occupied_channels(site.word)) -
             static_cast<std::int64_t>(occupied_channels(kQuiescentWord));
  }
  return total;
}

bool same_lattice(const ReferenceLattice& left, const ReferenceLattice& right) {
  return left.support() == right.support();
}

ReferenceLattice translated(const ReferenceLattice& lattice, const Z3Coordinate& displacement) {
  ReferenceLattice result;
  for (const ReferenceSite& site : lattice.support()) {
    result.write(site.coordinate + displacement, site.word);
  }
  return result;
}

bool is_basis_permutation(const BasisPermutation& permutation) {
  bool seen[4] = {false, false, false, false};
  for (const std::uint32_t basis : permutation) {
    if (basis >= 4u || seen[basis])
      return false;
    seen[basis] = true;
  }
  return true;
}

Z3Coordinate transformed_coordinate(const Z3Coordinate& coordinate,
                                    const BasisPermutation& permutation) {
  if (!is_basis_permutation(permutation)) {
    throw std::invalid_argument("BCC-32 basis transform is not a permutation");
  }
  const CoordinateComponent coefficients[3] = {coordinate.x, coordinate.y, coordinate.z};
  Z3Coordinate result{};
  for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
    const Int3 image = basis_offset(static_cast<Basis>(permutation[axis]));
    result.x += coefficients[axis] * image.x;
    result.y += coefficients[axis] * image.y;
    result.z += coefficients[axis] * image.z;
  }
  return result;
}

namespace {

void transfer_channel(SiteWord source, SiteWord source_bit, SiteWord& destination,
                      SiteWord destination_bit) {
  if ((source & source_bit) != 0u)
    destination |= destination_bit;
}

}  // namespace

SiteWord transformed_word(SiteWord word, const BasisPermutation& permutation) {
  if (!is_basis_permutation(permutation)) {
    throw std::invalid_argument("BCC-32 channel transform is not a permutation");
  }
  SiteWord result = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    const std::uint32_t image = permutation[basis];
    transfer_channel(word, carrier_bit(basis), result, carrier_bit(image));
    transfer_channel(word, carrier_bit(basis + 4u), result, carrier_bit(image + 4u));
    transfer_channel(word, face_bit(basis), result, face_bit(image));
    transfer_channel(word, face_bit(basis + 4u), result, face_bit(image + 4u));
    transfer_channel(word, owned_bond_bit(basis), result, owned_bond_bit(image));
    transfer_channel(word, channel_bit(kConformationShift, basis), result,
                     channel_bit(kConformationShift, image));
    transfer_channel(word, channel_bit(kReactiveShift, basis), result,
                     channel_bit(kReactiveShift, image));
    transfer_channel(word, energy_bit(basis), result, energy_bit(image));
  }
  return result;
}

ReferenceLattice tetrahedral_transform(const ReferenceLattice& lattice,
                                       const BasisPermutation& permutation) {
  ReferenceLattice result;
  for (const ReferenceSite& site : lattice.support()) {
    result.write(transformed_coordinate(site.coordinate, permutation),
                 transformed_word(site.word, permutation));
  }
  return result;
}

}  // namespace substrate::bcc32
