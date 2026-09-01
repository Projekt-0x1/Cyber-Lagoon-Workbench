#include "bcc32_reference_world_bridge.hpp"

#include <algorithm>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {
namespace {

ExactCoordinate translated(const Z3Coordinate& at, const ExactCoordinate& origin) {
  return {at.x + origin.x, at.y + origin.y, at.z + origin.z};
}

}  // namespace

std::vector<ReferenceSite> translated_support(const ReferenceLattice& lattice,
                                              const ExactCoordinate& origin) {
  std::vector<ReferenceSite> result = lattice.support();
  for (ReferenceSite& site : result)
    site.coordinate = translated(site.coordinate, origin);
  return result;
}

bool write_lattice_into_store(const ReferenceLattice& lattice,
                              const ExactCoordinate& origin, WorldStore* store,
                              std::string* error) {
  if (store == nullptr) {
    if (error != nullptr) *error = "null world store";
    return false;
  }
  // support() is already compacted to non-quiescent sites, so this writes
  // exactly the route's matter and never fabricates a quiescent word.
  for (const ReferenceSite& site : lattice.support()) {
    if (site.word == kQ) {
      if (error != nullptr) *error = "quiescent word in reference support";
      return false;
    }
    if (!store->write_site(translated(site.coordinate, origin), site.word,
                           error))
      return false;
  }
  return true;
}

bool read_lattice_from_store(const WorldStore& store,
                             const ExactCoordinate& origin,
                             std::span<const Z3Coordinate> coordinates,
                             ReferenceLattice* lattice, std::string* error) {
  if (lattice == nullptr) {
    if (error != nullptr) *error = "null reference lattice";
    return false;
  }
  std::vector<ReferenceSite> support;
  support.reserve(coordinates.size());
  for (const Z3Coordinate& at : coordinates) {
    const SiteWord word = store.read_site(translated(at, origin));
    // A quiescent read is not an error: it means the store legitimately holds
    // no matter there.  It is simply absent from the rebuilt support, which is
    // exactly how ReferenceLattice represents it.
    if (word == kQ) continue;
    support.push_back(ReferenceSite{at, word});
  }
  // replace_support rejects duplicate coordinates, so a caller that passes the
  // same coordinate twice is reported rather than silently collapsed.
  try {
    lattice->replace_support(support);
  } catch (const std::exception& failure) {
    if (error != nullptr) *error = failure.what();
    return false;
  }
  return true;
}

bool same_support(const ReferenceLattice& left, const ReferenceLattice& right) {
  const std::vector<ReferenceSite> left_support = left.support();
  const std::vector<ReferenceSite> right_support = right.support();
  return left_support == right_support;
}

}  // namespace substrate::bcc32
