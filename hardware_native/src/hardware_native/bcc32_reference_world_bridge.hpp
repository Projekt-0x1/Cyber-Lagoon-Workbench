#pragma once

#include <span>
#include <string>
#include <vector>

#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"
#include "bcc32_world_store.hpp"

namespace substrate::bcc32 {

// Exact translation between the reference law's sparse per-site lattice and the
// durable chunked world store.
//
// WHY THIS EXISTS.  Plan section 5G.5 items 1 and 2 both terminate in the same
// edit: the conditioned-matter owner advances every route's matter as a
// ReferenceLattice on the CPU, and `PagedWorldExecutor::advance_store` operates
// on a WorldStore.  Before this header, ZERO files under src/hardware_native/
// named both types -- the bridge did not exist, and it is the true blocker for
// both items.  See docs/audits/2026-07-28-conditioned-matter-capacity-wall.md.
//
// WHY THE ROUND TRIP CAN BE EXACT.  Both representations agree on quiescence:
//   bcc32_law.cuh:15-16   constexpr SiteWord kQ = 0x000000ffu;
//                         static_assert(kQ == kCarrierMask);
//   bcc32_types.cuh:37    constexpr SiteWord kQuiescentWord = kCarrierMask;
// ReferenceLattice compacts kQ entries out of its support; WorldStore treats an
// absent chunk as all-quiescent and erases a chunk that becomes all-quiescent.
// So "absent" means the same word in both, and support maps word-for-word.
//
// THIS HEADER MOVES MATTER; IT DOES NOT ADVANCE IT.  No law step, no schedule,
// no page publication happens here.  It is a representation change only, and it
// is deliberately reversible so a contract can prove it lost nothing.
//
// PLACEMENT IS THE CALLER'S, AND IT IS NOT COSMETIC.  A WorldStore chunk is
// kChunkSites = 1,000,000 sites = 4 MB, materialized densely on first write to
// that chunk (bcc32_world_store.cpp:110-118).  A conditioned route occupies 32
// sites.  Giving each route its own store, or its own chunk, therefore costs
// 4 MB per route regardless of the 32 sites used.  Routes MUST be translated to
// disjoint offsets inside a SHARED store so that many routes share one chunk.
// `translated_support` exists to make that placement explicit at the call site
// rather than implicit in a per-entry store.

// Returns `lattice`'s support translated by `origin`.  Coordinates only; the
// words are carried unchanged.
[[nodiscard]] std::vector<ReferenceSite> translated_support(
    const ReferenceLattice& lattice, const ExactCoordinate& origin);

// Writes every non-quiescent site of `lattice`, translated by `origin`, into
// `store`.  Fails without partial effect only if `store` is otherwise untouched
// by the caller; on an addressing failure it reports the error and stops, so a
// caller that needs all-or-nothing must stage into a scratch store first.
bool write_lattice_into_store(const ReferenceLattice& lattice,
                              const ExactCoordinate& origin, WorldStore* store,
                              std::string* error);

// Reads the words at `coordinates` (each translated by `origin`) out of `store`
// and rebuilds a lattice at the UNtranslated coordinates.  The caller supplies
// the coordinate list because a shared store legitimately holds other routes'
// matter; reading "everything non-quiescent" would harvest a neighbour.
bool read_lattice_from_store(const WorldStore& store,
                             const ExactCoordinate& origin,
                             std::span<const Z3Coordinate> coordinates,
                             ReferenceLattice* lattice, std::string* error);

// True when `left` and `right` have identical support: same coordinates, same
// words, same count.  Used by the round-trip contract.
[[nodiscard]] bool same_support(const ReferenceLattice& left,
                                const ReferenceLattice& right);

}  // namespace substrate::bcc32
