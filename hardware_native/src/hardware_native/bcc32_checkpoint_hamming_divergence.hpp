#pragma once

#include "bcc32_world_store.hpp"

#include <cstdint>

namespace substrate::bcc32 {

// Bitwise Hamming distance between two world snapshots: the number of bits
// that differ across every chunk present in either store. A chunk absent
// from one store reads as fully kQuiescentWord (WorldStore::read_chunk's own
// contract), so this is well-defined even when the two stores materialize
// different chunk footprints.
//
// This is the measurement primitive for the two-checkpoint divergence
// falsifier: run identical future contact from S_0 and from a copy S_0' that
// differs by one preregistered bit, then track D(t) = checkpoint_hamming_distance
// across successive checkpoints. Structured divergence away from zero without
// saturating near the maximum is the target regime; this function only
// computes D at one pair of snapshots; comparing across time is the caller's
// job.
[[nodiscard]] std::uint64_t checkpoint_hamming_distance(const WorldStore& a,
                                                         const WorldStore& b);

}  // namespace substrate::bcc32
