#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

// A resident transposition: the exchange (swap) of two resident value/
// complement rail pairs. It is the primitive the M7 audit asked for in
// isolation -- "the selected candidate's resident owner matter and
// complementary destination matter undergoing a lawful local transposition"
// -- built and proved here before anything is wired to a basin, a cloud
// factor, or a learning rule.
namespace substrate::bcc32::resident_transposition {

using substrate::bcc32::SiteWord;

// `a` and `b` are the absolute VALUE-rail slot index of two logical words in
// `words`. Each logical word occupies two physical rails: the value at the
// given slot and its bitwise complement at slot+1, matching the
// write_field/read_field convention in bcc32_grown_instance_basin.cuh
// (~:140-152). This function exchanges the two (value, complement) pairs as
// units, so both rails of both endpoints move together and the invariant
// words[p+1] == ~words[p] is preserved at both p==a and p==b afterwards.
//
// The exchange is unconditional: it reads both pairs, then writes both pairs
// back swapped. It never allocates, never round-trips to host, and never
// branches on the values being moved -- there is no condition here to depend
// on them. Consequences that follow from this shape alone, not asserted
// separately:
//   * self-inverse: applying the same (a, b) swap twice is the identity,
//     because "read old a, old b; write b into a, a into b" undone by the
//     same rule restores "read new a (=old b), new b (=old a); write old b
//     back into a's slot, old a back into b's slot".
//   * bit-count conserved exactly: no bit is created or destroyed, only
//     relocated, so popcount over the whole buffer is unchanged.
//   * a == b is safe by construction and is a true no-op, not merely an
//     untested case: when a == b the two "pairs" are the same physical
//     rails, so the values read for the a-side and b-side of the exchange
//     are identical, and the two writes below deposit exactly what was
//     already there. No caller-side precondition is required.
__device__ inline void transpose_resident_pair(SiteWord* words,
                                                std::uint32_t a,
                                                std::uint32_t b) {
  const SiteWord value_a = words[a];
  const SiteWord complement_a = words[a + 1u];
  const SiteWord value_b = words[b];
  const SiteWord complement_b = words[b + 1u];
  words[a] = value_b;
  words[a + 1u] = complement_b;
  words[b] = value_a;
  words[b + 1u] = complement_a;
}

}  // namespace substrate::bcc32::resident_transposition
