#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>

#include <boost/multiprecision/cpp_int.hpp>

namespace substrate::bcc32 {

// MEASURED AND REJECTED, recorded so it is not retried blind: replacing this
// with `cpp_int_backend<128, 0, ...>` -- same arbitrary precision, but with a
// 128-bit internal buffer so small components never touch the heap -- compiled
// clean, kept every contract byte-identical, and bought NOTHING (46.60s ->
// 46.58s on the composition witness, inside noise).
//
// ⛔ AND THE FIRST EXPLANATION FOR THAT NULL WAS PROBABLY WRONG. It read: "the
// residual is variable-length bookkeeping, so the fix is a fixed-width fast
// path for this component." External review supplied a more mundane and more
// likely account: DEFAULT `cpp_int` (MinBits == 0) ALREADY HAS INLINE CAPACITY
// -- Boost sizes it to fill the union occupied by the dynamic-storage header.
// Every coordinate in this project is a handful of bits, so those values were
// very likely allocation-free BEFORE the experiment. Enlarging an already
// sufficient buffer changes nothing, which is exactly what was measured.
// ⇒ Before anyone infers a mechanism from that null again: instrument the
// allocator and count allocations attributable to coordinate operations.
//
// ⛔⛔ DO NOT GIVE THIS TYPE A SMALL-OR-BIG FAST PATH. It was the named next
// target for about an hour and it is the wrong move: a fast path inside the
// canonical component contaminates serialization, equality, ordering, and
// hashing for every consumer -- and a stale or non-canonical small form makes a
// key logically unreachable in a hash container -- while leaving intact the
// algorithmic pattern that actually produces the flat profile.
//
// ⭐ THE ACTUAL NEXT TARGET IS THE EXECUTION SHAPE, NOT THE INTEGER. The law is
// local, yet the interpreter performs thousands of INDEPENDENT exact sparse-map
// lookups per step, reconstructing three multiprecision components and rehashing
// for neighbours it just visited. Keep this type canonical at ingress, egress,
// checkpoints and aperture mapping; add an execution view of exact origin plus
// CHECKED native relative offsets (chunk- or aperture-shaped), run the law over
// dense local slots, and commit changed sites back through exact arithmetic.
// See docs/audits/2026-08-05-reference-lattice-latency.md for the profile, the
// enumerated failure modes, and why each cheaper-looking option is worse.
using CoordinateComponent = boost::multiprecision::cpp_int;

// One exact logical coordinate type is shared by the reference law and durable
// world storage. GPU executors receive only bounded aperture-local topology;
// they never narrow or reinterpret this material identity.
struct ExactCoordinate {
    CoordinateComponent x = 0;
    CoordinateComponent y = 0;
    CoordinateComponent z = 0;

    friend bool operator==(const ExactCoordinate&, const ExactCoordinate&) = default;
};

using Z3Coordinate = ExactCoordinate;
using SiteCoord = ExactCoordinate;
using ChunkCoord = ExactCoordinate;

struct CoordinateLess {
    [[nodiscard]] bool operator()(const ExactCoordinate& left,
                                  const ExactCoordinate& right) const;
};

using Z3CoordinateLess = CoordinateLess;

// A cheap hash so associative LOOKUP stops paying O(log n) comparisons of
// three arbitrary-precision integers. This changes no ordering anywhere:
// CoordinateLess remains the only order, and every observable that exposes an
// order still sorts with it.
//
// MEASURED, and the reason this reads the way it does: a first version masked
// each component to its low 64 bits with `value & mask` before narrowing. That
// is a cpp_int BITWISE AND, which allocates an arbitrary-precision result on
// every lookup, and it went straight to the top of the profile at 17.1% --
// more than ReferenceLattice::read itself. It traded O(log n) cheap
// comparisons for one very expensive allocation and bought almost nothing.
//
// Boost specializes std::hash for its number types and hashes the limb array
// directly: no arithmetic, no allocation, and total over every magnitude, so
// nothing has to assume a component fits in a machine word.
struct CoordinateHash {
    [[nodiscard]] std::size_t operator()(const ExactCoordinate& coordinate) const {
        const std::hash<CoordinateComponent> component_hash;
        std::size_t mixed = component_hash(coordinate.x);
        mixed ^= component_hash(coordinate.y) + 0x9e3779b97f4a7c15ull + (mixed << 6) +
                 (mixed >> 2);
        mixed ^= component_hash(coordinate.z) + 0x9e3779b97f4a7c15ull + (mixed << 6) +
                 (mixed >> 2);
        return mixed;
    }
};

using Z3CoordinateHash = CoordinateHash;

[[nodiscard]] ExactCoordinate operator+(const ExactCoordinate& left,
                                        const ExactCoordinate& right);
[[nodiscard]] ExactCoordinate operator-(const ExactCoordinate& left,
                                        const ExactCoordinate& right);
[[nodiscard]] CoordinateComponent floor_divide(
    const CoordinateComponent& value,
    std::uint32_t positive_divisor);
[[nodiscard]] std::uint32_t floor_modulo(
    const CoordinateComponent& value,
    std::uint32_t positive_modulus);
[[nodiscard]] std::string canonical_coordinate_component(
    const CoordinateComponent& value);

}  // namespace substrate::bcc32
