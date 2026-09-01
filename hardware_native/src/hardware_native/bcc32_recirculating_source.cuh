#pragma once

// Recirculating field source -- the gradient-ecology spec's central primitive,
// as a candidate local reaction under test. NOT yet wired into F.
//
// Why this shape, from measurement rather than from the spec:
//
//   adc02db33c  a written byte is DISPERSED by one tick of F, not displaced:
//               zero decodable pairs anywhere in the support, in any of six
//               directions, while conservation and exact inverse hold. So the
//               readable form cannot be recovered by any local pair decoder --
//               it has to be RECONSTITUTED.
//   bb51259987  fixed-address storage does not survive F on either stack, in a
//               body or isolated. Capacity was never the blocker.
//   2026-08-03  the existing-law source search is exhausted: tested=512
//               period_limit=69 candidates=0 NO_CLOSED_SOURCE_ORBIT, and the
//               follow-up F-native probe failed as an ANATOMY failure -- "the
//               current F prefix transforms the candidate before the proposed
//               return operation can preserve a source-and-return orbit."
//
// Those three say the same thing from three directions: state that is parked
// does not survive, and state that is merely transported cannot be read. What
// is left is state that LEAVES AND COMES BACK, with the return reconstituting
// the readable form.
//
// The cycle:
//
//   reservoir --emit--> outbound packet --...F carries it...--> inbound packet
//        ^                                                            |
//        +---------------------- return -----------------------------+
//
// No packet disappears. A local gradient is represented by arrival duty cycle,
// dwell time, directional imbalance and return latency -- never by a float
// concentration, a region ID, or a stored word at an address.
//
// EXACT REVERSIBILITY BY CONSTRUCTION. Every operation below is a permutation
// of the local state: emit and return are inverse partial maps guarded by
// disjoint predicates, so the pair composes to the identity on any state where
// both guards can fire in sequence. Nothing is added and nothing is destroyed;
// a quantum moves between the reservoir lane and the packet lane and back.
//
// WHAT THIS IS NOT. There is no semantic source, no destination coordinate, no
// region label, no host scheduler, and no private journal. The reaction is
// target-independent: it reads only its own two lanes and a marker.

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

namespace substrate::bcc32::recirculating_source {

using substrate::bcc32::SiteWord;

// Lane layout on one site word. Reservoir and packet are dual-rail counted in
// the same convention every resident factor in this tree already uses, so a
// quantum moved between them is bit-count neutral by construction.
inline constexpr std::uint32_t kReservoirShift = 0u;
inline constexpr std::uint32_t kPacketShift = 8u;
inline constexpr std::uint32_t kPhaseShift = 16u;
inline constexpr SiteWord kLaneMask = 0xffu;

// Phase is the packet's position in its own cycle. It is ORDINARY MATTER on the
// site, not a host variable, and it is what makes emit and return distinguish
// themselves without a scheduler.
enum Phase : std::uint32_t {
  kAtRest = 0u,     // quantum in the reservoir, nothing outbound
  kOutbound = 1u,   // quantum has left, has not yet returned
  kInbound = 2u,    // quantum is back adjacent, return not yet taken
};

[[nodiscard]] __host__ __device__ inline std::uint32_t lane(SiteWord word,
                                                            std::uint32_t shift) {
  return static_cast<std::uint32_t>((word >> shift) & kLaneMask);
}

[[nodiscard]] __host__ __device__ inline SiteWord with_lane(SiteWord word,
                                                            std::uint32_t shift,
                                                            std::uint32_t value) {
  return (word & ~(kLaneMask << shift)) |
         (static_cast<SiteWord>(value & kLaneMask) << shift);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t phase(SiteWord word) {
  return lane(word, kPhaseShift);
}

// EMIT. Guard: at rest with a non-empty reservoir. One quantum moves reservoir
// -> packet and the phase becomes outbound. Refusing when the reservoir is
// empty is an ABSTENTION, not a clamp: no lane is touched, so the operation is
// still a permutation on the states where it fires.
[[nodiscard]] __host__ __device__ inline SiteWord emit(SiteWord word) {
  if (phase(word) != kAtRest) return word;
  const std::uint32_t reservoir = lane(word, kReservoirShift);
  if (reservoir == 0u) return word;
  SiteWord out = with_lane(word, kReservoirShift, reservoir - 1u);
  out = with_lane(out, kPacketShift, lane(word, kPacketShift) + 1u);
  return with_lane(out, kPhaseShift, kOutbound);
}

// ARRIVE. The packet has been carried and is now adjacent again. This is the
// only step that depends on transport having happened; it moves no quantum, it
// only advances the phase, so it cannot manufacture matter.
[[nodiscard]] __host__ __device__ inline SiteWord arrive(SiteWord word) {
  if (phase(word) != kOutbound) return word;
  return with_lane(word, kPhaseShift, kInbound);
}

// RETURN. Guard: inbound with a non-empty packet lane. The quantum moves back
// packet -> reservoir and the phase returns to rest. This is the exact inverse
// of emit composed with arrive, which is what reconstitutes the readable form.
[[nodiscard]] __host__ __device__ inline SiteWord take_return(SiteWord word) {
  if (phase(word) != kInbound) return word;
  const std::uint32_t packet = lane(word, kPacketShift);
  if (packet == 0u) return word;
  SiteWord out = with_lane(word, kPacketShift, packet - 1u);
  out = with_lane(out, kReservoirShift, lane(word, kReservoirShift) + 1u);
  return with_lane(out, kPhaseShift, kAtRest);
}

// Exact inverses, each the mirror of its forward partner.
[[nodiscard]] __host__ __device__ inline SiteWord undo_emit(SiteWord word) {
  if (phase(word) != kOutbound) return word;
  const std::uint32_t packet = lane(word, kPacketShift);
  if (packet == 0u) return word;
  SiteWord out = with_lane(word, kPacketShift, packet - 1u);
  out = with_lane(out, kReservoirShift, lane(word, kReservoirShift) + 1u);
  return with_lane(out, kPhaseShift, kAtRest);
}

[[nodiscard]] __host__ __device__ inline SiteWord undo_arrive(SiteWord word) {
  if (phase(word) != kInbound) return word;
  return with_lane(word, kPhaseShift, kOutbound);
}

[[nodiscard]] __host__ __device__ inline SiteWord undo_return(SiteWord word) {
  if (phase(word) != kAtRest) return word;
  const std::uint32_t reservoir = lane(word, kReservoirShift);
  if (reservoir == 0u) return word;
  SiteWord out = with_lane(word, kReservoirShift, reservoir - 1u);
  out = with_lane(out, kPacketShift, lane(word, kPacketShift) + 1u);
  return with_lane(out, kPhaseShift, kInbound);
}

// The conserved quantity: quanta are neither created nor destroyed by any step
// above, so reservoir + packet is invariant across a whole cycle. This is the
// local analogue of exact DeltaN_Q and is checked directly in the contract.
[[nodiscard]] __host__ __device__ inline std::uint32_t total_quanta(SiteWord word) {
  return lane(word, kReservoirShift) + lane(word, kPacketShift);
}

}  // namespace substrate::bcc32::recirculating_source
