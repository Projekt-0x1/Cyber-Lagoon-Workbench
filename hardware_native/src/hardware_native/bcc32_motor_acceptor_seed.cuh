#pragma once

// Compact matter-only seed for the first lawful motor acceptor grammar.
//
// The motor's measured landing word carries DeltaNQ=+2, whereas a world
// unit-object carries +1.  A one-cell receiver cannot remove that surplus.
// This seed therefore supplies one zero-net neighbour: one owned-bond quantum
// and one carrier hole.  It is a possible *place for the surplus to go*, not
// a serialized reaction, timer, target cell, or output instruction.  F alone
// decides whether any hash actually becomes a transducer.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using MotorAcceptorSeedHash = std::uint16_t;

inline constexpr std::uint32_t kMotorAcceptorBondShift = 0u;
inline constexpr std::uint32_t kMotorAcceptorHoleShift = 2u;
inline constexpr std::uint32_t kMotorAcceptorPlacementShift = 5u;
inline constexpr std::uint32_t kMotorAcceptorCatalystBasisShift = 8u;
inline constexpr std::uint32_t kMotorAcceptorCatalystPlacementShift = 10u;
inline constexpr MotorAcceptorSeedHash kMotorAcceptorSeedMask = 0x1fffu;
inline constexpr std::size_t kMotorAcceptorSeedSiteCount = 1u;

constexpr std::uint32_t motor_acceptor_bond_basis(MotorAcceptorSeedHash hash) {
  return (hash >> kMotorAcceptorBondShift) & 0x3u;
}

constexpr std::uint32_t motor_acceptor_hole_lane(MotorAcceptorSeedHash hash) {
  return (hash >> kMotorAcceptorHoleShift) & 0x7u;
}

constexpr std::uint32_t motor_acceptor_placement(MotorAcceptorSeedHash hash) {
  return (hash >> kMotorAcceptorPlacementShift) & 0x7u;
}

constexpr std::uint32_t motor_acceptor_catalyst_basis(MotorAcceptorSeedHash hash) {
  return (hash >> kMotorAcceptorCatalystBasisShift) & 0x3u;
}

constexpr std::uint32_t motor_acceptor_catalyst_placement(MotorAcceptorSeedHash hash) {
  return (hash >> kMotorAcceptorCatalystPlacementShift) & 0x7u;
}

constexpr MotorAcceptorSeedHash make_motor_acceptor_seed_hash(std::uint32_t bond_basis,
                                                               std::uint32_t hole_lane,
                                                               std::uint32_t placement) {
  return static_cast<MotorAcceptorSeedHash>(
      ((bond_basis & 0x3u) << kMotorAcceptorBondShift) |
      ((hole_lane & 0x7u) << kMotorAcceptorHoleShift) |
      ((placement & 0x7u) << kMotorAcceptorPlacementShift));
}

constexpr MotorAcceptorSeedHash make_motor_acceptor_br_catalyst_seed_hash(
    std::uint32_t bond_basis, std::uint32_t hole_lane, std::uint32_t placement,
    std::uint32_t catalyst_basis, std::uint32_t catalyst_placement) {
  return static_cast<MotorAcceptorSeedHash>(
      make_motor_acceptor_seed_hash(bond_basis, hole_lane, placement) |
      ((catalyst_basis & 0x3u) << kMotorAcceptorCatalystBasisShift) |
      ((catalyst_placement & 0x7u) << kMotorAcceptorCatalystPlacementShift));
}

constexpr SiteWord motor_acceptor_sink_word(MotorAcceptorSeedHash hash) {
  return static_cast<SiteWord>((kQ & ~carrier_bit(motor_acceptor_hole_lane(hash))) |
                               owned_bond_bit(motor_acceptor_bond_basis(hash)));
}

// The canonical C0 motor lands at (-1,0,0).  The hash chooses exactly one of
// that landing site's eight BCC neighbours.  Direction +u0 is the emitter and
// is rejected by the contract rather than silently overlapping founder matter.
constexpr DevelopmentalSeedSite motor_acceptor_sink_site(MotorAcceptorSeedHash hash) {
  const Int3 offset = direction_offset(static_cast<Direction>(motor_acceptor_placement(hash)));
  return {static_cast<std::int8_t>(-1 + offset.x), static_cast<std::int8_t>(offset.y),
          static_cast<std::int8_t>(offset.z), motor_acceptor_sink_word(hash)};
}

constexpr std::array<DevelopmentalSeedSite, kMotorAcceptorSeedSiteCount> motor_acceptor_seed(
    MotorAcceptorSeedHash hash) {
  return {{motor_acceptor_sink_site(hash)}};
}

// The third cell is not an arbitrary palette extension.  B/R controlled by a
// negative face is native edge gate 11, so an R|X- neighbour is the first
// deterministic catalyst that can interact with the B+hole sink by an
// established local rule.
constexpr DevelopmentalSeedSite motor_acceptor_br_catalyst_site(MotorAcceptorSeedHash hash) {
  const DevelopmentalSeedSite sink = motor_acceptor_sink_site(hash);
  const Int3 offset =
      direction_offset(static_cast<Direction>(motor_acceptor_catalyst_placement(hash)));
  return {static_cast<std::int8_t>(sink.x + offset.x), static_cast<std::int8_t>(sink.y + offset.y),
          static_cast<std::int8_t>(sink.z + offset.z),
          static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, motor_acceptor_catalyst_basis(hash)) |
                               face_bit(motor_acceptor_catalyst_basis(hash) + 4u))};
}

constexpr std::array<DevelopmentalSeedSite, 2u> motor_acceptor_br_catalyst_seed(
    MotorAcceptorSeedHash hash) {
  return {{motor_acceptor_sink_site(hash), motor_acceptor_br_catalyst_site(hash)}};
}

// Gate 6 is the first native edge rule that addresses the measured surplus
// itself: transpose R with P- under C control.  The canonical motor's surplus
// is P4 at landing-u0, so this compact receiver is fixed by that observed
// interface rather than by a chosen output time.
using MotorGate6AcceptorSeedHash = std::uint8_t;
inline constexpr MotorGate6AcceptorSeedHash kMotorGate6AcceptorSeedHash = 0u;

constexpr SiteWord motor_gate6_receiver_word(std::uint32_t basis) {
  return static_cast<SiteWord>(kQ | channel_bit(kConformationShift, basis) |
                               channel_bit(kReactiveShift, basis));
}

constexpr DevelopmentalSeedSite motor_gate6_receiver_site(MotorGate6AcceptorSeedHash hash) {
  const std::uint32_t basis = hash & 0x3u;
  // The canonical motor emits on basis 0; other values only expose the
  // same compact local word for explicit wrong-basis controls.
  const Int3 offset = direction_offset(static_cast<Direction>(basis + 4u));
  return {static_cast<std::int8_t>(-1 + offset.x), static_cast<std::int8_t>(offset.y),
          static_cast<std::int8_t>(offset.z), motor_gate6_receiver_word(basis)};
}

static_assert(motor_acceptor_sink_word(make_motor_acceptor_seed_hash(0u, 0u, 1u)) ==
              ((kQ & ~carrier_bit(0u)) | owned_bond_bit(0u)));

}  // namespace substrate::bcc32
