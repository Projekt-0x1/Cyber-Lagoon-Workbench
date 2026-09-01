#pragma once

#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"
#include "bcc32_raw_byte_tape.cuh"

namespace substrate::bcc32 {

// This is one finite appraisal/selection episode. It is a fixed contact
// schedule around four ordinary complete F steps, not a learning rule.
constexpr std::uint32_t kEndogenousRoleSwitchBits = 8u;

template <typename T, std::size_t Count>
struct EndogenousRoleSwitchArray {
  T values[Count]{};

  [[nodiscard]] __host__ __device__ constexpr T& operator[](std::size_t index) {
    return values[index];
  }
  [[nodiscard]] __host__ __device__ constexpr const T& operator[](std::size_t index) const {
    return values[index];
  }

  friend constexpr bool operator==(const EndogenousRoleSwitchArray&,
                                   const EndogenousRoleSwitchArray&) = default;
};

struct EndogenousRoleSwitchLaneWords {
  SiteWord active_source = kQ;
  SiteWord role_source = kQ;
  SiteWord shadow_destination = kQ;
  SiteWord role_destination = kQ;

  friend constexpr bool operator==(const EndogenousRoleSwitchLaneWords&,
                                   const EndogenousRoleSwitchLaneWords&) = default;
};

struct EndogenousRoleSwitchTape {
  RawByteRails route_surface{};
  RawByteRails route_tape{};
  RawByteRails shadow_route_surface{};
  RawByteRails shadow_route_tape{};
  RawByteRails actual_surface{};
  RawByteRails actual_tape{};
  RawByteRails selector_route_surface{};
  RawByteRails selector_route_tape{};
  RawByteRails motor_surface{};
  RawByteRails motor_tape{};
  EndogenousRoleSwitchArray<SiteWord, kEndogenousRoleSwitchBits> converter{};
  EndogenousRoleSwitchArray<SiteWord, kEndogenousRoleSwitchBits> rotor{};
  EndogenousRoleSwitchArray<SiteWord, kEndogenousRoleSwitchBits> selector_source{};
  EndogenousRoleSwitchArray<SiteWord, kEndogenousRoleSwitchBits> selector_destination{};

  friend constexpr bool operator==(const EndogenousRoleSwitchTape&,
                                   const EndogenousRoleSwitchTape&) = default;
};

[[nodiscard]] __host__ __device__ constexpr SiteWord role_switch_tape_bit(std::uint32_t index) {
  return SiteWord{1u} << index;
}

// All converter contacts receive the value which the scratch conduit directly
// prepared. A reciprocal contact then leaves that value at the world port and
// retains the displaced bit in this represented tape.
constexpr SiteWord kRoleSwitchConverterTargets =
    role_switch_tape_bit(3u) | role_switch_tape_bit(9u) | role_switch_tape_bit(14u) |
    role_switch_tape_bit(15u) | role_switch_tape_bit(16u);

// The third F sees the two P- rotor controls, three P+ controls, and the role
// edge in this exact state. The tape is fixed before the episode begins.
constexpr SiteWord kRoleSwitchRotorTargets = role_switch_tape_bit(2u) | role_switch_tape_bit(3u) |
                                             role_switch_tape_bit(5u) | role_switch_tape_bit(6u) |
                                             role_switch_tape_bit(7u) | role_switch_tape_bit(10u);

// F3 is the ordinary selector fixture from the exact exploratory table. The
// source receives route and its complement from represented dual rails; every
// other channel is restored by fixed contacts. C0 at the destination is the
// sole preserved resident role and is never inspected by a boundary gate.
constexpr SiteWord kRoleSwitchSelectorSourceTargets = kQ;
constexpr SiteWord kRoleSwitchSelectorDestinationTargets = kQ;

[[nodiscard]] __host__ __device__ constexpr EndogenousRoleSwitchLaneWords
endogenous_role_switch_initial_lane(bool role) {
  EndogenousRoleSwitchLaneWords lane{};

  lane.active_source &= ~(carrier_bit(0u) | face_bit(0u) | owned_bond_bit(0u) | energy_bit(0u));

  lane.role_source &=
      ~(carrier_bit(0u) | carrier_bit(1u) | carrier_bit(2u) | carrier_bit(3u) | carrier_bit(4u) |
        face_bit(1u) | face_bit(4u) | face_bit(0u) | owned_bond_bit(0u) | owned_bond_bit(1u) |
        energy_bit(0u) | energy_bit(1u) | channel_bit(kConformationShift, 0u));
  lane.role_source |= channel_bit(kReactiveShift, 0u);

  lane.shadow_destination &=
      ~(carrier_bit(5u) | face_bit(5u) | channel_bit(kConformationShift, 1u));
  lane.shadow_destination |= channel_bit(kReactiveShift, 1u);

  lane.role_destination &=
      ~(face_bit(4u) | channel_bit(kReactiveShift, 0u) | channel_bit(kConformationShift, 0u));
  lane.role_destination |= carrier_bit(4u);
  if (role)
    lane.role_destination |= channel_bit(kConformationShift, 0u);
  return lane;
}

[[nodiscard]] __host__ __device__ constexpr EndogenousRoleSwitchTape
endogenous_role_switch_initial_tape(std::uint8_t route, std::uint8_t actual,
                                    std::uint8_t displaced_motor = 0u) {
  EndogenousRoleSwitchTape tape{};
  tape.route_surface = with_raw_byte_carriers(tape.route_surface, 0u);
  tape.route_tape = with_raw_byte_faces(tape.route_tape, route);
  tape.shadow_route_surface = with_raw_byte_carriers(tape.shadow_route_surface, 0u);
  tape.shadow_route_tape = with_raw_byte_faces(tape.shadow_route_tape, route);
  tape.actual_surface = with_raw_byte_carriers(tape.actual_surface, 0u);
  tape.actual_tape = with_raw_byte_faces(tape.actual_tape, actual);
  tape.selector_route_surface = with_raw_byte_carriers(tape.selector_route_surface, 0u);
  tape.selector_route_tape = with_raw_byte_faces(tape.selector_route_tape, route);
  tape.motor_surface = with_raw_byte_carriers(tape.motor_surface, 0u);
  tape.motor_tape = with_raw_byte_faces(tape.motor_tape, displaced_motor);
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    tape.converter[bit] = kRoleSwitchConverterTargets;
    tape.rotor[bit] = kRoleSwitchRotorTargets;
    tape.selector_source[bit] = kRoleSwitchSelectorSourceTargets;
    tape.selector_destination[bit] = kRoleSwitchSelectorDestinationTargets;
  }
  return tape;
}

__host__ __device__ constexpr void endogenous_role_switch_contact(SiteWord& world_word,
                                                                  SiteWord world_bit,
                                                                  SiteWord& tape_word,
                                                                  std::uint32_t tape_index) {
  reciprocal_quantum_exchange(world_word, world_bit, tape_word, role_switch_tape_bit(tape_index));
}

__host__ __device__ constexpr void endogenous_role_switch_input_forward(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  reciprocal_raw_byte_exchange(tape.route_surface, tape.route_tape);
  reciprocal_raw_byte_exchange(tape.shadow_route_surface, tape.shadow_route_tape);
  reciprocal_raw_byte_exchange(tape.actual_surface, tape.actual_tape);
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    reciprocal_quantum_exchange(lanes[bit].active_source, carrier_bit(0u), tape.route_surface.one,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_source, carrier_bit(1u),
                                tape.shadow_route_surface.one, carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_source, carrier_bit(4u), tape.actual_surface.one,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].shadow_destination, carrier_bit(5u),
                                tape.actual_surface.zero, carrier_bit(bit));
  }
}

__host__ __device__ constexpr void endogenous_role_switch_input_inverse(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t reverse = kEndogenousRoleSwitchBits; reverse > 0u; --reverse) {
    const std::uint32_t bit = reverse - 1u;
    reciprocal_quantum_exchange(lanes[bit].shadow_destination, carrier_bit(5u),
                                tape.actual_surface.zero, carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_source, carrier_bit(4u), tape.actual_surface.one,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_source, carrier_bit(1u),
                                tape.shadow_route_surface.one, carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].active_source, carrier_bit(0u), tape.route_surface.one,
                                carrier_bit(bit));
  }
  reciprocal_raw_byte_exchange(tape.actual_surface, tape.actual_tape);
  reciprocal_raw_byte_exchange(tape.shadow_route_surface, tape.shadow_route_tape);
  reciprocal_raw_byte_exchange(tape.route_surface, tape.route_tape);
}

__host__ __device__ constexpr void endogenous_role_switch_converter_forward(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& contact_tape = tape.converter[bit];
    endogenous_role_switch_contact(lane.active_source, carrier_bit(0u), contact_tape, 0u);
    endogenous_role_switch_contact(lane.active_source, face_bit(0u), contact_tape, 1u);
    endogenous_role_switch_contact(lane.active_source, owned_bond_bit(0u), contact_tape, 2u);
    endogenous_role_switch_contact(lane.active_source, energy_bit(0u), contact_tape, 3u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(4u), contact_tape, 4u);
    endogenous_role_switch_contact(lane.role_source, face_bit(4u), contact_tape, 5u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(1u), contact_tape, 6u);
    endogenous_role_switch_contact(lane.role_source, face_bit(1u), contact_tape, 7u);
    endogenous_role_switch_contact(lane.role_source, owned_bond_bit(1u), contact_tape, 8u);
    endogenous_role_switch_contact(lane.role_source, energy_bit(1u), contact_tape, 9u);
    endogenous_role_switch_contact(lane.shadow_destination, carrier_bit(5u), contact_tape, 10u);
    endogenous_role_switch_contact(lane.shadow_destination, face_bit(5u), contact_tape, 11u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(2u), contact_tape, 12u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(3u), contact_tape, 13u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(5u), contact_tape, 14u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(6u), contact_tape, 15u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(7u), contact_tape, 16u);
  }
}

__host__ __device__ constexpr void endogenous_role_switch_converter_inverse(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t reverse = kEndogenousRoleSwitchBits; reverse > 0u; --reverse) {
    const std::uint32_t bit = reverse - 1u;
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& contact_tape = tape.converter[bit];
    endogenous_role_switch_contact(lane.role_source, carrier_bit(7u), contact_tape, 16u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(6u), contact_tape, 15u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(5u), contact_tape, 14u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(3u), contact_tape, 13u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(2u), contact_tape, 12u);
    endogenous_role_switch_contact(lane.shadow_destination, face_bit(5u), contact_tape, 11u);
    endogenous_role_switch_contact(lane.shadow_destination, carrier_bit(5u), contact_tape, 10u);
    endogenous_role_switch_contact(lane.role_source, energy_bit(1u), contact_tape, 9u);
    endogenous_role_switch_contact(lane.role_source, owned_bond_bit(1u), contact_tape, 8u);
    endogenous_role_switch_contact(lane.role_source, face_bit(1u), contact_tape, 7u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(1u), contact_tape, 6u);
    endogenous_role_switch_contact(lane.role_source, face_bit(4u), contact_tape, 5u);
    endogenous_role_switch_contact(lane.role_source, carrier_bit(4u), contact_tape, 4u);
    endogenous_role_switch_contact(lane.active_source, energy_bit(0u), contact_tape, 3u);
    endogenous_role_switch_contact(lane.active_source, owned_bond_bit(0u), contact_tape, 2u);
    endogenous_role_switch_contact(lane.active_source, face_bit(0u), contact_tape, 1u);
    endogenous_role_switch_contact(lane.active_source, carrier_bit(0u), contact_tape, 0u);
  }
}

__host__ __device__ constexpr void endogenous_role_switch_rotor_forward(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& contact_tape = tape.rotor[bit];
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      endogenous_role_switch_contact(lane.role_source, carrier_bit(basis + 4u), contact_tape,
                                     basis);
      if (basis != 0u) {
        endogenous_role_switch_contact(lane.role_source, carrier_bit(basis), contact_tape,
                                       basis + 4u);
      }
    }
    endogenous_role_switch_contact(lane.role_source, face_bit(0u), contact_tape, 8u);
    endogenous_role_switch_contact(lane.role_source, owned_bond_bit(0u), contact_tape, 9u);
    endogenous_role_switch_contact(lane.role_destination, carrier_bit(4u), contact_tape, 10u);
    endogenous_role_switch_contact(lane.role_destination, face_bit(4u), contact_tape, 11u);
    endogenous_role_switch_contact(lane.role_destination, channel_bit(kReactiveShift, 0u),
                                   contact_tape, 12u);
  }
}

__host__ __device__ constexpr void endogenous_role_switch_rotor_inverse(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t reverse = kEndogenousRoleSwitchBits; reverse > 0u; --reverse) {
    const std::uint32_t bit = reverse - 1u;
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& contact_tape = tape.rotor[bit];
    endogenous_role_switch_contact(lane.role_destination, channel_bit(kReactiveShift, 0u),
                                   contact_tape, 12u);
    endogenous_role_switch_contact(lane.role_destination, face_bit(4u), contact_tape, 11u);
    endogenous_role_switch_contact(lane.role_destination, carrier_bit(4u), contact_tape, 10u);
    endogenous_role_switch_contact(lane.role_source, owned_bond_bit(0u), contact_tape, 9u);
    endogenous_role_switch_contact(lane.role_source, face_bit(0u), contact_tape, 8u);
    for (std::uint32_t basis = 4u; basis > 0u; --basis) {
      const std::uint32_t index = basis - 1u;
      if (index != 0u) {
        endogenous_role_switch_contact(lane.role_source, carrier_bit(index), contact_tape,
                                       index + 4u);
      }
      endogenous_role_switch_contact(lane.role_source, carrier_bit(index + 4u), contact_tape,
                                     index);
    }
  }
}

__host__ __device__ constexpr void endogenous_role_switch_selector_forward(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  reciprocal_raw_byte_exchange(tape.selector_route_surface, tape.selector_route_tape);
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& source_tape = tape.selector_source[bit];
    SiteWord& destination_tape = tape.selector_destination[bit];
    for (std::uint32_t channel = 0u; channel < 32u; ++channel) {
      const SiteWord target = role_switch_tape_bit(channel);
      if (target != carrier_bit(0u) && target != energy_bit(0u))
        endogenous_role_switch_contact(lane.role_source, target, source_tape, channel);
    }
    for (std::uint32_t channel = 0u; channel < 32u; ++channel) {
      const SiteWord target = role_switch_tape_bit(channel);
      if (target != channel_bit(kConformationShift, 0u))
        endogenous_role_switch_contact(lane.role_destination, target, destination_tape, channel);
    }
    reciprocal_quantum_exchange(lane.role_source, carrier_bit(0u), tape.selector_route_surface.one,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lane.role_source, energy_bit(0u), tape.selector_route_surface.zero,
                                carrier_bit(bit));
  }
}

__host__ __device__ constexpr void endogenous_role_switch_selector_inverse(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t reverse = kEndogenousRoleSwitchBits; reverse > 0u; --reverse) {
    const std::uint32_t bit = reverse - 1u;
    EndogenousRoleSwitchLaneWords& lane = lanes[bit];
    SiteWord& source_tape = tape.selector_source[bit];
    SiteWord& destination_tape = tape.selector_destination[bit];
    reciprocal_quantum_exchange(lane.role_source, energy_bit(0u), tape.selector_route_surface.zero,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lane.role_source, carrier_bit(0u), tape.selector_route_surface.one,
                                carrier_bit(bit));
    for (std::uint32_t channel = 32u; channel > 0u; --channel) {
      const std::uint32_t index = channel - 1u;
      const SiteWord target = role_switch_tape_bit(index);
      if (target != channel_bit(kConformationShift, 0u))
        endogenous_role_switch_contact(lane.role_destination, target, destination_tape, index);
    }
    for (std::uint32_t channel = 32u; channel > 0u; --channel) {
      const std::uint32_t index = channel - 1u;
      const SiteWord target = role_switch_tape_bit(index);
      if (target != carrier_bit(0u) && target != energy_bit(0u))
        endogenous_role_switch_contact(lane.role_source, target, source_tape, index);
    }
  }
  reciprocal_raw_byte_exchange(tape.selector_route_surface, tape.selector_route_tape);
}

__host__ __device__ constexpr void endogenous_role_switch_output_forward(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    reciprocal_quantum_exchange(lanes[bit].role_destination, carrier_bit(0u),
                                tape.motor_surface.one, carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_source, energy_bit(0u), tape.motor_surface.zero,
                                carrier_bit(bit));
  }
  reciprocal_raw_byte_exchange(tape.motor_surface, tape.motor_tape);
}

__host__ __device__ constexpr void endogenous_role_switch_output_inverse(
    EndogenousRoleSwitchArray<EndogenousRoleSwitchLaneWords, kEndogenousRoleSwitchBits>& lanes,
    EndogenousRoleSwitchTape& tape) {
  reciprocal_raw_byte_exchange(tape.motor_surface, tape.motor_tape);
  for (std::uint32_t reverse = kEndogenousRoleSwitchBits; reverse > 0u; --reverse) {
    const std::uint32_t bit = reverse - 1u;
    reciprocal_quantum_exchange(lanes[bit].role_source, energy_bit(0u), tape.motor_surface.zero,
                                carrier_bit(bit));
    reciprocal_quantum_exchange(lanes[bit].role_destination, carrier_bit(0u),
                                tape.motor_surface.one, carrier_bit(bit));
  }
}

[[nodiscard]] __host__ __device__ constexpr std::int64_t endogenous_role_switch_tape_delta_n_q(
    const EndogenousRoleSwitchTape& tape) {
  std::int64_t total = 0;
  const auto add = [&total](SiteWord word) {
    total += static_cast<std::int64_t>(occupied_channels(word)) - 8;
  };
  add(tape.route_surface.zero);
  add(tape.route_surface.one);
  add(tape.route_tape.zero);
  add(tape.route_tape.one);
  add(tape.shadow_route_surface.zero);
  add(tape.shadow_route_surface.one);
  add(tape.shadow_route_tape.zero);
  add(tape.shadow_route_tape.one);
  add(tape.actual_surface.zero);
  add(tape.actual_surface.one);
  add(tape.actual_tape.zero);
  add(tape.actual_tape.one);
  add(tape.selector_route_surface.zero);
  add(tape.selector_route_surface.one);
  add(tape.selector_route_tape.zero);
  add(tape.selector_route_tape.one);
  add(tape.motor_surface.zero);
  add(tape.motor_surface.one);
  add(tape.motor_tape.zero);
  add(tape.motor_tape.one);
  for (std::uint32_t bit = 0u; bit < kEndogenousRoleSwitchBits; ++bit) {
    add(tape.converter[bit]);
    add(tape.rotor[bit]);
    add(tape.selector_source[bit]);
    add(tape.selector_destination[bit]);
  }
  return total;
}

}  // namespace substrate::bcc32
