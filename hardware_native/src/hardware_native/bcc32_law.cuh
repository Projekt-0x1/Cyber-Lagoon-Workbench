#pragma once

#include <cstdint>

#include "bcc32_geometry.cuh"
#include "bcc32_law_netlist.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

class ReferenceLattice;

// Q is a complete homogeneous P8 medium word, not absent storage and not life.
constexpr SiteWord kQ = 0x000000ffu;
static_assert(kQ == kCarrierMask);

[[nodiscard]] __host__ __device__ constexpr SiteWord face_channel(Direction direction) {
  return face_bit(direction_index(direction));
}

// Every elementary change is a transposition. The enable predicate must not
// inspect either target, so reversing a circuit means only reversing gate order.
__host__ __device__ constexpr void controlled_transpose(SiteWord& first_word, SiteWord first_bit,
                                                        SiteWord& second_word, SiteWord second_bit,
                                                        bool enabled) {
  if (!enabled || (&first_word == &second_word && first_bit == second_bit)) {
    return;
  }
  const bool first = (first_word & first_bit) != 0u;
  const bool second = (second_word & second_bit) != 0u;
  if (first != second) {
    first_word ^= first_bit;
    second_word ^= second_bit;
  }
}

[[nodiscard]] __host__ __device__ constexpr SiteWord tetrad_mask(std::uint32_t shift) {
  return SiteWord{0x0fu} << shift;
}

__host__ __device__ constexpr void complement_pair_tetrad(SiteWord& word, std::uint32_t shift) {
  const SiteWord mask = tetrad_mask(shift);
  if (occupied_channels(word & mask) == 2u) {
    word ^= mask;
  }
}

// A represented unordered pair of BCC bases rotates the target tetrad. There
// is no rank, minimum, unique winner, coordinate, or tie-to-identity fallback.
__host__ __device__ constexpr void pair_rotor(SiteWord& word, std::uint32_t target_shift,
                                              std::uint32_t control_shift) {
  const std::uint32_t control = static_cast<std::uint32_t>((word >> control_shift) & 0x0fu);
  if (occupied_channels(control) != 2u) {
    return;
  }
  std::uint32_t first = 0u;
  while (((control >> first) & 1u) == 0u) {
    ++first;
  }
  std::uint32_t second = first + 1u;
  while (((control >> second) & 1u) == 0u) {
    ++second;
  }
  controlled_transpose(word, channel_bit(target_shift, first), word,
                       channel_bit(target_shift, second), true);
}

// A locally grown one-bond receptor catalyses a three-factor molecular
// coincidence by relocating its represented bond to the active basis:
//
//   B_s + M_s + E_b + C_b + R_b + collar(-X-_b)
//       <-> B_b + M_s + P+_b + X+_s + X-_b,  s != b
//
// The three-lobed X- collar identifies b while the unique B_s is physical
// receptor matter, not a tag. M_s = E_s + R_s + P+_s - C_s - P-_s is a
// represented scaffold phenotype that differentiates the receptor from
// unrelated one-bond molecules. The product's X+_s remembers which scaffold
// bond moved, so the inverse can restore the exact molecule. Four occupied
// channels become four occupied channels. This is S4-covariant and contains
// no coordinate, preferred basis, or hidden write. The product remains
// ordinary matter subject to every later F gate.
[[nodiscard]] __host__ __device__ constexpr bool differentiated_scaffold_marker(
    SiteWord word, std::uint32_t scaffold) {
  const SiteWord required =
      energy_bit(scaffold) | channel_bit(kReactiveShift, scaffold) | carrier_bit(scaffold);
  const SiteWord forbidden = channel_bit(kConformationShift, scaffold) | carrier_bit(scaffold + 4u);
  return (word & required) == required && (word & forbidden) == 0u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
differentiated_three_factor_candidate_mask(SiteWord word) {
  const std::uint32_t face_state = faces(word);
  const std::uint32_t positive_faces = face_state & 0x0fu;
  const std::uint32_t negative_faces = (face_state >> 4u) & 0x0fu;
  // Both reversible sides have a sparse, exact collar envelope. The collar
  // also identifies the only possible reaction basis, so derive that basis
  // directly instead of scanning all four. This is the same S4-covariant
  // predicate with no preferred basis; it keeps the receptor cheap on bulk
  // matter while preserving its complete truth table.
  const bool possible_forward =
      positive_faces == 0u &&
      (negative_faces == 0x07u || negative_faces == 0x0bu ||
       negative_faces == 0x0du || negative_faces == 0x0eu);
  const bool possible_reverse =
      negative_faces == 0x0fu &&
      (positive_faces == 0x01u || positive_faces == 0x02u ||
       positive_faces == 0x04u || positive_faces == 0x08u);
  if (!possible_forward && !possible_reverse)
    return 0u;

  if (possible_forward) {
    const std::uint32_t basis_mask = 0x0fu ^ negative_faces;
    std::uint32_t basis = 0u;
    while (((basis_mask >> basis) & 1u) == 0u)
      ++basis;
    const SiteWord molecular = energy_bit(basis) | channel_bit(kConformationShift, basis) |
                               channel_bit(kReactiveShift, basis);
    const std::uint32_t non_target_bonds = owned_bonds(word) & ~basis_mask;
    if (occupied_channels(non_target_bonds) != 1u)
      return 0u;
    std::uint32_t scaffold = 0u;
    while (((non_target_bonds >> scaffold) & 1u) == 0u)
      ++scaffold;
    const bool forward =
        differentiated_scaffold_marker(word, scaffold) &&
        (word & owned_bond_bit(basis)) == 0u && (word & carrier_bit(basis)) == 0u &&
        (word & molecular) == molecular &&
        carriers(word) == (0xffu ^ basis_mask ^ (1u << (scaffold + 4u))) &&
        ((word >> kEnergyShift) & 0x0fu) == ((1u << scaffold) | basis_mask);
    return forward ? basis_mask : 0u;
  }

  const std::uint32_t bond_state = owned_bonds(word);
  if (occupied_channels(bond_state) != 1u)
    return 0u;
  std::uint32_t basis = 0u;
  while (((bond_state >> basis) & 1u) == 0u)
    ++basis;
  std::uint32_t scaffold = 0u;
  while (((positive_faces >> scaffold) & 1u) == 0u)
    ++scaffold;
  const SiteWord molecular = energy_bit(basis) | channel_bit(kConformationShift, basis) |
                             channel_bit(kReactiveShift, basis);
  const bool reverse =
      differentiated_scaffold_marker(word, scaffold) &&
      (positive_faces & (1u << basis)) == 0u && (word & carrier_bit(basis)) != 0u &&
      (word & molecular) == 0u &&
      carriers(word) == (0xffu ^ (1u << (scaffold + 4u))) &&
      ((word >> kEnergyShift) & 0x0fu) == (1u << scaffold);
  return reverse ? (1u << basis) : 0u;
}

__host__ __device__ constexpr void differentiated_three_factor_capture(SiteWord& word) {
  const std::uint32_t candidates = differentiated_three_factor_candidate_mask(word);
  if (occupied_channels(candidates) != 1u)
    return;
  std::uint32_t basis = 0u;
  while (((candidates >> basis) & 1u) == 0u)
    ++basis;
  const std::uint32_t face_state = faces(word);
  const bool forward = (word & (energy_bit(basis) | channel_bit(kConformationShift, basis) |
                                channel_bit(kReactiveShift, basis))) != 0u;
  const std::uint32_t scaffold_mask =
      forward ? (owned_bonds(word) & ~(1u << basis)) : (face_state & 0x0fu);
  std::uint32_t scaffold = 0u;
  while (((scaffold_mask >> scaffold) & 1u) == 0u)
    ++scaffold;
  const SiteWord molecular = energy_bit(basis) | channel_bit(kConformationShift, basis) |
                             channel_bit(kReactiveShift, basis) | owned_bond_bit(scaffold);
  const SiteWord captured =
      owned_bond_bit(basis) | carrier_bit(basis) | face_bit(scaffold) | face_bit(basis + 4u);
  const SiteWord after = word ^ molecular ^ captured;
  if (differentiated_three_factor_candidate_mask(after) == candidates) {
    word = after;
  }
}

// Persistent synapse molecule:
//   X-_b + one marker in {E_b, B_b} + the native P+_b/P-_b holes.
//
// Credit is a two-hole molecule on any two of the other three bases. Positive
// holes move E_b -> B_b and leave as negative holes; negative holes perform
// the exact inverse. Two holes are deliberate: with the synapse's own hole,
// the carrier tetrad has popcount one rather than two, so ordinary H does not
// scatter it before this differentiated collision. The reciprocal product
// streams away in the same F tick, leaving the known two-cell fixed-point
// synapse rearmed. All basis and signal-pair choices obey the same local rule.
__host__ __device__ constexpr void differentiated_signed_synapse(SiteWord& word) {
  const SiteWord structure = word & ~kCarrierMask;
  // Nearly every ordinary site rejects on this one native instruction. The
  // exact molecule has only a receptor face plus one marker outside carriers.
  if (occupied_channels(structure) != 2u)
    return;
  const std::uint32_t receptor = (structure >> 12u) & 0x0fu;
  if (occupied_channels(receptor) != 1u)
    return;
  std::uint32_t marker_basis = 0u;
  while ((receptor & (1u << marker_basis)) == 0u)
    ++marker_basis;
  const SiteWord zero = face_bit(marker_basis + 4u) | energy_bit(marker_basis);
  const SiteWord one = face_bit(marker_basis + 4u) | owned_bond_bit(marker_basis);
  if (structure != zero && structure != one)
    return;
  const bool inactive = structure == zero;
  if (
      (word & (carrier_bit(marker_basis) | carrier_bit(marker_basis + 4u))) != 0u)
    return;

  // A three-hole positive probe is distinct from the two-hole credit
  // molecule. Active synapses scatter it onto the reciprocal rails without
  // moving their marker; inactive synapses transmit it. This is the physical
  // read surface used by a convergent regional receiver.
  if (!inactive) {
    std::uint32_t positive_probe = 0u;
    std::uint32_t negative_probe = 0u;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      if (basis == marker_basis)
        continue;
      const bool positive = (word & carrier_bit(basis)) != 0u;
      const bool negative = (word & carrier_bit(basis + 4u)) != 0u;
      if (!positive && negative)
        positive_probe |= 1u << basis;
      if (positive && !negative)
        negative_probe |= 1u << basis;
    }
    const std::uint32_t probe =
        occupied_channels(positive_probe) == 3u ? positive_probe : negative_probe;
    if (occupied_channels(probe) == 3u) {
      for (std::uint32_t basis = 0u; basis < 4u; ++basis)
        if ((probe & (1u << basis)) != 0u)
          word ^= carrier_bit(basis) | carrier_bit(basis + 4u);
      return;
    }
  }

  std::uint32_t signals = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    if (basis == marker_basis)
      continue;
    const bool positive = (word & carrier_bit(basis)) != 0u;
    const bool negative = (word & carrier_bit(basis + 4u)) != 0u;
    const bool selected = inactive ? (!positive && negative) : (positive && !negative);
    if (selected) {
      signals |= 1u << basis;
    } else if (!positive || !negative) {
      return;
    }
  }
  if (occupied_channels(signals) != 2u)
    return;

  word ^= energy_bit(marker_basis) | owned_bond_bit(marker_basis);
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    if ((signals & (1u << basis)) != 0u)
      word ^= carrier_bit(basis) | carrier_bit(basis + 4u);
}

// Stage one of a processive scalar-weight collision. H has complemented the
// tetrad containing the permanent marker hole plus one incoming root hole.
// The exact post-H body is transposed into a one-channel token while restoring
// the fixed carrier mask. The permanent reciprocal holes retain marker
// identity; token role and basis retain action and road direction.
__host__ __device__ constexpr void differentiated_processive_stage(SiteWord& word) {
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    const SiteWord zero_marker = face_bit(marker + 4u) | energy_bit(marker);
    const SiteWord one_marker = face_bit(marker + 4u) | owned_bond_bit(marker);
    const SiteWord rest = kCarrierMask ^ carrier_bit(marker) ^ carrier_bit(marker + 4u);
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      const SiteWord raw_positive = rest ^ carrier_bit(path);
      const SiteWord post_positive = raw_positive ^ SiteWord { 0x0fu };
      const SiteWord raw_negative = rest ^ carrier_bit(path + 4u);
      const SiteWord post_negative = raw_negative ^ SiteWord { 0xf0u };
      const SiteWord bodies[4u]{zero_marker | post_positive, one_marker | post_positive,
                                one_marker | post_negative, zero_marker | post_negative};
      const SiteWord tokens[4u]{owned_bond_bit(path) | rest,
                                channel_bit(kReactiveShift, path) | rest, energy_bit(path) | rest,
                                channel_bit(kConformationShift, path) | rest};
      for (std::uint32_t action = 0u; action < 4u; ++action) {
        if (word == bodies[action]) {
          word = tokens[action];
          return;
        }
        if (word == tokens[action]) {
          word = bodies[action];
          return;
        }
      }
    }
  }
}

// Stage two runs after the energy rotors. It releases the staged token as the
// strengthened/weakened fixed-point body plus a reflected carrier hole, or as
// an unchanged body plus a transmitted saturation/underflow hole.
__host__ __device__ constexpr void differentiated_processive_release(SiteWord& word) {
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    const SiteWord rest = kCarrierMask ^ carrier_bit(marker) ^ carrier_bit(marker + 4u);
    const SiteWord zero_marker = face_bit(marker + 4u) | energy_bit(marker);
    const SiteWord one_marker = face_bit(marker + 4u) | owned_bond_bit(marker);
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      const SiteWord raw_positive = rest ^ carrier_bit(path);
      const SiteWord raw_negative = rest ^ carrier_bit(path + 4u);
      const SiteWord tokens[4u]{owned_bond_bit(path) | rest,
                                channel_bit(kReactiveShift, path) | rest, energy_bit(path) | rest,
                                channel_bit(kConformationShift, path) | rest};
      const SiteWord products[4u]{one_marker | raw_negative, one_marker | raw_positive,
                                  zero_marker | raw_positive, zero_marker | raw_negative};
      for (std::uint32_t action = 0u; action < 4u; ++action) {
        if (word == tokens[action]) {
          word = products[action];
          return;
        }
        if (word == products[action]) {
          word = tokens[action];
          return;
        }
      }
    }
  }
}

// A complete carrier collar and one positive-face marker differentiate a
// terminal maturation molecule. For every ordered marker/path pair m != p,
// exchange C_p+R_p <-> R_p+E_m. The shared R_p bit retains the ordered path
// while the marker basis receives the terminal energy. Exact-word matching
// makes mixed or decorated molecules abstain and keeps the exchange involutive.
#ifndef BCC32_TERMINAL_CANDIDATE_FIRST
#define BCC32_TERMINAL_CANDIDATE_FIRST 4
#endif
#ifndef BCC32_TERMINAL_CANDIDATE_SECOND
#define BCC32_TERMINAL_CANDIDATE_SECOND 7
#endif

[[nodiscard]] __host__ __device__ constexpr SiteWord terminal_candidate_bit(
    std::uint32_t selector, std::uint32_t marker, std::uint32_t path) {
  const std::uint32_t role = selector / 2u;
  const std::uint32_t basis = (selector & 1u) == 0u ? path : marker;
  switch (role) {
    case 0u: return owned_bond_bit(basis);
    case 1u: return channel_bit(kConformationShift, basis);
    case 2u: return channel_bit(kReactiveShift, basis);
    case 3u: return energy_bit(basis);
    case 4u: return face_bit(basis + 4u);
  }
  return 0u;
}

__host__ __device__ constexpr void differentiated_terminal_maturation(
    SiteWord& word) {
  if (carriers(word) != 0xffu)
    return;
  const std::uint32_t face_state = faces(word);
  const std::uint32_t positive_faces = face_state & 0x0fu;
  if ((face_state & 0xf0u) != 0u ||
      occupied_channels(positive_faces) != 1u)
    return;
  std::uint32_t marker = 0u;
  while ((positive_faces & (1u << marker)) == 0u)
    ++marker;
  for (std::uint32_t path = 0u; path < 4u; ++path) {
    if (path == marker)
      continue;
    const SiteWord immature =
        kQ | face_bit(marker) |
        channel_bit(kConformationShift, path) |
        channel_bit(kReactiveShift, path);
    const SiteWord first = terminal_candidate_bit(
        BCC32_TERMINAL_CANDIDATE_FIRST, marker, path);
    const SiteWord second = terminal_candidate_bit(
        BCC32_TERMINAL_CANDIDATE_SECOND, marker, path);
    const SiteWord mature = kQ | face_bit(marker) | first | second;
    if (word == immature || word == mature) {
      word ^= channel_bit(kConformationShift, path) ^
              channel_bit(kReactiveShift, path) ^ first ^ second;
      return;
    }
  }
}

// K_site is one S4-covariant collision. H supplies pair-complement scattering;
// the two E rotors remove basis-parity sectors using represented P collisions.
__host__ __device__ constexpr void apply_site_gate(SiteWord& word, const SiteGate& gate) {
  switch (gate.opcode) {
    case SiteOpcode::complement_pair_tetrad:
      complement_pair_tetrad(word, gate.target_shift);
      return;
    case SiteOpcode::pair_rotor:
      pair_rotor(word, gate.target_shift, gate.control_shift);
      return;
    case SiteOpcode::differentiated_three_factor_capture:
      differentiated_three_factor_capture(word);
      return;
    case SiteOpcode::differentiated_signed_synapse:
      differentiated_signed_synapse(word);
      return;
    case SiteOpcode::differentiated_processive_stage:
      differentiated_processive_stage(word);
      return;
    case SiteOpcode::differentiated_terminal_maturation:
      differentiated_terminal_maturation(word);
      return;
  }
}

[[nodiscard]] __host__ __device__ constexpr bool differentiated_carrier_envelope(
    SiteWord word) {
  const std::uint32_t carrier_state = carriers(word);
  const std::uint32_t positive = carrier_state & 0x0fu;
  const std::uint32_t negative = (carrier_state >> 4u) & 0x0fu;
  const bool negative_three =
      negative == 0x07u || negative == 0x0bu ||
      negative == 0x0du || negative == 0x0eu;
  const bool positive_three_or_four =
      positive == 0x07u || positive == 0x0bu ||
      positive == 0x0du || positive == 0x0eu || positive == 0x0fu;
  return negative_three && positive_three_or_four;
}

template <std::uint32_t Index>
__host__ __device__ constexpr void apply_site_word_forward_impl(SiteWord& word) {
  if constexpr (Index < kSiteGateCount) {
    apply_site_gate(word, site_gate(Index));
    apply_site_word_forward_impl<Index + 1u>(word);
  }
}

template <std::uint32_t Remaining>
__host__ __device__ constexpr void apply_site_word_inverse_impl(SiteWord& word) {
  if constexpr (Remaining > 3u) {
    apply_site_gate(word, site_gate(Remaining - 1u));
    apply_site_word_inverse_impl<Remaining - 1u>(word);
  }
}

__host__ __device__ constexpr void apply_site_word_forward(SiteWord& word) {
  // Netlist gates 0..2 are fused without changing their order. Candidate
  // receptor matter always has three negative carriers and either three or
  // four positive carriers, so the two carrier tetrads cannot fire around a
  // real capture. Ordinary matter pays only this cheap envelope test; the
  // expensive molecular predicate remains restricted to its physical collar.
  if (differentiated_carrier_envelope(word))
    differentiated_three_factor_capture(word);
  complement_pair_tetrad(word, 0u);
  complement_pair_tetrad(word, 4u);
  apply_site_word_forward_impl<3u>(word);
}

__host__ __device__ constexpr void apply_site_word_inverse(SiteWord& word) {
  apply_site_word_inverse_impl<kSiteGateCount>(word);
  complement_pair_tetrad(word, 4u);
  complement_pair_tetrad(word, 0u);
  if (differentiated_carrier_envelope(word))
    differentiated_three_factor_capture(word);
}

[[nodiscard]] __host__ __device__ constexpr bool bit_is_set(SiteWord word, SiteWord bit) {
  return (word & bit) != 0u;
}

[[nodiscard]] __host__ __device__ constexpr bool role_uses_source(EdgeRole role) {
  return role == EdgeRole::positive_carrier || role == EdgeRole::positive_face ||
         role == EdgeRole::owned_bond || role == EdgeRole::energy;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord role_bit(EdgeRole role, std::uint32_t basis) {
  switch (role) {
    case EdgeRole::positive_carrier:
      return carrier_bit(basis);
    case EdgeRole::negative_carrier:
      return carrier_bit(basis + 4u);
    case EdgeRole::positive_face:
      return face_bit(basis);
    case EdgeRole::negative_face:
      return face_bit(basis + 4u);
    case EdgeRole::owned_bond:
      return owned_bond_bit(basis);
    case EdgeRole::conformation:
      return channel_bit(kConformationShift, basis);
    case EdgeRole::reactive:
      return channel_bit(kReactiveShift, basis);
    case EdgeRole::energy:
      return energy_bit(basis);
  }
  return 0u;
}

[[nodiscard]] __host__ __device__ constexpr bool role_is_set(SiteWord source, SiteWord destination,
                                                             EdgeRole role, std::uint32_t basis) {
  return bit_is_set(role_uses_source(role) ? source : destination, role_bit(role, basis));
}

__host__ __device__ constexpr void toggle_role(SiteWord& source, SiteWord& destination,
                                               EdgeRole role, std::uint32_t basis) {
  if (role_uses_source(role)) {
    source ^= role_bit(role, basis);
  } else {
    destination ^= role_bit(role, basis);
  }
}

__host__ __device__ constexpr void transpose_roles(SiteWord& source, SiteWord& destination,
                                                   EdgeRole first, EdgeRole second, bool enabled,
                                                   std::uint32_t basis) {
  if (!enabled || first == second)
    return;
  const bool first_value = role_is_set(source, destination, first, basis);
  const bool second_value = role_is_set(source, destination, second, basis);
  if (first_value != second_value) {
    toggle_role(source, destination, first, basis);
    toggle_role(source, destination, second, basis);
  }
}

// The sole four-target exchange pairs carrier+free-energy with bond+reaction.
// Other edge gates remain active for all other local states, so this is a local
// reversible reaction, not a multi-site motif password that disables chemistry.
__host__ __device__ constexpr void apply_edge_gate(SiteWord& source, SiteWord& destination,
                                                   const EdgeGate& gate, std::uint32_t basis) {
  if (gate.opcode == EdgeOpcode::pair_exchange) {
    const bool first = role_is_set(source, destination, gate.first, basis);
    const bool second = role_is_set(source, destination, gate.second, basis);
    const bool third = role_is_set(source, destination, gate.third, basis);
    const bool fourth = role_is_set(source, destination, gate.fourth, basis);
    if ((first && second && !third && !fourth) || (!first && !second && third && fourth)) {
      toggle_role(source, destination, gate.first, basis);
      toggle_role(source, destination, gate.second, basis);
      toggle_role(source, destination, gate.third, basis);
      toggle_role(source, destination, gate.fourth, basis);
    }
    return;
  }
  bool enabled = false;
  switch (gate.control) {
    case EdgeControl::none:
      enabled = true;
      break;
    case EdgeControl::bit:
      enabled = role_is_set(source, destination, gate.control_first, basis);
      break;
    case EdgeControl::xor_bits:
      enabled = role_is_set(source, destination, gate.control_first, basis) !=
                role_is_set(source, destination, gate.control_second, basis);
      break;
  }
  transpose_roles(source, destination, gate.first, gate.second, enabled, basis);
}

// Each basis selects one bit-disjoint block:
// source {P+, X+, B, E}, destination {P-, X-, C, R}.
// Across all sites and bases these blocks partition every persistent bit once.
__host__ __device__ constexpr void apply_edge_block_forward(SiteWord& source, SiteWord& destination,
                                                            std::uint32_t basis) {
  for (std::uint32_t index = 0u; index < kEdgeGateCount; ++index) {
    apply_edge_gate(source, destination, edge_gate(index), basis);
  }
}

__host__ __device__ constexpr void apply_edge_block_inverse(SiteWord& source, SiteWord& destination,
                                                            std::uint32_t basis) {
  for (std::uint32_t index = kEdgeGateCount; index > 0u; --index) {
    apply_edge_gate(source, destination, edge_gate(index - 1u), basis);
  }
}

// Canonical superstep:
// S_P o K_developmental_credit_service o K_prediction_residual_toggle o
// K_processive_release o K_processive_rearm o K_corner o K_pair_splitter o
// K_edge o K_site o K_eligibility_residual o K_developmental_append o
// K_developmental_learned_receptor. The learned A/B event-journal claim
// therefore exists before site/edge staging and is consumed into its exact
// source-specific negative event by processive release in the same tick. The
// eligibility junction reads boundary-live trace matter before
// ordinary site/edge turnover advances it.
// Production exposes F and F^-1; factor entry points exist for mechanical
// proof code only.
void apply_k_site(ReferenceLattice& lattice);
void apply_k_site_inverse(ReferenceLattice& lattice);
void apply_k_edge(ReferenceLattice& lattice);
void apply_k_edge_inverse(ReferenceLattice& lattice);
void apply_k_carrier_pair_splitter(ReferenceLattice& lattice, bool inverse);
void apply_k_processive_rearm(ReferenceLattice& lattice, bool inverse);
void apply_k_processive_release(ReferenceLattice& lattice, bool inverse);
void apply_k_carrier_corner(ReferenceLattice& lattice, bool inverse);
void apply_k_eligibility_residual_junction(ReferenceLattice& lattice,
                                           bool inverse);
void apply_s_p(ReferenceLattice& lattice);
void apply_s_p_inverse(ReferenceLattice& lattice);
void apply_factor(ReferenceLattice& lattice, LawFactor factor, bool inverse);
void apply_superstep(ReferenceLattice& lattice);
void apply_superstep_inverse(ReferenceLattice& lattice);

}  // namespace substrate::bcc32
