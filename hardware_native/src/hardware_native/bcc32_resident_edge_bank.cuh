#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

// A dedicated reversible EDGE RECORD, in ancilla matter, for the coincidence
// association between an instance-basin owner and a cloud destination.
//
// This replaces transpose_resident_pair() (bcc32_resident_transposition.cuh)
// in that role. The swap was refuted: applying the SAME coincidence twice is
// an involution (returns to byte-identical never-learned -- it does not
// learn), and rebinding an owner O from X to Y leaves X holding O's token
// with no reciprocal O-Y relation. An external audit named the reason a swap
// cannot be patched: "You cannot preserve two endpoint identities, preserve
// prior relations, and add a new distinguishable relation without storing
// additional information somewhere. That is an information-capacity
// impossibility." So the relation now lives in NEW matter -- neither
// endpoint rail is ever written by this factor. Only this ancilla is.
//
// Declared semantics (also enforced by the contract, not merely asserted
// here):
//   * duplicate -- the SAME (owner_token, dest_token) pair occurring twice
//     is idempotent: the existing record is found and left untouched (no
//     second XOR, which would toggle the record back to neutral and erase
//     it). Nothing is allocated, nothing is doubled.
//   * rebinding -- owner_token binding to a second dest_token leaves the
//     first (owner_token, dest_token) record completely intact in its own
//     slot; the new pair gets its OWN free slot. The two relations are
//     simply two different edges; forming the second one never touches the
//     first one's rails (and never touches either endpoint's rails, which
//     this factor never writes at all).
//   * capacity -- when every slot is occupied and the (owner_token,
//     dest_token) pair does not match any existing record, formation
//     abstains atomically: no rail in this ancilla is written.
//
// Reversibility: every write here is `rail ^= delta` against a slot that
// starts at (and, absent an occupant, always returns to) zero. XOR is its
// own inverse, so undo_edge_device() with the identical delta exactly
// restores the pre-formation ancilla state -- no per-tick generic journal is
// needed, only "did an edge form, and at which index" (carried by
// AssociationOutcome / resident_association_history_ in
// bcc32_developmental_adult.cuh, the same place owner_slot/dest_slot already
// live).
//
// Value/complement rail-pair convention matches bcc32_grown_instance_basin.cuh
// (value at physical index 2k, complement at 2k+1). XOR preserves the
// invariant word[2k+1] == ~word[2k] automatically: if new_value =
// old_value ^ delta, then ~new_value = ~(old_value ^ delta) = ~old_value ^
// delta = old_complement ^ delta, i.e. XOR-ing the SAME delta into the
// complement rail keeps it the exact complement of the value rail, with no
// separate case for the complement.
namespace substrate::bcc32::resident_edge_bank {

using substrate::bcc32::SiteWord;

// Small declared capacity -- 8 edge records, not a large table.
inline constexpr std::uint32_t kEdgeCount = 8u;

enum GlobalField : std::uint32_t {
  kFactorMarker = 0u,
  kLayoutVersion,
  kGlobalFieldCount,
};

enum EdgeField : std::uint32_t {
  kOwnerToken = 0u,
  kDestToken,
  kOccupied,
  kEdgeFieldCount,
};

enum EdgeAction : std::uint32_t {
  kActionNone = 0u,
  kActionFormed = 1u,
  kActionDuplicate = 2u,
  kActionCapacityAbstained = 3u,
};

inline constexpr std::uint32_t kResidentFieldCount =
    kGlobalFieldCount + kEdgeCount * kEdgeFieldCount;
inline constexpr std::uint32_t kResidentPhysicalRailCount =
    kResidentFieldCount * 2u;
// No generic per-tick journal: every write is a self-inverse XOR and the
// caller (combined_resident_factors_step_kernel /
// combined_resident_factors_inverse_kernel in bcc32_developmental_adult.cuh)
// is told exactly which edge index to undo via AssociationOutcome. The full
// founder domain IS the resident domain.
inline constexpr std::uint32_t kPhysicalRailCount = kResidentPhysicalRailCount;

inline constexpr SiteWord kFactorMarkerValue = 0xe4b3a17du;
inline constexpr SiteWord kLayoutVersionValue = 0xe4b30001u;

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kPhysicalRailCount]{};
};

struct EdgeOutcome {
  std::uint32_t action = kActionNone;
  std::uint32_t edge_index = 0xffffffffu;
};

// A small, standalone cuboid disjoint from every other producer's footprint
// -- proved by a dedicated host-only enumeration probe over every producer's
// OWN physical_offset()-style function across its own full founder-index
// domain (not the approximate extents quoted in prose, which measurably
// undercounted at least the cloud factor's true journal-inclusive range).
// Chosen box: x in [200,203], y in [100,112], z fixed at 150 -- global
// coordinates (centre 250) land at x[450,453] y[350,362] z[400,400], well
// inside the [26,473] clearance window on all three axes, and outside every
// other producer's measured footprint (situation, credit, credit-form,
// sparse_event, selective_state, instance_basin, sensorimotor, cloud,
// readout_f_route, grounded-context, and the calculator's symbolic_arithmetic
// factor, which is not touched by this change but was still checked).
__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {200 + static_cast<std::int32_t>(index % 4u),
          100 + static_cast<std::int32_t>((index / 4u) % 13u), 150};
}

__host__ __device__ inline std::uint32_t resident_index(
    std::uint32_t field) {
  return field * 2u;
}

__host__ __device__ inline std::uint32_t edge_field(std::uint32_t edge,
                                                     std::uint32_t field) {
  return kGlobalFieldCount + edge * kEdgeFieldCount + field;
}

__device__ inline SiteWord read_field(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t field) {
  return words[layout.rails[resident_index(field)]];
}

__device__ inline void write_field(SiteWord* words, const DeviceLayout& layout,
                                   std::uint32_t field, SiteWord value) {
  const std::uint32_t index = resident_index(field);
  words[layout.rails[index]] = value;
  words[layout.rails[index + 1u]] = ~value;
}

// Self-inverse: XOR-ing the identical delta twice into a value/complement
// pair restores both rails exactly, and preserves word[2k+1] == ~word[2k] at
// every intermediate step (see the file comment above for why).
__device__ inline void xor_field(SiteWord* words, const DeviceLayout& layout,
                                 std::uint32_t field, SiteWord delta) {
  const std::uint32_t index = resident_index(field);
  words[layout.rails[index]] ^= delta;
  words[layout.rails[index + 1u]] ^= delta;
}

__device__ inline bool factor_available(const SiteWord* words,
                                        const DeviceLayout& layout) {
  return read_field(words, layout, kFactorMarker) == kFactorMarkerValue &&
         read_field(words, layout, kLayoutVersion) == kLayoutVersionValue;
}

// Forms (or idempotently refreshes, or capacity-abstains from) the edge
// (owner_token, dest_token). Never writes either endpoint's own rail --
// callers pass in VALUES already read from those rails, not slot indices;
// this factor has no way to reach any rail outside its own DeviceLayout.
__device__ inline EdgeOutcome form_edge_device(SiteWord* words,
                                               const DeviceLayout& layout,
                                               SiteWord owner_token,
                                               SiteWord dest_token) {
  EdgeOutcome outcome{};
  if (!factor_available(words, layout)) {
    // Founder matter not attached: abstain silently, exactly as if capacity
    // were exhausted. No rail is touched.
    outcome.action = kActionCapacityAbstained;
    return outcome;
  }
  // Duplicate check FIRST: the same (owner, dest) pair occurring twice must
  // not erase (double-XOR) or double-allocate the edge.
  for (std::uint32_t edge = 0u; edge < kEdgeCount; ++edge) {
    if (read_field(words, layout, edge_field(edge, kOccupied)) != 0u &&
        read_field(words, layout, edge_field(edge, kOwnerToken)) ==
            owner_token &&
        read_field(words, layout, edge_field(edge, kDestToken)) ==
            dest_token) {
      outcome.action = kActionDuplicate;
      outcome.edge_index = edge;
      return outcome;
    }
  }
  // Rebinding: owner_token matching a DIFFERENT dest_token is not a
  // duplicate -- it falls through to allocate its own new slot below,
  // leaving whatever existing (owner_token, other_dest_token) record there
  // is completely untouched (it is a different edge, in a different slot).
  for (std::uint32_t edge = 0u; edge < kEdgeCount; ++edge) {
    if (read_field(words, layout, edge_field(edge, kOccupied)) == 0u) {
      xor_field(words, layout, edge_field(edge, kOwnerToken), owner_token);
      xor_field(words, layout, edge_field(edge, kDestToken), dest_token);
      xor_field(words, layout, edge_field(edge, kOccupied), 1u);
      outcome.action = kActionFormed;
      outcome.edge_index = edge;
      return outcome;
    }
  }
  // All kEdgeCount slots occupied and this pair matches none of them:
  // abstain atomically. No rail above this point was touched by this call
  // (the duplicate scan only reads), so this is a true no-op.
  outcome.action = kActionCapacityAbstained;
  return outcome;
}

// Read-only recall: given an owner_token, find the first occupied edge whose
// owner matches and report its dest_token. Purely additive -- never writes
// any rail, so it cannot affect form_edge_device()/undo_edge_device()'s
// declared semantics or any existing consumer. Used by the form-domain
// recall-bias mechanism (bcc32_cuda_form_domain_recall_bias_contract.cu) to
// look up a destination token in the SAME domain the caller's owner_token
// came from, generically (no candidate-index special-casing here -- this
// only returns a token, the caller decides what to do with it).
struct RecallOutcome {
  std::uint32_t found = 0u;
  SiteWord dest_token = 0u;
  std::uint32_t edge_index = 0xffffffffu;
};

__device__ inline RecallOutcome recall_edge_device(const SiteWord* words,
                                                    const DeviceLayout& layout,
                                                    SiteWord owner_token) {
  RecallOutcome outcome{};
  if (!factor_available(words, layout)) return outcome;
  for (std::uint32_t edge = 0u; edge < kEdgeCount; ++edge) {
    if (read_field(words, layout, edge_field(edge, kOccupied)) != 0u &&
        read_field(words, layout, edge_field(edge, kOwnerToken)) ==
            owner_token) {
      outcome.found = 1u;
      outcome.dest_token = read_field(words, layout, edge_field(edge, kDestToken));
      outcome.edge_index = edge;
      return outcome;
    }
  }
  return outcome;
}

// Exact inverse of a kActionFormed outcome: XOR the identical values back in
// to restore the pre-formation (neutral) state of that one edge slot. Must
// NOT be called for kActionDuplicate/kActionCapacityAbstained/kActionNone --
// those forward calls never wrote anything, so there is nothing to undo.
__device__ inline void undo_edge_device(SiteWord* words,
                                        const DeviceLayout& layout,
                                        std::uint32_t edge_index,
                                        SiteWord owner_token,
                                        SiteWord dest_token) {
  xor_field(words, layout, edge_field(edge_index, kOccupied), 1u);
  xor_field(words, layout, edge_field(edge_index, kDestToken), dest_token);
  xor_field(words, layout, edge_field(edge_index, kOwnerToken), owner_token);
}

}  // namespace substrate::bcc32::resident_edge_bank
