#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

// Four INDEPENDENT resident factor basins -- actor, action, modifier,
// object -- each holding a single FORM identity (a raw byte, in the same
// domain the language path decodes via decode_form_raw() ->
// RawByteDecode.value; never a signature hash).
//
// WHY this exists: bcc32_grown_sparse_event_memory.cuh's select_trajectory()
// iterates kTrajectoryAssemblyCount whole stored trajectory ASSEMBLIES,
// picks the best-matching complete trajectory, and arm_selected_trajectory()
// arms it. Generation walks one chosen sentence's phases; no independently
// selected actor/action/modifier/object factor exists anywhere in the
// substrate, so a held-out probe emits a REPLAYED training surface (wrong
// participant at word one) because the whole trajectory was chosen before
// any word. This file does not touch select_trajectory,
// arm_selected_trajectory, or trajectory_adjacency_index -- it builds the
// factor substrate ALONGSIDE them, not a replacement for generation.
//
// The structural rule this file satisfies: no resident record that selects
// a complete trajectory may own more than one independently variable
// factor. Each of the four basins below is SEPARATELY ADDRESSABLE matter --
// reading or writing one factor touches only that factor's own two rails
// (value + complement), never another factor's rails. That is the whole
// point: if actor and modifier shared a record, the crossed-substitution
// contract (bcc32_cuda_resident_factor_basins_contract.cu) would fail its
// own cross-contamination check.
//
// Value/complement rail-pair convention matches
// bcc32_grown_instance_basin.cuh and bcc32_resident_edge_bank.cuh (value at
// physical index 2k, complement at 2k+1): write_field sets word[2k]=value,
// word[2k+1]=~value directly (a plain set, like grown_instance_basin's
// write_field -- not the edge bank's XOR-delta form/undo, because there is
// no capacity/duplicate bookkeeping here, just one decoded byte per slot).
// Reversibility is exact LIFO: every write_field call here is undone by a
// second write_field call restoring the exact previous value (0 at fresh
// founder-matter attachment), never a generic per-tick journal -- this
// factor never runs a tick loop, so there is nothing to journal.
namespace substrate::bcc32::resident_factor_basins {

using substrate::bcc32::SiteWord;

enum GlobalField : std::uint32_t {
  kFactorMarker = 0u,
  kLayoutVersion,
  kGlobalFieldCount,
};

// The four independently variable factors. Each is exactly one field --
// one form-identity byte, stored full-width in a SiteWord rail pair.
enum FactorSlot : std::uint32_t {
  kActor = 0u,
  kAction,
  kModifier,
  kObject,
  kFactorSlotCount,
};

inline constexpr std::uint32_t kResidentFieldCount =
    static_cast<std::uint32_t>(kGlobalFieldCount) +
    static_cast<std::uint32_t>(kFactorSlotCount);
inline constexpr std::uint32_t kResidentPhysicalRailCount =
    kResidentFieldCount * 2u;
// No per-tick journal: this factor is never advanced by a develop() tick.
// Its only writes are direct populate_factor_device() sets, each undone by
// a direct write_factor_device() call restoring the known prior value.
inline constexpr std::uint32_t kPhysicalRailCount = kResidentPhysicalRailCount;

inline constexpr SiteWord kFactorMarkerValue = 0xfac7b451u;
inline constexpr SiteWord kLayoutVersionValue = 0xfac70001u;

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kPhysicalRailCount]{};
};

// A small, standalone cuboid disjoint from every other producer's founder
// footprint -- proved by a dedicated host-only enumeration probe (see
// hardware_native/tests/bcc32_cuda_resident_factor_basins_contract.cu's
// companion placement probe, run separately at
// /tmp/.../placement_probe.cu during development) over every producer's own
// physical_offset()-style function across its own full founder-index domain
// (cloud, form_credit resident + form sub-domain, instance_basin,
// selective_state_space, sensorimotor, sparse_event_memory (journal
// included -- journal_index() feeds back into physical_offset()'s own
// domain there), the calculator's symbolic_arithmetic factor (untouched,
// still checked), edge_bank, readout_f_route, and grounded_context's four
// fixed points). Chosen box: x in [204,207], y in [100,102], z fixed at
// 150 -- immediately adjacent to (never overlapping) the edge bank's
// x in [200,203] y in [100,112] z=150, so it shares the edge bank's z-plane
// but starts one x-step past its right edge. Global coordinates (centre
// 250) land at x[454,457] y[350,352] z[400,400], comfortably inside the
// [26,473] clearance window on all three axes, and pairwise disjoint from
// every measured producer footprint above (bounding-box overlap check,
// zero collisions).
__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {204 + static_cast<std::int32_t>(index % 4u),
          100 + static_cast<std::int32_t>((index / 4u) % 3u), 150};
}

__host__ __device__ inline std::uint32_t resident_index(
    std::uint32_t field) {
  return field * 2u;
}

// FALSIFIER SWITCH (required by the brief): flip to true, rebuild, rerun
// bcc32_cuda_resident_factor_basins_contract -- this aliases kModifier's
// rails onto kActor's own field, at the single source both read_field and
// write_field go through, exactly the class of defect the contract's
// cross-contamination check exists to catch. With this true, the
// actor-only and modifier-only substitution checks must go RED with a
// nonzero cross-contamination count (transplanting "actor" also
// transplants what modifier reads, because they are literally the same
// two rails). Revert to false to restore the PASS baseline.
inline constexpr bool kFalsifierAliasModifierOntoActor = false;

__host__ __device__ inline std::uint32_t factor_field(std::uint32_t slot) {
  if (kFalsifierAliasModifierOntoActor && slot == kModifier) slot = kActor;
  return kGlobalFieldCount + slot;
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

__device__ inline bool factor_available(const SiteWord* words,
                                        const DeviceLayout& layout) {
  return read_field(words, layout, kFactorMarker) == kFactorMarkerValue &&
         read_field(words, layout, kLayoutVersion) == kLayoutVersionValue;
}

// Reads one factor slot's own decoded form byte (and validity -- valid
// whenever founder matter is attached; this factor has no "unset" sentinel
// beyond the fresh-founder value of 0, so callers that need to distinguish
// "never populated" from "populated with byte 0" must track that
// separately, exactly as RawByteDecode.valid does for the raw byte tape).
struct FactorReadout {
  std::uint8_t value = 0u;
  bool valid = false;
};

__device__ inline FactorReadout read_factor_device(const SiteWord* words,
                                                    const DeviceLayout& layout,
                                                    std::uint32_t slot) {
  FactorReadout out{};
  if (!factor_available(words, layout) || slot >= kFactorSlotCount) return out;
  out.value = static_cast<std::uint8_t>(
      read_field(words, layout, factor_field(slot)) & 0xffu);
  out.valid = true;
  return out;
}

// Sets exactly one factor slot's own two rails to `value` (a decoded raw
// byte, e.g. sparse_event_memory::decode_form_raw(...).value). Touches no
// other slot's rails and no global field. Returns the PREVIOUS value so the
// caller can undo with an identical write_factor_device() call restoring it
// -- exact LIFO reversibility, no journal needed because this factor never
// runs a tick loop.
__device__ inline SiteWord write_factor_device(SiteWord* words,
                                               const DeviceLayout& layout,
                                               std::uint32_t slot,
                                               SiteWord value) {
  if (!factor_available(words, layout) || slot >= kFactorSlotCount) return 0u;
  const std::uint32_t field = factor_field(slot);
  const SiteWord previous = read_field(words, layout, field);
  write_field(words, layout, field, value);
  return previous;
}

}  // namespace substrate::bcc32::resident_factor_basins
