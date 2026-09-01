#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_grown_sparse_event_memory.cuh"
#include "hardware_native/bcc32_resident_factor_basins.cuh"

// A generic ordered renderer over the JOINT resident factor state
// (bcc32_resident_factor_basins.cuh's four independent basins: actor,
// action, modifier, object). Generation in
// bcc32_grown_sparse_event_memory.cuh works by CHOOSING one stored
// trajectory ASSEMBLY (select_trajectory()) and walking ITS phases
// (arm_selected_trajectory()) -- so it can only ever emit a surface that
// some prior training scene produced whole. This renderer instead composes
// a byte sequence directly from the four LIVE factor basins, one byte per
// slot, and never reads any stored trajectory to do it. It can therefore
// emit a combination for which no stored trajectory exists -- that is the
// whole point of building it, and the comparison in
// bcc32_cuda_factor_renderer_contract.cu against the OTHER (untouched)
// mechanism is the result this header exists to support.
//
// This file does not modify select_trajectory, arm_selected_trajectory, or
// trajectory_adjacency_index. It sits ALONGSIDE bcc32_grown_sparse_event_
// memory.cuh, reusing its RawByteDecode/decode_form_raw/trajectory_phase_
// index only for the FALSIFIER switch below (a deliberately wrong build that
// must fail the contract's checks 2 and/or 4).
//
// ============================================================================
// order_source=authored -- READ BEFORE CITING THIS FILE AS "DISCOVERING"
// WORD ORDER.
// ============================================================================
// kRenderOrder below is a fixed, hand-written sequence: actor, action,
// modifier, object. It is NOT computed at runtime from stored trajectory
// data. It was not possible to derive it from data already resident in the
// tree within this task's time-box, because nothing in the tree stores any
// TRAINED trajectory before this contract creates one (bcc32_resident_
// factor_basins_contract.cu never stores a trajectory at all -- it only
// exercises the four basins directly), so any "phase order shared by
// existing stored trajectories" would first have to be authored by
// whichever kernel populates the training corpus, then merely reflected
// back by the renderer -- which would misrepresent an authored order as a
// discovered one. Rather than launder that, this file authors the order
// ONCE, here, in the open: the same actor-action-modifier-object order the
// English training sentences use, and the SAME positional convention
// (phase 0..3) that bcc32_grown_sparse_event_memory.cuh's own
// trajectory_phase_index/arm_selected_trajectory already use for ITS stored
// sentences. The claim this renderer supports is scoped to CONTENT
// COMPOSITION (which byte occupies which of the four already-numbered
// slots) -- never to WORD ORDER, which remains scaffolding authored by a
// human, exactly as the brief requires it be disclosed.

namespace substrate::bcc32::resident_factor_renderer {

namespace factor = resident_factor_basins;
namespace sparse = grown_sparse_event_memory;
using substrate::bcc32::RawByteDecode;
using substrate::bcc32::SiteWord;

inline constexpr const char* kOrderSource = "authored";

inline constexpr std::uint32_t kSlotCount = factor::kFactorSlotCount;

struct RenderedSurface {
  std::uint8_t bytes[kSlotCount]{};
  std::uint32_t valid[kSlotCount]{};
};

// FALSIFIER SWITCH (required by the brief): flip to true, rebuild, rerun
// bcc32_cuda_factor_renderer_contract -- this makes the MODIFIER position of
// the emitted sequence read a fixed stored trajectory's phase instead of the
// live modifier factor basin. With this true: check 2 (the untrained Bo+blu
// probe) must go RED, because the renderer will emit whatever modifier byte
// kFalsifierTrajectory happened to store at training time (the first
// trained trajectory, Al+red -> "red") instead of the live "blu" factor;
// check 4's modifier-only control must also go RED, because changing only
// the live modifier factor no longer changes the emitted modifier position.
// Revert to false to restore the PASS baseline.
inline constexpr bool kFalsifierReadModifierFromTrajectory = false;
inline constexpr std::uint32_t kFalsifierTrajectory = 0u;
inline constexpr std::uint32_t kFalsifierTrajectoryModifierPhase = 2u;

// Reads the four live factor basins (bcc32_resident_factor_basins.cuh) in
// the fixed order above and composes them into one byte sequence. Never
// walks trajectory_phase_index, never calls select_trajectory or
// arm_selected_trajectory, never consults kTrajectoryAssemblyCount -- the
// ONLY exception is the falsifier switch above, whose entire purpose is to
// prove checks 2/4 actually catch that class of defect when it is
// deliberately introduced.
__device__ inline RenderedSurface render_from_factors(
    const SiteWord* words, const factor::DeviceLayout& factor_layout,
    const sparse::DeviceLayout& sparse_layout) {
  (void)sparse_layout;
  RenderedSurface out{};
  // order_source=authored (see file header). This local constexpr array is
  // the one place the fixed actor->action->modifier->object sequence is
  // written down (kept function-local, not namespace-scope, so it compiles
  // as ordinary device-local data rather than requiring __device__ global
  // storage).
  constexpr std::uint32_t kRenderOrder[kSlotCount] = {
      static_cast<std::uint32_t>(factor::kActor),
      static_cast<std::uint32_t>(factor::kAction),
      static_cast<std::uint32_t>(factor::kModifier),
      static_cast<std::uint32_t>(factor::kObject)};
  for (std::uint32_t position = 0u; position < kSlotCount; ++position) {
    const std::uint32_t slot = kRenderOrder[position];
    if (kFalsifierReadModifierFromTrajectory &&
        slot == static_cast<std::uint32_t>(factor::kModifier)) {
      const std::uint32_t form = sparse::read_unsigned(
          words, sparse::trajectory_phase_index(
                     kFalsifierTrajectory, kFalsifierTrajectoryModifierPhase,
                     0u));
      const RawByteDecode raw = sparse::decode_form_raw(words, form);
      out.bytes[position] = raw.value;
      out.valid[position] = raw.valid ? 1u : 0u;
      continue;
    }
    const factor::FactorReadout readout =
        factor::read_factor_device(words, factor_layout, slot);
    out.bytes[position] = readout.value;
    out.valid[position] = readout.valid ? 1u : 0u;
  }
  return out;
}

}  // namespace substrate::bcc32::resident_factor_renderer
