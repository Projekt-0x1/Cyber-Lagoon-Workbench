#pragma once

#include <cstdint>

#include "bcc32_types.cuh"

namespace substrate::bcc32 {

// Single source of truth for the aperture edge length, in chunks. This is
// consumed by bcc32_developmental_adult.cuh AND by every
// grown_*_factor.cuh fixed_physical_slot() implementation to compute global
// slot addresses. The factor headers cannot include
// bcc32_developmental_adult.cuh to reach it there (the adult header includes
// them -- that would be a cycle), so the value lives here instead: a leaf
// header that depends only on bcc32_types.cuh (for kChunkEdge), never on the
// adult header. developmental_adult::kApertureEdgeChunks is an alias onto
// this constant, not a second definition.
// Overridable ONLY to measure how the aperture bound scales; the default is
// unchanged and every shipped target uses it. The organism's own coordinates
// are computed relative to kApertureCentre below, so raising this moves the
// FACES outward while leaving the body where it was -- which is what makes an
// edge sweep discriminating rather than merely relocating the same structures.
#ifndef BCC32_APERTURE_EDGE_CHUNKS
#define BCC32_APERTURE_EDGE_CHUNKS 5
#endif
inline constexpr std::int64_t kApertureEdgeChunks = BCC32_APERTURE_EDGE_CHUNKS;

// Derived aperture geometry shared by every fixed_physical_slot()
// implementation that folds a factor's physical offsets into one flat global
// slot index, in aperture-chunk-major order. Hoisted here because all sites
// that computed these from kApertureEdgeChunks and kChunkEdge did so
// identically; verify that still holds before adding more call sites.
inline constexpr std::uint64_t kApertureChunkEdge =
    static_cast<std::uint64_t>(kChunkEdge);
inline constexpr std::uint64_t kApertureChunkSites =
    kApertureChunkEdge * kApertureChunkEdge * kApertureChunkEdge;
inline constexpr std::int32_t kApertureCentre = static_cast<std::int32_t>(
    (static_cast<std::uint64_t>(kApertureEdgeChunks) * kApertureChunkEdge) /
    2u);

}  // namespace substrate::bcc32
