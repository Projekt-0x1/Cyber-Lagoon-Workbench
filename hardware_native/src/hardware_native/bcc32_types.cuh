#pragma once

#include "bcc32_hd_annotation.h"

#include <cstdint>

namespace substrate::bcc32 {

using SiteWord = std::uint32_t;

constexpr std::uint64_t kProductionSites = 2'500'000'000ull;
constexpr std::uint64_t kBitsPerSite = 32ull;
constexpr std::uint64_t kProductionBits = 80'000'000'000ull;
constexpr std::uint64_t kProductionBytes = 10'000'000'000ull;

constexpr std::uint32_t kChunkEdge = 100u;
constexpr std::uint64_t kChunkSites = 1'000'000ull;
constexpr std::uint64_t kChunkBytes = 4'000'000ull;
constexpr std::uint32_t kProductionChunkSlots = 2'500u;

constexpr std::uint32_t kCarrierShift = 0u;
constexpr std::uint32_t kFaceShift = 8u;
constexpr std::uint32_t kOwnedBondShift = 16u;
constexpr std::uint32_t kConformationShift = 20u;
constexpr std::uint32_t kReactiveShift = 24u;
constexpr std::uint32_t kEnergyShift = 28u;

constexpr SiteWord kCarrierMask = 0x000000ffu;
constexpr SiteWord kFaceMask = 0x0000ff00u;
constexpr SiteWord kOwnedBondMask = 0x000f0000u;
constexpr SiteWord kConformationMask = 0x00f00000u;
constexpr SiteWord kReactiveMask = 0x0f000000u;
constexpr SiteWord kEnergyMask = 0xf0000000u;

// The uniform carrier bath is a complete ordinary word. Every structural
// channel is quiescent; all eight ballistic channels are occupied.
constexpr SiteWord kQuiescentWord = kCarrierMask;

[[nodiscard]] __host__ __device__ constexpr std::uint32_t carriers(
    SiteWord word) {
    return word & kCarrierMask;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t faces(SiteWord word) {
    return (word & kFaceMask) >> kFaceShift;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t owned_bonds(
    SiteWord word) {
    return (word & kOwnedBondMask) >> kOwnedBondShift;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t conformation(
    SiteWord word) {
    return (word & kConformationMask) >> kConformationShift;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t reactive(
    SiteWord word) {
    return (word & kReactiveMask) >> kReactiveShift;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_field(
    SiteWord word,
    SiteWord mask,
    std::uint32_t shift,
    std::uint32_t value) {
    return static_cast<SiteWord>((word & ~mask) |
                                 ((static_cast<SiteWord>(value) << shift) & mask));
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_carriers(
    SiteWord word,
    std::uint32_t value) {
    return with_field(word, kCarrierMask, kCarrierShift, value);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_faces(
    SiteWord word,
    std::uint32_t value) {
    return with_field(word, kFaceMask, kFaceShift, value);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_owned_bonds(
    SiteWord word,
    std::uint32_t value) {
    return with_field(word, kOwnedBondMask, kOwnedBondShift, value);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_conformation(
    SiteWord word,
    std::uint32_t value) {
    return with_field(word, kConformationMask, kConformationShift, value);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord with_reactive(
    SiteWord word,
    std::uint32_t value) {
    return with_field(word, kReactiveMask, kReactiveShift, value);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord channel_bit(
    std::uint32_t shift,
    std::uint32_t index) {
    return SiteWord{1u} << (shift + index);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord carrier_bit(
    std::uint32_t direction) {
    return channel_bit(kCarrierShift, direction);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord face_bit(
    std::uint32_t direction) {
    return channel_bit(kFaceShift, direction);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord owned_bond_bit(
    std::uint32_t basis) {
    return channel_bit(kOwnedBondShift, basis);
}

// E4 consists of four independently represented Boolean quanta.  The law
// exchanges these bits; it never forms or consumes a scalar energy value.
[[nodiscard]] __host__ __device__ constexpr SiteWord energy_bit(
    std::uint32_t quantum) {
    return channel_bit(kEnergyShift, quantum);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t occupied_channels(
    SiteWord word) {
    std::uint32_t count = 0u;
    while (word != 0u) {
        count += word & 1u;
        word >>= 1u;
    }
    return count;
}

static_assert(sizeof(SiteWord) * 8u == kBitsPerSite);
static_assert(kProductionSites * kBitsPerSite == kProductionBits);
static_assert(kProductionSites * sizeof(SiteWord) == kProductionBytes);
static_assert(kChunkSites * sizeof(SiteWord) == kChunkBytes);
static_assert(static_cast<std::uint64_t>(kProductionChunkSlots) * kChunkSites ==
              kProductionSites);
static_assert((kCarrierMask | kFaceMask | kOwnedBondMask | kConformationMask |
               kReactiveMask | kEnergyMask) == 0xffffffffu);
static_assert((kCarrierMask & kFaceMask) == 0u);
static_assert(occupied_channels(kQuiescentWord) == 8u);

}  // namespace substrate::bcc32
