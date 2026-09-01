#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32 {

struct Int3 {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    friend __host__ __device__ constexpr bool operator==(const Int3&,
                                                          const Int3&) = default;
};

[[nodiscard]] __host__ __device__ constexpr Int3 operator+(Int3 left,
                                                             Int3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

[[nodiscard]] __host__ __device__ constexpr Int3 operator-(Int3 value) {
    return {-value.x, -value.y, -value.z};
}

enum class Basis : std::uint32_t {
    u0 = 0u,
    u1 = 1u,
    u2 = 2u,
    u3 = 3u,
};

enum class Direction : std::uint32_t {
    positive_u0 = 0u,
    positive_u1 = 1u,
    positive_u2 = 2u,
    positive_u3 = 3u,
    negative_u0 = 4u,
    negative_u1 = 5u,
    negative_u2 = 6u,
    negative_u3 = 7u,
};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t direction_index(
    Direction direction) {
    return static_cast<std::uint32_t>(direction);
}

[[nodiscard]] __host__ __device__ constexpr Basis direction_basis(
    Direction direction) {
    return static_cast<Basis>(direction_index(direction) & 3u);
}

[[nodiscard]] __host__ __device__ constexpr bool direction_is_positive(
    Direction direction) {
    return direction_index(direction) < 4u;
}

[[nodiscard]] __host__ __device__ constexpr Direction opposite_direction(
    Direction direction) {
    return static_cast<Direction>(direction_index(direction) ^ 4u);
}

// These oblique integer generators make the logical Z^3 graph connected.
// Their physical embedding below is the regular tetrahedral BCC star.
[[nodiscard]] __host__ __device__ constexpr Int3 basis_offset(Basis basis) {
    switch (basis) {
        case Basis::u0: return {1, 0, 0};
        case Basis::u1: return {0, 1, 0};
        case Basis::u2: return {0, 0, 1};
        case Basis::u3: return {-1, -1, -1};
    }
    return {};
}

[[nodiscard]] __host__ __device__ constexpr Int3 direction_offset(
    Direction direction) {
    const Int3 value = basis_offset(direction_basis(direction));
    return direction_is_positive(direction) ? value : -value;
}

// Twice-scaled physical vectors preserve integer arithmetic while exposing the
// tetrahedral BCC embedding. Logical addressing always uses direction_offset.
[[nodiscard]] __host__ __device__ constexpr Int3 physical_tetrahedral_vector(
    Direction direction) {
    constexpr Int3 positive[4] = {
        {1, 1, -1},
        {-1, 1, 1},
        {1, -1, 1},
        {-1, -1, -1},
    };
    const Int3 value = positive[static_cast<std::uint32_t>(
        direction_basis(direction))];
    return direction_is_positive(direction) ? value : -value;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t bipartite_color(
    std::int64_t x,
    std::int64_t y,
    std::int64_t z) {
    const std::int64_t parity = (x + y + z) % 2;
    return static_cast<std::uint32_t>(parity < 0 ? parity + 2 : parity);
}

static_assert(basis_offset(Basis::u0) + basis_offset(Basis::u1) +
                  basis_offset(Basis::u2) + basis_offset(Basis::u3) ==
              Int3{});
static_assert(physical_tetrahedral_vector(Direction::positive_u0) +
                  physical_tetrahedral_vector(Direction::positive_u1) +
                  physical_tetrahedral_vector(Direction::positive_u2) +
                  physical_tetrahedral_vector(Direction::positive_u3) ==
              Int3{});
static_assert(direction_offset(Direction::positive_u0) ==
              -direction_offset(Direction::negative_u0));
static_assert(direction_offset(Direction::positive_u1) ==
              -direction_offset(Direction::negative_u1));
static_assert(direction_offset(Direction::positive_u2) ==
              -direction_offset(Direction::negative_u2));
static_assert(direction_offset(Direction::positive_u3) ==
              -direction_offset(Direction::negative_u3));
static_assert(bipartite_color(0, 0, 0) != bipartite_color(1, 0, 0));
static_assert(bipartite_color(0, 0, 0) != bipartite_color(-1, -1, -1));

}  // namespace substrate::bcc32
