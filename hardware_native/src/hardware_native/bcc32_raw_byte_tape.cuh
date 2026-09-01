#pragma once

#include <cstdint>

#include "bcc32_reciprocal_tape.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

// A byte is a body-level eight-bit observation, not a character, token, or
// symbol. Dual rail keeps exactly one represented quantum per bit position.
struct RawByteRails {
  SiteWord zero = kQuiescentWord;
  SiteWord one = kQuiescentWord;

  friend constexpr bool operator==(const RawByteRails&, const RawByteRails&) = default;
};

struct RawByteDecode {
  std::uint8_t value = 0u;
  bool valid = false;

  friend constexpr bool operator==(const RawByteDecode&, const RawByteDecode&) = default;
};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t raw_byte_zero_rail(std::uint8_t value) {
  return static_cast<std::uint32_t>(static_cast<std::uint8_t>(~value));
}

[[nodiscard]] __host__ __device__ constexpr RawByteRails with_raw_byte_carriers(
    RawByteRails rails, std::uint8_t value) {
  rails.zero = with_carriers(rails.zero, raw_byte_zero_rail(value));
  rails.one = with_carriers(rails.one, value);
  return rails;
}

[[nodiscard]] __host__ __device__ constexpr RawByteRails with_raw_byte_faces(RawByteRails rails,
                                                                             std::uint8_t value) {
  rails.zero = with_faces(rails.zero, raw_byte_zero_rail(value));
  rails.one = with_faces(rails.one, value);
  return rails;
}

[[nodiscard]] __host__ __device__ constexpr RawByteDecode decode_raw_byte_carriers(
    const RawByteRails& rails) {
  const std::uint32_t zero = carriers(rails.zero);
  const std::uint32_t one = carriers(rails.one);
  return {static_cast<std::uint8_t>(one),
          static_cast<std::uint8_t>(zero ^ one) == 0xffu && (zero & one) == 0u};
}

[[nodiscard]] __host__ __device__ constexpr RawByteDecode decode_raw_byte_faces(
    const RawByteRails& rails) {
  const std::uint32_t zero = faces(rails.zero);
  const std::uint32_t one = faces(rails.one);
  return {static_cast<std::uint8_t>(one),
          static_cast<std::uint8_t>(zero ^ one) == 0xffu && (zero & one) == 0u};
}

// The external tape is part of the reversible state. After this exchange the
// world carries the incoming byte and the tape carries the displaced byte.
// Reapplying the same operation is the exact inverse.
__host__ __device__ constexpr void reciprocal_raw_byte_exchange(RawByteRails& world,
                                                                RawByteRails& tape) {
  reciprocal_field_exchange<kCarrierShift, kFaceShift, 8u>(world.zero, tape.zero);
  reciprocal_field_exchange<kCarrierShift, kFaceShift, 8u>(world.one, tape.one);
}

static_assert(decode_raw_byte_carriers(with_raw_byte_carriers(RawByteRails{},
                                                              static_cast<std::uint8_t>(0xa5u))) ==
              RawByteDecode{0xa5u, true});
static_assert(decode_raw_byte_faces(with_raw_byte_faces(RawByteRails{},
                                                        static_cast<std::uint8_t>(0x3cu))) ==
              RawByteDecode{0x3cu, true});

}  // namespace substrate::bcc32
