#ifndef HARDWARE_NATIVE_DIRECT_CONTENT_ADDRESS_CUH
#define HARDWARE_NATIVE_DIRECT_CONTENT_ADDRESS_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_genome.cuh"

namespace substrate::direct_network {

struct DirectSha256Address {
  std::uint8_t byte[32]{};
};
static_assert(std::is_standard_layout_v<DirectSha256Address> &&
              std::is_trivially_copyable_v<DirectSha256Address> &&
              std::has_unique_object_representations_v<DirectSha256Address>);

#if defined(__CUDACC__)
#define DIRECT_CONTENT_HD __host__ __device__
#else
#define DIRECT_CONTENT_HD
#endif

DIRECT_CONTENT_HD inline bool operator==(const DirectSha256Address& left,
                                         const DirectSha256Address& right) {
  for (std::uint32_t i = 0u; i < 32u; ++i)
    if (left.byte[i] != right.byte[i]) return false;
  return true;
}

DIRECT_CONTENT_HD inline bool operator!=(const DirectSha256Address& left,
                                         const DirectSha256Address& right) {
  return !(left == right);
}

namespace detail {

DIRECT_CONTENT_HD inline std::uint32_t sha256_rotr(std::uint32_t value,
                                                   std::uint32_t bits) {
  return (value >> bits) | (value << (32u - bits));
}

DIRECT_CONTENT_HD inline std::uint32_t sha256_round_constant(std::uint32_t i) {
  constexpr std::uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  return k[i];
}

struct DirectSha256State {
  std::uint32_t h[8] = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::uint8_t block[64]{};
  std::uint64_t total_bytes = 0u;
  std::uint32_t block_bytes = 0u;

  DIRECT_CONTENT_HD void transform(const std::uint8_t* input) {
    std::uint32_t w[64];
    for (std::uint32_t i = 0u; i < 16u; ++i) {
      const std::uint32_t j = i * 4u;
      w[i] = (static_cast<std::uint32_t>(input[j]) << 24u) |
             (static_cast<std::uint32_t>(input[j + 1u]) << 16u) |
             (static_cast<std::uint32_t>(input[j + 2u]) << 8u) |
             static_cast<std::uint32_t>(input[j + 3u]);
    }
    for (std::uint32_t i = 16u; i < 64u; ++i) {
      const std::uint32_t x = w[i - 15u];
      const std::uint32_t y = w[i - 2u];
      const std::uint32_t s0 = sha256_rotr(x, 7u) ^ sha256_rotr(x, 18u) ^ (x >> 3u);
      const std::uint32_t s1 = sha256_rotr(y, 17u) ^ sha256_rotr(y, 19u) ^ (y >> 10u);
      w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }

    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (std::uint32_t i = 0u; i < 64u; ++i) {
      const std::uint32_t s1 = sha256_rotr(e, 6u) ^ sha256_rotr(e, 11u) ^
                               sha256_rotr(e, 25u);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + s1 + choose + sha256_round_constant(i) + w[i];
      const std::uint32_t s0 = sha256_rotr(a, 2u) ^ sha256_rotr(a, 13u) ^
                               sha256_rotr(a, 22u);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + majority;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  DIRECT_CONTENT_HD void update(const void* source, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(source);
    total_bytes += static_cast<std::uint64_t>(size);
    while (size != 0u) {
      const std::size_t room = 64u - block_bytes;
      const std::size_t take = size < room ? size : room;
      for (std::size_t i = 0u; i < take; ++i) block[block_bytes + i] = bytes[i];
      block_bytes += static_cast<std::uint32_t>(take);
      bytes += take;
      size -= take;
      if (block_bytes == 64u) {
        transform(block);
        block_bytes = 0u;
      }
    }
  }

  DIRECT_CONTENT_HD DirectSha256Address finish() {
    const std::uint64_t bit_count = total_bytes * 8u;
    block[block_bytes++] = 0x80u;
    if (block_bytes > 56u) {
      while (block_bytes < 64u) block[block_bytes++] = 0u;
      transform(block);
      block_bytes = 0u;
    }
    while (block_bytes < 56u) block[block_bytes++] = 0u;
    for (std::uint32_t i = 0u; i < 8u; ++i)
      block[63u - i] = static_cast<std::uint8_t>(bit_count >> (i * 8u));
    transform(block);

    DirectSha256Address address{};
    for (std::uint32_t i = 0u; i < 8u; ++i) {
      address.byte[i * 4u] = static_cast<std::uint8_t>(h[i] >> 24u);
      address.byte[i * 4u + 1u] = static_cast<std::uint8_t>(h[i] >> 16u);
      address.byte[i * 4u + 2u] = static_cast<std::uint8_t>(h[i] >> 8u);
      address.byte[i * 4u + 3u] = static_cast<std::uint8_t>(h[i]);
    }
    return address;
  }
};

}  // namespace detail

DIRECT_CONTENT_HD inline bool direct_sha256_content_address(
    const void* bytes, std::size_t size, DirectSha256Address* out) {
  if (out == nullptr || (size != 0u && bytes == nullptr) ||
      size > (static_cast<std::uint64_t>(~std::uint64_t{0}) >> 3u))
    return false;
  detail::DirectSha256State state{};
  if (size != 0u) state.update(bytes, size);
  *out = state.finish();
  return true;
}

DIRECT_CONTENT_HD inline bool direct_sha256_genome_address(
    const DirectGenomeV1& genome, DirectSha256Address* out) {
  if (out == nullptr ||
      (genome.header.abi_version != kDirectGenomeAbiV1 &&
       genome.header.abi_version != kDirectGenomeAbiV2) ||
      genome.header.territory_count > kDirectMaxTerritoriesV1 ||
      genome.header.field_count > kDirectMaxFieldsV1 ||
      genome.header.rule_count > kDirectMaxRulesV1)
    return false;

  detail::DirectSha256State state{};
  state.update(&genome.header, sizeof(genome.header));
  state.update(genome.territories,
               sizeof(genome.territories[0]) * genome.header.territory_count);
  state.update(genome.fields,
               sizeof(genome.fields[0]) * genome.header.field_count);
  if (genome.header.abi_version == kDirectGenomeAbiV1) {
    for (std::uint32_t i = 0u; i < genome.header.rule_count; ++i)
      state.update(&genome.rules[i], offsetof(DirectRuleSpecV1, tract_delay));
  } else {
    state.update(genome.rules,
                 sizeof(genome.rules[0]) * genome.header.rule_count);
  }
  *out = state.finish();
  return true;
}

#undef DIRECT_CONTENT_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_CONTENT_ADDRESS_CUH
