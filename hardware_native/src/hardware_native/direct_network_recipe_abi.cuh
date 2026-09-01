// Patch 0002: __host__ __device__ operations over the network_recipe ABI
// (direct_network_recipe.hpp) for use inside CUDA construction/certification
// kernels (patches 0004+) without a host round-trip. The struct definitions
// themselves stay in the plain-C++ header so CPU-only tooling never needs
// nvcc; this file only adds device-callable behavior on top of them.

#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_ABI_CUH

#include <cuda_runtime.h>

#include "hardware_native/direct_network_recipe.hpp"

namespace substrate::direct_network::recipe {

// Deterministic, non-cryptographic 256-bit content hash (FNV-1a spread
// across 8 lanes). This is a build/dedup/replay-identity digest, not a
// security primitive: its job is that two byte-identical Genomes always
// produce the same Root256 and two different Genomes almost never collide,
// on both host and device, in lockstep -- it is not defending against an
// adversarial genome author.
__host__ __device__ inline Root256 content_root(const void* bytes, std::size_t size) {
  const unsigned char* data = static_cast<const unsigned char*>(bytes);
  std::uint32_t lanes[8];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) {
    lanes[lane] = 0x811c9dc5u ^ (static_cast<std::uint32_t>(lane) * 0x01000193u);
  }
  for (std::size_t i = 0; i < size; ++i) {
    std::uint32_t& lane = lanes[i & 7u];
    lane ^= static_cast<std::uint32_t>(data[i]);
    lane *= 0x01000193u;  // FNV prime
  }
  Root256 root;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) root.word[lane] = lanes[lane];
  return root;
}

// A Genome's own genome_root is computed over every field except the
// genome_root slot itself (which would otherwise depend on its own value).
// Callers hash header.{abi_version, life_function_version, seed_count,
// field_count, rule_count, development_end_tick, matter_budget, flags,
// development_seed, parent_root, delta_root} followed by the seeds/fields/
// rules arrays, in declaration order -- exactly the header layout with
// genome_root skipped.
__host__ __device__ inline Root256 canonical_genome_root(const Genome& genome) {
  struct HashableHeader {
    std::uint32_t abi_version;
    std::uint32_t life_function_version;
    std::uint32_t seed_count;
    std::uint32_t field_count;
    std::uint32_t rule_count;
    std::uint32_t development_end_tick;
    std::uint32_t matter_budget;
    std::uint32_t flags;
    std::uint64_t development_seed;
    Root256 parent_root;
    Root256 delta_root;
  };
  const HashableHeader hashable{
      genome.header.abi_version,          genome.header.life_function_version,
      genome.header.seed_count,           genome.header.field_count,
      genome.header.rule_count,           genome.header.development_end_tick,
      genome.header.matter_budget,        genome.header.flags,
      genome.header.development_seed,     genome.header.parent_root,
      genome.header.delta_root,
  };
  std::uint32_t lanes[8];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) {
    lanes[lane] = 0x811c9dc5u ^ (static_cast<std::uint32_t>(lane) * 0x01000193u);
  }
  auto absorb = [&lanes](const void* bytes, std::size_t size) {
    const unsigned char* data = static_cast<const unsigned char*>(bytes);
    for (std::size_t i = 0; i < size; ++i) {
      std::uint32_t& lane = lanes[i & 7u];
      lane ^= static_cast<std::uint32_t>(data[i]);
      lane *= 0x01000193u;
    }
  };
  absorb(&hashable, sizeof(hashable));
  absorb(genome.seeds, sizeof(SeedBlock) * genome.header.seed_count);
  absorb(genome.fields, sizeof(FieldBlock) * genome.header.field_count);
  absorb(genome.rules, sizeof(ConstructionRule) * genome.header.rule_count);
  Root256 root;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) root.word[lane] = lanes[lane];
  return root;
}

enum class GenomeValidationError : std::uint32_t {
  kNone = 0,
  kAbiVersionMismatch,
  kSeedCountOutOfRange,
  kFieldCountOutOfRange,
  kRuleCountOutOfRange,
  kRuleOpcodeOutOfRange,
  kFieldReferenceOutOfRange,
  kRootMismatch,
};

// Structural legality only (fixed-capacity bounds, opcode range, field/
// child-slot cross-references, and that the caller-supplied genome_root
// matches the recomputed content root). This does not judge whether the
// recipe is a *good* one -- that is the miner's ranking law (patch 0007) --
// only whether it is a well-formed member of the ABI at all, matching
// constitutional_dependency_queue.py's role as observer/structural gate
// rather than semantic authority.
__host__ __device__ inline GenomeValidationError validate_genome(const Genome& genome) {
  if (genome.header.abi_version != kCurrentAbiVersion) {
    return GenomeValidationError::kAbiVersionMismatch;
  }
  if (genome.header.seed_count > kMaxSeeds) {
    return GenomeValidationError::kSeedCountOutOfRange;
  }
  if (genome.header.field_count > kMaxFields) {
    return GenomeValidationError::kFieldCountOutOfRange;
  }
  if (genome.header.rule_count > kMaxRules) {
    return GenomeValidationError::kRuleCountOutOfRange;
  }
  for (std::uint32_t i = 0; i < genome.header.rule_count; ++i) {
    const ConstructionRule& rule = genome.rules[i];
    if (static_cast<std::uint32_t>(rule.opcode) >= kRuleOpcodeCount) {
      return GenomeValidationError::kRuleOpcodeOutOfRange;
    }
    if (rule.field >= genome.header.field_count && rule.field != 0xffffffffu) {
      return GenomeValidationError::kFieldReferenceOutOfRange;
    }
  }
  if (canonical_genome_root(genome) != genome.header.genome_root) {
    return GenomeValidationError::kRootMismatch;
  }
  return GenomeValidationError::kNone;
}

}  // namespace substrate::direct_network::recipe

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_ABI_CUH
