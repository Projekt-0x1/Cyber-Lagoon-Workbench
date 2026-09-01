#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::resident_roles {

inline constexpr std::uint32_t kStructuralRoleCount = 64u;
inline constexpr std::uint32_t kRoleProjectionBits = 6u;
inline constexpr std::uint32_t kRoleProjectionStride = kRoleProjectionBits + 1u;
inline constexpr std::uint32_t kRoleBigramTableSize = kStructuralRoleCount * kStructuralRoleCount;
inline constexpr std::uint32_t kRoleTrigramTableSize =
    kStructuralRoleCount * kStructuralRoleCount * kStructuralRoleCount;
inline constexpr std::uint32_t kResidentRoleBlockSize = 256u;

inline constexpr std::uint32_t kBigramOutgoingTag = 0x243f6a88u;
inline constexpr std::uint32_t kBigramIncomingTag = 0x85a308d3u;
inline constexpr std::uint32_t kTrigramOutgoingTag = 0x13198a2eu;
inline constexpr std::uint32_t kTrigramBridgeTag = 0x03707344u;
inline constexpr std::uint32_t kTrigramIncomingTag = 0xa4093822u;

struct DefaultNgramAccess {
  template <typename BigramKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t bigram_previous(const BigramKey& key) {
    return static_cast<std::uint32_t>(key.previous);
  }

  template <typename BigramKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t bigram_next(const BigramKey& key) {
    return static_cast<std::uint32_t>(key.next);
  }

  template <typename TrigramKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t trigram_first(const TrigramKey& key) {
    return static_cast<std::uint32_t>(key.first);
  }

  template <typename TrigramKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t trigram_second(const TrigramKey& key) {
    return static_cast<std::uint32_t>(key.second);
  }

  template <typename TrigramKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t trigram_next(const TrigramKey& key) {
    return static_cast<std::uint32_t>(key.next);
  }
};

struct MutableStructuralRole {
  std::uint32_t role = 0u;
  std::uint32_t confidence = 0u;
  std::uint32_t evidence_depth = 0u;
};

struct DenseRoleCountTables {
  std::uint64_t* bigrams = nullptr;
  std::uint64_t* trigrams = nullptr;
};

struct ConstDenseRoleCountTables {
  const std::uint64_t* bigrams = nullptr;
  const std::uint64_t* trigrams = nullptr;
};

struct MutableRoleEvidenceTables {
  DenseRoleCountTables base_grammar{};
  DenseRoleCountTables online_content{};
};

struct RoleEvidenceTables {
  ConstDenseRoleCountTables base_grammar{};
  ConstDenseRoleCountTables online_content{};
};

struct RoleCompatibilityScore {
  std::uint64_t base_grammar = 0u;
  std::uint64_t online_content = 0u;

  [[nodiscard]] __host__ __device__ std::uint64_t total() const {
    return base_grammar + online_content;
  }
};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t mix_role_bits(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t integer_log_depth(std::uint64_t count) {
  std::uint32_t depth = 0u;
  while (count != 0u) {
    ++depth;
    count >>= 1u;
  }
  return depth;
}

// ---------------------------------------------------------------------------
// ROLE-SIGNAL CANONICALIZATION (lesionable via a nullptr map).
//
// Measured problem: severe variant fragmentation of the resident vocabulary
// ("growth" / "Growth" / "growth," are distinct units), so every unit's role
// projection is computed from a sparse per-variant sliver of context and the
// composer's per-role filler pools collapse to 1-3 candidates. The canonical
// map sends each unit to a REPRESENTATIVE unit sharing its canonical byte
// form (ASCII case fold + leading/trailing non-core-byte strip, computed
// ON-DEVICE from the unit_content the adult already stores), and role
// derivation pools the distributional context of all variants into the
// representative's projection row. Every variant then reads the SAME role,
// computed from their COMBINED (robust) context.
//
// This is byte normalization of the role STATISTIC only -- no authored
// linguistics (no word lists, no morphology rules), and the surface form of
// every unit is untouched: emitted bytes, segmentation, and the byte-suffix
// agreement signal all still see the raw stored variants.
// ---------------------------------------------------------------------------
[[nodiscard]] __host__ __device__ inline std::uint32_t canonical_unit(
    const std::uint32_t* canon, std::uint32_t unit) {
  return canon == nullptr ? unit : canon[unit];
}

// A CORE byte anchors the canonical form: ASCII alphanumerics plus all
// high-half bytes (UTF-8 payload). Edge bytes outside this set (punctuation
// variants riding on the token) are stripped from the SIGNATURE only.
[[nodiscard]] __host__ __device__ constexpr bool canon_core_byte(std::uint32_t value) {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'z') ||
         (value >= 'A' && value <= 'Z') || value >= 0x80u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t canon_fold_byte(std::uint32_t value) {
  return value >= 'A' && value <= 'Z' ? value + 32u : value;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t role_feature_hash(
    std::uint32_t first, std::uint32_t second, std::uint32_t tag, std::uint32_t projection) {
  std::uint32_t value = mix_role_bits(first ^ (tag + 0x9e3779b9u));
  value = mix_role_bits(value ^ second ^ 0x7f4a7c15u);
  return mix_role_bits(value ^ (projection + 1u) * 0x94d049bbu);
}

[[nodiscard]] __host__ __device__ constexpr std::int32_t role_feature_sign(
    std::uint32_t first, std::uint32_t second, std::uint32_t tag, std::uint32_t projection) {
  return (role_feature_hash(first, second, tag, projection) & 1u) != 0u ? 1 : -1;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t role_bigram_index(std::uint32_t previous,
                                                                            std::uint32_t next) {
  return previous * kStructuralRoleCount + next;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t role_trigram_index(std::uint32_t first,
                                                                             std::uint32_t previous,
                                                                             std::uint32_t next) {
  return (first * kStructuralRoleCount + previous) * kStructuralRoleCount + next;
}

[[nodiscard]] __host__ __device__ inline MutableStructuralRole role_from_projection(
    const std::int32_t* projection) {
  MutableStructuralRole result{};
  std::uint64_t margin = 0u;
  for (std::uint32_t bit = 0u; bit < kRoleProjectionBits; ++bit) {
    const std::int64_t value = projection[bit];
    if (value > 0)
      result.role |= 1u << bit;
    margin += static_cast<std::uint64_t>(value < 0 ? -value : value);
  }
  result.evidence_depth = projection[kRoleProjectionBits] > 0
                              ? static_cast<std::uint32_t>(projection[kRoleProjectionBits])
                              : 0u;
  const std::uint64_t possible =
      static_cast<std::uint64_t>(result.evidence_depth) * kRoleProjectionBits;
  result.confidence =
      possible == 0u ? 0u : static_cast<std::uint32_t>((margin * 255u + possible / 2u) / possible);
  if (result.confidence > 255u)
    result.confidence = 255u;
  return result;
}

[[nodiscard]] __host__ __device__ constexpr ConstDenseRoleCountTables as_const(
    DenseRoleCountTables tables) {
  return {tables.bigrams, tables.trigrams};
}

[[nodiscard]] __host__ __device__ constexpr RoleEvidenceTables as_const(
    MutableRoleEvidenceTables tables) {
  return {as_const(tables.base_grammar), as_const(tables.online_content)};
}

[[nodiscard]] __host__ __device__ inline std::uint32_t minimum_confidence(std::uint32_t first,
                                                                          std::uint32_t second) {
  return first < second ? first : second;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t score_dense_role_table(
    std::uint32_t first, std::uint32_t previous, std::uint32_t candidate,
    const MutableStructuralRole* roles, std::uint32_t unit_count,
    ConstDenseRoleCountTables tables) {
  if (first >= unit_count || previous >= unit_count || candidate >= unit_count ||
      tables.bigrams == nullptr || tables.trigrams == nullptr) {
    return 0u;
  }
  const MutableStructuralRole first_role = roles[first];
  const MutableStructuralRole previous_role = roles[previous];
  const MutableStructuralRole candidate_role = roles[candidate];
  if (first_role.role >= kStructuralRoleCount || previous_role.role >= kStructuralRoleCount ||
      candidate_role.role >= kStructuralRoleCount) {
    return 0u;
  }
  const std::uint32_t pair_confidence =
      minimum_confidence(previous_role.confidence, candidate_role.confidence);
  const std::uint32_t triple_confidence =
      minimum_confidence(first_role.confidence,
                         minimum_confidence(previous_role.confidence, candidate_role.confidence));
  const std::uint32_t pair_depth =
      integer_log_depth(tables.bigrams[role_bigram_index(previous_role.role, candidate_role.role)]);
  const std::uint32_t triple_depth = integer_log_depth(tables.trigrams[role_trigram_index(
      first_role.role, previous_role.role, candidate_role.role)]);
  return static_cast<std::uint64_t>(pair_depth) * (64u + pair_confidence) +
         static_cast<std::uint64_t>(triple_depth) * (64u + triple_confidence) * 3u;
}

[[nodiscard]] __host__ __device__ inline RoleCompatibilityScore score_role_compatibility(
    std::uint32_t first, std::uint32_t previous, std::uint32_t candidate,
    const MutableStructuralRole* roles, std::uint32_t unit_count, RoleEvidenceTables tables) {
  return {
      score_dense_role_table(first, previous, candidate, roles, unit_count, tables.base_grammar),
      score_dense_role_table(first, previous, candidate, roles, unit_count, tables.online_content)};
}

__device__ inline void add_role_projection_feature(std::int32_t* projections, std::uint32_t unit,
                                                   std::uint32_t first, std::uint32_t second,
                                                   std::uint32_t tag, std::uint32_t depth) {
  std::int32_t* row = projections + static_cast<std::size_t>(unit) * kRoleProjectionStride;
  for (std::uint32_t bit = 0u; bit < kRoleProjectionBits; ++bit) {
    atomicAdd(row + bit,
              role_feature_sign(first, second, tag, bit) * static_cast<std::int32_t>(depth));
  }
  atomicAdd(row + kRoleProjectionBits, static_cast<std::int32_t>(depth));
}

template <typename BigramKey, typename Access = DefaultNgramAccess>
__global__ void accumulate_bigram_role_projections_kernel(const BigramKey* keys,
                                                          const std::uint32_t* counts,
                                                          std::uint32_t key_count,
                                                          std::uint32_t unit_count,
                                                          std::int32_t* projections,
                                                          const std::uint32_t* canon) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_count || counts[index] == 0u)
    return;
  const std::uint32_t raw_previous = Access::bigram_previous(keys[index]);
  const std::uint32_t raw_next = Access::bigram_next(keys[index]);
  if (raw_previous >= unit_count || raw_next >= unit_count)
    return;
  // Canonicalized on BOTH sides: the destination row pools every variant's
  // context, and the context identity itself is variant-invariant (so
  // "growth is" and "Growth, is" contribute the SAME feature).
  const std::uint32_t previous = canonical_unit(canon, raw_previous);
  const std::uint32_t next = canonical_unit(canon, raw_next);
  const std::uint32_t depth = integer_log_depth(counts[index]);
  add_role_projection_feature(projections, previous, next, 0u, kBigramOutgoingTag, depth);
  add_role_projection_feature(projections, next, previous, 0u, kBigramIncomingTag, depth);
}

template <typename TrigramKey, typename Access = DefaultNgramAccess>
__global__ void accumulate_trigram_role_projections_kernel(const TrigramKey* keys,
                                                           const std::uint32_t* counts,
                                                           std::uint32_t key_count,
                                                           std::uint32_t unit_count,
                                                           std::int32_t* projections,
                                                           const std::uint32_t* canon) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_count || counts[index] == 0u)
    return;
  const std::uint32_t raw_first = Access::trigram_first(keys[index]);
  const std::uint32_t raw_second = Access::trigram_second(keys[index]);
  const std::uint32_t raw_next = Access::trigram_next(keys[index]);
  if (raw_first >= unit_count || raw_second >= unit_count || raw_next >= unit_count)
    return;
  const std::uint32_t first = canonical_unit(canon, raw_first);
  const std::uint32_t second = canonical_unit(canon, raw_second);
  const std::uint32_t next = canonical_unit(canon, raw_next);
  const std::uint32_t depth = integer_log_depth(counts[index]);
  add_role_projection_feature(projections, first, second, next, kTrigramOutgoingTag, depth);
  add_role_projection_feature(projections, second, first, next, kTrigramBridgeTag, depth);
  add_role_projection_feature(projections, next, first, second, kTrigramIncomingTag, depth);
}

static __global__ void finalize_structural_roles_kernel(const std::int32_t* projections,
                                                        std::uint32_t unit_count,
                                                        MutableStructuralRole* roles,
                                                        const std::uint32_t* canon) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count)
    return;
  // Every variant reads its canonical representative's pooled projection.
  roles[unit] = role_from_projection(
      projections +
      static_cast<std::size_t>(canonical_unit(canon, unit)) * kRoleProjectionStride);
}

// Canonical byte-form signature of each unit (FNV-1a over the case-folded
// core byte span). 0 marks a unit with no core byte (pure punctuation /
// separator matter): such units stay self-canonical. Runs entirely on-device
// over the resident unit_content -- the bytes never cross to the host.
static __global__ void compute_role_canon_signature_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit_count,
    unsigned long long* signatures) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count)
    return;
  const std::uint32_t length = unit_lengths[unit];
  std::uint32_t first = length;
  std::uint32_t last = 0u;
  bool any = false;
  for (std::uint32_t offset = 0u; offset < length; ++offset) {
    const std::uint32_t word = unit_content[unit * unit_words + offset / 4u];
    const std::uint32_t value = (word >> ((offset % 4u) * 8u)) & 0xffu;
    if (!canon_core_byte(value))
      continue;
    if (!any) {
      first = offset;
      any = true;
    }
    last = offset;
  }
  if (!any) {
    signatures[unit] = 0ull;
    return;
  }
  unsigned long long hash = 1469598103934665603ull;
  for (std::uint32_t offset = first; offset <= last; ++offset) {
    const std::uint32_t word = unit_content[unit * unit_words + offset / 4u];
    const std::uint32_t value = (word >> ((offset % 4u) * 8u)) & 0xffu;
    hash = (hash ^ canon_fold_byte(value)) * 1099511628211ull;
  }
  signatures[unit] = hash == 0ull ? 1ull : hash;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t role_canon_probe(
    unsigned long long signature) {
  return mix_role_bits(static_cast<std::uint32_t>(signature) ^
                       static_cast<std::uint32_t>(signature >> 32u));
}

// Claim, per canonical signature, the lowest-numbered unit as representative
// (open-addressed device hash table; 64-bit signatures make cross-form
// collisions negligible at resident vocabulary scale).
static __global__ void claim_role_canon_representatives_kernel(
    const unsigned long long* signatures, std::uint32_t unit_count,
    unsigned long long* table_keys, std::uint32_t* table_reps,
    std::uint32_t table_mask) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count)
    return;
  const unsigned long long signature = signatures[unit];
  if (signature == 0ull)
    return;
  std::uint32_t probe = role_canon_probe(signature) & table_mask;
  for (std::uint32_t attempt = 0u; attempt <= table_mask; ++attempt) {
    const unsigned long long previous = atomicCAS(&table_keys[probe], 0ull, signature);
    if (previous == 0ull || previous == signature) {
      atomicMin(&table_reps[probe], unit);
      return;
    }
    probe = (probe + 1u) & table_mask;
  }
}

static __global__ void resolve_role_canon_kernel(
    const unsigned long long* signatures, std::uint32_t unit_count,
    const unsigned long long* table_keys, const std::uint32_t* table_reps,
    std::uint32_t table_mask, std::uint32_t* canon) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count)
    return;
  const unsigned long long signature = signatures[unit];
  if (signature == 0ull) {
    canon[unit] = unit;
    return;
  }
  std::uint32_t probe = role_canon_probe(signature) & table_mask;
  for (std::uint32_t attempt = 0u; attempt <= table_mask; ++attempt) {
    if (table_keys[probe] == signature) {
      canon[unit] = table_reps[probe];
      return;
    }
    probe = (probe + 1u) & table_mask;
  }
  canon[unit] = unit;  // unreachable when the claim pass completed
}

// Build the unit -> canonical-representative map fully on-device.
// table_size must be a power of two and comfortably exceed unit_count.
inline cudaError_t build_role_canonical_map_cuda(
    std::uint32_t unit_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    unsigned long long* signature_scratch, unsigned long long* table_keys,
    std::uint32_t* table_reps, std::uint32_t table_size, std::uint32_t* canon,
    cudaStream_t stream = nullptr) {
  if (unit_count == 0u)
    return cudaSuccess;
  cudaError_t status =
      cudaMemsetAsync(table_keys, 0, table_size * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(table_reps, 0xff, table_size * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  const std::uint32_t blocks = (unit_count + kResidentRoleBlockSize - 1u) / kResidentRoleBlockSize;
  compute_role_canon_signature_kernel<<<blocks, kResidentRoleBlockSize, 0, stream>>>(
      unit_lengths, unit_content, unit_words, unit_count, signature_scratch);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  const std::uint32_t table_mask = table_size - 1u;
  claim_role_canon_representatives_kernel<<<blocks, kResidentRoleBlockSize, 0, stream>>>(
      signature_scratch, unit_count, table_keys, table_reps, table_mask);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  resolve_role_canon_kernel<<<blocks, kResidentRoleBlockSize, 0, stream>>>(
      signature_scratch, unit_count, table_keys, table_reps, table_mask, canon);
  return cudaPeekAtLastError();
}

template <typename BigramKey, typename Access = DefaultNgramAccess>
__global__ void build_role_bigram_table_kernel(const BigramKey* keys, const std::uint32_t* counts,
                                               std::uint32_t key_count,
                                               const MutableStructuralRole* roles,
                                               std::uint32_t unit_count, std::uint64_t* table) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_count || counts[index] == 0u)
    return;
  const std::uint32_t previous = Access::bigram_previous(keys[index]);
  const std::uint32_t next = Access::bigram_next(keys[index]);
  if (previous >= unit_count || next >= unit_count)
    return;
  const std::uint32_t previous_role = roles[previous].role;
  const std::uint32_t next_role = roles[next].role;
  if (previous_role >= kStructuralRoleCount || next_role >= kStructuralRoleCount)
    return;
  atomicAdd(
      reinterpret_cast<unsigned long long*>(table + role_bigram_index(previous_role, next_role)),
      static_cast<unsigned long long>(counts[index]));
}

template <typename TrigramKey, typename Access = DefaultNgramAccess>
__global__ void build_role_trigram_table_kernel(const TrigramKey* keys, const std::uint32_t* counts,
                                                std::uint32_t key_count,
                                                const MutableStructuralRole* roles,
                                                std::uint32_t unit_count, std::uint64_t* table) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_count || counts[index] == 0u)
    return;
  const std::uint32_t first = Access::trigram_first(keys[index]);
  const std::uint32_t second = Access::trigram_second(keys[index]);
  const std::uint32_t next = Access::trigram_next(keys[index]);
  if (first >= unit_count || second >= unit_count || next >= unit_count)
    return;
  const std::uint32_t first_role = roles[first].role;
  const std::uint32_t second_role = roles[second].role;
  const std::uint32_t next_role = roles[next].role;
  if (first_role >= kStructuralRoleCount || second_role >= kStructuralRoleCount ||
      next_role >= kStructuralRoleCount) {
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(
                table + role_trigram_index(first_role, second_role, next_role)),
            static_cast<unsigned long long>(counts[index]));
}

[[nodiscard]] inline constexpr std::size_t role_projection_scratch_words(std::uint32_t unit_count) {
  return static_cast<std::size_t>(unit_count) * kRoleProjectionStride;
}

[[nodiscard]] inline constexpr std::uint32_t resident_role_blocks(std::uint32_t count) {
  return (count + kResidentRoleBlockSize - 1u) / kResidentRoleBlockSize;
}

template <typename BigramKey, typename TrigramKey, typename Access = DefaultNgramAccess>
inline cudaError_t derive_structural_roles_cuda(
    std::uint32_t unit_count, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, const std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count, const BigramKey* online_bigrams,
    const std::uint32_t* online_bigram_counts, std::uint32_t online_bigram_count,
    const TrigramKey* online_trigrams, const std::uint32_t* online_trigram_counts,
    std::uint32_t online_trigram_count, std::int32_t* projection_scratch,
    MutableStructuralRole* roles, const std::uint32_t* canon = nullptr,
    cudaStream_t stream = nullptr) {
  if (unit_count == 0u)
    return cudaSuccess;
  cudaError_t status =
      cudaMemsetAsync(projection_scratch, 0,
                      role_projection_scratch_words(unit_count) * sizeof(std::int32_t), stream);
  if (status != cudaSuccess)
    return status;

  if (base_bigram_count != 0u) {
    accumulate_bigram_role_projections_kernel<BigramKey, Access>
        <<<resident_role_blocks(base_bigram_count), kResidentRoleBlockSize, 0, stream>>>(
            base_bigrams, base_bigram_counts, base_bigram_count, unit_count, projection_scratch,
            canon);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  if (base_trigram_count != 0u) {
    accumulate_trigram_role_projections_kernel<TrigramKey, Access>
        <<<resident_role_blocks(base_trigram_count), kResidentRoleBlockSize, 0, stream>>>(
            base_trigrams, base_trigram_counts, base_trigram_count, unit_count, projection_scratch,
            canon);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  if (online_bigram_count != 0u) {
    accumulate_bigram_role_projections_kernel<BigramKey, Access>
        <<<resident_role_blocks(online_bigram_count), kResidentRoleBlockSize, 0, stream>>>(
            online_bigrams, online_bigram_counts, online_bigram_count, unit_count,
            projection_scratch, canon);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  if (online_trigram_count != 0u) {
    accumulate_trigram_role_projections_kernel<TrigramKey, Access>
        <<<resident_role_blocks(online_trigram_count), kResidentRoleBlockSize, 0, stream>>>(
            online_trigrams, online_trigram_counts, online_trigram_count, unit_count,
            projection_scratch, canon);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  finalize_structural_roles_kernel<<<resident_role_blocks(unit_count), kResidentRoleBlockSize, 0,
                                     stream>>>(projection_scratch, unit_count, roles, canon);
  return cudaPeekAtLastError();
}

template <typename BigramKey, typename TrigramKey, typename Access = DefaultNgramAccess>
inline cudaError_t build_dense_role_tables_cuda(
    const MutableStructuralRole* roles, std::uint32_t unit_count, const BigramKey* bigrams,
    const std::uint32_t* bigram_counts, std::uint32_t bigram_count, const TrigramKey* trigrams,
    const std::uint32_t* trigram_counts, std::uint32_t trigram_count, DenseRoleCountTables output,
    cudaStream_t stream = nullptr) {
  cudaError_t status =
      cudaMemsetAsync(output.bigrams, 0, kRoleBigramTableSize * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status =
      cudaMemsetAsync(output.trigrams, 0, kRoleTrigramTableSize * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  if (bigram_count != 0u) {
    build_role_bigram_table_kernel<BigramKey, Access>
        <<<resident_role_blocks(bigram_count), kResidentRoleBlockSize, 0, stream>>>(
            bigrams, bigram_counts, bigram_count, roles, unit_count, output.bigrams);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  if (trigram_count != 0u) {
    build_role_trigram_table_kernel<TrigramKey, Access>
        <<<resident_role_blocks(trigram_count), kResidentRoleBlockSize, 0, stream>>>(
            trigrams, trigram_counts, trigram_count, roles, unit_count, output.trigrams);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  return cudaSuccess;
}

template <typename BigramKey, typename TrigramKey, typename Access = DefaultNgramAccess>
inline cudaError_t build_role_evidence_tables_cuda(
    const MutableStructuralRole* roles, std::uint32_t unit_count, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, const std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count, const BigramKey* online_bigrams,
    const std::uint32_t* online_bigram_counts, std::uint32_t online_bigram_count,
    const TrigramKey* online_trigrams, const std::uint32_t* online_trigram_counts,
    std::uint32_t online_trigram_count, MutableRoleEvidenceTables output,
    cudaStream_t stream = nullptr) {
  cudaError_t status = build_dense_role_tables_cuda<BigramKey, TrigramKey, Access>(
      roles, unit_count, base_bigrams, base_bigram_counts, base_bigram_count, base_trigrams,
      base_trigram_counts, base_trigram_count, output.base_grammar, stream);
  if (status != cudaSuccess)
    return status;
  return build_dense_role_tables_cuda<BigramKey, TrigramKey, Access>(
      roles, unit_count, online_bigrams, online_bigram_counts, online_bigram_count, online_trigrams,
      online_trigram_counts, online_trigram_count, output.online_content, stream);
}

// ---------------------------------------------------------------------------
// NEIGHBOR-ENTROPY GLUE DISCOVERY (context dispersion, on-device).
//
// Measured regression this replaces: the frequency-rank closed-class
// statistic absorbs high-FREQUENCY CONTENT words ("work", "economic") as
// glue, excluding them from the composer's content-slot filler pools -- the
// composer then reaches for the wrong word ("human worm" instead of "human
// work"). Frequency is the WRONG statistic for glue: what distinguishes a
// function word is not how often it occurs but how FREELY it combines.
//
// The replacement statistic: per canonical group, the mean of the left and
// right neighbor-entropy of the group's resident bigram distribution
// (canonical-pair pooled). True glue (the/of/and/to/that/...) combines with
// nearly anything -> HIGH entropy on BOTH sides; content words -- however
// frequent -- keep selective contexts -> lower entropy. The glue set is cut
// by a threshold DISCOVERED from the entropy distribution itself (Otsu split
// of the upper half-range), not by an authored value and not by a fixed
// frequency rank. Asymmetric artifacts (high dispersion on one side only,
// e.g. discourse tics like "see") are guarded by requiring the WEAKER side's
// entropy to sit within a fixed slack of the cutoff.
//
// Doctrine: entirely on-device from the resident neighbor (bigram) counts --
// no unit byte and no count crosses to the host for this signal; the mask is
// written by a device kernel. Learned (counts come from residence +
// assimilation), resident (publishes into the same closed-class mask all
// construction kernels already read), lesionable (caller keeps the legacy
// frequency mask when lesioned).
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kEntropyGlueHistogramBins = 2048u;
inline constexpr float kEntropyGlueHistogramMaxBits = 16.0f;
// The weaker side of a glue group must reach cutoff minus this slack (bits).
inline constexpr float kEntropyGlueSideSlackBits = 1.0f;

[[nodiscard]] __host__ __device__ inline unsigned long long entropy_glue_mix(
    unsigned long long key) {
  key += 0x9e3779b97f4a7c15ull;
  key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
  key = (key ^ (key >> 27)) * 0x94d049bb133111ebull;
  return key ^ (key >> 31);
}

// Accumulate canonical-pair neighbor counts into an open-addressed device
// hash table. Keys pack (canonical_prev + 1, canonical_next + 1) so the empty
// slot sentinel 0 can never collide with a real pair.
template <typename BigramKey, typename Access = DefaultNgramAccess>
__global__ void accumulate_canonical_neighbor_pairs_kernel(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t pair_count,
    const std::uint32_t* canon, std::uint32_t unit_count,
    unsigned long long* table_keys, unsigned long long* table_counts,
    std::uint32_t table_mask) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pair_count) return;
  const std::uint32_t raw_previous = Access::bigram_previous(keys[index]);
  const std::uint32_t raw_next = Access::bigram_next(keys[index]);
  if (raw_previous >= unit_count || raw_next >= unit_count) return;
  const std::uint32_t weight = counts[index];
  if (weight == 0u) return;
  const unsigned long long previous = canonical_unit(canon, raw_previous);
  const unsigned long long next = canonical_unit(canon, raw_next);
  const unsigned long long key = ((previous + 1ull) << 32) | (next + 1ull);
  std::uint32_t slot =
      static_cast<std::uint32_t>(entropy_glue_mix(key)) & table_mask;
  for (std::uint32_t probe = 0u; probe <= table_mask; ++probe) {
    const unsigned long long resident = atomicCAS(&table_keys[slot], 0ull, key);
    if (resident == 0ull || resident == key) {
      atomicAdd(&table_counts[slot], static_cast<unsigned long long>(weight));
      return;
    }
    slot = (slot + 1u) & table_mask;
  }
}

// Per-group neighbor totals from the deduplicated canonical-pair table.
static __global__ void reduce_neighbor_totals_kernel(
    const unsigned long long* table_keys, const unsigned long long* table_counts,
    std::uint32_t table_size, unsigned long long* right_totals,
    unsigned long long* left_totals) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= table_size) return;
  const unsigned long long key = table_keys[index];
  if (key == 0ull) return;
  const std::uint32_t previous = static_cast<std::uint32_t>(key >> 32) - 1u;
  const std::uint32_t next = static_cast<std::uint32_t>(key & 0xffffffffull) - 1u;
  const unsigned long long weight = table_counts[index];
  atomicAdd(&right_totals[previous], weight);
  atomicAdd(&left_totals[next], weight);
}

// Per-group sum of c*log2(c) over the deduplicated canonical-pair counts;
// with the totals this yields H = log2(T) - S/T without a second pass over
// normalized probabilities.
static __global__ void reduce_neighbor_plogp_kernel(
    const unsigned long long* table_keys, const unsigned long long* table_counts,
    std::uint32_t table_size, double* right_plogp, double* left_plogp) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= table_size) return;
  const unsigned long long key = table_keys[index];
  if (key == 0ull) return;
  const std::uint32_t previous = static_cast<std::uint32_t>(key >> 32) - 1u;
  const std::uint32_t next = static_cast<std::uint32_t>(key & 0xffffffffull) - 1u;
  const double weight = static_cast<double>(table_counts[index]);
  const double contribution = weight * log2(weight);
  atomicAdd(&right_plogp[previous], contribution);
  atomicAdd(&left_plogp[next], contribution);
}

// Finalize per-group entropies (mean and weaker side), histogram the means,
// and track the maximum. Groups missing either side stay unqualified (-1).
static __global__ void finalize_neighbor_entropy_kernel(
    std::uint32_t unit_count, const unsigned long long* right_totals,
    const unsigned long long* left_totals, const double* right_plogp,
    const double* left_plogp, float* entropy_mean, float* entropy_min,
    std::uint32_t* histogram, int* max_entropy_bits) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const unsigned long long right_total = right_totals[unit];
  const unsigned long long left_total = left_totals[unit];
  if (right_total == 0ull || left_total == 0ull) {
    entropy_mean[unit] = -1.0f;
    entropy_min[unit] = -1.0f;
    return;
  }
  const double right_t = static_cast<double>(right_total);
  const double left_t = static_cast<double>(left_total);
  const double right_entropy = log2(right_t) - right_plogp[unit] / right_t;
  const double left_entropy = log2(left_t) - left_plogp[unit] / left_t;
  const float mean = static_cast<float>(0.5 * (right_entropy + left_entropy));
  const float weaker = static_cast<float>(
      right_entropy < left_entropy ? right_entropy : left_entropy);
  entropy_mean[unit] = mean;
  entropy_min[unit] = weaker;
  const float clamped = mean < 0.0f ? 0.0f
                        : (mean >= kEntropyGlueHistogramMaxBits
                               ? kEntropyGlueHistogramMaxBits
                               : mean);
  std::uint32_t bin = static_cast<std::uint32_t>(
      clamped * (static_cast<float>(kEntropyGlueHistogramBins) /
                 kEntropyGlueHistogramMaxBits));
  if (bin >= kEntropyGlueHistogramBins) bin = kEntropyGlueHistogramBins - 1u;
  atomicAdd(&histogram[bin], 1u);
  atomicMax(max_entropy_bits, __float_as_int(mean));
}

// Discover the glue cutoff from the entropy distribution itself: Otsu's
// between-class split applied to the UPPER HALF of the observed range. The
// lower half (rare and mid-selectivity content) would drag a whole-range
// split into the content bulk; the upper half isolates exactly the
// content-tail vs free-combining-glue boundary. Single thread: the histogram
// is tiny and this runs once per construction learning pass.
static __global__ void select_entropy_glue_cutoff_kernel(
    const std::uint32_t* histogram, const int* max_entropy_bits,
    float* cutoff_out) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const float max_entropy = __int_as_float(*max_entropy_bits);
  if (!(max_entropy > 0.0f)) {
    // Degenerate stream (no qualified group): publish an unreachable cutoff
    // so the mask stays empty rather than absorbing everything as glue.
    *cutoff_out = 3.402823466e+38f;
    return;
  }
  const float bin_width =
      kEntropyGlueHistogramMaxBits / static_cast<float>(kEntropyGlueHistogramBins);
  std::uint32_t low_bin =
      static_cast<std::uint32_t>((0.5f * max_entropy) / bin_width);
  if (low_bin >= kEntropyGlueHistogramBins) low_bin = kEntropyGlueHistogramBins - 1u;
  double total_weight = 0.0;
  double total_moment = 0.0;
  for (std::uint32_t bin = low_bin; bin < kEntropyGlueHistogramBins; ++bin) {
    const double weight = static_cast<double>(histogram[bin]);
    const double center = (static_cast<double>(bin) + 0.5) * bin_width;
    total_weight += weight;
    total_moment += weight * center;
  }
  if (total_weight < 2.0) {
    *cutoff_out = 3.402823466e+38f;
    return;
  }
  double below_weight = 0.0;
  double below_moment = 0.0;
  double best_variance = -1.0;
  float best_cutoff = 3.402823466e+38f;
  for (std::uint32_t bin = low_bin; bin + 1u < kEntropyGlueHistogramBins; ++bin) {
    const double weight = static_cast<double>(histogram[bin]);
    const double center = (static_cast<double>(bin) + 0.5) * bin_width;
    below_weight += weight;
    below_moment += weight * center;
    const double above_weight = total_weight - below_weight;
    if (below_weight <= 0.0 || above_weight <= 0.0) continue;
    const double below_mean = below_moment / below_weight;
    const double above_mean = (total_moment - below_moment) / above_weight;
    const double separation = below_mean - above_mean;
    const double variance = below_weight * above_weight * separation * separation;
    if (variance > best_variance) {
      best_variance = variance;
      // Threshold sits at the UPPER edge of the below-class bin: glue is
      // everything strictly above the split.
      best_cutoff = static_cast<float>((static_cast<double>(bin) + 1.0) * bin_width);
    }
  }
  *cutoff_out = best_cutoff;
}

// Publish the glue mask: a unit is glue when its canonical group's mean
// neighbor-entropy clears the discovered cutoff AND its weaker side stays
// within the slack (kills one-sided dispersion artifacts). Every surface
// variant reads its canonical representative's pooled statistic.
static __global__ void publish_entropy_glue_mask_kernel(
    std::uint32_t unit_count, const std::uint32_t* canon,
    const float* entropy_mean, const float* entropy_min, const float* cutoff,
    std::uint32_t* closed_class_mask) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const std::uint32_t representative = canonical_unit(canon, unit);
  const float mean = entropy_mean[representative];
  const float weaker = entropy_min[representative];
  const float threshold = *cutoff;
  const bool glue =
      mean >= threshold && weaker >= threshold - kEntropyGlueSideSlackBits;
  closed_class_mask[unit] = glue ? 1u : 0u;
}

// Scratch geometry for discover_entropy_glue_cuda; the caller owns the
// device allocations so transient learning workspace stays in its hands.
[[nodiscard]] inline std::uint32_t entropy_glue_table_size(
    std::uint32_t base_bigram_count, std::uint32_t online_bigram_count) {
  const std::uint64_t needed =
      2ull * (static_cast<std::uint64_t>(base_bigram_count) +
              static_cast<std::uint64_t>(online_bigram_count));
  std::uint32_t size = 1024u;
  while (size < needed && size < (1u << 24)) size <<= 1u;
  return size;
}

// Compute the neighbor-entropy glue mask fully on-device from the resident
// bigram tables. entropy_mean/entropy_min/cutoff stay resident for
// diagnostics; the mask is the published authority.
template <typename BigramKey, typename Access = DefaultNgramAccess>
inline cudaError_t discover_entropy_glue_cuda(
    std::uint32_t unit_count, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count, const std::uint32_t* canon,
    unsigned long long* table_keys, unsigned long long* table_counts,
    std::uint32_t table_size, unsigned long long* right_totals,
    unsigned long long* left_totals, double* right_plogp, double* left_plogp,
    std::uint32_t* histogram, int* max_entropy_bits, float* entropy_mean,
    float* entropy_min, float* cutoff, std::uint32_t* closed_class_mask,
    cudaStream_t stream = nullptr) {
  if (unit_count == 0u) return cudaSuccess;
  cudaError_t status =
      cudaMemsetAsync(table_keys, 0, table_size * sizeof(unsigned long long), stream);
  if (status != cudaSuccess) return status;
  status =
      cudaMemsetAsync(table_counts, 0, table_size * sizeof(unsigned long long), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(right_totals, 0, unit_count * sizeof(unsigned long long),
                           stream);
  if (status != cudaSuccess) return status;
  status =
      cudaMemsetAsync(left_totals, 0, unit_count * sizeof(unsigned long long), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(right_plogp, 0, unit_count * sizeof(double), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(left_plogp, 0, unit_count * sizeof(double), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(histogram, 0,
                           kEntropyGlueHistogramBins * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(max_entropy_bits, 0, sizeof(int), stream);
  if (status != cudaSuccess) return status;
  const std::uint32_t table_mask = table_size - 1u;
  if (base_bigram_count != 0u) {
    accumulate_canonical_neighbor_pairs_kernel<BigramKey, Access>
        <<<resident_role_blocks(base_bigram_count), kResidentRoleBlockSize, 0, stream>>>(
            base_bigrams, base_bigram_counts, base_bigram_count, canon, unit_count,
            table_keys, table_counts, table_mask);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
  }
  if (online_bigram_count != 0u) {
    accumulate_canonical_neighbor_pairs_kernel<BigramKey, Access>
        <<<resident_role_blocks(online_bigram_count), kResidentRoleBlockSize, 0,
           stream>>>(online_bigrams, online_bigram_counts, online_bigram_count, canon,
                     unit_count, table_keys, table_counts, table_mask);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
  }
  reduce_neighbor_totals_kernel<<<resident_role_blocks(table_size),
                                  kResidentRoleBlockSize, 0, stream>>>(
      table_keys, table_counts, table_size, right_totals, left_totals);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  reduce_neighbor_plogp_kernel<<<resident_role_blocks(table_size),
                                 kResidentRoleBlockSize, 0, stream>>>(
      table_keys, table_counts, table_size, right_plogp, left_plogp);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  finalize_neighbor_entropy_kernel<<<resident_role_blocks(unit_count),
                                     kResidentRoleBlockSize, 0, stream>>>(
      unit_count, right_totals, left_totals, right_plogp, left_plogp, entropy_mean,
      entropy_min, histogram, max_entropy_bits);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  select_entropy_glue_cutoff_kernel<<<1u, 1u, 0, stream>>>(histogram,
                                                           max_entropy_bits, cutoff);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  publish_entropy_glue_mask_kernel<<<resident_role_blocks(unit_count),
                                     kResidentRoleBlockSize, 0, stream>>>(
      unit_count, canon, entropy_mean, entropy_min, cutoff, closed_class_mask);
  return cudaPeekAtLastError();
}

}  // namespace substrate::bcc32::resident_roles
