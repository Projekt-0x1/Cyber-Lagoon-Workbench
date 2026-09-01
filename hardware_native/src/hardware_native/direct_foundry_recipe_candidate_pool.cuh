#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_RECIPE_CANDIDATE_POOL_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_RECIPE_CANDIDATE_POOL_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

enum class DirectFoundryCandidateKindV1 : std::uint32_t {
  recipe_revision = 1u,
  guarded_solver_lowering = 2u,
  network_motif_or_macro = 3u,
  n_next_composition = 4u,
  p_n_plus_one_recipe = 5u,
};

// This is the class of evidence claimed by the submitted artifact. Merely
// recording a claim in the pool does not verify it or authorize its use.
enum class DirectFoundryProofClassV1 : std::uint32_t {
  unverified_proposal = 1u,
  logical_equivalence_claim = 2u,
  physical_efficiency_claim = 3u,
  developmental_prior_claim = 4u,
};

// Observer-side, semantic-opaque candidate metadata. Every address names an
// immutable external artifact or receipt. The pool has no Adult pointer,
// occurrence identity, activation bit, score, popularity, or winner field.
struct DirectFoundryCandidateRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint32_t abi_version = 1u;
  std::uint32_t generation = 0u;
  std::uint32_t derivation_rank = 0u;
  std::uint32_t formal_port_count = 0u;
  DirectFoundryCandidateKindV1 kind =
      DirectFoundryCandidateKindV1::recipe_revision;
  DirectFoundryProofClassV1 proof_class =
      DirectFoundryProofClassV1::unverified_proposal;
  DirectSha256Address candidate_body{};
  DirectSha256Address parent_candidate{};
  DirectSha256Address construction_ancestry{};
  DirectSha256Address authorship{};
  DirectSha256Address formal_ports{};
  DirectSha256Address domain{};
  DirectSha256Address guard{};
  DirectSha256Address proof_claim{};
  DirectSha256Address compatible_evaluator{};
  DirectSha256Address compatible_species{};
  DirectSha256Address resource_receipt{};
  DirectSha256Address contextual_failures{};
  DirectSha256Address falsifiers{};
};

struct DirectFoundryCandidateEntryV1 {
  DirectFoundryCandidateRecordV1 record{};
  DirectSha256Address candidate_address{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

static_assert(std::is_standard_layout_v<DirectFoundryCandidateRecordV1> &&
              std::is_trivially_copyable_v<DirectFoundryCandidateRecordV1>);
static_assert(std::is_standard_layout_v<DirectFoundryCandidateEntryV1> &&
              std::is_trivially_copyable_v<DirectFoundryCandidateEntryV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_HD
#endif

DIRECT_FOUNDRY_HD inline bool direct_foundry_address_is_nonzero(
    const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte)
    any |= byte;
  return any != 0u;
}

template <std::size_t Capacity>
class DirectFoundryRecipeCandidatePool {
 public:
  static_assert(Capacity > 0u);

  bool append(const DirectFoundryCandidateRecordV1& record) {
    if (size_ == Capacity || record.sequence != size_ || !valid_record(record))
      return false;

    const DirectSha256Address candidate = candidate_address(record);
    if (find_index(candidate, size_) != size_)
      return false;

    if (record.generation == 0u) {
      if (direct_foundry_address_is_nonzero(record.parent_candidate))
        return false;
    } else {
      const std::size_t parent = find_index(record.parent_candidate, size_);
      if (parent == size_ ||
          record.generation != entries_[parent].record.generation + 1u)
        return false;
    }

    DirectFoundryCandidateEntryV1 entry{};
    entry.record = record;
    entry.candidate_address = candidate;
    entry.prior_head = head_;
    entry.record_address = record_address(record.sequence, candidate);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return size_; }
  static constexpr std::size_t capacity() { return Capacity; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectFoundryCandidateEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  bool find(const DirectSha256Address& candidate,
            DirectFoundryCandidateEntryV1* out) const {
    const std::size_t index = find_index(candidate, size_);
    return index != size_ && read(index, out);
  }

  bool parent_of(const DirectFoundryCandidateEntryV1& child,
                 DirectFoundryCandidateEntryV1* out) const {
    if (child.record.generation == 0u || out == nullptr)
      return false;
    return find(child.record.parent_candidate, out);
  }

  bool verify() const {
    return verify_entries(entries_.data(), size_, head_);
  }

  static bool verify_entries(const DirectFoundryCandidateEntryV1* entries,
                             std::size_t count,
                             const DirectSha256Address& expected_head) {
    if ((count != 0u && entries == nullptr) || count > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t i = 0u; i < count; ++i) {
      const auto& entry = entries[i];
      if (entry.record.sequence != i || !valid_record(entry.record) ||
          entry.candidate_address != candidate_address(entry.record) ||
          entry.prior_head != prior ||
          entry.record_address != record_address(i, entry.candidate_address) ||
          entry.chain_head != next_head(prior, entry.record_address))
        return false;

      for (std::size_t j = 0u; j < i; ++j)
        if (entries[j].candidate_address == entry.candidate_address)
          return false;

      if (entry.record.generation == 0u) {
        if (direct_foundry_address_is_nonzero(entry.record.parent_candidate))
          return false;
      } else {
        const std::size_t parent = find_index_in(
            entries, i, entry.record.parent_candidate);
        if (parent == i ||
            entry.record.generation != entries[parent].record.generation + 1u)
          return false;
      }
      prior = entry.chain_head;
    }
    return prior == expected_head;
  }

  // Candidate identity is independent of append order. Sequence participates
  // only in the append-only record and chain addresses below.
  DIRECT_FOUNDRY_HD static DirectSha256Address candidate_address(
      const DirectFoundryCandidateRecordV1& record) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-recipe-candidate-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u32(&state, record.abi_version);
    update_u32(&state, record.generation);
    update_u32(&state, record.derivation_rank);
    update_u32(&state, record.formal_port_count);
    update_u32(&state, static_cast<std::uint32_t>(record.kind));
    update_u32(&state, static_cast<std::uint32_t>(record.proof_class));
    update_address(&state, record.candidate_body);
    update_address(&state, record.parent_candidate);
    update_address(&state, record.construction_ancestry);
    update_address(&state, record.authorship);
    update_address(&state, record.formal_ports);
    update_address(&state, record.domain);
    update_address(&state, record.guard);
    update_address(&state, record.proof_claim);
    update_address(&state, record.compatible_evaluator);
    update_address(&state, record.compatible_species);
    update_address(&state, record.resource_receipt);
    update_address(&state, record.contextual_failures);
    update_address(&state, record.falsifiers);
    return state.finish();
  }

 private:
  static bool known_kind(DirectFoundryCandidateKindV1 kind) {
    return kind == DirectFoundryCandidateKindV1::recipe_revision ||
           kind == DirectFoundryCandidateKindV1::guarded_solver_lowering ||
           kind == DirectFoundryCandidateKindV1::network_motif_or_macro ||
           kind == DirectFoundryCandidateKindV1::n_next_composition ||
           kind == DirectFoundryCandidateKindV1::p_n_plus_one_recipe;
  }

  static bool known_proof_class(DirectFoundryProofClassV1 proof_class) {
    return proof_class == DirectFoundryProofClassV1::unverified_proposal ||
           proof_class ==
               DirectFoundryProofClassV1::logical_equivalence_claim ||
           proof_class ==
               DirectFoundryProofClassV1::physical_efficiency_claim ||
           proof_class ==
               DirectFoundryProofClassV1::developmental_prior_claim;
  }

  static bool valid_record(const DirectFoundryCandidateRecordV1& record) {
    return record.abi_version == 1u && known_kind(record.kind) &&
           known_proof_class(record.proof_class) &&
           record.formal_port_count != 0u &&
           direct_foundry_address_is_nonzero(record.candidate_body) &&
           direct_foundry_address_is_nonzero(record.construction_ancestry) &&
           direct_foundry_address_is_nonzero(record.authorship) &&
           direct_foundry_address_is_nonzero(record.formal_ports) &&
           direct_foundry_address_is_nonzero(record.domain) &&
           direct_foundry_address_is_nonzero(record.guard) &&
           direct_foundry_address_is_nonzero(record.proof_claim) &&
           direct_foundry_address_is_nonzero(record.compatible_evaluator) &&
           direct_foundry_address_is_nonzero(record.compatible_species) &&
           direct_foundry_address_is_nonzero(record.resource_receipt) &&
           direct_foundry_address_is_nonzero(record.contextual_failures) &&
           direct_foundry_address_is_nonzero(record.falsifiers);
  }

  std::size_t find_index(const DirectSha256Address& candidate,
                         std::size_t count) const {
    return find_index_in(entries_.data(), count, candidate);
  }

  static std::size_t find_index_in(
      const DirectFoundryCandidateEntryV1* entries, std::size_t count,
      const DirectSha256Address& candidate) {
    for (std::size_t i = 0u; i < count; ++i)
      if (entries[i].candidate_address == candidate)
        return i;
    return count;
  }

  DIRECT_FOUNDRY_HD static void update_u32(detail::DirectSha256State* state,
                                           std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_HD static void update_u64(detail::DirectSha256State* state,
                                           std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_HD static void update_address(
      detail::DirectSha256State* state, const DirectSha256Address& address) {
    state->update(address.byte, sizeof(address.byte));
  }

  DIRECT_FOUNDRY_HD static DirectSha256Address record_address(
      std::uint64_t sequence, const DirectSha256Address& candidate) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-recipe-pool-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, sequence);
    update_address(&state, candidate);
    return state.finish();
  }

  DIRECT_FOUNDRY_HD static DirectSha256Address next_head(
      const DirectSha256Address& prior,
      const DirectSha256Address& record) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-recipe-pool-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_address(&state, prior);
    update_address(&state, record);
    return state.finish();
  }

  std::array<DirectFoundryCandidateEntryV1, Capacity> entries_{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

#undef DIRECT_FOUNDRY_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_RECIPE_CANDIDATE_POOL_CUH
