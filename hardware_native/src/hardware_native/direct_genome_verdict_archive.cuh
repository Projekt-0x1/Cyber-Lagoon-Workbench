#ifndef HARDWARE_NATIVE_DIRECT_GENOME_VERDICT_ARCHIVE_CUH
#define HARDWARE_NATIVE_DIRECT_GENOME_VERDICT_ARCHIVE_CUH

#include <array>
#include <cstddef>
#include <cstdint>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

enum class DirectGenomeFailureKindV1 : std::uint32_t {
  failed_mutation = 1u,
  lethal_trajectory = 2u,
};

// Observer-side evidence only. These opaque roots do not confer runtime,
// learning, promotion, or inheritance authority on the archived verdict.
struct DirectGenomeVerdictRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint32_t abi_version = 1u;
  DirectGenomeFailureKindV1 kind = DirectGenomeFailureKindV1::failed_mutation;
  DirectSha256Address candidate_genome{};
  DirectSha256Address mutation{};
  DirectSha256Address development_context{};
  DirectSha256Address birth_receipt{};
  DirectSha256Address experiment{};
  DirectSha256Address evidence_receipt{};
};

struct DirectGenomeVerdictEntryV1 {
  DirectGenomeVerdictRecordV1 record{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

inline bool direct_genome_verdict_matches_context(
    const DirectGenomeVerdictRecordV1& archived,
    const DirectSha256Address& candidate_genome,
    const DirectSha256Address& mutation,
    const DirectSha256Address& development_context,
    const DirectSha256Address& experiment) {
  return archived.candidate_genome == candidate_genome &&
         archived.mutation == mutation &&
         archived.development_context == development_context &&
         archived.experiment == experiment;
}

inline bool direct_verdict_address_is_nonzero(const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte)
    any |= byte;
  return any != 0u;
}

template <std::size_t Capacity>
class DirectGenomeGoodArchive;

template <std::size_t Capacity>
class DirectGenomeVerdictArchive {
 public:
  static_assert(Capacity > 0u);

  bool append(const DirectGenomeVerdictRecordV1& record) {
    if (size_ == Capacity || record.sequence != size_ || !valid_record(record))
      return false;
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].record.candidate_genome == record.candidate_genome)
        return false;

    DirectGenomeVerdictEntryV1 entry{};
    entry.record = record;
    entry.prior_head = head_;
    entry.record_address = address_record(record);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  template <std::size_t GoodCapacity>
  bool append_complementary(const DirectGenomeVerdictRecordV1& record,
                            const DirectGenomeGoodArchive<GoodCapacity>& goods);

  std::size_t size() const { return size_; }
  static constexpr std::size_t capacity() { return Capacity; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectGenomeVerdictEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  bool verify() const { return verify_entries(entries_.data(), size_, head_); }

  bool contains_trial(const DirectSha256Address& trial) const {
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].record.evidence_receipt == trial)
        return true;
    return false;
  }

  static bool verify_entries(const DirectGenomeVerdictEntryV1* entries, std::size_t count,
                             const DirectSha256Address& expected_head) {
    if ((count != 0u && entries == nullptr) || count > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t i = 0u; i < count; ++i) {
      const auto& entry = entries[i];
      if (entry.record.sequence != i || !valid_record(entry.record) || entry.prior_head != prior ||
          entry.record_address != address_record(entry.record) ||
          entry.chain_head != next_head(prior, entry.record_address))
        return false;
      for (std::size_t j = 0u; j < i; ++j)
        if (entries[j].record.candidate_genome == entry.record.candidate_genome)
          return false;
      prior = entry.chain_head;
    }
    return prior == expected_head;
  }

 private:
  static bool valid_record(const DirectGenomeVerdictRecordV1& record) {
    const bool known_kind = record.kind == DirectGenomeFailureKindV1::failed_mutation ||
                            record.kind == DirectGenomeFailureKindV1::lethal_trajectory;
    return record.abi_version == 1u && known_kind &&
           direct_verdict_address_is_nonzero(record.candidate_genome) &&
           direct_verdict_address_is_nonzero(record.mutation) &&
           direct_verdict_address_is_nonzero(record.development_context) &&
           direct_verdict_address_is_nonzero(record.birth_receipt) &&
           direct_verdict_address_is_nonzero(record.experiment) &&
           direct_verdict_address_is_nonzero(record.evidence_receipt);
  }

  static void update_u32(detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u), static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  static void update_u64(detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  static DirectSha256Address address_record(const DirectGenomeVerdictRecordV1& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-verdict-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, record.sequence);
    update_u32(&state, record.abi_version);
    update_u32(&state, static_cast<std::uint32_t>(record.kind));
    for (const DirectSha256Address* value :
         {&record.candidate_genome, &record.mutation, &record.development_context,
          &record.birth_receipt, &record.experiment, &record.evidence_receipt})
      state.update(value->byte, sizeof(value->byte));
    return state.finish();
  }

  static DirectSha256Address next_head(const DirectSha256Address& prior,
                                       const DirectSha256Address& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-verdict-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    state.update(prior.byte, sizeof(prior.byte));
    state.update(record.byte, sizeof(record.byte));
    return state.finish();
  }

  std::array<DirectGenomeVerdictEntryV1, Capacity> entries_{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

struct DirectGenomeGoodRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint32_t abi_version = 1u;
  DirectSha256Address candidate_genome{};
  DirectSha256Address trial_history{};
  DirectSha256Address first_birth_receipt{};
  DirectSha256Address second_birth_receipt{};
  DirectSha256Address canalization_signature{};
  DirectSha256Address canalization_guard{};
  DirectSha256Address milestone_receipt{};
};

struct DirectGenomeGoodEntryV1 {
  DirectGenomeGoodRecordV1 record{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

inline bool direct_good_record_matches_replay(const DirectGenomeGoodRecordV1& archived,
                                              const DirectSha256Address& candidate_genome,
                                              const DirectSha256Address& canalization_signature,
                                              const DirectSha256Address& canalization_guard) {
  return archived.candidate_genome == candidate_genome &&
         archived.canalization_signature == canalization_signature &&
         archived.canalization_guard == canalization_guard;
}

// Finite observer archive. Admission requires completed birth, canalization,
// milestone, and trial-history receipts and consults the sibling failure ledger.
// It supplies evidence storage only, never promotion or inheritance authority.
template <std::size_t Capacity>
class DirectGenomeGoodArchive {
 public:
  static_assert(Capacity > 0u);

  template <std::size_t BadCapacity>
  bool append(const DirectGenomeGoodRecordV1& record,
              const DirectGenomeVerdictArchive<BadCapacity>& bads) {
    if (size_ == Capacity || record.sequence != size_ || !valid_record(record) ||
        bads.contains_trial(record.trial_history))
      return false;
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].record.candidate_genome == record.candidate_genome)
        return false;

    DirectGenomeGoodEntryV1 entry{};
    entry.record = record;
    entry.prior_head = head_;
    entry.record_address = address_record(record);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return size_; }
  static constexpr std::size_t capacity() { return Capacity; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectGenomeGoodEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  bool verify() const { return verify_entries(entries_.data(), size_, head_); }

  bool contains_trial(const DirectSha256Address& trial) const {
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].record.trial_history == trial)
        return true;
    return false;
  }

  static bool verify_entries(const DirectGenomeGoodEntryV1* entries, std::size_t count,
                             const DirectSha256Address& expected_head) {
    if ((count != 0u && entries == nullptr) || count > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t i = 0u; i < count; ++i) {
      const auto& entry = entries[i];
      if (entry.record.sequence != i || !valid_record(entry.record) || entry.prior_head != prior ||
          entry.record_address != address_record(entry.record) ||
          entry.chain_head != next_head(prior, entry.record_address))
        return false;
      for (std::size_t j = 0u; j < i; ++j)
        if (entries[j].record.candidate_genome == entry.record.candidate_genome)
          return false;
      prior = entry.chain_head;
    }
    return prior == expected_head;
  }

 private:
  static bool valid_record(const DirectGenomeGoodRecordV1& record) {
    return record.abi_version == 1u && direct_verdict_address_is_nonzero(record.candidate_genome) &&
           direct_verdict_address_is_nonzero(record.trial_history) &&
           direct_verdict_address_is_nonzero(record.first_birth_receipt) &&
           direct_verdict_address_is_nonzero(record.second_birth_receipt) &&
           record.first_birth_receipt != record.second_birth_receipt &&
           direct_verdict_address_is_nonzero(record.canalization_signature) &&
           direct_verdict_address_is_nonzero(record.canalization_guard) &&
           direct_verdict_address_is_nonzero(record.milestone_receipt);
  }

  static void update_u32(detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u), static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  static void update_u64(detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  static DirectSha256Address address_record(const DirectGenomeGoodRecordV1& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-good-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, record.sequence);
    update_u32(&state, record.abi_version);
    for (const DirectSha256Address* value :
         {&record.candidate_genome, &record.trial_history, &record.first_birth_receipt,
          &record.second_birth_receipt, &record.canalization_signature, &record.canalization_guard,
          &record.milestone_receipt})
      state.update(value->byte, sizeof(value->byte));
    return state.finish();
  }

  static DirectSha256Address next_head(const DirectSha256Address& prior,
                                       const DirectSha256Address& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-good-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    state.update(prior.byte, sizeof(prior.byte));
    state.update(record.byte, sizeof(record.byte));
    return state.finish();
  }

  std::array<DirectGenomeGoodEntryV1, Capacity> entries_{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

template <std::size_t Capacity>
template <std::size_t GoodCapacity>
bool DirectGenomeVerdictArchive<Capacity>::append_complementary(
    const DirectGenomeVerdictRecordV1& record, const DirectGenomeGoodArchive<GoodCapacity>& goods) {
  return !goods.contains_trial(record.evidence_receipt) && append(record);
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_GENOME_VERDICT_ARCHIVE_CUH
