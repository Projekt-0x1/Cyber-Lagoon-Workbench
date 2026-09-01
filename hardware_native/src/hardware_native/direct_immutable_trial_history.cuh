#ifndef HARDWARE_NATIVE_DIRECT_IMMUTABLE_TRIAL_HISTORY_CUH
#define HARDWARE_NATIVE_DIRECT_IMMUTABLE_TRIAL_HISTORY_CUH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

enum class DirectTrialRecordKindV1 : std::uint32_t {
  evolution_trial = 1u,
  adult_evaluation = 2u,
};

struct DirectTrialRecordV1 {
  std::uint64_t sequence = 0u;
  DirectSha256Address candidate_genome{};
  DirectSha256Address context{};
  DirectSha256Address adult_state{};
  DirectSha256Address evidence{};
  std::uint32_t abi_version = 1u;
  DirectTrialRecordKindV1 kind = DirectTrialRecordKindV1::evolution_trial;
};

struct DirectImmutableTrialEntryV1 {
  DirectTrialRecordV1 record{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

class DirectImmutableTrialHistory {
 public:
  bool append(const DirectTrialRecordV1& record) {
    if (!valid_record(record) || record.sequence != entries_.size()) return false;

    DirectImmutableTrialEntryV1 entry{};
    entry.record = record;
    entry.prior_head = head_;
    entry.record_address = record_address(record);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_.push_back(entry);
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return entries_.size(); }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectImmutableTrialEntryV1* out) const {
    if (out == nullptr || index >= entries_.size()) return false;
    *out = entries_[index];
    return true;
  }

 private:
  static bool valid_record(const DirectTrialRecordV1& record) {
    return record.abi_version == 1u &&
           (record.kind == DirectTrialRecordKindV1::evolution_trial ||
            record.kind == DirectTrialRecordKindV1::adult_evaluation);
  }

  static void update_u32_le(detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  static void update_u64_le(detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  static DirectSha256Address record_address(const DirectTrialRecordV1& record) {
    static constexpr char kDomain[] = "0x1-direct-trial-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64_le(&state, record.sequence);
    state.update(record.candidate_genome.byte, sizeof(record.candidate_genome.byte));
    state.update(record.context.byte, sizeof(record.context.byte));
    state.update(record.adult_state.byte, sizeof(record.adult_state.byte));
    state.update(record.evidence.byte, sizeof(record.evidence.byte));
    update_u32_le(&state, record.abi_version);
    update_u32_le(&state, static_cast<std::uint32_t>(record.kind));
    return state.finish();
  }

  static DirectSha256Address next_head(const DirectSha256Address& prior,
                                      const DirectSha256Address& record) {
    static constexpr char kDomain[] = "0x1-direct-trial-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    state.update(prior.byte, sizeof(prior.byte));
    state.update(record.byte, sizeof(record.byte));
    return state.finish();
  }

  std::vector<DirectImmutableTrialEntryV1> entries_;
  DirectSha256Address head_{};
};

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_IMMUTABLE_TRIAL_HISTORY_CUH
