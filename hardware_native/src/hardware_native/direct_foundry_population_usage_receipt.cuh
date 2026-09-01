#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_POPULATION_USAGE_RECEIPT_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_POPULATION_USAGE_RECEIPT_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

// Privacy-bounded population aggregate. It intentionally contains no Adult,
// occurrence, account, source-text, or semantic identifier. Distribution roots
// address opaque aggregate histograms; exact private histories remain outside.
struct DirectFoundryPopulationUsageReceiptV1 {
  std::uint64_t sequence = 0u;
  std::uint32_t abi_version = 1u;
  std::uint32_t cohort_size = 0u;
  std::uint32_t trial_count = 0u;
  std::uint32_t max_individual_trial_steps = 0u;
  std::uint32_t successful_closure_count = 0u;
  std::uint32_t guarded_failure_count = 0u;
  std::uint64_t exact_causal_participation_count = 0u;
  std::uint64_t redundant_occurrence_count = 0u;
  std::uint64_t cooccurrence_count = 0u;
  std::uint64_t latency_p95_ns = 0u;
  std::uint64_t peak_vram_bytes = 0u;
  std::uint64_t energy_p95_nj = 0u;
  DirectSha256Address candidate{};
  DirectSha256Address cohort{};
  DirectSha256Address task{};
  DirectSha256Address guard{};
  DirectSha256Address body_regime{};
  DirectSha256Address evaluator{};
  DirectSha256Address resource_regime{};
  DirectSha256Address trial_history{};
  DirectSha256Address positive_receipts{};
  DirectSha256Address negative_receipts{};
  DirectSha256Address successful_closure_distribution{};
  DirectSha256Address causal_participation_distribution{};
  DirectSha256Address latency_distribution{};
  DirectSha256Address vram_distribution{};
  DirectSha256Address redundancy_distribution{};
  DirectSha256Address cooccurrence_distribution{};
  DirectSha256Address guarded_failure_distribution{};
  DirectSha256Address resource_profiles{};
};

struct DirectFoundryPopulationUsageEntryV1 {
  DirectFoundryPopulationUsageReceiptV1 receipt{};
  DirectSha256Address receipt_address{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

static_assert(std::is_standard_layout_v<DirectFoundryPopulationUsageReceiptV1> &&
              std::is_trivially_copyable_v<DirectFoundryPopulationUsageReceiptV1>);
static_assert(std::is_standard_layout_v<DirectFoundryPopulationUsageEntryV1> &&
              std::is_trivially_copyable_v<DirectFoundryPopulationUsageEntryV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_USAGE_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_USAGE_HD
#endif

DIRECT_FOUNDRY_USAGE_HD inline bool direct_foundry_usage_address_nonzero(
    const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte)
    any |= byte;
  return any != 0u;
}

template <std::size_t Capacity>
class DirectFoundryPopulationUsageArchive {
 public:
  static_assert(Capacity > 0u);
  static constexpr std::uint32_t kMinimumPrivacyCohort = 2u;

  bool append(const DirectFoundryPopulationUsageReceiptV1& receipt) {
    if (size_ == Capacity || receipt.sequence != size_ ||
        !valid_receipt(receipt))
      return false;
    const DirectSha256Address content = receipt_address(receipt);
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].receipt_address == content)
        return false;

    DirectFoundryPopulationUsageEntryV1 entry{};
    entry.receipt = receipt;
    entry.receipt_address = content;
    entry.prior_head = head_;
    entry.record_address = record_address(receipt.sequence, content);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return size_; }
  static constexpr std::size_t capacity() { return Capacity; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index,
            DirectFoundryPopulationUsageEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  bool verify() const {
    return verify_entries(entries_.data(), size_, head_);
  }

  static bool verify_entries(
      const DirectFoundryPopulationUsageEntryV1* entries, std::size_t count,
      const DirectSha256Address& expected_head) {
    if ((count != 0u && entries == nullptr) || count > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t i = 0u; i < count; ++i) {
      const auto& entry = entries[i];
      if (entry.receipt.sequence != i || !valid_receipt(entry.receipt) ||
          entry.receipt_address != receipt_address(entry.receipt) ||
          entry.prior_head != prior ||
          entry.record_address != record_address(i, entry.receipt_address) ||
          entry.chain_head != next_head(prior, entry.record_address))
        return false;
      for (std::size_t j = 0u; j < i; ++j)
        if (entries[j].receipt_address == entry.receipt_address)
          return false;
      prior = entry.chain_head;
    }
    return prior == expected_head;
  }

  // Sequence is archive chronology, so population receipt identity is stable
  // when the same aggregate is imported into another append-only archive.
  DIRECT_FOUNDRY_USAGE_HD static DirectSha256Address receipt_address(
      const DirectFoundryPopulationUsageReceiptV1& receipt) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-population-usage-receipt-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u32(&state, receipt.abi_version);
    update_u32(&state, receipt.cohort_size);
    update_u32(&state, receipt.trial_count);
    update_u32(&state, receipt.max_individual_trial_steps);
    update_u32(&state, receipt.successful_closure_count);
    update_u32(&state, receipt.guarded_failure_count);
    update_u64(&state, receipt.exact_causal_participation_count);
    update_u64(&state, receipt.redundant_occurrence_count);
    update_u64(&state, receipt.cooccurrence_count);
    update_u64(&state, receipt.latency_p95_ns);
    update_u64(&state, receipt.peak_vram_bytes);
    update_u64(&state, receipt.energy_p95_nj);
    update_address(&state, receipt.candidate);
    update_address(&state, receipt.cohort);
    update_address(&state, receipt.task);
    update_address(&state, receipt.guard);
    update_address(&state, receipt.body_regime);
    update_address(&state, receipt.evaluator);
    update_address(&state, receipt.resource_regime);
    update_address(&state, receipt.trial_history);
    update_address(&state, receipt.positive_receipts);
    update_address(&state, receipt.negative_receipts);
    update_address(&state, receipt.successful_closure_distribution);
    update_address(&state, receipt.causal_participation_distribution);
    update_address(&state, receipt.latency_distribution);
    update_address(&state, receipt.vram_distribution);
    update_address(&state, receipt.redundancy_distribution);
    update_address(&state, receipt.cooccurrence_distribution);
    update_address(&state, receipt.guarded_failure_distribution);
    update_address(&state, receipt.resource_profiles);
    return state.finish();
  }

 private:
  static bool valid_receipt(
      const DirectFoundryPopulationUsageReceiptV1& receipt) {
    if (receipt.abi_version != 1u ||
        receipt.cohort_size < kMinimumPrivacyCohort ||
        receipt.trial_count < receipt.cohort_size ||
        receipt.max_individual_trial_steps == 0u ||
        receipt.successful_closure_count > receipt.trial_count ||
        receipt.guarded_failure_count > receipt.trial_count ||
        receipt.exact_causal_participation_count > 0xffffffffu ||
        receipt.redundant_occurrence_count > 0xffffffffu ||
        receipt.latency_p95_ns == 0u || receipt.peak_vram_bytes == 0u ||
        receipt.energy_p95_nj == 0u)
      return false;
    const DirectSha256Address* roots[] = {
        &receipt.candidate,
        &receipt.cohort,
        &receipt.task,
        &receipt.guard,
        &receipt.body_regime,
        &receipt.evaluator,
        &receipt.resource_regime,
        &receipt.trial_history,
        &receipt.positive_receipts,
        &receipt.negative_receipts,
        &receipt.successful_closure_distribution,
        &receipt.causal_participation_distribution,
        &receipt.latency_distribution,
        &receipt.vram_distribution,
        &receipt.redundancy_distribution,
        &receipt.cooccurrence_distribution,
        &receipt.guarded_failure_distribution,
        &receipt.resource_profiles};
    for (const DirectSha256Address* root : roots)
      if (!direct_foundry_usage_address_nonzero(*root))
        return false;
    return true;
  }

  DIRECT_FOUNDRY_USAGE_HD static void update_u32(
      detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_USAGE_HD static void update_u64(
      detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_USAGE_HD static void update_address(
      detail::DirectSha256State* state, const DirectSha256Address& address) {
    state->update(address.byte, sizeof(address.byte));
  }

  DIRECT_FOUNDRY_USAGE_HD static DirectSha256Address record_address(
      std::uint64_t sequence, const DirectSha256Address& receipt) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-population-usage-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, sequence);
    update_address(&state, receipt);
    return state.finish();
  }

  DIRECT_FOUNDRY_USAGE_HD static DirectSha256Address next_head(
      const DirectSha256Address& prior,
      const DirectSha256Address& record) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-population-usage-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_address(&state, prior);
    update_address(&state, record);
    return state.finish();
  }

  std::array<DirectFoundryPopulationUsageEntryV1, Capacity> entries_{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

#undef DIRECT_FOUNDRY_USAGE_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_POPULATION_USAGE_RECEIPT_CUH
