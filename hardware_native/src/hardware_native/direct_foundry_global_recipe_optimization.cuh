#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_GLOBAL_RECIPE_OPTIMIZATION_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_GLOBAL_RECIPE_OPTIMIZATION_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_foundry_population_usage_receipt.cuh"
#include "hardware_native/direct_foundry_recipe_candidate_pool.cuh"

namespace substrate::direct_network {

enum class DirectFoundrySearchOperationV1 : std::uint32_t {
  invent = 1u,
  cross = 2u,
  mine = 3u,
  specialize = 4u,
  simplify = 5u,
  optimize = 6u,
};

// One finite, reproducible observer-side search nomination. search_generation
// is descriptive ancestry: no fixed generation ceiling or recursive meta-level
// is enforced here. The record can nominate possible cognition only.
struct DirectFoundryOptimizationRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint64_t search_generation = 0u;
  std::uint32_t abi_version = 1u;
  DirectFoundrySearchOperationV1 operation =
      DirectFoundrySearchOperationV1::mine;
  std::uint32_t trial_count = 0u;
  std::uint32_t successful_closure_count = 0u;
  std::uint32_t guarded_failure_count = 0u;
  std::uint64_t exact_causal_participation_count = 0u;
  std::uint64_t redundant_occurrence_count = 0u;
  std::uint64_t cooccurrence_count = 0u;
  std::uint64_t latency_p95_ns = 0u;
  std::uint64_t peak_vram_bytes = 0u;
  std::uint64_t energy_p95_nj = 0u;
  DirectSha256Address candidate{};
  DirectSha256Address parent_candidate{};
  DirectSha256Address fallback_candidate{};
  DirectSha256Address construction_ancestry{};
  DirectSha256Address compatible_species{};
  DirectSha256Address usage_receipt{};
  DirectSha256Address search_proposal{};
  DirectSha256Address task{};
  DirectSha256Address guard{};
  DirectSha256Address body_regime{};
  DirectSha256Address evaluator{};
  DirectSha256Address resource_regime{};
  DirectSha256Address trial_history{};
  DirectSha256Address positive_receipts{};
  DirectSha256Address negative_receipts{};
  DirectSha256Address reproducibility{};
};

struct DirectFoundryOptimizationEntryV1 {
  DirectFoundryOptimizationRecordV1 record{};
  DirectSha256Address nomination_address{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

static_assert(std::is_standard_layout_v<DirectFoundryOptimizationRecordV1> &&
              std::is_trivially_copyable_v<DirectFoundryOptimizationRecordV1>);
static_assert(std::is_standard_layout_v<DirectFoundryOptimizationEntryV1> &&
              std::is_trivially_copyable_v<DirectFoundryOptimizationEntryV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_OPT_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_OPT_HD
#endif

template <std::size_t Capacity>
class DirectFoundryGlobalRecipeOptimization {
 public:
  static_assert(Capacity > 0u);

  bool append(
      const DirectFoundryOptimizationRecordV1& record,
      const DirectFoundryCandidateEntryV1& candidate,
      const DirectFoundryCandidateEntryV1& fallback,
      const DirectFoundryPopulationUsageEntryV1& usage) {
    if (size_ == Capacity || record.sequence != size_ ||
        !valid_record(record) || !consistent_sources(record, candidate,
                                                     fallback, usage))
      return false;
    const DirectSha256Address nomination = nomination_address(record);
    for (std::size_t i = 0u; i < size_; ++i)
      if (entries_[i].nomination_address == nomination)
        return false;

    DirectFoundryOptimizationEntryV1 entry{};
    entry.record = record;
    entry.nomination_address = nomination;
    entry.prior_head = head_;
    entry.record_address = record_address(record.sequence, nomination);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return size_; }
  static constexpr std::size_t capacity() { return Capacity; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectFoundryOptimizationEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  bool verify() const {
    return verify_entries(entries_.data(), size_, head_);
  }

  static bool verify_entries(const DirectFoundryOptimizationEntryV1* entries,
                             std::size_t count,
                             const DirectSha256Address& expected_head) {
    if ((count != 0u && entries == nullptr) || count > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t i = 0u; i < count; ++i) {
      const auto& entry = entries[i];
      if (entry.record.sequence != i || !valid_record(entry.record) ||
          entry.nomination_address != nomination_address(entry.record) ||
          entry.prior_head != prior ||
          entry.record_address != record_address(i, entry.nomination_address) ||
          entry.chain_head != next_head(prior, entry.record_address))
        return false;
      for (std::size_t j = 0u; j < i; ++j)
        if (entries[j].nomination_address == entry.nomination_address)
          return false;
      prior = entry.chain_head;
    }
    return prior == expected_head;
  }

  // A Pareto view is always guard/task/body/evaluator/resource-specific. It
  // neither mutates history nor returns a universal winner. If capacity is too
  // small, nothing is written.
  bool pareto_front(
      const DirectSha256Address& task, const DirectSha256Address& guard,
      const DirectSha256Address& body_regime,
      const DirectSha256Address& evaluator,
      const DirectSha256Address& resource_regime,
      DirectFoundryOptimizationEntryV1* out, std::size_t out_capacity,
      std::size_t* out_count) const {
    if (out_count == nullptr ||
        (out_capacity != 0u && out == nullptr) ||
        !direct_foundry_usage_address_nonzero(task) ||
        !direct_foundry_usage_address_nonzero(guard) ||
        !direct_foundry_usage_address_nonzero(body_regime) ||
        !direct_foundry_usage_address_nonzero(evaluator) ||
        !direct_foundry_usage_address_nonzero(resource_regime))
      return false;

    std::size_t selected = 0u;
    for (std::size_t i = 0u; i < size_; ++i)
      if (matches_context(entries_[i].record, task, guard, body_regime,
                          evaluator, resource_regime) &&
          !is_dominated(i))
        ++selected;
    if (selected > out_capacity)
      return false;

    std::size_t cursor = 0u;
    for (std::size_t i = 0u; i < size_; ++i)
      if (matches_context(entries_[i].record, task, guard, body_regime,
                          evaluator, resource_regime) &&
          !is_dominated(i))
        out[cursor++] = entries_[i];
    *out_count = selected;
    return true;
  }

  DIRECT_FOUNDRY_OPT_HD static DirectSha256Address nomination_address(
      const DirectFoundryOptimizationRecordV1& record) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-global-optimization-nomination-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, record.search_generation);
    update_u32(&state, record.abi_version);
    update_u32(&state, static_cast<std::uint32_t>(record.operation));
    update_u32(&state, record.trial_count);
    update_u32(&state, record.successful_closure_count);
    update_u32(&state, record.guarded_failure_count);
    update_u64(&state, record.exact_causal_participation_count);
    update_u64(&state, record.redundant_occurrence_count);
    update_u64(&state, record.cooccurrence_count);
    update_u64(&state, record.latency_p95_ns);
    update_u64(&state, record.peak_vram_bytes);
    update_u64(&state, record.energy_p95_nj);
    update_address(&state, record.candidate);
    update_address(&state, record.parent_candidate);
    update_address(&state, record.fallback_candidate);
    update_address(&state, record.construction_ancestry);
    update_address(&state, record.compatible_species);
    update_address(&state, record.usage_receipt);
    update_address(&state, record.search_proposal);
    update_address(&state, record.task);
    update_address(&state, record.guard);
    update_address(&state, record.body_regime);
    update_address(&state, record.evaluator);
    update_address(&state, record.resource_regime);
    update_address(&state, record.trial_history);
    update_address(&state, record.positive_receipts);
    update_address(&state, record.negative_receipts);
    update_address(&state, record.reproducibility);
    return state.finish();
  }

 private:
  using CandidateAddressing = DirectFoundryRecipeCandidatePool<1u>;
  using UsageAddressing = DirectFoundryPopulationUsageArchive<1u>;

  static bool known_operation(DirectFoundrySearchOperationV1 operation) {
    return operation == DirectFoundrySearchOperationV1::invent ||
           operation == DirectFoundrySearchOperationV1::cross ||
           operation == DirectFoundrySearchOperationV1::mine ||
           operation == DirectFoundrySearchOperationV1::specialize ||
           operation == DirectFoundrySearchOperationV1::simplify ||
           operation == DirectFoundrySearchOperationV1::optimize;
  }

  static bool valid_record(const DirectFoundryOptimizationRecordV1& record) {
    if (record.abi_version != 1u || !known_operation(record.operation) ||
        record.trial_count == 0u ||
        record.successful_closure_count > record.trial_count ||
        record.guarded_failure_count > record.trial_count ||
        record.exact_causal_participation_count > 0xffffffffu ||
        record.redundant_occurrence_count > 0xffffffffu ||
        record.latency_p95_ns == 0u || record.peak_vram_bytes == 0u ||
        record.energy_p95_nj == 0u)
      return false;
    const DirectSha256Address* roots[] = {
        &record.candidate,
        &record.fallback_candidate,
        &record.construction_ancestry,
        &record.compatible_species,
        &record.usage_receipt,
        &record.search_proposal,
        &record.task,
        &record.guard,
        &record.body_regime,
        &record.evaluator,
        &record.resource_regime,
        &record.trial_history,
        &record.positive_receipts,
        &record.negative_receipts,
        &record.reproducibility};
    for (const DirectSha256Address* root : roots)
      if (!direct_foundry_usage_address_nonzero(*root))
        return false;
    return true;
  }

  static bool consistent_sources(
      const DirectFoundryOptimizationRecordV1& record,
      const DirectFoundryCandidateEntryV1& candidate,
      const DirectFoundryCandidateEntryV1& fallback,
      const DirectFoundryPopulationUsageEntryV1& usage) {
    if (candidate.candidate_address !=
            CandidateAddressing::candidate_address(candidate.record) ||
        fallback.candidate_address !=
            CandidateAddressing::candidate_address(fallback.record) ||
        usage.receipt_address !=
            UsageAddressing::receipt_address(usage.receipt))
      return false;
    const auto& observed = usage.receipt;
    return record.candidate == candidate.candidate_address &&
           record.parent_candidate == candidate.record.parent_candidate &&
           record.fallback_candidate == fallback.candidate_address &&
           record.construction_ancestry ==
               candidate.record.construction_ancestry &&
           record.compatible_species == candidate.record.compatible_species &&
           record.usage_receipt == usage.receipt_address &&
           record.task == observed.task && record.guard == observed.guard &&
           record.guard == candidate.record.guard &&
           record.body_regime == observed.body_regime &&
           record.evaluator == observed.evaluator &&
           record.evaluator == candidate.record.compatible_evaluator &&
           record.resource_regime == observed.resource_regime &&
           record.trial_history == observed.trial_history &&
           record.positive_receipts == observed.positive_receipts &&
           record.negative_receipts == observed.negative_receipts &&
           record.trial_count == observed.trial_count &&
           record.successful_closure_count ==
               observed.successful_closure_count &&
           record.guarded_failure_count == observed.guarded_failure_count &&
           record.exact_causal_participation_count ==
               observed.exact_causal_participation_count &&
           record.redundant_occurrence_count ==
               observed.redundant_occurrence_count &&
           record.cooccurrence_count == observed.cooccurrence_count &&
           record.latency_p95_ns == observed.latency_p95_ns &&
           record.peak_vram_bytes == observed.peak_vram_bytes &&
           record.energy_p95_nj == observed.energy_p95_nj;
  }

  static bool matches_context(
      const DirectFoundryOptimizationRecordV1& record,
      const DirectSha256Address& task, const DirectSha256Address& guard,
      const DirectSha256Address& body_regime,
      const DirectSha256Address& evaluator,
      const DirectSha256Address& resource_regime) {
    return record.task == task && record.guard == guard &&
           record.body_regime == body_regime &&
           record.evaluator == evaluator &&
           record.resource_regime == resource_regime;
  }

  static bool ratio_at_least(std::uint64_t left_numerator,
                             std::uint32_t left_denominator,
                             std::uint64_t right_numerator,
                             std::uint32_t right_denominator) {
    return left_numerator * right_denominator >=
           right_numerator * left_denominator;
  }

  static bool ratio_at_most(std::uint64_t left_numerator,
                            std::uint32_t left_denominator,
                            std::uint64_t right_numerator,
                            std::uint32_t right_denominator) {
    return left_numerator * right_denominator <=
           right_numerator * left_denominator;
  }

  static bool ratio_equal(std::uint64_t left_numerator,
                          std::uint32_t left_denominator,
                          std::uint64_t right_numerator,
                          std::uint32_t right_denominator) {
    return left_numerator * right_denominator ==
           right_numerator * left_denominator;
  }

  static bool dominates(const DirectFoundryOptimizationRecordV1& left,
                        const DirectFoundryOptimizationRecordV1& right) {
    if (!matches_context(left, right.task, right.guard, right.body_regime,
                         right.evaluator, right.resource_regime))
      return false;
    const bool success = ratio_at_least(
        left.successful_closure_count, left.trial_count,
        right.successful_closure_count, right.trial_count);
    const bool participation = ratio_at_least(
        left.exact_causal_participation_count, left.trial_count,
        right.exact_causal_participation_count, right.trial_count);
    const bool failures = ratio_at_most(
        left.guarded_failure_count, left.trial_count,
        right.guarded_failure_count, right.trial_count);
    const bool redundancy = ratio_at_most(
        left.redundant_occurrence_count, left.trial_count,
        right.redundant_occurrence_count, right.trial_count);
    const bool latency = left.latency_p95_ns <= right.latency_p95_ns;
    const bool vram = left.peak_vram_bytes <= right.peak_vram_bytes;
    const bool energy = left.energy_p95_nj <= right.energy_p95_nj;
    if (!(success && participation && failures && redundancy && latency &&
          vram && energy))
      return false;
    return !ratio_equal(left.successful_closure_count, left.trial_count,
                        right.successful_closure_count, right.trial_count) ||
           !ratio_equal(left.exact_causal_participation_count,
                        left.trial_count,
                        right.exact_causal_participation_count,
                        right.trial_count) ||
           !ratio_equal(left.guarded_failure_count, left.trial_count,
                        right.guarded_failure_count, right.trial_count) ||
           !ratio_equal(left.redundant_occurrence_count, left.trial_count,
                        right.redundant_occurrence_count,
                        right.trial_count) ||
           left.latency_p95_ns != right.latency_p95_ns ||
           left.peak_vram_bytes != right.peak_vram_bytes ||
           left.energy_p95_nj != right.energy_p95_nj;
  }

  bool is_dominated(std::size_t candidate) const {
    for (std::size_t i = 0u; i < size_; ++i)
      if (i != candidate &&
          dominates(entries_[i].record, entries_[candidate].record))
        return true;
    return false;
  }

  DIRECT_FOUNDRY_OPT_HD static void update_u32(
      detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_OPT_HD static void update_u64(
      detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  DIRECT_FOUNDRY_OPT_HD static void update_address(
      detail::DirectSha256State* state, const DirectSha256Address& address) {
    state->update(address.byte, sizeof(address.byte));
  }

  DIRECT_FOUNDRY_OPT_HD static DirectSha256Address record_address(
      std::uint64_t sequence, const DirectSha256Address& nomination) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-global-optimization-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, sequence);
    update_address(&state, nomination);
    return state.finish();
  }

  DIRECT_FOUNDRY_OPT_HD static DirectSha256Address next_head(
      const DirectSha256Address& prior,
      const DirectSha256Address& record) {
    static constexpr char kDomain[] =
        "0x1-direct-foundry-global-optimization-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_address(&state, prior);
    update_address(&state, record);
    return state.finish();
  }

  std::array<DirectFoundryOptimizationEntryV1, Capacity> entries_{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

#undef DIRECT_FOUNDRY_OPT_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_GLOBAL_RECIPE_OPTIMIZATION_CUH
