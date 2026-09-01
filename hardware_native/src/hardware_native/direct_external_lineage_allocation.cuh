#ifndef HARDWARE_NATIVE_DIRECT_EXTERNAL_LINEAGE_ALLOCATION_CUH
#define HARDWARE_NATIVE_DIRECT_EXTERNAL_LINEAGE_ALLOCATION_CUH

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

// Observer-side foundry accounting only. The schema deliberately contains no
// genome mutation, resident state, sensory, reward, eligibility, or credit field.
struct DirectExternalLineageAllocationRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint64_t work_units = 0u;
  DirectSha256Address lineage_record{};
  DirectSha256Address receipt{};
  DirectSha256Address sponsor{};
  DirectSha256Address terms{};
};

struct DirectExternalLineageAllocationEntryV1 {
  DirectExternalLineageAllocationRecordV1 record{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

inline bool allocation_address_is_nonzero(const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte) any |= byte;
  return any != 0u;
}

class DirectExternalLineageAllocation {
 public:
  bool append(const DirectExternalLineageAllocationRecordV1& record) {
    if (!valid(record) || record.sequence != entries_.size() ||
        record.work_units > std::numeric_limits<std::uint64_t>::max() - total_work_units_)
      return false;
    for (const DirectExternalLineageAllocationEntryV1& entry : entries_)
      if (entry.record.lineage_record == record.lineage_record &&
          entry.record.sponsor == record.sponsor && entry.record.terms == record.terms)
        return false;

    DirectExternalLineageAllocationEntryV1 entry{};
    entry.record = record;
    entry.prior_head = head_;
    entry.record_address = address_record(record);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_.push_back(entry);
    total_work_units_ += record.work_units;
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return entries_.size(); }
  std::uint64_t total_work_units() const { return total_work_units_; }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectExternalLineageAllocationEntryV1* out) const {
    if (out == nullptr || index >= entries_.size()) return false;
    *out = entries_[index];
    return true;
  }

 private:
  static bool valid(const DirectExternalLineageAllocationRecordV1& record) {
    return record.work_units != 0u && allocation_address_is_nonzero(record.lineage_record) &&
           allocation_address_is_nonzero(record.receipt) &&
           allocation_address_is_nonzero(record.sponsor) &&
           allocation_address_is_nonzero(record.terms);
  }

  static void update_u64(detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  static DirectSha256Address address_record(
      const DirectExternalLineageAllocationRecordV1& record) {
    static constexpr char kDomain[] = "0x1-direct-external-lineage-allocation-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, record.sequence);
    update_u64(&state, record.work_units);
    for (const DirectSha256Address* value :
         {&record.lineage_record, &record.receipt, &record.sponsor, &record.terms})
      state.update(value->byte, sizeof(value->byte));
    return state.finish();
  }

  static DirectSha256Address next_head(const DirectSha256Address& prior,
                                      const DirectSha256Address& record) {
    static constexpr char kDomain[] = "0x1-direct-external-allocation-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    state.update(prior.byte, sizeof(prior.byte));
    state.update(record.byte, sizeof(record.byte));
    return state.finish();
  }

  std::vector<DirectExternalLineageAllocationEntryV1> entries_;
  std::uint64_t total_work_units_ = 0u;
  DirectSha256Address head_{};
};

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_EXTERNAL_LINEAGE_ALLOCATION_CUH
