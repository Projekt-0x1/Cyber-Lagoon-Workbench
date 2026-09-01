#ifndef HARDWARE_NATIVE_DIRECT_GENOME_LINEAGE_PROVENANCE_CUH
#define HARDWARE_NATIVE_DIRECT_GENOME_LINEAGE_PROVENANCE_CUH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

enum class DirectGenomeLineageKindV1 : std::uint32_t {
  founder = 1u,
  mutation = 2u,
};

struct DirectGenomeLineageRecordV1 {
  std::uint64_t sequence = 0u;
  std::uint32_t generation = 0u;
  DirectGenomeLineageKindV1 kind = DirectGenomeLineageKindV1::founder;
  DirectSha256Address genome{};
  DirectSha256Address parent_genome{};
  DirectSha256Address parent_record{};
  DirectSha256Address mutation{};
  DirectSha256Address experiment{};
  DirectSha256Address receipt{};
};

struct DirectGenomeLineageEntryV1 {
  DirectGenomeLineageRecordV1 record{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

inline bool lineage_address_is_nonzero(const DirectSha256Address& a) {
  std::uint8_t n = 0u; for (std::uint8_t b : a.byte) n |= b; return n != 0u;
}

class DirectGenomeLineageProvenance {
 public:
  bool append_observer_prose_as_mutation(const DirectGenomeLineageRecordV1&, const void*, std::size_t) { return false; }
  bool append_g1_executable_seed_mutation(const DirectGenomeLineageRecordV1& r, const DirectSha256Address& s) { return r.kind == DirectGenomeLineageKindV1::mutation && r.mutation == s && lineage_address_is_nonzero(s) && append(r); } bool append_g2_neonatal_sibling_assay(const DirectGenomeLineageRecordV1& r, const DirectSha256Address& s) { return append_g1_executable_seed_mutation(r, s); }
  bool append(const DirectGenomeLineageRecordV1& record) {
    if (record.sequence != entries_.size() || !valid_record(record) ||
        find_genome(record.genome) != entries_.size())
      return false;

    if (record.kind == DirectGenomeLineageKindV1::founder) {
      if (!entries_.empty() || record.generation != 0u ||
          lineage_address_is_nonzero(record.parent_genome) ||
          lineage_address_is_nonzero(record.parent_record) ||
          lineage_address_is_nonzero(record.mutation))
        return false;
    } else {
      const std::size_t parent = find_record(record.parent_record);
      if (parent == entries_.size() || record.parent_genome != entries_[parent].record.genome ||
          record.generation != entries_[parent].record.generation + 1u ||
          record.genome == record.parent_genome || !lineage_address_is_nonzero(record.mutation))
        return false;
    }

    DirectGenomeLineageEntryV1 entry{};
    entry.record = record;
    entry.prior_head = head_;
    entry.record_address = address_record(record);
    entry.chain_head = next_head(entry.prior_head, entry.record_address);
    entries_.push_back(entry);
    head_ = entry.chain_head;
    return true;
  }

  std::size_t size() const { return entries_.size(); }
  DirectSha256Address head() const { return head_; }

  bool read(std::size_t index, DirectGenomeLineageEntryV1* out) const {
    if (out == nullptr || index >= entries_.size()) return false;
    *out = entries_[index];
    return true;
  }

  bool find(const DirectSha256Address& genome, DirectGenomeLineageEntryV1* out) const {
    const std::size_t index = find_genome(genome);
    return index != entries_.size() && read(index, out);
  }

  bool parent_of(const DirectGenomeLineageEntryV1& child,
                 DirectGenomeLineageEntryV1* out) const {
    if (child.record.kind != DirectGenomeLineageKindV1::mutation || out == nullptr) return false;
    const std::size_t index = find_record(child.record.parent_record);
    return index != entries_.size() && read(index, out);
  }

 private:
  static bool valid_record(const DirectGenomeLineageRecordV1& record) {
    return (record.kind == DirectGenomeLineageKindV1::founder ||
            record.kind == DirectGenomeLineageKindV1::mutation) &&
           lineage_address_is_nonzero(record.genome) &&
           lineage_address_is_nonzero(record.experiment) &&
           lineage_address_is_nonzero(record.receipt);
  }

  std::size_t find_genome(const DirectSha256Address& genome) const {
    for (std::size_t i = 0u; i < entries_.size(); ++i)
      if (entries_[i].record.genome == genome) return i;
    return entries_.size();
  }

  std::size_t find_record(const DirectSha256Address& record) const {
    for (std::size_t i = 0u; i < entries_.size(); ++i)
      if (entries_[i].record_address == record) return i;
    return entries_.size();
  }

  static void update_u32(detail::DirectSha256State* state, std::uint32_t value) {
    const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8u), static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 24u)};
    state->update(bytes, sizeof(bytes));
  }

  static void update_u64(detail::DirectSha256State* state, std::uint64_t value) {
    std::uint8_t bytes[8];
    for (std::uint32_t i = 0u; i < 8u; ++i)
      bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
    state->update(bytes, sizeof(bytes));
  }

  static DirectSha256Address address_record(const DirectGenomeLineageRecordV1& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-lineage-record-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    update_u64(&state, record.sequence);
    update_u32(&state, record.generation);
    update_u32(&state, static_cast<std::uint32_t>(record.kind));
    for (const DirectSha256Address* value : {&record.genome, &record.parent_genome,
                                             &record.parent_record, &record.mutation,
                                             &record.experiment, &record.receipt})
      state.update(value->byte, sizeof(value->byte));
    return state.finish();
  }

  static DirectSha256Address next_head(const DirectSha256Address& prior,
                                      const DirectSha256Address& record) {
    static constexpr char kDomain[] = "0x1-direct-genome-lineage-chain-v1";
    detail::DirectSha256State state{};
    state.update(kDomain, sizeof(kDomain) - 1u);
    state.update(prior.byte, sizeof(prior.byte));
    state.update(record.byte, sizeof(record.byte));
    return state.finish();
  }

  std::vector<DirectGenomeLineageEntryV1> entries_;
  DirectSha256Address head_{};
};

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_GENOME_LINEAGE_PROVENANCE_CUH
