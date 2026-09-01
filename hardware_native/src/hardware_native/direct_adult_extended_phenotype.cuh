#ifndef HARDWARE_NATIVE_DIRECT_ADULT_EXTENDED_PHENOTYPE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_EXTENDED_PHENOTYPE_CUH

// i.extended_phenotype_actions (#1617). Actions may alter state that lives
// outside the body membrane: an environment surface the adult does not own.
// A cell accepts a mutation only from a settled verified intervention keyed
// to its ticket identity; once written, the alteration is verified against
// that external surface itself and persists independently of the actor's
// local evidence tables.

#include <cstdint>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kEnvironmentCellCapacity = 8u;

struct DirectEnvironmentCell {
  std::uint64_t altering_ticket;
  std::uint32_t value;
  std::uint32_t revision;
};

struct DirectExtendedPhenotypeLedger {
  DirectEnvironmentCell cells[kEnvironmentCellCapacity] = {};
  std::int32_t occupied_mask;
  std::uint32_t refused_unverified;
  std::uint64_t ledger_identity;
};

static_assert(std::is_trivially_copyable_v<DirectExtendedPhenotypeLedger>);

__host__ __device__ inline std::uint64_t phenotype_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

// Attempt one environment mutation from one intervention. The claim counts
// only when the ticket settled with an exact verified world return; anything
// else is a counted refusal and the cell keeps its prior bytes.
__device__ inline bool record_environment_mutation(
    DirectExtendedPhenotypeLedger* ledger, std::uint32_t cell_index,
    std::uint32_t new_value,
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, const DirectExactHistoryRecord* records,
    std::uint32_t record_count) {
  if (ledger == nullptr || tickets == nullptr || records == nullptr ||
      cell_index >= kEnvironmentCellCapacity)
    return false;
  const std::uint32_t bounded =
      ticket_count < direct_adult_core::kMaxAsynchronousTickets
          ? ticket_count
          : direct_adult_core::kMaxAsynchronousTickets;
  const direct_adult_core::AsynchronousTicket* ticket = nullptr;
  for (std::uint32_t slot = 0u; slot < bounded; ++slot)
    if (tickets[slot].ticket_id != 0u &&
        tickets[slot].ticket_id ==
            (static_cast<std::uint64_t>(0x1617ull) << 48u) + cell_index &&
        tickets[slot].settled != 0u)
      ticket = &tickets[slot];
  if (ticket == nullptr ||
      affect_verified_world_return(*ticket, records, record_count) == nullptr) {
    ++ledger->refused_unverified;
    return false;
  }
  DirectEnvironmentCell& cell = ledger->cells[cell_index];
  cell.altering_ticket = ticket->ticket_id;
  cell.value = new_value;
  ++cell.revision;
  ledger->occupied_mask |= (1 << cell_index);
  ledger->ledger_identity =
      phenotype_fold(phenotype_fold(ledger->ledger_identity,
                                    ticket->ticket_id),
                     static_cast<std::uint64_t>(new_value));
  return true;
}

// Verify the external surface: every altered cell must still carry a
// consistent alteration record. This reads the environment ledger alone --
// no actor tables participate.
__host__ __device__ inline bool verify_environment_state(
    const DirectExtendedPhenotypeLedger& ledger,
    std::uint32_t expected_cells) {
  std::uint32_t altered = 0u;
  for (std::uint32_t i = 0u; i < kEnvironmentCellCapacity; ++i) {
    const bool marked = (ledger.occupied_mask & (1 << i)) != 0u;
    const bool consistent =
        ledger.cells[i].altering_ticket != 0u &&
        ledger.cells[i].revision > 0u && ledger.cells[i].value != 0u;
    if (marked != consistent) return false;
    if (marked) ++altered;
  }
  return altered == expected_cells;
}

}  // namespace substrate::direct_network

#endif
