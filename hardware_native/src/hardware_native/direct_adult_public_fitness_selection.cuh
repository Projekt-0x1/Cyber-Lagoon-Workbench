#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PUBLIC_FITNESS_SELECTION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PUBLIC_FITNESS_SELECTION_CUH

// i.outer_selection_information (#1593).
// Evolutionary selection consumes public fitness receipts only: boundary-
// measured outcomes bound to each construction's root digest.  The selector
// holds no resident pointers, so no private internal oracle can leak into
// the choice, and zero-evidence receipts refuse fail-closed.

#include <cstdint>
#include <type_traits>

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kPublicFitnessMaxCandidates = 16u;

struct DirectPublicFitnessReceipt {
  std::uint64_t construction_root;  // genome/birth digest of the candidate
  std::uint64_t assimilated_consequences;
  std::uint64_t emitted_actions;
  std::uint32_t lived_epochs;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectPublicFitnessReceipt> &&
              std::is_standard_layout_v<DirectPublicFitnessReceipt>);

inline bool direct_public_fitness_receipt_valid(
    const DirectPublicFitnessReceipt& receipt) {
  return receipt.construction_root != 0u &&
         receipt.assimilated_consequences > 0ull &&
         receipt.emitted_actions > 0ull && receipt.lived_epochs > 0u;
}

// Lexicographic public ordering: consequences, then actions, then epochs,
// with the construction root as the deterministic final tie-break.
inline int compare_public_fitness(const DirectPublicFitnessReceipt& a,
                                  const DirectPublicFitnessReceipt& b) {
  if (a.assimilated_consequences != b.assimilated_consequences)
    return a.assimilated_consequences < b.assimilated_consequences ? -1 : 1;
  if (a.emitted_actions != b.emitted_actions)
    return a.emitted_actions < b.emitted_actions ? -1 : 1;
  if (a.lived_epochs != b.lived_epochs)
    return a.lived_epochs < b.lived_epochs ? -1 : 1;
  if (a.construction_root != b.construction_root)
    return a.construction_root < b.construction_root ? -1 : 1;
  return 0;
}

// Selects the fittest candidate index from receipts alone.  Returns false
// for empty input or any zero-evidence receipt (fail-closed).
inline bool select_fittest_public_receipt(
    const DirectPublicFitnessReceipt* receipts, std::uint32_t count,
    std::uint32_t* winner_index) {
  if (receipts == nullptr || winner_index == nullptr || count == 0u ||
      count > kPublicFitnessMaxCandidates)
    return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (!direct_public_fitness_receipt_valid(receipts[i])) return false;
  std::uint32_t best = 0u;
  for (std::uint32_t i = 1u; i < count; ++i)
    if (compare_public_fitness(receipts[i], receipts[best]) > 0) best = i;
  *winner_index = best;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
