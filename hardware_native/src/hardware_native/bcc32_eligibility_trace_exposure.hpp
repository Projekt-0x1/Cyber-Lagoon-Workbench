#pragma once

// Bounded site-local candidate for projecting a generic positive-carrier
// vacancy into a temporary E-bearing word.  This is intentionally not
// registered in canonical F and does not distinguish eligibility or H4.  It is
// a value-neutral involution over two exact ordinary-matter words and abstains
// on every other 32-bit state.

#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {

inline constexpr std::uint32_t kCarrierVacancyExposureNoBasis = 4u;

[[nodiscard]] __host__ __device__ constexpr SiteWord
carrier_vacancy_hidden_word(std::uint32_t basis) {
  return basis < 4u ? static_cast<SiteWord>(
                          kQ ^ carrier_bit(basis))
                    : kQ;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
carrier_vacancy_exposed_word(std::uint32_t basis) {
  return basis < 4u
             ? static_cast<SiteWord>(
                   (carrier_vacancy_hidden_word(basis) ^
                    carrier_bit(basis + 4u)) |
                   energy_bit(basis))
             : kQ;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
carrier_vacancy_exposure_basis(SiteWord word) {
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    if (word == carrier_vacancy_hidden_word(basis) ||
        word == carrier_vacancy_exposed_word(basis))
      return basis;
  }
  return kCarrierVacancyExposureNoBasis;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
carrier_vacancy_exposure_word(SiteWord word) {
  const std::uint32_t basis = carrier_vacancy_exposure_basis(word);
  if (basis == kCarrierVacancyExposureNoBasis)
    return word;
  return static_cast<SiteWord>(word ^ carrier_bit(basis + 4u) ^
                               energy_bit(basis));
}

// Dense and active entry points share the exact word circuit. Out-of-range
// slots and every occurrence of a duplicated active slot fail closed without
// touching that word, giving CPU and CUDA the same deterministic projection.
inline void apply_carrier_vacancy_exposure_dense(SiteWord* words,
                                                 std::size_t word_count) {
  if (words == nullptr)
    return;
  for (std::size_t index = 0u; index < word_count; ++index)
    words[index] = carrier_vacancy_exposure_word(words[index]);
}

[[nodiscard]] __host__ __device__ inline bool
carrier_vacancy_active_slot_is_unique(const std::uint64_t* active_slots,
                                      std::size_t active_count,
                                      std::size_t index) {
  const std::uint64_t slot = active_slots[index];
  for (std::size_t other = 0u; other < active_count; ++other)
    if (other != index && active_slots[other] == slot)
      return false;
  return true;
}

inline void apply_carrier_vacancy_exposure_active(
    SiteWord* words, std::size_t word_count, const std::uint64_t* active_slots,
    std::size_t active_count) {
  if (words == nullptr || active_slots == nullptr)
    return;
  for (std::size_t index = 0u; index < active_count; ++index) {
    const std::uint64_t slot = active_slots[index];
    if (slot < word_count && carrier_vacancy_active_slot_is_unique(
                                active_slots, active_count, index))
      words[slot] = carrier_vacancy_exposure_word(words[slot]);
  }
}

static_assert(carrier_vacancy_exposure_word(carrier_vacancy_hidden_word(0u)) ==
              carrier_vacancy_exposed_word(0u));
static_assert(carrier_vacancy_exposure_word(carrier_vacancy_exposed_word(0u)) ==
              carrier_vacancy_hidden_word(0u));

}  // namespace substrate::bcc32
