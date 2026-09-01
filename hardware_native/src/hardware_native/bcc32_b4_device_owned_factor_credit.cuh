#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_b3_factor_amplitude_matter.cuh"
#include "hardware_native/bcc32_cuda_paged_resident_route_bank.cuh"

namespace bcc32::device_owned_factor_credit {

namespace factor = bcc32_b3_factor_amplitude_matter;
namespace bank = paged_resident_credit;
using substrate::bcc32::SiteWord;

constexpr std::uint32_t kRegionCount = factor::kFactorFamilies;
constexpr std::uint32_t kRouteCount =
    factor::kFactorFamilies * factor::kFactorWords;

struct ResidentClock {
  std::uint64_t contact = 0u;
};

struct FactorRouteBinding {
  bank::RouteKey key{};
  std::uint32_t factor_index = bank::kInvalidIndex;
  std::uint32_t slot = bank::kInvalidSlot;
  std::uint64_t lesion_generation = 0u;
  std::uint32_t bound = 0u;
};

struct WorldResidualView {
  const SiteWord* world = nullptr;
  std::uint64_t world_words = 0u;
  std::uint64_t positive_endpoint = 0u;
  std::uint64_t negative_endpoint = 0u;
  SiteWord vacancy = 0u;
  SiteWord* positive_supply = nullptr;
  SiteWord* negative_supply = nullptr;
  std::uint32_t region = 0u;
  std::uint32_t enabled = 0u;
};

struct CreditReceipt {
  bank::OperationCode code = bank::OperationCode::kNotRun;
  std::uint32_t admitted = 0u;
  std::uint32_t matched_regions = 0u;
  std::uint32_t consumed_regions = 0u;
  std::uint32_t unbound = 0u;
  std::uint32_t journal_count = 0u;
  bank::OwnerScalars bank_before{};
  bank::OwnerScalars bank_after{};
  ResidentClock clock_before{};
  ResidentClock clock_after{};
  SiteWord positive_before[kRegionCount]{};
  SiteWord negative_before[kRegionCount]{};
  SiteWord matched_before[kRegionCount]{};
  SiteWord escrow_before[kRegionCount]{};
  SiteWord transduced_positive[kRegionCount]{};
  SiteWord transduced_negative[kRegionCount]{};
  SiteWord escrow_after[kRegionCount]{};
  SiteWord world_positive = 0u;
  SiteWord world_negative = 0u;
  SiteWord positive_supply_before = 0u;
  SiteWord negative_supply_before = 0u;
  SiteWord positive_supply_after = 0u;
  SiteWord negative_supply_after = 0u;
  std::uint32_t world_region = 0u;
  std::uint32_t world_enabled = 0u;
  std::uint64_t binding_hash = 0u;
};

struct DeviceCreditView {
  bank::PagedBankView bank{};
  factor::DeviceFactorRing factors{};
  const FactorRouteBinding* bindings = nullptr;
  std::uint32_t binding_count = 0u;
  ResidentClock* clock = nullptr;
  SiteWord* positive_regions = nullptr;
  SiteWord* negative_regions = nullptr;
  SiteWord* matched_regions = nullptr;
  SiteWord* residual_escrow = nullptr;
  bank::JournalEntry* journal = nullptr;
  std::uint32_t journal_capacity = 0u;
  CreditReceipt* receipt = nullptr;
  CreditReceipt* attempt = nullptr;
  WorldResidualView world{};
};

__device__ inline bool valid_view(const DeviceCreditView& view) {
  return bank::valid_shape(view.bank) && view.clock != nullptr &&
         view.bindings != nullptr && view.binding_count != 0u &&
         view.positive_regions != nullptr && view.negative_regions != nullptr &&
         view.matched_regions != nullptr && view.residual_escrow != nullptr &&
         view.journal != nullptr && view.receipt != nullptr &&
         view.attempt != nullptr &&
         view.journal_capacity >= view.binding_count;
}

__device__ inline bool valid_world_view(const WorldResidualView& world) {
  return world.enabled == 0u ||
         (world.world != nullptr && world.positive_supply != nullptr &&
          world.negative_supply != nullptr && world.vacancy != 0u &&
          world.world_words != 0u &&
          world.positive_endpoint < world.world_words &&
          world.negative_endpoint < world.world_words &&
          world.positive_endpoint != world.negative_endpoint &&
          world.positive_supply != world.negative_supply &&
          world.region < kRegionCount);
}

__device__ inline void restore_wrapper(const DeviceCreditView& view,
                                       const CreditReceipt& receipt) {
  *view.clock = receipt.clock_before;
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    view.positive_regions[region] = receipt.positive_before[region];
    view.negative_regions[region] = receipt.negative_before[region];
    view.matched_regions[region] = receipt.matched_before[region];
    view.residual_escrow[region] = receipt.escrow_before[region];
  }
  if (receipt.world_enabled != 0u) {
    *view.world.positive_supply = receipt.positive_supply_before;
    *view.world.negative_supply = receipt.negative_supply_before;
  }
}

__device__ inline void clear_commit_receipt(CreditReceipt* receipt) {
  receipt->admitted = 0u;
  receipt->consumed_regions = 0u;
  receipt->journal_count = 0u;
  receipt->bank_after = bank::OwnerScalars{};
  receipt->clock_after = ResidentClock{};
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    receipt->transduced_positive[region] = 0u;
    receipt->transduced_negative[region] = 0u;
    receipt->escrow_after[region] = 0u;
  }
  receipt->positive_supply_after = 0u;
  receipt->negative_supply_after = 0u;
}

__device__ inline std::uint64_t binding_state_hash(
    const FactorRouteBinding* bindings, std::uint32_t count) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  auto mix = [&hash](std::uint64_t value, std::uint32_t bytes) {
    for (std::uint32_t byte = 0u; byte < bytes; ++byte) {
      hash ^= static_cast<unsigned char>(value >> (byte * 8u));
      hash *= UINT64_C(1099511628211);
    }
  };
  for (std::uint32_t index = 0u; index < count; ++index) {
    const FactorRouteBinding binding = bindings[index];
    mix(binding.key.anchor, sizeof(binding.key.anchor));
    mix(binding.key.previous, sizeof(binding.key.previous));
    mix(binding.key.next, sizeof(binding.key.next));
    mix(binding.key.region, sizeof(binding.key.region));
    mix(binding.factor_index, sizeof(binding.factor_index));
    mix(binding.slot, sizeof(binding.slot));
    mix(binding.lesion_generation, sizeof(binding.lesion_generation));
    mix(binding.bound, sizeof(binding.bound));
  }
  return hash;
}

// Applies one factor-credit transaction while the caller owns transaction_lock.
// The local receipt is published only by the outer transaction after commit.
__device__ inline bool apply_credit_locked(DeviceCreditView view,
                                           CreditReceipt* local_ptr) {
  CreditReceipt& local = *local_ptr;
  local.bank_before = *view.bank.scalars;
  local.bank_before.transaction_lock = 0u;
  local.clock_before = *view.clock;
  local.binding_hash =
      binding_state_hash(view.bindings, view.binding_count);
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    local.positive_before[region] = view.positive_regions[region];
    local.negative_before[region] = view.negative_regions[region];
    local.matched_before[region] = view.matched_regions[region];
    local.escrow_before[region] = view.residual_escrow[region];
  }
  if (view.world.enabled != 0u) {
    local.world_enabled = 1u;
    local.world_region = view.world.region;
    local.world_positive = view.world.world[view.world.positive_endpoint];
    local.world_negative = view.world.world[view.world.negative_endpoint];
    local.positive_supply_before = *view.world.positive_supply;
    local.negative_supply_before = *view.world.negative_supply;
    const bool positive =
        (local.world_positive & view.world.vacancy) == 0u;
    const bool negative =
        (local.world_negative & view.world.vacancy) == 0u;
    if (positive && negative) {
      local.code = bank::OperationCode::kRejected;
      return false;
    }
    SiteWord* const region =
        positive ? view.positive_regions + view.world.region
                 : view.negative_regions + view.world.region;
    SiteWord* const supply =
        positive ? view.world.positive_supply : view.world.negative_supply;
    if ((positive || negative) &&
        (*region != 0u || __popc(*supply) != 1)) {
      local.code = bank::OperationCode::kRejected;
      return false;
    }
    if (positive || negative) {
      *region = *supply;
      *supply = 0u;
    }
  }

  const std::uint32_t lane_index =
      static_cast<std::uint32_t>(view.clock->contact % factor::kHorizon);
  const SiteWord* const eligibility = view.factors.lane[lane_index];
  bool invalid = eligibility == nullptr ||
                 view.bank.scalars->state_epoch == UINT64_MAX ||
                 view.clock->contact == UINT64_MAX;
  std::int32_t signs[kRegionCount]{};
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    const bool positive = view.positive_regions[region] != 0u;
    const bool negative = view.negative_regions[region] != 0u;
    const bool matched = view.matched_regions[region] != 0u;
    const std::uint32_t active =
        static_cast<std::uint32_t>(positive) +
        static_cast<std::uint32_t>(negative) +
        static_cast<std::uint32_t>(matched);
    if (active > 1u || (active != 0u && view.residual_escrow[region] != 0u))
      invalid = true;
    signs[region] = positive ? 1 : (negative ? -1 : 0);
    local.matched_regions += matched ? 1u : 0u;
  }
  if (invalid) {
    restore_wrapper(view, local);
    local.code = bank::OperationCode::kRejected;
    local.bank_after = bank::OwnerScalars{};
    local.clock_after = ResidentClock{};
    return false;
  }

  for (std::uint32_t binding_index = 0u;
       binding_index < view.binding_count; ++binding_index) {
    const FactorRouteBinding binding = view.bindings[binding_index];
    if (binding.bound == 0u) {
      ++local.unbound;
      continue;
    }
    if (binding.factor_index >= kRouteCount ||
        binding.slot >= view.bank.capacity ||
        !bank::valid_key(binding.key)) {
      invalid = true;
      break;
    }
    const std::uint32_t factor_index = binding.factor_index;
    const SiteWord eligible_word = eligibility[factor_index];
    if (eligible_word == 0u) continue;
    if (!factor::is_canonical_amplitude(eligible_word)) {
      invalid = true;
      break;
    }
    const std::uint32_t region = factor_index / factor::kFactorWords;
    const std::int32_t sign = signs[region];
    if (sign == 0) continue;
    bank::RouteMatter* const route =
        bank::route_at(view.bank, binding.slot);
    if (route == nullptr || route->occupied == 0u ||
        !resident_credit::same_key(route->key, binding.key) ||
        route->lesion_generation != binding.lesion_generation ||
        route->lesion_active != 0u) {
      invalid = true;
      break;
    }
    bank::JournalEntry entry{};
    entry.route_before = *route;
    entry.slot = binding.slot;
    SiteWord* const destination =
        sign > 0 ? &route->positive_word : &route->negative_word;
    SiteWord* const source =
        sign > 0 ? &view.positive_regions[region]
                 : &view.negative_regions[region];
    const SiteWord occupied = route->positive_word | route->negative_word |
                              route->lesion_positive_word |
                              route->lesion_negative_word;
    const SiteWord available = *source & ~occupied;
    if (available == 0u) {
      continue;
    }
    const SiteWord quantum =
        static_cast<SiteWord>(1u << (__ffs(available) - 1));
    *source &= ~quantum;
    *destination |= quantum;
    entry.route_after = *route;
    view.journal[local.journal_count++] = entry;
    if (sign > 0)
      local.transduced_positive[region] |= quantum;
    else
      local.transduced_negative[region] |= quantum;
    ++local.admitted;
  }

  if (invalid) {
    bank::rollback_entries(view.bank, view.journal, local.journal_count,
                           local.bank_before);
    view.bank.scalars->transaction_lock = 1u;
    restore_wrapper(view, local);
    bank::clear_journal_entries(view.journal, local.journal_count);
    clear_commit_receipt(&local);
    local.journal_count = 0u;
    local.code = bank::OperationCode::kRejected;
    local.bank_after = bank::OwnerScalars{};
    return false;
  }

  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if (signs[region] > 0) {
      view.residual_escrow[region] = view.positive_regions[region];
      local.escrow_after[region] = view.residual_escrow[region];
      view.positive_regions[region] = 0u;
      ++local.consumed_regions;
    } else if (signs[region] < 0) {
      view.residual_escrow[region] = view.negative_regions[region];
      local.escrow_after[region] = view.residual_escrow[region];
      view.negative_regions[region] = 0u;
      ++local.consumed_regions;
    } else if (local.matched_before[region] != 0u) {
      view.residual_escrow[region] = view.matched_regions[region];
      local.escrow_after[region] = view.residual_escrow[region];
      view.matched_regions[region] = 0u;
      ++local.consumed_regions;
    }
  }
  ++view.clock->contact;
  ++view.bank.scalars->committed_transactions;
  ++view.bank.scalars->state_epoch;
  local.code = bank::OperationCode::kOk;
  local.bank_after = *view.bank.scalars;
  local.bank_after.transaction_lock = 0u;
  local.clock_after = *view.clock;
  if (view.world.enabled != 0u) {
    local.positive_supply_after = *view.world.positive_supply;
    local.negative_supply_after = *view.world.negative_supply;
  }
  return true;
}

// The host launches this kernel but supplies no route, sign, lane, or target.
// Credit may mutate only an already occupied route named by a resident binding.
static __global__ void apply_credit_kernel(DeviceCreditView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.attempt == nullptr) return;
  CreditReceipt local{};
  if (!valid_view(view) || !valid_world_view(view.world)) {
    local.code = bank::OperationCode::kInvalidInput;
    *view.attempt = local;
    return;
  }
  if (atomicCAS(&view.bank.scalars->transaction_lock, 0u, 1u) != 0u) {
    local.code = bank::OperationCode::kBusy;
    *view.attempt = local;
    return;
  }
  const bool committed = apply_credit_locked(view, &local);
  view.bank.scalars->transaction_lock = 0u;
  if (committed) *view.receipt = local;
  *view.attempt = local;
}

__device__ inline bank::OperationCode inverse_credit_locked(
    DeviceCreditView view) {
  if (view.receipt->journal_count > view.binding_count ||
      view.receipt->journal_count > view.journal_capacity) {
    return bank::OperationCode::kStale;
  }
  if (view.receipt->code != bank::OperationCode::kOk ||
      !bank::same_owner_state(*view.bank.scalars,
                              view.receipt->bank_after) ||
      view.clock->contact != view.receipt->clock_after.contact ||
      binding_state_hash(view.bindings, view.binding_count) !=
          view.receipt->binding_hash) {
    return bank::OperationCode::kStale;
  }
  if (view.receipt->world_enabled != view.world.enabled ||
      (view.receipt->world_enabled != 0u &&
       (view.receipt->world_region != view.world.region ||
        view.world.world[view.world.positive_endpoint] !=
            view.receipt->world_positive ||
        view.world.world[view.world.negative_endpoint] !=
            view.receipt->world_negative ||
        *view.world.positive_supply !=
            view.receipt->positive_supply_after ||
        *view.world.negative_supply !=
            view.receipt->negative_supply_after))) {
    return bank::OperationCode::kStale;
  }
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if (view.positive_regions[region] != 0u ||
        view.negative_regions[region] != 0u ||
        view.matched_regions[region] != 0u ||
        view.residual_escrow[region] != view.receipt->escrow_after[region]) {
      return bank::OperationCode::kStale;
    }
  }
  for (std::uint32_t index = 0u; index < view.receipt->journal_count;
       ++index) {
    if (view.journal[index].slot >= view.bank.capacity) {
      return bank::OperationCode::kStale;
    }
  }
  for (std::uint32_t index = 0u; index < view.receipt->journal_count;
       ++index) {
    const bank::JournalEntry entry = view.journal[index];
    bool last_for_slot = true;
    for (std::uint32_t later = index + 1u;
         later < view.receipt->journal_count; ++later) {
      if (view.journal[later].slot == entry.slot) {
        last_for_slot = false;
        break;
      }
    }
    if (!last_for_slot) continue;
    const bank::RouteMatter* route = bank::route_at(view.bank, entry.slot);
    if (route == nullptr || !bank::same_route_state(*route, entry.route_after)) {
      return bank::OperationCode::kStale;
    }
  }
  bank::rollback_entries(view.bank, view.journal,
                         view.receipt->journal_count,
                         view.receipt->bank_before);
  view.bank.scalars->transaction_lock = 1u;
  restore_wrapper(view, *view.receipt);
  bank::clear_journal_entries(view.journal, view.receipt->journal_count);
  *view.receipt = CreditReceipt{};
  return bank::OperationCode::kOk;
}

static __global__ void inverse_credit_kernel(DeviceCreditView view,
                                             bank::OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = bank::OperationReceipt{};
  if (!valid_view(view) || !valid_world_view(view.world)) {
    result->code = bank::OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&view.bank.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = bank::OperationCode::kBusy;
    return;
  }
  result->code = inverse_credit_locked(view);
  view.bank.scalars->transaction_lock = 0u;
}

}  // namespace bcc32::device_owned_factor_credit
