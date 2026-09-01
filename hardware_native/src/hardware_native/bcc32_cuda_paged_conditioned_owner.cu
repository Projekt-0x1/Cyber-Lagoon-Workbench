#include <array>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "hardware_native/bcc32_b4_device_owned_factor_credit.cuh"
#include "hardware_native/bcc32_cuda_paged_conditioned_owner.hpp"
#include "hardware_native/bcc32_cuda_paged_conditioned_owner_impl.cuh"

namespace bcc32::paged_conditioned_owner {

namespace {

static_assert(kFactorRegionCount == bcc32::device_owned_factor_credit::kRegionCount);
static_assert(kFactorHorizon == bcc32_b3_factor_amplitude_matter::kHorizon);
static_assert(kFactorWordsPerRegion ==
              bcc32_b3_factor_amplitude_matter::kFactorWords);
static_assert(kResidentFactorWords ==
              bcc32::device_owned_factor_credit::kRouteCount);
static_assert(static_cast<std::uint32_t>(OperationCode::kOk) ==
              static_cast<std::uint32_t>(bank::OperationCode::kOk));
static_assert(static_cast<std::uint32_t>(OperationCode::kRejected) ==
              static_cast<std::uint32_t>(bank::OperationCode::kRejected));
static_assert(kInvalidFactorSlot == bank::kInvalidSlot);

constexpr std::uint64_t kFactorCheckpointMagic = UINT64_C(0x3143414643323342);
constexpr std::uint32_t kFactorCheckpointVersion = 2u;

struct FactorCheckpointHeader {
  std::uint64_t magic = kFactorCheckpointMagic;
  std::uint32_t version = kFactorCheckpointVersion;
  std::uint32_t lane_words = 0u;
  std::uint32_t binding_count = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t physical_hash = 0u;
};

template <typename T>
void write_factor_plain(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output)
    throw std::runtime_error("owner factor checkpoint write failed");
}

template <typename T>
T read_factor_plain(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input)
    throw std::runtime_error("owner factor checkpoint truncated");
  return value;
}

void hash_bytes(std::uint64_t* hash, const void* data, std::size_t bytes) {
  const auto* raw = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0u; index < bytes; ++index) {
    *hash ^= raw[index];
    *hash *= UINT64_C(1099511628211);
  }
}

__global__ void initialize_eligibility_supply_kernel(
    std::uint32_t* supply, std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (supply == nullptr || index >= count) return;
  supply[index] =
      bcc32_b3_factor_amplitude_matter::encode_amplitude(1);
}

__global__ void resolve_factor_bindings_kernel(bank::PagedBankView owner,
                                               ResidentFactorBinding* bindings,
                                               std::uint32_t binding_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= binding_count || bindings == nullptr)
    return;
  auto binding = bindings[index];
  const bank::RouteKey key{binding.anchor, binding.previous, binding.next, 0u};
  binding.slot = bank::find_route(owner, key);
  const bank::RouteMatter* route = bank::route_at(owner, binding.slot);
  if (route == nullptr || route->lesion_active != 0u) {
    binding.slot = bank::kInvalidSlot;
    binding.lesion_generation = 0u;
    binding.bound = 0u;
  } else {
    binding.lesion_generation = route->lesion_generation;
    binding.bound = 1u;
  }
  bindings[index] = binding;
}

__global__ void adapt_factor_bindings_kernel(
    const ResidentFactorBinding* source,
    bcc32::device_owned_factor_credit::FactorRouteBinding* destination,
    std::uint32_t binding_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= binding_count || source == nullptr || destination == nullptr)
    return;
  const ResidentFactorBinding binding = source[index];
  destination[index] = {{binding.anchor, binding.previous, binding.next, 0u},
                        binding.factor_index,
                        binding.slot,
                        binding.lesion_generation,
                        binding.bound};
}

struct ParticipationReceiptInternal {
  bank::OperationCode code = bank::OperationCode::kNotRun;
  std::uint32_t admitted = 0u;
  std::uint32_t factor_index = bank::kInvalidIndex;
  std::uint32_t slot = bank::kInvalidSlot;
  bank::OwnerScalars bank_before{};
  bank::OwnerScalars bank_after{};
  ResidentFactorBinding binding_before{};
  ResidentFactorBinding binding_after{};
  std::uint32_t lane_before = 0u;
  std::uint32_t lane_after = 0u;
  std::uint32_t source_before = 0u;
  std::uint32_t source_after = 0u;
  std::uint32_t* lane = nullptr;
  std::uint32_t* source = nullptr;
};

struct PredictionWitnessReceiptInternal {
  std::uint32_t requested = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t rejected = 0u;
};

struct ExpiredEligibilityEntry {
  std::uint32_t factor_index = kInvalidFactorSlot;
  std::uint32_t lane_before = 0u;
  std::uint32_t supply_before = 0u;
};

struct DelayedFactorKeyIndexEntry {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::uint32_t head = kInvalidFactorSlot;
};

struct DelayedFactorCreditReceiptInternal {
  bank::OperationCode code = bank::OperationCode::kNotRun;
  std::uint32_t requested = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t matched = 0u;
  std::uint32_t errors = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t expired = 0u;
  std::uint32_t journal_count = 0u;
  std::uint32_t expiry_count = 0u;
  bank::OwnerScalars bank_before{};
  bank::OwnerScalars bank_after{};
  ResidentFactorClock clock_before{};
  ResidentFactorClock clock_after{};
};

__device__ bool same_binding(const ResidentFactorBinding& left,
                             const ResidentFactorBinding& right) {
  return left.anchor == right.anchor && left.previous == right.previous &&
         left.next == right.next && left.region == right.region &&
         left.factor_index == right.factor_index && left.slot == right.slot &&
         left.lesion_generation == right.lesion_generation &&
         left.bound == right.bound && left.reserved == right.reserved;
}

__device__ bool binding_matches_route(
    const ResidentFactorBinding& binding,
    const ResidentPredictionWitness& witness, std::uint32_t route_slot,
    std::uint64_t lesion_generation) {
  return binding.bound != 0u && binding.anchor == witness.anchor &&
         binding.previous == witness.previous &&
         binding.next == witness.next && binding.slot == route_slot &&
         binding.lesion_generation == lesion_generation;
}

__device__ bool factor_lanes_are_idle(const ResidentFactorRing& factors,
                                      std::uint32_t factor_index) {
  for (std::uint32_t lane = 0u; lane < kFactorHorizon; ++lane) {
    if (factors.lane[lane] == nullptr ||
        factors.lane[lane][factor_index] != 0u)
      return false;
  }
  return true;
}

__device__ bool factor_endpoint_is_idle(
    const ResidentFactorRing& factors,
    const std::uint32_t* eligibility_supply, std::uint32_t factor_index) {
  if (!bcc32_b3_factor_amplitude_matter::is_canonical_amplitude(
          eligibility_supply[factor_index]))
    return false;
  return factor_lanes_are_idle(factors, factor_index);
}

__device__ bool factor_endpoint_source_is_idle(
    const ResidentFactorRing& factors, const std::uint32_t* source,
    std::uint32_t factor_index) {
  if (source == nullptr ||
      !bcc32_b3_factor_amplitude_matter::is_canonical_amplitude(*source))
    return false;
  return factor_lanes_are_idle(factors, factor_index);
}

__device__ std::uint32_t select_resident_factor_endpoint(
    const ResidentFactorRing& factors,
    const ResidentFactorBinding* bindings,
    const std::uint32_t* eligibility_supply, std::uint32_t lane_words,
  const ResidentPredictionWitness& witness, std::uint32_t route_slot,
    std::uint64_t lesion_generation) {
  const std::uint32_t preferred = route_slot % lane_words;
  std::uint32_t first_idle = kInvalidFactorSlot;
  for (std::uint32_t probe = 0u; probe < lane_words; ++probe) {
    const std::uint32_t candidate = (preferred + probe) % lane_words;
    if (binding_matches_route(bindings[candidate], witness, route_slot,
                              lesion_generation))
      return candidate;
    const bool idle =
        factor_endpoint_is_idle(factors, eligibility_supply, candidate);
    if (idle && first_idle == kInvalidFactorSlot) first_idle = candidate;
    // Bindings are retained as tombstones after expiry. An unbound endpoint
    // therefore terminates the probe chain: this route cannot have an exact
    // binding beyond it.
    if (bindings[candidate].bound == 0u) return first_idle;
  }
  return first_idle;
}

__device__ bool capture_factor_participation_locked(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    const ResidentFactorParticipation& event,
    ParticipationReceiptInternal* local) {
  local->bank_before = *owner.scalars;
  local->bank_before.transaction_lock = 0u;
  const std::uint32_t factor_region =
      event.factor_index / bcc32_b3_factor_amplitude_matter::kFactorWords;
  if (event.valid == 0u || event.eligibility_source == nullptr ||
      event.factor_index >= lane_words ||
      event.factor_index >= binding_count ||
      factor_region >= kFactorRegionCount || event.region != factor_region ||
      owner.scalars->state_epoch == UINT64_MAX) {
    local->code = bank::OperationCode::kRejected;
    return false;
  }
  const bank::RouteKey key{event.anchor, event.previous, event.next, 0u};
  const std::uint32_t slot = bank::find_route(owner, key);
  const bank::RouteMatter* route = bank::route_at(owner, slot);
  std::uint32_t* const lane =
      factors.lane[clock->contact % kFactorHorizon];
  if (route == nullptr || route->lesion_active != 0u || lane == nullptr ||
      !bcc32_b3_factor_amplitude_matter::is_canonical_amplitude(
          *event.eligibility_source) ||
      lane[event.factor_index] != 0u) {
    local->code = bank::OperationCode::kRejected;
    return false;
  }
  const ResidentFactorBinding prior = bindings[event.factor_index];
  if (prior.bound != 0u &&
      (prior.anchor != event.anchor || prior.previous != event.previous ||
       prior.next != event.next || prior.region != event.region ||
       prior.factor_index != event.factor_index || prior.slot != slot ||
       prior.lesion_generation != route->lesion_generation) &&
      !factor_endpoint_source_is_idle(factors, event.eligibility_source,
                                      event.factor_index)) {
    local->code = bank::OperationCode::kRejected;
    return false;
  }

  local->factor_index = event.factor_index;
  local->slot = slot;
  local->binding_before = prior;
  local->lane_before = lane[event.factor_index];
  local->source_before = *event.eligibility_source;
  local->lane = lane;
  local->source = event.eligibility_source;
  bindings[event.factor_index] =
      {event.anchor, event.previous, event.next, event.region,
       event.factor_index, slot, route->lesion_generation, 1u, 0u};
  lane[event.factor_index] = *event.eligibility_source;
  *event.eligibility_source = local->lane_before;
  local->binding_after = bindings[event.factor_index];
  local->lane_after = lane[event.factor_index];
  local->source_after = *event.eligibility_source;
  local->code = bank::OperationCode::kOk;
  local->admitted = 1u;
  return true;
}

__device__ void restore_factor_participation_locked(
    bank::PagedBankView owner, ResidentFactorRing,
    ResidentFactorBinding* bindings, ResidentFactorClock*,
    const ParticipationReceiptInternal& receipt) {
  bindings[receipt.factor_index] = receipt.binding_before;
  receipt.lane[receipt.factor_index] = receipt.lane_before;
  *receipt.source = receipt.source_before;
  *owner.scalars = receipt.bank_before;
  owner.scalars->transaction_lock = 1u;
}

__global__ void capture_factor_participation_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    const ResidentFactorParticipation* participation,
    ParticipationReceiptInternal* receipt,
    ParticipationReceiptInternal* attempt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr ||
      attempt == nullptr)
    return;
  ParticipationReceiptInternal local{};
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      participation == nullptr || lane_words == 0u || binding_count == 0u) {
    local.code = bank::OperationCode::kInvalidInput;
    *attempt = local;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    local.code = bank::OperationCode::kBusy;
    *attempt = local;
    return;
  }
  const ResidentFactorParticipation event = *participation;
  if (!capture_factor_participation_locked(
          owner, factors, bindings, lane_words, binding_count, clock, event,
          &local)) {
    owner.scalars->transaction_lock = 0u;
    *attempt = local;
    return;
  }
  ++owner.scalars->committed_transactions;
  ++owner.scalars->state_epoch;
  owner.scalars->transaction_lock = 0u;
  local.bank_after = *owner.scalars;
  local.code = bank::OperationCode::kOk;
  local.admitted = 1u;
  *receipt = local;
  *attempt = local;
}

__global__ void capture_prediction_witness_batch_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    std::uint32_t* eligibility_supply,
    const ResidentPredictionWitness* witnesses, std::uint32_t count,
    PredictionWitnessReceiptInternal* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  PredictionWitnessReceiptInternal result{};
  result.requested = count;
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      eligibility_supply == nullptr || (count != 0u && witnesses == nullptr) ||
      lane_words == 0u || binding_count == 0u) {
    result.rejected = count;
    *receipt = result;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    result.rejected = count;
    *receipt = result;
    return;
  }
  const std::uint32_t lane_index =
      static_cast<std::uint32_t>(clock->contact % kFactorHorizon);
  const std::uint32_t* const lane = factors.lane[lane_index];
  bool invalid = lane == nullptr || binding_count > lane_words;
  for (std::uint32_t index = 0u; !invalid && index < count; ++index) {
    const ResidentPredictionWitness witness = witnesses[index];
    if (witness.valid == 0u) continue;
    if (witness.factor_index != 0xffffffffu) {
      invalid = true;
      break;
    }
  }
  if (invalid) {
    result.rejected = count;
    owner.scalars->transaction_lock = 0u;
    *receipt = result;
    return;
  }
  for (std::uint32_t index = 0u; index < count; ++index) {
    const ResidentPredictionWitness witness = witnesses[index];
    if (witness.valid == 0u) {
      ++result.abstained;
      continue;
    }
    const std::uint32_t slot = bank::find_route(
        owner, {witness.anchor, witness.previous, witness.next, 0u});
    const bank::RouteMatter* route = bank::route_at(owner, slot);
    if (route == nullptr || route->lesion_active != 0u) {
      ++result.abstained;
      continue;
    }
    const std::uint32_t factor_index = select_resident_factor_endpoint(
        factors, bindings, eligibility_supply, lane_words, witness, slot,
        route->lesion_generation);
    if (factor_index == kInvalidFactorSlot) {
      ++result.abstained;
      continue;
    }
    const std::uint32_t region = factor_index / kFactorWordsPerRegion;
    const ResidentFactorBinding prior = bindings[factor_index];
    if (binding_matches_route(prior, witness, slot,
                              route->lesion_generation) &&
        eligibility_supply[factor_index] == 0u) {
      ++result.abstained;
      continue;
    }
    ResidentFactorParticipation participation{
        witness.anchor, witness.previous, witness.next, region,
        factor_index,
        factor_index < lane_words
            ? eligibility_supply + factor_index
            : nullptr,
        1u};
    ParticipationReceiptInternal local{};
    if (capture_factor_participation_locked(
            owner, factors, bindings, lane_words, binding_count, clock,
            participation, &local)) {
      ++result.admitted;
    } else {
      ++result.rejected;
    }
  }
  if (result.admitted != 0u) {
    ++owner.scalars->committed_transactions;
    ++owner.scalars->state_epoch;
  }
  owner.scalars->transaction_lock = 0u;
  *receipt = result;
}

__device__ bool same_credit_key(
    const substrate::bcc32::ConditionedMatterDeviceCredit& event,
    const ResidentFactorBinding& binding) {
  return event.anchor == binding.anchor &&
         event.previous == binding.previous && event.next == binding.next;
}

__device__ std::uint32_t delayed_credit_key_hash(
    std::uint32_t anchor, std::uint32_t previous, std::uint32_t next) {
  std::uint32_t hash = 2166136261u;
  hash = (hash ^ anchor) * 16777619u;
  hash = (hash ^ previous) * 16777619u;
  hash = (hash ^ next) * 16777619u;
  return hash;
}

__device__ bool same_delayed_credit_key(
    const DelayedFactorKeyIndexEntry& entry,
    const substrate::bcc32::ConditionedMatterDeviceCredit& event) {
  return entry.anchor == event.anchor && entry.previous == event.previous &&
         entry.next == event.next;
}

__device__ bool build_delayed_factor_key_index(
    const ResidentFactorBinding* bindings, std::uint32_t binding_count,
    DelayedFactorKeyIndexEntry* key_index,
    std::uint32_t key_index_capacity, std::uint32_t* key_next) {
  for (std::uint32_t index = 0u; index < key_index_capacity; ++index)
    key_index[index].head = kInvalidFactorSlot;

  for (std::uint32_t binding_index = 0u;
       binding_index < binding_count; ++binding_index) {
    key_next[binding_index] = kInvalidFactorSlot;
    const ResidentFactorBinding binding = bindings[binding_index];
    if (binding.bound == 0u) continue;
    const std::uint32_t hash = delayed_credit_key_hash(
        binding.anchor, binding.previous, binding.next);
    const std::uint32_t preferred = hash & (key_index_capacity - 1u);
    bool inserted = false;
    for (std::uint32_t probe = 0u; probe < key_index_capacity; ++probe) {
      DelayedFactorKeyIndexEntry& entry =
          key_index[(preferred + probe) & (key_index_capacity - 1u)];
      if (entry.head == kInvalidFactorSlot) {
        entry.anchor = binding.anchor;
        entry.previous = binding.previous;
        entry.next = binding.next;
        entry.head = binding_index;
        inserted = true;
        break;
      }
      if (entry.anchor != binding.anchor ||
          entry.previous != binding.previous || entry.next != binding.next)
        continue;
      key_next[binding_index] = entry.head;
      entry.head = binding_index;
      inserted = true;
      break;
    }
    if (!inserted) return false;
  }
  return true;
}

__global__ void apply_delayed_factor_credit_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    std::uint32_t* eligibility_supply,
    const substrate::bcc32::ConditionedMatterDeviceCredit* events,
    std::uint32_t count, bank::JournalEntry* journal,
    std::uint32_t journal_capacity, ExpiredEligibilityEntry* expiry_journal,
    std::uint32_t expiry_capacity,
    DelayedFactorKeyIndexEntry* key_index,
    std::uint32_t key_index_capacity, std::uint32_t* key_next,
    std::uint8_t* key_state,
    DelayedFactorCreditReceiptInternal* receipt,
    DelayedFactorCreditReceiptInternal* attempt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || attempt == nullptr) return;
  DelayedFactorCreditReceiptInternal local{};
  local.requested = count;
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      eligibility_supply == nullptr || (count != 0u && events == nullptr) ||
      journal == nullptr || expiry_journal == nullptr ||
      journal_capacity < count || expiry_capacity < binding_count ||
      key_index == nullptr || key_next == nullptr || key_state == nullptr ||
      key_index_capacity < binding_count ||
      (key_index_capacity & (key_index_capacity - 1u)) != 0u ||
      lane_words == 0u || binding_count == 0u ||
      binding_count > lane_words ||
      clock->contact == UINT64_MAX) {
    local.code = bank::OperationCode::kInvalidInput;
    *attempt = local;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    local.code = bank::OperationCode::kBusy;
    *attempt = local;
    return;
  }
  local.bank_before = *owner.scalars;
  local.bank_before.transaction_lock = 0u;
  local.clock_before = *clock;
  bool invalid = false;
  for (std::uint32_t lane_index = 0u; lane_index < kFactorHorizon;
       ++lane_index)
    invalid |= factors.lane[lane_index] == nullptr;

  if (!invalid) {
    for (std::uint32_t binding_index = 0u;
         binding_index < binding_count; ++binding_index)
      key_state[binding_index] = 0u;
    invalid = !build_delayed_factor_key_index(
        bindings, binding_count, key_index, key_index_capacity, key_next);
  }

  for (std::uint32_t index = 0u; !invalid && index < count; ++index) {
    const substrate::bcc32::ConditionedMatterDeviceCredit event =
        events[index];
    if (event.valid == 0u || event.polarity == 0) continue;
    // A prediction error dominates the observation carrying the same
    // transaction identity regardless of fixed-slot adjacency.
    if (event.polarity > 0) {
      bool has_error = false;
      for (std::uint32_t paired_index = 0u; paired_index < count;
           ++paired_index) {
        const auto paired = events[paired_index];
        has_error = has_error ||
                    (paired.valid != 0u && paired.polarity < 0 &&
                     paired.source_event == event.source_event &&
                     paired.anchor == event.anchor &&
                     paired.previous == event.previous &&
                     paired.next == event.next);
      }
      if (has_error) continue;
    }

    std::uint32_t selected = kInvalidFactorSlot;
    const std::uint32_t hash = delayed_credit_key_hash(
        event.anchor, event.previous, event.next);
    const std::uint32_t preferred = hash & (key_index_capacity - 1u);
    const DelayedFactorKeyIndexEntry* key_entry = nullptr;
    for (std::uint32_t probe = 0u; probe < key_index_capacity; ++probe) {
      const DelayedFactorKeyIndexEntry& candidate =
          key_index[(preferred + probe) & (key_index_capacity - 1u)];
      if (candidate.head == kInvalidFactorSlot) break;
      if (same_delayed_credit_key(candidate, event)) {
        key_entry = &candidate;
        break;
      }
    }
    for (std::uint32_t binding_index =
             key_entry == nullptr ? kInvalidFactorSlot : key_entry->head;
         !invalid && binding_index != kInvalidFactorSlot;
         binding_index = key_next[binding_index]) {
      const ResidentFactorBinding binding = bindings[binding_index];
      if (binding.bound == 0u || binding.factor_index >= lane_words ||
          binding.factor_index != binding_index ||
          binding.region != binding_index / kFactorWordsPerRegion ||
          !same_credit_key(event, binding))
        continue;
      std::uint8_t state = key_state[binding_index];
      if (state == 0u) {
        bool live = false;
        for (std::uint32_t lane_index = 0u; lane_index < kFactorHorizon;
             ++lane_index) {
          const std::uint32_t word =
              factors.lane[lane_index][binding.factor_index];
          if (word != 0u &&
              !bcc32_b3_factor_amplitude_matter::is_canonical_amplitude(
                  word)) {
            invalid = true;
            break;
          }
          live |= word != 0u;
        }
        if (invalid) break;
        state = live ? 2u : 1u;
        key_state[binding_index] = state;
      }
      if (state != 2u) continue;
      if (selected != kInvalidFactorSlot) {
        invalid = true;
        break;
      }
      selected = binding_index;
    }
    if (invalid) break;
    if (selected == kInvalidFactorSlot) {
      if (event.polarity < 0) {
        ++local.abstained;
        continue;
      }
      const bank::RouteKey key{event.anchor, event.previous, event.next, 0u};
      std::uint32_t bucket = bank::kInvalidIndex;
      bool inserted = false;
      const std::uint32_t slot =
          bank::provision_route(owner, key, &bucket, &inserted);
      bank::RouteMatter* const route = bank::route_at(owner, slot);
      if (slot == bank::kInvalidSlot || route == nullptr ||
          route->lesion_active != 0u) {
        invalid = true;
        break;
      }
      bank::JournalEntry entry{};
      entry.route_before = inserted ? bank::RouteMatter{} : *route;
      entry.slot = slot;
      entry.bucket = bucket;
      entry.inserted = inserted ? 1u : 0u;
      std::uint32_t quantum = 0u;
      if (!bank::acquire_quantum(owner, route->positive_word,
                                 &entry.reservoir_index, &quantum,
                                 &entry.reservoir_before)) {
        if (inserted) {
          *route = bank::RouteMatter{};
          owner.directory[bucket] = 0u;
          --owner.scalars->route_count;
        }
        ++local.abstained;
        continue;
      }
      route->positive_word |= quantum;
      entry.route_after = *route;
      journal[local.journal_count++] = entry;
      ++local.admitted;
      continue;
    }

    const ResidentFactorBinding binding = bindings[selected];
    bank::RouteMatter* const route = bank::route_at(owner, binding.slot);
    if (route == nullptr || route->occupied == 0u ||
        route->key.anchor != binding.anchor ||
        route->key.previous != binding.previous ||
        route->key.next != binding.next || route->key.region != 0u) {
      invalid = true;
      break;
    }
    if (route->lesion_active != 0u ||
        route->lesion_generation != binding.lesion_generation) {
      ++local.abstained;
      continue;
    }
    bank::JournalEntry entry{};
    entry.route_before = *route;
    entry.slot = binding.slot;
    std::uint32_t reservoir_index = bank::kInvalidIndex;
    std::uint32_t quantum = 0u;
    std::uint32_t reservoir_before = 0u;
    std::uint32_t* const destination =
        event.polarity > 0 ? &route->positive_word : &route->negative_word;
    if (!bank::acquire_quantum(owner, *destination, &reservoir_index,
                               &quantum, &reservoir_before)) {
      ++local.abstained;
      continue;
    }
    entry.reservoir_index = reservoir_index;
    entry.reservoir_before = reservoir_before;
    *destination |= quantum;
    entry.route_after = *route;
    journal[local.journal_count++] = entry;
    ++local.admitted;
    if (event.polarity > 0)
      ++local.matched;
    else
      ++local.errors;
  }

  if (!invalid) {
    ++clock->contact;
    const std::uint32_t expiry_lane_index =
        static_cast<std::uint32_t>(clock->contact % kFactorHorizon);
    std::uint32_t* const expiry_lane = factors.lane[expiry_lane_index];
    invalid = expiry_lane == nullptr;
    for (std::uint32_t factor_index = 0u;
         !invalid && factor_index < binding_count; ++factor_index) {
      const std::uint32_t word = expiry_lane[factor_index];
      if (word == 0u) continue;
      if (!bcc32_b3_factor_amplitude_matter::is_canonical_amplitude(word) ||
          eligibility_supply[factor_index] != 0u) {
        invalid = true;
        break;
      }
      expiry_journal[local.expiry_count++] = {
          factor_index, word, eligibility_supply[factor_index]};
      eligibility_supply[factor_index] = word;
      expiry_lane[factor_index] = 0u;
      ++local.expired;
    }
  }

  if (invalid) {
    for (std::uint32_t offset = 0u; offset < local.expiry_count; ++offset) {
      const auto entry = expiry_journal[local.expiry_count - 1u - offset];
      factors.lane[static_cast<std::uint32_t>(
          clock->contact % kFactorHorizon)][entry.factor_index] =
          entry.lane_before;
      eligibility_supply[entry.factor_index] = entry.supply_before;
    }
    bank::rollback_entries(owner, journal, local.journal_count,
                           local.bank_before);
    *clock = local.clock_before;
    bank::clear_journal_entries(journal, local.journal_count);
    local.code = bank::OperationCode::kRejected;
    local.admitted = 0u;
    local.matched = 0u;
    local.errors = 0u;
    local.journal_count = 0u;
    local.expiry_count = 0u;
    local.expired = 0u;
    owner.scalars->transaction_lock = 0u;
    *attempt = local;
    return;
  }

  ++owner.scalars->committed_transactions;
  ++owner.scalars->state_epoch;
  local.bank_after = *owner.scalars;
  local.bank_after.transaction_lock = 0u;
  local.clock_after = *clock;
  local.code = bank::OperationCode::kOk;
  owner.scalars->transaction_lock = 0u;
  *receipt = local;
  *attempt = local;
}

__global__ void inverse_delayed_factor_credit_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    std::uint32_t* eligibility_supply, bank::JournalEntry* journal,
    std::uint32_t journal_capacity,
    ExpiredEligibilityEntry* expiry_journal,
    std::uint32_t expiry_capacity,
    DelayedFactorCreditReceiptInternal* receipt,
    bank::OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = bank::OperationReceipt{};
  if (!bank::valid_shape(owner) || clock == nullptr ||
      eligibility_supply == nullptr || journal == nullptr ||
      expiry_journal == nullptr || receipt == nullptr) {
    result->code = bank::OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = bank::OperationCode::kBusy;
    return;
  }
  const std::uint32_t expiry_lane_index =
      static_cast<std::uint32_t>(clock->contact % kFactorHorizon);
  std::uint32_t* const expiry_lane = factors.lane[expiry_lane_index];
  bool stale =
      receipt->code != bank::OperationCode::kOk ||
      receipt->journal_count > journal_capacity ||
      receipt->expiry_count > binding_count ||
      receipt->expiry_count > expiry_capacity ||
      !bank::same_owner_state(*owner.scalars, receipt->bank_after) ||
      clock->contact != receipt->clock_after.contact ||
      expiry_lane == nullptr;
  for (std::uint32_t index = 0u; !stale &&
       index < receipt->journal_count; ++index) {
    const bank::JournalEntry entry = journal[index];
    bool last_for_slot = true;
    for (std::uint32_t later = index + 1u;
         later < receipt->journal_count; ++later) {
      if (journal[later].slot == entry.slot) {
        last_for_slot = false;
        break;
      }
    }
    if (!last_for_slot) continue;
    const bank::RouteMatter* route = bank::route_at(owner, entry.slot);
    if (route == nullptr ||
        !bank::same_route_state(*route, entry.route_after))
      stale = true;
  }
  for (std::uint32_t index = 0u; !stale &&
       index < receipt->expiry_count; ++index) {
    const auto entry = expiry_journal[index];
    if (entry.factor_index >= binding_count ||
        expiry_lane[entry.factor_index] != 0u ||
        eligibility_supply[entry.factor_index] != entry.lane_before)
      stale = true;
  }
  if (stale) {
    owner.scalars->transaction_lock = 0u;
    result->code = bank::OperationCode::kStale;
    return;
  }
  bank::rollback_entries(owner, journal, receipt->journal_count,
                         receipt->bank_before);
  for (std::uint32_t offset = 0u; offset < receipt->expiry_count; ++offset) {
    const auto entry =
        expiry_journal[receipt->expiry_count - 1u - offset];
    expiry_lane[entry.factor_index] = entry.lane_before;
    eligibility_supply[entry.factor_index] = entry.supply_before;
  }
  *clock = receipt->clock_before;
  owner.scalars->transaction_lock = 0u;
  bank::clear_journal_entries(journal, receipt->journal_count);
  *receipt = DelayedFactorCreditReceiptInternal{};
  result->code = bank::OperationCode::kOk;
}

__global__ void inverse_factor_participation_kernel(
    bank::PagedBankView owner, ResidentFactorRing,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    const ResidentFactorParticipation* participation,
    ParticipationReceiptInternal* receipt, bank::OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = bank::OperationReceipt{};
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      participation == nullptr || receipt == nullptr) {
    result->code = bank::OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = bank::OperationCode::kBusy;
    return;
  }
  const ResidentFactorParticipation event = *participation;
  if (receipt->code != bank::OperationCode::kOk ||
      receipt->factor_index >= lane_words ||
      receipt->factor_index >= binding_count || receipt->lane == nullptr ||
      event.eligibility_source != receipt->source ||
      !bank::same_owner_state(*owner.scalars, receipt->bank_after) ||
      !same_binding(bindings[receipt->factor_index],
                    receipt->binding_after) ||
      receipt->lane[receipt->factor_index] != receipt->lane_after ||
      *receipt->source != receipt->source_after) {
    owner.scalars->transaction_lock = 0u;
    result->code = bank::OperationCode::kStale;
    return;
  }
  bindings[receipt->factor_index] = receipt->binding_before;
  receipt->lane[receipt->factor_index] = receipt->lane_before;
  *receipt->source = receipt->source_before;
  *owner.scalars = receipt->bank_before;
  owner.scalars->transaction_lock = 0u;
  *receipt = ParticipationReceiptInternal{};
  result->code = bank::OperationCode::kOk;
}

struct AtomicParticipationCreditReceipt {
  ParticipationReceiptInternal participation{};
  bcc32::device_owned_factor_credit::CreditReceipt credit{};
};

__global__ void apply_atomic_participation_credit_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    const ResidentFactorParticipation* participation,
    bcc32::device_owned_factor_credit::DeviceCreditView credit_view,
    AtomicParticipationCreditReceipt* committed,
    bcc32::device_owned_factor_credit::CreditReceipt* attempt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || attempt == nullptr) return;
  bcc32::device_owned_factor_credit::CreditReceipt credit{};
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      participation == nullptr || committed == nullptr || lane_words == 0u ||
      binding_count == 0u ||
      !bcc32::device_owned_factor_credit::valid_world_view(
          credit_view.world)) {
    credit.code = bank::OperationCode::kInvalidInput;
    *attempt = credit;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    credit.code = bank::OperationCode::kBusy;
    *attempt = credit;
    return;
  }

  ParticipationReceiptInternal captured{};
  const ResidentFactorParticipation event = *participation;
  if (!capture_factor_participation_locked(
          owner, factors, bindings, lane_words, binding_count, clock, event,
          &captured)) {
    owner.scalars->transaction_lock = 0u;
    credit.code = captured.code;
    *attempt = credit;
    return;
  }

  const ResidentFactorBinding resident = captured.binding_after;
  bcc32::device_owned_factor_credit::FactorRouteBinding selected{
      {resident.anchor, resident.previous, resident.next, resident.region},
      resident.factor_index,
      resident.slot,
      resident.lesion_generation,
      resident.bound};
  credit_view.bindings = &selected;
  credit_view.binding_count = 1u;
  credit_view.receipt = &credit;
  credit_view.attempt = &credit;
  if (!bcc32::device_owned_factor_credit::valid_view(credit_view) ||
      !bcc32::device_owned_factor_credit::apply_credit_locked(credit_view,
                                                               &credit)) {
    restore_factor_participation_locked(owner, factors, bindings, clock,
                                        captured);
    owner.scalars->transaction_lock = 0u;
    *attempt = credit;
    return;
  }

  captured.bank_after = credit.bank_after;
  captured.code = bank::OperationCode::kOk;
  AtomicParticipationCreditReceipt transaction{captured, credit};
  owner.scalars->transaction_lock = 0u;
  *committed = transaction;
  *attempt = credit;
}

__global__ void inverse_atomic_participation_credit_kernel(
    bank::PagedBankView owner, ResidentFactorRing factors,
    ResidentFactorBinding* bindings, std::uint32_t lane_words,
    std::uint32_t binding_count, ResidentFactorClock* clock,
    const ResidentFactorParticipation* participation,
    bcc32::device_owned_factor_credit::DeviceCreditView credit_view,
    AtomicParticipationCreditReceipt* committed,
    bank::OperationReceipt* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr) return;
  *result = bank::OperationReceipt{};
  if (!bank::valid_shape(owner) || bindings == nullptr || clock == nullptr ||
      participation == nullptr || committed == nullptr || lane_words == 0u ||
      binding_count == 0u ||
      !bcc32::device_owned_factor_credit::valid_world_view(
          credit_view.world)) {
    result->code = bank::OperationCode::kInvalidInput;
    return;
  }
  if (atomicCAS(&owner.scalars->transaction_lock, 0u, 1u) != 0u) {
    result->code = bank::OperationCode::kBusy;
    return;
  }

  ParticipationReceiptInternal& captured = committed->participation;
  const ResidentFactorParticipation event = *participation;
  if (captured.code != bank::OperationCode::kOk ||
      captured.factor_index >= lane_words ||
      captured.factor_index >= binding_count || captured.lane == nullptr ||
      captured.source == nullptr ||
      event.eligibility_source != captured.source ||
      !same_binding(bindings[captured.factor_index],
                    captured.binding_after) ||
      captured.lane[captured.factor_index] != captured.lane_after ||
      *captured.source != captured.source_after) {
    owner.scalars->transaction_lock = 0u;
    result->code = bank::OperationCode::kStale;
    return;
  }

  const ResidentFactorBinding resident =
      bindings[captured.factor_index];
  bcc32::device_owned_factor_credit::FactorRouteBinding selected{
      {resident.anchor, resident.previous, resident.next, resident.region},
      resident.factor_index,
      resident.slot,
      resident.lesion_generation,
      resident.bound};
  credit_view.bindings = &selected;
  credit_view.binding_count = 1u;
  credit_view.receipt = &committed->credit;
  credit_view.attempt = &committed->credit;
  const bank::OperationCode code =
      bcc32::device_owned_factor_credit::inverse_credit_locked(credit_view);
  if (code != bank::OperationCode::kOk) {
    owner.scalars->transaction_lock = 0u;
    result->code = code;
    return;
  }
  restore_factor_participation_locked(owner, factors, bindings, clock,
                                      captured);
  *committed = AtomicParticipationCreditReceipt{};
  owner.scalars->transaction_lock = 0u;
  result->code = bank::OperationCode::kOk;
}

}  // namespace

struct PagedConditionedOwner::Impl {
  explicit Impl(std::uint32_t capacity) : engine(capacity) {}
  ~Impl() {
    clear_resident_factor_state();
    cudaFree(factor_operation);
    cudaFree(factor_attempt);
    cudaFree(factor_receipt);
    cudaFree(factor_journal);
    cudaFree(factor_bindings);
    cudaFree(participation_receipt);
    cudaFree(participation_attempt);
    cudaFree(participation_operation);
    cudaFree(atomic_participation_credit_receipt);
    cudaFree(prediction_witness_receipt);
    cudaFree(delayed_credit_journal);
    cudaFree(delayed_expiry_journal);
    cudaFree(delayed_key_index);
    cudaFree(delayed_key_next);
    cudaFree(delayed_key_state);
    cudaFree(delayed_credit_receipt);
    cudaFree(delayed_credit_attempt);
    cudaFree(delayed_credit_operation);
  }

  void clear_resident_factor_state() noexcept {
    cudaFree(resident_factor_words);
    cudaFree(resident_bindings);
    cudaFree(resident_clock);
    cudaFree(resident_regions);
    cudaFree(resident_eligibility_supply);
    resident_factor_words = nullptr;
    resident_bindings = nullptr;
    resident_clock = nullptr;
    resident_regions = nullptr;
    resident_eligibility_supply = nullptr;
    resident_lane_words = 0u;
    resident_binding_count = 0u;
  }

  void reset_resident_factor_state(std::uint32_t lane_words,
                                   std::uint32_t binding_count) {
    clear_resident_factor_state();
    if (lane_words == 0u && binding_count == 0u)
      return;
    if (lane_words == 0u || binding_count == 0u)
      throw std::runtime_error("invalid resident factor-state shape");
    const std::size_t factor_word_count =
        static_cast<std::size_t>(lane_words) * kFactorHorizon;
    cuda_require(cudaMalloc(&resident_factor_words,
                            factor_word_count *
                                sizeof(*resident_factor_words)),
                 "allocate resident factor lanes");
    cuda_require(cudaMalloc(&resident_bindings,
                            static_cast<std::size_t>(binding_count) *
                                sizeof(*resident_bindings)),
                 "allocate resident factor bindings");
    cuda_require(cudaMalloc(&resident_clock, sizeof(*resident_clock)),
                 "allocate resident factor clock");
    cuda_require(cudaMalloc(&resident_regions,
                            4u * kFactorRegionCount *
                                sizeof(*resident_regions)),
                 "allocate resident residual regions");
    cuda_require(cudaMalloc(&resident_eligibility_supply,
                            static_cast<std::size_t>(lane_words) *
                                sizeof(*resident_eligibility_supply)),
                 "allocate resident eligibility supply");
    cuda_require(cudaMemset(resident_factor_words, 0,
                            factor_word_count *
                                sizeof(*resident_factor_words)),
                 "clear resident factor lanes");
    cuda_require(cudaMemset(resident_bindings, 0,
                            static_cast<std::size_t>(binding_count) *
                                sizeof(*resident_bindings)),
                 "clear resident factor bindings");
    cuda_require(cudaMemset(resident_clock, 0, sizeof(*resident_clock)),
                 "clear resident factor clock");
    cuda_require(cudaMemset(resident_regions, 0,
                            4u * kFactorRegionCount *
                                sizeof(*resident_regions)),
                 "clear resident residual regions");
    initialize_eligibility_supply_kernel<<<(lane_words + 255u) / 256u, 256u>>>(
        resident_eligibility_supply, lane_words);
    cuda_require(cudaGetLastError(),
                 "initialize resident eligibility supply");
    cuda_require(cudaDeviceSynchronize(),
                 "complete resident eligibility initialization");
    resident_lane_words = lane_words;
    resident_binding_count = binding_count;
  }

  ResidentFactorStateView resident_factor_state_device() const {
    ResidentFactorStateView result{};
    if (resident_lane_words == 0u)
      return result;
    for (std::uint32_t lane = 0u; lane < kFactorHorizon; ++lane) {
      result.factors.lane[lane] =
          resident_factor_words +
          static_cast<std::size_t>(lane) * resident_lane_words;
    }
    result.eligibility_supply = resident_eligibility_supply;
    result.bindings = resident_bindings;
    result.binding_count = resident_binding_count;
    result.clock = resident_clock;
    result.positive_regions = resident_regions;
    result.negative_regions = resident_regions + kFactorRegionCount;
    result.matched_regions =
        resident_regions + 2u * kFactorRegionCount;
    result.residual_escrow =
        resident_regions + 3u * kFactorRegionCount;
    return result;
  }

  // ⭐ ONE HOST-SIDE IMAGE OF THE RESIDENT FACTOR STATE.
  //
  // Hashing the state, checkpointing the state and verifying a restored
  // checkpoint all need the SAME five buffers.  Each path used to declare its
  // own copy of them and run its own five-`cudaMemcpy` download burst, so a
  // single `save()` paid the full-state device read twice and a single
  // `load()` paid a full-state upload plus a full-state read-back.  This
  // struct is the shared staging surface: one declaration, one download.
  //
  // ⚠ This is transport, not selection.  Nothing here chooses WHICH resident
  // memory is relevant -- the caller already named the whole configured
  // factor state and this only moves it.
  struct StagedFactorState {
    std::vector<std::uint32_t> factors;
    std::vector<std::uint32_t> supply;
    std::vector<ResidentFactorBinding> bindings;
    ResidentFactorClock clock{};
    std::array<std::uint32_t, 4u * kFactorRegionCount> regions{};
  };

  void size_staged_factor_state(StagedFactorState& staged) const {
    staged.factors.assign(
        static_cast<std::size_t>(resident_lane_words) * kFactorHorizon, 0u);
    staged.supply.assign(resident_lane_words, 0u);
    staged.bindings.assign(resident_binding_count, ResidentFactorBinding{});
  }

  void download_resident_factor_state(StagedFactorState& staged) const {
    size_staged_factor_state(staged);
    cuda_require(cudaMemcpy(staged.factors.data(), resident_factor_words,
                            staged.factors.size() * sizeof(staged.factors[0]),
                            cudaMemcpyDeviceToHost),
                 "stage resident factor lanes");
    cuda_require(cudaMemcpy(staged.supply.data(), resident_eligibility_supply,
                            staged.supply.size() * sizeof(staged.supply[0]),
                            cudaMemcpyDeviceToHost),
                 "stage resident eligibility supply");
    cuda_require(cudaMemcpy(staged.bindings.data(), resident_bindings,
                            staged.bindings.size() *
                                sizeof(staged.bindings[0]),
                            cudaMemcpyDeviceToHost),
                 "stage resident factor bindings");
    cuda_require(cudaMemcpy(&staged.clock, resident_clock,
                            sizeof(staged.clock), cudaMemcpyDeviceToHost),
                 "stage resident factor clock");
    cuda_require(cudaMemcpy(staged.regions.data(), resident_regions,
                            staged.regions.size() * sizeof(staged.regions[0]),
                            cudaMemcpyDeviceToHost),
                 "stage resident residual regions");
  }

  std::uint64_t hash_staged_factor_state(const StagedFactorState& staged,
                                         bool include_supply) const {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash_bytes(&hash, &resident_lane_words, sizeof(resident_lane_words));
    hash_bytes(&hash, &resident_binding_count,
               sizeof(resident_binding_count));
    hash_bytes(&hash, staged.factors.data(),
               staged.factors.size() * sizeof(staged.factors[0]));
    if (include_supply)
      hash_bytes(&hash, staged.supply.data(),
                 staged.supply.size() * sizeof(staged.supply[0]));
    hash_bytes(&hash, staged.bindings.data(),
               staged.bindings.size() * sizeof(staged.bindings[0]));
    hash_bytes(&hash, &staged.clock, sizeof(staged.clock));
    hash_bytes(&hash, staged.regions.data(),
               staged.regions.size() * sizeof(staged.regions[0]));
    return hash;
  }

  std::uint64_t resident_factor_hash(bool include_supply = true) const {
    if (resident_lane_words == 0u)
      return 0u;
    StagedFactorState staged;
    download_resident_factor_state(staged);
    return hash_staged_factor_state(staged, include_supply);
  }

  void save_resident_factor_state(std::ostream& output) const {
    // ⭐ DOWNLOAD ONCE, THEN HASH AND WRITE THE SAME BYTES.
    //
    // The header carries a hash of exactly the state this function goes on to
    // write, and it used to obtain that hash by calling `resident_factor_hash()`
    // -- a second, independent full-state download of bytes already destined
    // for a second download two statements later.  Nothing mutates the device
    // between them, so the two bursts were reading identical memory.  Staging
    // first and hashing the staged image halves the device traffic of every
    // checkpoint and yields a byte-identical header and payload.
    StagedFactorState staged;
    if (resident_lane_words != 0u)
      download_resident_factor_state(staged);
    write_factor_plain(
        output, FactorCheckpointHeader{
                    kFactorCheckpointMagic, kFactorCheckpointVersion,
                    resident_lane_words, resident_binding_count, 0u,
                    resident_lane_words == 0u
                        ? 0u
                        : hash_staged_factor_state(staged, true)});
    if (resident_lane_words == 0u)
      return;
    output.write(reinterpret_cast<const char*>(staged.factors.data()),
                 static_cast<std::streamsize>(
                     staged.factors.size() * sizeof(staged.factors[0])));
    output.write(reinterpret_cast<const char*>(staged.supply.data()),
                 static_cast<std::streamsize>(
                     staged.supply.size() * sizeof(staged.supply[0])));
    output.write(reinterpret_cast<const char*>(staged.bindings.data()),
                 static_cast<std::streamsize>(
                     staged.bindings.size() * sizeof(staged.bindings[0])));
    write_factor_plain(output, staged.clock);
    output.write(reinterpret_cast<const char*>(staged.regions.data()),
                 static_cast<std::streamsize>(
                     staged.regions.size() * sizeof(staged.regions[0])));
    if (!output)
      throw std::runtime_error("owner factor checkpoint write failed");
  }

  void load_resident_factor_state(std::istream& input) {
    if (input.peek() == std::char_traits<char>::eof()) {
      input.clear(input.rdstate() & ~std::ios::eofbit);
      return;
    }
    const FactorCheckpointHeader header =
        read_factor_plain<FactorCheckpointHeader>(input);
    if (header.magic != kFactorCheckpointMagic ||
        (header.version != 1u &&
         header.version != kFactorCheckpointVersion) ||
        ((header.lane_words == 0u) != (header.binding_count == 0u)))
      throw std::runtime_error("incompatible owner factor checkpoint");
    reset_resident_factor_state(header.lane_words, header.binding_count);
    if (header.lane_words == 0u) {
      if (header.physical_hash != 0u)
        throw std::runtime_error("invalid empty owner factor checkpoint");
      return;
    }
    StagedFactorState staged;
    size_staged_factor_state(staged);
    input.read(reinterpret_cast<char*>(staged.factors.data()),
               static_cast<std::streamsize>(
                   staged.factors.size() * sizeof(staged.factors[0])));
    if (header.version >= 2u) {
      input.read(reinterpret_cast<char*>(staged.supply.data()),
                 static_cast<std::streamsize>(
                     staged.supply.size() * sizeof(staged.supply[0])));
    }
    input.read(reinterpret_cast<char*>(staged.bindings.data()),
               static_cast<std::streamsize>(
                   staged.bindings.size() * sizeof(staged.bindings[0])));
    input.read(reinterpret_cast<char*>(&staged.clock), sizeof(staged.clock));
    input.read(reinterpret_cast<char*>(staged.regions.data()),
               static_cast<std::streamsize>(
                   staged.regions.size() * sizeof(staged.regions[0])));
    if (!input)
      throw std::runtime_error("owner factor checkpoint truncated");
    // ⭐ VERIFY THE CHECKPOINT, NOT THE PCIe BUS.
    //
    // The header hash certifies the CHECKPOINT BYTES, and those bytes are in
    // `staged` right now.  The previous shape uploaded them and then ran a
    // full-state download (`resident_factor_hash()`) to hash what it had just
    // written -- a whole extra device read of the state per restore, whose
    // only additional coverage was the copy engine itself, which
    // `cuda_require` on each upload already reports.  Hashing the staged
    // image tests exactly the same predicate on exactly the same bytes and
    // throws on exactly the same corrupt checkpoints.
    //
    // ⚠ ORDER MATTERS: hash BEFORE uploading, so a checkpoint that fails
    // verification never reaches resident memory.
    if (hash_staged_factor_state(staged, header.version >= 2u) !=
        header.physical_hash)
      throw std::runtime_error("owner factor checkpoint hash mismatch");
    cuda_require(cudaMemcpy(resident_factor_words, staged.factors.data(),
                            staged.factors.size() * sizeof(staged.factors[0]),
                            cudaMemcpyHostToDevice),
                 "restore resident factor lanes");
    if (header.version >= 2u) {
      cuda_require(cudaMemcpy(resident_eligibility_supply,
                              staged.supply.data(),
                              staged.supply.size() * sizeof(staged.supply[0]),
                              cudaMemcpyHostToDevice),
                   "restore resident eligibility supply");
    }
    cuda_require(cudaMemcpy(resident_bindings, staged.bindings.data(),
                            staged.bindings.size() *
                                sizeof(staged.bindings[0]),
                            cudaMemcpyHostToDevice),
                 "restore resident factor bindings");
    cuda_require(cudaMemcpy(resident_clock, &staged.clock,
                            sizeof(staged.clock), cudaMemcpyHostToDevice),
                 "restore resident factor clock");
    cuda_require(cudaMemcpy(resident_regions, staged.regions.data(),
                            staged.regions.size() * sizeof(staged.regions[0]),
                            cudaMemcpyHostToDevice),
                 "restore resident residual regions");
  }

  void ensure_factor_scratch(std::uint32_t binding_count) {
    if (binding_count <= factor_capacity)
      return;
    cudaFree(factor_journal);
    cudaFree(factor_bindings);
    factor_journal = nullptr;
    factor_bindings = nullptr;
    cuda_require(cudaMalloc(&factor_journal,
                            static_cast<std::size_t>(binding_count) * sizeof(bank::JournalEntry)),
                 "allocate owner factor-credit journal");
    cuda_require(cudaMalloc(&factor_bindings,
                            static_cast<std::size_t>(binding_count) * sizeof(*factor_bindings)),
                 "allocate owner factor-binding adapter");
    factor_capacity = binding_count;
  }

  void ensure_delayed_credit_scratch(std::uint32_t event_count,
                                     std::uint32_t binding_count) {
    if (event_count > delayed_event_capacity) {
      cudaFree(delayed_credit_journal);
      delayed_credit_journal = nullptr;
      cuda_require(
          cudaMalloc(&delayed_credit_journal,
                     static_cast<std::size_t>(event_count) *
                         sizeof(*delayed_credit_journal)),
          "allocate delayed factor-credit journal");
      delayed_event_capacity = event_count;
    }
    if (binding_count > delayed_expiry_capacity) {
      cudaFree(delayed_expiry_journal);
      delayed_expiry_journal = nullptr;
      cuda_require(
          cudaMalloc(&delayed_expiry_journal,
                     static_cast<std::size_t>(binding_count) *
                         sizeof(*delayed_expiry_journal)),
          "allocate delayed eligibility-expiry journal");
      delayed_expiry_capacity = binding_count;
    }
    const std::uint32_t required_key_capacity =
        binding_count > (std::numeric_limits<std::uint32_t>::max() / 2u)
            ? std::numeric_limits<std::uint32_t>::max()
            : binding_count * 2u;
    const std::uint32_t key_capacity =
        next_power_of_two(std::max<std::uint64_t>(4u, required_key_capacity));
    if (binding_count > delayed_key_capacity ||
        key_capacity > delayed_key_index_capacity) {
      cudaFree(delayed_key_index);
      cudaFree(delayed_key_next);
      cudaFree(delayed_key_state);
      delayed_key_index = nullptr;
      delayed_key_next = nullptr;
      delayed_key_state = nullptr;
      cuda_require(cudaMalloc(&delayed_key_index,
                              static_cast<std::size_t>(key_capacity) *
                                  sizeof(*delayed_key_index)),
                   "allocate delayed factor key index");
      cuda_require(cudaMalloc(&delayed_key_next,
                              static_cast<std::size_t>(binding_count) *
                                  sizeof(*delayed_key_next)),
                   "allocate delayed factor key chains");
      cuda_require(cudaMalloc(&delayed_key_state,
                              static_cast<std::size_t>(binding_count) *
                                  sizeof(*delayed_key_state)),
                   "allocate delayed factor key state");
      delayed_key_capacity = binding_count;
      delayed_key_index_capacity = key_capacity;
    }
    if (delayed_credit_receipt == nullptr) {
      cuda_require(cudaMalloc(&delayed_credit_receipt,
                              sizeof(*delayed_credit_receipt)),
                   "allocate delayed factor-credit receipt");
      cuda_require(cudaMalloc(&delayed_credit_attempt,
                              sizeof(*delayed_credit_attempt)),
                   "allocate delayed factor-credit attempt");
      cuda_require(cudaMalloc(&delayed_credit_operation,
                              sizeof(*delayed_credit_operation)),
                   "allocate delayed factor-credit inverse result");
      cuda_require(cudaMemset(delayed_credit_receipt, 0,
                              sizeof(*delayed_credit_receipt)),
                   "clear delayed factor-credit receipt");
    }
  }

  void adapt_factor_bindings(const ResidentFactorBinding* bindings, std::uint32_t binding_count) {
    adapt_factor_bindings_kernel<<<(binding_count + 255u) / 256u, 256u>>>(bindings, factor_bindings,
                                                                          binding_count);
    cuda_require(cudaGetLastError(), "launch owner factor-binding adapter");
  }

  bcc32::device_owned_factor_credit::DeviceCreditView factor_view(
      const ResidentFactorRing& resident_factors, std::uint32_t binding_count,
      ResidentFactorClock* clock, std::uint32_t* positive_regions, std::uint32_t* negative_regions,
      std::uint32_t* matched_regions, std::uint32_t* residual_escrow,
      const WorldResidualSource* world) {
    bcc32_b3_factor_amplitude_matter::DeviceFactorRing factors{};
    for (std::uint32_t lane = 0u; lane < kFactorHorizon; ++lane)
      factors.lane[lane] = resident_factors.lane[lane];
    static_assert(
        sizeof(ResidentFactorClock) == sizeof(bcc32::device_owned_factor_credit::ResidentClock) &&
        alignof(ResidentFactorClock) == alignof(bcc32::device_owned_factor_credit::ResidentClock));
    bcc32::device_owned_factor_credit::WorldResidualView world_view{};
    if (world != nullptr) {
      world_view = {world->world,
                    world->world_words,
                    world->positive_endpoint,
                    world->negative_endpoint,
                    world->vacancy,
                    world->positive_supply,
                    world->negative_supply,
                    world->region,
                    1u};
    }
    return {engine.device_view(),
            factors,
            factor_bindings,
            binding_count,
            reinterpret_cast<bcc32::device_owned_factor_credit::ResidentClock*>(clock),
            positive_regions,
            negative_regions,
            matched_regions,
            residual_escrow,
            factor_journal,
            factor_capacity,
            factor_receipt,
            factor_attempt,
            world_view};
  }

  PagedConditionedOwnerEngine engine;
  bank::JournalEntry* factor_journal = nullptr;
  bcc32::device_owned_factor_credit::FactorRouteBinding* factor_bindings = nullptr;
  bcc32::device_owned_factor_credit::CreditReceipt* factor_receipt = nullptr;
  bcc32::device_owned_factor_credit::CreditReceipt* factor_attempt = nullptr;
  bank::OperationReceipt* factor_operation = nullptr;
  ParticipationReceiptInternal* participation_receipt = nullptr;
  ParticipationReceiptInternal* participation_attempt = nullptr;
  bank::OperationReceipt* participation_operation = nullptr;
  AtomicParticipationCreditReceipt* atomic_participation_credit_receipt =
      nullptr;
  PredictionWitnessReceiptInternal* prediction_witness_receipt = nullptr;
  bank::JournalEntry* delayed_credit_journal = nullptr;
  ExpiredEligibilityEntry* delayed_expiry_journal = nullptr;
  DelayedFactorKeyIndexEntry* delayed_key_index = nullptr;
  std::uint32_t* delayed_key_next = nullptr;
  std::uint8_t* delayed_key_state = nullptr;
  DelayedFactorCreditReceiptInternal* delayed_credit_receipt = nullptr;
  DelayedFactorCreditReceiptInternal* delayed_credit_attempt = nullptr;
  bank::OperationReceipt* delayed_credit_operation = nullptr;
  std::uint32_t factor_capacity = 0u;
  std::uint32_t delayed_event_capacity = 0u;
  std::uint32_t delayed_expiry_capacity = 0u;
  std::uint32_t delayed_key_capacity = 0u;
  std::uint32_t delayed_key_index_capacity = 0u;
  std::uint32_t* resident_factor_words = nullptr;
  ResidentFactorBinding* resident_bindings = nullptr;
  ResidentFactorClock* resident_clock = nullptr;
  std::uint32_t* resident_regions = nullptr;
  std::uint32_t* resident_eligibility_supply = nullptr;
  std::uint32_t resident_lane_words = 0u;
  std::uint32_t resident_binding_count = 0u;
};

#include "bcc32_cuda_paged_conditioned_owner_methods.inl"

}  // namespace bcc32::paged_conditioned_owner
