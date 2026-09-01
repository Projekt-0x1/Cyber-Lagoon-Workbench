#pragma once
#include "causal_rewrite_universe.cuh"

#if defined(__CUDACC__)
#define BCC32_RECURRENT_CARRIER_HD __host__ __device__
#else
#define BCC32_RECURRENT_CARRIER_HD
#endif

namespace substrate::bcc32::resident_recurrent_carrier {
namespace rewrite = substrate::bcc32::causal_rewrite;

// Route fragment:
// lane[0] form
// lane[1] route owner (opaque resident owner; never a concept id)
// lane[2] source state key (locator only)
// lane[3] target state key (locator only)
// lane[4] remaining-delay class earned by topology; Cloud-1 fixtures author it
// lane[5] strength_q8 (matter/competition weight, not output confidence)
// lane[6] source revision / topology generation
// lane[7] zero; reserved[] zero
//
// Carrier:
// lane[0] form
// lane[1] carrier owner
// lane[2] current state key (locator only)
// lane[3] local generation age/epoch count; reset on every successful hop
// lane[4] remaining delay
// lane[5] local transient energy_q8
// lane[6] originating resident revision / lineage provenance
// lane[7] zero; reserved[] zero
//
// Record::revision distinguishes successive local carrier generations. A
// successful route traversal consumes the current generation and constructs
// a successor generation at the target. The first successor may recycle the
// same physical Record slot, but it receives a fresh revision and age zero;
// therefore storage reuse is not carrier-lifetime continuity. lane[6]
// deliberately survives the replacement so causal lineage remains intact.

struct AdvanceReceipt {
  std::uint32_t carriers_before = 0u;
  std::uint32_t carriers_after = 0u;
  std::uint32_t advanced = 0u;
  std::uint32_t branched = 0u;
  std::uint32_t renewed = 0u;
  std::uint32_t extinguished = 0u;
  std::uint32_t malformed = 0u;
};

BCC32_RECURRENT_CARRIER_HD inline bool is_route_fragment(const rewrite::Record& r) {
  return r.matter_q8 != 0u &&
         r.lane[0] == rewrite::kFormRecurrentRouteFragment &&
         r.lane[1] != 0u && r.lane[1] != rewrite::kInvalid &&
         r.lane[2] != 0u && r.lane[2] != rewrite::kInvalid &&
         r.lane[3] != 0u && r.lane[3] != rewrite::kInvalid &&
         r.lane[5] != 0u && r.lane[7] == 0u &&
         r.reserved[0] == 0u && r.reserved[1] == 0u;
}

BCC32_RECURRENT_CARRIER_HD inline bool is_carrier(const rewrite::Record& r) {
  return r.matter_q8 != 0u &&
         r.lane[0] == rewrite::kFormRecurrentCarrier &&
         r.lane[1] != 0u && r.lane[1] != rewrite::kInvalid &&
         r.lane[2] != 0u && r.lane[2] != rewrite::kInvalid &&
         r.lane[5] != 0u && r.lane[7] == 0u &&
         r.reserved[0] == 0u && r.reserved[1] == 0u;
}

// Fixture-only: test fixtures may seed route fragments directly to exercise
// the transport engine below. Production raw experience must NOT call this,
// and hand-seeding a topology as the mechanism gate is explicitly RED for
// 0X1-189/0X1-190 as of their 2026-08-15 architecture correction -- these
// seeders exist to prove the transport engine, never to stand in for grown
// topology.
BCC32_RECURRENT_CARRIER_HD inline bool seed_route_fragment_for_contract(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t from, std::uint32_t to, std::uint32_t delay,
    std::uint32_t strength_q8) {
  if (state == nullptr || owner == 0u || owner == rewrite::kInvalid ||
      from == 0u || from == rewrite::kInvalid ||
      to == 0u || to == rewrite::kInvalid || strength_q8 == 0u)
    return false;
  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;
  rewrite::Record& r = state->records[slot];
  const std::uint32_t revision = r.revision + 1u;
  r = rewrite::Record{};
  r.lane[0] = rewrite::kFormRecurrentRouteFragment;
  r.lane[1] = owner;
  r.lane[2] = from;
  r.lane[3] = to;
  r.lane[4] = delay;
  r.lane[5] = strength_q8;
  r.lane[6] = static_cast<std::uint32_t>(state->revision);
  r.revision = revision;
  return true;
}

BCC32_RECURRENT_CARRIER_HD inline bool seed_carrier_for_contract(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t state_key, std::uint32_t energy_q8) {
  if (state == nullptr || owner == 0u || owner == rewrite::kInvalid ||
      state_key == 0u || state_key == rewrite::kInvalid || energy_q8 == 0u)
    return false;
  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;
  rewrite::Record& r = state->records[slot];
  const std::uint32_t revision = r.revision + 1u;
  r = rewrite::Record{};
  r.lane[0] = rewrite::kFormRecurrentCarrier;
  r.lane[1] = owner;
  r.lane[2] = state_key;
  r.lane[5] = energy_q8;
  r.lane[6] = static_cast<std::uint32_t>(state->revision);
  r.revision = revision;
  return true;
}

// Bounded scratch: fixture topologies are deliberately small. A carrier
// mid-transit (remaining delay > 0) only decrements its local delay and ages
// that one local generation. Energy is spent only when a resting carrier
// actually traverses a route; transit delay therefore remains transport
// latency rather than an implicit global lifetime threshold.
//
// At rest, the carrier looks up every live route fragment sharing its owner
// and current state key. Zero matches extinguishes that local transient.
// Each match consumes the current generation and constructs a successor
// generation at the target. One q8 energy quantum is spent by the hop, while
// the traversed route locally renews the successor to at least its own
// strength_q8:
//
//   successor_energy = max(incoming_energy - 1, route_strength_q8)
//
// Thus static recurrent route matter can mutually renew finite carrier
// generations without a host recurrence counter, global persistence bit, or
// semantic teaching signal. The first successor recycles the parent's Record
// storage with a fresh Record::revision and age zero; additional outgoing
// routes allocate sibling generations. lane[6] remains the originating
// resident revision for the complete causal lineage.
//
// Matching routes are read-only in this pass. Newly produced carrier
// generations are not in the opening snapshot and therefore cannot advance
// again in the same epoch.
inline constexpr std::uint32_t kRecurrentCarrierScratchCapacity = 64u;

BCC32_RECURRENT_CARRIER_HD inline std::uint32_t renew_carrier_energy_q8(
    std::uint32_t incoming_energy_q8, std::uint32_t route_strength_q8) {
  if (incoming_energy_q8 == 0u || route_strength_q8 == 0u) return 0u;
  const std::uint32_t residual = incoming_energy_q8 - 1u;
  return residual > route_strength_q8 ? residual : route_strength_q8;
}

BCC32_RECURRENT_CARRIER_HD inline AdvanceReceipt advance_recurrent_carriers(
    rewrite::ResidentRewriteState* state) {
  AdvanceReceipt receipt{};
  if (state == nullptr) return receipt;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);

  std::uint32_t carrier_slot[kRecurrentCarrierScratchCapacity]{};
  std::uint32_t carrier_owner[kRecurrentCarrierScratchCapacity]{};
  std::uint32_t carrier_revision[kRecurrentCarrierScratchCapacity]{};
  std::uint32_t carrier_count = 0u;
  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& record = state->records[slot];
    if (!is_carrier(record)) continue;
    ++receipt.carriers_before;
    if (carrier_count < kRecurrentCarrierScratchCapacity) {
      carrier_slot[carrier_count] = slot;
      carrier_owner[carrier_count] = record.lane[1];
      carrier_revision[carrier_count] = record.revision;
      ++carrier_count;
    }
  }

  // Logical-tuple order, not slot order: (owner, revision). A permuted
  // Record allocation must still process every carrier in the same order.
  for (std::uint32_t i = 1u; i < carrier_count; ++i) {
    const std::uint32_t slot = carrier_slot[i];
    const std::uint32_t owner = carrier_owner[i];
    const std::uint32_t revision = carrier_revision[i];
    std::uint32_t j = i;
    while (j > 0u &&
           (carrier_owner[j - 1u] > owner ||
            (carrier_owner[j - 1u] == owner &&
             carrier_revision[j - 1u] > revision))) {
      carrier_slot[j] = carrier_slot[j - 1u];
      carrier_owner[j] = carrier_owner[j - 1u];
      carrier_revision[j] = carrier_revision[j - 1u];
      --j;
    }
    carrier_slot[j] = slot;
    carrier_owner[j] = owner;
    carrier_revision[j] = revision;
  }

  for (std::uint32_t i = 0u; i < carrier_count; ++i) {
    const std::uint32_t slot = carrier_slot[i];
    rewrite::Record& carrier = state->records[slot];
    // Revalidate: this pass only mutates carrier records it has not yet
    // visited, and never touches route fragments, so a not-yet-visited
    // carrier's own slot cannot have changed underneath it. This check is
    // a cheap invariant guard, not a real race in this bounded single-pass
    // algorithm.
    if (!is_carrier(carrier) || carrier.revision != carrier_revision[i]) {
      ++receipt.malformed;
      continue;
    }
    if (carrier.lane[4] != 0u) {
      carrier.lane[4] -= 1u;
      carrier.lane[3] += 1u;
      ++carrier.revision;
      ++receipt.advanced;
      continue;
    }
    const std::uint32_t owner = carrier.lane[1];
    const std::uint32_t current_key = carrier.lane[2];
    const std::uint32_t energy_q8 = carrier.lane[5];
    const std::uint32_t origin_revision = carrier.lane[6];

    std::uint32_t route_target[kRecurrentCarrierScratchCapacity]{};
    std::uint32_t route_delay[kRecurrentCarrierScratchCapacity]{};
    std::uint32_t route_strength[kRecurrentCarrierScratchCapacity]{};
    std::uint32_t route_revision[kRecurrentCarrierScratchCapacity]{};
    std::uint32_t route_count = 0u;
    for (std::uint32_t rslot = 0u; rslot < capacity; ++rslot) {
      const rewrite::Record& route = state->records[rslot];
      if (!is_route_fragment(route) || route.lane[1] != owner ||
          route.lane[2] != current_key)
        continue;
      if (route_count < kRecurrentCarrierScratchCapacity) {
        route_target[route_count] = route.lane[3];
        route_delay[route_count] = route.lane[4];
        route_strength[route_count] = route.lane[5];
        route_revision[route_count] = route.revision;
        ++route_count;
      }
    }
    // Deterministic order among sibling routes at the same source: sort by
    // (target, revision), independent of Record allocation slot order.
    for (std::uint32_t a = 1u; a < route_count; ++a) {
      const std::uint32_t target = route_target[a];
      const std::uint32_t delay = route_delay[a];
      const std::uint32_t strength = route_strength[a];
      const std::uint32_t revision = route_revision[a];
      std::uint32_t b = a;
      while (b > 0u &&
             (route_target[b - 1u] > target ||
              (route_target[b - 1u] == target &&
               route_revision[b - 1u] > revision))) {
        route_target[b] = route_target[b - 1u];
        route_delay[b] = route_delay[b - 1u];
        route_strength[b] = route_strength[b - 1u];
        route_revision[b] = route_revision[b - 1u];
        --b;
      }
      route_target[b] = target;
      route_delay[b] = delay;
      route_strength[b] = strength;
      route_revision[b] = revision;
    }

    if (route_count == 0u) {
      rewrite::clear_record(&carrier);
      ++receipt.extinguished;
      continue;
    }

    // The old local carrier generation ends here. Reusing its storage for the
    // first successor avoids turning ordinary renewal into allocator
    // pressure, but revision+age make the generation boundary explicit.
    const std::uint32_t first_revision = carrier.revision + 1u;
    const std::uint32_t first_energy =
        renew_carrier_energy_q8(energy_q8, route_strength[0u]);
    if (first_energy == 0u) {
      rewrite::clear_record(&carrier);
      ++receipt.extinguished;
      continue;
    }
    carrier = rewrite::Record{};
    carrier.lane[0] = rewrite::kFormRecurrentCarrier;
    carrier.lane[1] = owner;
    carrier.lane[2] = route_target[0u];
    carrier.lane[3] = 0u;
    carrier.lane[4] = route_delay[0u];
    carrier.lane[5] = first_energy;
    carrier.lane[6] = origin_revision;
    carrier.revision = first_revision;
    ++receipt.advanced;
    ++receipt.renewed;

    for (std::uint32_t k = 1u; k < route_count; ++k) {
      const std::uint32_t branch_energy =
          renew_carrier_energy_q8(energy_q8, route_strength[k]);
      if (branch_energy == 0u) continue;
      const std::uint32_t new_slot = rewrite::allocate_record(state);
      if (new_slot == rewrite::kInvalid) continue;
      rewrite::Record& branch = state->records[new_slot];
      const std::uint32_t branch_revision = branch.revision + 1u;
      branch = rewrite::Record{};
      branch.lane[0] = rewrite::kFormRecurrentCarrier;
      branch.lane[1] = owner;
      branch.lane[2] = route_target[k];
      branch.lane[3] = 0u;
      branch.lane[4] = route_delay[k];
      branch.lane[5] = branch_energy;
      branch.lane[6] = origin_revision;
      branch.revision = branch_revision;
      ++receipt.branched;
      ++receipt.renewed;
    }
  }

  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    if (is_carrier(state->records[slot])) ++receipt.carriers_after;
  }
  return receipt;
}

}  // namespace substrate::bcc32::resident_recurrent_carrier

#undef BCC32_RECURRENT_CARRIER_HD
