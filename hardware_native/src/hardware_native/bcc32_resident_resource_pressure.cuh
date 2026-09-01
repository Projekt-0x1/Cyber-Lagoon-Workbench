#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

// causal_rewrite_universe.cuh intentionally undefines its private annotation
// macro after its own declarations.  This independently includable donor has
// its own equivalent annotation rather than depending on that implementation
// detail.
#if defined(__CUDACC__)
#define BCC32_RESIDENT_RESOURCE_HD __host__ __device__
#else
#define BCC32_RESIDENT_RESOURCE_HD
#endif

// Resident structural-pressure matter for the RWR0 adult.
//
// This is deliberately a small vocabulary of ordinary Record forms, not a
// scalar budget, host reward, objective, action writer, or semantic authority
// path. Every quantity below is derived by scanning Record::matter_q8 in the
// resident world or the existing physical-lesion escrow. A caller cannot pass
// availability into any predicate or transaction.
namespace substrate::bcc32::resident_resource_pressure {

namespace rewrite = causal_rewrite;

inline constexpr std::uint32_t kFormFreeMatter = 0x5a7c34e1u;
inline constexpr std::uint32_t kFormCommittedMatter = 0x93d1e746u;
inline constexpr std::uint32_t kFormDamageMatter = 0x2f86b95cu;
inline constexpr std::uint32_t kFormRepairEscrow = 0xc4a16d73u;
inline constexpr std::uint32_t kLesionRecordCapacity = 8u;
inline constexpr std::uint32_t kMaximumChargedExtent = 8u;

enum class PaymentPurpose : std::uint32_t {
  support_increment = 1u,
  program_construction = 2u,
  span_program_construction = 3u,
  action_proposal = 4u,
};

struct MatterSnapshot {
  std::uint64_t free_q8 = 0u;
  std::uint64_t committed_q8 = 0u;
  std::uint64_t damage_q8 = 0u;
  std::uint64_t escrow_q8 = 0u;
};

struct Payment {
  std::uint32_t owner = rewrite::kInvalid;
  std::uint32_t selected_extent = 0u;
  std::uint32_t charged_q8 = 0u;
  PaymentPurpose purpose = PaymentPurpose::support_increment;
  std::uint32_t slots[kMaximumChargedExtent]{};
  std::uint32_t transaction_tag = 0u;
  std::uint64_t pre_commit_revision = 0u;
  std::uint64_t commit_revision = 0u;
};

// This is a resident pre-commit observation.  The canonical selector supplies
// only the extent it has already selected; this donor neither names nor picks
// an action, word, program, or public target.
struct AffordabilityRead {
  std::uint32_t owner = rewrite::kInvalid;
  std::uint32_t selected_extent = 0u;
  std::uint32_t charged_q8 = 0u;
  PaymentPurpose purpose = PaymentPurpose::support_increment;
  std::uint32_t slots[kMaximumChargedExtent]{};
  std::int32_t pre_margin_q8 = 0;
  std::uint64_t state_revision = 0u;
};

// The canonical resident path must create this only after the charged public
// or learning commitment has a later resident consequence.  The tag binds one
// consequence to one recorded charge; it is not a source, selector, or credit.
struct ResidentConsequence {
  std::uint32_t owner = rewrite::kInvalid;
  std::uint32_t actual_extent = 0u;
  PaymentPurpose purpose = PaymentPurpose::support_increment;
  std::uint32_t transaction_tag = 0u;
  std::uint64_t commit_revision = 0u;
  std::uint64_t resident_tick = 0u;
  std::uint64_t consequence_id = 0u;
};

BCC32_RESIDENT_RESOURCE_HD inline bool valid_payment_purpose(
    PaymentPurpose purpose) {
  return purpose == PaymentPurpose::support_increment ||
         purpose == PaymentPurpose::program_construction ||
         purpose == PaymentPurpose::span_program_construction ||
         purpose == PaymentPurpose::action_proposal;
}

BCC32_RESIDENT_RESOURCE_HD inline std::uint64_t saturating_add_u64(
    std::uint64_t left, std::uint64_t right) {
  constexpr std::uint64_t kU64Max = ~std::uint64_t{0};
  return left > kU64Max - right ? kU64Max : left + right;
}

// This is intentionally independently testable at full integer endpoints.
// Resident scans are bounded well below these limits, but the arithmetic must
// remain fail-closed if a corrupt or future wider record account reaches them.
BCC32_RESIDENT_RESOURCE_HD inline std::int32_t saturating_signed_margin_q8(
    std::uint64_t free_q8, std::uint64_t committed_q8,
    std::uint64_t damage_q8, std::uint64_t escrow_q8) {
  const std::uint64_t debit = saturating_add_u64(
      saturating_add_u64(committed_q8, damage_q8), escrow_q8);
  constexpr std::uint64_t kI32Max = 0x7fffffffu;
  constexpr std::uint64_t kI32Magnitude = kI32Max + 1u;
  if (free_q8 >= debit) {
    const std::uint64_t surplus = free_q8 - debit;
    return static_cast<std::int32_t>(surplus > kI32Max ? kI32Max : surplus);
  }
  const std::uint64_t deficit = debit - free_q8;
  if (deficit >= kI32Magnitude) return (-2147483647 - 1);
  return -static_cast<std::int32_t>(deficit);
}

BCC32_RESIDENT_RESOURCE_HD inline bool valid_owner(std::uint32_t owner) {
  return owner != rewrite::kInvalid;
}

BCC32_RESIDENT_RESOURCE_HD inline bool lesion_is_bounded(
    const rewrite::ResidentRewriteState& state) {
  return state.lesion.count <= kLesionRecordCapacity;
}

BCC32_RESIDENT_RESOURCE_HD inline bool live_owned_form(const rewrite::Record& record,
                                              std::uint32_t form,
                                              std::uint32_t owner) {
  return record.matter_q8 != 0u && record.lane[0] == form &&
         record.lane[1] == owner;
}

BCC32_RESIDENT_RESOURCE_HD inline void add_form_matter(MatterSnapshot* snapshot,
                                             const rewrite::Record& record,
                                             std::uint32_t owner) {
  if (record.matter_q8 == 0u || record.lane[1] != owner) return;
  if (record.lane[0] == kFormFreeMatter) {
    snapshot->free_q8 = saturating_add_u64(snapshot->free_q8,
                                            record.matter_q8);
  } else if (record.lane[0] == kFormCommittedMatter) {
    snapshot->committed_q8 = saturating_add_u64(snapshot->committed_q8,
                                                 record.matter_q8);
  } else if (record.lane[0] == kFormDamageMatter) {
    snapshot->damage_q8 = saturating_add_u64(snapshot->damage_q8,
                                              record.matter_q8);
  } else if (record.lane[0] == kFormRepairEscrow) {
    snapshot->escrow_q8 = saturating_add_u64(snapshot->escrow_q8,
                                              record.matter_q8);
  }
}

BCC32_RESIDENT_RESOURCE_HD inline MatterSnapshot snapshot(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  MatterSnapshot result{};
  if (!valid_owner(owner) || !lesion_is_bounded(state)) return result;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&state);
       ++slot)
    add_form_matter(&result, state.records[slot], owner);
  for (std::uint32_t entry = 0u; entry < state.lesion.count; ++entry)
    add_form_matter(&result, state.lesion.displaced[entry], owner);
  return result;
}

BCC32_RESIDENT_RESOURCE_HD inline std::int32_t structural_margin_q8(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  const MatterSnapshot matter = snapshot(state, owner);
  return saturating_signed_margin_q8(matter.free_q8, matter.committed_q8,
                                     matter.damage_q8, matter.escrow_q8);
}

BCC32_RESIDENT_RESOURCE_HD inline std::uint32_t find_owned_form(
    const rewrite::ResidentRewriteState& state, std::uint32_t form,
    std::uint32_t owner) {
  if (!valid_owner(owner)) return rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&state);
       ++slot)
    if (live_owned_form(state.records[slot], form, owner)) return slot;
  return rewrite::kInvalid;
}

BCC32_RESIDENT_RESOURCE_HD inline void increment_revision(rewrite::Record* record) {
  if (record->revision != 0xffffffffu) ++record->revision;
}

BCC32_RESIDENT_RESOURCE_HD inline void increment_revision(
    rewrite::ResidentRewriteState* state) {
  if (state->revision != ~std::uint64_t{0}) ++state->revision;
}

BCC32_RESIDENT_RESOURCE_HD inline void write_owned_form(rewrite::Record* record,
                                               std::uint32_t form,
                                               std::uint32_t owner) {
  const std::uint32_t matter = record->matter_q8;
  const std::uint32_t revision = record->revision;
  *record = rewrite::Record{};
  record->lane[0] = form;
  record->lane[1] = owner;
  record->lane[2] = matter;
  record->matter_q8 = matter;
  record->revision = revision;
  increment_revision(record);
}

// Minting is a named conversion of one actually allocatable empty Record. It
// does not create matter; capacity failure leaves the complete state unchanged.
BCC32_RESIDENT_RESOURCE_HD inline bool mint_free_matter(
    rewrite::ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || !valid_owner(owner) || state->fault != 0u) return false;
  if (!lesion_is_bounded(*state)) return false;
  bool has_allocatable_empty = false;
  for (std::uint32_t probe = 0u;
       probe < rewrite::live_record_capacity(state); ++probe) {
    const rewrite::Record& candidate = state->records[probe];
    if (candidate.matter_q8 != 0u && candidate.lane[0] == rewrite::kFormEmpty) {
      has_allocatable_empty = true;
      break;
    }
  }
  if (!has_allocatable_empty) return false;
  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;
  write_owned_form(&state->records[slot], kFormFreeMatter, owner);
  increment_revision(state);
  return true;
}

// Read affordability before the canonical path commits its already-selected
// extent.  The transaction is fail-closed: a nonpositive margin, a mismatched
// extent, or fewer whole free records than selected yields no receipt.
BCC32_RESIDENT_RESOURCE_HD inline bool read_affordability(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner,
    std::uint32_t selected_extent, PaymentPurpose purpose,
    AffordabilityRead* read) {
  if (read == nullptr || !valid_owner(owner) || !lesion_is_bounded(state) ||
      !valid_payment_purpose(purpose) || selected_extent == 0u ||
      selected_extent > kMaximumChargedExtent)
    return false;
  const std::uint64_t charged = static_cast<std::uint64_t>(selected_extent) *
                                rewrite::kRecordMatterQ8;
  const std::int32_t margin = structural_margin_q8(state, owner);
  if (margin <= 0 || static_cast<std::uint64_t>(margin) < charged) return false;

  AffordabilityRead candidate{};
  candidate.owner = owner;
  candidate.selected_extent = selected_extent;
  candidate.charged_q8 = static_cast<std::uint32_t>(charged);
  candidate.purpose = purpose;
  candidate.pre_margin_q8 = margin;
  candidate.state_revision = state.revision;
  std::uint32_t found = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&state);
       ++slot) {
    if (!live_owned_form(state.records[slot], kFormFreeMatter, owner) ||
        state.records[slot].matter_q8 != rewrite::kRecordMatterQ8)
      continue;
    candidate.slots[found++] = slot;
    if (found == selected_extent) {
      *read = candidate;
      return true;
    }
  }
  return false;
}

BCC32_RESIDENT_RESOURCE_HD inline bool can_pay_exact(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner,
    std::uint32_t matter_q8) {
  if (matter_q8 != rewrite::kRecordMatterQ8) return false;
  AffordabilityRead ignored{};
  return read_affordability(state, owner, 1u,
                            PaymentPurpose::support_increment, &ignored);
}

BCC32_RESIDENT_RESOURCE_HD inline bool can_pay_support_increment(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  return can_pay_exact(state, owner, rewrite::kRecordMatterQ8);
}

BCC32_RESIDENT_RESOURCE_HD inline bool can_pay_program_construction(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  return can_pay_exact(state, owner, rewrite::kRecordMatterQ8);
}

BCC32_RESIDENT_RESOURCE_HD inline bool can_pay_span_program_construction(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  return can_pay_exact(state, owner, rewrite::kRecordMatterQ8);
}

BCC32_RESIDENT_RESOURCE_HD inline bool can_pay_action_proposal(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  return can_pay_exact(state, owner, rewrite::kRecordMatterQ8);
}

// Commit exactly the extent whose affordability was read.  A stale read,
// forged owner/purpose/cost, or changed free record rejects before any Record
// mutation.  It only converts represented matter; it never writes semantics.
BCC32_RESIDENT_RESOURCE_HD inline bool commit_selected_extent(
    rewrite::ResidentRewriteState* state, const AffordabilityRead& read,
    Payment* payment) {
  if (state == nullptr || payment == nullptr || state->fault != 0u ||
      !valid_owner(read.owner) || !lesion_is_bounded(*state) ||
      !valid_payment_purpose(read.purpose) ||
      read.selected_extent == 0u || read.selected_extent > kMaximumChargedExtent ||
      read.charged_q8 != read.selected_extent * rewrite::kRecordMatterQ8 ||
      read.state_revision != state->revision)
    return false;
  if (structural_margin_q8(*state, read.owner) <= 0 ||
      static_cast<std::uint64_t>(structural_margin_q8(*state, read.owner)) <
          read.charged_q8)
    return false;
  for (std::uint32_t i = 0u; i < read.selected_extent; ++i) {
    if (read.slots[i] >= rewrite::live_record_capacity(state) ||
        !live_owned_form(state->records[read.slots[i]], kFormFreeMatter,
                         read.owner) ||
        state->records[read.slots[i]].matter_q8 != rewrite::kRecordMatterQ8)
      return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (read.slots[prior] == read.slots[i]) return false;
  }

  Payment committed{};
  committed.owner = read.owner;
  committed.selected_extent = read.selected_extent;
  committed.charged_q8 = read.charged_q8;
  committed.purpose = read.purpose;
  committed.pre_commit_revision = state->revision;
  committed.commit_revision = state->revision == ~std::uint64_t{0}
                                  ? state->revision
                                  : state->revision + 1u;
  committed.transaction_tag = rewrite::rewrite_mix(
      read.owner ^ read.selected_extent,
      static_cast<std::uint32_t>(read.purpose),
      static_cast<std::uint32_t>(state->revision));
  if (committed.transaction_tag == 0u) committed.transaction_tag = 1u;
  for (std::uint32_t i = 0u; i < read.selected_extent; ++i) {
    committed.slots[i] = read.slots[i];
    rewrite::Record& record = state->records[read.slots[i]];
    write_owned_form(&record, kFormCommittedMatter, read.owner);
    record.lane[3] = static_cast<std::uint32_t>(read.purpose);
    record.lane[4] = committed.transaction_tag;
    record.lane[5] = static_cast<std::uint32_t>(committed.commit_revision);
    record.lane[6] = static_cast<std::uint32_t>(committed.commit_revision >> 32u);
  }
  increment_revision(state);
  *payment = committed;
  return true;
}

// Refund is not a credit source.  It returns the exact committed forms only
// after one later matching resident consequence; absent, forged, mismatched,
// or repeated consequences leave state byte-identical.
BCC32_RESIDENT_RESOURCE_HD inline bool refund_after_consequence(
    rewrite::ResidentRewriteState* state, const Payment& payment,
    const ResidentConsequence& consequence) {
  if (state == nullptr || state->fault != 0u || !lesion_is_bounded(*state) ||
      !valid_owner(payment.owner) || payment.selected_extent == 0u ||
      payment.selected_extent > kMaximumChargedExtent ||
      payment.charged_q8 != payment.selected_extent * rewrite::kRecordMatterQ8 ||
      payment.transaction_tag == 0u || consequence.owner != payment.owner ||
      consequence.actual_extent != payment.selected_extent ||
      consequence.purpose != payment.purpose ||
      !valid_payment_purpose(payment.purpose) ||
      consequence.transaction_tag != payment.transaction_tag ||
      consequence.commit_revision != payment.commit_revision ||
      consequence.consequence_id == 0u ||
      consequence.resident_tick <= payment.commit_revision)
    return false;
  for (std::uint32_t i = 0u; i < payment.selected_extent; ++i) {
    if (payment.slots[i] >= rewrite::live_record_capacity(state) ||
        !live_owned_form(state->records[payment.slots[i]],
                         kFormCommittedMatter, payment.owner) ||
        state->records[payment.slots[i]].matter_q8 != rewrite::kRecordMatterQ8 ||
        state->records[payment.slots[i]].lane[3] !=
            static_cast<std::uint32_t>(payment.purpose) ||
        state->records[payment.slots[i]].lane[4] != payment.transaction_tag ||
        state->records[payment.slots[i]].lane[5] !=
            static_cast<std::uint32_t>(payment.commit_revision) ||
        state->records[payment.slots[i]].lane[6] !=
            static_cast<std::uint32_t>(payment.commit_revision >> 32u))
      return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (payment.slots[prior] == payment.slots[i]) return false;
  }
  for (std::uint32_t i = 0u; i < payment.selected_extent; ++i)
    write_owned_form(&state->records[payment.slots[i]], kFormFreeMatter,
                     payment.owner);
  increment_revision(state);
  return true;
}

// Escrow is represented by consuming one owned free Record. It can repair one
// focal displaced free Record exactly once; it is not a host refill.
BCC32_RESIDENT_RESOURCE_HD inline bool stage_repair_escrow(
    rewrite::ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || state->fault != 0u ||
      !can_pay_exact(*state, owner, rewrite::kRecordMatterQ8))
    return false;
  const std::uint32_t slot = find_owned_form(*state, kFormFreeMatter, owner);
  write_owned_form(&state->records[slot], kFormRepairEscrow, owner);
  increment_revision(state);
  return true;
}

// Classify only a Record already displaced by the existing physical lesion
// transaction. No caller chooses a semantic object: the relation owner is
// carried by the displaced free-matter Record itself.
BCC32_RESIDENT_RESOURCE_HD inline bool classify_focal_damage(
    rewrite::ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || state->fault != 0u || !valid_owner(owner) ||
      !lesion_is_bounded(*state))
    return false;
  for (std::uint32_t entry = 0u; entry < state->lesion.count; ++entry) {
    rewrite::Record& record = state->lesion.displaced[entry];
    if (!live_owned_form(record, kFormFreeMatter, owner)) continue;
    write_owned_form(&record, kFormDamageMatter, owner);
    increment_revision(state);
    return true;
  }
  return false;
}

BCC32_RESIDENT_RESOURCE_HD inline std::uint32_t find_damage_entry(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  if (!valid_owner(owner) || !lesion_is_bounded(state)) return rewrite::kInvalid;
  for (std::uint32_t entry = 0u; entry < state.lesion.count; ++entry)
    if (live_owned_form(state.lesion.displaced[entry], kFormDamageMatter, owner))
      return entry;
  return rewrite::kInvalid;
}

BCC32_RESIDENT_RESOURCE_HD inline void erase_lesion_entry(
    rewrite::ResidentRewriteState* state, std::uint32_t entry) {
  const std::uint32_t last = state->lesion.count - 1u;
  if (entry != last) {
    state->lesion.displaced[entry] = state->lesion.displaced[last];
    state->lesion.original_slot[entry] = state->lesion.original_slot[last];
  }
  state->lesion.displaced[last] = rewrite::Record{};
  state->lesion.displaced[last].matter_q8 = 0u;
  state->lesion.original_slot[last] = 0u;
  --state->lesion.count;
}

BCC32_RESIDENT_RESOURCE_HD inline bool repair_focal_damage(
    rewrite::ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || state->fault != 0u || !valid_owner(owner) ||
      !lesion_is_bounded(*state))
    return false;
  const std::uint32_t escrow = find_owned_form(*state, kFormRepairEscrow, owner);
  const std::uint32_t damage = find_damage_entry(*state, owner);
  if (escrow == rewrite::kInvalid || damage == rewrite::kInvalid) return false;
  const std::uint32_t original_slot = state->lesion.original_slot[damage];
  const rewrite::Record displaced = state->lesion.displaced[damage];
  if (original_slot >= rewrite::live_record_capacity(state) ||
      state->records[original_slot].matter_q8 != 0u ||
      state->records[original_slot].lane[0] != 0u ||
      displaced.matter_q8 == 0u ||
      state->records[escrow].matter_q8 != displaced.matter_q8 ||
      state->lesion.removed_matter_q8 < displaced.matter_q8)
    return false;

  rewrite::Record restored = displaced;
  write_owned_form(&restored, kFormFreeMatter, owner);
  state->records[original_slot] = restored;
  rewrite::clear_record(&state->records[escrow]);
  state->lesion.removed_matter_q8 -= displaced.matter_q8;
  erase_lesion_entry(state, damage);
  increment_revision(state);
  return true;
}

// A relation-insensitive digest permits allocation-permutation controls only
// where ownership is already carried by the records. It is not authority.
BCC32_RESIDENT_RESOURCE_HD inline std::uint64_t ownership_digest(
    const rewrite::ResidentRewriteState& state, std::uint32_t owner) {
  const MatterSnapshot matter = snapshot(state, owner);
  const std::uint32_t left = rewrite::rewrite_mix(
      static_cast<std::uint32_t>(matter.free_q8),
      static_cast<std::uint32_t>(matter.committed_q8), owner);
  const std::uint32_t right = rewrite::rewrite_mix(
      static_cast<std::uint32_t>(matter.damage_q8),
      static_cast<std::uint32_t>(matter.escrow_q8), owner);
  return (static_cast<std::uint64_t>(left) << 32u) | right;
}

}  // namespace substrate::bcc32::resident_resource_pressure

#undef BCC32_RESIDENT_RESOURCE_HD
