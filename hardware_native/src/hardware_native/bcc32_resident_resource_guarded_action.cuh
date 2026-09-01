#pragma once

#include "bcc32_persistent_kernel.hpp"
#include "bcc32_resident_resource_pressure.cuh"

#include <cstdint>

// A deliberately narrow adapter between resident structural affordability and
// the existing opaque action-return ticket.  It does not issue a ticket,
// choose an action, select a word, or interpret a return.  The canonical
// writer must obtain the ticket and later validate raw return provenance.
namespace substrate::bcc32::resident_resource_guarded_action {

namespace rewrite = causal_rewrite;
namespace pressure = resident_resource_pressure;
namespace kernel = persistent_kernel;

#if defined(__CUDACC__)
#define BCC32_GUARDED_ACTION_HD __host__ __device__
#else
#define BCC32_GUARDED_ACTION_HD
#endif

struct GuardedAction {
  kernel::ActionReturnTicket ticket{};
  pressure::AffordabilityRead affordability{};
  pressure::Payment payment{};
  std::uint32_t ticket_binding = 0u;
};

struct GuardedActionConsequence {
  kernel::TickReceipt accepted_return{};
  std::uint64_t accepted_before = 0u;
  pressure::ResidentConsequence resident{};
};

// The adapter exposes only what an existing canonical egress writer already
// owns: its opaque ticket and the selected physical extent.  It intentionally
// carries no word, target, reward, label, or alternate output buffer.
struct CanonicalTicketedActionView {
  kernel::ActionReturnTicket ticket{};
  std::uint32_t selected_extent = 0u;
  pressure::PaymentPurpose purpose = pressure::PaymentPurpose::action_proposal;
};

// The canonical writer returns this opaque publication witness after deciding
// to expose the same ticketed action.  It carries no action word or semantic
// target: only existence, ticket identity, and the canonical public sequence.
struct CanonicalPublicActionReceipt {
  kernel::ActionReturnTicket ticket{};
  std::uint64_t public_sequence = 0u;
  std::uint32_t action_present = 0u;
};

// The canonical language publisher may attach this opaque receipt when the
// same ticketed public action carries a sentence it has already learned and
// completed.  No bytes, tokens, answer text, or semantic role crosses this
// adapter, so resource pressure cannot author language.
struct CanonicalLearnedSentenceReceipt {
  kernel::ActionReturnTicket ticket{};
  std::uint64_t public_sequence = 0u;
  std::uint64_t learned_sentence_sequence = 0u;
  std::uint32_t action_present = 0u;
  std::uint32_t complete_sentence_present = 0u;
};

// A passive view of the current canonical accepted-return receipt.  The
// callback-based adapter does not synthesize it; ingress must provide it only
// after its own raw-ticket validation and contact digest commit.
struct AcceptedReturnView {
  const kernel::TickReceipt* receipt = nullptr;
  std::uint64_t accepted_before = 0u;
};

BCC32_GUARDED_ACTION_HD inline bool valid_ticket(
    const kernel::ActionReturnTicket& ticket) {
  return ticket.action_sequence != 0u && ticket.nonce != 0u;
}

BCC32_GUARDED_ACTION_HD inline bool same_ticket(
    const kernel::ActionReturnTicket& left,
    const kernel::ActionReturnTicket& right) {
  return left.action_sequence == right.action_sequence &&
         left.nonce == right.nonce;
}

// Action sequence is the current runtime's unique device-issued ticket
// identity.  Store it on every charged Record so the same issued action
// cannot acquire a second charge before its return path resolves.
BCC32_GUARDED_ACTION_HD inline std::uint32_t ticket_sequence_low(
    const kernel::ActionReturnTicket& ticket) {
  return static_cast<std::uint32_t>(ticket.action_sequence);
}

BCC32_GUARDED_ACTION_HD inline std::uint32_t ticket_sequence_high(
    const kernel::ActionReturnTicket& ticket) {
  return static_cast<std::uint32_t>(ticket.action_sequence >> 32u);
}

BCC32_GUARDED_ACTION_HD inline std::uint32_t ticket_binding(
    const kernel::ActionReturnTicket& ticket,
    const pressure::AffordabilityRead& read) {
  return rewrite::rewrite_mix(
      ticket_sequence_low(ticket) ^ static_cast<std::uint32_t>(ticket.nonce),
      ticket_sequence_high(ticket) ^ static_cast<std::uint32_t>(ticket.nonce >> 32u),
      read.owner ^ read.selected_extent ^
          static_cast<std::uint32_t>(read.purpose) ^
          static_cast<std::uint32_t>(read.state_revision));
}

// A returned/refunded ticket remains marked on its restored free matter, so a
// replay cannot purchase a second action.  The existing ticket issuer remains
// the authority for nonce authenticity; this only preserves local once-ness.
BCC32_GUARDED_ACTION_HD inline bool ticket_is_bound_or_consumed(
    const rewrite::ResidentRewriteState& state,
    const kernel::ActionReturnTicket& ticket) {
  if (!valid_ticket(ticket)) return true;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&state);
       ++slot) {
    const rewrite::Record& record = state.records[slot];
    if (record.matter_q8 != 0u &&
        (record.lane[0] == pressure::kFormCommittedMatter ||
         record.lane[0] == pressure::kFormFreeMatter) &&
        record.reserved[0] == ticket_sequence_low(ticket) &&
        record.reserved[1] == ticket_sequence_high(ticket))
      return true;
  }
  return false;
}

// This is the sole boolean a canonical egress writer needs to consult before
// publishing the same ticketed action.  It is deliberately a guard, not an
// alternate egress/action writer: false means withhold the already-selected
// public action; true still requires charge_before_action_commit first.
BCC32_GUARDED_ACTION_HD inline bool allow_canonical_ticketed_public_commit(
    const rewrite::ResidentRewriteState& state, const GuardedAction& pending) {
  return valid_ticket(pending.ticket) &&
         pending.ticket_binding ==
             ticket_binding(pending.ticket, pending.affordability) &&
         !ticket_is_bound_or_consumed(state, pending.ticket) &&
         pending.affordability.state_revision == state.revision &&
         pressure::structural_margin_q8(state, pending.affordability.owner) > 0;
}

BCC32_GUARDED_ACTION_HD inline CanonicalTicketedActionView canonical_view(
    const GuardedAction& action) {
  CanonicalTicketedActionView view{};
  view.ticket = action.ticket;
  view.selected_extent = action.affordability.selected_extent;
  view.purpose = action.affordability.purpose;
  return view;
}

BCC32_GUARDED_ACTION_HD inline bool is_exact_public_action_receipt(
    const CanonicalPublicActionReceipt& receipt,
    const CanonicalTicketedActionView& action) {
  return receipt.action_present == 1u && receipt.public_sequence != 0u &&
         same_ticket(receipt.ticket, action.ticket);
}

BCC32_GUARDED_ACTION_HD inline bool is_exact_learned_sentence_receipt(
    const CanonicalLearnedSentenceReceipt& receipt,
    const CanonicalTicketedActionView& action) {
  return receipt.action_present == 1u &&
         receipt.complete_sentence_present == 1u &&
         receipt.public_sequence != 0u &&
         receipt.learned_sentence_sequence != 0u &&
         same_ticket(receipt.ticket, action.ticket);
}

// This is the required pre-public-commit snapshot.  selected_extent is
// supplied by the existing selector; the adapter never derives it from a
// token, route, action value, or host scalar.
BCC32_GUARDED_ACTION_HD inline bool snapshot_before_action_commit(
    const rewrite::ResidentRewriteState& state,
    const kernel::ActionReturnTicket& ticket, std::uint32_t owner,
    std::uint32_t selected_extent, pressure::PaymentPurpose purpose,
    GuardedAction* pending) {
  if (pending == nullptr || !valid_ticket(ticket) ||
      ticket_is_bound_or_consumed(state, ticket))
    return false;
  GuardedAction candidate{};
  candidate.ticket = ticket;
  if (!pressure::read_affordability(state, owner, selected_extent, purpose,
                                    &candidate.affordability))
    return false;
  candidate.ticket_binding = ticket_binding(ticket, candidate.affordability);
  *pending = candidate;
  return true;
}

// Charge exactly once before the caller commits the corresponding public
// action.  The record conversion happens before this returns true; callers
// must withhold the action on false and must not call this after egress.
BCC32_GUARDED_ACTION_HD inline bool charge_before_action_commit(
    rewrite::ResidentRewriteState* state, const GuardedAction& pending,
    GuardedAction* charged) {
  if (state == nullptr || charged == nullptr ||
      !allow_canonical_ticketed_public_commit(*state, pending))
    return false;
  GuardedAction candidate = pending;
  if (!pressure::commit_selected_extent(state, pending.affordability,
                                        &candidate.payment))
    return false;
  for (std::uint32_t i = 0u; i < candidate.payment.selected_extent; ++i) {
    rewrite::Record& record = state->records[candidate.payment.slots[i]];
    record.reserved[0] = ticket_sequence_low(candidate.ticket);
    record.reserved[1] = ticket_sequence_high(candidate.ticket);
  }
  *charged = candidate;
  return true;
}

// Invoke the supplied canonical commit callback exactly once and only after an
// exact pre-commit charge.  The callback remains the only action authority;
// this adapter never writes public action bytes.  If that authority declines
// after charge, matter stays committed fail-closed rather than being minted
// back without a later resident consequence.
template <typename CanonicalCommit>
inline bool commit_via_canonical_ticketed_action(
    rewrite::ResidentRewriteState* state, const GuardedAction& pending,
    CanonicalCommit&& canonical_commit, GuardedAction* charged,
    CanonicalPublicActionReceipt* public_receipt) {
  if (state == nullptr || charged == nullptr ||
      public_receipt == nullptr ||
      !allow_canonical_ticketed_public_commit(*state, pending))
    return false;
  GuardedAction candidate{};
  if (!charge_before_action_commit(state, pending, &candidate)) return false;
  CanonicalPublicActionReceipt receipt{};
  if (!canonical_commit(canonical_view(candidate), &receipt) ||
      !is_exact_public_action_receipt(receipt, canonical_view(candidate)))
    return false;
  *charged = candidate;
  *public_receipt = receipt;
  return true;
}

// This is the language-bearing production candidate.  The callback is the
// only canonical publisher and must return its own opaque learned-sentence
// receipt.  The guard never supplies sentence bytes or chooses an answer.
template <typename CanonicalSentenceCommit>
inline bool commit_via_canonical_learned_sentence_action(
    rewrite::ResidentRewriteState* state, const GuardedAction& pending,
    CanonicalSentenceCommit&& canonical_commit, GuardedAction* charged,
    CanonicalLearnedSentenceReceipt* public_receipt) {
  if (state == nullptr || charged == nullptr || public_receipt == nullptr ||
      !allow_canonical_ticketed_public_commit(*state, pending))
    return false;
  GuardedAction candidate{};
  if (!charge_before_action_commit(state, pending, &candidate)) return false;
  CanonicalLearnedSentenceReceipt receipt{};
  if (!canonical_commit(canonical_view(candidate), &receipt) ||
      !is_exact_learned_sentence_receipt(receipt, canonical_view(candidate)))
    return false;
  *charged = candidate;
  *public_receipt = receipt;
  return true;
}

// The existing action-return ingress owns the strongest available raw receipt:
// accepted count advances, the ticket sequence is recorded, and a nonzero raw
// contact digest/sequence is committed before rewrite learning.  This adapter
// accepts no host reward, label, or hand-authored consequence in its place.
BCC32_GUARDED_ACTION_HD inline bool is_accepted_raw_action_return(
    const kernel::TickReceipt& receipt, std::uint64_t accepted_before,
    const kernel::ActionReturnTicket& ticket) {
  return valid_ticket(ticket) && receipt.action_return_accepted > accepted_before &&
         receipt.action_return_action_sequence == ticket.action_sequence &&
         receipt.action_return_contact_sequence != 0u &&
         receipt.action_return_contact != ContentAddress{};
}

// The consequence must be bound to an accepted raw action-return receipt for
// this exact ticket.  Raw ingress, not this donor, remains the provenance
// authority that decides whether that receipt exists.
BCC32_GUARDED_ACTION_HD inline bool refund_after_ticketed_consequence(
    rewrite::ResidentRewriteState* state, const GuardedAction& charged,
    const GuardedActionConsequence& consequence) {
  if (state == nullptr || !valid_ticket(charged.ticket) ||
      !is_accepted_raw_action_return(consequence.accepted_return,
                                     consequence.accepted_before,
                                     charged.ticket) ||
      charged.ticket_binding !=
          ticket_binding(charged.ticket, charged.affordability) ||
      ticket_is_bound_or_consumed(*state, charged.ticket) == false)
    return false;
  for (std::uint32_t i = 0u; i < charged.payment.selected_extent; ++i) {
    if (charged.payment.slots[i] >= rewrite::kRecordCapacity) return false;
    const rewrite::Record& record = state->records[charged.payment.slots[i]];
    if (record.reserved[0] != ticket_sequence_low(charged.ticket) ||
        record.reserved[1] != ticket_sequence_high(charged.ticket))
      return false;
  }
  if (!pressure::refund_after_consequence(state, charged.payment,
                                          consequence.resident))
    return false;
  for (std::uint32_t i = 0u; i < charged.payment.selected_extent; ++i) {
    rewrite::Record& record = state->records[charged.payment.slots[i]];
    record.reserved[0] = ticket_sequence_low(charged.ticket);
    record.reserved[1] = ticket_sequence_high(charged.ticket);
  }
  return true;
}

// Accept the existing canonical raw-return receipt by passive view.  The
// consequence fields remain record-bound and must still match the exact charge.
inline bool refund_via_accepted_return_view(
    rewrite::ResidentRewriteState* state, const GuardedAction& charged,
    const AcceptedReturnView& accepted, pressure::ResidentConsequence resident) {
  if (accepted.receipt == nullptr) return false;
  GuardedActionConsequence consequence{};
  consequence.accepted_return = *accepted.receipt;
  consequence.accepted_before = accepted.accepted_before;
  consequence.resident = resident;
  return refund_after_ticketed_consequence(state, charged, consequence);
}

}  // namespace substrate::bcc32::resident_resource_guarded_action

#undef BCC32_GUARDED_ACTION_HD
