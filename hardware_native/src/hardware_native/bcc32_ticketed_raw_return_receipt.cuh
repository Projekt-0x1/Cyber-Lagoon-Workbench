#pragma once

#include "bcc32_persistent_kernel.hpp"
#include "bcc32_resident_mixed_provenance_evidence.cuh"
#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_TICKETED_RETURN_HD __host__ __device__
#else
#define BCC32_TICKETED_RETURN_HD
#endif

namespace substrate::bcc32::causal_rewrite::ticketed_raw_return_receipt {

namespace kernel = ::substrate::bcc32::persistent_kernel;
using ActionReturnTicket = kernel::ActionReturnTicket;

inline constexpr std::uint32_t kFormTicketedRawReturnReceipt = 0xafd8c6e1u;
inline constexpr std::uint32_t kReceiptAccepted = 1u;
inline constexpr std::uint32_t kMaximumContactWords = 64u;

// This is a view of fields already published by the current canonical return
// path. It is not a second ingress ring or a host-selected reward channel.
struct CanonicalReturnView {
  ActionReturnTicket ticket{};
  std::uint32_t action_word = 0u;
  std::uint32_t returned_word = 0u;
  std::uint32_t contact_word_count = 0u;
  std::uint64_t ingress_sequence = 0u;
  std::uint32_t contact_digest_word = 0u;
};

struct RawPayloadView {
  const std::uint32_t* words = nullptr;
  std::uint32_t count = 0u;
  std::uint32_t contact_digest_word = 0u;
  std::uint64_t ingress_sequence = 0u;
};

struct Receipt {
  std::uint32_t accepted = 0u;
  std::uint32_t status = 0u;
  std::uint64_t action_sequence = 0u;
  std::uint64_t ingress_sequence = 0u;
  std::uint32_t contact_digest_word = 0u;
  std::uint32_t resident_revision = 0u;
};

struct LearningReceipt {
  Receipt receipt{};
  std::uint32_t trajectory_owner = kInvalid;
  std::uint32_t trajectory_revision = 0u;
  std::uint32_t raw_word = 0u;
};

BCC32_TICKETED_RETURN_HD inline bool valid_ticket(
    const ActionReturnTicket& ticket) {
  return ticket.action_sequence != 0u && ticket.nonce != 0u;
}

BCC32_TICKETED_RETURN_HD inline bool is_channel_one(
    std::uint32_t word) {
  return ((word & kRawChannelMask) >> 24u) == 1u;
}

BCC32_TICKETED_RETURN_HD inline bool same_ticket(
    const ActionReturnTicket& left, const ActionReturnTicket& right) {
  return left.action_sequence == right.action_sequence &&
         left.nonce == right.nonce;
}

BCC32_TICKETED_RETURN_HD inline std::uint32_t payload_digest_word(
    const std::uint32_t* words, std::uint32_t count) {
  if (words == nullptr || count == 0u || count > kMaximumContactWords)
    return 0u;
  std::uint32_t digest = rewrite_mix(kFormTicketedRawReturnReceipt, count, 0u);
  for (std::uint32_t index = 0u; index < count; ++index)
    digest = rewrite_mix(digest, words[index], index);
  return digest == 0u || digest == kInvalid ? digest ^ 0x7135c9a2u : digest;
}

BCC32_TICKETED_RETURN_HD inline bool receipt_matches(
    const Record& record, const CanonicalReturnView& returned) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormTicketedRawReturnReceipt &&
         record.lane[1] == static_cast<std::uint32_t>(
                                returned.ticket.action_sequence) &&
         record.lane[2] == static_cast<std::uint32_t>(
                                returned.ticket.action_sequence >> 32u) &&
         record.lane[3] ==
             static_cast<std::uint32_t>(returned.ticket.nonce) &&
         record.lane[4] == static_cast<std::uint32_t>(
                                returned.ticket.nonce >> 32u);
}

BCC32_TICKETED_RETURN_HD inline bool duplicate_receipt(
    const ResidentRewriteState* state,
    const CanonicalReturnView& returned) {
  if (state == nullptr) return false;
  for (std::uint32_t slot = 0u; slot < kRecordCapacity; ++slot)
    if (receipt_matches(state->records[slot], returned)) return true;
  return false;
}

BCC32_TICKETED_RETURN_HD inline bool yoked_receipt(
    const ResidentRewriteState* state,
    const CanonicalReturnView& returned) {
  if (state == nullptr) return false;
  for (std::uint32_t slot = 0u; slot < kRecordCapacity; ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == kFormTicketedRawReturnReceipt &&
        record.lane[5] == returned.action_word &&
        record.reserved[1] == returned.contact_digest_word)
      return true;
  }
  return false;
}

// One canonical-callable primitive. It binds the already-issued ticket,
// emitted action, ingress sequence/count, and raw contact digest into one
// resident Record. No public language or motor field is written.
BCC32_TICKETED_RETURN_HD inline Receipt admit(
    ResidentRewriteState* state, const ActionReturnTicket& issued_ticket,
    const CanonicalReturnView& returned) {
  Receipt receipt{};
  receipt.action_sequence = returned.ticket.action_sequence;
  receipt.ingress_sequence = returned.ingress_sequence;
  receipt.contact_digest_word = returned.contact_digest_word;
  if (state == nullptr || state->fault != 0u ||
      !valid_ticket(issued_ticket) || !valid_ticket(returned.ticket) ||
      !same_ticket(issued_ticket, returned.ticket) ||
      !is_channel_one(returned.action_word) ||
      returned.contact_word_count > kMaximumContactWords ||
      returned.ingress_sequence == 0u || returned.contact_digest_word == 0u ||
      returned.action_word == 0u)
    return receipt;
  // A body echo of the emitted action is not a returned consequence.
  if (returned.contact_word_count == 1u &&
      returned.returned_word == returned.action_word)
    return receipt;
  if (duplicate_receipt(state, returned) || yoked_receipt(state, returned) ||
      free_record_count(state) == 0u)
    return receipt;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return receipt;
  Record& record = state->records[slot];
  record.lane[0] = kFormTicketedRawReturnReceipt;
  record.lane[1] = static_cast<std::uint32_t>(returned.ticket.action_sequence);
  record.lane[2] = static_cast<std::uint32_t>(
      returned.ticket.action_sequence >> 32u);
  record.lane[3] = static_cast<std::uint32_t>(returned.ticket.nonce);
  record.lane[4] = static_cast<std::uint32_t>(returned.ticket.nonce >> 32u);
  record.lane[5] = returned.action_word;
  record.lane[6] = returned.contact_word_count;
  record.lane[7] = kReceiptAccepted;
  record.reserved[0] = static_cast<std::uint32_t>(returned.ingress_sequence);
  record.reserved[1] = returned.contact_digest_word;
  ++record.revision;
  ++state->revision;
  refresh_receipt(state);
  receipt.accepted = 1u;
  receipt.status = kReceiptAccepted;
  receipt.resident_revision = static_cast<std::uint32_t>(state->revision);
  return receipt;
}

// Canonical callback for an accepted raw payload. The digest is recomputed
// from the payload before resident mutation, so post-publication payload
// mutation cannot be laundered as the original contact receipt.
//
// issued_ticket and returned_ticket are deliberately separate parameters:
// issued_ticket is the ticket this resident actually dispatched for the
// action, returned_ticket is whatever ticket the caller now presents
// alongside the raw payload. admit()'s same_ticket() check below is the
// only place a forged/stale presented ticket is rejected; collapsing both
// parameters into one caller-supplied value (as an earlier revision of
// this wrapper did) makes that check compare a value against itself and
// therefore never fail.
BCC32_TICKETED_RETURN_HD inline Receipt admit_payload(
    ResidentRewriteState* state, const ActionReturnTicket& issued_ticket,
    const ActionReturnTicket& returned_ticket, std::uint32_t action_word,
    const RawPayloadView& payload) {
  Receipt receipt{};
  if (payload.words == nullptr || payload.count == 0u ||
      payload.count > kMaximumContactWords ||
      payload_digest_word(payload.words, payload.count) !=
          payload.contact_digest_word)
    return receipt;
  CanonicalReturnView returned{};
  returned.ticket = returned_ticket;
  returned.action_word = action_word;
  returned.returned_word = payload.words[0];
  returned.contact_word_count = payload.count;
  returned.ingress_sequence = payload.ingress_sequence;
  returned.contact_digest_word = payload.contact_digest_word;
  return admit(state, issued_ticket, returned);
}

BCC32_TICKETED_RETURN_HD inline std::uint32_t find_receipt_slot(
    const ResidentRewriteState* state, const ActionReturnTicket& ticket) {
  if (state == nullptr) return kInvalid;
  CanonicalReturnView key{};
  key.ticket = ticket;
  for (std::uint32_t slot = 0u; slot < kRecordCapacity; ++slot)
    if (receipt_matches(state->records[slot], key)) return slot;
  return kInvalid;
}

// Hand the accepted raw word to the existing resident external-ingress and
// provenance path. This is the only learning handoff here: it creates an
// ordinary one-word external trajectory for later resident Programs to read;
// it does not author language, reward, or a semantic answer.
BCC32_TICKETED_RETURN_HD inline LearningReceipt admit_payload_to_learning(
    ResidentRewriteState* state, const ActionReturnTicket& issued_ticket,
    const ActionReturnTicket& returned_ticket, std::uint32_t action_word,
    const RawPayloadView& payload) {
  LearningReceipt result{};
  if (state == nullptr || payload.words == nullptr || payload.count != 1u ||
      find_current_trajectory(state) != kInvalid ||
      free_record_count(state) < 4u)
    return result;
  result.receipt =
      admit_payload(state, issued_ticket, returned_ticket, action_word,
                    payload);
  if (result.receipt.accepted == 0u) return result;
  // admit()/admit_payload() stamp the receipt Record with returned.ticket
  // (the presented ticket), not issued_ticket -- receipt_matches() compares
  // against that same field, so the lookup key here must be returned_ticket
  // or a legitimate lookup for a forgery-rejected call would silently find
  // nothing while an accepted call under a mismatched issued_ticket search
  // key would never locate its own just-written record.
  const std::uint32_t receipt_slot =
      find_receipt_slot(state, returned_ticket);
  ResidentRewriteEngine engine(state);
  const bool ingressed = mixed_provenance::consume_external_event(
      engine, RawRewriteEvent{payload.words[0], 1u, kEventFrameNone}, false);
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  const bool grounded =
      ingressed && state->fault == 0u && trajectory_slot != kInvalid &&
      mixed_provenance::tagged_history(state, state->records[trajectory_slot]);
  if (!grounded) {
    if (receipt_slot != kInvalid) clear_record(&state->records[receipt_slot]);
    if (trajectory_slot != kInvalid)
      mixed_provenance::clear_trajectory_and_provenance(state, trajectory_slot);
    if (state->revision != 0u) --state->revision;
    refresh_receipt(state);
    result = LearningReceipt{};
    return result;
  }
  mixed_provenance::Origin origin = mixed_provenance::Origin::generated;
  std::uint32_t producer = kInvalid;
  const Record& trajectory = state->records[trajectory_slot];
  const bool exactly_one_external =
      trajectory.lane[2] == 1u && mixed_provenance::origin_at(
          state, trajectory, 0u, &origin, &producer) &&
      origin == mixed_provenance::Origin::external && producer == kInvalid;
  if (!exactly_one_external) {
    if (receipt_slot != kInvalid) clear_record(&state->records[receipt_slot]);
    mixed_provenance::clear_trajectory_and_provenance(state, trajectory_slot);
    if (state->revision != 0u) --state->revision;
    refresh_receipt(state);
    return LearningReceipt{};
  }
  result.trajectory_owner = trajectory.lane[1];
  result.trajectory_revision = trajectory.revision;
  result.raw_word = payload.words[0];
  return result;
}

}  // namespace substrate::bcc32::causal_rewrite::ticketed_raw_return_receipt

#undef BCC32_TICKETED_RETURN_HD
