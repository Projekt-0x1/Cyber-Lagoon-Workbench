#pragma once

#include "bcc32_persistent_kernel.hpp"
#include "bcc32_resident_egress_history.cuh"
#include "bcc32_resident_open_inquiry.cuh"

#include <cstdint>
#include <type_traits>

// Exact RWR0 clarification transaction.  The resident inquiry and its live
// alternative Programs determine what is unresolved; the canonical egress
// history determines which completed public act receives a return ticket.
// This seam carries no answer, referent, surface buffer, party row, or
// caller-selected Record locus.
namespace substrate::bcc32::causal_rewrite::open_inquiry_return_gate {

namespace inquiry = open_inquiry;
namespace history = persistent_kernel::egress_history;
namespace kernel = persistent_kernel;

#if defined(__CUDACC__)
#define BCC32_INQUIRY_RETURN_HD __host__ __device__
#else
#define BCC32_INQUIRY_RETURN_HD
#endif

struct Publication {
  kernel::ActionReturnTicket ticket{};
  std::uint64_t world_revision = 0u;
  std::uint64_t action_sequence = 0u;
  std::uint64_t digest = 0u;
  std::uint32_t inquiry_owner = kInvalid;
  std::uint32_t constructor_locus = kInvalid;
  std::uint32_t unresolved_identity = 0u;
  std::uint32_t alternative_count = 0u;
};

static_assert(std::is_trivially_copyable_v<Publication>);

BCC32_INQUIRY_RETURN_HD inline std::uint64_t mix64(std::uint64_t digest,
                                                    std::uint64_t value) {
  digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) + (digest >> 2u);
  return digest == 0u ? 1u : digest;
}

BCC32_INQUIRY_RETURN_HD inline std::uint64_t publication_digest(
    const Publication& publication) {
  std::uint64_t digest = mix64(0x52575230494e5159ull,
                               publication.ticket.issuer_instance);
  digest = mix64(digest, publication.ticket.action_sequence);
  digest = mix64(digest, publication.ticket.nonce);
  digest = mix64(digest, publication.world_revision);
  digest = mix64(digest, publication.action_sequence);
  digest = mix64(digest, publication.inquiry_owner);
  digest = mix64(digest, publication.constructor_locus);
  digest = mix64(digest, publication.unresolved_identity);
  return mix64(digest, publication.alternative_count);
}

BCC32_INQUIRY_RETURN_HD inline bool publication_self_consistent(
    const ResidentRewriteState* state, const Publication& publication) {
  return state != nullptr && publication.ticket.issuer_instance != 0u &&
         publication.ticket.action_sequence != 0u &&
         publication.ticket.nonce != 0u &&
         publication.ticket.action_sequence == publication.action_sequence &&
         publication.world_revision != 0u &&
         publication.inquiry_owner != 0u &&
         publication.inquiry_owner != kInvalid &&
         publication.constructor_locus < live_record_capacity(state) &&
         publication.unresolved_identity != 0u &&
         publication.unresolved_identity != kInvalid &&
         publication.alternative_count == 2u && publication.digest != 0u &&
         publication.digest == publication_digest(publication);
}

BCC32_INQUIRY_RETURN_HD inline bool derive_unresolved_identity(
    const ResidentRewriteState* state, const Record& inquiry_header,
    std::uint32_t* identity) {
  if (state == nullptr || identity == nullptr || inquiry_header.lane[4] != 2u)
    return false;
  const std::uint32_t suspended_slot = cross_contact::find_trajectory_by_owner(
      state, inquiry_header.lane[2]);
  if (suspended_slot == kInvalid ||
      inquiry_header.lane[3] != state->records[suspended_slot].lane[2])
    return false;
  inquiry::AlternativeSnapshot alternatives[2]{};
  if (!inquiry::collect_exact_fork(state, state->records[suspended_slot],
                                   alternatives))
    return false;
  bool matched[2]{false, false};
  std::uint32_t binding_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != inquiry::kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry_header.lane[1])
      continue;
    if (binding_count == 2u) return false;
    bool binding_matched = false;
    for (std::uint32_t index = 0u; index < 2u; ++index) {
      if (matched[index] || binding.lane[2] != alternatives[index].owner ||
          binding.lane[3] != alternatives[index].revision ||
          binding.lane[4] != alternatives[index].consequence)
        continue;
      matched[index] = true;
      binding_matched = true;
      break;
    }
    if (!binding_matched) return false;
    ++binding_count;
  }
  if (binding_count != 2u || !matched[0] || !matched[1]) return false;
  std::uint32_t digest = rewrite_mix(inquiry_header.lane[1],
                                     inquiry_header.lane[2],
                                     inquiry_header.lane[3]);
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    // collect_exact_fork performed the complete resident validation and the
    // binding scan above proves that both stored alternatives name that same
    // snapshot. Do not call the full authority reader again here: a grounded
    // VersionSpace alternative is deliberately non-executable after
    // counterevidence, and repeating its resident scan on every epoch turns
    // this receipt path back into the quadratic stall the bounded collector
    // removed.
    digest = rewrite_mix(digest, alternatives[index].owner,
                         alternatives[index].revision);
    digest = rewrite_mix(digest, alternatives[index].consequence, index + 1u);
  }
  if (digest == 0u || digest == kInvalid) return false;
  *identity = digest;
  return true;
}

// The final learned question word becomes return-capable only after its
// constructor emission cursor has disappeared.  Earlier bytes cannot acquire
// a ticket, and an unrelated channel-1 action cannot masquerade as this
// inquiry.  The egress event is looked up from the ticket sequence rather than
// nominated by the caller.
BCC32_INQUIRY_RETURN_HD inline Publication bind_completed_public_inquiry(
    ResidentRewriteState* state, const history::State* egress,
    const kernel::ActionReturnTicket& ticket) {
  Publication publication{};
  if (state == nullptr || egress == nullptr || state->fault != 0u ||
      ticket.issuer_instance == 0u || ticket.action_sequence == 0u ||
      ticket.nonce == 0u)
    return publication;
  history::Event action{};
  if (!history::lookup(egress, ticket.action_sequence, &action) ||
      action.sequence != ticket.action_sequence || action.completed_tick == 0u)
    return publication;
  const std::uint32_t inquiry_slot = inquiry::unique_active_inquiry(state);
  if (inquiry_slot == kInvalid) return publication;
  const Record& header = state->records[inquiry_slot];
  if ((header.lane[7] & (inquiry::kInquiryAwaitingReply |
                         inquiry::kInquirySurfaceEmitted)) !=
          (inquiry::kInquiryAwaitingReply |
           inquiry::kInquirySurfaceEmitted) ||
      (header.lane[7] & (inquiry::kInquirySurfaceCaptured |
                         inquiry::kInquiryReplyBound |
                         inquiry::kInquirySettled)) != 0u ||
      header.reserved[1] >= live_record_capacity(state) ||
      action.producer_locus != header.reserved[1] ||
      !inquiry::inquiry_constructor_authoritative(state, header.reserved[1]))
    return publication;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& emission = state->records[slot];
    if (emission.matter_q8 != 0u &&
        emission.lane[0] == inquiry::kFormOpenInquiryEmission &&
        emission.lane[1] == header.lane[1])
      return publication;
  }
  std::uint32_t identity = 0u;
  if (!derive_unresolved_identity(state, header, &identity)) return publication;
  publication.ticket = ticket;
  publication.world_revision = state->revision;
  publication.action_sequence = action.sequence;
  publication.inquiry_owner = header.lane[1];
  publication.constructor_locus = header.reserved[1];
  publication.unresolved_identity = identity;
  publication.alternative_count = header.lane[4];
  publication.digest = publication_digest(publication);
  if (!publication_self_consistent(state, publication)) return Publication{};
  // Binding the completed public writer is the authority transition. Keep
  // ordinary generated-surface detachment neutral because the same physical
  // operation is also used by the adult's ordinary teacher-contact path. A
  // copied/replayed resident state carries this aperture bit, so raw contact
  // cannot promote itself into an external return after publication.
  state->open_inquiry_public_return_pending = 1u;
  return publication;
}

BCC32_INQUIRY_RETURN_HD inline bool still_same_unresolved_inquiry(
    const ResidentRewriteState* state, const Publication& publication) {
  if (state == nullptr || state->fault != 0u ||
      !publication_self_consistent(state, publication) ||
      state->revision < publication.world_revision)
    return false;
  const std::uint32_t inquiry_slot = inquiry::unique_active_inquiry(state);
  if (inquiry_slot == kInvalid) return false;
  const Record& header = state->records[inquiry_slot];
  std::uint32_t identity = 0u;
  return state->open_inquiry_public_return_pending != 0u &&
         header.lane[1] == publication.inquiry_owner &&
         header.reserved[1] == publication.constructor_locus &&
         derive_unresolved_identity(state, header, &identity) &&
         identity == publication.unresolved_identity;
}

// Acquisition remains ordinary contact: before the adult has ever learned a
// clarification surface, an externally presented teaching surface may be
// followed by one externally presented reply.  Once a resident surface was
// emitted publicly, this aperture closes and only the exact ticketed function
// below can bind a reply.
BCC32_INQUIRY_RETURN_HD inline bool bind_unemitted_teacher_reply_before_end(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  if (state->open_inquiry_public_return_pending != 0u) return false;
  const std::uint32_t slot = inquiry::unique_active_inquiry(state);
  if (slot == kInvalid) return false;
  const Record& header = state->records[slot];
  if ((header.lane[7] & inquiry::kInquirySurfaceCaptured) == 0u ||
      (header.lane[7] & inquiry::kInquirySurfaceEmitted) != 0u)
    return false;
  return inquiry::bind_one_fresh_external_reply_before_end(state);
}

// Called only inside the runtime's shadow return world after its canonical
// ticket/nonce/producer commitment gate accepted the complete raw return.  A
// failure leaves that shadow uncommittable, so bind+settle is atomic at the
// resident root.  Selection comes only from the raw reply matching one stored
// continuation; this API has no selected-alternative argument.
BCC32_INQUIRY_RETURN_HD inline bool settle_ticketed_reply_before_end(
    ResidentRewriteState* state, const Publication& publication) {
  if (!still_same_unresolved_inquiry(state, publication) ||
      state->open_inquiry_public_return_pending == 0u)
    return false;
  // The publication digest and resident identity have just been checked.
  // Temporarily consume the resident aperture so generic settlement cannot
  // close a public inquiry without this exact ticketed path. Restore it on
  // every failed attempt so a later valid return remains possible.
  state->open_inquiry_public_return_pending = 0u;
  if (!inquiry::bind_one_fresh_external_reply_before_end(state) ||
      !inquiry::settle_one_at_end(state)) {
    state->open_inquiry_public_return_pending = 1u;
    return false;
  }
  std::uint32_t settled = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != inquiry::kFormOpenInquiry ||
        header.lane[1] != publication.inquiry_owner)
      continue;
    if (settled != kInvalid ||
        (header.lane[7] & (inquiry::kInquiryReplyBound |
                           inquiry::kInquirySettled)) !=
            (inquiry::kInquiryReplyBound | inquiry::kInquirySettled) ||
        (header.lane[7] & inquiry::kInquiryAwaitingReply) != 0u)
      return false;
    settled = slot;
  }
  return settled != kInvalid;
}

}  // namespace substrate::bcc32::causal_rewrite::open_inquiry_return_gate

#undef BCC32_INQUIRY_RETURN_HD
