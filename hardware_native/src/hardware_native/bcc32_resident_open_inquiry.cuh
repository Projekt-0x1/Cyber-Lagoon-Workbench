#pragma once

#include "bcc32_resident_cross_contact_context.cuh"
#include "bcc32_resident_mixed_provenance_evidence.cuh"
#include "bcc32_resident_pending_means.cuh"
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_OPEN_INQUIRY_HD __host__ __device__
#else
#define BCC32_OPEN_INQUIRY_HD
#endif

// RWR24 is a bounded resident clarification bootstrap.  It does not contain an
// ASK opcode, a semantic selector, or a second output route.  It records only
// lived fork -> external surface -> external exact-continuation reply -> resumed
// program episodes, factors their byte topology, and lets the existing root decide when
// these hooks run at physical PAUSE/END and after ordinary program advancement.
namespace substrate::bcc32::causal_rewrite::open_inquiry {

inline constexpr std::uint32_t kFormOpenInquiry = 0x94c6a217u;
inline constexpr std::uint32_t kFormOpenInquiryAlternative = 0xa5d7b328u;
inline constexpr std::uint32_t kFormOpenInquirySurfaceTerm = 0xb6e8c439u;
inline constexpr std::uint32_t kFormOpenInquiryReplyWitness = 0xc7f9d54au;
inline constexpr std::uint32_t kFormOpenInquiryReplyTerm = 0x6a1d3f90u;
inline constexpr std::uint32_t kFormOpenInquiryResumeWitness = 0xd80ae65bu;
inline constexpr std::uint32_t kFormOpenInquiryConstructor = 0xe91bf76cu;
inline constexpr std::uint32_t kFormOpenInquiryConstructorTerm = 0xfa2c087du;
inline constexpr std::uint32_t kFormOpenInquiryConstructorWitness = 0x8b3d198eu;
inline constexpr std::uint32_t kFormOpenInquiryEmission = 0x7c4e2a9fu;

// Inquiry header lanes:
//   [1] owner; [2,3] exact suspended owner/length; [4] alternative count;
//   [5,6] selected alternative owner/revision after reply binding; [7] state;
//   reserved[0] exact captured external surface length (zero for held-out use);
//   reserved[1] exact learned Constructor slot for held-out emission, else invalid.
// Alternative lanes: [1] inquiry owner; [2,3,4] program owner/revision/first
// continuation word.  The owner/revision name the complete continuation; the
// first word remains only the legacy one-word wrapper discriminator.
// Reply witness lanes: [1] inquiry owner; [2] reply owner; [3,4,5] selected
// program owner/revision and first selected word; [6] reply-source revision;
// reserved[0,1] reply word count and trajectory digest. Reply terms retain the
// external raw topology. Constructor terms contain only learned byte topology;
// no host-selected text, entity, or answer role exists here.
struct InquiryView {
  std::uint32_t inquiry_slot = kInvalid;
  std::uint32_t suspended_slot = kInvalid;
  std::uint32_t reply_slot = kInvalid;
  std::uint32_t matching_alternative_slot = kInvalid;
  std::uint32_t constructor_slot = kInvalid;
};

struct AlternativeSnapshot {
  std::uint32_t owner = kInvalid;
  std::uint32_t revision = 0u;
  std::uint32_t consequence = 0u;
  std::uint32_t slot = kInvalid;
};
#include "bcc32_resident_open_inquiry_core.inl"
#include "bcc32_resident_open_inquiry_alternatives.inl"

BCC32_OPEN_INQUIRY_HD inline bool snapshot_still_live(
    const ResidentRewriteState* state, const Record& suspended,
    const Record& binding, std::uint32_t* slot,
    bool require_enabled = true) {
  if (slot != nullptr) *slot = kInvalid;
  if (state == nullptr || binding.matter_q8 == 0u ||
      binding.lane[0] != kFormOpenInquiryAlternative)
    return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t candidate_slot = 0u; candidate_slot < live_record_capacity(state);
       ++candidate_slot) {
    const Record& program = state->records[candidate_slot];
    if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        program.lane[1] != binding.lane[2])
      continue;
    if (found != kInvalid || program.revision != binding.lane[3] ||
        !live_resident_continuation_alternative(state, candidate_slot,
                                                suspended, nullptr,
                                                require_enabled))
      return false;
    std::uint32_t word = 0u;
    if (!alternative_continuation_word_at(
            state, AlternativeSnapshot{program.lane[1], program.revision, 0u,
                                       candidate_slot},
            suspended.lane[2], 0u, &word) ||
        word != binding.lane[4])
      return false;
    found = candidate_slot;
  }
  if (found == kInvalid) return false;
  if (slot != nullptr) *slot = found;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_active_inquiry(
    const ResidentRewriteState* state) {
  std::uint32_t found = kInvalid;
  if (state == nullptr) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& inquiry = state->records[slot];
    if (inquiry.matter_q8 == 0u || inquiry.lane[0] != kFormOpenInquiry ||
        (inquiry.lane[7] & kInquiryAwaitingReply) == 0u ||
        (inquiry.lane[7] & kInquirySettled) != 0u)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_current_trajectory(
    const ResidentRewriteState* state, bool* ambiguous = nullptr) {
  if (ambiguous != nullptr) *ambiguous = false;
  std::uint32_t found = kInvalid;
  if (state == nullptr) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& trajectory = state->records[slot];
    if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory ||
        trajectory.lane[3] != 0u)
      continue;
    if (found != kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      return kInvalid;
    }
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_surface_word_at(
    const ResidentRewriteState* state, const Record& inquiry,
    std::uint32_t index, std::uint32_t* word) {
  if (word == nullptr || index >= inquiry.reserved[0]) return false;
  const std::uint32_t slot = unique_owned_slot(
      state, kFormOpenInquirySurfaceTerm, inquiry.lane[1], index);
  if (slot == kInvalid) return false;
  *word = state->records[slot].lane[3];
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool continuation_occurs_once_in_surface(
    const ResidentRewriteState* state, const Record& inquiry,
    const AlternativeSnapshot& alternative, std::uint32_t suspended_length) {
  std::uint32_t length = 0u;
  std::uint32_t digest = 0u;
  if (!alternative_continuation_digest(state, alternative, suspended_length,
                                       &length, &digest) ||
      length == 0u || length > inquiry.reserved[0])
    return false;
  std::uint32_t matches = 0u;
  for (std::uint32_t start = 0u; start + length <= inquiry.reserved[0]; ++start) {
    bool exact = true;
    for (std::uint32_t index = 0u; index < length; ++index) {
      std::uint32_t surface_word = 0u;
      std::uint32_t label_word = 0u;
      if (!inquiry_surface_word_at(state, inquiry, start + index, &surface_word) ||
          !alternative_continuation_word_at(state, alternative, suspended_length,
                                            index, &label_word) ||
          surface_word != label_word) {
        exact = false;
        break;
      }
    }
    matches += exact;
  }
  return matches == 1u;
}

BCC32_OPEN_INQUIRY_HD inline bool reply_matches_continuation(
    const ResidentRewriteState* state, const Record& reply,
    const AlternativeSnapshot& alternative, std::uint32_t suspended_length,
    std::uint32_t* digest) {
  std::uint32_t length = 0u;
  std::uint32_t expected_digest = 0u;
  if (digest != nullptr) *digest = 0u;
  if (!alternative_continuation_digest(state, alternative, suspended_length,
                                       &length, &expected_digest) ||
      reply.lane[2] != length)
    return false;
  std::uint32_t observed_digest = 0u;
  for (std::uint32_t index = 0u; index < length; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    if (!trajectory_word_at(state, reply.lane[1], index, &observed) ||
        !alternative_continuation_word_at(state, alternative, suspended_length,
                                          index, &expected) ||
        observed != expected)
      return false;
    observed_digest = rewrite_mix(observed_digest, observed, index);
  }
  if (observed_digest != expected_digest) return false;
  if (digest != nullptr) *digest = observed_digest;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t inquiry_surface_source_owner(
    const ResidentRewriteState* state, const Record& inquiry) {
  if (state == nullptr || inquiry.reserved[0] == 0u) return kInvalid;
  std::uint32_t owner = kInvalid;
  std::uint32_t source_revision = 0u;
  std::uint32_t source_digest = 0u;
  std::uint32_t recomputed_digest = 0u;
  for (std::uint32_t index = 0u; index < inquiry.reserved[0]; ++index) {
    const std::uint32_t slot = unique_owned_slot(
        state, kFormOpenInquirySurfaceTerm, inquiry.lane[1], index);
    if (slot == kInvalid) return kInvalid;
    const std::uint32_t candidate = state->records[slot].lane[4];
    if (candidate == 0u || candidate == kInvalid ||
        (owner != kInvalid && owner != candidate))
      return kInvalid;
    const Record& term = state->records[slot];
    if (term.lane[5] == 0u || term.lane[7] != kInquiryExternalWitness ||
        (source_revision != 0u && source_revision != term.lane[5]) ||
        (index != 0u && source_digest != term.lane[6]))
      return kInvalid;
    owner = candidate;
    source_revision = term.lane[5];
    source_digest = term.lane[6];
    recomputed_digest = rewrite_mix(recomputed_digest, term.lane[3], index);
  }
  return recomputed_digest == source_digest ? owner : kInvalid;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t inquiry_reply_source_owner(
    const ResidentRewriteState* state, const Record& inquiry) {
  if (state == nullptr) return kInvalid;
  ++state->oi_reply_source_owner_attempts;
  std::uint32_t owner = kInvalid;
  std::uint32_t witness_matches = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormOpenInquiryReplyWitness ||
        witness.lane[1] != inquiry.lane[1])
      continue;
    ++witness_matches;
    if (owner != kInvalid) {
      ++state->oi_reply_source_owner_decline_ambiguous;
      return kInvalid;
    }
    if (witness.lane[2] == 0u || witness.lane[2] == kInvalid) {
      ++state->oi_reply_source_owner_decline_witness_owner_bad;
      return kInvalid;
    }
    if (witness.lane[6] == 0u) {
      ++state->oi_reply_source_owner_decline_witness_lane6_zero;
      return kInvalid;
    }
    if (witness.lane[7] != kInquiryExternalWitness) {
      ++state->oi_reply_source_owner_decline_witness_not_external;
      return kInvalid;
    }
    if (!reply_terms_valid(state, inquiry, witness)) {
      ++state->oi_reply_source_owner_decline_reply_terms_invalid;
      return kInvalid;
    }
    owner = witness.lane[2];
  }
  if (owner == kInvalid) {
    if (witness_matches == 0u)
      ++state->oi_reply_source_owner_decline_no_witness_found;
    return kInvalid;
  }
  ++state->oi_reply_source_owner_found;
  return owner;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_episode_consequences(
    const ResidentRewriteState* state, const Record& inquiry,
    std::uint32_t consequences[2]) {
  if (state == nullptr || consequences == nullptr) return false;
  consequences[0] = kInvalid;
  consequences[1] = kInvalid;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    if (count == 2u) return false;
    consequences[count++] = binding.lane[4];
  }
  if (count != 2u || consequences[0] == consequences[1]) return false;
  if (consequences[1] < consequences[0]) {
    const std::uint32_t temporary = consequences[0];
    consequences[0] = consequences[1];
    consequences[1] = temporary;
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_has_exact_snapshots(
    const ResidentRewriteState* state, const Record& inquiry,
    const Record& suspended, bool require_live, std::uint32_t* selected_slot,
    bool require_enabled = true) {
  if (selected_slot != nullptr) *selected_slot = kInvalid;
  if (state == nullptr || inquiry.lane[4] != 2u) return false;
  std::uint32_t bindings = 0u;
  std::uint32_t selected = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    if (bindings == 2u) return false;
    std::uint32_t live_slot = kInvalid;
    if (require_live && !snapshot_still_live(state, suspended, binding,
                                              &live_slot, require_enabled))
      return false;
    if (binding.lane[2] == inquiry.lane[5]) {
      if (binding.lane[3] != inquiry.lane[6]) return false;
      selected = live_slot;
    }
    ++bindings;
  }
  if (bindings != 2u) return false;
  if (selected_slot != nullptr) *selected_slot = selected;
  return true;
}

#include "bcc32_resident_open_inquiry_lifecycle.inl"

// Called before ordinary END closes the external teacher contact.  A generated
// question cannot become teacher matter because only an un-emitted inquiry may
// capture this wholly external trajectory.
BCC32_OPEN_INQUIRY_HD inline bool capture_teacher_surface_before_end(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  ++state->oi_capture_surface_attempts;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  const std::uint32_t surface_slot = unique_current_trajectory(state);
  if (inquiry_slot == kInvalid || surface_slot == kInvalid) {
    // Purely local, read-only reclassification of which side was kInvalid
    // and why (no matching candidate vs. more than one -- ambiguous). This
    // recount does not alter the decision above; it only selects which
    // counter to bump for diagnostic visibility.
    if (inquiry_slot == kInvalid) {
      std::uint32_t candidates = 0u;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& candidate = state->records[slot];
        if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormOpenInquiry ||
            (candidate.lane[7] & kInquiryAwaitingReply) == 0u ||
            (candidate.lane[7] & kInquirySettled) != 0u)
          continue;
        ++candidates;
      }
      if (candidates == 0u)
        ++state->oi_capture_decline_no_active_inquiry;
      else
        ++state->oi_capture_decline_ambiguous_inquiry;
    }
    if (surface_slot == kInvalid) {
      std::uint32_t candidates = 0u;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& candidate = state->records[slot];
        if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormTrajectory ||
            candidate.lane[3] != 0u)
          continue;
        ++candidates;
      }
      if (candidates == 0u)
        ++state->oi_capture_decline_no_current_trajectory;
      else
        ++state->oi_capture_decline_ambiguous_trajectory;
    }
    return false;
  }
  Record& inquiry = state->records[inquiry_slot];
  const Record& surface = state->records[surface_slot];
  if ((inquiry.lane[7] & (kInquirySurfaceCaptured | kInquirySurfaceEmitted |
                          kInquiryReplyBound)) != 0u) {
    ++state->oi_capture_decline_already_progressed;
    return false;
  }
  if (inquiry.reserved[1] != kInvalid) {
    ++state->oi_capture_decline_reserved_pending;
    return false;
  }
  if (surface.lane[1] == inquiry.lane[2]) {
    ++state->oi_capture_decline_surface_owner_is_suspended;
    return false;
  }
  if (!wholly_external_trajectory(state, surface)) {
    ++state->oi_capture_decline_not_wholly_external;
    return false;
  }
  if (surface.lane[2] == 0u) {
    ++state->oi_capture_decline_surface_zero_length;
    return false;
  }
  if (surface.lane[2] > kMaximumSurfaceWords) {
    ++state->oi_capture_decline_surface_too_long;
    return false;
  }
  if (free_record_count(state) < surface.lane[2]) {
    ++state->oi_capture_decline_insufficient_free_records;
    return false;
  }
  for (std::uint32_t index = 0u; index < surface.lane[2]; ++index) {
    std::uint32_t ignored = 0u;
    if (!trajectory_word_at(state, surface.lane[1], index, &ignored)) return false;
  }
  for (std::uint32_t index = 0u; index < surface.lane[2]; ++index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    std::uint32_t word = 0u;
    (void)trajectory_word_at(state, surface.lane[1], index, &word);
    Record& term = state->records[slot];
    term.lane[0] = kFormOpenInquirySurfaceTerm;
    term.lane[1] = inquiry.lane[1];
    term.lane[2] = index;
    term.lane[3] = word;
    term.lane[4] = surface.lane[1];
    term.lane[5] = surface.revision;
    term.lane[6] = surface.lane[6];
    term.lane[7] = kInquiryExternalWitness;
    ++term.revision;
  }
  inquiry.reserved[0] = surface.lane[2];
  inquiry.lane[7] |= kInquirySurfaceCaptured;
  ++inquiry.revision;
  const std::uint32_t surface_owner = surface.lane[1];
  mixed_provenance::clear_provenance(state, surface_owner);
  clear_trajectory(state, surface_slot);
  ++state->revision;
  refresh_receipt(state);
  ++state->oi_capture_surface_admitted;
  return true;
}

// A completed self-generated question is public history, not the later
// external reply contact.  On the first physical reply word, retire only that
// exactly attributed generated trajectory so the ordinary ingress path creates
// a fresh wholly external owner.  This hook never inspects the reply value.
BCC32_OPEN_INQUIRY_HD inline bool emitted_surface_length(
    const ResidentRewriteState* state, const Record& inquiry,
    const Record& constructor, const Record& suspended, std::uint32_t* length) {
  if (state == nullptr || length == nullptr || constructor.lane[2] == 0u ||
      constructor.lane[2] > kMaximumSurfaceWords)
    return false;
  AlternativeSnapshot alternatives[2]{};
  if (!collect_exact_fork(state, suspended, alternatives)) return false;
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t term_slot = unique_owned_slot(
        state, kFormOpenInquiryConstructorTerm, constructor.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    if (term.lane[3] == kTermLiteral) {
      ++total;
    } else if (term.lane[3] == kTermFirstAlternative ||
               term.lane[3] == kTermSecondAlternative) {
      std::uint32_t label_length = 0u;
      std::uint32_t label_digest = 0u;
      const AlternativeSnapshot& alternative =
          term.lane[3] == kTermFirstAlternative ? alternatives[0] : alternatives[1];
      if (term.lane[4] != 0u ||
          !alternative_continuation_digest(state, alternative, inquiry.lane[3],
                                           &label_length, &label_digest))
        return false;
      total += label_length;
    } else {
      return false;
    }
    if (total > kMaximumSurfaceWords) return false;
  }
  *length = total;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool
detach_emitted_surface_before_external_reply(ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  const std::uint32_t current_slot = unique_current_trajectory(state);
  if (inquiry_slot == kInvalid || current_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  const Record& current = state->records[current_slot];
  const std::uint32_t constructor_slot = inquiry.reserved[1];
  if ((inquiry.lane[7] & kInquirySurfaceEmitted) == 0u ||
      (inquiry.lane[7] & kInquiryReplyBound) != 0u ||
      constructor_slot == kInvalid ||
      constructor_slot >= live_record_capacity(state) ||
      state->records[constructor_slot].lane[1] != inquiry.lane[5] ||
      state->records[constructor_slot].revision != inquiry.lane[6] ||
      !inquiry_constructor_authoritative(state, constructor_slot) ||
      current.lane[5] != 0u ||
      (current.lane[7] & kTrajectoryHasGenerated) == 0u ||
      !mixed_provenance::tagged_history(state, current))
    return false;
  std::uint32_t expected_length = 0u;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid ||
      !emitted_surface_length(state, inquiry, state->records[constructor_slot],
                              state->records[suspended_slot], &expected_length) ||
      current.lane[2] != expected_length)
    return false;
  for (std::uint32_t index = 0u; index < current.lane[2]; ++index) {
    mixed_provenance::Origin origin = mixed_provenance::Origin::external;
    std::uint32_t producer = kInvalid;
    if (!mixed_provenance::origin_at(state, current, index, &origin,
                                     &producer) ||
        origin != mixed_provenance::Origin::generated ||
        producer != constructor_slot)
      return false;
  }
  const std::uint32_t owner = current.lane[1];
  mixed_provenance::clear_provenance(state, owner);
  clear_trajectory(state, current_slot);
  // The learned Constructor was needed to generate and validate the surface,
  // but it is not an answer selection.  Once the surface is physically
  // detached, release these lanes for the later external alternative witness.
  // reserved[1] keeps the Constructor slot only for validating the learned
  // reply topology; lane[5]/lane[6] must not make an unanswered inquiry look
  // as though one alternative had already been chosen.
  inquiry.lane[5] = kInvalid;
  inquiry.lane[6] = kInvalid;
  ++inquiry.revision;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

// The legacy reply wrapper is byte-topological: optional homogeneous literal
// spans around one selected first word. Exact multiword continuations are the
// second admitted shape below; neither route contains a token or semantic role.
BCC32_OPEN_INQUIRY_HD inline bool reply_shape_from_words(
    const std::uint32_t* words, std::uint32_t count, std::uint32_t chosen,
    bool require_recurrent, std::uint32_t* flags, std::uint32_t* prefix,
    std::uint32_t* suffix) {
  if (flags == nullptr || prefix == nullptr || suffix == nullptr ||
      words == nullptr || count == 0u || count > kMaximumSurfaceWords ||
      chosen == kInvalid)
    return false;
  std::uint32_t chosen_index = kInvalid;
  for (std::uint32_t index = 0u; index < count; ++index) {
    if (words[index] != chosen) continue;
    if (chosen_index != kInvalid) return false;
    chosen_index = index;
  }
  if (chosen_index == kInvalid) return false;
  std::uint32_t result = 0u;
  *prefix = 0u;
  *suffix = 0u;
  if (chosen_index != 0u) {
    if (require_recurrent && chosen_index < 2u) return false;
    *prefix = words[0];
    for (std::uint32_t index = 1u; index < chosen_index; ++index)
      if (words[index] != *prefix) return false;
    result |= kReplyTemplatePrefix;
  }
  const std::uint32_t suffix_count = count - chosen_index - 1u;
  if (suffix_count != 0u) {
    if (require_recurrent && suffix_count < 2u) return false;
    *suffix = words[chosen_index + 1u];
    for (std::uint32_t index = chosen_index + 2u; index < count; ++index)
      if (words[index] != *suffix) return false;
    result |= kReplyTemplateSuffix;
  }
  *flags = result;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool reply_shape_from_trajectory(
    const ResidentRewriteState* state, const Record& reply,
    std::uint32_t chosen, bool require_recurrent, std::uint32_t* flags,
    std::uint32_t* prefix, std::uint32_t* suffix) {
  if (state == nullptr || reply.lane[2] == 0u ||
      reply.lane[2] > kMaximumSurfaceWords)
    return false;
  std::uint32_t words[kMaximumSurfaceWords]{};
  for (std::uint32_t index = 0u; index < reply.lane[2]; ++index)
    if (!trajectory_word_at(state, reply.lane[1], index, &words[index]))
      return false;
  return reply_shape_from_words(words, reply.lane[2], chosen, require_recurrent,
                                flags, prefix, suffix);
}

BCC32_OPEN_INQUIRY_HD inline bool reply_terms_valid(
    const ResidentRewriteState* state, const Record& inquiry,
    const Record& witness, std::uint32_t* chosen_index) {
  if (chosen_index != nullptr) *chosen_index = kInvalid;
  if (state == nullptr) return false;
  ++state->oi_reply_terms_attempts;
  if (witness.lane[0] != kFormOpenInquiryReplyWitness ||
      witness.lane[1] != inquiry.lane[1] || witness.reserved[0] == 0u ||
      witness.reserved[0] > kMaximumSurfaceWords ||
      witness.lane[7] != kInquiryExternalWitness || witness.revision != 1u) {
    ++state->oi_reply_terms_decline_witness_guard;
    return false;
  }
  std::uint32_t words[kMaximumSurfaceWords]{};
  for (std::uint32_t index = 0u; index < witness.reserved[0]; ++index) {
    const std::uint32_t term_slot = unique_owned_slot(
        state, kFormOpenInquiryReplyTerm, inquiry.lane[1], index);
    if (term_slot == kInvalid) {
      ++state->oi_reply_terms_decline_term_lookup_failed;
      return false;
    }
    const Record& term = state->records[term_slot];
    if (term.lane[4] != witness.lane[2]) {
      ++state->oi_reply_terms_decline_term_lane4_owner_mismatch;
      return false;
    }
    if (term.lane[5] != witness.lane[6]) {
      ++state->oi_reply_terms_decline_term_lane5_revision_mismatch;
      return false;
    }
    if (term.lane[6] != witness.reserved[0]) {
      ++state->oi_reply_terms_decline_term_lane6_length_mismatch;
      return false;
    }
    if (term.lane[7] != kInquiryExternalWitness) {
      ++state->oi_reply_terms_decline_term_not_external;
      return false;
    }
    if (term.revision != 1u) {
      ++state->oi_reply_terms_decline_term_revision_not_one;
      state->oi_reply_terms_last_bad_term_revision = term.revision;
      state->oi_reply_terms_last_bad_term_slot = term_slot;
      return false;
    }
    words[index] = term.lane[3];
  }
  std::uint32_t term_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 != 0u && term.lane[0] == kFormOpenInquiryReplyTerm &&
        term.lane[1] == inquiry.lane[1])
      ++term_count;
  }
  if (term_count != witness.reserved[0]) {
    ++state->oi_reply_terms_decline_term_count_mismatch;
    return false;
  }
  std::uint32_t flags = 0u;
  std::uint32_t prefix = 0u;
  std::uint32_t suffix = 0u;
  if (reply_shape_from_words(words, witness.reserved[0], witness.lane[5], false,
                             &flags, &prefix, &suffix)) {
    for (std::uint32_t index = 0u; index < witness.reserved[0]; ++index)
      if (words[index] == witness.lane[5]) {
        if (chosen_index != nullptr) *chosen_index = index;
        ++state->oi_reply_terms_fastpath_matched;
        return true;
      }
  }
  std::uint32_t selected_slot = kInvalid;
  bool program_ambiguous = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        program.lane[1] != witness.lane[3])
      continue;
    if (program.revision != witness.lane[4] ||
        !resident_program_authoritative(state, slot))
      continue;
    if (selected_slot != kInvalid) {
      ++state->oi_reply_terms_decline_program_ambiguous;
      program_ambiguous = true;
      break;
    }
    selected_slot = slot;
  }
  if (program_ambiguous) return false;
  if (selected_slot == kInvalid) {
    ++state->oi_reply_terms_decline_program_not_found;
    return false;
  }
  {
    AlternativeSnapshot selected{witness.lane[3], witness.lane[4],
                                 witness.lane[5], selected_slot};
    std::uint32_t length = 0u;
    std::uint32_t digest = 0u;
    bool exact = alternative_continuation_digest(
        state, selected, inquiry.lane[3], &length, &digest);
    exact = exact && length == witness.reserved[0];
    for (std::uint32_t index = 0u; exact && index < length; ++index) {
      std::uint32_t expected = 0u;
      exact = alternative_continuation_word_at(
                  state, selected, inquiry.lane[3], index, &expected) &&
              words[index] == expected;
    }
    if (exact) {
      if (chosen_index != nullptr) *chosen_index = 0u;
      ++state->oi_reply_terms_program_exact_matched;
      return true;
    }
    ++state->oi_reply_terms_decline_program_exact_mismatch;
  }
  return false;
}

// Accept an exact continuation or the learned one-word wrapper.
BCC32_OPEN_INQUIRY_HD inline bool bind_one_fresh_external_reply_before_end(
    ResidentRewriteState* state, InquiryView* view = nullptr) {
  if (view != nullptr) *view = InquiryView{};
  if (state == nullptr || state->fault != 0u) return false;
  ++state->oi_bind_reply_attempts;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  const std::uint32_t reply_slot = unique_current_trajectory(state);
  if (inquiry_slot == kInvalid || reply_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  const Record& reply = state->records[reply_slot];
  if ((inquiry.lane[7] & (kInquirySurfaceCaptured | kInquirySurfaceEmitted)) == 0u ||
      (inquiry.lane[7] & kInquiryReplyBound) != 0u || reply.lane[2] == 0u ||
      reply.lane[2] > kMaximumSurfaceWords ||
      !wholly_external_trajectory(state, reply) ||
      ((inquiry.lane[7] & kInquirySurfaceCaptured) != 0u &&
       inquiry_surface_source_owner(state, inquiry) == reply.lane[1]))
    return false;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid) return false;
  const Record& suspended = state->records[suspended_slot];
  if (suspended.lane[3] == 0u || suspended.lane[2] != inquiry.lane[3] ||
      (suspended.lane[7] & kTrajectoryOpenInquiry) == 0u)
    return false;
  std::uint32_t matches = 0u;
  std::uint32_t chosen_owner = kInvalid;
  std::uint32_t chosen_revision = 0u;
  std::uint32_t chosen_slot = kInvalid;
  std::uint32_t chosen_word = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    std::uint32_t program_slot = kInvalid;
    if (!snapshot_still_live(state, suspended, binding, &program_slot)) return false;
    AlternativeSnapshot alternative{};
    alternative.owner = binding.lane[2];
    alternative.revision = binding.lane[3];
    alternative.consequence = binding.lane[4];
    alternative.slot = program_slot;
    std::uint32_t candidate_digest = 0u;
    const bool exact = reply_matches_continuation(
        state, reply, alternative, suspended.lane[2], &candidate_digest);
    std::uint32_t reply_flags = 0u;
    std::uint32_t reply_prefix = 0u;
    std::uint32_t reply_suffix = 0u;
    const bool wrapped = reply_shape_from_trajectory(
        state, reply, alternative.consequence, false, &reply_flags,
        &reply_prefix, &reply_suffix);
    if (exact || wrapped) {
      if ((inquiry.lane[7] & kInquirySurfaceCaptured) != 0u &&
          !continuation_occurs_once_in_surface(state, inquiry, alternative,
                                               suspended.lane[2]))
        return false;
      if ((inquiry.lane[7] & kInquirySurfaceEmitted) != 0u) {
        const std::uint32_t constructor_slot = inquiry.reserved[1];
        if (constructor_slot >= live_record_capacity(state)) return false;
        const Record& constructor = state->records[constructor_slot];
        const bool exact_allowed =
            exact && constructor.lane[6] == 0u &&
            constructor.reserved[0] == 0u && constructor.reserved[1] == 0u;
        const bool wrapped_allowed =
            wrapped && constructor.lane[6] == reply_flags &&
            ((reply_flags & kReplyTemplatePrefix) == 0u ||
             constructor.reserved[0] == reply_prefix) &&
            ((reply_flags & kReplyTemplateSuffix) == 0u ||
             constructor.reserved[1] == reply_suffix);
        if (!exact_allowed && !wrapped_allowed) continue;
      }
      ++matches;
      chosen_owner = binding.lane[2];
      chosen_revision = binding.lane[3];
      chosen_slot = program_slot;
      chosen_word = alternative.consequence;
    }
  }
  if (matches != 1u || chosen_slot == kInvalid) return false;
  // The fresh reply selected one exact resident continuation.  Keep that
  // selection as an inquiry-owned emission cursor instead of asking the
  // generic candidate collector to rediscover it from the still-ambiguous
  // VersionSpace population.  Equivalent Programs may legitimately coexist;
  // disabling whichever slots happened to be visible here would mutate
  // knowledge globally and still leave allocation-order duplicates live.
  if (free_record_count(state) < reply.lane[2] + 2u)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    if (state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormOpenInquiryEmission)
      return false;
  for (std::uint32_t index = 0u; index < reply.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!trajectory_word_at(state, reply.lane[1], index, &word)) return false;
  }
  const std::uint32_t witness_slot = allocate_record(state);
  if (witness_slot == kInvalid) return false;
  Record& witness = state->records[witness_slot];
  witness.lane[0] = kFormOpenInquiryReplyWitness;
  witness.lane[1] = inquiry.lane[1];
  witness.lane[2] = reply.lane[1];
  witness.lane[3] = chosen_owner;
  witness.lane[4] = chosen_revision;
  witness.lane[5] = chosen_word;
  witness.lane[6] = reply.revision;
  witness.lane[7] = kInquiryExternalWitness;
  witness.reserved[0] = reply.lane[2];
  witness.reserved[1] = reply.lane[6];
  ++witness.revision;
  for (std::uint32_t index = 0u; index < reply.lane[2]; ++index) {
    const std::uint32_t term_slot = allocate_record(state);
    if (term_slot == kInvalid) return false;
    Record& term = state->records[term_slot];
    term.lane[0] = kFormOpenInquiryReplyTerm;
    term.lane[1] = inquiry.lane[1];
    term.lane[2] = index;
    if (!trajectory_word_at(state, reply.lane[1], index, &term.lane[3]))
      return false;
    term.lane[4] = reply.lane[1];
    term.lane[5] = reply.revision;
    term.lane[6] = reply.lane[2];
    term.lane[7] = kInquiryExternalWitness;
    ++term.revision;
  }
  const std::uint32_t emission_slot = allocate_record(state);
  if (emission_slot == kInvalid) return false;
  Record& emission = state->records[emission_slot];
  emission.lane[0] = kFormOpenInquiryEmission;
  emission.lane[1] = inquiry.lane[1];
  emission.lane[2] = chosen_slot;
  emission.lane[3] = 0u;
  emission.lane[4] = chosen_owner;
  emission.lane[5] = chosen_revision;
  emission.lane[6] = kInquiryEmissionReplyContinuation;
  ++emission.revision;
  inquiry.lane[5] = chosen_owner;
  inquiry.lane[6] = chosen_revision;
  inquiry.lane[7] |= kInquiryReplyBound;
  ++inquiry.revision;
  ++state->revision;
  refresh_receipt(state);
  if (view != nullptr) {
    view->inquiry_slot = inquiry_slot;
    view->suspended_slot = suspended_slot;
    view->reply_slot = reply_slot;
    view->matching_alternative_slot = chosen_slot;
  }
  ++state->oi_bind_reply_admitted;
  return true;
}

// Called before ordinary END.  It does not allocate.  Every validation occurs
// before changing a Program flag or reactivating the exact suspended owner.
BCC32_OPEN_INQUIRY_HD inline bool settle_one_at_end(
    ResidentRewriteState* state, InquiryView* view = nullptr) {
  if (view != nullptr) *view = InquiryView{};
  if (state == nullptr || state->fault != 0u ||
      state->open_inquiry_public_return_pending != 0u)
    return false;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  const std::uint32_t reply_slot = unique_current_trajectory(state);
  if (inquiry_slot == kInvalid || reply_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  const Record& reply = state->records[reply_slot];
  if ((inquiry.lane[7] & kInquiryReplyBound) == 0u ||
      !wholly_external_trajectory(state, reply))
    return false;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid) return false;
  Record& suspended = state->records[suspended_slot];
  if (suspended.lane[3] == 0u || suspended.lane[2] != inquiry.lane[3] ||
      (suspended.lane[7] & kTrajectoryOpenInquiry) == 0u)
    return false;
  std::uint32_t witness_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormOpenInquiryReplyWitness ||
        witness.lane[1] != inquiry.lane[1])
      continue;
    const bool terms_valid = reply_terms_valid(state, inquiry, witness);
    if (witness.lane[2] != reply.lane[1] || witness.lane[3] != inquiry.lane[5] ||
        witness.lane[4] != inquiry.lane[6] ||
        witness.lane[6] != reply.revision ||
        witness.reserved[1] != reply.lane[6] ||
        !terms_valid)
      return false;
    ++witness_count;
  }
  if (witness_count != 1u) return false;
  std::uint32_t selected_slot = kInvalid;
  if (!inquiry_has_exact_snapshots(state, inquiry, suspended, true,
                                   &selected_slot) ||
      selected_slot == kInvalid)
    return false;
  AlternativeSnapshot live[2]{};
  if (!collect_exact_fork(state, suspended, live))
    return false;
  bool selected_seen = false;
  for (std::uint32_t index = 0u; index < 2u; ++index)
    selected_seen |= live[index].owner == inquiry.lane[5] &&
                     live[index].revision == inquiry.lane[6];
  if (!selected_seen)
    return false;

  std::uint32_t binding_slots[2]{kInvalid, kInvalid};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    bool matched = false;
    for (std::uint32_t index = 0u; index < 2u; ++index) {
      if (binding.lane[2] != live[index].owner ||
          binding.lane[3] != live[index].revision ||
          binding.lane[4] != live[index].consequence)
        continue;
      if (binding_slots[index] != kInvalid) return false;
      binding_slots[index] = slot;
      matched = true;
    }
    if (!matched) return false;
  }
  if (binding_slots[0] == kInvalid || binding_slots[1] == kInvalid)
    return false;

  std::uint32_t selected_word = kInvalid;
  for (std::uint32_t index = 0u; index < 2u; ++index)
    if (live[index].owner == inquiry.lane[5] &&
        live[index].revision == inquiry.lane[6]) {
      selected_word = live[index].consequence;
    }
  if (selected_word == kInvalid) return false;

  // Inquiry resolution is local to this suspended episode.  Both resident
  // alternatives remain authoritative so later contact can still expose the
  // original uncertainty, revise either branch, or teach a different reply.
  // The reply has already become an exact zero-authority witness. Retire its
  // source trajectory before generic END processing so it cannot be promoted
  // as an unrelated carried prefix or pay ordinary Program support.
  const std::uint32_t reply_owner = reply.lane[1];
  mixed_provenance::clear_provenance(state, reply_owner);
  clear_trajectory(state, reply_slot);
  inquiry.lane[7] = (inquiry.lane[7] & ~kInquiryAwaitingReply) |
                    kInquirySettled;
  ++inquiry.revision;
  ++state->revision;
  refresh_receipt(state);
  if (view != nullptr) {
    view->inquiry_slot = inquiry_slot;
    view->suspended_slot = suspended_slot;
    view->reply_slot = reply_slot;
    view->matching_alternative_slot = selected_slot;
  }
  return true;
}

// Ordinary END must close the fresh reply owner before the exact suspended
// prefix becomes current again.  This post-END step prevents two current
// trajectories and makes re-entry depend on the physical reply boundary.
BCC32_OPEN_INQUIRY_HD inline bool reactivate_settled_suspended_after_end(
    ResidentRewriteState* state) {
  bool ambiguous_current = false;
  const std::uint32_t current =
      unique_current_trajectory(state, &ambiguous_current);
  if (state == nullptr || state->fault != 0u || ambiguous_current ||
      current != kInvalid)
    return false;
  ++state->oi_reactivate_attempts;
  std::uint32_t inquiry_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& inquiry = state->records[slot];
    if (inquiry.matter_q8 == 0u || inquiry.lane[0] != kFormOpenInquiry ||
        (inquiry.lane[7] & kInquirySettled) == 0u ||
        (inquiry.lane[7] & kInquiryResumed) != 0u)
      continue;
    if (inquiry_slot != kInvalid) return false;
    inquiry_slot = slot;
  }
  if (inquiry_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid) return false;
  Record& suspended = state->records[suspended_slot];
  if (suspended.lane[3] == 0u || suspended.lane[2] != inquiry.lane[3] ||
      (suspended.lane[7] & kTrajectoryOpenInquiry) == 0u ||
      !inquiry_has_exact_snapshots(state, inquiry, suspended, true, nullptr,
                                   false))
    return false;
  suspended.lane[3] = 0u;
  // The selected reply is the new physical permission to continue the exact
  // yielded prefix. The inquiry-owned emission cursor still appends every
  // generated word to this ordinary current trajectory, so the yielded latch
  // remains the physical resume boundary rather than an observer-side choice.
  suspended.lane[4] = 1u;
  suspended.lane[7] =
      (suspended.lane[7] & ~kTrajectoryOpenInquiry) |
      kTrajectoryVersionSpaceSelected;
  ++suspended.revision;
  ++state->revision;
  refresh_receipt(state);
  ++state->oi_reactivate_admitted;
  return true;
}

// A distributed public word has no producer Record.  It may nevertheless
// complete an already-bound inquiry when the common emitter's receipt still
// agrees with the live trajectory and the complete relation accounting.  The
// receipt is deliberately checked as a cross-field integrity witness here;
// it is not promoted into an inquiry-owned Record or a semantic answer cell.
BCC32_OPEN_INQUIRY_HD inline bool distributed_emission_matches_inquiry(
    const ResidentRewriteState* state, const Record& inquiry,
    const Record& yielded_prefix, std::uint32_t* selected_word) {
  if (selected_word != nullptr) *selected_word = kInvalid;
  if (state == nullptr || selected_word == nullptr ||
      state->generated_word_valid == 0u || state->generated_locus != kInvalid ||
      state->generated_receipt_valid == 0u ||
      state->causal_relation_generated_events == 0u ||
      state->causal_relation_component_digest == 0u ||
      state->causal_relation_component_revision_digest == 0u ||
      state->causal_relation_external_provenance_digest == 0u ||
      state->causal_relation_external_leaves == 0u ||
      state->generated_receipt_owner == 0u ||
      state->generated_receipt_owner == kInvalid ||
      state->generated_receipt_participant_records == 0u ||
      state->generated_receipt_external_leaves == 0u ||
      state->generated_receipt_provenance_digest == 0u ||
      state->generated_receipt_independent_sources < 2u ||
      state->generated_receipt_source_contributions < 2u ||
      state->generated_receipt_topology_digest !=
          state->causal_relation_component_digest ||
      state->generated_receipt_revision_digest !=
          state->causal_relation_component_revision_digest ||
      state->generated_receipt_participation_digest !=
          causal_relation_mix(
              causal_relation_mix(state->generated_receipt_topology_digest,
                                  state->generated_receipt_revision_digest),
              state->generated_receipt_provenance_digest) ||
      state->generated_receipt_participant_records !=
          state->causal_relation_participating_records ||
      state->generated_receipt_independent_sources !=
          state->causal_relation_independent_sources ||
      state->generated_receipt_source_contributions !=
          state->causal_relation_source_contributions)
    return false;

  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid ||
      state->records[current_slot].lane[1] != state->generated_receipt_owner ||
      state->records[current_slot].lane[2] == 0u ||
      state->records[current_slot].lane[3] != 0u ||
      !mixed_provenance::tagged_history(state, state->records[current_slot]))
    return false;
  mixed_provenance::Origin origin = mixed_provenance::Origin::external;
  std::uint32_t producer = kInvalid;
  if (!mixed_provenance::origin_at(
          state, state->records[current_slot],
          state->records[current_slot].lane[2] - 1u, &origin, &producer) ||
      origin != mixed_provenance::Origin::distributed || producer != kInvalid)
    return false;

  // The inquiry's selected alternative remains a contextual witness.  Its
  // Program slot is revalidated only to prove that the fresh reply selected a
  // live continuation; it is never used as the distributed emission locus.
  if (inquiry.lane[5] == kInvalid || inquiry.lane[6] == 0u ||
      inquiry.lane[6] == kInvalid)
    return false;
  std::uint32_t selected_bindings = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1] ||
        binding.lane[2] != inquiry.lane[5] ||
        binding.lane[3] != inquiry.lane[6])
      continue;
    std::uint32_t program_slot = kInvalid;
    if (!snapshot_still_live(state, yielded_prefix, binding, &program_slot))
      return false;
    ++selected_bindings;
    if (binding.lane[4] == kInvalid ||
        binding.lane[4] != state->generated_word)
      return false;
    *selected_word = binding.lane[4];
  }
  return selected_bindings == 1u;
}

// The root calls this only after ordinary resident execution produced a word.
// It records the causal result of a training episode; held-out generated
// questions carry kInquirySurfaceEmitted and can never be construction support.
BCC32_OPEN_INQUIRY_HD inline bool observe_resumed_consequence(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u || state->generated_word_valid == 0u)
    return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& inquiry = state->records[slot];
    if (inquiry.matter_q8 != 0u && inquiry.lane[0] == kFormOpenInquiry &&
        (inquiry.lane[7] & kInquirySettled) != 0u &&
        (inquiry.lane[7] & kInquiryResumed) == 0u &&
        (inquiry.lane[7] &
         (kInquirySurfaceCaptured | kInquirySurfaceEmitted)) != 0u) {
      if (found != kInvalid) return false;
      found = slot;
    }
  }
  if (found == kInvalid) return false;
  Record& inquiry = state->records[found];
  const std::uint32_t resumed_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (resumed_slot == kInvalid || state->records[resumed_slot].lane[3] != 0u ||
      (state->records[resumed_slot].lane[7] &
       kTrajectoryVersionSpaceSelected) == 0u)
    return false;
  Record yielded_prefix = state->records[resumed_slot];
  yielded_prefix.lane[2] = inquiry.lane[3];
  const bool distributed_emission =
      state->generated_locus == kInvalid &&
      state->generated_receipt_valid != 0u;
  std::uint32_t selected_slot = kInvalid;
  AlternativeSnapshot selected{};
  std::uint32_t selected_length = 0u;
  std::uint32_t selected_digest = 0u;
  std::uint32_t selected_word = kInvalid;
  if (distributed_emission) {
    if (!distributed_emission_matches_inquiry(
            state, inquiry, yielded_prefix, &selected_word))
      return false;
  } else {
    selected_slot = state->generated_locus;
    if (selected_slot >= live_record_capacity(state) ||
        state->records[selected_slot].matter_q8 == 0u ||
        state->records[selected_slot].lane[0] != kFormProgram ||
        state->records[selected_slot].lane[1] != inquiry.lane[5] ||
        state->records[selected_slot].revision != inquiry.lane[6])
      return false;
    selected = AlternativeSnapshot{inquiry.lane[5], inquiry.lane[6], kInvalid,
                                   selected_slot};
    if (!live_resident_continuation_alternative(
            state, selected_slot, yielded_prefix, &selected) ||
        selected.owner != inquiry.lane[5] ||
        selected.revision != inquiry.lane[6] ||
        causal_product_has_live_counterevidence(state, selected_slot))
      return false;
    if (!alternative_continuation_digest(
            state, selected, inquiry.lane[3], &selected_length,
            &selected_digest) ||
        !alternative_continuation_word_at(state, selected, inquiry.lane[3], 0u,
                                          &selected_word))
      return false;
  }
  const bool continuation_emitted =
      (inquiry.lane[7] & kInquiryReplyContinuationEmitted) != 0u;
  if (distributed_emission && continuation_emitted) return false;
  const std::uint32_t emission_slot = continuation_emitted
                                          ? unique_reply_continuation_emission(
                                                state, inquiry)
                                          : kInvalid;
  if (distributed_emission) {
    // The distributed writer emitted the selected consequence itself.  There
    // is no Program continuation cursor to inspect and no singleton locus to
    // promote; the receipt and selected inquiry witness above are the only
    // admissible evidence for this completion.
  } else if (continuation_emitted) {
    if (emission_slot == kInvalid ||
        state->records[emission_slot].lane[2] != selected_slot ||
        state->records[emission_slot].lane[3] != selected_length ||
        state->records[emission_slot].lane[4] != selected.owner ||
        state->records[emission_slot].lane[5] != selected.revision ||
        selected_length == 0u)
      return false;
    for (std::uint32_t index = 0u; index < selected_length; ++index) {
      std::uint32_t expected = 0u;
      std::uint32_t observed = 0u;
      if (!alternative_continuation_word_at(state, selected, inquiry.lane[3],
                                            index, &expected) ||
          !trajectory_word_at(state, state->records[resumed_slot].lane[1],
                              inquiry.lane[3] + index, &observed) ||
          expected != observed)
        return false;
    }
    std::uint32_t last = 0u;
    if (!alternative_continuation_word_at(state, selected, inquiry.lane[3],
                                          selected_length - 1u, &last) ||
        last != state->generated_word)
      return false;
  } else {
    // The legacy path emits one selected continuation byte through ordinary
    // Program execution.  Its authority was already checked through the same
    // continuation helper as an exact response emission.
    if (selected_length != 1u || selected_word != state->generated_word ||
        free_record_count(state) == 0u)
      return false;
  }

  const bool training_episode =
      (inquiry.lane[7] & kInquirySurfaceCaptured) != 0u &&
      (inquiry.lane[7] & kInquirySurfaceEmitted) == 0u;
  std::uint32_t binding_count = 0u;
  for (std::uint32_t binding_slot = 0u;
       training_episode && binding_slot < live_record_capacity(state); ++binding_slot) {
    const Record& binding = state->records[binding_slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    ++binding_count;
    if (binding.lane[2] == inquiry.lane[5]) {
      if (binding.lane[3] != inquiry.lane[6] ||
          binding.lane[4] != selected_word)
        return false;
      continue;
    }
    std::uint32_t program_slot = kInvalid;
    if (!snapshot_still_live(state, yielded_prefix, binding, &program_slot))
      return false;
  }
  if (training_episode && binding_count != 2u)
    return false;
  const std::uint32_t witness_slot = continuation_emitted
                                         ? emission_slot
                                         : allocate_record(state);
  if (witness_slot == kInvalid) return false;
  Record& witness = state->records[witness_slot];
  witness.lane[0] = kFormOpenInquiryResumeWitness;
  witness.lane[1] = inquiry.lane[1];
  witness.lane[2] = inquiry.lane[5];
  witness.lane[3] = inquiry.lane[6];
  witness.lane[4] = selected_word;
  witness.lane[5] = 0u;
  witness.lane[6] = 0u;
  witness.lane[7] = 0u;
  witness.reserved[0] = 0u;
  witness.reserved[1] = 0u;
  ++witness.revision;
  inquiry.lane[7] |= kInquiryResumed | kInquiryComplete;
  ++inquiry.revision;
  Record& resumed = state->records[resumed_slot];
  resumed.lane[7] &= ~kTrajectoryVersionSpaceSelected;
  ++resumed.revision;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool episode_alternatives(
    const ResidentRewriteState* state, const Record& inquiry,
    AlternativeSnapshot alternatives[2]) {
  if (state == nullptr || alternatives == nullptr || inquiry.lane[4] != 2u)
    return false;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    if (count == 2u) return false;
    std::uint32_t program_slot = kInvalid;
    if (!snapshot_program_still_grounded(state, binding, inquiry.lane[3],
                                         &program_slot))
      return false;
    alternatives[count++] = {binding.lane[2], binding.lane[3], binding.lane[4],
                             program_slot};
  }
  if (count != 2u) return false;
  for (std::uint32_t index = 0u; index < inquiry.lane[3]; ++index) {
    std::uint32_t first_prefix = 0u;
    std::uint32_t second_prefix = 0u;
    std::uint32_t first_meta = 0u;
    std::uint32_t second_meta = 0u;
    const Record& first_program = state->records[alternatives[0].slot];
    const Record& second_program = state->records[alternatives[1].slot];
    const bool first_valid =
        (first_program.lane[7] & kProgramFlagVersionSpace) != 0u
            ? version_space_effective_program_term_at(
                  state, first_program, index, &first_prefix, &first_meta)
            : resident_program_term_at(state, alternatives[0].slot, index,
                                       &first_prefix, &first_meta);
    const bool second_valid =
        (second_program.lane[7] & kProgramFlagVersionSpace) != 0u
            ? version_space_effective_program_term_at(
                  state, second_program, index, &second_prefix, &second_meta)
            : resident_program_term_at(state, alternatives[1].slot, index,
                                       &second_prefix, &second_meta);
    if (!first_valid || !second_valid ||
        first_meta != 0u || second_meta != 0u ||
        first_prefix != second_prefix)
      return false;
  }
  std::uint32_t first = 0u, second = 0u;
  if (!alternative_continuation_word_at(state, alternatives[0], inquiry.lane[3],
                                        0u, &first) ||
      !alternative_continuation_word_at(state, alternatives[1], inquiry.lane[3],
                                        0u, &second))
    return false;
  if (second < first ||
      (second == first &&
       alternatives[1].owner < alternatives[0].owner)) {
    const AlternativeSnapshot temporary = alternatives[0];
    alternatives[0] = alternatives[1];
    alternatives[1] = temporary;
  }
  return !alternative_continuations_equal(state, alternatives[0], alternatives[1],
                                          inquiry.lane[3]);
}

BCC32_OPEN_INQUIRY_HD inline bool find_continuation_once(
    const ResidentRewriteState* state, const Record& inquiry,
    const AlternativeSnapshot& alternative, std::uint32_t* start,
    std::uint32_t* length) {
  if (start == nullptr || length == nullptr || inquiry.reserved[0] == 0u)
    return false;
  std::uint32_t digest = 0u;
  if (!alternative_continuation_digest(state, alternative, inquiry.lane[3], length,
                                       &digest) || *length == 0u ||
      *length > inquiry.reserved[0])
    return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t candidate = 0u;
       candidate + *length <= inquiry.reserved[0]; ++candidate) {
    bool exact = true;
    for (std::uint32_t index = 0u; index < *length; ++index) {
      std::uint32_t surface_word = 0u;
      std::uint32_t label_word = 0u;
      if (!inquiry_surface_word_at(state, inquiry, candidate + index, &surface_word) ||
          !alternative_continuation_word_at(state, alternative, inquiry.lane[3],
                                            index, &label_word) ||
          surface_word != label_word) {
        exact = false;
        break;
      }
    }
    if (!exact) continue;
    if (found != kInvalid) return false;
    found = candidate;
  }
  if (found == kInvalid) return false;
  *start = found;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool reply_witness_matches_continuation(
    const ResidentRewriteState* state, const Record& inquiry,
    const AlternativeSnapshot alternatives[2]) {
  std::uint32_t witnesses = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormOpenInquiryReplyWitness ||
        witness.lane[1] != inquiry.lane[1])
      continue;
    if (witnesses != 0u || witness.lane[2] == 0u ||
        witness.lane[2] == kInvalid || witness.lane[6] == 0u ||
        witness.lane[7] != kInquiryExternalWitness ||
        !reply_terms_valid(state, inquiry, witness))
      return false;
    bool selected = false;
    for (std::uint32_t index = 0u; index < 2u; ++index)
      selected |= alternatives[index].owner == witness.lane[3] &&
                  alternatives[index].revision == witness.lane[4] &&
                  alternatives[index].consequence == witness.lane[5];
    if (!selected)
      return false;
    ++witnesses;
  }
  return witnesses == 1u;
}

BCC32_OPEN_INQUIRY_HD inline bool reply_template_from_episode(
    const ResidentRewriteState* state, const Record& inquiry,
    bool require_recurrent, std::uint32_t* flags, std::uint32_t* prefix,
    std::uint32_t* suffix) {
  if (state == nullptr || flags == nullptr || prefix == nullptr ||
      suffix == nullptr)
    return false;
  const Record* witness = nullptr;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormOpenInquiryReplyWitness ||
        candidate.lane[1] != inquiry.lane[1])
      continue;
    if (witness != nullptr || !reply_terms_valid(state, inquiry, candidate))
      return false;
    witness = &candidate;
  }
  if (witness == nullptr) return false;
  std::uint32_t words[kMaximumSurfaceWords]{};
  for (std::uint32_t index = 0u; index < witness->reserved[0]; ++index) {
    const std::uint32_t term_slot = unique_owned_slot(
        state, kFormOpenInquiryReplyTerm, inquiry.lane[1], index);
    if (term_slot == kInvalid) return false;
    words[index] = state->records[term_slot].lane[3];
  }
  if (reply_shape_from_words(words, witness->reserved[0], witness->lane[5],
                             require_recurrent, flags, prefix, suffix))
    return true;
  // reply_terms_valid() has already established the only other admitted shape:
  // the complete selected Program continuation.
  *flags = 0u;
  *prefix = 0u;
  *suffix = 0u;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool derive_reply_template(
    const ResidentRewriteState* state, std::uint32_t first,
    std::uint32_t second, std::uint32_t third, std::uint32_t* flags,
    std::uint32_t* prefix, std::uint32_t* suffix) {
  if (state == nullptr || first >= live_record_capacity(state) ||
      second >= live_record_capacity(state) || third >= live_record_capacity(state))
    return false;
  std::uint32_t fa = 0u, fb = 0u, fc = 0u;
  std::uint32_t pa = 0u, pb = 0u, pc = 0u;
  std::uint32_t sa = 0u, sb = 0u, sc = 0u;
  if (!reply_template_from_episode(
          state, state->records[first], true, &fa, &pa, &sa) ||
      !reply_template_from_episode(
          state, state->records[second], true, &fb, &pb, &sb) ||
      !reply_template_from_episode(
          state, state->records[third], true, &fc, &pc, &sc) ||
      fa != fb || fa != fc || pa != pb || pa != pc || sa != sb || sa != sc)
    return false;
  *flags = fa;
  *prefix = pa;
  *suffix = sa;
  return true;
}

// Factor literal topology around the two exact external Program continuations.
// The continuation bytes stay in their Programs; constructor terms merely name
// where they expand.  Any absent, duplicate, overlapping, or conflicting span
// leaves the constructor unmade.
BCC32_OPEN_INQUIRY_HD inline bool derive_continuation_template(
    const ResidentRewriteState* state, std::uint32_t first, std::uint32_t second,
    std::uint32_t third, std::uint32_t kinds[kMaximumSurfaceWords],
    std::uint32_t values[kMaximumSurfaceWords], std::uint32_t* length) {
  if (state == nullptr || kinds == nullptr || values == nullptr || length == nullptr ||
      !episode_complete_valid(state, first) || !episode_complete_valid(state, second) ||
      !episode_complete_valid(state, third))
    return false;
  const Record* episodes[3] = {&state->records[first], &state->records[second],
                               &state->records[third]};
  AlternativeSnapshot alternatives[3][2]{};
  std::uint32_t starts[3][2]{};
  std::uint32_t spans[3][2]{};
  for (std::uint32_t episode = 0u; episode < 3u; ++episode) {
    if (!episode_alternatives(state, *episodes[episode], alternatives[episode]) ||
        !reply_witness_matches_continuation(state, *episodes[episode],
                                            alternatives[episode]) ||
        !find_continuation_once(state, *episodes[episode], alternatives[episode][0],
                                &starts[episode][0], &spans[episode][0]) ||
        !find_continuation_once(state, *episodes[episode], alternatives[episode][1],
                                &starts[episode][1], &spans[episode][1]))
      return false;
    const bool first_before_second = starts[episode][0] < starts[episode][1];
    const std::uint32_t left = first_before_second ? 0u : 1u;
    const std::uint32_t right = first_before_second ? 1u : 0u;
    if (starts[episode][left] + spans[episode][left] > starts[episode][right])
      return false;
  }
  const bool first_before_second = starts[0][0] < starts[0][1];
  for (std::uint32_t episode = 1u; episode < 3u; ++episode)
    if ((starts[episode][0] < starts[episode][1]) != first_before_second)
      return false;
  const std::uint32_t left = first_before_second ? 0u : 1u;
  const std::uint32_t right = first_before_second ? 1u : 0u;
  const std::uint32_t literal_begin[3] = {0u, starts[0][left] + spans[0][left],
                                           starts[0][right] + spans[0][right]};
  const std::uint32_t literal_end[3] = {starts[0][left], starts[0][right],
                                         episodes[0]->reserved[0]};
  for (std::uint32_t segment = 0u; segment < 3u; ++segment) {
    const std::uint32_t expected = literal_end[segment] - literal_begin[segment];
    for (std::uint32_t episode = 1u; episode < 3u; ++episode) {
      const std::uint32_t begin = segment == 0u ? 0u
          : (segment == 1u ? starts[episode][left] + spans[episode][left]
                           : starts[episode][right] + spans[episode][right]);
      const std::uint32_t end = segment == 0u ? starts[episode][left]
          : (segment == 1u ? starts[episode][right] : episodes[episode]->reserved[0]);
      if (end < begin || end - begin != expected) return false;
      for (std::uint32_t index = 0u; index < expected; ++index) {
        std::uint32_t reference = 0u;
        std::uint32_t observed = 0u;
        if (!inquiry_surface_word_at(state, *episodes[0], literal_begin[segment] + index,
                                     &reference) ||
            !inquiry_surface_word_at(state, *episodes[episode], begin + index,
                                     &observed) || reference != observed)
          return false;
      }
    }
  }
  std::uint32_t out = 0u;
  for (std::uint32_t segment = 0u; segment < 3u; ++segment) {
    for (std::uint32_t index = literal_begin[segment];
         index < literal_end[segment]; ++index) {
      if (out == kMaximumSurfaceWords ||
          !inquiry_surface_word_at(state, *episodes[0], index, &values[out]))
        return false;
      kinds[out++] = kTermLiteral;
    }
    if (segment == 0u || segment == 1u) {
      if (out == kMaximumSurfaceWords) return false;
      kinds[out] = (segment == 0u ? left : right) == 0u
                       ? kTermFirstAlternative : kTermSecondAlternative;
      values[out++] = 0u;
    }
  }
  *length = out;
  return out != 0u;
}

BCC32_OPEN_INQUIRY_HD inline bool derive_template(
    const ResidentRewriteState* state, std::uint32_t first, std::uint32_t second,
    std::uint32_t third, std::uint32_t kinds[kMaximumSurfaceWords],
    std::uint32_t values[kMaximumSurfaceWords], std::uint32_t* length) {
  if (state == nullptr || kinds == nullptr || values == nullptr || length == nullptr ||
      !episode_complete_valid(state, first) || !episode_complete_valid(state, second) ||
      !episode_complete_valid(state, third))
    return false;
  const Record& a = state->records[first];
  const Record& b = state->records[second];
  const Record& c = state->records[third];
  if (a.lane[2] == b.lane[2] || a.lane[2] == c.lane[2] ||
      b.lane[2] == c.lane[2])
    return false;
  std::uint32_t count_a = 0u;
  std::uint32_t count_b = 0u;
  std::uint32_t count_c = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormOpenInquirySurfaceTerm)
      continue;
    count_a += term.lane[1] == a.lane[1];
    count_b += term.lane[1] == b.lane[1];
    count_c += term.lane[1] == c.lane[1];
  }
  if (count_a == 0u || count_a != count_b || count_a != count_c ||
      count_a > kMaximumSurfaceWords)
    return false;
  std::uint32_t refs_a = 0u;
  std::uint32_t refs_b = 0u;
  std::uint32_t alternatives_a[2]{};
  std::uint32_t alternatives_b[2]{};
  std::uint32_t alternatives_c[2]{};
  if (!inquiry_episode_consequences(state, a, alternatives_a) ||
      !inquiry_episode_consequences(state, b, alternatives_b) ||
      !inquiry_episode_consequences(state, c, alternatives_c))
    return false;
  for (std::uint32_t index = 0u; index < count_a; ++index) {
    std::uint32_t wa = 0u, wb = 0u, wc = 0u;
    if (!inquiry_surface_word_at(state, a, index, &wa) ||
        !inquiry_surface_word_at(state, b, index, &wb) ||
        !inquiry_surface_word_at(state, c, index, &wc))
      return false;
    if (wa == wb && wa == wc) {
      kinds[index] = kTermLiteral;
      values[index] = wa;
    } else if (wa == alternatives_a[0] && wb == alternatives_b[0] &&
               wc == alternatives_c[0]) {
      kinds[index] = kTermFirstAlternative;
      values[index] = 0u;
      ++refs_a;
    } else if (wa == alternatives_a[1] && wb == alternatives_b[1] &&
               wc == alternatives_c[1]) {
      kinds[index] = kTermSecondAlternative;
      values[index] = 0u;
      ++refs_b;
    } else {
      return false;
    }
  }
  if (refs_a == 0u || refs_b == 0u) return false;
  *length = count_a;
  return true;
}

// Factor exactly one allocation-order-independent template.  A conflicting
// supportable triple abstains rather than selecting a chronology or slot winner.
BCC32_OPEN_INQUIRY_HD inline bool settle_constructor_from_complete_episodes(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  ++state->oi_ctor_settle_attempts;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    if (inquiry_constructor_authoritative(state, slot)) {
      ++state->oi_ctor_settle_decline_already_authoritative;
      return false;
    }
  std::uint32_t episodes[kMaximumFactorEpisodes]{};
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!episode_complete_valid(state, slot)) continue;
    if (count == kMaximumFactorEpisodes) {
      ++state->oi_ctor_settle_decline_episode_overflow;
      return false;
    }
    episodes[count++] = slot;
  }
  state->oi_ctor_settle_last_episode_count = count;
  if (count < kRequiredIndependentEpisodes) {
    ++state->oi_ctor_settle_decline_insufficient_episodes;
    return false;
  }
  for (std::uint32_t index = 1u; index < count; ++index) {
    const std::uint32_t candidate = episodes[index];
    std::uint32_t position = index;
    while (position != 0u &&
           state->records[candidate].lane[1] <
               state->records[episodes[position - 1u]].lane[1]) {
      episodes[position] = episodes[position - 1u];
      --position;
    }
    episodes[position] = candidate;
  }
  bool have_template = false;
  std::uint32_t chosen[3]{kInvalid, kInvalid, kInvalid};
  std::uint32_t selected_kinds[kMaximumSurfaceWords]{};
  std::uint32_t selected_values[kMaximumSurfaceWords]{};
  std::uint32_t selected_length = 0u;
  std::uint32_t selected_reply_flags = 0u;
  std::uint32_t selected_reply_prefix = 0u;
  std::uint32_t selected_reply_suffix = 0u;
  for (std::uint32_t i = 0u; i + 2u < count; ++i) {
    for (std::uint32_t j = i + 1u; j + 1u < count; ++j) {
      for (std::uint32_t k = j + 1u; k < count; ++k) {
        std::uint32_t kinds[kMaximumSurfaceWords]{};
        std::uint32_t values[kMaximumSurfaceWords]{};
        std::uint32_t length = 0u;
        if (!derive_continuation_template(state, episodes[i], episodes[j],
                                          episodes[k], kinds, values, &length) &&
            !derive_template(state, episodes[i], episodes[j], episodes[k], kinds,
                             values, &length))
          continue;
        std::uint32_t reply_flags = 0u;
        std::uint32_t reply_prefix = 0u;
        std::uint32_t reply_suffix = 0u;
        if (!derive_reply_template(state, episodes[i], episodes[j], episodes[k],
                                   &reply_flags, &reply_prefix, &reply_suffix))
          continue;
        if (!have_template) {
          have_template = true;
          chosen[0] = episodes[i]; chosen[1] = episodes[j]; chosen[2] = episodes[k];
          selected_length = length;
          selected_reply_flags = reply_flags;
          selected_reply_prefix = reply_prefix;
          selected_reply_suffix = reply_suffix;
          for (std::uint32_t n = 0u; n < length; ++n) {
            selected_kinds[n] = kinds[n];
            selected_values[n] = values[n];
          }
        } else if (length != selected_length ||
                   reply_flags != selected_reply_flags ||
                   reply_prefix != selected_reply_prefix ||
                   reply_suffix != selected_reply_suffix) {
          ++state->oi_ctor_settle_decline_conflicting_template;
          return false;
        } else {
          for (std::uint32_t n = 0u; n < length; ++n)
            if (selected_kinds[n] != kinds[n] || selected_values[n] != values[n]) {
              ++state->oi_ctor_settle_decline_conflicting_template;
              return false;
            }
        }
      }
    }
  }
  if (!have_template) {
    ++state->oi_ctor_settle_decline_no_template;
    return false;
  }
  if (free_record_count(state) < 1u + selected_length + 3u) {
    ++state->oi_ctor_settle_decline_insufficient_free_records;
    return false;
  }
  const std::uint32_t owner = make_inquiry_owner(
      state, kFormOpenInquiryConstructor,
      rewrite_mix(state->records[chosen[0]].lane[1], state->records[chosen[1]].lane[1],
                  state->records[chosen[2]].lane[1]));
  if (owner == kInvalid) {
    ++state->oi_ctor_settle_decline_owner_failed;
    return false;
  }
  const std::uint32_t header_slot = allocate_record(state);
  if (header_slot == kInvalid) {
    ++state->oi_ctor_settle_decline_header_alloc_failed;
    return false;
  }
  Record& constructor = state->records[header_slot];
  constructor.lane[0] = kFormOpenInquiryConstructor;
  constructor.lane[1] = owner;
  constructor.lane[2] = selected_length;
  constructor.lane[3] = kRequiredIndependentEpisodes;
  constructor.lane[4] = kReplyEqualsChosenAlternative;
  constructor.lane[6] = selected_reply_flags;
  constructor.lane[7] = kInquiryConstructorEnabled;
  constructor.reserved[0] = selected_reply_prefix;
  constructor.reserved[1] = selected_reply_suffix;
  ++constructor.revision;
  std::uint32_t template_digest = rewrite_mix(
      kFormOpenInquiryConstructor, constructor.lane[1], constructor.lane[2]);
  for (std::uint32_t index = 0u; index < selected_length; ++index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) {
      ++state->oi_ctor_settle_decline_term_alloc_failed;
      return false;
    }
    Record& term = state->records[slot];
    term.lane[0] = kFormOpenInquiryConstructorTerm;
    term.lane[1] = owner;
    term.lane[2] = index;
    term.lane[3] = selected_kinds[index];
    term.lane[4] = selected_values[index];
    ++term.revision;
    template_digest = rewrite_mix(
        template_digest,
        rewrite_mix(index, term.lane[3], term.lane[4]), term.revision);
  }
  if (template_digest == 0u || template_digest == kInvalid)
    template_digest ^= 0x7f4a7c15u;
  constructor.lane[5] = template_digest;
  for (std::uint32_t index = 0u; index < 3u; ++index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) {
      ++state->oi_ctor_settle_decline_witness_alloc_failed;
      return false;
    }
    Record& witness = state->records[slot];
    witness.lane[0] = kFormOpenInquiryConstructorWitness;
    witness.lane[1] = owner;
    witness.lane[2] = state->records[chosen[index]].lane[1];
    witness.lane[3] = state->records[chosen[index]].revision;
    witness.lane[4] = state->records[chosen[index]].lane[2];
    ++witness.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  const bool authoritative =
      inquiry_constructor_authoritative(state, header_slot);
  if (authoritative)
    ++state->oi_ctor_settle_admitted;
  else
    ++state->oi_ctor_settle_decline_final_check_failed;
  return authoritative;
}

BCC32_OPEN_INQUIRY_HD inline bool constructor_term_at(
    const ResidentRewriteState* state, const Record& constructor,
    std::uint32_t index, std::uint32_t* kind, std::uint32_t* value) {
  if (kind == nullptr || value == nullptr || index >= constructor.lane[2]) return false;
  const std::uint32_t slot = unique_owned_slot(
      state, kFormOpenInquiryConstructorTerm, constructor.lane[1], index);
  if (slot == kInvalid) return false;
  const Record& term = state->records[slot];
  if (term.lane[3] > kTermSecondAlternative ||
      (term.lane[3] == kTermLiteral && term.lane[4] == kInvalid) ||
      (term.lane[3] != kTermLiteral && term.lane[4] != 0u))
    return false;
  *kind = term.lane[3];
  *value = term.lane[4];
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool constructor_template_digest(
    const ResidentRewriteState* state, const Record& constructor,
    std::uint32_t* digest) {
  if (state == nullptr || digest == nullptr || constructor.lane[2] == 0u ||
      constructor.lane[2] > kMaximumSurfaceWords)
    return false;
  std::uint32_t result = rewrite_mix(
      kFormOpenInquiryConstructor, constructor.lane[1], constructor.lane[2]);
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t slot = unique_owned_slot(
        state, kFormOpenInquiryConstructorTerm, constructor.lane[1], index);
    if (slot == kInvalid) return false;
    const Record& term = state->records[slot];
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    if (!constructor_term_at(state, constructor, index, &kind, &value) ||
        term.revision == 0u)
      return false;
    result = rewrite_mix(result, rewrite_mix(index, kind, value), term.revision);
  }
  if (result == 0u || result == kInvalid) result ^= 0x7f4a7c15u;
  *digest = result;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_episode_matches_constructor(
    const ResidentRewriteState* state, const Record& constructor,
    const Record& inquiry) {
  if (state == nullptr || inquiry.reserved[0] == 0u)
    return false;
  AlternativeSnapshot alternatives[2]{};
  if (!episode_alternatives(state, inquiry, alternatives) ||
      !reply_witness_matches_continuation(state, inquiry, alternatives))
    return false;
  std::uint32_t surface_index = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    if (!constructor_term_at(state, constructor, index, &kind, &value))
      return false;
    if (kind == kTermLiteral) {
      std::uint32_t observed = 0u;
      if (surface_index >= inquiry.reserved[0] ||
          !inquiry_surface_word_at(state, inquiry, surface_index++, &observed) ||
          observed != value)
        return false;
      continue;
    }
    const AlternativeSnapshot& alternative =
        kind == kTermFirstAlternative ? alternatives[0] : alternatives[1];
    std::uint32_t label_length = 0u;
    std::uint32_t label_digest = 0u;
    if (!alternative_continuation_digest(state, alternative, inquiry.lane[3],
                                         &label_length, &label_digest) ||
        surface_index + label_length > inquiry.reserved[0])
      return false;
    for (std::uint32_t label_index = 0u; label_index < label_length; ++label_index) {
      std::uint32_t observed = 0u;
      std::uint32_t expected = 0u;
      if (!inquiry_surface_word_at(state, inquiry, surface_index++, &observed) ||
          !alternative_continuation_word_at(state, alternative, inquiry.lane[3],
                                            label_index, &expected) ||
          observed != expected)
        return false;
    }
  }
  if (surface_index != inquiry.reserved[0]) return false;
  std::uint32_t flags = 0u;
  std::uint32_t prefix = 0u;
  std::uint32_t suffix = 0u;
  return reply_template_from_episode(state, inquiry, false, &flags, &prefix,
                                     &suffix) &&
         flags == constructor.lane[6] &&
         ((flags & kReplyTemplatePrefix) == 0u ||
          prefix == constructor.reserved[0]) &&
         ((flags & kReplyTemplateSuffix) == 0u ||
          suffix == constructor.reserved[1]);
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_constructor_authoritative(
    const ResidentRewriteState* state, std::uint32_t constructor_slot) {
  if (state == nullptr || constructor_slot >= live_record_capacity(state)) return false;
  const Record& constructor = state->records[constructor_slot];
  if (constructor.matter_q8 == 0u ||
      constructor.lane[0] != kFormOpenInquiryConstructor ||
      constructor.lane[1] == 0u || constructor.lane[1] == kInvalid ||
      constructor.lane[2] == 0u ||
      constructor.lane[2] > kMaximumSurfaceWords ||
      constructor.lane[3] != kRequiredIndependentEpisodes ||
      constructor.lane[4] != kReplyEqualsChosenAlternative ||
      constructor.lane[5] == 0u || constructor.lane[5] == kInvalid ||
      (constructor.lane[6] &
       ~(kReplyTemplatePrefix | kReplyTemplateSuffix)) != 0u ||
      ((constructor.lane[6] & kReplyTemplatePrefix) == 0u &&
       constructor.reserved[0] != 0u) ||
      ((constructor.lane[6] & kReplyTemplateSuffix) == 0u &&
       constructor.reserved[1] != 0u) ||
      constructor.lane[7] != kInquiryConstructorEnabled)
    return false;
  std::uint32_t term_count = 0u;
  std::uint32_t first_refs = 0u;
  std::uint32_t second_refs = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 != 0u &&
        term.lane[0] == kFormOpenInquiryConstructorTerm &&
        term.lane[1] == constructor.lane[1])
      ++term_count;
  }
  if (term_count != constructor.lane[2]) return false;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    if (!constructor_term_at(state, constructor, index, &kind, &value))
      return false;
    first_refs += kind == kTermFirstAlternative;
    second_refs += kind == kTermSecondAlternative;
  }
  if (first_refs == 0u || second_refs == 0u) return false;
  std::uint32_t template_digest = 0u;
  if (!constructor_template_digest(state, constructor, &template_digest) ||
      template_digest != constructor.lane[5])
    return false;

  std::uint32_t witness_count = 0u;
  std::uint32_t episode_owners[kRequiredIndependentEpisodes]{
      kInvalid, kInvalid, kInvalid};
  std::uint32_t suspended_owners[kRequiredIndependentEpisodes]{
      kInvalid, kInvalid, kInvalid};
  std::uint32_t surface_owners[kRequiredIndependentEpisodes]{
      kInvalid, kInvalid, kInvalid};
  std::uint32_t reply_owners[kRequiredIndependentEpisodes]{
      kInvalid, kInvalid, kInvalid};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormOpenInquiryConstructorWitness ||
        witness.lane[1] != constructor.lane[1])
      continue;
    if (witness_count == kRequiredIndependentEpisodes ||
        witness.lane[2] == 0u || witness.lane[2] == kInvalid ||
        witness.lane[3] == 0u || witness.lane[4] == 0u ||
        witness.lane[4] == kInvalid)
      return false;
    const std::uint32_t episode_slot = unique_header_by_owner(
        state, kFormOpenInquiry, witness.lane[2]);
    if (episode_slot == kInvalid ||
        !episode_complete_valid(state, episode_slot))
      return false;
    const Record& episode = state->records[episode_slot];
    const std::uint32_t surface_owner =
        inquiry_surface_source_owner(state, episode);
    const std::uint32_t reply_owner =
        inquiry_reply_source_owner(state, episode);
    if (episode.revision != witness.lane[3] ||
        episode.lane[2] != witness.lane[4] ||
        surface_owner == kInvalid || reply_owner == kInvalid ||
        surface_owner == reply_owner ||
        !inquiry_episode_matches_constructor(state, constructor, episode))
      return false;
    for (std::uint32_t prior = 0u; prior < witness_count; ++prior)
      if (episode_owners[prior] == witness.lane[2] ||
          suspended_owners[prior] == witness.lane[4] ||
          surface_owners[prior] == surface_owner ||
          reply_owners[prior] == reply_owner)
        return false;
    episode_owners[witness_count] = witness.lane[2];
    suspended_owners[witness_count] = witness.lane[4];
    surface_owners[witness_count] = surface_owner;
    reply_owners[witness_count] = reply_owner;
    ++witness_count;
  }
  return witness_count == kRequiredIndependentEpisodes;
}

#include "bcc32_resident_open_inquiry_selection.inl"
#include "bcc32_resident_open_inquiry_emission.inl"

}  // namespace substrate::bcc32::causal_rewrite::open_inquiry

#undef BCC32_OPEN_INQUIRY_HD
