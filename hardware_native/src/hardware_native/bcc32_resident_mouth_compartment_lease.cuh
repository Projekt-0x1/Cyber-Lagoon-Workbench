#pragma once

// ---------------------------------------------------------------------------
// MOUTH COMPARTMENT LEASE -- the handbook's named repair for the public
// surface's accidental global monopoly (docs/0x1_handbook_complete.md
// ~L30466-30473): "remove the accidental global monopoly on the mouth ...
// shard it into local mouth-compartment leases, each of which must still be
// powered by real body/peer/world consequence ... a substrate repair, not a
// semantic scheduler."
//
// REBUILD, NOT RECOVERY. A prior donor with this name existed (GREEN,
// host+device, reversible, matter account closed) but was handed off
// unclaimed and its bytes are not recoverable from any ref or worktree in
// this repository (checked before writing this file). This is a fresh
// implementation of the same audited design intent, with this file's own
// dosing -- it makes no claim to reproduce the prior donor byte-for-byte.
//
// WHY IT MATTERS (from the audit, unchanged): today's public surface is one
// global register triple -- generated_word / generated_word_valid /
// generated_locus, ResidentRewriteState (causal_rewrite_universe.cuh:226-
// 228) -- written by every producer in the resident construction pipeline.
// With one gate, removing the single mouth locus always collapses the whole
// surface, so the handbook's distributed-field contrast (targeted public-
// actuator removal vs matched random removal) can only ever be asserted,
// never earned: there is nothing sharded to remove part of.
//
// WHAT THIS FILE ADDS: kMouthCompartmentCount independent output slots, each
// gated on its own consequence-backed payment. A compartment may publish a
// word only while paid_matter_q8 > 0; withdrawing that payment silences only
// that one compartment. This is deliberately NOT a new Record/Form: identity
// comes from the EXISTING make_record_owner allocator (the same authority
// every other resident record's owner comes from), and payment is a plain
// counter this file owns -- zero new cardinality in the shared Record
// ecology, matching the audit's note that the design "adds no cardinality,
// only a lease on an existing make_record_owner family".
//
// WIRING (0X1-158): mouth_compartment_for_locus/mouth_compartment_gate_
// public_emission below are the reusable glue the canonical public-emission
// checkpoint in bcc32_resident_rewrite_runtime.cu now calls, gating
// egress_history::append itself rather than any of the nine upstream
// generated_word/generated_word_valid/generated_locus writer sites. Those
// nine sites are still untouched -- they keep writing the single
// state->world.generated_word/generated_locus triple exactly as before;
// wiring each producer into its own compartment identity is a distinct,
// larger change. What changed is that the LAST gate before a generated word
// becomes a public act now consults the compartment (by the generated
// locus's own value, not a host pick) instead of only checking locus
// validity, so a withdrawn compartment silences public emission for loci
// that hash to it while other loci still publish (CodexCapabilityCoord0812D
// flagged the prior donor as a possible observer-word-ontology concern --
// the answer here is the same as before: no new Form, no new vocabulary,
// only a payment-gated lease on an allocator every resident record already
// shares).
// ---------------------------------------------------------------------------

#include <cstdint>

#include "causal_rewrite_universe.cuh"

#if defined(__CUDACC__)
#define BCC32_MOUTH_LEASE_HD __host__ __device__
#else
#define BCC32_MOUTH_LEASE_HD
#endif

namespace substrate::bcc32::causal_rewrite {

inline constexpr std::uint32_t kMouthCompartmentCount = 3u;
// A payment keeps one compartment alive for a bounded resident-clock span.
// Accepted raw consequence renews it; quiet epochs let it expire.
inline constexpr std::uint64_t kMouthLeaseDurationEpochs = 64u;

struct MouthCompartment {
  std::uint32_t owner = kInvalid;      // identity: make_record_owner, once
  std::uint32_t paid_matter_q8 = 0u;   // > 0 iff currently consequence-backed
  std::uint32_t speak_count = 0u;      // diagnostic only: never gates anything
  std::uint64_t expiry_epoch = 0u;      // 0 iff no live payment
  std::uint64_t last_consequence_revision = 0u;
  std::uint32_t renewal_count = 0u;
  std::uint32_t lesion_count = 0u;
};

struct MouthCompartmentField {
  MouthCompartment compartment[kMouthCompartmentCount]{};
};

BCC32_MOUTH_LEASE_HD inline bool mouth_compartment_owner_exists(
    const MouthCompartmentField* field, std::uint32_t owner,
    std::uint32_t except_index) {
  if (field == nullptr || owner == kInvalid || owner == 0u) return false;
  for (std::uint32_t index = 0u; index < kMouthCompartmentCount; ++index)
    if (index != except_index && field->compartment[index].owner == owner)
      return true;
  return false;
}

// One shared, single-identity gate: the literal shape of today's global
// triple, kept here only as the comparison arm for the falsifier below. A
// real single-gate topology would route every producer through this one
// balance instead of a MouthCompartmentField.
struct SingleGateField {
  std::uint32_t owner = kInvalid;
  std::uint32_t paid_matter_q8 = 0u;
  std::uint32_t generated_word = 0u;
  std::uint32_t generated_word_valid = 0u;
  std::uint32_t generated_locus = kInvalid;
};

// Admits a returned-consequence event as this compartment's payment. Mirrors
// the capacity-then-drop idiom already used for resident admission elsewhere
// in this substrate (compare insert_relation_triple's attempt/drop
// accounting): a compartment holds at most one live payment at a time, so a
// second consequence event aimed at an already-paid compartment is REJECTED,
// not queued or summed. This is what makes "admission" a real capacity
// property instead of an unconditional accept.
BCC32_MOUTH_LEASE_HD inline bool admit_mouth_compartment_consequence(
    ResidentRewriteState* state, MouthCompartmentField* field,
    std::uint32_t index, std::uint32_t consequence_matter_q8,
    std::uint64_t epoch);

BCC32_MOUTH_LEASE_HD inline bool admit_mouth_compartment_consequence(
    ResidentRewriteState* state, MouthCompartmentField* field,
    std::uint32_t index, std::uint32_t consequence_matter_q8) {
  return admit_mouth_compartment_consequence(
      state, field, index, consequence_matter_q8, 0u);
}

BCC32_MOUTH_LEASE_HD inline bool admit_mouth_compartment_consequence(
    ResidentRewriteState* state, MouthCompartmentField* field,
    std::uint32_t index, std::uint32_t consequence_matter_q8,
    std::uint64_t epoch) {
  if (state == nullptr || field == nullptr || state->fault != 0u ||
      index >= kMouthCompartmentCount || consequence_matter_q8 == 0u)
    return false;
  MouthCompartment& compartment = field->compartment[index];
  if (compartment.paid_matter_q8 != 0u) return false;  // capacity exhausted
  if (compartment.owner == kInvalid) {
    compartment.owner =
        make_record_owner(state, 0x6d6f7574u ^ (index + 1u));  // 'mout' ^ idx
    if (compartment.owner == kInvalid ||
        mouth_compartment_owner_exists(field, compartment.owner, index)) {
      compartment.owner = kInvalid;
      return false;
    }
  }
  if (compartment.owner == 0u ||
      mouth_compartment_owner_exists(field, compartment.owner, index))
    return false;
  compartment.paid_matter_q8 = consequence_matter_q8;
  compartment.expiry_epoch = epoch + kMouthLeaseDurationEpochs;
  compartment.last_consequence_revision = state->revision;
  return true;
}

// A later accepted raw consequence renews the resident-selected producer's
// lease. It may reacquire an expired/lesioned compartment, but cannot mint an
// identity, select another compartment, or replay one resident revision.
BCC32_MOUTH_LEASE_HD inline bool renew_mouth_compartment_consequence(
    ResidentRewriteState* state, MouthCompartmentField* field,
    std::uint32_t index, std::uint32_t consequence_matter_q8,
    std::uint64_t epoch) {
  if (state == nullptr || field == nullptr || state->fault != 0u ||
      index >= kMouthCompartmentCount || consequence_matter_q8 == 0u)
    return false;
  MouthCompartment& compartment = field->compartment[index];
  if (compartment.owner == kInvalid || compartment.owner == 0u ||
      mouth_compartment_owner_exists(field, compartment.owner, index) ||
      (state->revision != 0u &&
       compartment.last_consequence_revision == state->revision))
    return false;
  compartment.paid_matter_q8 = consequence_matter_q8;
  compartment.expiry_epoch = epoch + kMouthLeaseDurationEpochs;
  compartment.last_consequence_revision = state->revision;
  if (compartment.renewal_count != kInvalid) ++compartment.renewal_count;
  return true;
}

BCC32_MOUTH_LEASE_HD inline void advance_mouth_compartment_epoch(
    MouthCompartmentField* field, std::uint64_t epoch) {
  if (field == nullptr) return;
  for (std::uint32_t index = 0u; index < kMouthCompartmentCount; ++index) {
    MouthCompartment& compartment = field->compartment[index];
    if (compartment.paid_matter_q8 != 0u &&
        compartment.expiry_epoch != 0u &&
        epoch >= compartment.expiry_epoch) {
      compartment.paid_matter_q8 = 0u;
      compartment.expiry_epoch = 0u;
    }
  }
}

// Targeted removal: withdraw exactly one compartment's consequence backing.
// Reversible -- a later admit_mouth_compartment_consequence on the same
// index pays it again, matching the audit's "reversible" measurement.
BCC32_MOUTH_LEASE_HD inline void withdraw_mouth_compartment(
    MouthCompartmentField* field, std::uint32_t index) {
  if (field == nullptr || index >= kMouthCompartmentCount) return;
  MouthCompartment& compartment = field->compartment[index];
  compartment.paid_matter_q8 = 0u;
  compartment.expiry_epoch = 0u;
  if (compartment.lesion_count != kInvalid) ++compartment.lesion_count;
}

// Translate a raw physical lesion over resident loci into localized mouth
// tissue. No semantic compartment id is supplied by the host.
BCC32_MOUTH_LEASE_HD inline void withdraw_mouth_compartments_for_lesion(
    MouthCompartmentField* field, std::uint32_t start,
    std::uint32_t count) {
  if (field == nullptr) return;
  for (std::uint32_t offset = 0u; offset < count; ++offset)
    withdraw_mouth_compartment(
        field, (start + offset) % kMouthCompartmentCount);
}

BCC32_MOUTH_LEASE_HD inline bool mouth_compartment_may_speak(
    const MouthCompartmentField* field, std::uint32_t index) {
  if (field == nullptr || index >= kMouthCompartmentCount) return false;
  const MouthCompartment& compartment = field->compartment[index];
  return compartment.owner != kInvalid && compartment.owner != 0u &&
      !mouth_compartment_owner_exists(field, compartment.owner, index) &&
      compartment.paid_matter_q8 > 0u && compartment.expiry_epoch != 0u;
}

// The substrate repair itself: gate publication on consequence-backed
// payment, never on a semantic role, word identity, or content. A
// compartment that cannot currently pay cannot speak, full stop.
BCC32_MOUTH_LEASE_HD inline bool set_generated_word_in_compartment(
    const ResidentRewriteState* state, MouthCompartmentField* field,
    std::uint32_t index, std::uint32_t word, std::uint32_t locus) {
  // The canonical adult keeps the payload in ResidentRewriteState and sends
  // it directly to the one egress writer. A mouth compartment is only a
  // consequence-backed permission; it must not become a second per-word
  // memory or writer. Keep the historical arguments for call-site stability,
  // but deliberately do not retain either value here.
  (void)word;
  if (state == nullptr || locus == kInvalid ||
      locus >= live_record_capacity(state) ||
      !mouth_compartment_may_speak(field, index))
    return false;
  MouthCompartment& compartment = field->compartment[index];
  if (compartment.speak_count != kInvalid) ++compartment.speak_count;
  return true;
}

BCC32_MOUTH_LEASE_HD inline std::uint32_t mouth_compartments_speaking(
    const MouthCompartmentField* field) {
  if (field == nullptr) return 0u;
  std::uint32_t speaking = 0u;
  for (std::uint32_t i = 0u; i < kMouthCompartmentCount; ++i)
    if (mouth_compartment_may_speak(field, i)) ++speaking;
  return speaking;
}

// Maps a producer's own generated locus onto one of the kMouthCompartmentCount
// compartments. Deterministic and resident-controlled -- the locus is the
// record slot the resident's own construction/search already selected this
// epoch, never a value the host picks -- so which compartment a given
// public-emission attempt lands in is itself an ordinary consequence of
// which record the resident is currently generating from.
BCC32_MOUTH_LEASE_HD inline std::uint32_t mouth_compartment_for_locus(
    std::uint32_t generated_locus) {
  if (generated_locus == kInvalid) return kInvalid;
  return generated_locus % kMouthCompartmentCount;
}

// THE canonical public-emission gate. Called once, at the single checkpoint
// immediately before a generated word is appended to egress history (see
// bcc32_resident_rewrite_runtime.cu, resident_rewrite_epoch_kernel). Returns
// whether the compartment covering this locus may currently speak, and
// reports which compartment that was so the caller can record the
// publication into it afterward via set_generated_word_in_compartment. This
// is the one place the sharded/single-register distinction actually bites in
// production: a withdrawn compartment silences only the loci that hash to
// it, not the whole public surface.
BCC32_MOUTH_LEASE_HD inline bool mouth_compartment_gate_public_emission(
    const ResidentRewriteState* state, const MouthCompartmentField* field,
    std::uint32_t generated_locus, std::uint32_t* compartment_index_out) {
  if (state == nullptr || generated_locus == kInvalid ||
      generated_locus >= live_record_capacity(state)) {
    if (compartment_index_out != nullptr) *compartment_index_out = kInvalid;
    return false;
  }
  const std::uint32_t index = mouth_compartment_for_locus(generated_locus);
  if (compartment_index_out != nullptr) *compartment_index_out = index;
  return mouth_compartment_may_speak(field, index);
}

// Single-gate analogue of admit_mouth_compartment_consequence, capacity 1
// (today's topology has exactly one identity to pay, so at most one live
// payment exists regardless of how many events target it).
BCC32_MOUTH_LEASE_HD inline bool admit_single_gate_consequence(
    ResidentRewriteState* state, SingleGateField* field,
    std::uint32_t consequence_matter_q8) {
  if (state == nullptr || field == nullptr || consequence_matter_q8 == 0u)
    return false;
  if (field->paid_matter_q8 != 0u) return false;
  if (field->owner == kInvalid) {
    field->owner = make_record_owner(state, 0x6d6f7574u);
    if (field->owner == kInvalid) return false;
  }
  field->paid_matter_q8 = consequence_matter_q8;
  return true;
}

BCC32_MOUTH_LEASE_HD inline void withdraw_single_gate(SingleGateField* field) {
  if (field == nullptr) return;
  field->paid_matter_q8 = 0u;
}

BCC32_MOUTH_LEASE_HD inline bool single_gate_may_speak(
    const SingleGateField* field) {
  return field != nullptr && field->paid_matter_q8 > 0u;
}

}  // namespace substrate::bcc32::causal_rewrite

#undef BCC32_MOUTH_LEASE_HD
