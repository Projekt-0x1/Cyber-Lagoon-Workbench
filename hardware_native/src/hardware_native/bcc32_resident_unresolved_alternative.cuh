#pragma once

#include "bcc32_resident_open_inquiry.cuh"

#include <cstdint>

// bcc32_resident_open_inquiry.cuh #undefs its own BCC32_OPEN_INQUIRY_HD at
// end of file (it is header-private, not meant to leak to includers), so
// this file needs its own copy of the same host/device qualifier macro.
#if defined(__CUDACC__)
#define BCC32_OPEN_INQUIRY_HD __host__ __device__
#else
#define BCC32_OPEN_INQUIRY_HD
#endif

// A focused bridge from RWR25's fail-closed dormant-continuation tie to
// RWR24's learned inquiry surface.  The retained records name only exact live
// resident histories and the SpanPrograms which made them equally eligible.
// They store no question text, desired answer, cell, goal, or host-selected
// winner.  Before opening and before every emitted word, a const reader must
// rederive both alternatives from the still-live source matter.
namespace substrate::bcc32::causal_rewrite::open_inquiry::
    unresolved_alternative {

inline constexpr std::uint32_t kFormResidentUnresolvedSet = 0x2ead8054u;
inline constexpr std::uint32_t kFormResidentUnresolvedAlternative =
    0x3fbe9165u;
inline constexpr std::uint32_t kUnresolvedSetCaptured = 1u;
inline constexpr std::uint32_t kUnresolvedSetOpened = 1u << 1u;
inline constexpr std::uint32_t kInquiryUnresolvedAlternativeSurface = 2u;
inline constexpr std::uint32_t kUnresolvedInquiryMarker = 0x554e5231u;

// RUA-2 is the provider-neutral family. A set can contain as many alternatives
// as resident matter permits. Its members persist no Record slot and copy no
// continuation: each retains one provider owner/revision/structural identity
// and one half-open binding-range tuple. The legacy pair bridge below remains
// a two-surface adapter over its learned inquiry constructor.
inline constexpr std::uint32_t kFormResidentAlternativeSet = 0xb540c8e7u;
inline constexpr std::uint32_t kFormResidentAlternativeProvider = 0xc9a31d62u;
inline constexpr std::uint32_t kAlternativeSetBuilding = 1u;
inline constexpr std::uint32_t kAlternativeSetReady = 2u;
inline constexpr std::uint32_t kAlternativeSetRetired = 3u;

struct ProviderIdentity {
  std::uint32_t form = kFormEmpty;
  std::uint32_t owner = kInvalid;
  std::uint32_t revision = 0u;
  std::uint64_t structural_identity = 0u;
  std::uint32_t binding_begin = 0u;
  std::uint32_t binding_extent = 0u;
};

struct AlternativeSetBuilder {
  std::uint32_t owner = kInvalid;
  std::uint32_t expected = 0u;
};

struct AlternativeSetView {
  std::uint32_t owner = kInvalid;
  std::uint32_t count = 0u;
  std::uint64_t identity = 0u;
};

BCC32_OPEN_INQUIRY_HD inline std::uint64_t rotate_provider_identity(
    std::uint64_t value, std::uint32_t distance) {
  const std::uint32_t shift = distance & 63u;
  return shift == 0u ? value : (value << shift) | (value >> (64u - shift));
}

BCC32_OPEN_INQUIRY_HD inline std::uint64_t
owner_independent_record_structure(const Record& record) {
  std::uint32_t low = rewrite_mix(record.lane[0], record.revision,
                                  record.matter_q8);
  std::uint32_t high = rewrite_mix(record.reserved[0], record.reserved[1],
                                   record.lane[0]);
  for (std::uint32_t lane = 2u; lane < kLaneCount; ++lane) {
    low = rewrite_mix(low, record.lane[lane], lane);
    high = rewrite_mix(high, record.lane[lane], lane ^ 0xa5u);
  }
  return (static_cast<std::uint64_t>(high) << 32u) | low;
}

// Structural identity covers the provider header and every live owner-bound
// fragment without depending on allocation order. Slot movement is therefore
// harmless, while a term/binding mutation invalidates every unpublished or
// ready set that named the old structure.
BCC32_OPEN_INQUIRY_HD inline std::uint64_t provider_structural_identity(
    const ResidentRewriteState* state, std::uint32_t provider_slot) {
  if (state == nullptr || provider_slot >= live_record_capacity(state))
    return 0u;
  const Record& provider = state->records[provider_slot];
  if (provider.matter_q8 == 0u || provider.lane[0] == kFormEmpty ||
      provider.lane[0] == kFormResidentAlternativeSet ||
      provider.lane[0] == kFormResidentAlternativeProvider ||
      provider.lane[1] == 0u || provider.lane[1] == kInvalid ||
      provider.revision == 0u || provider.lane[2] == 0u)
    return 0u;
  std::uint64_t xor_fold = 0u;
  std::uint64_t sum_fold = 0u;
  std::uint32_t fragments = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[1] != provider.lane[1])
      continue;
    const std::uint64_t digest = owner_independent_record_structure(record);
    xor_fold ^= rotate_provider_identity(digest, record.lane[0] & 63u);
    sum_fold += digest * 0x9e3779b185ebca87ull;
    ++fragments;
  }
  const std::uint64_t result = xor_fold ^ rotate_provider_identity(
      sum_fold, fragments & 63u) ^
      (static_cast<std::uint64_t>(fragments) << 32u) ^ provider.lane[0];
  return result == 0u ? 0x9e3779b185ebca87ull : result;
}

BCC32_OPEN_INQUIRY_HD inline bool provider_identity_at(
    const ResidentRewriteState* state, std::uint32_t provider_slot,
    std::uint32_t binding_begin, std::uint32_t binding_extent,
    ProviderIdentity* output) {
  if (output != nullptr) *output = ProviderIdentity{};
  if (state == nullptr || output == nullptr ||
      provider_slot >= live_record_capacity(state) || binding_extent == 0u)
    return false;
  const Record& provider = state->records[provider_slot];
  const std::uint64_t identity =
      provider_structural_identity(state, provider_slot);
  if (identity == 0u || binding_begin > provider.lane[2] ||
      binding_extent > provider.lane[2] - binding_begin)
    return false;
  *output = ProviderIdentity{provider.lane[0], provider.lane[1],
                             provider.revision, identity, binding_begin,
                             binding_extent};
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool same_provider_identity(
    const ProviderIdentity& left, const ProviderIdentity& right) {
  return left.form == right.form && left.owner == right.owner &&
      left.revision == right.revision &&
      left.structural_identity == right.structural_identity &&
      left.binding_begin == right.binding_begin &&
      left.binding_extent == right.binding_extent;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_provider_slot(
    const ResidentRewriteState* state, const ProviderIdentity& identity) {
  if (state == nullptr || identity.form == kFormEmpty ||
      identity.owner == 0u || identity.owner == kInvalid ||
      identity.revision == 0u || identity.structural_identity == 0u ||
      identity.binding_extent == 0u)
    return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& provider = state->records[slot];
    if (provider.matter_q8 == 0u || provider.lane[0] != identity.form ||
        provider.lane[1] != identity.owner ||
        provider.revision != identity.revision)
      continue;
    if (found != kInvalid || identity.binding_begin > provider.lane[2] ||
        identity.binding_extent > provider.lane[2] - identity.binding_begin ||
        provider_structural_identity(state, slot) !=
            identity.structural_identity)
      return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline std::uint64_t alternative_member_digest(
    const ProviderIdentity& identity, std::uint32_t ordinal) {
  std::uint64_t result = identity.structural_identity ^
      (static_cast<std::uint64_t>(identity.form) << 32u) ^ identity.owner;
  result ^= rotate_provider_identity(
      (static_cast<std::uint64_t>(identity.revision) << 32u) |
          identity.binding_begin,
      17u);
  result ^= rotate_provider_identity(
      (static_cast<std::uint64_t>(identity.binding_extent) << 32u) | ordinal,
      41u);
  return result;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_alternative_set_slot(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t lifecycle) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& set = state->records[slot];
    if (set.matter_q8 == 0u ||
        set.lane[0] != kFormResidentAlternativeSet ||
        set.lane[1] != owner || set.lane[7] != lifecycle)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline bool begin_alternative_set(
    ResidentRewriteState* state, std::uint32_t expected,
    std::uint32_t resident_seed, AlternativeSetBuilder* output) {
  if (output != nullptr) *output = AlternativeSetBuilder{};
  if (state == nullptr || output == nullptr || state->fault != 0u ||
      expected < 2u || expected >= live_record_capacity(state) ||
      free_record_count(state) < expected + 1u)
    return false;
  const std::uint32_t owner = make_inquiry_owner(
      state, kFormResidentAlternativeSet,
      rewrite_mix(resident_seed, expected, state->revision));
  const std::uint32_t slot = allocate_record(state);
  if (owner == kInvalid || slot == kInvalid) return false;
  Record& set = state->records[slot];
  set.lane[0] = kFormResidentAlternativeSet;
  set.lane[1] = owner;
  set.lane[2] = expected;
  set.lane[7] = kAlternativeSetBuilding;
  ++set.revision;
  ++state->revision;
  *output = AlternativeSetBuilder{owner, expected};
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool append_alternative_provider(
    ResidentRewriteState* state, const AlternativeSetBuilder& builder,
    std::uint32_t ordinal, const ProviderIdentity& identity) {
  if (state == nullptr || state->fault != 0u ||
      ordinal >= builder.expected ||
      unique_provider_slot(state, identity) == kInvalid)
    return false;
  const std::uint32_t set_slot = unique_alternative_set_slot(
      state, builder.owner, kAlternativeSetBuilding);
  if (set_slot == kInvalid ||
      state->records[set_slot].lane[2] != builder.expected)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& member = state->records[slot];
    if (member.matter_q8 == 0u ||
        member.lane[0] != kFormResidentAlternativeProvider ||
        member.lane[1] != builder.owner)
      continue;
    const ProviderIdentity present{
        member.lane[2], member.lane[3], member.lane[4],
        (static_cast<std::uint64_t>(member.lane[6]) << 32u) | member.lane[5],
        member.lane[7], member.reserved[0]};
    if (member.reserved[1] == ordinal ||
        same_provider_identity(present, identity))
      return false;
  }
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& member = state->records[slot];
  member.lane[0] = kFormResidentAlternativeProvider;
  member.lane[1] = builder.owner;
  member.lane[2] = identity.form;
  member.lane[3] = identity.owner;
  member.lane[4] = identity.revision;
  member.lane[5] = static_cast<std::uint32_t>(identity.structural_identity);
  member.lane[6] =
      static_cast<std::uint32_t>(identity.structural_identity >> 32u);
  member.lane[7] = identity.binding_begin;
  member.reserved[0] = identity.binding_extent;
  member.reserved[1] = ordinal;
  ++member.revision;
  ++state->revision;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool alternative_provider_at(
    const ResidentRewriteState* state, std::uint32_t set_owner,
    std::uint32_t ordinal, ProviderIdentity* output,
    std::uint32_t* ephemeral_provider_slot = nullptr) {
  if (output != nullptr) *output = ProviderIdentity{};
  if (ephemeral_provider_slot != nullptr)
    *ephemeral_provider_slot = kInvalid;
  if (state == nullptr || output == nullptr) return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& member = state->records[slot];
    if (member.matter_q8 == 0u ||
        member.lane[0] != kFormResidentAlternativeProvider ||
        member.lane[1] != set_owner || member.reserved[1] != ordinal)
      continue;
    if (found != kInvalid) return false;
    found = slot;
  }
  if (found == kInvalid) return false;
  const Record& member = state->records[found];
  const ProviderIdentity identity{
      member.lane[2], member.lane[3], member.lane[4],
      (static_cast<std::uint64_t>(member.lane[6]) << 32u) | member.lane[5],
      member.lane[7], member.reserved[0]};
  const std::uint32_t provider_slot = unique_provider_slot(state, identity);
  if (provider_slot == kInvalid) return false;
  *output = identity;
  if (ephemeral_provider_slot != nullptr)
    *ephemeral_provider_slot = provider_slot;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool publish_alternative_set_ready(
    ResidentRewriteState* state, const AlternativeSetBuilder& builder,
    AlternativeSetView* output = nullptr) {
  if (output != nullptr) *output = AlternativeSetView{};
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t set_slot = unique_alternative_set_slot(
      state, builder.owner, kAlternativeSetBuilding);
  if (set_slot == kInvalid || builder.expected < 2u ||
      state->records[set_slot].lane[2] != builder.expected)
    return false;
  std::uint64_t identity = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < builder.expected; ++ordinal) {
    ProviderIdentity provider{};
    if (!alternative_provider_at(
            state, builder.owner, ordinal, &provider))
      return false;
    identity ^= rotate_provider_identity(
        alternative_member_digest(provider, ordinal), ordinal);
  }
  std::uint32_t member_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    member_count += state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormResidentAlternativeProvider &&
        state->records[slot].lane[1] == builder.owner;
  if (member_count != builder.expected || identity == 0u) return false;
  Record& set = state->records[set_slot];
  set.lane[3] = static_cast<std::uint32_t>(identity);
  set.lane[4] = static_cast<std::uint32_t>(identity >> 32u);
  ++set.revision;
  // Publication is last: readers never accept any member while this one-word
  // lifecycle remains building.
  set.lane[7] = kAlternativeSetReady;
  ++state->revision;
  refresh_receipt(state);
  if (output != nullptr)
    *output = AlternativeSetView{builder.owner, builder.expected, identity};
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool rederive_alternative_set(
    const ResidentRewriteState* state, std::uint32_t set_owner,
    AlternativeSetView* output) {
  if (output != nullptr) *output = AlternativeSetView{};
  if (state == nullptr || output == nullptr) return false;
  const std::uint32_t set_slot = unique_alternative_set_slot(
      state, set_owner, kAlternativeSetReady);
  if (set_slot == kInvalid) return false;
  const Record& set = state->records[set_slot];
  if (set.lane[2] < 2u) return false;
  std::uint64_t identity = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < set.lane[2]; ++ordinal) {
    ProviderIdentity provider{};
    if (!alternative_provider_at(state, set_owner, ordinal, &provider))
      return false;
    identity ^= rotate_provider_identity(
        alternative_member_digest(provider, ordinal), ordinal);
  }
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    count += state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormResidentAlternativeProvider &&
        state->records[slot].lane[1] == set_owner;
  const std::uint64_t stored =
      (static_cast<std::uint64_t>(set.lane[4]) << 32u) | set.lane[3];
  if (count != set.lane[2] || identity == 0u || identity != stored)
    return false;
  *output = AlternativeSetView{set_owner, count, identity};
  return true;
}

struct DormantSpanAlternative {
  ProviderIdentity provider{};
  std::uint64_t binding_identity = 0u;
};

struct DormantSpanTieReceipt {
  std::uint32_t ready = 0u;
  std::uint32_t overflow = 0u;
  std::uint32_t count = 0u;
  std::uint32_t extent = 0u;
  std::uint32_t support = 0u;
  std::uint64_t state_revision = 0u;
};

// The binding identity intentionally excludes provider owner and revision. It
// names only provider structure plus the half-open structural range, so two
// independently grown providers can expose the same binding identity without
// being collapsed into one authority lineage.
BCC32_OPEN_INQUIRY_HD inline std::uint64_t owner_independent_binding_identity(
    const ProviderIdentity& provider) {
  std::uint64_t result = provider.structural_identity ^
      (static_cast<std::uint64_t>(provider.form) << 32u);
  result ^= rotate_provider_identity(
      (static_cast<std::uint64_t>(provider.binding_begin) << 32u) |
          provider.binding_extent,
      29u);
  return result == 0u ? 0xd6e8feb86659fd93ull : result;
}

BCC32_OPEN_INQUIRY_HD inline bool eligible_dormant_span_alternative(
    ResidentRewriteState* state, std::uint32_t dormant_slot,
    std::uint32_t current_slot, std::uint32_t required_extent,
    std::uint32_t required_support, DormantSpanAlternative* output,
    std::uint32_t* observed_extent = nullptr,
    std::uint32_t* observed_support = nullptr) {
  if (output != nullptr) *output = DormantSpanAlternative{};
  if (state == nullptr || dormant_slot >= live_record_capacity(state) ||
      current_slot >= live_record_capacity(state) ||
      !cross_contact::is_dormant_history(state->records[dormant_slot]))
    return false;
  ProgramCandidateConsensus consensus{};
  if (!cross_contact::probe_dormant_join(
          state, dormant_slot, current_slot, &consensus) ||
      !consensus.have_candidate || consensus.conflict ||
      consensus.span_saw_unbound || consensus.span_saw_ambiguous ||
      !consensus.selected_from_span ||
      consensus.selected_support < kSpanProgramMatureSupport ||
      consensus.diagnostic_locus >= live_record_capacity(state) ||
      state->records[consensus.diagnostic_locus].lane[0] != kFormSpanProgram ||
      !resident_program_authoritative(state, consensus.diagnostic_locus))
    return false;
  const Record& dormant = state->records[dormant_slot];
  const Record& current = state->records[current_slot];
  const std::uint32_t extent = dormant.lane[2] + current.lane[2];
  if (observed_extent != nullptr) *observed_extent = extent;
  if (observed_support != nullptr)
    *observed_support = consensus.selected_support;
  if ((required_extent != 0u && extent != required_extent) ||
      (required_support != 0u &&
       consensus.selected_support != required_support))
    return false;
  const Record& provider = state->records[consensus.diagnostic_locus];
  if (extent >= provider.lane[2]) return false;
  ProviderIdentity identity{};
  if (!provider_identity_at(
          state, consensus.diagnostic_locus, extent,
          provider.lane[2] - extent, &identity))
    return false;
  if (output != nullptr) {
    output->provider = identity;
    output->binding_identity = owner_independent_binding_identity(identity);
  }
  return true;
}

// P1 is observational: the canonical dormant join probe borrows Records but
// rolls every byte back, and this collector itself performs no allocation or
// publication. The caller supplies storage; a too-small aperture is reported
// as overflow and never returns a truncated resident alternative set.
BCC32_OPEN_INQUIRY_HD inline DormantSpanTieReceipt
collect_dormant_span_tie_alternatives(
    ResidentRewriteState* state, std::uint32_t current_slot,
    DormantSpanAlternative* alternatives, std::uint32_t capacity) {
  DormantSpanTieReceipt receipt{};
  receipt.state_revision = state == nullptr ? 0u : state->revision;
  if (state == nullptr || alternatives == nullptr || capacity < 2u ||
      current_slot >= live_record_capacity(state))
    return receipt;
  const Record current = state->records[current_slot];
  if (current.matter_q8 == 0u || current.lane[0] != kFormTrajectory ||
      current.lane[2] == 0u || current.lane[3] != 0u || current.lane[7] != 0u)
    return receipt;
  std::uint32_t best_extent = 0u;
  std::uint32_t best_support = 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    std::uint32_t extent = 0u;
    std::uint32_t support = 0u;
    if (!eligible_dormant_span_alternative(
            state, slot, current_slot, 0u, 0u, nullptr, &extent, &support))
      continue;
    if (count == 0u || extent > best_extent ||
        (extent == best_extent && support > best_support)) {
      best_extent = extent;
      best_support = support;
      count = 1u;
    } else if (extent == best_extent && support == best_support) {
      ++count;
    }
  }
  receipt.count = count;
  receipt.extent = best_extent;
  receipt.support = best_support;
  receipt.overflow = count > capacity ? 1u : 0u;
  if (count < 2u || count > capacity || state->revision != receipt.state_revision)
    return receipt;
  std::uint32_t written = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    DormantSpanAlternative candidate{};
    if (!eligible_dormant_span_alternative(
            state, slot, current_slot, best_extent, best_support,
            &candidate))
      continue;
    for (std::uint32_t prior = 0u; prior < written; ++prior)
      if (same_provider_identity(
              alternatives[prior].provider, candidate.provider))
        return DormantSpanTieReceipt{};
    alternatives[written++] = candidate;
  }
  if (written != count || state->revision != receipt.state_revision)
    return DormantSpanTieReceipt{};
  receipt.ready = 1u;
  return receipt;
}

BCC32_OPEN_INQUIRY_HD inline bool publish_dormant_span_tie_set(
    ResidentRewriteState* state, const DormantSpanTieReceipt& tie,
    const DormantSpanAlternative* alternatives, std::uint32_t resident_seed,
    AlternativeSetView* output = nullptr) {
  if (output != nullptr) *output = AlternativeSetView{};
  if (state == nullptr || alternatives == nullptr || tie.ready == 0u ||
      tie.overflow != 0u || tie.count < 2u ||
      tie.state_revision != state->revision)
    return false;
  AlternativeSetBuilder builder{};
  if (!begin_alternative_set(state, tie.count, resident_seed, &builder))
    return false;
  for (std::uint32_t ordinal = 0u; ordinal < tie.count; ++ordinal) {
    if (alternatives[ordinal].binding_identity !=
            owner_independent_binding_identity(
                alternatives[ordinal].provider) ||
        !append_alternative_provider(
            state, builder, ordinal, alternatives[ordinal].provider)) {
      const std::uint32_t set_slot = unique_alternative_set_slot(
          state, builder.owner, kAlternativeSetBuilding);
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state);
           ++slot) {
        Record& record = state->records[slot];
        if (record.matter_q8 != 0u &&
            record.lane[0] == kFormResidentAlternativeProvider &&
            record.lane[1] == builder.owner)
          clear_record(&record);
      }
      if (set_slot != kInvalid) clear_record(&state->records[set_slot]);
      ++state->revision;
      refresh_receipt(state);
      return false;
    }
  }
  return publish_alternative_set_ready(state, builder, output);
}

struct AlternativeSource {
  std::uint32_t record_slot = kInvalid;
  std::uint32_t dormant_slot = kInvalid;
  std::uint32_t program_slot = kInvalid;
  std::uint32_t label_word = 0u;
};

struct RederivedSet {
  std::uint32_t set_slot = kInvalid;
  std::uint32_t suspended_slot = kInvalid;
  AlternativeSource alternatives[2]{};
};

struct Candidate {
  std::uint32_t dormant_slot = kInvalid;
  std::uint32_t program_slot = kInvalid;
  std::uint32_t extent = 0u;
  std::uint32_t support = 0u;
};

inline constexpr bool candidate_better(const Candidate& candidate,
                                       const Candidate& incumbent) {
  return candidate.extent > incumbent.extent ||
         (candidate.extent == incumbent.extent &&
          candidate.support > incumbent.support);
}

inline constexpr bool candidate_tied(const Candidate& left,
                                     const Candidate& right) {
  return left.extent == right.extent && left.support == right.support;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_record_by_owner_revision(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner, std::uint32_t revision) {
  if (state == nullptr || owner == 0u || owner == kInvalid || revision == 0u)
    return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != form ||
        record.lane[1] != owner || record.revision != revision)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_unresolved_set(
    const ResidentRewriteState* state) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormResidentUnresolvedSet ||
        (record.lane[7] & kUnresolvedSetCaptured) == 0u ||
        (record.lane[7] & kUnresolvedSetOpened) != 0u)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_trajectory_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormTrajectory ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline bool source_word_at(
    const ResidentRewriteState* state, const Record& dormant,
    std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr ||
      !cross_contact::is_dormant_history(dormant) || index >= dormant.lane[2])
    return false;
  if (cross_contact::is_compact_dormant_history(dormant))
    return cross_contact::compact_dormant_word_at(state, dormant, index, word);
  return trajectory_word_at(state, dormant.lane[1], index, word);
}

BCC32_OPEN_INQUIRY_HD inline bool first_distinguishing_words(
    const ResidentRewriteState* state, const Record& left, const Record& right,
    std::uint32_t* left_word, std::uint32_t* right_word) {
  if (state == nullptr || left_word == nullptr || right_word == nullptr)
    return false;
  const std::uint32_t shared = left.lane[2] < right.lane[2]
                                   ? left.lane[2]
                                   : right.lane[2];
  for (std::uint32_t index = 0u; index < shared; ++index) {
    std::uint32_t left_value = 0u;
    std::uint32_t right_value = 0u;
    if (!source_word_at(state, left, index, &left_value) ||
        !source_word_at(state, right, index, &right_value))
      return false;
    if (left_value == right_value) continue;
    *left_word = left_value;
    *right_word = right_value;
    return true;
  }
  // A strict-prefix difference has no second resident word at the divergence
  // and therefore cannot be surfaced without inventing a host sentinel.
  return false;
}

BCC32_OPEN_INQUIRY_HD inline bool exact_trajectory_digest(
    const ResidentRewriteState* state, const Record& trajectory,
    bool dormant, std::uint32_t* digest) {
  if (state == nullptr || digest == nullptr || trajectory.lane[2] == 0u)
    return false;
  std::uint32_t result = 0u;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t word = 0u;
    const bool read = dormant
                          ? source_word_at(state, trajectory, index, &word)
                          : trajectory_word_at(state, trajectory.lane[1], index,
                                               &word);
    if (!read) return false;
    result = rewrite_mix(result, word, index);
  }
  *digest = result;
  return result != 0u;
}

// Const rederivation is the authority boundary.  The snapshot records are not
// accepted unless their complete current/dormant/SpanProgram sources remain
// unique, byte-identical, equally supported, and independently distinguishable.
BCC32_OPEN_INQUIRY_HD inline bool rederive(
    const ResidentRewriteState* state, std::uint32_t set_slot,
    RederivedSet* output) {
  if (output != nullptr) *output = RederivedSet{};
  if (state == nullptr || set_slot >= live_record_capacity(state)) return false;
  const Record& set = state->records[set_slot];
  if (set.matter_q8 == 0u || set.lane[0] != kFormResidentUnresolvedSet ||
      set.lane[1] == 0u || set.lane[1] == kInvalid || set.lane[2] == 0u ||
      set.lane[2] == kInvalid || set.lane[3] == 0u || set.lane[4] == 0u ||
      set.lane[5] == 0u || set.lane[6] != 2u ||
      (set.lane[7] & ~(kUnresolvedSetCaptured | kUnresolvedSetOpened)) != 0u ||
      (set.lane[7] & kUnresolvedSetCaptured) == 0u ||
      set.reserved[0] == 0u || set.reserved[1] == 0u)
    return false;

  const std::uint32_t suspended_slot =
      unique_trajectory_by_owner(state, set.lane[2]);
  if (suspended_slot == kInvalid) return false;
  const Record& suspended = state->records[suspended_slot];
  const std::uint32_t lifecycle_delta =
      (set.lane[7] & kUnresolvedSetOpened) != 0u ? 2u : 1u;
  std::uint32_t suspended_digest = 0u;
  if (set.lane[3] > 0xffffffffu - lifecycle_delta ||
      suspended.revision != set.lane[3] + lifecycle_delta ||
      suspended.lane[2] != set.lane[4] || suspended.lane[6] != set.lane[5] ||
      (suspended.lane[7] & kTrajectoryWasYielded) == 0u ||
      !exact_trajectory_digest(state, suspended, false, &suspended_digest) ||
      suspended_digest != set.lane[5])
    return false;

  AlternativeSource alternatives[2]{};
  std::uint32_t alternative_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& source = state->records[slot];
    if (source.matter_q8 == 0u ||
        source.lane[0] != kFormResidentUnresolvedAlternative ||
        source.lane[1] != set.lane[1])
      continue;
    if (alternative_count == 2u || source.lane[2] == 0u ||
        source.lane[2] == kInvalid || source.lane[3] == 0u ||
        source.lane[4] == 0u || source.lane[4] == kInvalid ||
        source.lane[5] == 0u || source.lane[6] == 0u ||
        source.lane[6] == kInvalid || source.lane[7] == 0u ||
        source.reserved[0] != set.reserved[0] ||
        source.reserved[1] != set.reserved[1])
      return false;
    const std::uint32_t dormant_slot = unique_record_by_owner_revision(
        state, kFormTrajectory, source.lane[2], source.lane[3]);
    const std::uint32_t program_slot = unique_record_by_owner_revision(
        state, kFormSpanProgram, source.lane[6], source.lane[7]);
    std::uint32_t dormant_digest = 0u;
    if (dormant_slot == kInvalid || program_slot == kInvalid ||
        !cross_contact::is_dormant_history(state->records[dormant_slot]) ||
        state->records[dormant_slot].lane[2] != source.lane[4] ||
        state->records[dormant_slot].lane[6] != source.lane[5] ||
        !exact_trajectory_digest(state, state->records[dormant_slot], true,
                                 &dormant_digest) ||
        dormant_digest != source.lane[5] ||
        state->records[dormant_slot].lane[2] + suspended.lane[2] !=
            source.reserved[0] ||
        !resident_program_authoritative(state, program_slot) ||
        !raw_span_program_preflight(state, program_slot) ||
        state->records[program_slot].lane[3] != source.reserved[1])
      return false;
    alternatives[alternative_count++] =
        AlternativeSource{slot, dormant_slot, program_slot, 0u};
  }
  if (alternative_count != 2u ||
      alternatives[0].dormant_slot == alternatives[1].dormant_slot)
    return false;

  std::uint32_t left = 0u;
  std::uint32_t right = 0u;
  if (!first_distinguishing_words(
          state, state->records[alternatives[0].dormant_slot],
          state->records[alternatives[1].dormant_slot], &left, &right) ||
      left == right)
    return false;
  alternatives[0].label_word = left;
  alternatives[1].label_word = right;
  if (right < left ||
      (right == left && state->records[alternatives[1].dormant_slot].lane[1] <
                            state->records[alternatives[0].dormant_slot].lane[1])) {
    const AlternativeSource temporary = alternatives[0];
    alternatives[0] = alternatives[1];
    alternatives[1] = temporary;
  }
  if (output != nullptr) {
    output->set_slot = set_slot;
    output->suspended_slot = suspended_slot;
    output->alternatives[0] = alternatives[0];
    output->alternatives[1] = alternatives[1];
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool capture_before_pause(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u ||
      unique_unresolved_set(state) != kInvalid ||
      unique_active_inquiry(state) != kInvalid)
    return false;
  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid) return false;
  const Record current = state->records[current_slot];
  if (current.lane[2] == 0u || current.lane[3] != 0u || current.lane[7] != 0u)
    return false;

  Candidate best{};
  Candidate tied[2]{};
  std::uint32_t tied_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!cross_contact::is_dormant_history(state->records[slot])) continue;
    ProgramCandidateConsensus consensus{};
    if (!cross_contact::probe_dormant_join(
            state, slot, current_slot, &consensus) ||
        !consensus.have_candidate || consensus.conflict ||
        consensus.span_saw_unbound || consensus.span_saw_ambiguous ||
        !consensus.selected_from_span ||
        consensus.selected_support < kSpanProgramMatureSupport ||
        consensus.diagnostic_locus >= live_record_capacity(state) ||
        state->records[consensus.diagnostic_locus].lane[0] != kFormSpanProgram ||
        !resident_program_authoritative(state, consensus.diagnostic_locus))
      continue;
    const Candidate candidate{
        slot, consensus.diagnostic_locus,
        state->records[slot].lane[2] + current.lane[2],
        consensus.selected_support};
    if (tied_count == 0u || candidate_better(candidate, best)) {
      best = candidate;
      tied[0] = candidate;
      tied_count = 1u;
    } else if (candidate_tied(candidate, best)) {
      if (tied_count == 2u) return false;
      tied[tied_count++] = candidate;
    }
  }
  if (tied_count != 2u || tied[0].dormant_slot == tied[1].dormant_slot ||
      free_record_count(state) < 3u)
    return false;
  std::uint32_t ignored_left = 0u;
  std::uint32_t ignored_right = 0u;
  if (!first_distinguishing_words(
          state, state->records[tied[0].dormant_slot],
          state->records[tied[1].dormant_slot], &ignored_left,
          &ignored_right))
    return false;

  const std::uint32_t owner = make_inquiry_owner(
      state, kFormResidentUnresolvedSet,
      rewrite_mix(current.lane[1], state->records[tied[0].dormant_slot].lane[1],
                  state->records[tied[1].dormant_slot].lane[1]));
  if (owner == kInvalid) return false;
  const std::uint32_t set_slot = allocate_record(state);
  const std::uint32_t first_slot = allocate_record(state);
  const std::uint32_t second_slot = allocate_record(state);
  if (set_slot == kInvalid || first_slot == kInvalid || second_slot == kInvalid)
    return false;

  Record& set = state->records[set_slot];
  set.lane[0] = kFormResidentUnresolvedSet;
  set.lane[1] = owner;
  set.lane[2] = current.lane[1];
  set.lane[3] = current.revision;
  set.lane[4] = current.lane[2];
  set.lane[5] = current.lane[6];
  set.lane[6] = 2u;
  set.lane[7] = kUnresolvedSetCaptured;
  set.reserved[0] = best.extent;
  set.reserved[1] = best.support;
  ++set.revision;
  const std::uint32_t slots[2] = {first_slot, second_slot};
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const Record& dormant = state->records[tied[index].dormant_slot];
    const Record& program = state->records[tied[index].program_slot];
    Record& alternative = state->records[slots[index]];
    alternative.lane[0] = kFormResidentUnresolvedAlternative;
    alternative.lane[1] = owner;
    alternative.lane[2] = dormant.lane[1];
    alternative.lane[3] = dormant.revision;
    alternative.lane[4] = dormant.lane[2];
    alternative.lane[5] = dormant.lane[6];
    alternative.lane[6] = program.lane[1];
    alternative.lane[7] = program.revision;
    alternative.reserved[0] = best.extent;
    alternative.reserved[1] = best.support;
    ++alternative.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool open_after_pause(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u ||
      unique_active_inquiry(state) != kInvalid)
    return false;
  const std::uint32_t set_slot = unique_unresolved_set(state);
  RederivedSet sources{};
  if (set_slot == kInvalid || !rederive(state, set_slot, &sources)) return false;
  Record& set = state->records[set_slot];
  if ((set.lane[7] & kUnresolvedSetOpened) != 0u) return false;
  const std::uint32_t constructor = unique_authoritative_constructor(state);
  if (constructor == kInvalid || free_record_count(state) < 4u) return false;
  const Record constructor_snapshot = state->records[constructor];
  const std::uint32_t owner = make_inquiry_owner(
      state, kFormOpenInquiry,
      rewrite_mix(set.lane[1], constructor_snapshot.lane[1],
                  constructor_snapshot.revision));
  if (owner == kInvalid) return false;

  const std::uint32_t inquiry_slot = allocate_record(state);
  const std::uint32_t first_slot = allocate_record(state);
  const std::uint32_t second_slot = allocate_record(state);
  const std::uint32_t emission_slot = allocate_record(state);
  if (inquiry_slot == kInvalid || first_slot == kInvalid ||
      second_slot == kInvalid || emission_slot == kInvalid)
    return false;
  Record& inquiry = state->records[inquiry_slot];
  inquiry.lane[0] = kFormOpenInquiry;
  inquiry.lane[1] = owner;
  inquiry.lane[2] = set.lane[1];
  inquiry.lane[3] = set.lane[4];
  inquiry.lane[4] = 2u;
  inquiry.lane[5] = constructor_snapshot.lane[1];
  inquiry.lane[6] = constructor_snapshot.revision;
  inquiry.lane[7] = kInquiryAwaitingReply;
  inquiry.reserved[0] = kUnresolvedInquiryMarker;
  inquiry.reserved[1] = constructor;
  ++inquiry.revision;
  const std::uint32_t bindings[2] = {first_slot, second_slot};
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const Record& source = state->records[sources.alternatives[index].record_slot];
    Record& binding = state->records[bindings[index]];
    binding.lane[0] = kFormOpenInquiryAlternative;
    binding.lane[1] = owner;
    binding.lane[2] = source.lane[2];
    binding.lane[3] = source.lane[3];
    binding.lane[4] = sources.alternatives[index].label_word;
    ++binding.revision;
  }
  Record& emission = state->records[emission_slot];
  emission.lane[0] = kFormOpenInquiryEmission;
  emission.lane[1] = owner;
  emission.lane[2] = constructor;
  emission.lane[3] = 0u;
  emission.lane[4] = constructor_snapshot.lane[1];
  emission.lane[5] = constructor_snapshot.revision;
  emission.lane[6] = kInquiryUnresolvedAlternativeSurface;
  ++emission.revision;
  Record& suspended = state->records[sources.suspended_slot];
  suspended.lane[3] = 1u;
  suspended.lane[7] |= kTrajectoryOpenInquiry;
  ++suspended.revision;
  set.lane[7] |= kUnresolvedSetOpened;
  ++set.revision;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

// This is the same learned Constructor -> ordinary generated-trajectory seam
// used by open inquiry.  The special cursor exists only so the canonical
// VersionSpace reader cannot mistake dormant-history owners for Programs.
BCC32_OPEN_INQUIRY_HD inline bool advance_surface_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return false;
  std::uint32_t emission_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& emission = state->records[slot];
    if (emission.matter_q8 == 0u || emission.lane[0] != kFormOpenInquiryEmission ||
        emission.lane[6] != kInquiryUnresolvedAlternativeSurface)
      continue;
    if (emission_slot != kInvalid) return false;
    emission_slot = slot;
  }
  if (emission_slot == kInvalid) return false;
  Record& emission = state->records[emission_slot];
  const std::uint32_t inquiry_slot =
      unique_header_by_owner(state, kFormOpenInquiry, emission.lane[1]);
  if (inquiry_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  if (inquiry.reserved[0] != kUnresolvedInquiryMarker ||
      inquiry.reserved[1] != emission.lane[2] ||
      (inquiry.lane[7] & kInquirySettled) != 0u)
    return false;
  const std::uint32_t set_slot =
      unique_header_by_owner(state, kFormResidentUnresolvedSet, inquiry.lane[2]);
  RederivedSet sources{};
  if (set_slot == kInvalid || !rederive(state, set_slot, &sources)) return false;
  const Record& constructor = state->records[emission.lane[2]];
  if (constructor.lane[1] != emission.lane[4] ||
      constructor.revision != emission.lane[5] ||
      inquiry.lane[5] != constructor.lane[1] ||
      inquiry.lane[6] != constructor.revision ||
      !inquiry_constructor_authoritative(state, emission.lane[2]) ||
      emission.lane[3] >= constructor.lane[2])
    return false;
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  if (!constructor_term_at(state, constructor, emission.lane[3], &kind, &value))
    return false;
  std::uint32_t word = value;
  if (kind == kTermFirstAlternative)
    word = sources.alternatives[0].label_word;
  else if (kind == kTermSecondAlternative)
    word = sources.alternatives[1].label_word;
  else if (kind != kTermLiteral)
    return false;
  if (!append_trajectory_word(state, word, true)) return false;
  state->generated_word = word;
  state->generated_word_valid = 1u;
  state->generated_locus = emission.lane[2];
  state->active_locus = emission.lane[2];
  ++emission.lane[3];
  ++emission.revision;
  inquiry.lane[7] |= kInquirySurfaceEmitted;
  ++inquiry.revision;
  if (emission.lane[3] == constructor.lane[2]) clear_record(&emission);
  ++state->revision;
  refresh_receipt(state);
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite::open_inquiry::
   // unresolved_alternative

#undef BCC32_OPEN_INQUIRY_HD
