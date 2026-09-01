#pragma once

// Resident construction/slot composer.
//
// Mechanism (doctrine-legal generative grammar):
//   LEARN  During assimilation every episode (sentence-like segment) of legal
//          shape is abstracted into a CONSTRUCTION SKELETON: the sequence of
//          its recurrent, structurally selective closed-class units kept as
//          literal unit ids (shared grammatical glue, discovered from resident
//          frequency plus structural-role population evidence), with every
//          content-word position replaced by a
//          TYPED SLOT carrying only the unit's emergent structural role
//          (resident_roles projection, 64 classes). The store keeps NO
//          content words for slot positions -- copying slot content back out
//          is structurally impossible.
//   SELECT At generation time a skeleton is chosen by GRAMMATICAL FIT: can
//          its slot-role multiset be covered by the currently active content
//          matter (resident subject field + composed completion units)?
//          Score = corpus-wide support (generality) x activation mass of the
//          role-matching pool. The query byte string is never compared with
//          corpus spans here; the only cue influence is the resident subject
//          field the cue conditioning already latched.
//   BIND   Each slot is filled with the strongest role-matching active
//          content unit (reuse-penalised); literals + fillers realize in
//          construction order. Source novelty remains an explicit promotion
//          gate rather than an asserted property of this store.
//
// Resident, mutable, lesionable: lesioning the store (or an empty store)
// falls back to the legacy generation path.

#include <cuda_runtime.h>

#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

namespace substrate::bcc32::resident_construction {

inline constexpr std::uint32_t kConstructionCap = 8192u;
// Founder exposure must not consume every writable construction slot. The
// retained quarter remains ordinary resident matter and is available to all
// later contacts; it is not assigned to a language category or endpoint.
inline constexpr std::uint32_t kInitialConstructionCap =
    kConstructionCap - kConstructionCap / 4u;
inline constexpr std::uint32_t kConstructionHashCap = 32768u;  // power of two
inline constexpr std::uint32_t kConstructionMaxTokens = 24u;
inline constexpr std::uint32_t kConstructionMinSlots = 2u;
inline constexpr std::uint32_t kConstructionMinTokens =
    kConstructionMinSlots + 1u;
inline constexpr std::uint32_t kConstructionMaxSlots = 4u;
inline constexpr std::uint32_t kQuestionAnswerArityCount =
    kConstructionMaxSlots + 1u;
inline constexpr std::uint32_t kConstructionSlotPopulationCap = 16u;
inline constexpr std::uint32_t kConstructionMinRoleEvidence = 2u;
inline constexpr std::uint32_t kConstructionMaxLiteralRun = 4u;
inline constexpr std::uint32_t kConstructionPoolCap = 96u;
inline constexpr std::uint32_t kConstructionFuncRank = 80u;
inline constexpr std::uint32_t kConstructionMinFillerBytes = 4u;
inline constexpr std::uint32_t kConstructionBlock = 256u;
inline constexpr std::uint32_t kConstructionRoleCount =
    resident_roles::kStructuralRoleCount;
inline constexpr std::uint32_t kSlotFlag = 0x80000000u;
inline constexpr std::uint32_t kNoConstruction = 0xffffffffu;
inline constexpr std::uint32_t kAmbiguousConstruction = 0xfffffffeu;
// Morphological agreement signal: opaque byte-suffix classes (hash of the
// word-final byte window) whose adjacency statistics are LEARNED from the
// resident stream on-device. No byte value or linguistic rule is authored;
// the classes are just byte-derived tags and the compatibility is pure
// co-occurrence counting.
inline constexpr std::uint32_t kSuffixClassCount = 1024u;
inline constexpr std::uint32_t kSuffixWindow = 3u;
inline constexpr std::uint32_t kNoSuffixClass = kSuffixClassCount;
inline constexpr std::uint32_t kSuffixGateMaxShift = 12u;
inline constexpr std::uint32_t kHashClaim = 0xffffffffu;
inline constexpr std::uint32_t kHashDead = 0xfffffffeu;

[[nodiscard]] __host__ __device__ inline bool token_is_slot(std::uint32_t token) {
  return (token & kSlotFlag) != 0u;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t token_role(std::uint32_t token) {
  return token & ~kSlotFlag;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t construction_role(
    resident_roles::MutableStructuralRole role) {
  return role.role;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t construction_mix(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

[[nodiscard]] __device__ inline std::uint32_t construction_pattern_hash(
    const std::uint32_t* tokens, std::uint32_t index, std::uint32_t extent) {
  std::uint32_t hash = 2166136261u;
  const std::size_t base = static_cast<std::size_t>(index) * kConstructionMaxTokens;
  for (std::uint32_t i = 0u; i < extent; ++i)
    hash = (hash ^ tokens[base + i]) * 16777619u;
  return construction_mix(hash ^ extent);
}

[[nodiscard]] __host__ __device__ inline std::size_t construction_slot_index(
    std::uint32_t construction_index, std::uint32_t slot) {
  return static_cast<std::size_t>(construction_index) * kConstructionMaxSlots + slot;
}

[[nodiscard]] __host__ __device__ inline std::size_t construction_slot_member_index(
    std::uint32_t construction_index, std::uint32_t slot, std::uint32_t member) {
  return construction_slot_index(construction_index, slot) *
             kConstructionSlotPopulationCap +
         member;
}

// A construction slot is a learned resident population, not only a role hash.
// Every recurrent observation contributes one unit of mass to the concrete unit
// that occupied that slot. Capacity loss is explicit: once overflowed, the
// population may no longer authorize public realization.
__device__ inline void observe_construction_slot_population(
    const std::uint32_t* sequence, std::uint32_t begin, std::uint32_t extent,
    const std::uint32_t* local_tokens, std::uint32_t construction_index,
    std::uint32_t* slot_units, std::uint32_t* slot_masses,
    std::uint32_t* slot_totals, std::uint32_t* slot_overflow) {
  std::uint32_t slot = 0u;
  for (std::uint32_t position = 0u; position < extent; ++position) {
    if (!token_is_slot(local_tokens[position]))
      continue;
    const std::uint32_t unit = sequence[begin + position];
    const std::size_t slot_index = construction_slot_index(construction_index, slot);
    atomicAdd(slot_totals + slot_index, 1u);
    bool retained = false;
    for (std::uint32_t member = 0u; member < kConstructionSlotPopulationCap;
         ++member) {
      const std::size_t index =
          construction_slot_member_index(construction_index, slot, member);
      const std::uint32_t prior = atomicCAS(slot_units + index, kNoConstruction, unit);
      if (prior == kNoConstruction || prior == unit) {
        atomicAdd(slot_masses + index, 1u);
        retained = true;
        break;
      }
    }
    if (!retained)
      atomicAdd(slot_overflow + slot_index, 1u);
    ++slot;
  }
}

static __global__ void count_role_population_kernel(
    const resident_roles::MutableStructuralRole* roles, std::uint32_t unit_count,
    std::uint32_t* role_population) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count || roles[unit].confidence == 0u ||
      roles[unit].role >= resident_roles::kStructuralRoleCount)
    return;
  atomicAdd(role_population + roles[unit].role, 1u);
}

// The closed-class decision is computed ONCE on the host at learn time (from
// resident vitality rank + role confidence, empirically validated against the
// store yield) and published as a per-unit mask. Device code only reads the
// mask, so learn/pool/bind can never diverge on what counts as glue.
[[nodiscard]] __device__ inline bool construction_closed_class(
    std::uint32_t unit, const std::uint32_t* closed_class_mask) {
  return closed_class_mask[unit] != 0u;
}

[[nodiscard]] __device__ inline std::uint32_t construction_unit_byte(
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    std::uint32_t unit, std::uint32_t offset) {
  const std::uint32_t word = unit_content[unit * unit_words + offset / 4u];
  return (word >> ((offset % 4u) * 8u)) & 0xffu;
}

// Terminal (last non-boundary) byte of a unit; 256u when none.
[[nodiscard]] __device__ inline std::uint32_t construction_terminal_byte(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit,
    const std::uint32_t* boundary_mask) {
  const std::uint32_t length = unit_lengths[unit];
  for (std::uint32_t back = 0u; back < length; ++back) {
    const std::uint32_t value =
        construction_unit_byte(unit_content, unit_words, unit, length - 1u - back);
    if (boundary_mask[value] == 0u) return value;
  }
  return 256u;
}

// A unit ends with closure when its observed terminal byte is one of the
// discovered sentence-closure bytes. Boundary membership cannot veto that
// evidence: learned unit boundaries commonly include the very terminal byte
// whose transition statistics made it a closure witness.
[[nodiscard]] __device__ inline bool construction_unit_ends_with_closure(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit,
    const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count) {
  (void)boundary_mask;
  const std::uint32_t length = unit_lengths[unit];
  if (length == 0u) return false;
  const std::uint32_t trailing =
      construction_unit_byte(unit_content, unit_words, unit, length - 1u);
  for (std::uint32_t back = 0u; back < length; ++back) {
    const std::uint32_t value =
        construction_unit_byte(unit_content, unit_words, unit, length - 1u - back);
    for (std::uint32_t index = 0u; index < closure_count; ++index) {
      if (value == closure_bytes[index]) return true;
    }
    if (value != trailing) return false;
  }
  return false;
}

// Opaque suffix class of a unit: hash of its last kSuffixWindow bytes ending
// at the terminal non-boundary byte (trailing separators skipped, exactly the
// terminal-byte convention above). Derived from the raw byte content the
// adult already stores on-device -- NOT an authored morphological rule; a
// class is just a byte-window tag ("-ion", "-was", "-th." collapse to opaque
// ids), and any linguistic meaning is discovered by the transition counts.
[[nodiscard]] __device__ inline std::uint32_t construction_suffix_class(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit,
    const std::uint32_t* boundary_mask) {
  const std::uint32_t length = unit_lengths[unit];
  std::uint32_t terminal = length;
  for (std::uint32_t back = 0u; back < length; ++back) {
    const std::uint32_t value =
        construction_unit_byte(unit_content, unit_words, unit, length - 1u - back);
    if (boundary_mask[value] == 0u) {
      terminal = length - 1u - back;
      break;
    }
  }
  if (terminal >= length) return kNoSuffixClass;
  const std::uint32_t window =
      terminal + 1u < kSuffixWindow ? terminal + 1u : kSuffixWindow;
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t i = 0u; i < window; ++i) {
    const std::uint32_t value = construction_unit_byte(
        unit_content, unit_words, unit, terminal + 1u - window + i);
    hash = (hash ^ value) * 16777619u;
  }
  return construction_mix(hash ^ window) & (kSuffixClassCount - 1u);
}

// LEARN suffix-class adjacency from the assimilated resident stream: for
// every adjacent unit pair, count (suffix-class of left) -> (suffix-class of
// right). Runs entirely on-device over the device-resident sequence -- the
// raw bytes never cross to the host for this signal. This is where the
// organism discovers, from co-occurrence alone, that a given subject-suffix
// population tends to be followed by a given verb-suffix population.
static __global__ void learn_suffix_transitions_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    std::uint32_t* suffix_transitions) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position + 1u >= sequence_count) return;
  const std::uint32_t left = construction_suffix_class(
      unit_lengths, unit_content, unit_words, sequence[position], boundary_mask);
  const std::uint32_t right = construction_suffix_class(
      unit_lengths, unit_content, unit_words, sequence[position + 1u],
      boundary_mask);
  if (left >= kSuffixClassCount || right >= kSuffixClassCount) return;
  atomicAdd(suffix_transitions + left * kSuffixClassCount + right, 1u);
}

// Learned agreement gate: log-depth of the observed suffix-class transition
// count. 0 when the adjacency was never seen; grows with evidence.
[[nodiscard]] __device__ inline std::uint32_t construction_suffix_gate_depth(
    const std::uint32_t* suffix_transitions, std::uint32_t left_class,
    std::uint32_t right_class) {
  if (left_class >= kSuffixClassCount || right_class >= kSuffixClassCount)
    return 0u;
  return resident_roles::integer_log_depth(
      suffix_transitions[left_class * kSuffixClassCount + right_class]);
}

// Abstract one closed sentence span [begin, end) into a skeleton and store it
// (dedup via open-addressed hash of the token pattern, support accumulated).
__device__ inline void learn_one_construction(
    const std::uint32_t* sequence, std::uint32_t begin, std::uint32_t end,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* closed_class_mask, const std::uint32_t* unit_lengths,
    std::uint32_t* tokens, std::uint32_t* lengths, std::uint32_t* slot_counts,
    std::uint32_t* supports, std::uint32_t* slot_units,
    std::uint32_t* slot_masses, std::uint32_t* slot_totals,
    std::uint32_t* slot_overflow, std::uint32_t* hash_slots,
    std::uint32_t* construction_count, std::uint64_t evidence_revision,
    std::uint64_t* construction_evidence_revision,
    std::uint64_t* construction_origin_revision,
    bool population_only = false, bool terminal_literal = true,
    std::uint32_t* learned_construction = nullptr,
    std::uint32_t allocation_capacity = kConstructionCap) {
  if (learned_construction != nullptr)
    learned_construction[0] = kNoConstruction;
  const std::uint32_t extent = end - begin;
  const std::uint32_t minimum_extent =
      population_only ? kConstructionMinSlots : kConstructionMinTokens;
  if (extent < minimum_extent || extent > kConstructionMaxTokens) return;
  // Frame hygiene: no fragment/formatting tokens, and the closure-carrying
  // final token must be an ordinary short word (kills citation tails).
  for (std::uint32_t i = 0u; i < extent; ++i) {
    if (unit_lengths[sequence[begin + i]] < 2u) return;
  }
  if (unit_lengths[sequence[end - 1u]] > 12u) return;

  std::uint32_t local[kConstructionMaxTokens];
  std::uint32_t slots = 0u;
  std::uint32_t run = 0u;
  std::uint32_t max_run = 0u;
  for (std::uint32_t i = 0u; i < extent; ++i) {
    const std::uint32_t unit = sequence[begin + i];
    const bool closed_class =
        !population_only && construction_closed_class(unit, closed_class_mask);
    (void)terminal_literal;
    if (closed_class) {
      // Literal grammatical glue. A terminal content unit remains a typed
      // slot; retaining it only because it carries closure would inject the
      // training fact into later grounded realizations of this skeleton.
      local[i] = unit;
      ++run;
      if (run > max_run) max_run = run;
    } else {
      local[i] = kSlotFlag | construction_role(roles[unit]);
      ++slots;
      run = 0u;
    }
  }
  if (slots < kConstructionMinSlots || slots > kConstructionMaxSlots) return;
  if (!population_only && max_run > kConstructionMaxLiteralRun) return;
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t i = 0u; i < extent; ++i) hash = (hash ^ local[i]) * 16777619u;
  hash = construction_mix(hash ^ extent);

  std::uint32_t probe = hash & (kConstructionHashCap - 1u);
  for (std::uint32_t attempt = 0u; attempt < kConstructionHashCap; ++attempt) {
    std::uint32_t previous =
        atomicCAS(&hash_slots[probe], 0u, kHashClaim);
    bool claimed = previous == 0u;
    if (!claimed && previous == kHashDead &&
        allocation_capacity == kConstructionCap) {
      claimed =
          atomicCAS(&hash_slots[probe], kHashDead, kHashClaim) == kHashDead;
    }
    if (claimed) {
      // We claimed this slot: allocate a fresh skeleton.
      std::uint32_t index = construction_count[0];
      while (index < allocation_capacity) {
        const std::uint32_t observed =
            atomicCAS(construction_count, index, index + 1u);
        if (observed == index) break;
        index = observed;
      }
      if (index >= allocation_capacity) {
        hash_slots[probe] = kHashDead;
        return;
      }
      for (std::uint32_t i = 0u; i < extent; ++i)
        tokens[index * kConstructionMaxTokens + i] = local[i];
      lengths[index] = extent;
      slot_counts[index] = slots;
      supports[index] = 1u;
      observe_construction_slot_population(
          sequence, begin, extent, local, index, slot_units, slot_masses,
          slot_totals, slot_overflow);
      if (construction_evidence_revision != nullptr)
        construction_evidence_revision[index] = evidence_revision;
      if (construction_origin_revision != nullptr)
        construction_origin_revision[index] = evidence_revision;
      if (learned_construction != nullptr)
        learned_construction[0] = index;
      __threadfence();
      hash_slots[probe] = index + 1u;
      return;
    }
    std::uint32_t stored = previous;
    while (stored == kHashClaim) {
      // Another thread is publishing this slot; wait for its index.
      stored = *reinterpret_cast<volatile std::uint32_t*>(&hash_slots[probe]);
    }
    if (stored != kHashDead) {
      // Acquire side of the publish protocol: the volatile spin-wait above
      // only orders this thread's own read of hash_slots[probe]. Without a
      // fence here, this thread can observe the publisher's release write
      // (hash_slots[probe] = index + 1u) before it observes the publisher's
      // earlier stores to tokens[]/lengths[]/slot_counts[]/supports[] at
      // that index, reading stale/torn skeleton data (0X1-224).
      __threadfence();
      const std::uint32_t index = stored - 1u;
      if (lengths[index] == extent) {
        bool equal = true;
        for (std::uint32_t i = 0u; i < extent; ++i) {
          if (tokens[index * kConstructionMaxTokens + i] != local[i]) {
            equal = false;
            break;
          }
        }
        if (equal) {
          atomicAdd(&supports[index], 1u);
          observe_construction_slot_population(
              sequence, begin, extent, local, index, slot_units, slot_masses,
              slot_totals, slot_overflow);
          if (construction_evidence_revision != nullptr)
            construction_evidence_revision[index] = evidence_revision;
          if (learned_construction != nullptr)
            learned_construction[0] = index;
          return;
        }
      }
    }
    probe = (probe + 1u) & (kConstructionHashCap - 1u);
  }
}

// ---------------------------------------------------------------------------
// LEARN: one thread per resident episode. Episodes (paragraph-scale on this
// substrate) are re-segmented into sentences at units carrying the
// construction organ's OWN discovered sentence-closure bytes (next-start
// rarity statistic -- '.'-class, not the clause-level ':'/';' the episode
// machinery uses). Each complete closed sentence of legal shape becomes a
// skeleton.
// ---------------------------------------------------------------------------
static __global__ void learn_constructions_kernel(
    const std::uint32_t* sequence, const std::uint32_t* episode_offsets,
    std::uint32_t episode_count,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes,
    std::uint32_t closure_count, std::uint32_t* tokens, std::uint32_t* lengths,
    std::uint32_t* slot_counts, std::uint32_t* supports,
    std::uint32_t* slot_units, std::uint32_t* slot_masses,
    std::uint32_t* slot_totals, std::uint32_t* slot_overflow,
    std::uint32_t* hash_slots, std::uint32_t* construction_count,
    const std::uint64_t* evidence_revision,
    std::uint64_t* construction_evidence_revision,
    std::uint64_t* construction_origin_revision,
    std::uint32_t allocation_capacity) {
  const std::uint32_t episode = blockIdx.x * blockDim.x + threadIdx.x;
  if (episode >= episode_count) return;
  const std::uint32_t begin = episode_offsets[episode];
  const std::uint32_t end = episode_offsets[episode + 1u];
  if (end <= begin) return;
  const std::uint64_t event_revision = evidence_revision == nullptr
      ? 0u
      : ((evidence_revision[0] << 32u) |
         (static_cast<std::uint64_t>(begin) + 1u));

  // Some bodies learn stable episode boundaries before any particular byte
  // becomes a sentence-closure attractor.  In that regime the whole resident
  // episode is still repeated order evidence: retain its learned role
  // trajectory without inventing literal glue or requiring a host delimiter.
  if (closure_count == 0u) {
    learn_one_construction(sequence, begin, end, roles, closed_class_mask,
                           unit_lengths, tokens, lengths, slot_counts, supports,
                           slot_units, slot_masses, slot_totals, slot_overflow,
                           hash_slots, construction_count, event_revision,
                           construction_evidence_revision,
                           construction_origin_revision, true, true, nullptr,
                           allocation_capacity);
    return;
  }

  std::uint32_t sentence_begin = begin;
  bool whole_sentence = true;
  for (std::uint32_t i = begin; i < end; ++i) {
    const bool closes = construction_unit_ends_with_closure(
        unit_lengths, unit_content, unit_words, sequence[i], boundary_mask,
        closure_bytes, closure_count);
    if (closes) {
      if (whole_sentence) {
        learn_one_construction(sequence, sentence_begin, i + 1u, roles,
                               closed_class_mask, unit_lengths, tokens, lengths,
                               slot_counts, supports, slot_units, slot_masses,
                               slot_totals, slot_overflow, hash_slots,
                               construction_count, event_revision,
                               construction_evidence_revision,
                               construction_origin_revision, false, true,
                               nullptr, allocation_capacity);
      }
      sentence_begin = i + 1u;
      whole_sentence = true;
    } else if (i + 1u - sentence_begin > kConstructionMaxTokens) {
      whole_sentence = false;
    }
  }
}

// ---------------------------------------------------------------------------
// ASSOCIATION EXPANSION: accumulate resident association mass from the active
// subject units onto their partners. This is the adult's own learned
// association matter (subject -> associated content), not query-span
// retrieval: the query bytes never touch this path.
// ---------------------------------------------------------------------------
template <typename AssociationKeyT>
static __global__ void accumulate_subject_association_kernel(
    const AssociationKeyT* associations, const std::uint32_t* association_counts,
    std::uint32_t association_count, const std::uint32_t* subject_ids,
    const std::uint32_t* subject_count, std::uint32_t subject_cap,
    unsigned long long* mass) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= association_count || association_counts[i] == 0u) return;
  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  const std::uint32_t first = associations[i].first;
  const std::uint32_t second = associations[i].second;
  bool first_subject = false;
  bool second_subject = false;
  for (std::uint32_t k = 0u; k < subjects; ++k) {
    first_subject |= subject_ids[k] == first;
    second_subject |= subject_ids[k] == second;
  }
  if (first_subject && !second_subject)
    atomicAdd(mass + second, static_cast<unsigned long long>(association_counts[i]));
  if (second_subject && !first_subject)
    atomicAdd(mass + first, static_cast<unsigned long long>(association_counts[i]));
}

// ---------------------------------------------------------------------------
// CONTENT COMMITMENT: an ordered WHOLE-REPLY content plan formed BEFORE any
// frame is selected. From the resident subject field (latched by cue
// conditioning) and the adult's own online association matter (the
// subject-partner co-occurrence mass accumulated above), commit an ORDERED
// list of the content units the reply must express: the strongest legal
// subject unit first (the reply names its topic), then the association
// partners in DESCENDING resident association mass. The composer downstream
// may only RENDER this plan -- it never re-decides content. Resident,
// mutable, lesionable (skipping formation restores pool-driven slot
// filling). No corpus span is consulted: every committed unit is a single
// resident unit chosen by resident signals, recombined with learned glue.
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kCommitmentCap = 16u;

// Commitment-specific association accumulation. Differs from the pool's
// accumulate_subject_association_kernel in one way: only CONTENT subject
// units act as sources. The latched subject field also carries the
// question's glue ("what", "the"), and glue pairs with everything -- letting
// it radiate association mass swamps the topical signal the commitment
// ranks by. Writes its own mass buffer; the pool's buffer is untouched, so
// the lesioned path stays byte-identical.
template <typename AssociationKeyT>
static __global__ void accumulate_commitment_association_kernel(
    const AssociationKeyT* associations, const std::uint32_t* association_counts,
    std::uint32_t association_count, const std::uint32_t* subject_ids,
    const std::uint32_t* subject_count, std::uint32_t subject_cap,
    const std::uint32_t* closed_class_mask, unsigned long long* mass) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= association_count || association_counts[i] == 0u) return;
  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  const std::uint32_t first = associations[i].first;
  const std::uint32_t second = associations[i].second;
  bool first_subject = false;
  bool second_subject = false;
  for (std::uint32_t k = 0u; k < subjects; ++k) {
    first_subject |= subject_ids[k] == first;
    second_subject |= subject_ids[k] == second;
  }
  if (first_subject && !second_subject && closed_class_mask[first] == 0u)
    atomicAdd(mass + second,
              static_cast<unsigned long long>(association_counts[i]));
  if (second_subject && !first_subject && closed_class_mask[second] == 0u)
    atomicAdd(mass + first,
              static_cast<unsigned long long>(association_counts[i]));
}

[[nodiscard]] __device__ inline bool commitment_filler_legal(
    std::uint32_t unit, const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* filler_terminal_mask) {
  if (unit_lengths[unit] < kConstructionMinFillerBytes) return false;
  if (construction_closed_class(unit, closed_class_mask)) return false;
  const std::uint32_t terminal = construction_terminal_byte(
      unit_lengths, unit_content, unit_words, unit, boundary_mask);
  return terminal < 256u && filler_terminal_mask[terminal] != 0u;
}

// Near-duplicate guard (shared 4-byte prefix, the bind pass-0 convention) so
// the plan never commits "growth"/"growths"/"growth-" variants as distinct
// content items.
[[nodiscard]] __device__ inline bool commitment_near_duplicate(
    std::uint32_t unit, const std::uint32_t* committed, std::uint32_t count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words) {
  for (std::uint32_t k = 0u; k < count; ++k) {
    const std::uint32_t other = committed[k];
    const std::uint32_t span =
        min(4u, min(unit_lengths[unit], unit_lengths[other]));
    bool same = span != 0u;
    for (std::uint32_t b = 0u; b < span && same; ++b) {
      same = construction_unit_byte(unit_content, unit_words, unit, b) ==
             construction_unit_byte(unit_content, unit_words, other, b);
    }
    if (same) return true;
  }
  return false;
}

static __global__ void form_content_commitment_kernel(
    const unsigned long long* association_mass,
    const std::uint32_t* unit_vitality, std::uint32_t unit_count,
    const std::uint32_t* subject_ids, const std::uint32_t* subject_weights,
    const std::uint32_t* subject_count, std::uint32_t subject_cap,
    const std::uint32_t* closed_class_mask, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* boundary_mask,
    const std::uint32_t* filler_terminal_mask, std::uint32_t* commitment_units,
    std::uint32_t* commitment_meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t count = 0u;
  // 1) Topic: the strongest legal subject-field unit.
  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  std::uint32_t topic = kNoConstruction;
  std::uint32_t topic_weight = 0u;
  for (std::uint32_t i = 0u; i < subjects; ++i) {
    const std::uint32_t unit = subject_ids[i];
    if (!commitment_filler_legal(unit, closed_class_mask, unit_lengths,
                                 unit_content, unit_words, boundary_mask,
                                 filler_terminal_mask))
      continue;
    if (subject_weights[i] > topic_weight) {
      topic_weight = subject_weights[i];
      topic = unit;
    }
  }
  if (topic != kNoConstruction) commitment_units[count++] = topic;
  // 2) Predicating content: association partners in DESCENDING resident
  //    association STRENGTH -- co-occurrence mass normalized by the
  //    partner's own global vitality (a resident PMI analogue; raw counts
  //    only rank frequency, which surfaces generic filler). mass >= 2; ties
  //    resolve to the lowest unit id, deterministic. The scan never writes
  //    the mass array.
  while (count < kCommitmentCap) {
    unsigned long long best_strength = 0ull;
    std::uint32_t best_unit = kNoConstruction;
    for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
      const unsigned long long mass = association_mass[unit];
      if (mass < 2ull) continue;
      const unsigned long long strength =
          (mass << 12u) /
          (static_cast<unsigned long long>(unit_vitality[unit]) + 16ull);
      if (strength <= best_strength) continue;
      bool already = false;
      for (std::uint32_t k = 0u; k < count; ++k)
        already |= commitment_units[k] == unit;
      if (already) continue;
      if (!commitment_filler_legal(unit, closed_class_mask, unit_lengths,
                                   unit_content, unit_words, boundary_mask,
                                   filler_terminal_mask))
        continue;
      if (commitment_near_duplicate(unit, commitment_units, count,
                                    unit_lengths, unit_content, unit_words))
        continue;
      best_strength = strength;
      best_unit = unit;
    }
    if (best_unit == kNoConstruction) break;
    commitment_units[count++] = best_unit;
  }
  commitment_meta[0] = count;
  commitment_meta[1] = 0u;  // serialization cursor: nothing expressed yet
}

// RELATIONAL COMMITMENT: commit (S, P1, V1, P2, V2, ...) from the resident
// anchor-conditioned DIRECTED transition store (anchor, previous -> next)
// instead of symmetric co-occurrence mass. Co-occurrence ranks salience --
// "words about X" -- and carries no directionality; the conditioned store
// records which transitions the subject actually licenses, so a committed
// clause (P -> V) is an attested directed claim in the subject's contexts,
// not merely a co-active word. Clauses are ranked by DESCENDING attested
// count; both clause members must pass the same filler hygiene and
// near-duplicate guards as the co-occurrence plan. Writes nothing usable
// (meta[0] = 0) when the store holds no well-evidenced clause for the
// topic, so the caller falls back to the co-occurrence commitment.
// Lesionable upstream via BCC32_LESION_RELATION_COMMIT (the caller skips
// this kernel entirely -> exact co-occurrence commitment behavior).
inline constexpr std::uint32_t kRelationCommitMinCount = 2u;
inline constexpr std::uint32_t kRelationCommitGlueRun = 3u;

template <typename TransitionKeyT>
static __global__ void form_relational_commitment_kernel(
    const TransitionKeyT* transitions, const std::uint32_t* transition_counts,
    std::uint32_t transition_count, const std::uint32_t* unit_vitality,
    const std::uint32_t* subject_ids,
    const std::uint32_t* subject_weights, const std::uint32_t* subject_count,
    std::uint32_t subject_cap, const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* boundary_mask,
    const std::uint32_t* filler_terminal_mask, std::uint32_t* commitment_units,
    std::uint32_t* commitment_meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  commitment_meta[0] = 0u;
  commitment_meta[1] = 0u;
  // 1) Topic: the strongest legal subject-field unit (identical rule to the
  //    co-occurrence commitment, so ON/OFF differ only in what predicates).
  const std::uint32_t subjects =
      subject_count[0] < subject_cap ? subject_count[0] : subject_cap;
  std::uint32_t topic = kNoConstruction;
  std::uint32_t topic_weight = 0u;
  for (std::uint32_t i = 0u; i < subjects; ++i) {
    const std::uint32_t unit = subject_ids[i];
    if (!commitment_filler_legal(unit, closed_class_mask, unit_lengths,
                                 unit_content, unit_words, boundary_mask,
                                 filler_terminal_mask))
      continue;
    if (subject_weights[i] > topic_weight) {
      topic_weight = subject_weights[i];
      topic = unit;
    }
  }
  if (topic == kNoConstruction) return;
  // 2) The topic's directed range in the sorted (anchor, previous, next)
  //    store -- binary search, no helper dependency.
  std::uint32_t lo = 0u, hi = transition_count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (transitions[mid].anchor < topic) lo = mid + 1u; else hi = mid;
  }
  const std::uint32_t begin = lo;
  hi = transition_count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (transitions[mid].anchor <= topic) lo = mid + 1u; else hi = mid;
  }
  const std::uint32_t end = lo;
  if (begin >= end) return;
  // 3) Subject-anchored directed CHAIN: start AT the topic and repeatedly
  //    follow the strongest attested continuation (anchor = topic,
  //    previous = cursor -> next) -- the directed "what the stream actually
  //    says after the subject, in the subject's own contexts" signal. A
  //    first variant committed the globally strongest (P -> V) edge pairs
  //    from the topic's range; those turned out to be internal collocations
  //    of phrases merely NEAR the subject ("George Mason", "modeling
  //    agency"), not claims ABOUT it, so the chain starts at the subject
  //    instead. Content units are committed; closed-class/illegal units are
  //    traversed without being committed (the frame re-supplies glue at
  //    junctions), at most kRelationCommitGlueRun in a row.
  std::uint32_t count = 0u;
  commitment_units[count++] = topic;
  std::uint32_t visited[24u];
  std::uint32_t visited_count = 0u;
  visited[visited_count++] = topic;
  std::uint32_t cursor = topic;
  std::uint32_t glue_run = 0u;
  for (std::uint32_t step = 0u;
       step < 24u && count < kCommitmentCap && visited_count < 24u; ++step) {
    // Continuation choice uses the SAME vitality-normalized strength as the
    // co-occurrence commitment (the resident PMI analogue): raw counts are
    // dominated by function-word paths, which starves the chain of content
    // before the glue budget runs out.
    std::uint32_t best_next = kNoConstruction;
    unsigned long long best_strength = 0ull;
    for (std::uint32_t edge = begin; edge < end; ++edge) {
      if (transitions[edge].previous != cursor) continue;
      const std::uint32_t c = transition_counts[edge];
      if (c < kRelationCommitMinCount) continue;
      const std::uint32_t next = transitions[edge].next;
      if (next == cursor) continue;
      const unsigned long long strength =
          (static_cast<unsigned long long>(c) << 12u) /
          (static_cast<unsigned long long>(unit_vitality[next]) + 16ull);
      if (strength <= best_strength) continue;
      bool seen = false;
      for (std::uint32_t k = 0u; k < visited_count; ++k)
        seen |= visited[k] == next;
      if (seen) continue;
      best_strength = strength;
      best_next = next;
    }
    if (best_next == kNoConstruction) break;
    visited[visited_count++] = best_next;
    cursor = best_next;
    if (commitment_filler_legal(best_next, closed_class_mask, unit_lengths,
                                unit_content, unit_words, boundary_mask,
                                filler_terminal_mask) &&
        !commitment_near_duplicate(best_next, commitment_units, count,
                                   unit_lengths, unit_content, unit_words)) {
      commitment_units[count++] = best_next;
      glue_run = 0u;
    } else if (++glue_run > kRelationCommitGlueRun) {
      break;
    }
  }
  // A chain of fewer than two continuations is not a claim; fall back.
  if (count < 3u) return;
  commitment_meta[0] = count;
  commitment_meta[1] = 0u;
}

// ---------------------------------------------------------------------------
// RELATIONAL-TRIPLE CHANNEL: a resident store of TYPED TRIPLES (A, K, B)
// learned from the assimilation unit stream itself. beaf897fa established
// that every directional resident signal (conditioned transitions, directed
// n-grams) is DIRECTED SURFACE ADJACENCY -- ranking or chaining it yields
// collocations, never predication -- so answering needs a channel whose
// entries TYPE a relation. The type is not authored: it is the middle unit
// identity itself. The learned patterns are [content A] [one content K]
// [content B] and [content A] [1..kRelationTripleMaxGlue glue K] [content B]
// within one episode segment. The first admits predicates such as an
// experience-grown action/relation word only when the current cue activates
// that same K; the second retains copular/appositive/prepositional bridges.
// No parser, POS label, or byte-value rule participates. Each element is a
// SINGLE resident unit -- the store
// carries learned structure, never a corpus span. Resident (device hash
// table keyed by the whole triple, count-weighted), mutable (counts
// accumulate across assimilation calls), lesionable via
// BCC32_LESION_RELATION_TRIPLE (learning and commitment are both skipped
// -> exact prior commitment behavior).
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kRelationTripleHashCap = 1u << 21u;  // power of two
// A relation role records that a subject repeatedly participates in the same
// connective pattern, independent of the particular object word.  This is
// learned resident organization, not a per-word meaning table: it is the
// minimal index needed to let two subjects share an argument structure while
// their observed values remain different.
inline constexpr std::uint32_t kRelationRoleHashCap = 1u << 20u;
inline constexpr std::uint32_t kRelationRoleProbeCap = 32u;
inline constexpr std::uint32_t kRelationRoleSeedCount = 1u;
inline constexpr std::uint32_t kRelationTripleMaxGlue = 2u;
inline constexpr std::uint32_t kRelationTripleContentArgumentSpan = 6u;
inline constexpr std::uint32_t kRelationTripleProbeCap = 32u;
inline constexpr std::uint32_t kRelationTripleCandidateCap = 512u;
inline constexpr std::uint32_t kRelationTripleMaxClauses = 3u;
inline constexpr std::uint32_t kRelationSurfaceEvidenceCap = kCommitmentCap;
inline constexpr std::uint32_t kRelationTripleMinCount = 2u;
// A single observation is valid resident evidence, but it is not sufficient
// to speak by itself. Sparse observations may seed category comparison; the
// analogical reader still requires two shared contexts before it composes a
// novel relation.
inline constexpr std::uint32_t kRelationTripleSeedCount = 1u;
// Resolution of the question-time substitution probe's support histogram:
// bin i holds probes whose substituted triple carries exactly count i, and
// the last bin holds every probe at or above that count.
inline constexpr std::uint32_t kRelationProbeSupportBins = 8u;
// Bounded exact relation events preserve the provenance that aggregate
// triples lose. The ring stores extracted relation identities, never raw
// sentence or query surfaces.
inline constexpr std::uint32_t kWitnessedRelationEventCap = 1u << 16u;

// Physical chronology for retained local events. One subject position can
// yield several valid later arguments, so subject position alone is not an
// event identity: it makes the episode spine tie at its first branch. The
// extractor ordinal supplies the missing temporal coordinate without naming
// any linguistic role or content.
[[nodiscard]] __host__ __device__ inline std::uint64_t
relation_event_evidence_revision(std::uint64_t contact_revision,
                                 std::uint32_t subject_position,
                                 std::uint32_t ordinal) {
  const std::uint64_t local =
      (static_cast<std::uint64_t>(subject_position) + 1u) *
          (kRelationTripleContentArgumentSpan + 1u) +
      ordinal;
  return (contact_revision << 32u) | local;
}
// A connective TYPE whose triple mass is >= 1/2^kRelationTripleMirrorShift
// mirror-attested (both (A,K,B) and (B,K,A) seen) behaves like symmetric
// coordination, not predication -- a LEARNED per-type directionality
// statistic, no connective word is ever named.
inline constexpr std::uint32_t kRelationTripleMirrorShift = 2u;
inline constexpr std::uint32_t kNoTripleUnit = 0xffffffffu;
inline constexpr std::uint32_t kPendingTripleUnit = 0xfffffffeu;
inline constexpr std::uint32_t kQonsetQuestionByte = 63u;
inline constexpr std::uint32_t kQonsetTopicFloor = 1u;
inline constexpr std::uint32_t kRelationFieldCount = 4u;

struct RelationTriple {
  std::uint32_t subject;
  std::uint32_t connective;   // relation TYPE: first glue unit of the bridge
  std::uint32_t connective2;  // second glue unit of the bridge or kNoTripleUnit
  std::uint32_t value;
};

struct RelationRole {
  std::uint32_t subject = kNoTripleUnit;
  std::uint32_t connective = kNoTripleUnit;
  std::uint32_t connective2 = kNoTripleUnit;
};

// One exact extraction event before hash-table aggregation.  The resident
// triple store intentionally merges equal A-K-B observations; downstream
// organs that retain episode-specific evidence must instead consume this
// lossless event, including the physical positions and complete segment
// extent from which it arose.  No field carries a semantic role name.
struct RelationTripleEvent {
  RelationTriple triple{};
  std::uint32_t subject_position = 0u;
  std::uint32_t connective_begin = 0u;
  std::uint32_t connective_end = 0u;
  std::uint32_t value_position = 0u;
  std::uint32_t segment_begin = 0u;
  std::uint32_t segment_end = 0u;
  std::uint32_t valid = 0u;
};

struct WitnessedRelationEvent {
  RelationTriple triple{};
  std::uint64_t evidence_revision = 0u;
  std::uint32_t segment_begin = 0u;
  std::uint32_t opener = kNoTripleUnit;
  std::uint32_t terminal = kNoTripleUnit;
  std::uint32_t live = 0u;
};

[[nodiscard]] __host__ __device__ inline bool
relation_opener_has_question_evidence(
    std::uint32_t exact_opener,
    const std::uint64_t* qonset_evidence_revision) {
  return qonset_evidence_revision != nullptr &&
         qonset_evidence_revision[exact_opener] != 0u;
}

[[nodiscard]] __device__ inline bool decrement_question_onset(
    std::uint32_t* count) {
  std::uint32_t observed = atomicAdd(count, 0u);
  while (observed != 0u) {
    const std::uint32_t prior = atomicCAS(count, observed, observed - 1u);
    if (prior == observed) return true;
    observed = prior;
  }
  return false;
}

[[nodiscard]] __device__ inline std::uint32_t relation_triple_hash(
    std::uint32_t subject, std::uint32_t k1, std::uint32_t k2,
    std::uint32_t value) {
  std::uint32_t hash = construction_mix(subject ^ 0x9e3779b9u);
  hash = construction_mix(hash ^ k1);
  hash = construction_mix(hash ^ (k2 + 0x85ebca6bu));
  hash = construction_mix(hash ^ (value + 0x27d4eb2fu));
  return hash & (kRelationTripleHashCap - 1u);
}

[[nodiscard]] __device__ inline std::uint32_t relation_role_hash(
    std::uint32_t subject, std::uint32_t connective,
    std::uint32_t connective2) {
  std::uint32_t hash = construction_mix(subject ^ 0x243f6a88u);
  hash = construction_mix(hash ^ connective);
  hash = construction_mix(hash ^ (connective2 + 0x85ebca6bu));
  return hash & (kRelationRoleHashCap - 1u);
}

__device__ inline void insert_relation_role(
    RelationRole* table, std::uint32_t* table_counts,
    std::uint32_t subject, std::uint32_t connective,
    std::uint32_t connective2, std::uint32_t* drop_count = nullptr) {
  if (table == nullptr || table_counts == nullptr) return;
  std::uint32_t slot =
      relation_role_hash(subject, connective, connective2);
  for (std::uint32_t probe = 0u; probe < kRelationRoleProbeCap; ++probe) {
    std::uint32_t prior =
        atomicCAS(&table[slot].subject, kNoTripleUnit, kPendingTripleUnit);
    if (prior == kNoTripleUnit) {
      table[slot].connective = connective;
      table[slot].connective2 = connective2;
      atomicExch(table_counts + slot, 1u);
      __threadfence();
      atomicExch(&table[slot].subject, subject);
      return;
    }
    while (prior == kPendingTripleUnit)
      prior = atomicAdd(&table[slot].subject, 0u);
    const std::uint32_t published = prior;
    if (published == subject && table[slot].connective == connective &&
        table[slot].connective2 == connective2) {
      atomicAdd(table_counts + slot, 1u);
      return;
    }
    slot = (slot + 1u) & (kRelationRoleHashCap - 1u);
  }
  if (drop_count != nullptr) atomicAdd(drop_count, 1u);
}

[[nodiscard]] __device__ inline std::uint32_t relation_role_lookup(
    const RelationRole* table, const std::uint32_t* table_counts,
    std::uint32_t subject, std::uint32_t connective,
    std::uint32_t connective2) {
  if (table == nullptr || table_counts == nullptr) return 0u;
  std::uint32_t slot =
      relation_role_hash(subject, connective, connective2);
  for (std::uint32_t probe = 0u; probe < kRelationRoleProbeCap; ++probe) {
    if (table[slot].subject == kNoTripleUnit) return 0u;
    if (table[slot].subject == subject &&
        table[slot].connective == connective &&
        table[slot].connective2 == connective2)
      return table_counts[slot];
    slot = (slot + 1u) & (kRelationRoleHashCap - 1u);
  }
  return 0u;
}

// Admission census used by the adult before it allocates question-time
// synthesis work. It measures resident role formation only; it cannot choose
// a topic, answer, or output surface.
static __global__ void count_relation_role_evidence_kernel(
    const std::uint32_t* role_counts, std::uint32_t* occupied) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kRelationRoleHashCap ||
      role_counts[slot] < kRelationRoleSeedCount)
    return;
  atomicAdd(occupied, 1u);
}

__device__ inline void insert_relation_triple(
    RelationTriple* table, std::uint32_t* table_counts,
    std::uint64_t* table_evidence_revision, std::uint32_t subject,
    std::uint32_t connective, std::uint32_t connective2,
    std::uint32_t value, std::uint64_t evidence_revision,
    std::uint32_t* attempt_count = nullptr,
    std::uint32_t* drop_count = nullptr) {
  // Every attempt either claims an empty slot (count 1), finds its exact key
  // (count +1), or exhausts the probe window and is dropped. Counting both
  // ends closes attempted == sum(counts) + drops over the whole store.
  if (attempt_count != nullptr) atomicAdd(attempt_count, 1u);
  std::uint32_t slot =
      relation_triple_hash(subject, connective, connective2, value);
  for (std::uint32_t probe = 0u; probe < kRelationTripleProbeCap; ++probe) {
    std::uint32_t prior =
        atomicCAS(&table[slot].subject, kNoTripleUnit, kPendingTripleUnit);
    if (prior == kNoTripleUnit) {
      table[slot].connective = connective;
      table[slot].connective2 = connective2;
      table[slot].value = value;
      if (table_evidence_revision != nullptr)
        atomicExch(
            reinterpret_cast<unsigned long long*>(table_evidence_revision + slot),
            static_cast<unsigned long long>(evidence_revision));
      atomicExch(table_counts + slot, 1u);
      __threadfence();
      atomicExch(&table[slot].subject, subject);
      return;
    }
    while (prior == kPendingTripleUnit)
      prior = atomicAdd(&table[slot].subject, 0u);
    if (prior == subject && table[slot].connective == connective &&
        table[slot].connective2 == connective2 &&
        table[slot].value == value) {
      atomicAdd(table_counts + slot, 1u);
      if (table_evidence_revision != nullptr)
        atomicMax(
            reinterpret_cast<unsigned long long*>(table_evidence_revision + slot),
            static_cast<unsigned long long>(evidence_revision));
      return;
    }
    slot = (slot + 1u) & (kRelationTripleHashCap - 1u);
  }
  // The probe window is exhausted: this triple never reaches the store and
  // carries no count. It is lost mass, and only this counter records it.
  if (drop_count != nullptr) atomicAdd(drop_count, 1u);
}

// Return one event ordinal for a stream position.  Both aggregation and
// provenance export call this single extractor so the two resident views
// cannot silently acquire different notions of a relation.  Content
// predicates may bind several later arguments; glue bridges yield ordinal 0.
[[nodiscard]] __device__ inline bool extract_relation_triple_event(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* closed_class_mask,
    std::uint32_t position, std::uint32_t ordinal,
    RelationTripleEvent* event) {
  if (event == nullptr || position >= sequence_count)
    return false;
  *event = RelationTripleEvent{};
  const std::uint32_t segment = segment_ids[position];
  const std::uint32_t a = sequence[position];
  if (closed_class_mask[a] != 0u)
    return false;

  std::uint32_t segment_begin = position;
  while (segment_begin != 0u && segment_ids[segment_begin - 1u] == segment)
    --segment_begin;
  std::uint32_t segment_end = position + 1u;
  while (segment_end < sequence_count && segment_ids[segment_end] == segment)
    ++segment_end;

  const std::uint32_t next = position + 1u;
  // A content bridge may cross a short learned glue run before its argument.
  // This admits the ordinary observed shape A P [glue] B alongside adjacent
  // A P B without naming a lexical class: P and B are whatever the current
  // partition has learned as content, and the bounded scan remains inside the
  // same sensed segment.  The prior adjacent-only rule lost facts such as
  // "daxen moved the copper ..." before they could reach the resident graph.
  if (next < sequence_count && segment_ids[next] == segment &&
      closed_class_mask[sequence[next]] == 0u) {
    const std::uint32_t limit =
        position + kRelationTripleContentArgumentSpan + 1u < sequence_count
            ? position + kRelationTripleContentArgumentSpan + 1u
            : sequence_count;
    std::uint32_t candidate = 0u;
    std::uint32_t first_argument = kNoTripleUnit;
    std::uint32_t first_position = sequence_count;
    std::uint32_t second_argument = kNoTripleUnit;
    std::uint32_t second_position = sequence_count;
    for (std::uint32_t at = next + 1u; at < limit; ++at) {
      if (segment_ids[at] != segment)
        break;
      const std::uint32_t b = sequence[at];
      if (closed_class_mask[b] != 0u || b == a || b == sequence[next])
        continue;
      if (first_argument == kNoTripleUnit) {
        first_argument = b;
        first_position = at;
      } else if (second_argument == kNoTripleUnit) {
        second_argument = b;
        second_position = at;
      }
      if (candidate++ != ordinal)
        continue;
      event->triple = {a, sequence[next], kNoTripleUnit, b};
      event->subject_position = position;
      event->connective_begin = next;
      event->connective_end = next + 1u;
      event->value_position = at;
      event->segment_begin = segment_begin;
      event->segment_end = segment_end;
      event->valid = 1u;
      return true;
    }
    // Preserve the observed compound argument as one four-coordinate event in
    // addition to the established pairwise views. This is the same local
    // content rule at one greater order; no lexical class or authored role
    // decides which units participate.
    if (ordinal == candidate && first_argument != kNoTripleUnit &&
        second_argument != kNoTripleUnit) {
      event->triple = {a, sequence[next], first_argument, second_argument};
      event->subject_position = position;
      event->connective_begin = next;
      event->connective_end = first_position + 1u;
      event->value_position = second_position;
      event->segment_begin = segment_begin;
      event->segment_end = segment_end;
      event->valid = 1u;
      return true;
    }
    return false;
  }

  if (ordinal > 1u)
    return false;
  std::uint32_t k1 = kNoTripleUnit;
  std::uint32_t k2 = kNoTripleUnit;
  std::uint32_t run = 0u;
  std::uint32_t at = next;
  while (at < sequence_count && segment_ids[at] == segment &&
         run < kRelationTripleMaxGlue) {
    const std::uint32_t unit = sequence[at];
    if (closed_class_mask[unit] == 0u)
      break;
    if (run == 0u)
      k1 = unit;
    else
      k2 = unit;
    ++run;
    ++at;
  }
  if (run == 0u || at >= sequence_count || segment_ids[at] != segment)
    return false;
  const std::uint32_t b = sequence[at];
  if (closed_class_mask[b] != 0u || b == a)
    return false;
  if (ordinal == 0u) {
    event->triple = {a, k1, k2, b};
    event->value_position = at;
  } else {
    const std::uint32_t compound = at + 1u;
    if (compound >= sequence_count || segment_ids[compound] != segment ||
        closed_class_mask[sequence[compound]] != 0u ||
        sequence[compound] == a || sequence[compound] == b)
      return false;
    event->triple = {a, k1, b, sequence[compound]};
    event->value_position = compound;
  }
  event->subject_position = position;
  event->connective_begin = next;
  event->connective_end = next + run;
  event->segment_begin = segment_begin;
  event->segment_end = segment_end;
  event->valid = 1u;
  return true;
}

constexpr std::uint32_t kContactQuestionBoundary = 1u;
constexpr std::uint32_t kContactDeclarativeBoundary = 2u;

[[nodiscard]] __device__ inline std::uint32_t
learned_segment_boundary_modality(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t segment_begin, std::uint32_t segment_end,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* question_terminal_mass,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    const std::uint32_t* boundary_mask, std::uint32_t unit_count) {
  if (sequence == nullptr || segment_begin >= segment_end ||
      segment_end > sequence_count || unit_lengths == nullptr ||
      unit_content == nullptr || unit_words == 0u ||
      question_terminal_mass == nullptr || closure_bytes == nullptr ||
      closure_count == 0u || boundary_mask == nullptr)
    return 0u;
  const std::uint32_t unit = sequence[segment_end - 1u];
  if (unit >= unit_count) return 0u;
  bool learned_closure = false;
  for (std::uint32_t byte = 0u; byte < unit_lengths[unit]; ++byte) {
    const std::uint32_t value =
        construction_unit_byte(unit_content, unit_words, unit, byte);
    bool is_closure = false;
    for (std::uint32_t closure = 0u; closure < closure_count; ++closure) {
      is_closure |= value == closure_bytes[closure];
    }
    if (!is_closure) continue;
    bool suffix_is_boundary = true;
    for (std::uint32_t suffix = byte + 1u;
         suffix < unit_lengths[unit]; ++suffix) {
      const std::uint32_t suffix_value =
          construction_unit_byte(unit_content, unit_words, unit, suffix);
      suffix_is_boundary &=
          suffix_value <= 0xffu && boundary_mask[suffix_value] != 0u;
    }
    learned_closure |= suffix_is_boundary;
  }
  if (!learned_closure) return 0u;
  return question_terminal_mass[unit] != 0u
             ? kContactQuestionBoundary
             : kContactDeclarativeBoundary;
}

static __global__ void detect_learned_segment_boundary_modality_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t segment_begin, std::uint32_t segment_end,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, const std::uint32_t* question_terminal_mass,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    const std::uint32_t* boundary_mask, std::uint32_t unit_count,
    std::uint32_t* modality) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || modality == nullptr) return;
  modality[0] = learned_segment_boundary_modality(
      sequence, sequence_count, segment_begin, segment_end, unit_lengths,
      unit_content, unit_words, question_terminal_mass, closure_bytes,
      closure_count, boundary_mask, unit_count);
}

// LEARN: one thread per stream position. If the position holds a content
// unit A, record either A-contentK-contentB or A-glueK-contentB inside the
// same episode segment. Content predicates are useful only when cue-typed at
// retrieval; untyped retrieval remains restricted to the learned glue bridge.
// Record (A, K, B) with count into the resident
// hash table. Insertion is claim-by-CAS on the subject field; an identical
// key racing the claim's field writes can at worst duplicate a slot, which
// only splits a count (statistics noise, same tolerance as the suffix-class
// adjacency store).
static __global__ void learn_relation_triples_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* closed_class_mask,
    const std::uint32_t* role_canon,
    const std::uint64_t* qonset_evidence_revision,
    RelationTriple* table, std::uint32_t* table_counts,
    std::uint64_t* table_evidence_revision,
    RelationRole* relation_roles, std::uint32_t* relation_role_counts,
    const std::uint64_t* contact_evidence_revision,
    WitnessedRelationEvent* witnessed_events,
    std::uint32_t* witnessed_event_cursor,
    std::uint32_t* witnessed_event_constructions,
    std::uint32_t* witnessed_event_surface_units,
    std::uint32_t* witnessed_event_surface_counts,
    std::uint32_t* contact_event_count = nullptr,
    const std::uint32_t* contact_learned_request = nullptr,
    const std::uint32_t* question_terminal_mass = nullptr,
    const std::uint32_t* unit_lengths = nullptr,
    const std::uint32_t* unit_content = nullptr,
    std::uint32_t unit_words = 0u,
    const std::uint32_t* closure_bytes = nullptr,
    std::uint32_t closure_count = 0u,
    const std::uint32_t* boundary_mask = nullptr,
    std::uint32_t unit_count = kNoTripleUnit,
    std::uint32_t* relation_triple_attempted = nullptr,
    std::uint32_t* relation_triple_drops = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sequence_count) return;
  for (std::uint32_t ordinal = 0u;; ++ordinal) {
    RelationTripleEvent event{};
    if (!extract_relation_triple_event(sequence, sequence_count, segment_ids,
                                       closed_class_mask, i, ordinal, &event))
      break;
    const std::uint32_t exact_opener = sequence[event.segment_begin];
    std::uint32_t opener = exact_opener;
    if (role_canon != nullptr && role_canon[opener] < kNoTripleUnit)
      opener = role_canon[opener];
    // Role similarity may transfer a learned construction, but it cannot
    // rewrite the observed contact's modality. Otherwise a declarative opener
    // sharing a role class with an interrogative opener is discarded before
    // it can become a witnessed fact.
    const bool learned_request =
        contact_learned_request != nullptr &&
        contact_learned_request[0] != 0u;
    const std::uint32_t boundary_modality =
        learned_segment_boundary_modality(
            sequence, sequence_count, event.segment_begin, event.segment_end,
            unit_lengths, unit_content, unit_words, question_terminal_mass,
            closure_bytes, closure_count, boundary_mask, unit_count);
    (void)exact_opener;
    (void)qonset_evidence_revision;
    // Explicitly witnessed question contacts supervise the learned onset
    // field and do not become facts. A physical declarative boundary defeats
    // an opener-only false positive. With no terminal boundary, the resident
    // learned request prediction remains authoritative, enabling questions
    // whose surface omits '?'.
    if ((boundary_modality & kContactQuestionBoundary) != 0u ||
        (learned_request &&
         (boundary_modality & kContactDeclarativeBoundary) == 0u))
      continue;
    if (contact_event_count != nullptr)
      atomicAdd(contact_event_count, 1u);
    insert_relation_triple(
        table, table_counts, table_evidence_revision, event.triple.subject,
        event.triple.connective, event.triple.connective2, event.triple.value,
        relation_event_evidence_revision(
            contact_evidence_revision == nullptr ? 0u
                                                 : contact_evidence_revision[0],
            event.subject_position, ordinal),
        relation_triple_attempted, relation_triple_drops);
    insert_relation_role(relation_roles, relation_role_counts,
                         event.triple.subject, event.triple.connective,
                         event.triple.connective2);
    if (witnessed_events != nullptr &&
        witnessed_event_cursor != nullptr) {
      const std::uint32_t event_slot =
          atomicAdd(witnessed_event_cursor, 1u) &
          (kWitnessedRelationEventCap - 1u);
      WitnessedRelationEvent retained{};
      retained.triple = event.triple;
      retained.evidence_revision = relation_event_evidence_revision(
          contact_evidence_revision == nullptr ? 0u
                                               : contact_evidence_revision[0],
          event.subject_position, ordinal);
      retained.segment_begin = event.segment_begin;
      retained.opener = opener;
      retained.terminal =
          event.segment_end == 0u ? kNoTripleUnit
                                  : sequence[event.segment_end - 1u];
      retained.live = 1u;
      if (witnessed_event_constructions != nullptr)
        witnessed_event_constructions[event_slot] = kNoConstruction;
      witnessed_events[event_slot] = retained;
      if (witnessed_event_surface_units != nullptr &&
          witnessed_event_surface_counts != nullptr) {
        const std::uint32_t surface_end =
            event.value_position + 2u == event.segment_end
                ? event.segment_end
                : event.value_position + 1u;
        const std::uint32_t surface_count =
            surface_end - event.subject_position;
        witnessed_event_surface_counts[event_slot] =
            surface_count <= kConstructionMaxTokens ? surface_count : 0u;
        if (surface_count <= kConstructionMaxTokens) {
          for (std::uint32_t surface = 0u; surface < surface_count; ++surface)
            witnessed_event_surface_units[
                event_slot * kConstructionMaxTokens + surface] =
                sequence[event.subject_position + surface];
        }
      }
    }
  }
}

[[nodiscard]] __device__ inline std::uint32_t canonical_relation_unit(
    std::uint32_t unit, const std::uint32_t* role_canon) {
  if (unit == kNoTripleUnit || role_canon == nullptr)
    return unit;
  return role_canon[unit] < kNoTripleUnit ? role_canon[unit] : unit;
}

[[nodiscard]] __host__ __device__ inline std::size_t
question_answer_surface_index(std::uint32_t opener, std::uint32_t field,
                              std::uint32_t arity) {
  return (static_cast<std::size_t>(opener) * kRelationFieldCount + field) *
             kQuestionAnswerArityCount +
         arity;
}

[[nodiscard]] __device__ inline bool relation_unit_has_learned_closure(
    std::uint32_t unit, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    const std::uint32_t* boundary_mask, std::uint32_t unit_count) {
  if (unit >= unit_count || unit_lengths == nullptr || unit_content == nullptr ||
      unit_words == 0u || closure_bytes == nullptr || closure_count == 0u ||
      boundary_mask == nullptr)
    return false;
  for (std::uint32_t byte = 0u; byte < unit_lengths[unit]; ++byte) {
    const std::uint32_t value =
        construction_unit_byte(unit_content, unit_words, unit, byte);
    bool learned_closure = false;
    for (std::uint32_t closure = 0u; closure < closure_count; ++closure)
      learned_closure |= value == closure_bytes[closure];
    if (!learned_closure) continue;
    bool boundary_suffix = true;
    for (std::uint32_t suffix = byte + 1u; suffix < unit_lengths[unit];
         ++suffix) {
      const std::uint32_t suffix_value =
          construction_unit_byte(unit_content, unit_words, unit, suffix);
      boundary_suffix &=
          suffix_value <= 0xffu && boundary_mask[suffix_value] != 0u;
    }
    if (boundary_suffix) return true;
  }
  return false;
}

// Learn which opaque coordinate of an observed relation a recurrent
// interrogative construction leaves unresolved. The teacher is ordinary
// contact order: a learned question segment immediately followed by a
// non-question segment. The recurrent onset is the learned construction
// marker, not part of the relation body. Candidate relation bodies must agree
// in three coordinates and differ in exactly one. Equal support for multiple
// coordinates abstains, so no lexical item or authored semantic role decides
// what an opener requests.
//
// RETIRED FROM PRODUCTION (2026-08-14, 0X1-156): the differences==1u
// precondition above was measured to fire ZERO times over 12,000+ real
// question/answer pairs across four corpora, including Socratic dialogue
// chosen to maximize question/answer parallelism -- mass sits entirely at
// 3-4 differing coordinates, never exactly one. The production launch (in
// bcc32_cuda_adult_construction_learning.inl) is bypassed; requested_field is
// now derived live, per query, from the resident relation store by the
// causal-compatibility vote in
// bcc32_resident_relation_answer_side_causal_compatibility.cuh, wired into
// form_witnessed_relation_plan_kernel. This kernel definition and its own
// synthetic-fixture contract (bcc32_cuda_learned_question_gap_contract.cu)
// remain: the differences==1u rule is not internally broken, only never
// satisfied by real discourse, so nothing here needed to change to prove
// that separately-measured fact.
static __global__ void learn_question_gap_fields_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* closed_class_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* role_canon,
    const std::uint64_t* qonset_evidence_revision,
    std::uint32_t* question_gap_field_support,
    std::uint32_t* question_answer_construction,
    std::uint32_t* question_answer_construction_support,
    std::uint32_t* question_answer_slot_mapping,
    std::uint32_t* construction_tokens, std::uint32_t* construction_lengths,
    std::uint32_t* construction_slot_counts,
    std::uint32_t* construction_supports,
    std::uint32_t* construction_slot_units,
    std::uint32_t* construction_slot_masses,
    std::uint32_t* construction_slot_totals,
    std::uint32_t* construction_slot_overflow,
    std::uint32_t* construction_hash_slots,
    std::uint32_t* construction_count,
    const std::uint64_t* evidence_revision,
    std::uint64_t* construction_evidence_revision,
    std::uint64_t* construction_origin_revision,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    const std::uint32_t* boundary_mask,
    std::uint32_t unit_count, std::uint32_t allocation_capacity,
    // Diagnostic only, optional: an 8-bin histogram of the SAME `differences`
    // value the `== 1u` vote test below reads, counted once per evaluated
    // question/answer triple pair, whether or not the vote fires. Never read
    // by the organism, never checkpointed. nullptr disables it.
    std::uint32_t* gap_vote_histogram = nullptr,
    // Diagnostic only, optional: a kRelationFieldCount-bin histogram counting,
    // for every evaluated question/answer triple pair, WHICH coordinates the
    // SAME comparison below finds unequal -- one increment per differing
    // coordinate, including pairs whose `differences` is not 1. Never read by
    // the organism, never checkpointed. nullptr disables it.
    std::uint32_t* gap_coordinate_histogram = nullptr) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || sequence == nullptr ||
      segment_ids == nullptr || closed_class_mask == nullptr ||
      unit_lengths == nullptr || unit_content == nullptr ||
      qonset_evidence_revision == nullptr ||
      question_gap_field_support == nullptr ||
      question_answer_construction == nullptr ||
      question_answer_construction_support == nullptr || roles == nullptr ||
      question_answer_slot_mapping == nullptr ||
      construction_tokens == nullptr || construction_lengths == nullptr ||
      construction_slot_counts == nullptr || construction_supports == nullptr ||
      construction_slot_units == nullptr ||
      construction_slot_masses == nullptr ||
      construction_slot_totals == nullptr ||
      construction_slot_overflow == nullptr ||
      construction_hash_slots == nullptr || construction_count == nullptr ||
      closure_bytes == nullptr || closure_count == 0u ||
      boundary_mask == nullptr ||
      sequence_count < 4u)
    return;
  for (std::uint32_t question_begin = 0u;
       question_begin + 1u < sequence_count; ++question_begin) {
    const std::uint32_t opener = sequence[question_begin];
    if (opener >= unit_count || qonset_evidence_revision[opener] == 0u)
      continue;
    if (question_begin != 0u &&
        !relation_unit_has_learned_closure(
            sequence[question_begin - 1u], unit_lengths, unit_content,
            unit_words, closure_bytes, closure_count, boundary_mask,
            unit_count))
      continue;
    std::uint32_t question_end = question_begin;
    while (question_end < sequence_count &&
           !relation_unit_has_learned_closure(
               sequence[question_end], unit_lengths, unit_content, unit_words,
               closure_bytes, closure_count, boundary_mask, unit_count))
      ++question_end;
    if (question_end >= sequence_count) continue;
    const std::uint32_t answer_begin = question_end + 1u;
    if (answer_begin >= sequence_count) continue;
    std::uint32_t answer_end = answer_begin;
    while (answer_end < sequence_count &&
           !relation_unit_has_learned_closure(
               sequence[answer_end], unit_lengths, unit_content, unit_words,
               closure_bytes, closure_count, boundary_mask, unit_count))
      ++answer_end;
    if (answer_end < sequence_count) ++answer_end;

    const std::uint32_t answer_opener = sequence[answer_begin];
    const bool answer_is_question =
        answer_opener < unit_count &&
        qonset_evidence_revision[answer_opener] != 0u;
    if (!answer_is_question) {
      std::uint32_t votes[kRelationFieldCount] = {};
      RelationTripleEvent answer_events[kRelationFieldCount]{};
      bool answer_event_set[kRelationFieldCount] = {};
      bool answer_event_ambiguous[kRelationFieldCount] = {};
      for (std::uint32_t qpos = question_begin + 1u;
           qpos <= question_end; ++qpos) {
        for (std::uint32_t qordinal = 0u;; ++qordinal) {
          RelationTripleEvent question{};
          if (!extract_relation_triple_event(
                  sequence, sequence_count, segment_ids, closed_class_mask,
                  qpos, qordinal, &question))
            break;
          if (question.value_position > question_end) continue;
          for (std::uint32_t apos = answer_begin; apos < answer_end; ++apos) {
            for (std::uint32_t aordinal = 0u;; ++aordinal) {
              RelationTripleEvent answer{};
              if (!extract_relation_triple_event(
                      sequence, sequence_count, segment_ids, closed_class_mask,
                      apos, aordinal, &answer))
                break;
              if (answer.value_position >= answer_end) continue;
              const std::uint32_t qfields[kRelationFieldCount] = {
                  question.triple.subject, question.triple.connective,
                  question.triple.connective2, question.triple.value};
              const std::uint32_t afields[kRelationFieldCount] = {
                  answer.triple.subject, answer.triple.connective,
                  answer.triple.connective2, answer.triple.value};
              std::uint32_t changed = kRelationFieldCount;
              std::uint32_t differences = 0u;
              for (std::uint32_t field = 0u;
                   field < kRelationFieldCount; ++field) {
                const bool equal =
                    canonical_relation_unit(qfields[field], role_canon) ==
                    canonical_relation_unit(afields[field], role_canon);
                if (!equal) {
                  changed = field;
                  ++differences;
                  if (gap_coordinate_histogram != nullptr)
                    atomicAdd(&gap_coordinate_histogram[field], 1u);
                }
              }
              if (gap_vote_histogram != nullptr)
                atomicAdd(&gap_vote_histogram[min(differences, 7u)], 1u);
              if (differences == 1u) {
                ++votes[changed];
                if (!answer_event_set[changed]) {
                  answer_events[changed] = answer;
                  answer_event_set[changed] = true;
                } else {
                  const auto prior = answer_events[changed].triple;
                  answer_event_ambiguous[changed] |=
                      prior.subject != answer.triple.subject ||
                      prior.connective != answer.triple.connective ||
                      prior.connective2 != answer.triple.connective2 ||
                      prior.value != answer.triple.value;
                }
              }
            }
          }
        }
      }
      std::uint32_t winner = kRelationFieldCount;
      std::uint32_t winner_votes = 0u;
      bool ambiguous = false;
      for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
        if (votes[field] > winner_votes) {
          winner = field;
          winner_votes = votes[field];
          ambiguous = false;
        } else if (votes[field] != 0u && votes[field] == winner_votes) {
          ambiguous = true;
        }
      }
      if (winner < kRelationFieldCount && winner_votes != 0u && !ambiguous) {
        atomicAdd(&question_gap_field_support[
                      opener * kRelationFieldCount + winner],
                  1u);
        std::uint32_t learned_construction = kNoConstruction;
        const std::uint64_t answer_revision =
            evidence_revision == nullptr
                ? 0u
                : ((evidence_revision[0] << 32u) |
                   (static_cast<std::uint64_t>(answer_begin) + 1u));
        learn_one_construction(
            sequence, answer_begin, answer_end, roles, closed_class_mask,
            unit_lengths, construction_tokens, construction_lengths,
            construction_slot_counts, construction_supports,
            construction_slot_units, construction_slot_masses,
            construction_slot_totals, construction_slot_overflow,
            construction_hash_slots, construction_count, answer_revision,
            construction_evidence_revision, construction_origin_revision,
            false, false,
            &learned_construction, allocation_capacity);
        if (learned_construction != kNoConstruction &&
            answer_event_set[winner] && !answer_event_ambiguous[winner]) {
          const std::uint32_t answer_fields[kRelationFieldCount] = {
              answer_events[winner].triple.subject,
              answer_events[winner].triple.connective,
              answer_events[winner].triple.connective2,
              answer_events[winner].triple.value};
          std::uint32_t packed_mapping = 0u;
          std::uint32_t slot = 0u;
          bool mapping_valid = true;
          for (std::uint32_t position = answer_begin;
               position < answer_end && mapping_valid; ++position) {
            const std::uint32_t unit = sequence[position];
            if (construction_closed_class(unit, closed_class_mask)) continue;
            std::uint32_t matched_field = kRelationFieldCount;
            for (std::uint32_t field = 0u; field < kRelationFieldCount; ++field) {
              if (answer_fields[field] != unit) continue;
              if (matched_field != kRelationFieldCount) {
                mapping_valid = false;
                break;
              }
              matched_field = field;
            }
            if (matched_field == kRelationFieldCount ||
                slot >= kConstructionMaxSlots) {
              mapping_valid = false;
              break;
            }
            packed_mapping |= (matched_field + 1u) << (slot * 3u);
            ++slot;
          }
          mapping_valid &=
              slot == construction_slot_counts[learned_construction];
          if (!mapping_valid) continue;
          const std::size_t mapping =
              question_answer_surface_index(opener, winner, slot);
          const std::uint32_t prior = question_answer_construction[mapping];
          if (prior == kNoConstruction) {
            question_answer_construction[mapping] = learned_construction;
            question_answer_construction_support[mapping] = 1u;
            question_answer_slot_mapping[mapping] = packed_mapping;
          } else if (prior != kAmbiguousConstruction &&
                     question_answer_slot_mapping[mapping] == packed_mapping &&
                     construction_lengths[prior] ==
                         construction_lengths[learned_construction]) {
            bool same_surface = true;
            for (std::uint32_t position = 0u;
                 position < construction_lengths[prior] && same_surface;
                 ++position) {
              const std::uint32_t left =
                  construction_tokens[
                      prior * kConstructionMaxTokens + position];
              const std::uint32_t right =
                  construction_tokens[
                      learned_construction * kConstructionMaxTokens + position];
              same_surface =
                  (token_is_slot(left) && token_is_slot(right)) ||
                  left == right;
            }
            if (same_surface)
              ++question_answer_construction_support[mapping];
            else {
              question_answer_construction[mapping] = kAmbiguousConstruction;
              question_answer_construction_support[mapping] = 0u;
              question_answer_slot_mapping[mapping] = 0u;
            }
          } else {
            question_answer_construction[mapping] = kAmbiguousConstruction;
            question_answer_construction_support[mapping] = 0u;
            question_answer_slot_mapping[mapping] = 0u;
          }
        }
      }
    }
  }
}

// Materialize a minimal surface witness for each relation the organism has
// already discovered from the same resident sequence. Long factual contacts
// routinely contain more than kConstructionMaxSlots content positions. Keep
// the longest suffix ending at the event value that the learned construction
// capacity can actually retain. This preserves local observed order and glue
// without widening the checkpoint schema or turning intervening fillers into
// literals. The event extractor remains the sole authority for what counts as
// a relation; this kernel only gives that learned relation a surface form.
//
// The final unit is not forced literal here.  These are interior relation
// frames, not sentence closures, so forcing a lexical endpoint into the
// skeleton would leak the observed filler into a supposedly abstract slot.
struct RelationConstructionLearningReceipt {
  std::uint32_t attempted_events = 0u;
  std::uint32_t learned_constructions = 0u;
  std::uint32_t missing_constructions = 0u;
  std::uint32_t linked_events = 0u;
};

static __global__ void learn_relation_constructions_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids,
    const resident_roles::MutableStructuralRole* roles,
    const std::uint32_t* closed_class_mask, const std::uint32_t* unit_lengths,
    std::uint32_t* tokens, std::uint32_t* lengths, std::uint32_t* slot_counts,
    std::uint32_t* supports, std::uint32_t* slot_units,
    std::uint32_t* slot_masses, std::uint32_t* slot_totals,
    std::uint32_t* slot_overflow, std::uint32_t* hash_slots,
    std::uint32_t* construction_count, const std::uint64_t* evidence_revision,
    std::uint64_t* construction_evidence_revision,
    std::uint64_t* construction_origin_revision,
    const WitnessedRelationEvent* witnessed_events,
    const std::uint32_t* witnessed_event_cursor,
    std::uint32_t* witnessed_event_constructions,
    std::uint32_t* witnessed_event_surface_units,
    std::uint32_t* witnessed_event_surface_counts,
    std::uint32_t allocation_capacity,
    RelationConstructionLearningReceipt* receipt = nullptr) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= sequence_count) return;
  for (std::uint32_t ordinal = 0u;; ++ordinal) {
    RelationTripleEvent event{};
    if (!extract_relation_triple_event(sequence, sequence_count, segment_ids,
                                       closed_class_mask, position, ordinal,
                                       &event))
      break;
    if (receipt != nullptr) atomicAdd(&receipt->attempted_events, 1u);
    const std::uint64_t event_revision = relation_event_evidence_revision(
        evidence_revision == nullptr ? 0u : evidence_revision[0],
        event.subject_position, ordinal);
    const std::uint32_t event_end =
        event.value_position + 2u == event.segment_end
            ? event.segment_end
            : event.value_position + 1u;
    std::uint32_t learned_construction = kNoConstruction;
    std::uint32_t learned_begin = event.subject_position;
    for (std::uint32_t begin = event.subject_position; begin < event_end &&
         learned_construction == kNoConstruction;
         ++begin) {
      learn_one_construction(
          sequence, begin, event_end, roles, closed_class_mask,
          unit_lengths, tokens, lengths, slot_counts, supports, slot_units,
          slot_masses, slot_totals, slot_overflow, hash_slots,
          construction_count, event_revision, construction_evidence_revision,
          construction_origin_revision, false, false, &learned_construction,
          allocation_capacity);
      if (learned_construction != kNoConstruction) learned_begin = begin;
    }
    if (learned_construction == kNoConstruction) {
      if (receipt != nullptr)
        atomicAdd(&receipt->missing_constructions, 1u);
      continue;
    }
    if (receipt != nullptr)
      atomicAdd(&receipt->learned_constructions, 1u);
    if (
        witnessed_events == nullptr || witnessed_event_cursor == nullptr ||
        witnessed_event_constructions == nullptr)
      continue;
    const std::uint32_t extent =
        min(witnessed_event_cursor[0], kWitnessedRelationEventCap);
    for (std::uint32_t retained = 0u; retained < extent; ++retained) {
      if (witnessed_events[retained].live != 0u &&
          witnessed_events[retained].evidence_revision == event_revision) {
        witnessed_event_constructions[retained] = learned_construction;
        if (receipt != nullptr) atomicAdd(&receipt->linked_events, 1u);
        if (witnessed_event_surface_units != nullptr &&
            witnessed_event_surface_counts != nullptr) {
          const std::uint32_t surface_count =
              event_end - learned_begin;
          witnessed_event_surface_counts[retained] = surface_count;
          for (std::uint32_t surface = 0u; surface < surface_count; ++surface)
            witnessed_event_surface_units[
                retained * kConstructionMaxTokens + surface] =
                sequence[learned_begin + surface];
        }
      }
    }
  }
}

// Export the same extracted events into a bounded transient buffer.  The
// buffer is not a second knowledge store: callers consume it immediately to
// build exact resident organization and retain overflow as an explicit
// capacity receipt. `event_count` records attempted events; only the first
// `event_capacity` records are materialized.
static __global__ void emit_relation_triple_events_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* closed_class_mask,
    RelationTripleEvent* events, std::uint32_t event_capacity,
    std::uint32_t* event_count, std::uint32_t* overflow_count) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= sequence_count || event_count == nullptr ||
      overflow_count == nullptr)
    return;
  for (std::uint32_t ordinal = 0u;; ++ordinal) {
    RelationTripleEvent event{};
    if (!extract_relation_triple_event(sequence, sequence_count, segment_ids,
                                       closed_class_mask, position, ordinal,
                                       &event))
      break;
    const std::uint32_t slot = atomicAdd(event_count, 1u);
    if (events != nullptr && slot < event_capacity)
      events[slot] = event;
    else
      atomicAdd(overflow_count, 1u);
  }
}

// Exact-triple lookup (linear probe; retrieval runs after learning, so an
// empty slot terminates the chain).
[[nodiscard]] __device__ inline std::uint32_t relation_triple_lookup(
    const RelationTriple* table, const std::uint32_t* table_counts,
    std::uint32_t subject, std::uint32_t k1, std::uint32_t k2,
    std::uint32_t value) {
  std::uint32_t slot = relation_triple_hash(subject, k1, k2, value);
  for (std::uint32_t probe = 0u; probe < kRelationTripleProbeCap; ++probe) {
    if (table[slot].subject == kNoTripleUnit) return 0u;
    if (table[slot].subject == subject && table[slot].connective == k1 &&
        table[slot].connective2 == k2 && table[slot].value == value)
      return table_counts[slot];
    slot = (slot + 1u) & (kRelationTripleHashCap - 1u);
  }
  return 0u;
}

#include "bcc32_cuda_resident_construction_composer_relation_diagnostics.inl"
#include "bcc32_cuda_resident_construction_composer_tail.inl"
