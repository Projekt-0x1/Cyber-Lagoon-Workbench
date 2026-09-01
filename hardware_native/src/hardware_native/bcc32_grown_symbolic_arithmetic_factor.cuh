#pragma once

// A bounded, content-neutral resident factor for learning which physical
// source ports an arbitrary form should recruit.  The factor never receives an
// expected sum, a selected source mask, an answer identifier, a route index, or
// a correctness bit.  Source and operator identities are learned from raw
// four-byte forms and represented unary carrier tapes.  A query commits its
// route and motor witnesses before a later raw consequence is available.
//
// This factor deliberately does not implement arithmetic.  It can exchange
// one represented source quantum into an ordinary-F processive inlet and can
// later read the resulting processive bodies, but only the canonical law may
// change those bodies.  The full ordinary-F growth/attachment transaction is
// a separate, still-required causal axis.

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_law.cuh"

namespace substrate::bcc32::grown_symbolic_arithmetic_factor {

inline constexpr std::uint32_t kFormBytes = 4u;
inline constexpr std::uint32_t kMaxSources = 4u;
inline constexpr std::uint32_t kTokensPerSource = 6u;
inline constexpr std::uint32_t kFormSlotCount = 16u;
inline constexpr std::uint32_t kOperatorSlotCount = 8u;
inline constexpr std::uint32_t kRouteModeCount = 4u;
inline constexpr std::uint32_t kMaxRibbonBodies = 8u;
inline constexpr std::uint32_t kJournalDepth = 64u;
// Bits of the grown journal counter. Eight covers 0..255, comfortably above
// kJournalDepth, so the ceiling stays a policy choice rather than a capacity
// accident. See grown_journal_count().
inline constexpr std::uint32_t kJournalCounterBits = 8u;
inline constexpr std::uint32_t kNoSlot = 0xffffffffu;
inline constexpr std::uint32_t kNoSource = 0xffffffffu;
inline constexpr SiteWord kFactorMarkerValue = 0x5a71b01cu;

enum class Command : std::uint32_t {
  idle = 0u,
  ground_homeostasis = 1u,
  bind_quantity = 2u,
  prepare_query = 3u,
  route_one = 4u,
  finish_result = 5u,
  settle_consequence = 6u,
  age = 7u,
};

enum GlobalField : std::uint32_t {
  kFactorMarker = 0u,
  kJournalCount,
  kExplorerPhase,
  kAliasPhase,
  kHomeostaticReafference,
  kHomeostaticInternal,
  kEligibility,
  kEligibilityAge,
  kActiveOperatorSlot,
  kSelectedMode,
  kSelectedSourceMask,
  kRouteCursor,
  kMotorWitness,
  kEmittedForm,
  kResultSignature,
  kAttachedRibbonMask,
  kLastMoved,
  kLastMovedSource,
  kLastMovedToken,
  kGlobalFieldCount,
};

enum FormField : std::uint32_t {
  kFormActive = 0u,
  kFormWord,
  kFormQuantitySignature,
  kFormSupport,
  kFormOutputSupport,
  kFormFieldCount,
};

enum OperatorField : std::uint32_t {
  kOperatorActive = 0u,
  kOperatorWord,
  kOperatorFieldsBeforeModes,
};

inline constexpr std::uint32_t kOperatorFieldsPerMode = 4u;
inline constexpr std::uint32_t kOperatorFieldCount =
    kOperatorFieldsBeforeModes + kRouteModeCount * kOperatorFieldsPerMode;
inline constexpr std::uint32_t kFieldCount =
    kGlobalFieldCount + kFormSlotCount * kFormFieldCount + kOperatorSlotCount * kOperatorFieldCount;
inline constexpr std::uint32_t kResidentPhysicalRailCount = kFieldCount * 2u;
inline constexpr std::uint32_t kJournalPhysicalRailCount =
    kJournalDepth * kResidentPhysicalRailCount;
inline constexpr std::uint32_t kPhysicalRailCount =
    kResidentPhysicalRailCount + kJournalPhysicalRailCount;

enum OperatorModeField : std::uint32_t {
  kModeExpectedReafference = 0u,
  kModeExpectedInternal,
  kModePositive,
  kModeNegative,
};

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kPhysicalRailCount]{};
  std::uint64_t ribbon_bodies[kMaxRibbonBodies]{};
  std::uint64_t ribbon_inlet = 0u;
  std::uint32_t ribbon_path = 0u;
  SiteWord ribbon_zero = 0u;
  SiteWord ribbon_one = 0u;
  // ⭐ GROWN ATTACHMENT, added 2026-08-17. Each processive-weight cell owns four
  // seeded sites; site 0 is the body this factor reads as a unary digit, and
  // site 1 is the cell's structural bond word. Attachment is therefore already
  // written in the seeded region and does not need a scalar register to say so.
  //
  // These fields are ADDITIVE and default to zero. Every existing caller builds
  // a DeviceLayout without them and keeps its exact previous behaviour, because
  // only grown_attached_ribbon_mask() and ribbon_signature_grown() read them.
  // A layout that leaves them zero simply reports a grown mask of 0.
  std::uint64_t ribbon_cell_bonds[kMaxRibbonBodies]{};
  SiteWord ribbon_bond_word = 0u;
  // ⭐ GROWN JOURNAL COUNTER, added 2026-08-17. 01ffd4b262 measured that
  // kJournalCount is the single rail that stops route_fixed_schedule(): F leaves
  // kEligibility and kSelectedSourceMask corrupted-but-nonzero so they pass
  // their own gates, while the counter lands far above kJournalDepth and
  // begin_journal() refuses. These bodies carry the same counter on a second
  // hash-seeded processive-weight region, read as plain binary rather than as a
  // unary prefix -- a counter has no prefix semantics to preserve.
  //
  // ADDITIVE AND GATED. A layout that leaves journal_counter_one at zero keeps
  // the rail exactly as before, so every existing caller is untouched. See
  // grown_journal_counter_bound().
  std::uint64_t journal_counter_bodies[kJournalCounterBits]{};
  SiteWord journal_counter_zero = 0u;
  SiteWord journal_counter_one = 0u;
};

struct SourceDescriptor {
  const std::uint8_t* appearance = nullptr;
  std::uint32_t appearance_count = 0u;
  SiteWord* unary_tape = nullptr;
  std::uint32_t unary_capacity = 0u;
  std::uint32_t sensor_site = 0u;
};

struct DeviceInputs {
  const std::uint8_t* left_form = nullptr;
  const std::uint8_t* operator_form = nullptr;
  const std::uint8_t* right_form = nullptr;
  std::uint32_t left_count = 0u;
  std::uint32_t operator_count = 0u;
  std::uint32_t right_count = 0u;
  SourceDescriptor sources[kMaxSources]{};
  std::uint32_t source_count = 0u;
  const std::uint8_t* actual_reafference = nullptr;
  const std::uint8_t* actual_internal = nullptr;
  std::uint32_t actual_reafference_count = 0u;
  std::uint32_t actual_internal_count = 0u;
  std::uint32_t staged = 0u;
  Command command = Command::idle;
};

struct StepReceipt {
  std::uint64_t before_hash = 0u;
  std::uint64_t after_hash = 0u;
  SiteWord left_quantity = 0u;
  SiteWord right_quantity = 0u;
  SiteWord result_signature = 0u;
  SiteWord emitted_form = 0u;
  std::uint32_t operator_slot = kNoSlot;
  std::uint32_t selected_mode = kNoSlot;
  SiteWord selected_source_mask = 0u;
  std::uint32_t moved_source = kNoSource;
  std::uint32_t moved_token = kNoSlot;
  std::uint32_t bound = 0u;
  std::uint32_t prepared = 0u;
  std::uint32_t moved = 0u;
  std::uint32_t finished = 0u;
  std::uint32_t emitted = 0u;
  std::uint32_t revised = 0u;
  std::uint32_t matched = 0u;
  std::uint32_t homeostatic = 0u;
  std::uint32_t probation = 0u;
  std::uint32_t expired = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t inverse_ready = 0u;
};

struct CensusReceipt {
  std::uint32_t resident_matter = 0u;
  std::uint32_t tape_matter = 0u;
  std::uint64_t state_hash = 0u;
};

__host__ __device__ inline PhysicalOffset physical_offset(std::uint32_t index) {
  // Disjoint from the existing cloud, form, sensorimotor, instance, and
  // readout apertures.  The large journal is spread over a compact cuboid.
  return {80 + static_cast<std::int32_t>(index % 64u),
          80 + static_cast<std::int32_t>((index / 64u) % 64u),
          80 + static_cast<std::int32_t>(index / (64u * 64u))};
}

__host__ __device__ inline std::uint32_t resident_index(std::uint32_t field) {
  return field * 2u;
}

__host__ __device__ inline std::uint32_t journal_index(std::uint32_t event,
                                                       std::uint32_t physical_index) {
  return kResidentPhysicalRailCount + event * kResidentPhysicalRailCount + physical_index;
}

__host__ __device__ inline std::uint32_t form_field(std::uint32_t slot, FormField field) {
  return kGlobalFieldCount + slot * kFormFieldCount + static_cast<std::uint32_t>(field);
}

__host__ __device__ inline std::uint32_t operator_field(std::uint32_t slot, std::uint32_t field) {
  return kGlobalFieldCount + kFormSlotCount * kFormFieldCount + slot * kOperatorFieldCount + field;
}

__host__ __device__ inline std::uint32_t operator_mode_field(std::uint32_t slot, std::uint32_t mode,
                                                             OperatorModeField field) {
  return operator_field(slot, kOperatorFieldsBeforeModes + mode * kOperatorFieldsPerMode +
                                  static_cast<std::uint32_t>(field));
}

__device__ inline SiteWord read_field(const SiteWord* words, const DeviceLayout& layout,
                                      std::uint32_t field) {
  return words[layout.rails[resident_index(field)]];
}

__device__ inline void write_field(SiteWord* words, const DeviceLayout& layout, std::uint32_t field,
                                   SiteWord value) {
  const std::uint32_t index = resident_index(field);
  words[layout.rails[index]] = value;
  words[layout.rails[index + 1u]] = ~value;
}

__device__ inline SiteWord read_journal(const SiteWord* words, const DeviceLayout& layout,
                                        std::uint32_t event, std::uint32_t physical_index) {
  return words[layout.rails[journal_index(event, physical_index)]];
}

__device__ inline void write_journal(SiteWord* words, const DeviceLayout& layout,
                                     std::uint32_t event, std::uint32_t physical_index,
                                     SiteWord value) {
  words[layout.rails[journal_index(event, physical_index)]] = value;
}

__device__ inline std::uint64_t mix_hash(std::uint64_t hash, std::uint64_t value) {
  hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6u) + (hash >> 2u);
  return hash * UINT64_C(1099511628211);
}

__device__ inline std::uint64_t state_hash(const SiteWord* words, const DeviceLayout& layout) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index)
    hash = mix_hash(hash, words[layout.rails[index]]);
  return hash;
}

// A layout only uses the grown counter if it actually bound one. Two distinct
// non-zero alphabet words are required, so a half-filled layout falls back to
// the rail rather than reading zeros out of slot 0 and calling it a count.
__device__ inline bool grown_journal_counter_bound(const DeviceLayout& layout) {
  return layout.journal_counter_one != 0u && layout.journal_counter_zero != 0u &&
         layout.journal_counter_one != layout.journal_counter_zero;
}

// Plain binary over the counter region's bodies. `legible` is false if any body
// holds a word outside the seed's two-word alphabet -- the counter then has no
// value at all, which is a different thing from having a large one, and callers
// must not treat corruption as a count.
__device__ inline std::uint32_t grown_journal_count(const SiteWord* words,
                                                    const DeviceLayout& layout, bool* legible) {
  std::uint32_t value = 0u;
  *legible = true;
  for (std::uint32_t bit = 0u; bit < kJournalCounterBits; ++bit) {
    const SiteWord word = words[layout.journal_counter_bodies[bit]];
    if (word == layout.journal_counter_one)
      value |= 1u << bit;
    else if (word != layout.journal_counter_zero)
      *legible = false;
  }
  return value;
}

__device__ inline void write_grown_journal_count(SiteWord* words, const DeviceLayout& layout,
                                                 std::uint32_t value) {
  for (std::uint32_t bit = 0u; bit < kJournalCounterBits; ++bit)
    words[layout.journal_counter_bodies[bit]] = ((value >> bit) & 1u) != 0u
                                                    ? layout.journal_counter_one
                                                    : layout.journal_counter_zero;
}

__device__ inline std::uint32_t journal_count(const SiteWord* words, const DeviceLayout& layout,
                                              bool* legible) {
  if (!grown_journal_counter_bound(layout)) {
    *legible = true;
    return read_field(words, layout, kJournalCount);
  }
  return grown_journal_count(words, layout, legible);
}

__device__ inline void set_journal_count(SiteWord* words, const DeviceLayout& layout,
                                         std::uint32_t value) {
  if (!grown_journal_counter_bound(layout)) {
    write_field(words, layout, kJournalCount, value);
    return;
  }
  write_grown_journal_count(words, layout, value);
}

__device__ inline bool begin_journal(SiteWord* words, const DeviceLayout& layout) {
  bool legible = false;
  const std::uint32_t event = journal_count(words, layout, &legible);
  if (!legible || event >= kJournalDepth)
    return false;
  for (std::uint32_t index = 0u; index < kResidentPhysicalRailCount; ++index)
    write_journal(words, layout, event, index, words[layout.rails[index]]);
  set_journal_count(words, layout, event + 1u);
  return true;
}

__device__ inline void restore_last_journal(SiteWord* words, const DeviceLayout& layout) {
  bool legible = false;
  const std::uint32_t count = journal_count(words, layout, &legible);
  if (!legible || count == 0u)
    return;
  const std::uint32_t event = count - 1u;
  for (std::uint32_t index = 0u; index < kResidentPhysicalRailCount; ++index) {
    words[layout.rails[index]] = read_journal(words, layout, event, index);
    write_journal(words, layout, event, index, (index & 1u) == 0u ? 0u : 0xffffffffu);
  }
  // The loop above rewrote every resident rail from the journal, including the
  // vestigial kJournalCount rail. The grown counter is not a rail, so it has to
  // be wound back explicitly. begin_journal() snapshots BEFORE incrementing, so
  // the value that was current at write time is `event`, which is exactly what
  // the rail path used to restore.
  set_journal_count(words, layout, event);
}

__device__ inline SiteWord pack_form(const std::uint8_t* bytes, std::uint32_t count) {
  if (bytes == nullptr || count != kFormBytes)
    return 0u;
  SiteWord result = 0u;
  for (std::uint32_t index = 0u; index < kFormBytes; ++index)
    result |= static_cast<SiteWord>(bytes[index]) << (index * 8u);
  return result == 0u ? 1u : result;
}

__device__ inline bool token_present(SiteWord word, std::uint32_t lane) {
  return (word & carrier_bit(lane)) == 0u;
}

__device__ inline SiteWord quantity_signature(const SourceDescriptor& source, std::uint32_t lane) {
  if (source.unary_tape == nullptr || source.unary_capacity == 0u ||
      source.unary_capacity > kTokensPerSource)
    return 0u;
  std::uint64_t hash = UINT64_C(1469598103934665603);
  bool any = false;
  for (std::uint32_t index = 0u; index < kMaxRibbonBodies; ++index) {
    const bool present =
        index < source.unary_capacity && token_present(source.unary_tape[index], lane);
    any = any || present;
    hash = mix_hash(hash, present ? (index + 1u) : 0u);
  }
  const SiteWord folded = static_cast<SiteWord>(hash ^ (hash >> 32u));
  return any ? (folded == 0u ? 1u : folded) : 0u;
}

__device__ inline SiteWord increment_unary(SiteWord value) {
  const SiteWord free = ~value;
  return free == 0u ? value : value | (free & (0u - free));
}

__device__ inline std::uint32_t find_form(const SiteWord* words, const DeviceLayout& layout,
                                          SiteWord form) {
  for (std::uint32_t slot = 0u; slot < kFormSlotCount; ++slot)
    if (read_field(words, layout, form_field(slot, kFormActive)) != 0u &&
        read_field(words, layout, form_field(slot, kFormWord)) == form)
      return slot;
  return kNoSlot;
}

__device__ inline std::uint32_t free_form(const SiteWord* words, const DeviceLayout& layout) {
  for (std::uint32_t slot = 0u; slot < kFormSlotCount; ++slot)
    if (read_field(words, layout, form_field(slot, kFormActive)) == 0u)
      return slot;
  return kNoSlot;
}

__device__ inline std::uint32_t find_operator(const SiteWord* words, const DeviceLayout& layout,
                                              SiteWord form) {
  for (std::uint32_t slot = 0u; slot < kOperatorSlotCount; ++slot)
    if (read_field(words, layout, operator_field(slot, kOperatorActive)) != 0u &&
        read_field(words, layout, operator_field(slot, kOperatorWord)) == form)
      return slot;
  return kNoSlot;
}

__device__ inline std::uint32_t recruit_operator(SiteWord* words, const DeviceLayout& layout,
                                                 SiteWord form) {
  const std::uint32_t existing = find_operator(words, layout, form);
  if (existing != kNoSlot)
    return existing;
  for (std::uint32_t slot = 0u; slot < kOperatorSlotCount; ++slot) {
    if (read_field(words, layout, operator_field(slot, kOperatorActive)) != 0u)
      continue;
    write_field(words, layout, operator_field(slot, kOperatorActive), 1u);
    write_field(words, layout, operator_field(slot, kOperatorWord), form);
    return slot;
  }
  return kNoSlot;
}

__device__ inline bool bind_quantity_device(SiteWord* words, const DeviceLayout& layout,
                                            DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  if (inputs.source_count != 1u || inputs.sources[0].unary_tape == nullptr)
    return false;
  const SiteWord form = pack_form(inputs.left_form, inputs.left_count);
  const SiteWord quantity = quantity_signature(inputs.sources[0], layout.ribbon_path);
  if (form == 0u || quantity == 0u)
    return false;
  std::uint32_t slot = find_form(words, layout, form);
  if (slot == kNoSlot)
    slot = free_form(words, layout);
  if (slot == kNoSlot || !begin_journal(words, layout))
    return false;
  write_field(words, layout, form_field(slot, kFormActive), 1u);
  write_field(words, layout, form_field(slot, kFormWord), form);
  write_field(words, layout, form_field(slot, kFormQuantitySignature), quantity);
  write_field(words, layout, form_field(slot, kFormSupport),
              increment_unary(read_field(words, layout, form_field(slot, kFormSupport))));
  write_field(words, layout, form_field(slot, kFormOutputSupport),
              increment_unary(read_field(words, layout, form_field(slot, kFormOutputSupport))));
  inputs.staged = 0u;
  local.bound = 1u;
  local.left_quantity = quantity;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline bool ground_homeostasis_device(SiteWord* words, const DeviceLayout& layout,
                                                 DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  const SiteWord reafference =
      pack_form(inputs.actual_reafference, inputs.actual_reafference_count);
  const SiteWord internal = pack_form(inputs.actual_internal, inputs.actual_internal_count);
  if (reafference == 0u || internal == 0u ||
      read_field(words, layout, kHomeostaticReafference) != 0u ||
      read_field(words, layout, kHomeostaticInternal) != 0u || !begin_journal(words, layout))
    return false;
  write_field(words, layout, kHomeostaticReafference, reafference);
  write_field(words, layout, kHomeostaticInternal, internal);
  inputs.staged = 0u;
  local.bound = 1u;
  local.homeostatic = 1u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline std::uint32_t source_for_quantity(const DeviceInputs& inputs, SiteWord quantity,
                                                    std::uint32_t lane, bool* ambiguous) {
  std::uint32_t result = kNoSource;
  for (std::uint32_t source = 0u; source < inputs.source_count && source < kMaxSources; ++source) {
    if (quantity_signature(inputs.sources[source], lane) != quantity)
      continue;
    if (result != kNoSource) {
      *ambiguous = true;
      return kNoSource;
    }
    result = source;
  }
  return result;
}

__device__ inline std::int32_t mode_score(const SiteWord* words, const DeviceLayout& layout,
                                          std::uint32_t operator_slot, std::uint32_t mode) {
  return static_cast<std::int32_t>(__popc(
             read_field(words, layout, operator_mode_field(operator_slot, mode, kModePositive)))) -
         static_cast<std::int32_t>(__popc(
             read_field(words, layout, operator_mode_field(operator_slot, mode, kModeNegative))));
}

__device__ inline std::uint32_t choose_mode(SiteWord* words, const DeviceLayout& layout,
                                            std::uint32_t operator_slot, bool* abstained) {
  std::uint32_t total_positive = 0u;
  for (std::uint32_t mode = 0u; mode < kRouteModeCount; ++mode)
    total_positive +=
        __popc(read_field(words, layout, operator_mode_field(operator_slot, mode, kModePositive)));
  if (total_positive == 0u) {
    const std::uint32_t phase = read_field(words, layout, kExplorerPhase);
    write_field(words, layout, kExplorerPhase, phase + 1u);
    return phase % kRouteModeCount;
  }
  std::uint32_t best = 0u;
  std::int32_t best_score = mode_score(words, layout, operator_slot, 0u);
  bool tie = false;
  for (std::uint32_t mode = 1u; mode < kRouteModeCount; ++mode) {
    const std::int32_t score = mode_score(words, layout, operator_slot, mode);
    if (score > best_score) {
      best = mode;
      best_score = score;
      tie = false;
    } else if (score == best_score) {
      tie = true;
    }
  }
  if (tie || best_score <= 0) {
    *abstained = true;
    return kNoSlot;
  }
  return best;
}

__device__ inline bool prepare_query_device(SiteWord* words, const DeviceLayout& layout,
                                            DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  if (read_field(words, layout, kEligibility) != 0u || inputs.source_count == 0u ||
      inputs.source_count > kMaxSources)
    return false;
  const SiteWord left = pack_form(inputs.left_form, inputs.left_count);
  const SiteWord op = pack_form(inputs.operator_form, inputs.operator_count);
  const SiteWord right = pack_form(inputs.right_form, inputs.right_count);
  const std::uint32_t left_slot = find_form(words, layout, left);
  const std::uint32_t right_slot = find_form(words, layout, right);
  if (left_slot == kNoSlot || right_slot == kNoSlot || op == 0u)
    return false;
  const SiteWord left_quantity =
      read_field(words, layout, form_field(left_slot, kFormQuantitySignature));
  const SiteWord right_quantity =
      read_field(words, layout, form_field(right_slot, kFormQuantitySignature));
  bool ambiguous = false;
  const std::uint32_t left_source =
      source_for_quantity(inputs, left_quantity, layout.ribbon_path, &ambiguous);
  const std::uint32_t right_source =
      source_for_quantity(inputs, right_quantity, layout.ribbon_path, &ambiguous);
  if (ambiguous || left_source == kNoSource || right_source == kNoSource)
    return false;
  if (!begin_journal(words, layout))
    return false;
  const std::uint32_t operator_slot = recruit_operator(words, layout, op);
  if (operator_slot == kNoSlot) {
    restore_last_journal(words, layout);
    return false;
  }
  bool abstained = false;
  const std::uint32_t mode = choose_mode(words, layout, operator_slot, &abstained);
  SiteWord selected = 0u;
  if (!abstained) {
    if (mode == 0u) {
      selected = (1u << left_source) | (1u << right_source);
    } else if (mode == 1u) {
      selected = 1u << left_source;
    } else if (mode == 2u) {
      selected = 1u << right_source;
    } else {
      for (std::uint32_t source = 0u; source < inputs.source_count; ++source)
        if (source != left_source && source != right_source) {
          selected = 1u << source;
          break;
        }
      if (selected == 0u)
        abstained = true;
    }
  }
  write_field(words, layout, kActiveOperatorSlot, operator_slot);
  write_field(words, layout, kSelectedMode, abstained ? kNoSlot : mode);
  write_field(words, layout, kSelectedSourceMask, abstained ? 0u : selected);
  write_field(words, layout, kRouteCursor, 0u);
  write_field(words, layout, kMotorWitness, abstained ? 0u : (1u << mode));
  write_field(words, layout, kEligibility, abstained ? 0u : 1u);
  write_field(words, layout, kEligibilityAge, 0u);
  write_field(words, layout, kLastMoved, 0u);
  inputs.staged = 0u;
  local.left_quantity = left_quantity;
  local.right_quantity = right_quantity;
  local.operator_slot = operator_slot;
  local.selected_mode = abstained ? kNoSlot : mode;
  local.selected_source_mask = abstained ? 0u : selected;
  local.prepared = abstained ? 0u : 1u;
  local.abstained = abstained ? 1u : 0u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline bool route_one_device(SiteWord* words, const DeviceLayout& layout,
                                        DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  const SiteWord selected = read_field(words, layout, kSelectedSourceMask);
  if (read_field(words, layout, kEligibility) == 0u || selected == 0u)
    return false;
  if (!begin_journal(words, layout))
    return false;
  const std::uint32_t cursor = read_field(words, layout, kRouteCursor);
  std::uint32_t moved_source = kNoSource;
  std::uint32_t moved_token = kNoSlot;
  for (std::uint32_t offset = 0u; offset < kMaxSources * kTokensPerSource; ++offset) {
    const std::uint32_t flat = cursor + offset;
    const std::uint32_t source = flat / kTokensPerSource;
    const std::uint32_t token = flat % kTokensPerSource;
    if (source >= inputs.source_count || source >= kMaxSources)
      break;
    if ((selected & (1u << source)) == 0u || inputs.sources[source].unary_tape == nullptr ||
        token >= inputs.sources[source].unary_capacity)
      continue;
    SiteWord& tape = inputs.sources[source].unary_tape[token];
    const SiteWord lane = carrier_bit(layout.ribbon_path);
    if ((tape & lane) != 0u)
      continue;
    SiteWord& inlet = words[layout.ribbon_inlet];
    const SiteWord first = inlet & lane;
    const SiteWord second = tape & lane;
    inlet = (inlet & ~lane) | second;
    tape = (tape & ~lane) | first;
    moved_source = source;
    moved_token = token;
    write_field(words, layout, kRouteCursor, flat + 1u);
    break;
  }
  write_field(words, layout, kLastMoved, moved_source == kNoSource ? 0u : 1u);
  write_field(words, layout, kLastMovedSource, moved_source == kNoSource ? 0u : moved_source);
  write_field(words, layout, kLastMovedToken, moved_token == kNoSlot ? 0u : moved_token);
  inputs.staged = 0u;
  local.selected_source_mask = selected;
  local.moved_source = moved_source;
  local.moved_token = moved_token;
  local.moved = moved_source == kNoSource ? 0u : 1u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

// ⭐ THE GROWN ALTERNATIVE TO kAttachedRibbonMask.
//
// `kAttachedRibbonMask` is a placed scalar rail, and
// `bcc32_arithmetic_ribbon_controller_writeback_contract` (6ff1d5a5fc) measured
// ordinary F rotting it 0x7f -> 0x9bfd0bef over twelve ticks with no write at
// all -- while every one of the ribbon's own seeded sites stayed bit-identical
// across the same window. So the reader was the fragile half, not the carrier.
//
// Attachment does not need a register. A processive-weight cell carries its own
// structural bond word at site 1 of the cell, inside the same hash-seeded region
// F preserves. Reading attachment from there makes the mask grown matter rather
// than a scalar the host placed and F destroys.
//
// A layout with `ribbon_bond_word == 0` reports 0: nothing is silently declared
// attached because a caller did not fill the new fields in.
__device__ inline SiteWord grown_attached_ribbon_mask(const SiteWord* words,
                                                      const DeviceLayout& layout) {
  if (layout.ribbon_bond_word == 0u)
    return 0u;
  SiteWord attached = 0u;
  for (std::uint32_t index = 0u; index < kMaxRibbonBodies; ++index)
    if (words[layout.ribbon_cell_bonds[index]] == layout.ribbon_bond_word)
      attached |= static_cast<SiteWord>(1u) << index;
  return attached;
}

__device__ inline SiteWord ribbon_signature_with_mask(const SiteWord* words,
                                                      const DeviceLayout& layout, SiteWord attached,
                                                      bool* valid_prefix) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  bool saw_zero = false;
  bool any = false;
  *valid_prefix = true;
  for (std::uint32_t index = 0u; index < kMaxRibbonBodies; ++index) {
    const bool available = (attached & (1u << index)) != 0u;
    const SiteWord body = available ? words[layout.ribbon_bodies[index]] : kQ;
    const bool one = available && body == layout.ribbon_one;
    const bool zero = available && body == layout.ribbon_zero;
    if (available && !one && !zero)
      *valid_prefix = false;
    if (one && saw_zero)
      *valid_prefix = false;
    saw_zero = saw_zero || zero;
    any = any || one;
    hash = mix_hash(hash, one ? (index + 1u) : 0u);
  }
  const SiteWord folded = static_cast<SiteWord>(hash ^ (hash >> 32u));
  return any && *valid_prefix ? (folded == 0u ? 1u : folded) : 0u;
}

// Unchanged behaviour for every existing caller: attachment still comes from the
// placed rail. Kept as the control arm against which the grown reader below is
// compared, so the two can be measured on one organism in one window.
__device__ inline SiteWord ribbon_signature(const SiteWord* words, const DeviceLayout& layout,
                                            bool* valid_prefix) {
  return ribbon_signature_with_mask(words, layout,
                                    read_field(words, layout, kAttachedRibbonMask), valid_prefix);
}

// The same reader over grown attachment. No rail is consulted.
__device__ inline SiteWord ribbon_signature_grown(const SiteWord* words,
                                                  const DeviceLayout& layout, bool* valid_prefix) {
  return ribbon_signature_with_mask(words, layout, grown_attached_ribbon_mask(words, layout),
                                    valid_prefix);
}

__device__ inline bool finish_result_device(SiteWord* words, const DeviceLayout& layout,
                                            DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  if (read_field(words, layout, kEligibility) == 0u)
    return false;
  bool valid_prefix = false;
  // ⭐ 1b696c6ae1 -> live. Attachment now comes from the seeded region, not from
  // the kAttachedRibbonMask rail. The rail had exactly one reader (this line) and
  // exactly one writer (a test fixture), so this retires it from the live path:
  // the contract's own founder now POISONS that rail and still passes.
  const SiteWord result = ribbon_signature_grown(words, layout, &valid_prefix);
  if (!valid_prefix || result == 0u || !begin_journal(words, layout))
    return false;
  std::uint32_t best = kNoSlot;
  std::uint32_t best_support = 0u;
  bool tie = false;
  for (std::uint32_t slot = 0u; slot < kFormSlotCount; ++slot) {
    if (read_field(words, layout, form_field(slot, kFormActive)) == 0u ||
        read_field(words, layout, form_field(slot, kFormQuantitySignature)) != result)
      continue;
    const std::uint32_t support =
        __popc(read_field(words, layout, form_field(slot, kFormOutputSupport)));
    if (support > best_support) {
      best = slot;
      best_support = support;
      tie = false;
    } else if (support != 0u && support == best_support) {
      tie = true;
    }
  }
  const SiteWord form =
      best == kNoSlot || tie ? 0u : read_field(words, layout, form_field(best, kFormWord));
  write_field(words, layout, kResultSignature, result);
  write_field(words, layout, kEmittedForm, form);
  inputs.staged = 0u;
  local.result_signature = result;
  local.emitted_form = form;
  local.finished = 1u;
  local.emitted = form == 0u ? 0u : 1u;
  local.abstained = form == 0u ? 1u : 0u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline bool settle_consequence_device(SiteWord* words, const DeviceLayout& layout,
                                                 DeviceInputs& inputs, StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  if (read_field(words, layout, kEligibility) == 0u)
    return false;
  const std::uint32_t operator_slot = read_field(words, layout, kActiveOperatorSlot);
  const std::uint32_t mode = read_field(words, layout, kSelectedMode);
  const SiteWord reafference =
      pack_form(inputs.actual_reafference, inputs.actual_reafference_count);
  const SiteWord internal = pack_form(inputs.actual_internal, inputs.actual_internal_count);
  if (operator_slot >= kOperatorSlotCount || mode >= kRouteModeCount || reafference == 0u ||
      internal == 0u || !begin_journal(words, layout))
    return false;
  const std::uint32_t expected_reafference =
      operator_mode_field(operator_slot, mode, kModeExpectedReafference);
  const std::uint32_t expected_internal =
      operator_mode_field(operator_slot, mode, kModeExpectedInternal);
  const std::uint32_t positive = operator_mode_field(operator_slot, mode, kModePositive);
  const std::uint32_t negative = operator_mode_field(operator_slot, mode, kModeNegative);
  SiteWord expected_r = read_field(words, layout, expected_reafference);
  SiteWord expected_i = read_field(words, layout, expected_internal);
  const SiteWord homeostatic_r = read_field(words, layout, kHomeostaticReafference);
  const SiteWord homeostatic_i = read_field(words, layout, kHomeostaticInternal);
  if (homeostatic_r == 0u || homeostatic_i == 0u) {
    restore_last_journal(words, layout);
    return false;
  }
  if (expected_r == 0u && expected_i == 0u) {
    write_field(words, layout, expected_reafference, reafference);
    write_field(words, layout, expected_internal, internal);
    local.probation = 1u;
  } else {
    local.matched = expected_r == reafference && expected_i == internal;
  }
  // Route value is not predictability.  A separately grounded resident body
  // baseline supplies the comparison: only a consequence that returns both
  // raw channels to that maintained state earns positive support.  A stable
  // but non-homeostatic port remains negative.
  local.homeostatic = reafference == homeostatic_r && internal == homeostatic_i;
  if (local.homeostatic != 0u) {
    write_field(words, layout, positive, increment_unary(read_field(words, layout, positive)));
  } else {
    write_field(words, layout, negative, increment_unary(read_field(words, layout, negative)));
  }
  write_field(words, layout, kEligibility, 0u);
  write_field(words, layout, kEligibilityAge, 0u);
  write_field(words, layout, kMotorWitness, 0u);
  write_field(words, layout, kSelectedSourceMask, 0u);
  inputs.staged = 0u;
  local.operator_slot = operator_slot;
  local.selected_mode = mode;
  local.revised = 1u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline bool age_device(SiteWord* words, const DeviceLayout& layout, DeviceInputs& inputs,
                                  StepReceipt* receipt) {
  StepReceipt local{};
  local.before_hash = state_hash(words, layout);
  if (!begin_journal(words, layout))
    return false;
  if (read_field(words, layout, kEligibility) != 0u) {
    const std::uint32_t age = read_field(words, layout, kEligibilityAge);
    if (age >= 3u) {
      write_field(words, layout, kEligibility, 0u);
      write_field(words, layout, kMotorWitness, 0u);
      write_field(words, layout, kSelectedSourceMask, 0u);
      local.expired = 1u;
    } else {
      write_field(words, layout, kEligibilityAge, age + 1u);
    }
  }
  inputs.staged = 0u;
  local.after_hash = state_hash(words, layout);
  local.inverse_ready = 1u;
  if (receipt != nullptr)
    *receipt = local;
  return true;
}

__device__ inline bool step_device(SiteWord* words, const DeviceLayout& layout,
                                   DeviceInputs* inputs, StepReceipt* receipt) {
  if (inputs == nullptr || inputs->staged == 0u)
    return false;
  switch (inputs->command) {
    case Command::ground_homeostasis:
      return ground_homeostasis_device(words, layout, *inputs, receipt);
    case Command::bind_quantity:
      return bind_quantity_device(words, layout, *inputs, receipt);
    case Command::prepare_query:
      return prepare_query_device(words, layout, *inputs, receipt);
    case Command::route_one:
      return route_one_device(words, layout, *inputs, receipt);
    case Command::finish_result:
      return finish_result_device(words, layout, *inputs, receipt);
    case Command::settle_consequence:
      return settle_consequence_device(words, layout, *inputs, receipt);
    case Command::age:
      return age_device(words, layout, *inputs, receipt);
    default:
      return false;
  }
}

__device__ inline void inverse_step_device(SiteWord* words, const DeviceLayout& layout,
                                           DeviceInputs* inputs) {
  if (read_field(words, layout, kLastMoved) != 0u && inputs != nullptr) {
    const std::uint32_t source = read_field(words, layout, kLastMovedSource);
    const std::uint32_t token = read_field(words, layout, kLastMovedToken);
    if (source < inputs->source_count && source < kMaxSources &&
        inputs->sources[source].unary_tape != nullptr &&
        token < inputs->sources[source].unary_capacity) {
      const SiteWord lane = carrier_bit(layout.ribbon_path);
      SiteWord& inlet = words[layout.ribbon_inlet];
      SiteWord& tape = inputs->sources[source].unary_tape[token];
      const SiteWord first = inlet & lane;
      const SiteWord second = tape & lane;
      inlet = (inlet & ~lane) | second;
      tape = (tape & ~lane) | first;
    }
  }
  restore_last_journal(words, layout);
}

static __global__ void step_kernel(SiteWord* words, const DeviceLayout* layout,
                                   DeviceInputs* inputs, StepReceipt* receipt,
                                   std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *advanced = step_device(words, *layout, inputs, receipt) ? 1u : 0u;
}

static __global__ void inverse_step_kernel(SiteWord* words, const DeviceLayout* layout,
                                           DeviceInputs* inputs) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  inverse_step_device(words, *layout, inputs);
}

static __global__ void census_kernel(const SiteWord* words, const DeviceLayout* layout,
                                     const DeviceInputs* inputs, CensusReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  CensusReceipt local{};
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index)
    local.resident_matter += __popc(words[layout->rails[index]]);
  if (inputs != nullptr) {
    for (std::uint32_t source = 0u; source < inputs->source_count && source < kMaxSources; ++source)
      for (std::uint32_t token = 0u; token < inputs->sources[source].unary_capacity; ++token)
        local.tape_matter += __popc(inputs->sources[source].unary_tape[token]);
  }
  local.state_hash = state_hash(words, *layout);
  if (receipt != nullptr)
    *receipt = local;
}

}  // namespace substrate::bcc32::grown_symbolic_arithmetic_factor
