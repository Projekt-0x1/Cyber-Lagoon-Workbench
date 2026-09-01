#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bcc32_cuda_adult_device_memory.cuh"
#include "bcc32_cuda_resident_answer_frame.cuh"
#include "bcc32_cuda_resident_proposition_chain.cuh"
#include "bcc32_cuda_resident_proposition_tissue.cuh"
#include "bcc32_cuda_resident_question_goal.cuh"
#include "bcc32_cuda_resident_role_compositor.cuh"
#include "bcc32_cuda_resident_surface_organ_v2.cuh"
#include "bcc32_cuda_resident_construction_composer.cuh"
#include "bcc32_cuda_resident_ordered_relation_assimilation.cuh"
#include "bcc32_cuda_resident_credit_bank.cuh"
#include "bcc32_cuda_distributed_sequence_motor.cuh"
#include "bcc32_cuda_resident_population_surface.cuh"
#include "bcc32_cuda_adult_conditioned_credit.cuh"
#include "bcc32_cuda_resident_synthesis.cuh"

namespace bcc32_cuda_adult_v1 {

namespace answer_frame = substrate::bcc32::resident_answer_frame;
namespace role_compositor = substrate::bcc32::resident_role_compositor;
namespace surface_organ = substrate::bcc32::resident_surface_organ_v2;
namespace context_state = substrate::bcc32::resident_context_state;
namespace roles = substrate::bcc32::resident_roles;
namespace construction = substrate::bcc32::resident_construction;
namespace distributed_motor = bcc32_cuda_distributed_sequence_motor;
namespace proposition_chain = bcc32_cuda_resident_proposition_chain;
namespace proposition_tissue = bcc32_cuda_resident_proposition_tissue;
namespace question_goal = bcc32_cuda_resident_question_goal;
namespace synthesis = bcc32_cuda_resident_synthesis;
namespace ordered_relation = bcc32_cuda_resident_ordered_relation_assimilation;

constexpr std::uint32_t kBoundaryCount = 1u;
constexpr std::uint32_t kClosureCount = 2u;
constexpr std::uint32_t kStreamingCueCapacity = 4096u;
constexpr std::uint32_t kMaxUnitBytes = 48u;
constexpr std::uint32_t kUnitWords = kMaxUnitBytes / 4u;
constexpr std::uint32_t kTopK = 8u;
constexpr std::uint32_t kUnigramTop = 32u;
constexpr std::uint32_t kBlock = 256u;
constexpr std::uint32_t kOnlineUnitReserve = 262144u;
constexpr std::uint32_t kOnlineNgramCapacity = 2097152u;
constexpr std::uint32_t kOnlineAssociationCapacity = 16777216u;
constexpr std::uint32_t kOnlineConditionedTransitionCapacity = 33554432u;
constexpr std::uint32_t kResidentCreditBankCapacity = 262144u;
constexpr std::uint32_t kAssociationRadius = 8u;
constexpr std::uint32_t kOnlineEpisodeCapacity = 8388608u;
constexpr std::uint32_t kOnlineEpisodeBreakCapacity = 262144u;
constexpr std::uint32_t kOnlineAssimilationChunkUnits = 1048576u;
constexpr std::uint32_t kCompositionUnits = 64u;
constexpr std::uint32_t kEpisodeCompletionUnits = kCompositionUnits;
constexpr std::uint32_t kCompositionSnippetUnits = 4u;
constexpr std::uint32_t kCompositionBeamWidth = 32u;
constexpr std::uint32_t kCompositionBeamExpansions = 4u;
constexpr std::uint32_t kCompositionMinUnits = 8u;
constexpr std::uint32_t kCompositionMaxUnits = 64u;
constexpr std::uint32_t kCompositionSemanticLaunchUnits = 4u;
constexpr std::uint32_t kCompositionSourceRunLimit = 8u;
constexpr std::uint32_t kCompositionSplicedBit = 0x80000000u;
constexpr std::uint32_t kCueAnchorLimit = 32u;
constexpr std::uint32_t kCueAlignmentUnits = 16u;
constexpr std::uint32_t kCueNearIdentity = 57344u;
constexpr std::uint32_t kCueStrongIdentity = 58982u;
constexpr std::uint32_t kMotorWords = 16u;
// Persistent-topic (subject-field) parameters. Port of the measured resident
// working-set + drift re-anchoring mechanism (experiments/language_port/
// cuda_resident_adult.cu @ e1f39a68b) onto the unit-level generator: the cue's
// content units form a resident decaying activation field that (a) BIASES the
// continuation walk toward on-subject transitions, (b) is REINFORCED when the
// walk re-emits an on-subject unit, and (c) RE-ANCHORS the walk onto a resident
// subject unit when it drifts off the field for kSubjectDriftUnits units.
constexpr std::uint32_t kSubjectCap = 64u;
constexpr std::uint32_t kSubjectLocalCandidates = 4u;
constexpr std::uint32_t kSubjectInitWeight = 256u;
constexpr std::uint32_t kSubjectReinforce = 256u;
constexpr std::uint32_t kSubjectCapWeight = 3072u;
constexpr std::uint32_t kSubjectFloorWeight = 32u;
constexpr std::uint32_t kSubjectDriftUnits = 3u;
constexpr std::uint32_t kSubjectMinLen = 3u;
constexpr std::uint32_t kMotorRecentUnits = 16u;
constexpr std::uint32_t kResidentRelationTraceCapacity = 4096u;
constexpr std::uint32_t kResidentRelationTraceUnitCapacity =
    kResidentRelationTraceCapacity * kCompositionUnits;
// Cycle-break: the drift/re-anchor lever above only fires when subj_n > 0 (an
// active cue-derived subject field). A plain order-2 walk with NO active
// subject field (short/generic cue fragment; near-identity match found no
// content units) has no such lever at all, and a strong low-diversity
// (first,second)->next trigram edge can form a short deterministic cycle the
// walk revisits forever (observed: "wash their hands when" repeating >15x
// verbatim from a single corpus occurrence -- the exact 4-unit cycle does not
// appear anywhere in the corpus itself, confirming it is a walk artifact, not
// a copied span). kCtxHistWindow tracks recently-visited (first,second)
// context pairs; revisiting one forces a random unigram jump, independent of
// subj_n, so both biased and unbiased walks get the same safety net.
constexpr std::uint32_t kCtxHistWindow = 10u;
// Forced-run break: the cycle-break above only fires on a REVISITED
// (first,second) context -- by construction it cannot catch a long walk
// through a rare, distinctive corpus phrase, because each successive
// trigram context in a run of unique words is itself unique and never
// repeats. When a trigram/bigram context has evidence for exactly one
// candidate next unit (no competing continuation exists anywhere in the
// corpus or online experience), choose_weighted has nothing to sample
// from and the walk is forced to replay that one corpus occurrence
// verbatim. kMaxForcedRun counts consecutive single-candidate ("forced")
// steps and, once exceeded, breaks away the same way the cycle-break
// does -- a random unigram jump. Measured via the raw-text verbatim_span
// gate metric (bcc32_cuda_adult_stream_gate.py): unbroken forced runs
// produced reproducible 45-213 byte verbatim spans (an academic citation
// and a "Crusonia plant" passage quoted almost word for word); this is
// the located mechanism behind that copying, not the trigram/bigram
// tables themselves.
constexpr std::uint32_t kMaxForcedRun = 4u;
// QUESTION-ORIGINATION (--qorig, off by default). Port of the measured
// mechanism from experiments/language_port/cuda_resident_adult.cu @ 2fbae402b
// onto BCC-32: a unit enters the resident ONSET vocabulary once the online
// corpus has attested it as a question span's first unit at least
// kQOrigOnsetMin times AND that count clears the SPECIFICITY gate (onset in
// >= ~1/kQOrigSpecific of its total occurrences, so generic openers like
// "the"/"a" that only incidentally precede a '?' are excluded -- emergent
// from counts, no authored word list); the TERMINAL vocabulary gates on
// kQOrigTermMin alone (any unit repeatedly '?'-fused is a legal close).
constexpr std::uint32_t kQOrigMaxSpan = 24u;   // max units scanned back for a question span's onset
constexpr std::uint32_t kQOrigInject = 64u;    // bounded onset/terminal vocabulary size
constexpr std::uint32_t kQOrigOnsetMin = 8u;
constexpr std::uint32_t kQOrigTermMin = 6u;
constexpr std::uint32_t kQOrigSpecific = 16u;
constexpr std::uint32_t kQOrigCap = 48u;       // saturate the per-slot terminal-boost contribution
constexpr std::uint32_t kQOrigSoftCap = 9u;    // body units before the forced-close fallback engages
constexpr std::uint32_t kQOrigRate = 110u;     // onset-jump probability scale: P = drive*rate/256
constexpr std::uint32_t kQOrigSubjScale = 20u; // drive confidence = floor(log2(occurrence)) * scale
constexpr std::uint32_t kCueWindowRadius = 6u;
constexpr std::uint32_t kBaseWindowUnits = 16u;
constexpr std::uint32_t kBaseWindowStride = 4u;
constexpr std::uint32_t kResidentMassBudget = 2000000000u;
constexpr std::uint32_t kDistributedMotorPopulation = 262144u;
constexpr std::uint32_t kDistributedMotorActiveWidth = 8u;
constexpr std::uint32_t kDistributedMotorWindowBytes = 16u;
constexpr std::uint32_t kDistributedMotorScratchSteps = 65536u;
constexpr unsigned long long kDistributedMotorMassBudget = 1000000000ull;
constexpr std::uint32_t kResidentPropositionSynapseCapacity = 65536u;
constexpr std::uint32_t kResidentOrderedBindingCapacity = 65536u;
constexpr std::uint32_t kResidentPropositionOutputCapacity = 16u;
constexpr std::uint64_t kResidentPropositionMassBudget = 8000000ull;
constexpr std::uint32_t kSurfaceContextCapacity = 262144u;
constexpr std::uint32_t kSurfaceContextStateCapacity = 16384u;
constexpr std::uint32_t kSurfaceContextMembershipCapacity = 1048576u;
constexpr std::uint32_t kSurfaceContextTransitionCapacity = 1048576u;
constexpr std::uint64_t kSurfaceContextMassBudget = 1ull << 40u;
constexpr std::uint32_t kCooperativeGenerationThreshold = 1000000u;
constexpr std::uint32_t kEfferenceTraceBytes = 4096u;
constexpr std::uint32_t kInteractionShadowBytes = 4096u;

struct UnitKey {
  std::uint32_t hash_a;
  std::uint32_t hash_b;
  std::uint32_t length;

  __host__ __device__ bool operator<(const UnitKey& other) const {
    if (hash_a != other.hash_a) return hash_a < other.hash_a;
    if (hash_b != other.hash_b) return hash_b < other.hash_b;
    return length < other.length;
  }
  __host__ __device__ bool operator==(const UnitKey& other) const {
    return hash_a == other.hash_a && hash_b == other.hash_b && length == other.length;
  }
};

struct BigramKey {
  std::uint32_t previous;
  std::uint32_t next;

  __host__ __device__ bool operator<(const BigramKey& other) const {
    return previous < other.previous || (previous == other.previous && next < other.next);
  }
  __host__ __device__ bool operator==(const BigramKey& other) const {
    return previous == other.previous && next == other.next;
  }
};

struct TrigramKey {
  std::uint32_t first;
  std::uint32_t second;
  std::uint32_t next;

  __host__ __device__ bool operator<(const TrigramKey& other) const {
    if (first != other.first) return first < other.first;
    if (second != other.second) return second < other.second;
    return next < other.next;
  }
  __host__ __device__ bool operator==(const TrigramKey& other) const {
    return first == other.first && second == other.second && next == other.next;
  }
};

struct AssociationKey {
  std::uint32_t first;
  std::uint32_t second;

  __host__ __device__ bool operator<(const AssociationKey& other) const {
    return first < other.first || (first == other.first && second < other.second);
  }
  __host__ __device__ bool operator==(const AssociationKey& other) const {
    return first == other.first && second == other.second;
  }
};

struct LocalSeedCandidate {
  unsigned long long score;
  std::uint32_t position;
  std::uint32_t target;
  std::uint32_t cue_mask;
  std::uint32_t exact_cue_mask;
  std::uint32_t distinctive_cue_mask;
  std::uint32_t anchor_position;
  std::uint32_t anchor_cue;
  std::uint32_t contiguous_run;
  std::uint32_t start_offset;
  std::uint32_t local_launch_end;
  std::uint32_t backward_launch;
  std::uint32_t episode_begin;
  std::uint32_t episode_end;
  std::uint32_t distinctive_coverage;
  std::uint32_t distinctive_total;
};

struct LocalSeedRank {
  std::uint32_t novelty_scope;

  __host__ __device__ bool operator()(const LocalSeedCandidate& left,
                                      const LocalSeedCandidate& right) const {
    if (novelty_scope != 0u) {
      if ((left.score != 0u) != (right.score != 0u)) return left.score != 0u;
      if (left.distinctive_coverage != right.distinctive_coverage) {
        return left.distinctive_coverage > right.distinctive_coverage;
      }
      if (left.score != 0u && left.episode_begin != right.episode_begin) {
        return left.episode_begin > right.episode_begin;
      }
      if (left.contiguous_run != right.contiguous_run) {
        return left.contiguous_run > right.contiguous_run;
      }
    }
    if (left.score != right.score) return left.score > right.score;
    return left.position < right.position;
  }
};

struct WindowSignature {
  std::uint32_t hash_a;
  std::uint32_t hash_b;
  std::uint32_t position;

  __host__ __device__ bool operator<(const WindowSignature& other) const {
    if (hash_a != other.hash_a) return hash_a < other.hash_a;
    if (hash_b != other.hash_b) return hash_b < other.hash_b;
    return position < other.position;
  }
};

struct ResidentEfferenceState {
  std::uint32_t trace_bytes = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t exact_discharges = 0u;
  std::uint32_t mismatched_echoes = 0u;
  unsigned long long discharged_bytes = 0ull;
  unsigned long long mismatch_bytes = 0ull;
  std::uint32_t lesion_count = 0u;
};

struct ResidentInteractionShadowState {
  std::uint32_t trace_bytes = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t staged_contacts = 0u;
  std::uint32_t exact_confirmations = 0u;
  std::uint32_t replaced_contacts = 0u;
  unsigned long long confirmed_bytes = 0ull;
  std::uint32_t lesion_count = 0u;
};

// Per-contact evidence for the only lawful bridge from an ordered relation to
// a learned surface construction. It is part of the normal report so silence
// can be attributed to a failed learned condition, not a hidden debug mode.
// ⛔ Every field here MUST have a writer. Seven dead members were removed
// 2026-08-07 after they were read as measurements three times and cost three
// retracted claims -- their zero was the initialiser. Before adding a field,
// add its writer in the same change; before READING one, grep for a writer.
struct ConstructionAssociationReceipt {
  unsigned long long ordered_events = 0ull;
  unsigned long long ordered_accepted = 0ull;
  unsigned long long ordered_invalid = 0ull;
  unsigned long long ordered_incomplete_population = 0ull;
  unsigned long long ordered_below_recurrence = 0ull;
  unsigned long long ordered_whole_contact_abstained = 0ull;
  unsigned long long ordered_event_overflow = 0ull;
  unsigned long long ordered_capacity_overflow = 0ull;
  unsigned long long ordered_insufficient_mass = 0ull;
};

__global__ void accumulate_ordered_assimilation_receipt_kernel(
    const ordered_relation::AssimilationReceipt* source,
    ConstructionAssociationReceipt* destination) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || source == nullptr ||
      destination == nullptr)
    return;
  atomicAdd(&destination->ordered_events, source->events);
  atomicAdd(&destination->ordered_accepted, source->accepted);
  atomicAdd(&destination->ordered_invalid, source->invalid);
  atomicAdd(&destination->ordered_incomplete_population,
            source->incomplete_population);
  atomicAdd(&destination->ordered_below_recurrence,
            source->below_recurrence);
  atomicAdd(&destination->ordered_whole_contact_abstained,
            source->whole_contact_abstained);
  atomicAdd(&destination->ordered_event_overflow, source->event_overflow);
  atomicAdd(&destination->ordered_capacity_overflow, source->capacity_overflow);
  atomicAdd(&destination->ordered_insufficient_mass, source->insufficient_mass);
}

// SUBJECT ADMISSION CENSUS. A read-only account of WHY the resident subject
// field ended up the size it did. It exists because a zero subject field is
// silent: with subject_count==0 the construction pool has no source, the
// composer returns before its own diagnostics print, and the reply path falls
// through to a mode-2 decline -- a chain whose first link was previously
// unobservable. Each field counts units surviving one more predicate of
// collect_subject_field_deterministic, so the first count that collapses to
// zero NAMES the operative gate.
//
// ⚠ latch_gate_open is recorded SEPARATELY from the predicates, because the
// admission scan and the episode-count gate are different questions: the
// census answers "would any unit have been admitted", the gate answers "was
// the latch even allowed to run".
struct SubjectAdmissionCensus {
  std::uint32_t cue_units_seen = 0u;
  std::uint32_t cue_units_len_ok = 0u;
  std::uint32_t cue_units_nonfunc = 0u;
  std::uint32_t max_nonfunc_len_ok_cue_score = 0u;
  std::uint32_t near_identity_pass_count = 0u;
  std::uint32_t admitted_subject_count = 0u;
  std::uint32_t latch_gate_open = 0u;
  std::uint32_t unit_pos_available = 0u;
  // Recorded at the TOP of condition_on_raw_cue, because the cue path can exit
  // before the subject machinery is even declared: when scoped_episode_count
  // is 0 the exact-resident-Plan branch runs and returns, and `cue_scores`
  // does not exist above that return. Without these two fields a census that
  // never ran is indistinguishable from one that ran and admitted nothing.
  std::uint32_t scoped_episode_count = 0u;
  std::uint32_t exact_plan_branch_taken = 0u;
  std::uint32_t cue_conditioning_entered = 0u;
  std::uint32_t observed = 0u;
};

// Grow-only per-contact scratch behind condition_on_raw_cue. Transient
// workspace by construction: never serialized, and no array's contents are
// read before the contact that filled them re-initializes them.
struct RawCueScratch {
  DeviceArray<std::uint8_t> incoming_bytes;
  DeviceArray<std::uint8_t> device_bytes;
  DeviceArray<std::uint32_t> flags;
  DeviceArray<std::uint32_t> anchors;
  DeviceArray<std::uint32_t> scanned_ids;
  DeviceArray<std::uint32_t> starts;
  DeviceArray<std::uint32_t> best_ids;
  DeviceArray<std::uint32_t> best_scores;
  DeviceArray<std::uint32_t> cue_scores;
  DeviceArray<std::uint32_t> cue_orders;
  DeviceArray<std::uint32_t> route_scores;
  DeviceArray<std::uint32_t> route_orders;
  DeviceArray<std::uint32_t> cue_focus;
  DeviceArray<std::uint32_t> cue_masks;
  DeviceArray<std::uint32_t> salient_masks;
  DeviceArray<unsigned long long> association_mass;
  DeviceArray<unsigned long long> forward_scores;
  DeviceArray<unsigned long long> backward_scores;
  DeviceArray<std::uint32_t> forward_support;
  DeviceArray<std::uint32_t> backward_support;
  DeviceArray<std::uint32_t> forward_salient_support;
  DeviceArray<unsigned long long> strongest_evidence;
  DeviceArray<unsigned long long> second_evidence;
  DeviceArray<unsigned long long> cue_evidence;
  DeviceArray<LocalSeedCandidate> local_seed_candidates;
  DeviceArray<LocalSeedCandidate> clause_seed_candidates;
  DeviceArray<std::uint32_t> alternate_motor_context;
  DeviceArray<std::uint32_t> alternate_motor_completion;
  DeviceArray<std::uint32_t> episode_match_mask;
  DeviceArray<std::uint32_t> episode_exact_match_mask;
  DeviceArray<std::uint32_t> episode_match_spans;
  DeviceArray<std::uint32_t> scoped_episode_breaks;
  DeviceArray<std::uint32_t> matched_segment_counts;
  DeviceArray<std::uint32_t> admitted_segment_mask;
  DeviceArray<std::uint32_t> selected_anchor_ids;
  DeviceArray<unsigned long long> synthesis_cue_activation;
  DeviceArray<unsigned long long> synthesis_propagated_activation;
  DeviceArray<unsigned long long> synthesis_forward_activation;
  DeviceArray<unsigned long long> synthesis_backward_activation;
  DeviceArray<std::uint32_t> synthesis_cue_masks;
  DeviceArray<std::uint32_t> synthesis_salient_masks;
  DeviceArray<std::uint32_t> synthesis_propagated_masks;
  DeviceArray<std::uint32_t> synthesis_propagated_salient_masks;
  DeviceArray<std::uint32_t> synthesis_forward_support;
  DeviceArray<std::uint32_t> synthesis_backward_support;
  DeviceArray<synthesis::ResidentSourceWindowSignature> synthesis_source_windows;
  DeviceArray<std::int32_t> synthesis_role_projection;
  DeviceArray<roles::MutableStructuralRole> synthesis_roles;
  DeviceArray<std::uint64_t> synthesis_base_role_bigrams;
  DeviceArray<std::uint64_t> synthesis_base_role_trigrams;
  DeviceArray<std::uint64_t> synthesis_online_role_bigrams;
  DeviceArray<std::uint64_t> synthesis_online_role_trigrams;
  DeviceArray<std::uint32_t> learned_relation_predicates;
  DeviceArray<std::uint32_t> learned_relation_predicate_count;
  DeviceArray<std::uint32_t> learned_relation_cue_indices;
  DeviceArray<std::uint32_t> learned_relation_cue_scores;
  DeviceArray<std::uint32_t> learned_relation_predicate_owners;
  DeviceArray<std::uint32_t> learned_relation_binding_enabled;
  DeviceArray<std::uint32_t> learned_relation_subject_anchors;
  DeviceArray<std::uint32_t> learned_relation_cue_units;
  DeviceArray<std::uint32_t> learned_relation_cue_count_device;
  DeviceArray<unsigned long long> learned_relation_bindings;
  DeviceArray<unsigned long long> learned_relation_strongest_binding;
  DeviceArray<std::uint32_t> learned_relation_selected_predicate;
  DeviceArray<synthesis::ReverseBigramEdge> reverse_bigrams;
  DeviceArray<synthesis::ReverseTrigramEdge> reverse_trigrams;
  DeviceArray<std::uint32_t> synthesis_output_closure;
  DeviceArray<std::uint32_t> synthesis_seeds;
  DeviceArray<unsigned long long> synthesis_seed_scores;
  DeviceArray<std::uint32_t> synthesis_drafts;
  DeviceArray<std::uint32_t> synthesis_draft_lengths;
  DeviceArray<unsigned long long> synthesis_draft_scores;
  DeviceArray<std::uint32_t> synthesis_selected;
  DeviceArray<synthesis::ResidentSynthesisResult> synthesis_result;
  DeviceArray<std::uint32_t> frame_relation_tail;
  DeviceArray<std::uint32_t> frame_relation_tail_count;
  DeviceArray<std::uint32_t> frame_unit_flags;
  DeviceArray<std::uint32_t> frame_candidates;
  DeviceArray<std::uint32_t> frame_candidate_count_state;
  DeviceArray<answer_frame::SourceWindowSignature> frame_source_windows;
  DeviceArray<std::uint32_t> frame_source_window_count;
  DeviceArray<std::uint64_t> frame_role_bigrams;
  DeviceArray<std::uint64_t> frame_role_trigrams;
  DeviceArray<std::uint32_t> frame_drafts;
  DeviceArray<std::uint32_t> frame_draft_lengths;
  DeviceArray<unsigned long long> frame_draft_scores;
  DeviceArray<std::uint32_t> frame_output_units;
  DeviceArray<std::uint8_t> frame_output_bytes;
  DeviceArray<answer_frame::Result> frame_result;
  DeviceArray<std::uint32_t> role_boundary_evidence;
  DeviceArray<std::uint32_t> role_closure_evidence;
  DeviceArray<unsigned long long> role_cue_activation;
  DeviceArray<std::uint32_t> role_cue_groups;
  DeviceArray<role_compositor::SubjectConditionedRelationTrace> relation_traces;
  DeviceArray<std::uint32_t> relation_trace_units;
  DeviceArray<std::uint32_t> relation_trace_count;
  DeviceArray<std::uint32_t> relation_trace_unit_count;
  DeviceArray<role_compositor::MutableRoleCompositorPolicy> role_policy;
  DeviceArray<unsigned long long> role_trace_scores;
  DeviceArray<std::uint32_t> role_required_counts;
  DeviceArray<std::uint32_t> role_selected_counts;
  DeviceArray<role_compositor::RoleCompositorChoice> role_choice;
  DeviceArray<std::uint32_t> role_ordered_units;
  DeviceArray<std::uint8_t> role_output_bytes;
  DeviceArray<role_compositor::RoleCompositorResult> role_result;
};

struct AdultState {
  DeviceArray<std::uint32_t> boundary_mask;
  DeviceArray<std::uint32_t> boundary_bytes;
  DeviceArray<std::uint32_t> closure_bytes;
  DeviceArray<std::uint32_t> boundary_histogram;
  DeviceArray<std::uint32_t> boundary_pairs;
  DeviceArray<std::uint32_t> unit_lengths;
  DeviceArray<std::uint32_t> unit_content;
  DeviceArray<std::uint32_t> unit_vitality;
  // FUNC/NOUN/VERB/OTHER emergent word-class field. Only the bounded
  // build_unit_pos/FUNC-exclusion slice was extracted from historical commit
  // 473af804d; its bundled proposition and orchestration organs remain
  // excluded. Labels are rebuilt from resident vitality and online bigram
  // distributions when cue subject collection needs them, are mutable and
  // checkpointed, and never name a word class by authored vocabulary.
  DeviceArray<std::uint8_t> unit_pos;
  DeviceArray<std::uint32_t> unit_hash_slots;
  DeviceArray<std::uint32_t> unigram_top_ids;
  DeviceArray<BigramKey> bigrams;
  DeviceArray<std::uint32_t> bigram_counts;
  DeviceArray<TrigramKey> trigrams;
  DeviceArray<std::uint32_t> trigram_counts;
  DeviceArray<std::uint32_t> cached_bigram_contexts;
  DeviceArray<std::uint32_t> cached_bigram_entries;
  DeviceArray<BigramKey> cached_trigram_contexts;
  DeviceArray<std::uint32_t> cached_trigram_entries;
  DeviceArray<std::uint32_t> base_episode_units;
  DeviceArray<std::uint32_t> base_posting_positions;
  DeviceArray<std::uint32_t> base_posting_offsets;
  DeviceArray<WindowSignature> base_window_signatures;
  DeviceArray<BigramKey> online_bigrams;
  DeviceArray<std::uint32_t> online_bigram_counts;
  DeviceArray<TrigramKey> online_trigrams;
  DeviceArray<std::uint32_t> online_trigram_counts;
  DeviceArray<AssociationKey> online_associations;
  DeviceArray<std::uint32_t> online_association_counts;
  DeviceArray<ConditionedTransitionKey> online_conditioned_transitions;
  DeviceArray<std::uint32_t> online_conditioned_transition_counts;
  // Transient physical outlet measurements aligned with the sorted exact-key
  // transition inventory. Counts retain inventory/mass-accounting meaning;
  // every conditioned preference and prediction reads this surface instead.
  // It is rebuilt directly from the resident GPU credit bank.
  DeviceArray<std::uint32_t> online_conditioned_transition_conductance;
  // Parallel to conductance, same index space: the winning route's uncapped exposure
  // (RouteState::exposure) at publish time. Conductance alone cannot break a tie once
  // two candidates both saturate kSupportLimit; consumers that need to break such a
  // tie read this alongside conductance rather than re-deriving it from the bank.
  DeviceArray<std::uint32_t> online_conditioned_transition_exposure;
  DeviceArray<resident_credit::RouteState> conditioned_credit_routes;
  DeviceArray<resident_credit::BankScalars> conditioned_credit_scalars;
  DeviceArray<std::uint32_t> online_episode_units;
  DeviceArray<std::uint32_t> online_episode_breaks;
  DeviceArray<std::uint32_t> mutable_sizes;
  DeviceArray<std::uint32_t> motor_context;
  DeviceArray<std::uint32_t> motor_completion;
  // Read-only account of the last subject-admission decision. Never consumed
  // by any kernel; it is an observable, not a control input.
  SubjectAdmissionCensus subject_admission;
  DeviceArray<std::uint32_t> subject_ids;
  DeviceArray<std::uint32_t> subject_weights;
  DeviceArray<std::uint32_t> subject_count;
  // Resident interrogative construction matter. Two per-unit-slot integer
  // count fields are learned online from the SAME '?'
  // events the corpus itself places (a discovered byte, never an authored
  // rule): qonset_count[slot] counts how often the slot was the FIRST unit
  // of an online question span, qterm_count[slot] how often it was the
  // '?'-fused FINAL unit. qorig_onset/_w/_n and qorig_term/_w/_n are the
  // legacy transient bounded vocabulary lists gathered from those counts
  // only when --qorig is enabled (rebuilt
  // deterministically after each assimilation call, ascending-slot-order
  // host pass -- same idiom as collect_subject_field_deterministic, avoids
  // the atomicAdd-slot-race nondeterminism pattern). Mutable, lesionable,
  // integer, outside conserved vitality. The counts remain available as
  // construction priors even while legacy autonomous injection is disabled.
  DeviceArray<std::uint32_t> qonset_count;
  DeviceArray<std::uint32_t> qterm_count;
  DeviceArray<std::uint64_t> qonset_evidence_revision;
  DeviceArray<std::uint32_t> question_gap_field_support;
  // DIAGNOSTIC, not learned matter: 8 bins counting the `differences` value
  // computed by learn_question_gap_fields_kernel for every evaluated
  // question/answer triple pair (bin 1 is the only one that votes). Deliberately
  // NOT registered in the checkpoint macro list -- it is an instrument reading,
  // not organism state, and must not survive a save/restore.
  DeviceArray<std::uint32_t> question_gap_vote_histogram;
  // DIAGNOSTIC, not learned matter: kRelationFieldCount bins counting WHICH
  // coordinate of the question/answer triple pair the same comparison in
  // learn_question_gap_fields_kernel found unequal, one increment per differing
  // coordinate of every evaluated pair. Deliberately NOT registered in the
  // checkpoint macro list -- an instrument reading, not organism state.
  DeviceArray<std::uint32_t> question_gap_coordinate_histogram;
  DeviceArray<std::uint32_t> question_answer_construction;
  DeviceArray<std::uint32_t> question_answer_construction_support;
  DeviceArray<std::uint32_t> question_answer_slot_mapping;
  DeviceArray<std::uint32_t> qorig_onset;
  DeviceArray<std::uint32_t> qorig_onset_w;
  DeviceArray<std::uint32_t> qorig_onset_n;
  DeviceArray<std::uint32_t> qorig_term;
  DeviceArray<std::uint32_t> qorig_term_w;
  DeviceArray<std::uint32_t> qorig_term_n;
  bool qorig_on = false;  // --qorig opt-in flag; false = zero behavior change (default)
  // Persistent discourse state (BCC32_PROGRESSION): the theme the organism
  // was last developing, plus what it already said about it (cross-reply
  // refractory). Lets a quiet autonomous tick CONTINUE the discourse through
  // the licensed-novel progression channel instead of the n-gram walk.
  std::uint32_t discourse_theme = 0xffffffffu;
  std::vector<std::uint32_t> discourse_said;
  std::vector<std::uint32_t> discourse_front;  // recent active themes, newest last
  // Resident construction/slot composer store (learned, mutable, lesionable).
  DeviceArray<std::int32_t> construction_role_projection;
  DeviceArray<roles::MutableStructuralRole> construction_roles;
  DeviceArray<std::uint32_t> construction_closure_bytes;
  DeviceArray<std::uint32_t> construction_closed_class_mask;
  DeviceArray<std::uint32_t> construction_filler_terminal_mask;
  DeviceArray<std::uint32_t> construction_initial_form_mask;
  std::uint32_t construction_closure_count = 0u;
  DeviceArray<std::uint32_t> construction_role_population;
  DeviceArray<std::uint32_t> construction_tokens;
  DeviceArray<std::uint32_t> construction_lengths;
  DeviceArray<std::uint32_t> construction_slot_counts;
  DeviceArray<std::uint32_t> construction_supports;
  DeviceArray<std::uint32_t> construction_slot_units;
  DeviceArray<std::uint32_t> construction_slot_masses;
  DeviceArray<std::uint32_t> construction_slot_totals;
  DeviceArray<std::uint32_t> construction_slot_overflow;
  DeviceArray<std::uint32_t> construction_hash_slots;
  DeviceArray<std::uint32_t> construction_store_count;
  // Contact provenance is retained beside each learned construction. It is
  // only a same-event witness; it carries no authored linguistic meaning.
  DeviceArray<std::uint64_t> construction_evidence_revision;
  // Immutable acquisition provenance. Later observations may strengthen and
  // update a construction without making its originating contact independent.
  DeviceArray<std::uint64_t> construction_origin_revision;
  DeviceArray<std::uint32_t> construction_pool_units;
  DeviceArray<std::uint32_t> construction_pool_roles;
  DeviceArray<std::uint32_t> construction_pool_weights;
  DeviceArray<std::uint32_t> construction_pool_meta;
  DeviceArray<unsigned long long> construction_best;
  DeviceArray<std::uint32_t> construction_last_selected;
  DeviceArray<std::uint32_t> construction_plan;
  DeviceArray<std::uint32_t> construction_plan_meta;
  std::uint32_t construction_func_threshold = 0u;
  std::uint32_t construction_role_population_cutoff = 0u;
  std::uint32_t construction_count_host = 0u;
  // RETIRED as a causal input. This was a host-incremented counter fed into
  // resident selection as PRNG jitter -- an external RNG and a host causal
  // policy, both forbidden by Section 15. Selection now derives its variety
  // seed on-device from last_selected and the store size. The word is kept
  // only because AdultScalarState is written to disk by layout and is pinned
  // at 176 bytes; removing it is a checkpoint format bump, not this change.
  std::uint32_t retired_construction_nonce = 0u;
  bool construction_lesioned = false;
  // Learned byte-suffix-class adjacency counts (morphological agreement
  // signal). Accumulated on-device from the assimilated stream; read only by
  // the composer's bind step. Lesionable via BCC32_LESION_MORPH_AGREEMENT.
  DeviceArray<std::uint32_t> construction_suffix_transitions;
  bool morph_agreement_lesioned = false;
  // Role-signal canonicalization (unit -> canonical representative), built
  // ON-DEVICE from unit_content before each construction-role derivation so
  // variant-fragmented forms pool their distributional context into one
  // robust role. Read only by the role projection; the surface form of every
  // unit is untouched. Lesionable via BCC32_LESION_ROLE_CANON (nullptr map
  // restores exact legacy per-variant roles).
  DeviceArray<std::uint32_t> construction_role_canon;
  DeviceArray<unsigned long long> construction_role_canon_signatures;
  DeviceArray<unsigned long long> construction_role_canon_keys;
  DeviceArray<std::uint32_t> construction_role_canon_reps;
  std::uint32_t construction_role_canon_table_size = 0u;
  bool role_canon_lesioned = false;
  // Neighbor-entropy glue partition matter. The glue-vs-content decision is
  // made by CONTEXT DISPERSION (mean left/right neighbor-entropy per
  // canonical group, computed on-device from the resident bigram counts),
  // not by frequency rank: frequency wrongly absorbs high-frequency CONTENT
  // ("work", "economic") into the glue set and starves the composer's
  // content-slot pools. Lesionable via BCC32_LESION_ENTROPY_GLUE -> the
  // legacy frequency-rank mask stands.
  bool entropy_glue_lesioned = false;
  // Whole-reply CONTENT COMMITMENT: an ordered resident content plan
  // (units + [length, cursor] meta) formed on the question BEFORE any frame
  // selection, from the subject field + online association mass. The
  // composer only renders it in order. Lesionable via
  // BCC32_LESION_CONTENT_COMMIT -> exact pool-driven slot filling.
  DeviceArray<std::uint32_t> content_commitment_units;
  DeviceArray<std::uint32_t> content_commitment_meta;
  bool content_commit_lesioned = false;
  // RELATIONAL-TRIPLE CHANNEL matter: resident (A, connective, B) typed-
  // triple hash table + counts learned from the assimilation stream (the
  // connective inventory IS the discovered glue partition), plus the
  // propositional plan/meta and retrieval scratch. Lesionable via
  // BCC32_LESION_RELATION_TRIPLE -> learning and commitment both skipped,
  // exact prior commitment behavior.
  DeviceArray<construction::RelationTriple> relation_triples;
  DeviceArray<std::uint32_t> relation_triple_counts;
  // Role-pattern cloud: subject participation in recurrent connective
  // structures, independent of the particular observed value word.
  DeviceArray<construction::RelationRole> relation_roles;
  DeviceArray<std::uint32_t> relation_role_counts;
  // Exact acquisition revision for the retained aggregate slot. This is the
  // only lawful bridge to a learned construction witness; it is not inferred
  // from similarity at readout.
  DeviceArray<std::uint64_t> relation_triple_evidence_revision;
  DeviceArray<construction::WitnessedRelationEvent> witnessed_relation_events;
  DeviceArray<std::uint32_t> witnessed_relation_event_cursor;
  // Exact event-to-surface identity. A construction's latest evidence
  // revision is mutable aggregate metadata and cannot identify every event
  // that acquired the same learned skeleton.
  DeviceArray<std::uint32_t> witnessed_relation_constructions;
  DeviceArray<std::uint32_t> witnessed_relation_surface_units;
  DeviceArray<std::uint32_t> witnessed_relation_surface_counts;
  DeviceArray<std::uint32_t> relation_triple_type_total;
  DeviceArray<std::uint32_t> relation_triple_type_mirrored;
  DeviceArray<std::uint32_t> relation_triple_candidates;
  DeviceArray<std::uint32_t> relation_triple_cursor;
  // Lifetime totals of the insert channel. The store's summed counts record
  // only what landed; a triple that exhausts the probe window leaves no trace
  // in the table at all, so its loss is only visible here.
  DeviceArray<std::uint32_t> relation_triple_attempted;
  DeviceArray<std::uint32_t> relation_triple_drops;
  // Support carried by the triples the question-time substitution probe
  // actually reaches. The probed subset is gated on its source slot, so it
  // is not a sample of the store's global count distribution.
  DeviceArray<std::uint32_t> relation_probe_support_histogram;
  DeviceArray<std::uint32_t> relation_probe_total;
  // The same substituted lookup taken over the WHOLE occupied store, with
  // the source-slot count gate absent. Observer only: nothing downstream
  // reads these, they exist so the gated probe has a comparison arm, and
  // source_singletons says how much of the store the gate withholds.
  DeviceArray<std::uint32_t> relation_census_histogram;
  DeviceArray<std::uint32_t> relation_census_total;
  DeviceArray<std::uint32_t> relation_census_source_singletons;
  // The census counts how often substituting the topic finds support. That
  // alone cannot separate a relational store from an episodic membership
  // index: a store whose (K,V) pairs are merely common returns support for
  // ANY subject. These carry the matched-counterfactual arm -- same slot,
  // same (K,K2,V), varying only which subject is substituted -- so a run can
  // say whether the topic is doing any work at all. Observer only.
  DeviceArray<std::uint32_t> relation_counterfactual_histogram;
  DeviceArray<std::uint32_t> relation_counterfactual_total;
  DeviceArray<std::uint32_t> relation_topic_strictly_greater;
  DeviceArray<std::uint32_t> relation_counterfactual_strictly_greater;
  DeviceArray<std::uint32_t> relation_support_ties;
  DeviceArray<std::uint32_t> relation_triple_plan;
  DeviceArray<std::uint32_t> relation_triple_meta;
  // Persisted copy of the CURRENT cue's near-identity unit scores (written
  // by condition_on_raw_cue) so the triple channel can read which glue
  // words the question itself said.
  DeviceArray<std::uint32_t> relation_cue_scores;
  DeviceArray<std::uint32_t> relation_cue_orders;
  DeviceArray<std::uint32_t> relation_cue_exact;
  DeviceArray<std::uint32_t> proposition_cue_sequence;
  DeviceArray<std::uint32_t> proposition_cue_sequence_count;
  DeviceArray<std::uint32_t> relation_operator_order;
  // Raw body-surface continuity across transport frames. meta =
  // {byte_count, closure_ready, overflow}. The host may schedule from these
  // counts but never interprets the bytes or selects content.
  DeviceArray<std::uint8_t> streaming_cue_bytes;
  DeviceArray<std::uint32_t> streaming_cue_meta;
  RawCueScratch raw_cue_scratch;
  bool streaming_cue_mode = false;
  bool relation_triple_lesioned = false;
  DeviceArray<std::uint32_t> ledger;
  DeviceArray<std::uint32_t> rng;
  DeviceArray<std::uint8_t> efference_trace;
  DeviceArray<ResidentEfferenceState> efference_state;
  DeviceArray<std::uint8_t> interaction_shadow_trace;
  DeviceArray<ResidentInteractionShadowState> interaction_shadow_state;
  DeviceArray<bcc32_cuda_resident_synthesis::ResidentSynthesisPolicyState>
      synthesis_policy;
  DeviceArray<answer_frame::MutablePolicyState> answer_frame_policy;
  DeviceArray<answer_frame::MutableSelectionState> answer_frame_selection;
  DeviceArray<std::uint32_t> distributed_motor_mass;
  DeviceArray<std::uint32_t> distributed_motor_cell_support;
  DeviceArray<std::uint32_t> distributed_motor_support;
  DeviceArray<std::uint32_t> distributed_binding_keys;
  DeviceArray<std::uint32_t> distributed_binding_mass;
  DeviceArray<std::uint32_t> distributed_binding_support;
  DeviceArray<std::uint32_t> distributed_enabled;
  DeviceArray<std::uint8_t> distributed_history;
  DeviceArray<std::uint32_t> distributed_previous_active;
  DeviceArray<std::uint32_t> distributed_sequence_active;
  DeviceArray<std::uint32_t> distributed_current_active;
  DeviceArray<std::uint32_t> distributed_cue_active;
  DeviceArray<unsigned long long> distributed_completion_scores;
  DeviceArray<unsigned long long> distributed_motor_scores;
  DeviceArray<std::uint32_t> distributed_candidate_cells;
  DeviceArray<unsigned long long> distributed_candidate_scores;
  DeviceArray<std::uint32_t> distributed_path_cells;
  DeviceArray<std::uint32_t> distributed_path_phases;
  DeviceArray<std::uint8_t> distributed_output_tape;
  DeviceArray<std::uint32_t> distributed_scalars;
  DeviceArray<unsigned long long> distributed_mass_scalars;
  DeviceArray<std::int32_t> surface_role_projection;
  DeviceArray<roles::MutableStructuralRole> surface_roles;
  DeviceArray<std::uint32_t> surface_unit_population;
  DeviceArray<std::uint32_t> surface_unit_context_population;
  DeviceArray<std::uint16_t> surface_population_context_mass;
  DeviceArray<unsigned long long> surface_unit_activity;
  DeviceArray<std::uint32_t> surface_unit_phase;
  DeviceArray<std::uint32_t> surface_projection_state;
  DeviceArray<std::uint64_t> surface_unit_mass;
  DeviceArray<std::uint64_t> surface_unit_start_mass;
  DeviceArray<std::uint64_t> surface_unit_end_mass;
  DeviceArray<std::uint64_t> surface_role_mass;
  DeviceArray<std::uint64_t> surface_role_start_mass;
  DeviceArray<std::uint64_t> surface_role_end_mass;
  DeviceArray<std::uint64_t> surface_role_bigram_mass;
  DeviceArray<std::uint64_t> surface_role_bigram_context_mass;
  DeviceArray<std::uint64_t> surface_role_trigram_mass;
  DeviceArray<std::uint64_t> surface_role_trigram_context_mass;
  DeviceArray<surface_organ::SurfaceGrammarStats> surface_stats;
  DeviceArray<context_state::ContextCell> surface_context_cells;
  DeviceArray<context_state::MembershipCell> surface_context_memberships;
  DeviceArray<context_state::TransitionCell> surface_context_transitions;
  DeviceArray<context_state::UnitBinding> surface_context_bindings;
  DeviceArray<context_state::FieldScalars> surface_context_scalars;
  DeviceArray<std::uint64_t> surface_context_primary_ranks;
  DeviceArray<std::uint64_t> surface_context_alternate_ranks;
  DeviceArray<std::uint32_t> surface_context_state_counts;
  DeviceArray<surface_organ::SurfaceRolePathChoice> surface_bridges;
  DeviceArray<surface_organ::SurfaceRolePathChoice> surface_prefixes;
  DeviceArray<surface_organ::SurfaceRolePathChoice> surface_suffixes;
  DeviceArray<std::uint64_t> surface_permutation_scores;
  DeviceArray<std::uint32_t> surface_permutation_valid;
  DeviceArray<std::uint32_t> surface_output_units;
  DeviceArray<std::uint32_t> surface_output_anchor_mask;
  DeviceArray<std::uint8_t> surface_output_bytes;
  DeviceArray<surface_organ::SurfaceOrganResult> surface_result;
  DeviceArray<proposition_tissue::SparseBindingSynapse> proposition_synapses;
  DeviceArray<proposition_tissue::PopulationCohortEvidence> proposition_cohorts;
  DeviceArray<proposition_tissue::OrderedRoleBindingEvidence>
      proposition_ordered_bindings;
  // A construction witness is linked only when the learned slot populations
  // uniquely contain all retained identities of an ordered binding.  It is an
  // opaque co-acquisition record, not a semantic label or surface rule.
  DeviceArray<std::uint32_t> proposition_ordered_construction;
  DeviceArray<ConstructionAssociationReceipt> proposition_construction_association;
  DeviceArray<proposition_tissue::OrderedBindingAdmissionState>
      proposition_ordered_binding_admission;
  DeviceArray<std::uint64_t> proposition_ordered_evidence_revision;
  DeviceArray<proposition_tissue::TissueScalars> proposition_scalars;
  DeviceArray<std::uint64_t> proposition_completion_scores;
  DeviceArray<std::uint32_t> proposition_output_cells;
  DeviceArray<std::uint64_t> proposition_output_scores;
  DeviceArray<proposition_tissue::CompletionResult> proposition_completion_result;
  // Persistent information-gap commitment. Proposition tissue chooses its
  // opaque target; learned question constructions may realize it but never
  // write or replace its content.
  DeviceArray<question_goal::ResidentQuestionGoalState> question_goal_state;

  std::uint32_t unit_occurrences = 0u;
  std::uint32_t unit_count = 0u;
  std::uint32_t bigram_count = 0u;
  std::uint32_t trigram_count = 0u;
  std::uint32_t cached_bigram_count = 0u;
  std::uint32_t cached_trigram_count = 0u;
  std::uint32_t base_window_count = 0u;
  std::uint32_t unit_capacity = 0u;
  std::uint32_t surface_content_begin = 0u;
  std::uint32_t unit_hash_capacity = 0u;
  std::uint32_t online_bigram_count = 0u;
  std::uint32_t online_trigram_count = 0u;
  std::uint32_t online_association_count = 0u;
  std::uint32_t online_conditioned_transition_count = 0u;
  std::uint32_t online_episode_count = 0u;
  std::uint32_t online_episode_break_count = 0u;
  std::uint32_t novelty_episode_begin = 0u;
  std::uint32_t novelty_episode_break_begin = 0u;
  std::size_t resident_bytes = 0u;
  bool transitions_lesioned = false;
  bool base_episode_lesioned = false;
  bool novelty_epoch_active = false;
  bool novelty_epoch_pending = false;
  bool frame_emit = false;  // --frame-emit: multi-clause relation-frame emission (substrate generator)
  bool distributed_motor_enabled = false;
  bool surface_organ_enabled = false;
};

inline resident_credit::BankView resident_credit_bank_view(AdultState& state) {
  return resident_credit::BankView{
      state.conditioned_credit_routes.get(),
      state.conditioned_credit_scalars.get(),
      static_cast<std::uint32_t>(state.conditioned_credit_routes.size()),
      resident_credit::kSupportLimit};
}

inline proposition_tissue::TissueView resident_proposition_tissue_view(
    AdultState& state) {
  return {state.proposition_synapses.get(),
          static_cast<std::uint32_t>(state.proposition_synapses.size()),
          kDistributedMotorPopulation, state.proposition_scalars.get(),
          state.proposition_cohorts.get(),
          static_cast<std::uint32_t>(state.proposition_cohorts.size()),
          state.proposition_ordered_bindings.get(),
          static_cast<std::uint32_t>(state.proposition_ordered_bindings.size()),
          state.proposition_ordered_binding_admission.get()};
}

inline proposition_tissue::CompletionWorkspaceView
resident_proposition_workspace_view(AdultState& state) {
  return {state.proposition_completion_scores.get(),
          static_cast<std::uint32_t>(state.proposition_completion_scores.size()),
          state.proposition_output_cells.get(), state.proposition_output_scores.get(),
          static_cast<std::uint32_t>(state.proposition_output_cells.size())};
}

inline void originate_resident_bridge_gap(
    AdultState& state, proposition_tissue::SparsePopulationView topic,
    question_goal::OriginationResult* device_result = nullptr) {
  auto proposition_view = resident_proposition_tissue_view(state);
  auto topic_view = topic;
  auto* goal_state = state.question_goal_state.get();
  auto* result = device_result;
  void* arguments[] = {&proposition_view, &topic_view, &goal_state, &result};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(
                       question_goal::originate_information_gap_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "originate resident bridge-gap goal");
}

inline void refresh_resident_question_goal_after_contact(AdultState& state) {
  auto proposition_view = resident_proposition_tissue_view(state);
  auto* goal_state = state.question_goal_state.get();
  void* arguments[] = {&proposition_view, &goal_state};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(
                       question_goal::discharge_question_goal_from_tissue_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "refresh resident bridge-gap goal");
}

inline std::size_t resident_proposition_bytes(const AdultState& state) {
  return state.proposition_synapses.bytes() + state.proposition_cohorts.bytes() +
         state.proposition_ordered_bindings.bytes() +
         state.proposition_ordered_construction.bytes() +
         state.proposition_construction_association.bytes() +
         state.proposition_ordered_binding_admission.bytes() +
         state.proposition_ordered_evidence_revision.bytes() +
         state.proposition_scalars.bytes() +
         state.proposition_completion_scores.bytes() +
         state.proposition_output_cells.bytes() +
         state.proposition_output_scores.bytes() +
         state.proposition_completion_result.bytes() +
         state.question_goal_state.bytes() +
         state.proposition_cue_sequence.bytes() +
         state.proposition_cue_sequence_count.bytes();
}

inline void allocate_resident_question_goal(AdultState& state) {
  state.question_goal_state.allocate(1u);
  auto* goal_state = state.question_goal_state.get();
  void* arguments[] = {&goal_state};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(
                       question_goal::initialize_question_goal_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "initialize resident information-gap goal");
}

inline void allocate_resident_ordered_binding_tissue(AdultState& state) {
  state.proposition_ordered_bindings.allocate(kResidentOrderedBindingCapacity);
  state.proposition_ordered_construction.allocate(kResidentOrderedBindingCapacity);
  state.proposition_construction_association.allocate(1u);
  state.proposition_ordered_binding_admission.allocate(1u);
  state.proposition_ordered_evidence_revision.allocate(1u);
  cuda_require(cudaMemset(state.proposition_ordered_bindings.get(), 0,
                          state.proposition_ordered_bindings.bytes()),
               "clear resident ordered bindings");
  cuda_require(cudaMemset(state.proposition_ordered_construction.get(), 0xff,
                          state.proposition_ordered_construction.bytes()),
               "clear resident ordered construction links");
  cuda_require(cudaMemset(state.proposition_construction_association.get(), 0,
                          state.proposition_construction_association.bytes()),
               "clear resident construction association receipt");
  cuda_require(cudaMemset(state.proposition_ordered_binding_admission.get(), 0,
                          state.proposition_ordered_binding_admission.bytes()),
               "clear resident ordered binding admission");
  cuda_require(cudaMemset(state.proposition_ordered_evidence_revision.get(), 0,
                          state.proposition_ordered_evidence_revision.bytes()),
               "clear resident ordered binding evidence revision");
}

// Old checkpoints did not serialize the relation-to-construction bridge. A
// migration may create an empty bridge, but must never fabricate witnesses.
inline std::size_t ensure_ordered_construction_links(AdultState& state) {
  if (state.proposition_ordered_construction.get() != nullptr)
    return 0u;
  state.proposition_ordered_construction.allocate(kResidentOrderedBindingCapacity);
  cuda_require(cudaMemset(state.proposition_ordered_construction.get(), 0xff,
                          state.proposition_ordered_construction.bytes()),
               "initialize restored ordered construction links");
  return state.proposition_ordered_construction.bytes();
}

inline std::size_t ensure_construction_evidence_revisions(AdultState& state) {
  if (state.construction_evidence_revision.get() != nullptr)
    return 0u;
  state.construction_evidence_revision.allocate(construction::kConstructionCap);
  cuda_require(cudaMemset(state.construction_evidence_revision.get(), 0,
                          state.construction_evidence_revision.bytes()),
               "initialize restored construction evidence revisions");
  return state.construction_evidence_revision.bytes();
}

inline std::size_t ensure_construction_origin_revisions(AdultState& state) {
  if (state.construction_origin_revision.get() != nullptr)
    return 0u;
  state.construction_origin_revision.allocate(construction::kConstructionCap);
  if (state.construction_evidence_revision.get() != nullptr &&
      state.construction_evidence_revision.size() >=
          construction::kConstructionCap)
    cuda_require(cudaMemcpy(state.construction_origin_revision.get(),
                            state.construction_evidence_revision.get(),
                            state.construction_origin_revision.bytes(),
                            cudaMemcpyDeviceToDevice),
                 "migrate immutable construction origins");
  else
    cuda_require(cudaMemset(state.construction_origin_revision.get(), 0,
                            state.construction_origin_revision.bytes()),
                 "initialize restored construction origins");
  return state.construction_origin_revision.bytes();
}

inline std::size_t ensure_qonset_evidence_revisions(AdultState& state) {
  if (state.qonset_evidence_revision.get() != nullptr)
    return 0u;
  state.qonset_evidence_revision.allocate(state.unit_capacity);
  cuda_require(cudaMemset(state.qonset_evidence_revision.get(), 0,
                          state.qonset_evidence_revision.bytes()),
               "initialize restored question-onset evidence revisions");
  return state.qonset_evidence_revision.bytes();
}

inline std::size_t ensure_question_gap_field_support(AdultState& state) {
  // Diagnostic histogram rides along on the same restore path so the instrument
  // exists after a checkpoint load too. Its bytes are deliberately NOT added to
  // the returned resident-byte total: it is not resident matter.
  if (state.question_gap_vote_histogram.get() == nullptr) {
    state.question_gap_vote_histogram.allocate(8u);
    cuda_require(cudaMemset(state.question_gap_vote_histogram.get(), 0,
                            state.question_gap_vote_histogram.bytes()),
                 "initialize restored question-gap vote histogram");
  }
  if (state.question_gap_coordinate_histogram.get() == nullptr) {
    state.question_gap_coordinate_histogram.allocate(
        construction::kRelationFieldCount);
    cuda_require(cudaMemset(state.question_gap_coordinate_histogram.get(), 0,
                            state.question_gap_coordinate_histogram.bytes()),
                 "initialize restored question-gap coordinate histogram");
  }
  if (state.question_gap_field_support.get() != nullptr)
    return 0u;
  state.question_gap_field_support.allocate(
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount);
  cuda_require(cudaMemset(state.question_gap_field_support.get(), 0,
                          state.question_gap_field_support.bytes()),
               "initialize restored question-gap field support");
  return state.question_gap_field_support.bytes();
}

inline std::size_t ensure_question_answer_constructions(AdultState& state) {
  if (state.question_answer_construction.get() != nullptr &&
      state.question_answer_construction_support.get() != nullptr &&
      state.question_answer_slot_mapping.get() != nullptr)
    return 0u;
  const std::size_t extent =
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount *
      construction::kQuestionAnswerArityCount;
  state.question_answer_construction.allocate(extent);
  state.question_answer_construction_support.allocate(extent);
  state.question_answer_slot_mapping.allocate(extent);
  cuda_require(cudaMemset(state.question_answer_construction.get(), 0xff,
                          state.question_answer_construction.bytes()),
               "initialize restored question-answer constructions");
  cuda_require(cudaMemset(state.question_answer_construction_support.get(), 0,
                          state.question_answer_construction_support.bytes()),
               "initialize restored question-answer construction support");
  cuda_require(cudaMemset(state.question_answer_slot_mapping.get(), 0,
                          state.question_answer_slot_mapping.bytes()),
               "initialize restored question-answer slot mappings");
  return state.question_answer_construction.bytes() +
         state.question_answer_construction_support.bytes() +
         state.question_answer_slot_mapping.bytes();
}

// Checkpoints before the retained relation-event sidecar cannot supply an
// exact witness. Restore an empty sidecar rather than reconstructing one from
// aggregate counts; later contacts populate it through the normal extractor.
inline std::size_t ensure_relation_triple_evidence_revisions(AdultState& state) {
  if (state.relation_triple_evidence_revision.get() != nullptr)
    return 0u;
  state.relation_triple_evidence_revision.allocate(
      construction::kRelationTripleHashCap);
  cuda_require(cudaMemset(state.relation_triple_evidence_revision.get(), 0,
                          state.relation_triple_evidence_revision.bytes()),
               "initialize restored relation evidence revisions");
  return state.relation_triple_evidence_revision.bytes();
}

// Schema-v19 checkpoints predate the role-pattern cloud.  Migration creates
// an empty resident organization; it never infers role meaning from the exact
// relation table, because that would turn an observer-side reconstruction into
// learned authority.  New contacts rebuild it through the ordinary extractor.
inline std::size_t ensure_relation_role_contexts(AdultState& state) {
  if (state.relation_roles.get() != nullptr &&
      state.relation_role_counts.get() != nullptr)
    return 0u;
  state.relation_roles.allocate(construction::kRelationRoleHashCap);
  state.relation_role_counts.allocate(construction::kRelationRoleHashCap);
  cuda_require(cudaMemset(state.relation_roles.get(), 0xff,
                          state.relation_roles.bytes()),
               "initialize restored relation role cloud");
  cuda_require(cudaMemset(state.relation_role_counts.get(), 0,
                          state.relation_role_counts.bytes()),
               "initialize restored relation role cloud counts");
  return state.relation_roles.bytes() + state.relation_role_counts.bytes();
}

// A short question must be able to reach the resident composer once the adult
// has actually formed reusable construction and role matter.  Episode volume
// is not evidence of that formation: a one-sentence cue can arrive after a
// large adult, and a large raw stream can remain structurally singleton.
inline bool resident_construction_admission_open(AdultState& state) {
  if (state.construction_lesioned || state.relation_triple_lesioned ||
      state.construction_count_host == 0u ||
      state.relation_role_counts.get() == nullptr)
    return false;
  DeviceArray<std::uint32_t> occupied(1u);
  cuda_require(cudaMemset(occupied.get(), 0, occupied.bytes()),
               "clear resident relation-role admission census");
  construction::count_relation_role_evidence_kernel<<<
      (construction::kRelationRoleHashCap + kBlock - 1u) / kBlock, kBlock>>>(
      state.relation_role_counts.get(), occupied.get());
  cuda_require(cudaGetLastError(), "count resident relation-role evidence");
  std::uint32_t count = 0u;
  cuda_require(cudaMemcpy(&count, occupied.get(), sizeof(count),
                          cudaMemcpyDeviceToHost),
               "read resident relation-role admission census");
  return count != 0u;
}

inline std::size_t ensure_witnessed_relation_events(AdultState& state) {
  if (state.witnessed_relation_events.get() != nullptr &&
      state.witnessed_relation_event_cursor.get() != nullptr)
    return 0u;
  state.witnessed_relation_events.allocate(
      construction::kWitnessedRelationEventCap);
  state.witnessed_relation_event_cursor.allocate(1u);
  cuda_require(cudaMemset(state.witnessed_relation_events.get(), 0,
                          state.witnessed_relation_events.bytes()),
               "initialize restored witnessed relation events");
  cuda_require(cudaMemset(state.witnessed_relation_event_cursor.get(), 0,
                          state.witnessed_relation_event_cursor.bytes()),
               "initialize restored witnessed relation event cursor");
  return state.witnessed_relation_events.bytes() +
         state.witnessed_relation_event_cursor.bytes();
}

inline std::size_t ensure_witnessed_relation_constructions(AdultState& state) {
  if (state.witnessed_relation_constructions.get() != nullptr)
    return 0u;
  state.witnessed_relation_constructions.allocate(
      construction::kWitnessedRelationEventCap);
  cuda_require(cudaMemset(state.witnessed_relation_constructions.get(), 0xff,
                          state.witnessed_relation_constructions.bytes()),
               "initialize restored witnessed relation constructions");
  return state.witnessed_relation_constructions.bytes();
}

inline std::size_t ensure_witnessed_relation_surfaces(AdultState& state) {
  if (state.witnessed_relation_surface_units.get() != nullptr &&
      state.witnessed_relation_surface_counts.get() != nullptr)
    return 0u;
  state.witnessed_relation_surface_units.allocate(
      construction::kWitnessedRelationEventCap *
      construction::kConstructionMaxTokens);
  state.witnessed_relation_surface_counts.allocate(
      construction::kWitnessedRelationEventCap);
  cuda_require(cudaMemset(state.witnessed_relation_surface_units.get(), 0xff,
                          state.witnessed_relation_surface_units.bytes()),
               "initialize restored witnessed relation surface units");
  cuda_require(cudaMemset(state.witnessed_relation_surface_counts.get(), 0,
                          state.witnessed_relation_surface_counts.bytes()),
               "initialize restored witnessed relation surface counts");
  return state.witnessed_relation_surface_units.bytes() +
         state.witnessed_relation_surface_counts.bytes();
}

// Association receipts are transient observer matter. Schemas before v19
// predate this field, so migration creates a fresh zeroed receipt rather than
// manufacturing a learned value or suppressing the measurement. Current
// checkpoints serialize it for exact failure-atomic replay.
inline std::size_t ensure_construction_association_receipt(AdultState& state) {
  if (state.proposition_construction_association.get() != nullptr)
    return 0u;
  state.proposition_construction_association.allocate(1u);
  cuda_require(cudaMemset(state.proposition_construction_association.get(), 0,
                          state.proposition_construction_association.bytes()),
               "initialize restored construction association receipt");
  return state.proposition_construction_association.bytes();
}

inline void allocate_resident_proposition_tissue(AdultState& state) {
  state.proposition_synapses.allocate(kResidentPropositionSynapseCapacity);
  state.proposition_cohorts.allocate(
      kResidentPropositionSynapseCapacity * proposition_tissue::kMaximumCohortsPerSynapse);
  allocate_resident_ordered_binding_tissue(state);
  state.proposition_scalars.allocate(1u);
  state.proposition_completion_scores.allocate(kDistributedMotorPopulation);
  state.proposition_output_cells.allocate(kResidentPropositionOutputCapacity);
  state.proposition_output_scores.allocate(kResidentPropositionOutputCapacity);
  state.proposition_completion_result.allocate(1u);
  allocate_resident_question_goal(state);
  auto proposition_view = resident_proposition_tissue_view(state);
  std::uint32_t initialization_items = proposition_view.synapse_capacity;
  if (proposition_view.cohort_capacity > initialization_items)
    initialization_items = proposition_view.cohort_capacity;
  if (proposition_view.ordered_binding_capacity > initialization_items)
    initialization_items = proposition_view.ordered_binding_capacity;
  const dim3 initialization_block{256u, 1u, 1u};
  const dim3 initialization_grid{
      initialization_items == 0u
          ? 1u
          : (initialization_items + initialization_block.x - 1u) / initialization_block.x,
      1u, 1u};
  std::uint64_t initial_mass = kResidentPropositionMassBudget;
  void* initialization_arguments[] = {&proposition_view, &initial_mass};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(proposition_tissue::initialize_tissue_kernel),
                   initialization_grid, initialization_block, initialization_arguments, 0u,
                   nullptr),
               "initialize resident proposition tissue");
  cuda_require(cudaMemset(state.proposition_completion_scores.get(), 0,
                          state.proposition_completion_scores.bytes()),
               "clear resident proposition completion scores");
  cuda_require(cudaMemset(state.proposition_output_cells.get(), 0,
                          state.proposition_output_cells.bytes()),
               "clear resident proposition output cells");
  cuda_require(cudaMemset(state.proposition_output_scores.get(), 0,
                          state.proposition_output_scores.bytes()),
               "clear resident proposition output scores");
  cuda_require(cudaMemset(state.proposition_completion_result.get(), 0,
                          state.proposition_completion_result.bytes()),
               "clear resident proposition completion result");
  cuda_require(cudaGetLastError(), "initialize resident proposition tissue");
}

inline void assimilate_resident_proposition_contact(
    AdultState& state, proposition_tissue::SparsePopulationView earlier,
    proposition_tissue::SparsePopulationView later,
    proposition_tissue::SparsePopulationView context, bool efferent_contact = false,
    std::int32_t efferent_polarity = 0, bool outcome_present = true) {
  auto proposition_view = resident_proposition_tissue_view(state);
  auto earlier_view = earlier;
  auto later_view = later;
  auto context_view = context;
  std::uint32_t contact_word = efferent_contact ? 1u : 0u;
  std::int32_t polarity = efferent_polarity;
  std::uint32_t outcome_word = outcome_present ? 1u : 0u;
  void* arguments[] = {&proposition_view, &earlier_view, &later_view, &context_view,
                       &contact_word, &polarity, &outcome_word};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(
                       proposition_tissue::assimilate_experience_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "assimilate resident proposition contact");
}

inline void assimilate_resident_candidate_episodes(
    AdultState& state, const std::uint32_t* sequence,
    const std::uint32_t* segment_ids, std::uint32_t sequence_count,
    bool efferent_contact = false, std::int32_t efferent_polarity = 0,
    bool outcome_present = true) {
  if (sequence_count < 2u || state.proposition_synapses.get() == nullptr ||
      !state.surface_organ_enabled ||
      state.surface_unit_population.get() == nullptr)
    return;
  proposition_tissue::assimilate_candidate_episodes_warp_kernel<<<1u, 32u>>>(
      resident_proposition_tissue_view(state), sequence, segment_ids,
      sequence_count, efferent_contact ? 1u : 0u, efferent_polarity,
      outcome_present ? 1u : 0u, state.surface_unit_population.get(),
      state.unit_count, kDistributedMotorActiveWidth);
  cuda_require(cudaGetLastError(),
               "assimilate resident proposition candidate episodes");
}

inline proposition_tissue::CompletionResult complete_resident_proposition(
    AdultState& state, proposition_tissue::SparsePopulationView cue,
    bool reciprocal = false) {
  auto proposition_view = resident_proposition_tissue_view(state);
  auto cue_view = cue;
  std::uint32_t reciprocal_word = reciprocal ? 1u : 0u;
  auto workspace_view = resident_proposition_workspace_view(state);
  auto* completion_output = state.proposition_completion_result.get();
  void* arguments[] = {&proposition_view, &cue_view, &reciprocal_word,
                       &workspace_view, &completion_output};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(proposition_tissue::settle_completion_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "launch resident proposition completion");
  proposition_tissue::CompletionResult result{};
  cuda_require(cudaMemcpy(&result, state.proposition_completion_result.get(),
                          sizeof(result), cudaMemcpyDeviceToHost),
               "read resident proposition completion");
  return result;
}

inline proposition_tissue::CompletionResult complete_resident_discourse_proposition(
    AdultState& state, proposition_tissue::SparsePopulationView cue,
    bool reciprocal = false) {
  auto proposition_view = resident_proposition_tissue_view(state);
  auto cue_view = cue;
  std::uint32_t reciprocal_word = reciprocal ? 1u : 0u;
  auto workspace_view = resident_proposition_workspace_view(state);
  auto* completion_output = state.proposition_completion_result.get();
  void* arguments[] = {&proposition_view, &cue_view, &reciprocal_word,
                       &workspace_view, &completion_output};
  cuda_require(cudaLaunchKernel(
                   reinterpret_cast<const void*>(
                       proposition_tissue::settle_discourse_completion_kernel),
                   dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, arguments, 0u, nullptr),
               "launch resident discourse proposition completion");
  proposition_tissue::CompletionResult result{};
  cuda_require(cudaMemcpy(&result, state.proposition_completion_result.get(),
                          sizeof(result), cudaMemcpyDeviceToHost),
               "read resident discourse proposition completion");
  return result;
}

inline distributed_motor::DeviceStateView distributed_motor_view(AdultState& state) {
  return distributed_motor::DeviceStateView{
      kDistributedMotorPopulation,
      kDistributedMotorActiveWidth,
      kDistributedMotorWindowBytes,
      kDistributedMotorScratchSteps,
      state.distributed_motor_mass.get(),
      state.distributed_motor_cell_support.get(),
      state.distributed_motor_support.get(),
      state.distributed_binding_keys.get(),
      state.distributed_binding_mass.get(),
      state.distributed_binding_support.get(),
      state.distributed_enabled.get(),
      state.distributed_mass_scalars.get(),
      state.distributed_history.get(),
      state.distributed_scalars.get(),
      state.distributed_previous_active.get(),
      state.distributed_scalars.get() + 1u,
      state.distributed_sequence_active.get(),
      state.distributed_cue_active.get(),
      state.distributed_current_active.get(),
      state.distributed_scalars.get() + 2u,
      state.distributed_scalars.get() + 3u,
      state.distributed_completion_scores.get(),
      state.distributed_motor_scores.get(),
      state.distributed_candidate_cells.get(),
      state.distributed_candidate_scores.get(),
      state.distributed_path_cells.get(),
      state.distributed_path_phases.get(),
      state.distributed_output_tape.get(),
      state.distributed_scalars.get() + 5u,
      state.distributed_scalars.get() + 6u,
      state.distributed_scalars.get() + 7u,
      state.distributed_scalars.get() + 8u,
      state.distributed_scalars.get() + 9u,
      state.distributed_scalars.get() + 10u,
      state.distributed_scalars.get() + 11u,
      state.distributed_scalars.get() + 12u,
      state.distributed_scalars.get() + 4u,
      state.distributed_mass_scalars.get() + 1u};
}

inline void allocate_distributed_motor(AdultState& state) {
  const distributed_motor::BufferCounts counts =
      distributed_motor::required_buffer_counts(
          kDistributedMotorPopulation, kDistributedMotorActiveWidth,
          kDistributedMotorWindowBytes, kDistributedMotorScratchSteps);
  state.distributed_motor_mass.allocate(counts.motor_mass);
  state.distributed_motor_cell_support.allocate(counts.motor_cell_support);
  state.distributed_motor_support.allocate(counts.motor_support);
  state.distributed_binding_keys.allocate(counts.binding_keys);
  state.distributed_binding_mass.allocate(counts.binding_mass);
  state.distributed_binding_support.allocate(counts.binding_support);
  state.distributed_enabled.allocate(counts.enabled);
  state.distributed_history.allocate(counts.history);
  state.distributed_previous_active.allocate(counts.previous_active);
  state.distributed_sequence_active.allocate(counts.sequence_active);
  state.distributed_current_active.allocate(counts.current_active);
  state.distributed_cue_active.allocate(counts.cue_active);
  state.distributed_completion_scores.allocate(counts.completion_scores);
  state.distributed_motor_scores.allocate(counts.motor_scores);
  state.distributed_candidate_cells.allocate(counts.candidate_cells);
  state.distributed_candidate_scores.allocate(counts.candidate_scores);
  state.distributed_path_cells.allocate(counts.path_cells);
  state.distributed_path_phases.allocate(counts.path_phases);
  state.distributed_output_tape.allocate(counts.output_tape);
  state.distributed_scalars.allocate(counts.scalar_u32);
  state.distributed_mass_scalars.allocate(counts.scalar_u64);
}

inline context_state::FieldView surface_context_field_view(AdultState& state) {
  return {state.surface_context_cells.get(),
          state.surface_context_memberships.get(),
          state.surface_context_transitions.get(),
          state.surface_context_bindings.get(),
          state.surface_context_scalars.get(),
          kSurfaceContextCapacity,
          kSurfaceContextStateCapacity,
          kSurfaceContextMembershipCapacity,
          kSurfaceContextTransitionCapacity,
          state.unit_capacity};
}

inline context_state::LearningWorkspaceView surface_context_workspace_view(
    AdultState& state) {
  return {state.surface_context_primary_ranks.get(),
          state.surface_context_alternate_ranks.get(),
          state.surface_context_state_counts.get()};
}

inline surface_organ::MutableSurfaceGrammarEvidenceView surface_evidence_view(AdultState& state) {
  return {state.surface_unit_mass.get(),
          state.surface_unit_start_mass.get(),
          state.surface_unit_end_mass.get(),
          state.unit_capacity,
          state.surface_role_mass.get(),
          state.surface_role_start_mass.get(),
          state.surface_role_end_mass.get(),
          state.surface_role_bigram_mass.get(),
          state.surface_role_bigram_context_mass.get(),
          state.surface_role_trigram_mass.get(),
          state.surface_role_trigram_context_mass.get(),
          state.surface_stats.get(),
          context_state::as_device_view(surface_context_field_view(state))};
}

inline surface_organ::SurfaceSequenceEvidenceView<BigramKey, TrigramKey>
surface_sequence_evidence_view(AdultState& state) {
  return {state.bigrams.get(),
          state.bigram_counts.get(),
          state.bigram_count,
          state.trigrams.get(),
          state.trigram_counts.get(),
          state.trigram_count,
          state.online_bigrams.get(),
          state.online_bigram_counts.get(),
          state.online_bigram_count,
          state.online_trigrams.get(),
          state.online_trigram_counts.get(),
          state.online_trigram_count};
}

inline surface_organ::SurfaceUnitView surface_unit_view(AdultState& state) {
  return {state.unit_lengths.get(), state.unit_content.get(), state.surface_roles.get(),
          state.unit_count, kUnitWords};
}

inline bcc32_cuda_resident_population_surface::UnitPopulationView
resident_surface_population_view(AdultState& state) {
  return {state.surface_unit_population.get(), state.surface_unit_mass.get(),
          0u, state.unit_count, kDistributedMotorActiveWidth, nullptr,
          state.surface_population_context_mass.get(),
          state.surface_unit_context_population.get()};
}

inline void allocate_surface_organ(AdultState& state) {
  state.surface_role_projection.allocate(
      roles::role_projection_scratch_words(state.unit_capacity));
  state.surface_roles.allocate(state.unit_capacity);
  state.surface_unit_population.allocate(
      static_cast<std::size_t>(state.unit_capacity) * kDistributedMotorActiveWidth);
  state.surface_unit_context_population.allocate(
      static_cast<std::size_t>(state.unit_capacity) * kDistributedMotorActiveWidth);
  state.surface_population_context_mass.allocate(
      static_cast<std::size_t>(state.unit_capacity) * kDistributedMotorActiveWidth);
  state.surface_unit_activity.allocate(state.unit_capacity);
  state.surface_unit_phase.allocate(state.unit_capacity);
  state.surface_projection_state.allocate(2u);
  state.surface_unit_mass.allocate(state.unit_capacity);
  state.surface_unit_start_mass.allocate(state.unit_capacity);
  state.surface_unit_end_mass.allocate(state.unit_capacity);
  state.surface_role_mass.allocate(surface_organ::kSurfaceOrganRoleCount);
  state.surface_role_start_mass.allocate(surface_organ::kSurfaceOrganRoleCount);
  state.surface_role_end_mass.allocate(surface_organ::kSurfaceOrganRoleCount);
  state.surface_role_bigram_mass.allocate(surface_organ::surface_role_bigram_words());
  state.surface_role_bigram_context_mass.allocate(surface_organ::kSurfaceOrganRoleCount);
  state.surface_role_trigram_mass.allocate(surface_organ::surface_role_trigram_words());
  state.surface_role_trigram_context_mass.allocate(surface_organ::surface_role_bigram_words());
  state.surface_stats.allocate(1u);
  state.surface_context_cells.allocate(kSurfaceContextCapacity);
  state.surface_context_memberships.allocate(kSurfaceContextMembershipCapacity);
  state.surface_context_transitions.allocate(kSurfaceContextTransitionCapacity);
  state.surface_context_bindings.allocate(state.unit_capacity);
  state.surface_context_scalars.allocate(1u);
  state.surface_context_primary_ranks.allocate(state.unit_capacity);
  state.surface_context_alternate_ranks.allocate(state.unit_capacity);
  state.surface_context_state_counts.allocate(state.unit_capacity);
  state.surface_bridges.allocate(surface_organ::surface_role_bigram_words());
  state.surface_prefixes.allocate(surface_organ::kSurfaceOrganRoleCount);
  state.surface_suffixes.allocate(surface_organ::kSurfaceOrganRoleCount);
  const std::uint32_t permutations =
      surface_organ::surface_factorial(surface_organ::kSurfaceOrganMaxExhaustiveAnchors);
  state.surface_permutation_scores.allocate(permutations);
  state.surface_permutation_valid.allocate(permutations);
  state.surface_output_units.allocate(surface_organ::kSurfaceOrganMaxOutputUnits);
  state.surface_output_anchor_mask.allocate(surface_organ::kSurfaceOrganMaxOutputUnits);
  state.surface_output_bytes.allocate(4096u);
  state.surface_result.allocate(1u);
  cuda_require(cudaMemset(state.surface_role_projection.get(), 0,
                          state.surface_role_projection.bytes()),
               "clear persistent surface role projection");
  cuda_require(cudaMemset(state.surface_roles.get(), 0, state.surface_roles.bytes()),
               "clear persistent surface roles");
  cuda_require(cudaMemset(state.surface_population_context_mass.get(), 0,
                          state.surface_population_context_mass.bytes()),
               "clear resident population context mass");
  cuda_require(cudaMemset(state.surface_unit_context_population.get(), 0xff,
                          state.surface_unit_context_population.bytes()),
               "clear resident context populations");
  cuda_require(cudaMemset(state.surface_projection_state.get(), 0,
                          state.surface_projection_state.bytes()),
               "clear persistent surface projection state");
  cuda_require(context_state::initialize(
                   surface_context_field_view(state), surface_context_workspace_view(state),
                   kSurfaceContextMassBudget, 2u),
               "initialize resident surface context states");
  state.surface_organ_enabled = true;
}

inline std::size_t ensure_surface_population_context_mass(AdultState& state) {
  if (state.surface_unit_population.get() == nullptr ||
      (state.surface_population_context_mass.get() != nullptr &&
       state.surface_unit_context_population.get() != nullptr))
    return 0u;
  std::size_t bytes = 0u;
  if (state.surface_population_context_mass.get() == nullptr) {
    state.surface_population_context_mass.allocate(
        static_cast<std::size_t>(state.unit_capacity) *
        kDistributedMotorActiveWidth);
    cuda_require(cudaMemset(state.surface_population_context_mass.get(), 0,
                            state.surface_population_context_mass.bytes()),
                 "clear migrated resident population context mass");
    bytes += state.surface_population_context_mass.bytes();
  }
  if (state.surface_unit_context_population.get() == nullptr) {
    state.surface_unit_context_population.allocate(
        static_cast<std::size_t>(state.unit_capacity) *
        kDistributedMotorActiveWidth);
    cuda_require(cudaMemcpy(state.surface_unit_context_population.get(),
                            state.surface_unit_population.get(),
                            state.surface_unit_context_population.bytes(),
                            cudaMemcpyDeviceToDevice),
                 "migrate learned context populations away from identity");
    bytes += state.surface_unit_context_population.bytes();
  }
  return bytes;
}

// The stream checkpoint-only path consumes this constant before the adult's
// executable surface is selected. Keep it with the state ABI so a direct
// checkpoint include does not need to parse the executable adult header.
inline constexpr std::uint32_t kConditionedPredictionFactorWords = 12288u;

}  // namespace bcc32_cuda_adult_v1
