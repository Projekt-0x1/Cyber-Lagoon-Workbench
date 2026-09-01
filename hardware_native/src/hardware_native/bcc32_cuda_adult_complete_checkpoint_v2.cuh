#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// The complete checkpoint owns migration and byte layout, while the dedicated
// state surface owns AdultState storage. Direct checkpoint consumers therefore
// never need to parse the adult's executable kernels.
#include "bcc32_cuda_adult_state.cuh"

namespace bcc32_cuda_adult_complete_checkpoint_v2 {

namespace adult = bcc32_cuda_adult_v1;

constexpr std::uint64_t kMagic = 0x3256504341434342ull;
constexpr std::uint32_t kVersion = 20u;
constexpr std::uint32_t kVersion19LegacyVersion = 19u;
constexpr std::uint32_t kVersion17LegacyVersion = 17u;
constexpr std::uint32_t kVersion16LegacyVersion = 16u;
constexpr std::uint32_t kVersion15LegacyVersion = 15u;
constexpr std::uint32_t kVersion14LegacyVersion = 14u;
constexpr std::uint32_t kVersion13LegacyVersion = 13u;
constexpr std::uint32_t kVersion12LegacyVersion = 12u;
constexpr std::uint32_t kVersion11LegacyVersion = 11u;
constexpr std::uint32_t kVersion10LegacyVersion = 10u;
constexpr std::uint32_t kVersion9LegacyVersion = 9u;
constexpr std::uint32_t kVersion8LegacyVersion = 8u;
constexpr std::uint32_t kVersion7LegacyVersion = 7u;
constexpr std::uint32_t kVersion6LegacyVersion = 6u;
constexpr std::uint32_t kLegacyVersion = 5u;
constexpr std::uint32_t kVersion4LegacyVersion = 4u;
constexpr std::uint32_t kOlderLegacyVersion = 3u;
constexpr std::uint32_t kOldLegacyVersion = 2u;
constexpr std::uint64_t kMaximumFieldBytes = 1ull << 34u;

enum class FieldClass : std::uint32_t {
  persistent_learned_matter = 1u,
  transient_workspace = 2u,
  observer_diagnostic = 3u,
};

// This is the executable ownership inventory for every AdultState device
// allocation at schema v2.  Workspaces and observer receipts are saved too so
// a checkpoint taken at a legal F boundary resumes bit-exactly; their class
// states that they are not credited as learned meaning.
#define BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(X)                    \
  X(boundary_mask, persistent_learned_matter)                        \
  X(boundary_bytes, persistent_learned_matter)                       \
  X(closure_bytes, persistent_learned_matter)                        \
  X(boundary_histogram, persistent_learned_matter)                   \
  X(boundary_pairs, persistent_learned_matter)                       \
  X(unit_lengths, persistent_learned_matter)                         \
  X(unit_content, persistent_learned_matter)                         \
  X(unit_vitality, persistent_learned_matter)                        \
  X(unit_pos, persistent_learned_matter)                             \
  X(unit_hash_slots, transient_workspace)                            \
  X(unigram_top_ids, transient_workspace)                            \
  X(bigrams, persistent_learned_matter)                              \
  X(bigram_counts, persistent_learned_matter)                        \
  X(trigrams, persistent_learned_matter)                             \
  X(trigram_counts, persistent_learned_matter)                       \
  X(cached_bigram_contexts, transient_workspace)                     \
  X(cached_bigram_entries, transient_workspace)                      \
  X(cached_trigram_contexts, transient_workspace)                    \
  X(cached_trigram_entries, transient_workspace)                     \
  X(base_episode_units, persistent_learned_matter)                   \
  X(base_posting_positions, transient_workspace)                     \
  X(base_posting_offsets, transient_workspace)                       \
  X(base_window_signatures, persistent_learned_matter)               \
  X(online_bigrams, persistent_learned_matter)                       \
  X(online_bigram_counts, persistent_learned_matter)                 \
  X(online_trigrams, persistent_learned_matter)                      \
  X(online_trigram_counts, persistent_learned_matter)                \
  X(online_associations, persistent_learned_matter)                  \
  X(online_association_counts, persistent_learned_matter)            \
  X(online_conditioned_transitions, persistent_learned_matter)       \
  X(online_conditioned_transition_counts, persistent_learned_matter) \
  X(online_episode_units, persistent_learned_matter)                 \
  X(online_episode_breaks, persistent_learned_matter)                \
  X(mutable_sizes, persistent_learned_matter)                        \
  X(motor_context, transient_workspace)                              \
  X(motor_completion, transient_workspace)                           \
  X(subject_ids, persistent_learned_matter)                          \
  X(subject_weights, persistent_learned_matter)                      \
  X(subject_count, persistent_learned_matter)                        \
  X(qonset_count, persistent_learned_matter)                         \
  X(qterm_count, persistent_learned_matter)                          \
  X(qorig_onset, persistent_learned_matter)                          \
  X(qorig_onset_w, persistent_learned_matter)                        \
  X(qorig_onset_n, persistent_learned_matter)                        \
  X(qorig_term, persistent_learned_matter)                           \
  X(qorig_term_w, persistent_learned_matter)                         \
  X(qorig_term_n, persistent_learned_matter)                         \
  X(ledger, observer_diagnostic)                                     \
  X(rng, persistent_learned_matter)                                  \
  X(efference_trace, persistent_learned_matter)                      \
  X(efference_state, persistent_learned_matter)                      \
  X(interaction_shadow_trace, persistent_learned_matter)             \
  X(interaction_shadow_state, persistent_learned_matter)             \
  X(synthesis_policy, persistent_learned_matter)                     \
  X(answer_frame_policy, persistent_learned_matter)                  \
  X(answer_frame_selection, persistent_learned_matter)               \
  X(distributed_motor_mass, persistent_learned_matter)               \
  X(distributed_motor_cell_support, persistent_learned_matter)       \
  X(distributed_motor_support, persistent_learned_matter)            \
  X(distributed_binding_keys, persistent_learned_matter)             \
  X(distributed_binding_mass, persistent_learned_matter)             \
  X(distributed_binding_support, persistent_learned_matter)          \
  X(distributed_enabled, persistent_learned_matter)                  \
  X(distributed_history, persistent_learned_matter)                  \
  X(distributed_previous_active, persistent_learned_matter)          \
  X(distributed_sequence_active, persistent_learned_matter)          \
  X(distributed_current_active, persistent_learned_matter)           \
  X(distributed_cue_active, persistent_learned_matter)               \
  X(distributed_completion_scores, transient_workspace)              \
  X(distributed_motor_scores, transient_workspace)                   \
  X(distributed_candidate_cells, transient_workspace)                \
  X(distributed_candidate_scores, transient_workspace)               \
  X(distributed_path_cells, persistent_learned_matter)               \
  X(distributed_path_phases, persistent_learned_matter)              \
  X(distributed_output_tape, persistent_learned_matter)              \
  X(distributed_scalars, persistent_learned_matter)                  \
  X(distributed_mass_scalars, persistent_learned_matter)             \
  X(surface_role_projection, persistent_learned_matter)              \
  X(surface_roles, persistent_learned_matter)                        \
  X(surface_unit_population, persistent_learned_matter)              \
  X(surface_unit_activity, transient_workspace)                      \
  X(surface_unit_phase, persistent_learned_matter)                   \
  X(surface_projection_state, transient_workspace)                   \
  X(surface_unit_mass, persistent_learned_matter)                    \
  X(surface_unit_start_mass, persistent_learned_matter)              \
  X(surface_unit_end_mass, persistent_learned_matter)                \
  X(surface_role_mass, persistent_learned_matter)                    \
  X(surface_role_start_mass, persistent_learned_matter)              \
  X(surface_role_end_mass, persistent_learned_matter)                \
  X(surface_role_bigram_mass, persistent_learned_matter)             \
  X(surface_role_bigram_context_mass, persistent_learned_matter)     \
  X(surface_role_trigram_mass, persistent_learned_matter)            \
  X(surface_role_trigram_context_mass, persistent_learned_matter)    \
  X(surface_stats, persistent_learned_matter)                        \
  X(surface_bridges, persistent_learned_matter)                      \
  X(surface_prefixes, persistent_learned_matter)                     \
  X(surface_suffixes, persistent_learned_matter)                     \
  X(surface_permutation_scores, transient_workspace)                 \
  X(surface_permutation_valid, transient_workspace)                  \
  X(surface_output_units, transient_workspace)                       \
  X(surface_output_anchor_mask, transient_workspace)                 \
  X(surface_output_bytes, transient_workspace)                       \
  X(surface_result, observer_diagnostic)

#define BCC32_ADULT_CHECKPOINT_V3_ARRAY_FIELDS(X)                     \
  BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(X)                           \
  X(proposition_synapses, persistent_learned_matter)                  \
  X(proposition_scalars, persistent_learned_matter)                   \
  X(proposition_completion_scores, transient_workspace)              \
  X(proposition_output_cells, transient_workspace)                   \
  X(proposition_output_scores, transient_workspace)                  \
  X(proposition_completion_result, observer_diagnostic)

#define BCC32_ADULT_CHECKPOINT_V4_ARRAY_FIELDS(X)                     \
  BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(X)                           \
  X(proposition_synapses, persistent_learned_matter)                  \
  X(proposition_cohorts, persistent_learned_matter)                   \
  X(proposition_scalars, persistent_learned_matter)                   \
  X(proposition_completion_scores, transient_workspace)              \
  X(proposition_output_cells, transient_workspace)                   \
  X(proposition_output_scores, transient_workspace)                  \
  X(proposition_completion_result, observer_diagnostic)

#define BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(X)                     \
  BCC32_ADULT_CHECKPOINT_V4_ARRAY_FIELDS(X)                           \
  X(proposition_ordered_bindings, persistent_learned_matter)          \
  X(proposition_ordered_binding_admission, persistent_learned_matter) \
  X(proposition_ordered_evidence_revision, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V7_ARRAY_FIELDS(X)                     \
  BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(X)                           \
  X(question_goal_state, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V8_ADDED_ARRAY_FIELDS(X)                     \
  X(construction_role_projection, transient_workspace)                     \
  X(construction_roles, persistent_learned_matter)                          \
  X(construction_closure_bytes, persistent_learned_matter)                  \
  X(construction_closed_class_mask, persistent_learned_matter)              \
  X(construction_filler_terminal_mask, persistent_learned_matter)           \
  X(construction_initial_form_mask, persistent_learned_matter)              \
  X(construction_role_population, persistent_learned_matter)                \
  X(construction_tokens, persistent_learned_matter)                         \
  X(construction_lengths, persistent_learned_matter)                        \
  X(construction_slot_counts, persistent_learned_matter)                    \
  X(construction_supports, persistent_learned_matter)                       \
  X(construction_hash_slots, transient_workspace)                           \
  X(construction_store_count, persistent_learned_matter)                    \
  X(construction_pool_units, transient_workspace)                           \
  X(construction_pool_roles, transient_workspace)                           \
  X(construction_pool_weights, transient_workspace)                         \
  X(construction_pool_meta, transient_workspace)                            \
  X(construction_best, transient_workspace)                                 \
  X(construction_last_selected, persistent_learned_matter)                  \
  X(construction_plan, transient_workspace)                                 \
  X(construction_plan_meta, transient_workspace)                            \
  X(construction_suffix_transitions, persistent_learned_matter)             \
  X(construction_role_canon, persistent_learned_matter)                     \
  X(construction_role_canon_signatures, transient_workspace)                \
  X(construction_role_canon_keys, transient_workspace)                      \
  X(construction_role_canon_reps, transient_workspace)                      \
  X(content_commitment_units, transient_workspace)                          \
  X(content_commitment_meta, transient_workspace)                           \
  X(relation_triples, persistent_learned_matter)                            \
  X(relation_triple_counts, persistent_learned_matter)                      \
  X(relation_triple_type_total, persistent_learned_matter)                  \
  X(relation_triple_type_mirrored, persistent_learned_matter)               \
  X(relation_triple_candidates, transient_workspace)                        \
  X(relation_triple_cursor, transient_workspace)                            \
  X(relation_triple_plan, transient_workspace)                              \
  X(relation_triple_meta, transient_workspace)                              \
  X(relation_cue_scores, transient_workspace)                               \
  X(relation_cue_orders, transient_workspace)                               \
  X(relation_cue_exact, transient_workspace)                                \
  X(relation_operator_order, transient_workspace)                           \
  X(streaming_cue_bytes, transient_workspace)                               \
  X(streaming_cue_meta, transient_workspace)                                \
  X(surface_context_cells, persistent_learned_matter)                       \
  X(surface_context_memberships, persistent_learned_matter)                 \
  X(surface_context_transitions, persistent_learned_matter)                 \
  X(surface_context_bindings, persistent_learned_matter)                    \
  X(surface_context_scalars, persistent_learned_matter)                     \
  X(surface_context_primary_ranks, transient_workspace)                     \
  X(surface_context_alternate_ranks, transient_workspace)                   \
  X(surface_context_state_counts, transient_workspace)

#define BCC32_ADULT_CHECKPOINT_V8_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V7_ARRAY_FIELDS(X)                                 \
  BCC32_ADULT_CHECKPOINT_V8_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V9_ADDED_ARRAY_FIELDS(X)                     \
  X(proposition_cue_sequence, transient_workspace)                          \
  X(proposition_cue_sequence_count, transient_workspace)

#define BCC32_ADULT_CHECKPOINT_V10_ADDED_ARRAY_FIELDS(X)                    \
  X(construction_slot_units, persistent_learned_matter)                     \
  X(construction_slot_masses, persistent_learned_matter)                    \
  X(construction_slot_totals, persistent_learned_matter)                    \
  X(construction_slot_overflow, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V9_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V8_ARRAY_FIELDS(X)                                 \
  BCC32_ADULT_CHECKPOINT_V9_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V10_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V9_ARRAY_FIELDS(X)                                 \
  BCC32_ADULT_CHECKPOINT_V10_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V11_ADDED_ARRAY_FIELDS(X)                    \
  X(proposition_ordered_construction, persistent_learned_matter)             \
  X(construction_evidence_revision, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V11_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V10_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V11_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V12_ADDED_ARRAY_FIELDS(X)                    \
  X(relation_triple_evidence_revision, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V12_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V11_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V12_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V13_ADDED_ARRAY_FIELDS(X)                    \
  X(witnessed_relation_events, persistent_learned_matter)                   \
  X(witnessed_relation_event_cursor, persistent_learned_matter)             \
  X(qonset_evidence_revision, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V13_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V12_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V13_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V14_ADDED_ARRAY_FIELDS(X)                    \
  X(witnessed_relation_constructions, persistent_learned_matter)            \
  X(witnessed_relation_surface_units, persistent_learned_matter)            \
  X(witnessed_relation_surface_counts, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V14_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V13_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V14_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V15_ADDED_ARRAY_FIELDS(X)                    \
  X(question_gap_field_support, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V15_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V14_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V15_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V16_ADDED_ARRAY_FIELDS(X)                    \
  X(question_answer_construction, persistent_learned_matter)                \
  X(question_answer_construction_support, persistent_learned_matter)         \
  X(question_answer_slot_mapping, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V16_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V15_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V16_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V17_ADDED_ARRAY_FIELDS(X)                    \
  X(surface_population_context_mass, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V17_ARRAY_FIELDS(X)                          \
  BCC32_ADULT_CHECKPOINT_V16_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V17_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V18_ADDED_ARRAY_FIELDS(X)                    \
  X(surface_unit_context_population, persistent_learned_matter)              \
  X(construction_origin_revision, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_V19_ADDED_ARRAY_FIELDS(X)                    \
  X(proposition_construction_association, observer_diagnostic)

#define BCC32_ADULT_CHECKPOINT_V19_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V17_ARRAY_FIELDS(X)                                 \
  BCC32_ADULT_CHECKPOINT_V18_ADDED_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V19_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V20_ADDED_ARRAY_FIELDS(X)                    \
  X(relation_roles, persistent_learned_matter)                              \
  X(relation_role_counts, persistent_learned_matter)

#define BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(X)                              \
  BCC32_ADULT_CHECKPOINT_V19_ARRAY_FIELDS(X)                                \
  BCC32_ADULT_CHECKPOINT_V20_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_V9_AFTER_SYNAPSE_FIELDS(X)                   \
  X(proposition_cohorts, persistent_learned_matter)                         \
  X(proposition_scalars, persistent_learned_matter)                         \
  X(proposition_completion_scores, transient_workspace)                    \
  X(proposition_output_cells, transient_workspace)                         \
  X(proposition_output_scores, transient_workspace)                        \
  X(proposition_completion_result, observer_diagnostic)                    \
  X(proposition_ordered_bindings, persistent_learned_matter)               \
  X(proposition_ordered_binding_admission, persistent_learned_matter)       \
  X(proposition_ordered_evidence_revision, persistent_learned_matter)       \
  X(question_goal_state, persistent_learned_matter)                         \
  BCC32_ADULT_CHECKPOINT_V8_ADDED_ARRAY_FIELDS(X)                           \
  BCC32_ADULT_CHECKPOINT_V9_ADDED_ARRAY_FIELDS(X)

#define BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(X)                        \
  X(discourse_said, persistent_learned_matter)                              \
  X(discourse_front, persistent_learned_matter)

constexpr std::uint32_t kLegacyArrayFieldCount = 0u
#define BCC32_COUNT_LEGACY_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(BCC32_COUNT_LEGACY_FIELD)
#undef BCC32_COUNT_LEGACY_FIELD
    ;
static_assert(kLegacyArrayFieldCount == 103u);

constexpr std::uint32_t kVersion3ArrayFieldCount = 0u
#define BCC32_COUNT_VERSION3_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V3_ARRAY_FIELDS(BCC32_COUNT_VERSION3_FIELD)
#undef BCC32_COUNT_VERSION3_FIELD
    ;
static_assert(kVersion3ArrayFieldCount == 109u);

constexpr std::uint32_t kVersion4ArrayFieldCount = 0u
#define BCC32_COUNT_VERSION4_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V4_ARRAY_FIELDS(BCC32_COUNT_VERSION4_FIELD)
#undef BCC32_COUNT_VERSION4_FIELD
    ;
static_assert(kVersion4ArrayFieldCount == 110u);

constexpr std::uint32_t kDeviceArrayFieldCount = 0u
#define BCC32_COUNT_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_COUNT_FIELD)
#undef BCC32_COUNT_FIELD
    ;
static_assert(kDeviceArrayFieldCount == 189u,
              "AdultState ownership inventory changed; version the schema");

constexpr std::uint32_t kHostVectorFieldCount = 0u
#define BCC32_COUNT_HOST_VECTOR(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_COUNT_HOST_VECTOR)
#undef BCC32_COUNT_HOST_VECTOR
    ;
static_assert(kHostVectorFieldCount == 2u);
constexpr std::uint32_t kArrayFieldCount =
    kDeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion8DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION8_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V8_ARRAY_FIELDS(BCC32_COUNT_VERSION8_FIELD)
#undef BCC32_COUNT_VERSION8_FIELD
    ;
static_assert(kVersion8DeviceArrayFieldCount == 164u);
constexpr std::uint32_t kVersion8ArrayFieldCount =
    kVersion8DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion9DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION9_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V9_ARRAY_FIELDS(BCC32_COUNT_VERSION9_FIELD)
#undef BCC32_COUNT_VERSION9_FIELD
    ;
static_assert(kVersion9DeviceArrayFieldCount == 166u);
constexpr std::uint32_t kVersion9ArrayFieldCount =
    kVersion9DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion10DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION10_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V10_ARRAY_FIELDS(BCC32_COUNT_VERSION10_FIELD)
#undef BCC32_COUNT_VERSION10_FIELD
    ;
static_assert(kVersion10DeviceArrayFieldCount == 170u);
constexpr std::uint32_t kVersion10ArrayFieldCount =
    kVersion10DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion11DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION11_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V11_ARRAY_FIELDS(BCC32_COUNT_VERSION11_FIELD)
#undef BCC32_COUNT_VERSION11_FIELD
    ;
static_assert(kVersion11DeviceArrayFieldCount == 172u);
constexpr std::uint32_t kVersion11ArrayFieldCount =
    kVersion11DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion12DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION12_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V12_ARRAY_FIELDS(BCC32_COUNT_VERSION12_FIELD)
#undef BCC32_COUNT_VERSION12_FIELD
    ;
static_assert(kVersion12DeviceArrayFieldCount == 173u);
constexpr std::uint32_t kVersion12ArrayFieldCount =
    kVersion12DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion13DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION13_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V13_ARRAY_FIELDS(BCC32_COUNT_VERSION13_FIELD)
#undef BCC32_COUNT_VERSION13_FIELD
    ;
static_assert(kVersion13DeviceArrayFieldCount == 176u);
constexpr std::uint32_t kVersion13ArrayFieldCount =
    kVersion13DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion14DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION14_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V14_ARRAY_FIELDS(BCC32_COUNT_VERSION14_FIELD)
#undef BCC32_COUNT_VERSION14_FIELD
    ;
static_assert(kVersion14DeviceArrayFieldCount == 179u);
constexpr std::uint32_t kVersion14ArrayFieldCount =
    kVersion14DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion15DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION15_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V15_ARRAY_FIELDS(BCC32_COUNT_VERSION15_FIELD)
#undef BCC32_COUNT_VERSION15_FIELD
    ;
static_assert(kVersion15DeviceArrayFieldCount == 180u);
constexpr std::uint32_t kVersion15ArrayFieldCount =
    kVersion15DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion16DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION16_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V16_ARRAY_FIELDS(BCC32_COUNT_VERSION16_FIELD)
#undef BCC32_COUNT_VERSION16_FIELD
    ;
static_assert(kVersion16DeviceArrayFieldCount == 183u);
constexpr std::uint32_t kVersion16ArrayFieldCount =
    kVersion16DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion17DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION17_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V17_ARRAY_FIELDS(BCC32_COUNT_VERSION17_FIELD)
#undef BCC32_COUNT_VERSION17_FIELD
    ;
static_assert(kVersion17DeviceArrayFieldCount == 184u);
constexpr std::uint32_t kVersion17ArrayFieldCount =
    kVersion17DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion19DeviceArrayFieldCount = 0u
#define BCC32_COUNT_VERSION19_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V19_ARRAY_FIELDS(BCC32_COUNT_VERSION19_FIELD)
#undef BCC32_COUNT_VERSION19_FIELD
    ;
static_assert(kVersion19DeviceArrayFieldCount == 187u);
constexpr std::uint32_t kVersion19ArrayFieldCount =
    kVersion19DeviceArrayFieldCount + kHostVectorFieldCount;

constexpr std::uint32_t kVersion7ArrayFieldCount = 0u
#define BCC32_COUNT_VERSION7_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V7_ARRAY_FIELDS(BCC32_COUNT_VERSION7_FIELD)
#undef BCC32_COUNT_VERSION7_FIELD
    ;
static_assert(kVersion7ArrayFieldCount == 114u);

constexpr std::uint32_t kVersion5ArrayFieldCount = 0u
#define BCC32_COUNT_VERSION5_FIELD(name, classification) +1u
    BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(BCC32_COUNT_VERSION5_FIELD)
#undef BCC32_COUNT_VERSION5_FIELD
    ;
static_assert(kVersion5ArrayFieldCount == 113u);

struct Header {
  std::uint64_t magic = kMagic;
  std::uint32_t version = kVersion;
  std::uint32_t array_field_count = kArrayFieldCount;
  std::uint32_t scalar_bytes = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t schema_hash = 0u;
  std::uint64_t scalar_hash = 0u;
};

struct ArrayHeader {
  std::uint64_t name_hash = 0u;
  std::uint32_t classification = 0u;
  std::uint32_t element_bytes = 0u;
  std::uint64_t count = 0u;
  std::uint64_t data_hash = 0u;
};

struct SparseBindingSynapseV9 {
  std::uint32_t source_cell = 0u;
  std::uint32_t target_cell = 0u;
  std::uint32_t claimed = 0u;
  std::uint32_t overflow_mass = 0u;
  std::uint32_t cohort_overflow_mass = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t structure_mass = 0u;
};

struct AdultScalarStateV7 {
  std::uint32_t qorig_on = 0u;
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
  std::uint32_t reserved_alignment = 0u;
  std::uint64_t resident_bytes = 0u;
  std::uint32_t transitions_lesioned = 0u;
  std::uint32_t base_episode_lesioned = 0u;
  std::uint32_t novelty_epoch_active = 0u;
  std::uint32_t novelty_epoch_pending = 0u;
  std::uint32_t frame_emit = 0u;
  std::uint32_t distributed_motor_enabled = 0u;
  std::uint32_t surface_organ_enabled = 0u;
  std::uint32_t reserved_tail = 0u;
};

struct AdultScalarState {
  AdultScalarStateV7 v7{};
  std::uint32_t discourse_theme = 0xffffffffu;
  std::uint32_t construction_closure_count = 0u;
  std::uint32_t construction_func_threshold = 0u;
  std::uint32_t construction_role_population_cutoff = 0u;
  std::uint32_t construction_count_host = 0u;
  std::uint32_t retired_construction_nonce = 0u;
  std::uint32_t construction_role_canon_table_size = 0u;
  std::uint32_t construction_lesioned = 0u;
  std::uint32_t morph_agreement_lesioned = 0u;
  std::uint32_t role_canon_lesioned = 0u;
  std::uint32_t entropy_glue_lesioned = 0u;
  std::uint32_t content_commit_lesioned = 0u;
  std::uint32_t relation_triple_lesioned = 0u;
  std::uint32_t streaming_cue_mode = 0u;
};

static_assert(std::is_trivially_copyable_v<Header>);
static_assert(std::is_trivially_copyable_v<ArrayHeader>);
static_assert(std::is_trivially_copyable_v<AdultScalarStateV7>);
static_assert(std::is_trivially_copyable_v<AdultScalarState>);
static_assert(std::is_trivially_copyable_v<SparseBindingSynapseV9>);
static_assert(sizeof(SparseBindingSynapseV9) == 32u);
static_assert(sizeof(AdultScalarStateV7) == 120u,
              "schema-v7 scalars must retain their exact ABI");
static_assert(sizeof(AdultScalarState) == 176u,
              "schema-v8 scalars must not contain implicit padding");

struct Receipt {
  std::uint64_t schema_hash = 0u;
  std::uint64_t persistent_hash = 1469598103934665603ull;
  std::uint64_t workspace_hash = 1469598103934665603ull;
  std::uint64_t diagnostic_hash = 1469598103934665603ull;
  std::uint64_t payload_bytes = 0u;
  std::uint32_t array_fields = 0u;
};

inline std::uint64_t hash_bytes(const void* bytes, std::size_t count,
                                std::uint64_t seed = 1469598103934665603ull) {
  const auto* data = static_cast<const std::uint8_t*>(bytes);
  std::uint64_t hash = seed;
  for (std::size_t index = 0u; index < count; ++index) {
    hash ^= data[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

inline std::uint64_t hash_name(const char* name) {
  std::size_t size = 0u;
  while (name[size] != '\0')
    ++size;
  return hash_bytes(name, size);
}

inline void mix_hash(std::uint64_t* destination, std::uint64_t value) {
  *destination = hash_bytes(&value, sizeof(value), *destination);
}

inline AdultScalarStateV7 capture_v7_scalars(const adult::AdultState& state) {
  return {state.qorig_on ? 1u : 0u,
          state.unit_occurrences,
          state.unit_count,
          state.bigram_count,
          state.trigram_count,
          state.cached_bigram_count,
          state.cached_trigram_count,
          state.base_window_count,
          state.unit_capacity,
          state.surface_content_begin,
          state.unit_hash_capacity,
          state.online_bigram_count,
          state.online_trigram_count,
          state.online_association_count,
          state.online_conditioned_transition_count,
          state.online_episode_count,
          state.online_episode_break_count,
          state.novelty_episode_begin,
          state.novelty_episode_break_begin,
          0u,
          static_cast<std::uint64_t>(state.resident_bytes),
          state.transitions_lesioned ? 1u : 0u,
          state.base_episode_lesioned ? 1u : 0u,
          state.novelty_epoch_active ? 1u : 0u,
          state.novelty_epoch_pending ? 1u : 0u,
          state.frame_emit ? 1u : 0u,
          state.distributed_motor_enabled ? 1u : 0u,
          state.surface_organ_enabled ? 1u : 0u,
          0u};
}

inline AdultScalarState capture_scalars(const adult::AdultState& state) {
  AdultScalarState scalar{};
  scalar.v7 = capture_v7_scalars(state);
  scalar.discourse_theme = state.discourse_theme;
  scalar.construction_closure_count = state.construction_closure_count;
  scalar.construction_func_threshold = state.construction_func_threshold;
  scalar.construction_role_population_cutoff = state.construction_role_population_cutoff;
  scalar.construction_count_host = state.construction_count_host;
  scalar.retired_construction_nonce = state.retired_construction_nonce;
  scalar.construction_role_canon_table_size = state.construction_role_canon_table_size;
  scalar.construction_lesioned = state.construction_lesioned ? 1u : 0u;
  scalar.morph_agreement_lesioned = state.morph_agreement_lesioned ? 1u : 0u;
  scalar.role_canon_lesioned = state.role_canon_lesioned ? 1u : 0u;
  scalar.entropy_glue_lesioned = state.entropy_glue_lesioned ? 1u : 0u;
  scalar.content_commit_lesioned = state.content_commit_lesioned ? 1u : 0u;
  scalar.relation_triple_lesioned = state.relation_triple_lesioned ? 1u : 0u;
  scalar.streaming_cue_mode = state.streaming_cue_mode ? 1u : 0u;
  return scalar;
}

inline void restore_v7_scalars(adult::AdultState* state,
                               const AdultScalarStateV7& scalar) {
  state->qorig_on = scalar.qorig_on != 0u;
  state->unit_occurrences = scalar.unit_occurrences;
  state->unit_count = scalar.unit_count;
  state->bigram_count = scalar.bigram_count;
  state->trigram_count = scalar.trigram_count;
  state->cached_bigram_count = scalar.cached_bigram_count;
  state->cached_trigram_count = scalar.cached_trigram_count;
  state->base_window_count = scalar.base_window_count;
  state->unit_capacity = scalar.unit_capacity;
  state->surface_content_begin = scalar.surface_content_begin;
  state->unit_hash_capacity = scalar.unit_hash_capacity;
  state->online_bigram_count = scalar.online_bigram_count;
  state->online_trigram_count = scalar.online_trigram_count;
  state->online_association_count = scalar.online_association_count;
  state->online_conditioned_transition_count = scalar.online_conditioned_transition_count;
  state->online_episode_count = scalar.online_episode_count;
  state->online_episode_break_count = scalar.online_episode_break_count;
  state->novelty_episode_begin = scalar.novelty_episode_begin;
  state->novelty_episode_break_begin = scalar.novelty_episode_break_begin;
  state->resident_bytes = static_cast<std::size_t>(scalar.resident_bytes);
  state->transitions_lesioned = scalar.transitions_lesioned != 0u;
  state->base_episode_lesioned = scalar.base_episode_lesioned != 0u;
  state->novelty_epoch_active = scalar.novelty_epoch_active != 0u;
  state->novelty_epoch_pending = scalar.novelty_epoch_pending != 0u;
  state->frame_emit = scalar.frame_emit != 0u;
  state->distributed_motor_enabled = scalar.distributed_motor_enabled != 0u;
  state->surface_organ_enabled = scalar.surface_organ_enabled != 0u;
}

inline void restore_scalars(adult::AdultState* state, const AdultScalarState& scalar) {
  restore_v7_scalars(state, scalar.v7);
  state->discourse_theme = scalar.discourse_theme;
  state->construction_closure_count = scalar.construction_closure_count;
  state->construction_func_threshold = scalar.construction_func_threshold;
  state->construction_role_population_cutoff = scalar.construction_role_population_cutoff;
  state->construction_count_host = scalar.construction_count_host;
  state->retired_construction_nonce = scalar.retired_construction_nonce;
  state->construction_role_canon_table_size = scalar.construction_role_canon_table_size;
  state->construction_lesioned = scalar.construction_lesioned != 0u;
  state->morph_agreement_lesioned = scalar.morph_agreement_lesioned != 0u;
  state->role_canon_lesioned = scalar.role_canon_lesioned != 0u;
  state->entropy_glue_lesioned = scalar.entropy_glue_lesioned != 0u;
  state->content_commit_lesioned = scalar.content_commit_lesioned != 0u;
  state->relation_triple_lesioned = scalar.relation_triple_lesioned != 0u;
  state->streaming_cue_mode = scalar.streaming_cue_mode != 0u;
}

inline void validate_v7_scalars(const AdultScalarStateV7& scalar) {
  const auto boolean = [](std::uint32_t value) { return value <= 1u; };
  if (scalar.reserved_alignment != 0u || scalar.reserved_tail != 0u || !boolean(scalar.qorig_on) ||
      !boolean(scalar.transitions_lesioned) || !boolean(scalar.base_episode_lesioned) ||
      !boolean(scalar.novelty_epoch_active) || !boolean(scalar.novelty_epoch_pending) ||
      !boolean(scalar.frame_emit) || !boolean(scalar.distributed_motor_enabled) ||
      !boolean(scalar.surface_organ_enabled) ||
      scalar.resident_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("invalid complete adult checkpoint scalars");
  }
}

inline void validate_scalars(const AdultScalarState& scalar) {
  validate_v7_scalars(scalar.v7);
  const auto boolean = [](std::uint32_t value) { return value <= 1u; };
  if (!boolean(scalar.construction_lesioned) ||
      !boolean(scalar.morph_agreement_lesioned) ||
      !boolean(scalar.role_canon_lesioned) ||
      !boolean(scalar.entropy_glue_lesioned) ||
      !boolean(scalar.content_commit_lesioned) ||
      !boolean(scalar.relation_triple_lesioned) ||
      !boolean(scalar.streaming_cue_mode)) {
    throw std::runtime_error("invalid complete adult checkpoint behavior scalars");
  }
}

// Every *_schema_hash() function below ends by mixing in the scalar-state
// struct size and, for schemas that carry proposition tissue (v10+), the
// SparseBindingSynapse size, then returning. Shared here since it is the
// one part of each function that never varies with which field-list macro
// was applied above it.
inline std::uint64_t finalize_schema_hash(std::uint64_t hash,
                                          std::size_t scalar_bytes,
                                          bool include_proposition_synapse) {
  mix_hash(&hash, static_cast<std::uint64_t>(scalar_bytes));
  if (include_proposition_synapse) {
    mix_hash(&hash, static_cast<std::uint64_t>(
                         sizeof(adult::proposition_tissue::SparseBindingSynapse)));
  }
  return hash;
}

inline std::uint64_t legacy_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_FIELD(name, classification)                                              \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(BCC32_SCHEMA_FIELD)
#undef BCC32_SCHEMA_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarStateV7), false);
}

inline std::uint64_t version3_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_FIELD(name, classification)                                              \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V3_ARRAY_FIELDS(BCC32_SCHEMA_FIELD)
#undef BCC32_SCHEMA_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarStateV7), false);
}

inline std::uint64_t version4_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_FIELD(name, classification)                                              \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V4_ARRAY_FIELDS(BCC32_SCHEMA_FIELD)
#undef BCC32_SCHEMA_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarStateV7), false);
}

inline std::uint64_t version5_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_FIELD(name, classification)                                              \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(BCC32_SCHEMA_FIELD)
#undef BCC32_SCHEMA_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarStateV7), false);
}

inline std::uint64_t schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_FIELD(name, classification)                                              \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_SCHEMA_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_FIELD)
#undef BCC32_SCHEMA_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version11_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V11_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V11_ARRAY_FIELDS(BCC32_SCHEMA_V11_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V11_FIELD)
#undef BCC32_SCHEMA_V11_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version12_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V12_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V12_ARRAY_FIELDS(BCC32_SCHEMA_V12_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V12_FIELD)
#undef BCC32_SCHEMA_V12_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version13_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V13_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V13_ARRAY_FIELDS(BCC32_SCHEMA_V13_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V13_FIELD)
#undef BCC32_SCHEMA_V13_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version14_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V14_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V14_ARRAY_FIELDS(BCC32_SCHEMA_V14_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V14_FIELD)
#undef BCC32_SCHEMA_V14_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version15_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V15_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V15_ARRAY_FIELDS(BCC32_SCHEMA_V15_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V15_FIELD)
#undef BCC32_SCHEMA_V15_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version16_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V16_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V16_ARRAY_FIELDS(BCC32_SCHEMA_V16_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V16_FIELD)
#undef BCC32_SCHEMA_V16_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version17_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V17_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V17_ARRAY_FIELDS(BCC32_SCHEMA_V17_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V17_FIELD)
#undef BCC32_SCHEMA_V17_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version19_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V19_FIELD(name, classification)                         \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V19_ARRAY_FIELDS(BCC32_SCHEMA_V19_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V19_FIELD)
#undef BCC32_SCHEMA_V19_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version10_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V10_FIELD(name, classification)                        \
  do {                                                                       \
    const std::uint64_t name_value = hash_name(#name);                       \
    const std::uint32_t class_value =                                        \
        static_cast<std::uint32_t>(FieldClass::classification);              \
    mix_hash(&hash, name_value);                                              \
    mix_hash(&hash, class_value);                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V10_ARRAY_FIELDS(BCC32_SCHEMA_V10_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V10_FIELD)
#undef BCC32_SCHEMA_V10_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), true);
}

inline std::uint64_t version9_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V9_FIELD(name, classification)                         \
  do {                                                                      \
    const std::uint64_t name_value = hash_name(#name);                      \
    const std::uint32_t class_value =                                       \
        static_cast<std::uint32_t>(FieldClass::classification);             \
    mix_hash(&hash, name_value);                                             \
    mix_hash(&hash, class_value);                                            \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V9_ARRAY_FIELDS(BCC32_SCHEMA_V9_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V9_FIELD)
#undef BCC32_SCHEMA_V9_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), false);
}

inline std::uint64_t version8_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V8_FIELD(name, classification)                                           \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V8_ARRAY_FIELDS(BCC32_SCHEMA_V8_FIELD)
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_SCHEMA_V8_FIELD)
#undef BCC32_SCHEMA_V8_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarState), false);
}

inline std::uint64_t version7_schema_hash() {
  std::uint64_t hash = 1469598103934665603ull;
#define BCC32_SCHEMA_V7_FIELD(name, classification)                                           \
  do {                                                                                        \
    const std::uint64_t name_value = hash_name(#name);                                        \
    const std::uint32_t class_value = static_cast<std::uint32_t>(FieldClass::classification); \
    mix_hash(&hash, name_value);                                                              \
    mix_hash(&hash, class_value);                                                             \
  } while (false);
  BCC32_ADULT_CHECKPOINT_V7_ARRAY_FIELDS(BCC32_SCHEMA_V7_FIELD)
#undef BCC32_SCHEMA_V7_FIELD
  return finalize_schema_hash(hash, sizeof(AdultScalarStateV7), false);
}

// Schema v6 had the same field names as v7 but stored QuestionGoal v1
// elements.  The version, not the name-only schema hash, distinguishes the
// element ABI during deterministic migration.
inline std::uint64_t version6_schema_hash() { return version7_schema_hash(); }

inline std::size_t serialized_array_count(const adult::AdultState& state,
                                          const char* name,
                                          std::size_t capacity) {
  std::size_t count = capacity;
  if (std::strcmp(name, "online_bigrams") == 0 ||
      std::strcmp(name, "online_bigram_counts") == 0)
    count = state.online_bigram_count;
  else if (std::strcmp(name, "online_trigrams") == 0 ||
           std::strcmp(name, "online_trigram_counts") == 0)
    count = state.online_trigram_count;
  else if (std::strcmp(name, "online_associations") == 0 ||
           std::strcmp(name, "online_association_counts") == 0)
    count = state.online_association_count;
  else if (std::strcmp(name, "online_conditioned_transitions") == 0 ||
           std::strcmp(name, "online_conditioned_transition_counts") == 0)
    count = state.online_conditioned_transition_count;
  else if (std::strcmp(name, "online_episode_units") == 0)
    count = state.online_episode_count;
  else if (std::strcmp(name, "online_episode_breaks") == 0)
    count = state.online_episode_break_count;
  if (count > capacity)
    throw std::runtime_error("complete adult checkpoint live extent exceeds capacity: " +
                             std::string(name));
  return count;
}

inline std::size_t restored_array_capacity(const char* name,
                                           std::size_t serialized_count) {
  std::size_t capacity = serialized_count;
  if (std::strcmp(name, "online_bigrams") == 0 ||
      std::strcmp(name, "online_bigram_counts") == 0 ||
      std::strcmp(name, "online_trigrams") == 0 ||
      std::strcmp(name, "online_trigram_counts") == 0)
    capacity = adult::kOnlineNgramCapacity;
  else if (std::strcmp(name, "online_associations") == 0 ||
           std::strcmp(name, "online_association_counts") == 0)
    capacity = adult::kOnlineAssociationCapacity;
  else if (std::strcmp(name, "online_conditioned_transitions") == 0 ||
           std::strcmp(name, "online_conditioned_transition_counts") == 0)
    capacity = adult::kOnlineConditionedTransitionCapacity;
  else if (std::strcmp(name, "online_episode_units") == 0)
    capacity = adult::kOnlineEpisodeCapacity;
  else if (std::strcmp(name, "online_episode_breaks") == 0)
    capacity = adult::kOnlineEpisodeBreakCapacity;
  if (serialized_count > capacity)
    throw std::runtime_error("complete adult checkpoint field exceeds resident capacity: " +
                             std::string(name));
  return capacity;
}

#include "bcc32_cuda_adult_complete_checkpoint_io.inl"

inline std::uint64_t read_version9_proposition_synapses(
    std::ifstream* input,
    adult::DeviceArray<adult::proposition_tissue::SparseBindingSynapse>* array,
    Receipt* receipt) {
  ArrayHeader header{};
  input->read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!*input || header.name_hash != hash_name("proposition_synapses") ||
      header.classification !=
          static_cast<std::uint32_t>(FieldClass::persistent_learned_matter) ||
      header.element_bytes != sizeof(SparseBindingSynapseV9) ||
      header.count > std::numeric_limits<std::size_t>::max() ||
      header.count > kMaximumFieldBytes / sizeof(SparseBindingSynapseV9) ||
      header.count >
          kMaximumFieldBytes /
              sizeof(adult::proposition_tissue::SparseBindingSynapse))
    throw std::runtime_error("invalid schema-v9 proposition synapses");
  std::vector<SparseBindingSynapseV9> legacy(
      static_cast<std::size_t>(header.count));
  const std::uint64_t legacy_bytes =
      header.count * sizeof(SparseBindingSynapseV9);
  if (legacy_bytes != 0u)
    input->read(reinterpret_cast<char*>(legacy.data()),
                static_cast<std::streamsize>(legacy_bytes));
  if (!*input || hash_bytes(legacy.data(), static_cast<std::size_t>(legacy_bytes)) !=
                     header.data_hash)
    throw std::runtime_error("corrupt schema-v9 proposition synapses");
  std::vector<adult::proposition_tissue::SparseBindingSynapse> migrated(
      legacy.size());
  for (std::size_t index = 0u; index < legacy.size(); ++index) {
    const auto& source = legacy[index];
    auto& target = migrated[index];
    target.source_cell = source.source_cell;
    target.target_cell = source.target_cell;
    target.claimed = source.claimed;
    target.overflow_mass = source.overflow_mass;
    target.cohort_overflow_mass = source.cohort_overflow_mass;
    // V9 retained pressure but not its exact challenger. Zero marks that
    // pressure as unowned; the next exact attempt returns it to free matter
    // before establishing a new owner, so unrelated relations never pool it.
    // Version 9 conserved collision pressure but did not retain its challenger
    // identity. Mark it explicitly unowned so the first post-restore turnover
    // pass returns that mass before any new challenger can inherit it.
    target.overflow_binding =
        source.overflow_mass == 0u ? 0u : UINT64_MAX;
    target.structure_mass = source.structure_mass;
  }
  array->allocate(migrated.size());
  if (!migrated.empty())
    adult::cuda_require(
        cudaMemcpy(array->get(), migrated.data(), array->bytes(),
                   cudaMemcpyHostToDevice),
        "restore migrated schema-v9 proposition synapses");
  update_receipt(receipt, FieldClass::persistent_learned_matter,
                 header.name_hash, header.data_hash, legacy_bytes);
  return legacy_bytes;
}

inline void read_question_goal_v1_array(
    std::ifstream* input,
    adult::DeviceArray<adult::question_goal::ResidentQuestionGoalState>* array,
    Receipt* receipt) {
  using Legacy = adult::question_goal::ResidentQuestionGoalStateV1;
  ArrayHeader header{};
  input->read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!*input || header.name_hash != hash_name("question_goal_state") ||
      header.classification !=
          static_cast<std::uint32_t>(FieldClass::persistent_learned_matter) ||
      header.element_bytes != sizeof(Legacy) || header.count != 1u)
    throw std::runtime_error("invalid schema-v6 QuestionGoal field");
  Legacy legacy{};
  input->read(reinterpret_cast<char*>(&legacy), sizeof(legacy));
  if (!*input || hash_bytes(&legacy, sizeof(legacy)) != header.data_hash)
    throw std::runtime_error("corrupt schema-v6 QuestionGoal field");
  const auto migrated = adult::question_goal::migrate_v1(legacy);
  array->allocate(1u);
  adult::cuda_require(
      cudaMemcpy(array->get(), &migrated, sizeof(migrated),
                 cudaMemcpyHostToDevice),
      "restore migrated schema-v6 QuestionGoal");
  update_receipt(receipt, FieldClass::persistent_learned_matter,
                 header.name_hash, header.data_hash, sizeof(legacy));
}

inline std::uint64_t discard_legacy_array(std::ifstream* input, const char* name,
                                          FieldClass classification,
                                          std::uint32_t expected_element_bytes,
                                          Receipt* receipt,
                                          std::uint32_t alternate_element_bytes = 0u) {
  ArrayHeader header{};
  input->read(reinterpret_cast<char*>(&header), sizeof(header));
  const bool known_element_abi = header.element_bytes == expected_element_bytes ||
                                 (alternate_element_bytes != 0u &&
                                  header.element_bytes == alternate_element_bytes);
  if (!*input || header.name_hash != hash_name(name) ||
      header.classification != static_cast<std::uint32_t>(classification) ||
      !known_element_abi ||
      header.count > std::numeric_limits<std::size_t>::max() ||
      header.count > kMaximumFieldBytes / header.element_bytes) {
    throw std::runtime_error("invalid legacy adult checkpoint field: " + std::string(name));
  }
  const std::uint64_t bytes = header.count * header.element_bytes;
  std::vector<std::uint8_t> staged(static_cast<std::size_t>(bytes));
  if (!staged.empty())
    input->read(reinterpret_cast<char*>(staged.data()), staged.size());
  if (!*input || hash_bytes(staged.data(), staged.size()) != header.data_hash)
    throw std::runtime_error("corrupt legacy adult checkpoint field: " + std::string(name));
  update_receipt(receipt, classification, header.name_hash, header.data_hash, bytes);
  return bytes;
}

inline void validate_extents(const adult::AdultState& state) {
  const auto within = [](std::uint32_t count, std::size_t capacity) {
    return static_cast<std::size_t>(count) <= capacity;
  };
  if (state.unit_count > state.unit_capacity ||
      !within(state.unit_count, state.unit_lengths.size()) ||
      static_cast<std::uint64_t>(state.unit_capacity) * adult::kUnitWords !=
          state.unit_content.size() ||
      !within(state.bigram_count, state.bigrams.size()) ||
      !within(state.bigram_count, state.bigram_counts.size()) ||
      !within(state.trigram_count, state.trigrams.size()) ||
      !within(state.trigram_count, state.trigram_counts.size()) ||
      !within(state.online_bigram_count, state.online_bigrams.size()) ||
      !within(state.online_trigram_count, state.online_trigrams.size()) ||
      !within(state.online_association_count, state.online_associations.size()) ||
      !within(state.online_conditioned_transition_count,
              state.online_conditioned_transitions.size()) ||
      !within(state.online_episode_count, state.online_episode_units.size()) ||
      !within(state.online_episode_break_count, state.online_episode_breaks.size())) {
    throw std::runtime_error("complete adult checkpoint extents are inconsistent");
  }
}

inline std::size_t device_allocation_bytes(const adult::AdultState& state) {
  std::size_t total = 0u;
#define BCC32_SUM_DEVICE_BYTES(name, classification)                       \
  do {                                                                     \
    if (state.name.bytes() > std::numeric_limits<std::size_t>::max() - total) \
      throw std::runtime_error("complete adult allocation byte overflow"); \
    total += state.name.bytes();                                           \
  } while (false);
  BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_SUM_DEVICE_BYTES)
#undef BCC32_SUM_DEVICE_BYTES
  return total;
}

inline std::size_t allocate_empty_construction_slot_population(
    adult::AdultState* state) {
  if (state == nullptr)
    throw std::runtime_error("null construction-slot migration state");
  const std::size_t slot_count =
      static_cast<std::size_t>(adult::construction::kConstructionCap) *
      adult::construction::kConstructionMaxSlots;
  state->construction_slot_units.allocate(
      slot_count * adult::construction::kConstructionSlotPopulationCap);
  state->construction_slot_masses.allocate(
      slot_count * adult::construction::kConstructionSlotPopulationCap);
  state->construction_slot_totals.allocate(slot_count);
  state->construction_slot_overflow.allocate(slot_count);
  adult::cuda_require(
      cudaMemset(state->construction_slot_units.get(), 0xff,
                 state->construction_slot_units.bytes()),
      "clear migrated construction slot units");
  adult::cuda_require(
      cudaMemset(state->construction_slot_masses.get(), 0,
                 state->construction_slot_masses.bytes()),
      "clear migrated construction slot masses");
  adult::cuda_require(
      cudaMemset(state->construction_slot_totals.get(), 0,
                 state->construction_slot_totals.bytes()),
      "clear migrated construction slot totals");
  adult::cuda_require(
      cudaMemset(state->construction_slot_overflow.get(), 0,
                 state->construction_slot_overflow.bytes()),
      "clear migrated construction slot overflow");
  return state->construction_slot_units.bytes() +
         state->construction_slot_masses.bytes() +
         state->construction_slot_totals.bytes() +
         state->construction_slot_overflow.bytes();
}

inline Receipt save_checkpoint(const adult::AdultState& state, const std::string& path) {
  validate_extents(state);
  const AdultScalarState scalar = capture_scalars(state);
  Header header{};
  header.scalar_bytes = sizeof(scalar);
  header.schema_hash = schema_hash();
  header.scalar_hash = hash_bytes(&scalar, sizeof(scalar));
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot create complete adult v2 checkpoint: " + path);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(&scalar), sizeof(scalar));
  Receipt receipt{};
  receipt.schema_hash = header.schema_hash;
  CheckpointDeviceFields device_fields;
#define BCC32_STAGE_FIELD(name, classification)                              \
  device_fields.add(#name, FieldClass::classification, state.name,           \
                    serialized_array_count(state, #name, state.name.size()));
  BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_STAGE_FIELD)
#undef BCC32_STAGE_FIELD
  device_fields.write(&output, &receipt);
#define BCC32_WRITE_HOST_VECTOR(name, classification)                    \
  write_host_vector(&output, #name, FieldClass::classification,           \
                    state.name, &receipt);
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_WRITE_HOST_VECTOR)
#undef BCC32_WRITE_HOST_VECTOR
  mix_hash(&receipt.persistent_hash, header.scalar_hash);
  if (!output)
    throw std::runtime_error("complete adult v2 checkpoint write failed");
  return receipt;
}

inline Receipt receipt(const adult::AdultState& state);

inline adult::AdultState load_checkpoint(const std::string& path,
                                         Receipt* loaded_receipt = nullptr) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open complete adult v2 checkpoint: " + path);
  Header header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  const bool current_schema =
      header.version == kVersion && header.array_field_count == kArrayFieldCount &&
      header.schema_hash == schema_hash();
  const bool version19_legacy_schema =
      header.version == kVersion19LegacyVersion &&
      header.array_field_count == kVersion19ArrayFieldCount &&
      header.schema_hash == version19_schema_hash();
  const bool version17_legacy_schema =
      header.version == kVersion17LegacyVersion &&
      header.array_field_count == kVersion17ArrayFieldCount &&
      header.schema_hash == version17_schema_hash();
  const bool version16_legacy_schema =
      header.version == kVersion16LegacyVersion &&
      header.array_field_count == kVersion16ArrayFieldCount &&
      header.schema_hash == version16_schema_hash();
  const bool version15_legacy_schema =
      header.version == kVersion15LegacyVersion &&
      header.array_field_count == kVersion15ArrayFieldCount &&
      header.schema_hash == version15_schema_hash();
  const bool version14_legacy_schema =
      header.version == kVersion14LegacyVersion &&
      header.array_field_count == kVersion14ArrayFieldCount &&
      header.schema_hash == version14_schema_hash();
  const bool version13_legacy_schema =
      header.version == kVersion13LegacyVersion &&
      header.array_field_count == kVersion13ArrayFieldCount &&
      header.schema_hash == version13_schema_hash();
  const bool version12_legacy_schema =
      header.version == kVersion12LegacyVersion &&
      header.array_field_count == kVersion12ArrayFieldCount &&
      header.schema_hash == version12_schema_hash();
  const bool version11_legacy_schema =
      header.version == kVersion11LegacyVersion &&
      header.array_field_count == kVersion11ArrayFieldCount &&
      header.schema_hash == version11_schema_hash();
  const bool version10_legacy_schema =
      header.version == kVersion10LegacyVersion &&
      header.array_field_count == kVersion10ArrayFieldCount &&
      header.schema_hash == version10_schema_hash();
  const bool version9_legacy_schema =
      header.version == kVersion9LegacyVersion &&
      header.array_field_count == kVersion9ArrayFieldCount &&
      header.schema_hash == version9_schema_hash();
  const bool version8_legacy_schema =
      header.version == kVersion8LegacyVersion &&
      header.array_field_count == kVersion8ArrayFieldCount &&
      header.schema_hash == version8_schema_hash();
  const bool version7_legacy_schema =
      header.version == kVersion7LegacyVersion &&
      header.array_field_count == kVersion7ArrayFieldCount &&
      header.schema_hash == version7_schema_hash();
  const bool version6_legacy_schema =
      header.version == kVersion6LegacyVersion &&
      header.array_field_count == kVersion7ArrayFieldCount &&
      header.schema_hash == version6_schema_hash();
  const bool legacy_schema =
      header.version == kLegacyVersion &&
      header.array_field_count == kVersion5ArrayFieldCount &&
      header.schema_hash == version5_schema_hash();
  const bool version4_legacy_schema =
      header.version == kVersion4LegacyVersion &&
      header.array_field_count == kVersion4ArrayFieldCount &&
      header.schema_hash == version4_schema_hash();
  const bool older_legacy_schema =
      header.version == kOlderLegacyVersion &&
      header.array_field_count == kVersion3ArrayFieldCount &&
      header.schema_hash == version3_schema_hash();
  const bool old_legacy_schema =
      header.version == kOldLegacyVersion &&
      header.array_field_count == kLegacyArrayFieldCount &&
      header.schema_hash == legacy_schema_hash();
  if (!input || header.magic != kMagic ||
      (!current_schema && !version19_legacy_schema && !version17_legacy_schema &&
       !version16_legacy_schema &&
       !version15_legacy_schema &&
       !version14_legacy_schema &&
       !version13_legacy_schema &&
       !version12_legacy_schema &&
       !version11_legacy_schema && !version10_legacy_schema &&
       !version9_legacy_schema && !version8_legacy_schema &&
        !version7_legacy_schema &&
        !version6_legacy_schema && !legacy_schema &&
        !version4_legacy_schema &&
        !older_legacy_schema && !old_legacy_schema))
    throw std::runtime_error("incompatible complete adult checkpoint");
  AdultScalarState scalar{};
  if (current_schema || version19_legacy_schema || version17_legacy_schema ||
      version16_legacy_schema || version15_legacy_schema ||
      version14_legacy_schema ||
      version13_legacy_schema ||
      version12_legacy_schema ||
      version11_legacy_schema || version10_legacy_schema ||
      version9_legacy_schema ||
      version8_legacy_schema) {
    if (header.scalar_bytes != sizeof(scalar))
      throw std::runtime_error("incompatible complete adult current scalar ABI");
    input.read(reinterpret_cast<char*>(&scalar), sizeof(scalar));
    if (!input || header.scalar_hash != hash_bytes(&scalar, sizeof(scalar)))
      throw std::runtime_error("corrupt complete adult current scalars");
  } else {
    AdultScalarStateV7 legacy_scalar{};
    if (header.scalar_bytes != sizeof(legacy_scalar))
      throw std::runtime_error("incompatible complete adult legacy scalar ABI");
    input.read(reinterpret_cast<char*>(&legacy_scalar), sizeof(legacy_scalar));
    if (!input ||
        header.scalar_hash != hash_bytes(&legacy_scalar, sizeof(legacy_scalar)))
      throw std::runtime_error("corrupt complete adult legacy scalars");
    scalar.v7 = legacy_scalar;
  }
  validate_scalars(scalar);
  adult::AdultState state;
  restore_scalars(&state, scalar);
  Receipt receipt{};
  std::uint64_t legacy_proposition_bytes = 0u;
  std::uint64_t version9_proposition_bytes = 0u;
  receipt.schema_hash = header.schema_hash;
  CheckpointFieldReader field_reader(&receipt);
#define BCC32_READ_FIELD(name, classification)                              \
  field_reader.read(&input, #name, FieldClass::classification, &state.name);
  if (current_schema) {
    BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_HOST_VECTOR(name, classification)                       \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_HOST_VECTOR)
#undef BCC32_READ_HOST_VECTOR
  } else if (version19_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V19_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V19_HOST_VECTOR(name, classification)                    \
    read_host_vector(&input, #name, FieldClass::classification,              \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V19_HOST_VECTOR)
#undef BCC32_READ_V19_HOST_VECTOR
  } else if (version17_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V17_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V17_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V17_HOST_VECTOR)
#undef BCC32_READ_V17_HOST_VECTOR
  } else if (version16_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V16_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V16_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V16_HOST_VECTOR)
#undef BCC32_READ_V16_HOST_VECTOR
  } else if (version15_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V15_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V15_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V15_HOST_VECTOR)
#undef BCC32_READ_V15_HOST_VECTOR
  } else if (version14_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V14_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V14_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V14_HOST_VECTOR)
#undef BCC32_READ_V14_HOST_VECTOR
  } else if (version13_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V13_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V13_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V13_HOST_VECTOR)
#undef BCC32_READ_V13_HOST_VECTOR
  } else if (version12_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V12_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V12_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V12_HOST_VECTOR)
#undef BCC32_READ_V12_HOST_VECTOR
  } else if (version11_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V11_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V11_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V11_HOST_VECTOR)
#undef BCC32_READ_V11_HOST_VECTOR
  } else if (version10_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V10_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V10_HOST_VECTOR(name, classification)                   \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V10_HOST_VECTOR)
#undef BCC32_READ_V10_HOST_VECTOR
  } else if (version9_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
    version9_proposition_bytes = read_version9_proposition_synapses(
        &input, &state.proposition_synapses, &receipt);
    BCC32_ADULT_CHECKPOINT_V9_AFTER_SYNAPSE_FIELDS(BCC32_READ_FIELD)
#define BCC32_READ_V9_HOST_VECTOR(name, classification)                    \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V9_HOST_VECTOR)
#undef BCC32_READ_V9_HOST_VECTOR
  } else if (version8_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V8_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
#define BCC32_READ_V8_HOST_VECTOR(name, classification)                    \
    read_host_vector(&input, #name, FieldClass::classification,             \
                     &state.name, &receipt);
    BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_READ_V8_HOST_VECTOR)
#undef BCC32_READ_V8_HOST_VECTOR
  } else if (version7_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V7_ARRAY_FIELDS(BCC32_READ_FIELD)
  } else if (version6_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(BCC32_READ_FIELD)
    field_reader.flush();
    read_question_goal_v1_array(&input, &state.question_goal_state, &receipt);
  } else if (legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V5_ARRAY_FIELDS(BCC32_READ_FIELD)
  } else if (version4_legacy_schema) {
    BCC32_ADULT_CHECKPOINT_V4_ARRAY_FIELDS(BCC32_READ_FIELD)
  } else {
    BCC32_ADULT_CHECKPOINT_V2_ARRAY_FIELDS(BCC32_READ_FIELD)
    if (older_legacy_schema) {
      field_reader.flush();
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_synapses", FieldClass::persistent_learned_matter,
          64u, &receipt, 1312u);
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_scalars", FieldClass::persistent_learned_matter,
          sizeof(adult::proposition_tissue::TissueScalars), &receipt);
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_completion_scores", FieldClass::transient_workspace,
          sizeof(std::uint64_t), &receipt);
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_output_cells", FieldClass::transient_workspace,
          sizeof(std::uint32_t), &receipt);
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_output_scores", FieldClass::transient_workspace,
          sizeof(std::uint64_t), &receipt);
      legacy_proposition_bytes += discard_legacy_array(
          &input, "proposition_completion_result", FieldClass::observer_diagnostic,
          sizeof(adult::proposition_tissue::CompletionResult), &receipt);
    }
  }
  field_reader.finish();
#undef BCC32_READ_FIELD
  // The bridge was absent from every schema before v11. Allocate an empty
  // map on migration; only a later same-contact association may populate it.
  state.resident_bytes += adult::ensure_ordered_construction_links(state);
  state.resident_bytes += adult::ensure_construction_evidence_revisions(state);
  state.resident_bytes += adult::ensure_construction_origin_revisions(state);
  state.resident_bytes += adult::ensure_relation_triple_evidence_revisions(state);
  state.resident_bytes += adult::ensure_relation_role_contexts(state);
  state.resident_bytes += adult::ensure_witnessed_relation_events(state);
  state.resident_bytes += adult::ensure_witnessed_relation_constructions(state);
  state.resident_bytes += adult::ensure_witnessed_relation_surfaces(state);
  state.resident_bytes += adult::ensure_qonset_evidence_revisions(state);
  state.resident_bytes += adult::ensure_question_gap_field_support(state);
  state.resident_bytes += adult::ensure_question_answer_constructions(state);
  state.resident_bytes +=
      adult::ensure_surface_population_context_mass(state);
  // The receipt is observer-only and intentionally absent from prior
  // checkpoint schemas. Allocate it after the learned arrays are restored.
  state.resident_bytes += adult::ensure_construction_association_receipt(state);
  mix_hash(&receipt.persistent_hash, header.scalar_hash);
  if (input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("trailing complete adult v2 checkpoint bytes");
  if (version17_legacy_schema || version16_legacy_schema) {
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (version9_legacy_schema) {
    const std::size_t slot_population_bytes =
        allocate_empty_construction_slot_population(&state);
    const std::uint64_t migrated_bytes = state.proposition_synapses.bytes();
    if (migrated_bytes < version9_proposition_bytes)
      throw std::runtime_error("schema-v9 proposition migration shrank matter");
    const std::uint64_t growth = migrated_bytes - version9_proposition_bytes;
    if (growth > std::numeric_limits<std::size_t>::max() - state.resident_bytes)
      throw std::runtime_error("schema-v9 proposition resident extent overflow");
    if (slot_population_bytes >
        std::numeric_limits<std::size_t>::max() - state.resident_bytes -
            static_cast<std::size_t>(growth))
      throw std::runtime_error("schema-v9 slot resident extent overflow");
    state.resident_bytes +=
        static_cast<std::size_t>(growth) + slot_population_bytes;
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (version8_legacy_schema) {
    state.proposition_cue_sequence.allocate(adult::kCompositionUnits);
    state.proposition_cue_sequence_count.allocate(1u);
    adult::cuda_require(
        cudaMemset(state.proposition_cue_sequence_count.get(), 0,
                   state.proposition_cue_sequence_count.bytes()),
        "clear migrated ordered proposition cue extent");
    state.resident_bytes += state.proposition_cue_sequence.bytes() +
                            state.proposition_cue_sequence_count.bytes();
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (version7_legacy_schema) {
    // Schema v7 never serialized these organs.  Their default zero-length
    // DeviceArrays are the only honest migration: no learned construction,
    // relation, streaming, commitment, or surface-context matter is inferred.
    state.resident_bytes = device_allocation_bytes(state);
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (version6_legacy_schema) {
    state.resident_bytes +=
        sizeof(adult::question_goal::ResidentQuestionGoalState) -
        sizeof(adult::question_goal::ResidentQuestionGoalStateV1);
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (legacy_schema) {
    adult::allocate_resident_question_goal(state);
    state.resident_bytes += state.question_goal_state.bytes();
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (version4_legacy_schema) {
    adult::allocate_resident_ordered_binding_tissue(state);
    adult::allocate_resident_question_goal(state);
    state.resident_bytes += state.proposition_ordered_bindings.bytes() +
                            state.proposition_ordered_binding_admission.bytes() +
                            state.proposition_ordered_evidence_revision.bytes() +
                            state.question_goal_state.bytes();
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  } else if (older_legacy_schema || old_legacy_schema) {
    if (older_legacy_schema) {
      if (legacy_proposition_bytes > state.resident_bytes)
        throw std::runtime_error("legacy proposition bytes exceed resident extent");
      state.resident_bytes -= static_cast<std::size_t>(legacy_proposition_bytes);
    }
    adult::allocate_resident_proposition_tissue(state);
    state.resident_bytes += adult::resident_proposition_bytes(state);
    receipt = bcc32_cuda_adult_complete_checkpoint_v2::receipt(state);
  }
  validate_extents(state);
  adult::cuda_require(cudaDeviceSynchronize(), "complete adult v2 checkpoint restore");
  if (loaded_receipt != nullptr)
    *loaded_receipt = receipt;
  return state;
}

inline Receipt receipt(const adult::AdultState& state) {
  Receipt result{};
  result.schema_hash = schema_hash();
  CheckpointDeviceFields device_fields;
#define BCC32_RECEIPT_FIELD(name, classification)                            \
  device_fields.add(#name, FieldClass::classification, state.name,           \
                    serialized_array_count(state, #name, state.name.size()));
  BCC32_ADULT_CHECKPOINT_ARRAY_FIELDS(BCC32_RECEIPT_FIELD)
#undef BCC32_RECEIPT_FIELD
  device_fields.hash_into(&result);
#define BCC32_RECEIPT_HOST_VECTOR(name, classification)                    \
  update_receipt(&result, FieldClass::classification, hash_name(#name),    \
                 hash_bytes(state.name.data(),                             \
                            state.name.size() * sizeof(state.name[0])),    \
                 state.name.size() * sizeof(state.name[0]));
  BCC32_ADULT_CHECKPOINT_HOST_VECTOR_FIELDS(BCC32_RECEIPT_HOST_VECTOR)
#undef BCC32_RECEIPT_HOST_VECTOR
  const AdultScalarState scalar = capture_scalars(state);
  mix_hash(&result.persistent_hash, hash_bytes(&scalar, sizeof(scalar)));
  return result;
}

}  // namespace bcc32_cuda_adult_complete_checkpoint_v2
