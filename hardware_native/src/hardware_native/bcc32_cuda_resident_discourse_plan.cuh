#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bcc32_cuda_resident_discourse_plan {

#if defined(__CUDACC__)
#define BCC32_PLAN_HD __host__ __device__
#else
#define BCC32_PLAN_HD
#endif

constexpr std::uint64_t kPlanMagic = 0x32564e4c50524342ull;
constexpr std::uint32_t kPlanVersion = 6u;
constexpr std::uint32_t kMaxSteps = 8u;
constexpr std::uint32_t kMaxPopulationReferences = 64u;
constexpr std::uint32_t kMaxAnchorReferences = 64u;
constexpr std::uint32_t kMaxRealizedBytes = 4096u;

enum class PlanStatus : std::uint32_t {
  empty = 0u,
  settling = 1u,
  committed = 2u,
  emitting = 3u,
  completed = 4u,
  invalidated = 5u,
};

enum class PlanModality : std::uint32_t {
  unlicensed = 0u,
  discourse = 1u,
  causal = 2u,
  interrogative = 3u,
};

enum class PlanReferenceKind : std::uint32_t {
  none = 0u,
  opaque_population = 1u,
  ordered_binding = 2u,
  question_goal = 3u,
};

// Every reference is an opaque address into resident matter.  Neither an
// array offset nor a field name declares a concept, role, relation, or answer.
struct PlanStep {
  std::uint32_t population_begin = 0u;
  std::uint32_t population_count = 0u;
  std::uint32_t anchor_begin = 0u;
  std::uint32_t anchor_count = 0u;
  std::uint32_t dependency_mask = 0u;
  std::uint64_t evidence_revision = 0u;
  PlanModality modality = PlanModality::unlicensed;
  PlanReferenceKind reference_kind = PlanReferenceKind::none;
};

// One bounded persistent ABI between resident content selection and surface
// realization.  Counts are explicit; zero values inside any referenced
// population, anchor sequence, or realized byte string remain ordinary data.
struct ResidentDiscoursePlanState {
  std::uint64_t magic = kPlanMagic;
  std::uint32_t version = kPlanVersion;
  std::uint32_t revision = 0u;
  PlanStatus status = PlanStatus::empty;
  std::uint32_t step_count = 0u;
  std::uint32_t active_step = 0u;
  std::uint32_t population_reference_count = 0u;
  std::uint32_t anchor_reference_count = 0u;
  std::uint32_t surface_revision = 0u;
  std::uint64_t supporting_evidence_revision = 0u;
  // Conservative global contradiction epoch captured when the trajectory is
  // selected.  A later observation may advance the tissue revision without
  // contradicting this plan; accepted counterevidence may not.
  std::uint64_t conservative_counterevidence_epoch = 0u;
  // Only proposition-tissue trajectories carry this dependency.  Population
  // plans formed by other resident organs must not be retired by an unrelated
  // proposition lesion merely because they also use opaque populations.
  std::uint64_t conservative_proposition_lesion_epoch = 0u;
  std::uint32_t proposition_dependency = 0u;
  // Interrogative plans depend on one persistent QuestionGoal revision rather
  // than treating its slot number or surface construction as content.  A goal
  // change or lesion invalidates the Plan before further emission.
  std::uint32_t question_goal_dependency = 0u;
  std::uint32_t question_goal_revision = 0u;
  std::uint32_t question_goal_lesion_revision = 0u;
  PlanModality modality = PlanModality::unlicensed;
  std::uint32_t realized_step = 0u;
  // One retained surface frame may cover a contiguous dependency-linked span
  // of committed steps. The span advances atomically after its final byte.
  std::uint32_t realized_step_span = 0u;
  PlanModality realized_modality = PlanModality::unlicensed;
  std::uint32_t realized_byte_count = 0u;
  std::uint32_t emission_cursor = 0u;
  std::uint64_t invalidated_by_revision = 0u;
  PlanStep steps[kMaxSteps]{};
  std::uint32_t population_references[kMaxPopulationReferences]{};
  std::uint32_t anchor_references[kMaxAnchorReferences]{};
  std::uint8_t realized_bytes[kMaxRealizedBytes]{};
};

static_assert(std::is_trivially_copyable_v<PlanStep>);
static_assert(std::is_trivially_copyable_v<ResidentDiscoursePlanState>);

BCC32_PLAN_HD inline bool valid_status(PlanStatus status) {
  return static_cast<std::uint32_t>(status) <= static_cast<std::uint32_t>(PlanStatus::invalidated);
}

BCC32_PLAN_HD inline bool valid_modality(PlanModality modality) {
  return static_cast<std::uint32_t>(modality) <=
         static_cast<std::uint32_t>(PlanModality::interrogative);
}

BCC32_PLAN_HD inline bool valid_reference_kind(PlanReferenceKind kind) {
  return static_cast<std::uint32_t>(kind) <=
         static_cast<std::uint32_t>(PlanReferenceKind::question_goal);
}

BCC32_PLAN_HD inline bool valid(const ResidentDiscoursePlanState& state) {
  if (state.magic != kPlanMagic || state.version != kPlanVersion ||
      !valid_status(state.status) || !valid_modality(state.modality) ||
      !valid_modality(state.realized_modality) ||
      state.proposition_dependency > 1u ||
      state.question_goal_dependency > 1u ||
      (state.proposition_dependency != 0u &&
       state.question_goal_dependency != 0u) ||
      (state.proposition_dependency == 0u &&
       (state.conservative_counterevidence_epoch != 0u ||
        state.conservative_proposition_lesion_epoch != 0u)) ||
      (state.question_goal_dependency == 0u &&
       (state.question_goal_revision != 0u ||
        state.question_goal_lesion_revision != 0u)) ||
      (state.question_goal_dependency != 0u &&
       (state.modality != PlanModality::interrogative ||
        state.question_goal_revision == 0u)) ||
      state.step_count > kMaxSteps || state.active_step > state.step_count ||
      state.population_reference_count > kMaxPopulationReferences ||
      state.anchor_reference_count > kMaxAnchorReferences ||
      state.realized_step_span > state.step_count ||
      state.realized_byte_count > kMaxRealizedBytes ||
      state.emission_cursor > state.realized_byte_count) {
    return false;
  }
  for (std::uint32_t index = 0u; index < state.step_count; ++index) {
    const PlanStep& step = state.steps[index];
    if (!valid_modality(step.modality) || step.modality != state.modality ||
        !valid_reference_kind(step.reference_kind) ||
        (step.population_count == 0u &&
         step.reference_kind != PlanReferenceKind::none) ||
        (step.population_count != 0u &&
         step.reference_kind == PlanReferenceKind::none) ||
        (step.reference_kind == PlanReferenceKind::question_goal &&
         (state.question_goal_dependency == 0u ||
          state.modality != PlanModality::interrogative ||
          step.population_count != 1u)) ||
        (state.question_goal_dependency != 0u &&
         step.reference_kind != PlanReferenceKind::question_goal) ||
        step.population_begin > state.population_reference_count ||
        step.population_count > state.population_reference_count - step.population_begin ||
        step.anchor_begin > state.anchor_reference_count ||
        step.anchor_count > state.anchor_reference_count - step.anchor_begin ||
        (step.dependency_mask &
         ~((index == 0u) ? 0u : ((1u << index) - 1u))) != 0u) {
      return false;
    }
  }
  if (state.question_goal_dependency != 0u && state.step_count != 0u &&
      (state.step_count != 1u || state.population_reference_count != 1u ||
       state.population_references[0] != 0u))
    return false;
  if (state.status == PlanStatus::empty)
    return state.step_count == 0u && state.active_step == 0u &&
           state.population_reference_count == 0u && state.anchor_reference_count == 0u &&
           state.modality == PlanModality::unlicensed &&
           state.realized_step_span == 0u &&
           state.realized_modality == PlanModality::unlicensed &&
           state.realized_byte_count == 0u && state.emission_cursor == 0u;
  if (state.status == PlanStatus::settling)
    return state.active_step == 0u &&
           state.realized_step_span == 0u &&
           state.realized_modality == PlanModality::unlicensed &&
           state.realized_byte_count == 0u &&
           state.emission_cursor == 0u;
  if (state.status == PlanStatus::committed)
    return state.active_step < state.step_count &&
           state.realized_step_span == 0u &&
           state.realized_modality == PlanModality::unlicensed &&
           state.realized_byte_count == 0u &&
           state.emission_cursor == 0u;
  if (state.status == PlanStatus::emitting)
    return state.active_step < state.step_count &&
           state.realized_step == state.active_step &&
           state.realized_step_span != 0u &&
           state.realized_step_span <= state.step_count - state.active_step &&
           state.realized_byte_count != 0u &&
           state.realized_modality == state.steps[state.active_step].modality;
  if (state.status == PlanStatus::completed)
    return state.active_step == state.step_count &&
           state.realized_step_span == 0u &&
           state.realized_modality == PlanModality::unlicensed &&
           state.realized_byte_count == 0u &&
           state.emission_cursor == 0u;
  return state.realized_step_span == 0u &&
         state.realized_modality == PlanModality::unlicensed &&
         state.realized_byte_count == 0u && state.emission_cursor == 0u;
}

BCC32_PLAN_HD inline void clear(ResidentDiscoursePlanState* state) {
  if (state == nullptr)
    return;
  const std::uint32_t next_revision = state->revision + 1u;
  *state = ResidentDiscoursePlanState{};
  state->revision = next_revision;
}

// Return the maximal contiguous suffix whose steps are connected to the
// current span by resident dependencies. Independent later commitments remain
// separate motor frames even when they share modality or anchor storage.
BCC32_PLAN_HD inline std::uint32_t dependency_linked_step_span(
    const ResidentDiscoursePlanState& state, std::uint32_t first_step) {
  if (!valid(state) || first_step >= state.step_count)
    return 0u;
  const PlanModality modality = state.steps[first_step].modality;
  std::uint32_t included_mask = 1u << first_step;
  std::uint32_t span = 1u;
  for (std::uint32_t index = first_step + 1u; index < state.step_count;
       ++index) {
    const PlanStep& next = state.steps[index];
    if (next.modality != modality ||
        (next.dependency_mask & included_mask) == 0u) {
      break;
    }
    included_mask |= 1u << index;
    ++span;
  }
  return span;
}

// Build a prospective trajectory entirely from opaque resident population
// references. The caller may launch fixed CUDA stages, but cannot name or
// inspect the represented content while the plan is being formed.
BCC32_PLAN_HD inline bool begin_population_plan(
    ResidentDiscoursePlanState* state, std::uint64_t evidence_revision,
    std::uint32_t surface_revision,
    PlanModality modality = PlanModality::unlicensed,
    std::uint64_t conservative_counterevidence_epoch = 0u,
    std::uint32_t proposition_dependency = 0u,
    std::uint64_t conservative_proposition_lesion_epoch = 0u,
    std::uint32_t question_goal_dependency = 0u,
    std::uint32_t question_goal_revision = 0u,
    std::uint32_t question_goal_lesion_revision = 0u) {
  if (state == nullptr || proposition_dependency > 1u ||
      question_goal_dependency > 1u ||
      (proposition_dependency != 0u && question_goal_dependency != 0u) ||
      (proposition_dependency == 0u &&
       (conservative_counterevidence_epoch != 0u ||
        conservative_proposition_lesion_epoch != 0u)) ||
      (question_goal_dependency == 0u &&
       (question_goal_revision != 0u || question_goal_lesion_revision != 0u)) ||
      (question_goal_dependency != 0u &&
       (modality != PlanModality::interrogative ||
        question_goal_revision == 0u)) ||
      !valid(*state) ||
      (state->status != PlanStatus::empty &&
       state->status != PlanStatus::completed &&
       state->status != PlanStatus::invalidated))
    return false;
  clear(state);
  state->status = PlanStatus::settling;
  state->modality = modality;
  state->supporting_evidence_revision = evidence_revision;
  state->conservative_counterevidence_epoch =
      conservative_counterevidence_epoch;
  state->proposition_dependency = proposition_dependency;
  state->conservative_proposition_lesion_epoch =
      conservative_proposition_lesion_epoch;
  state->question_goal_dependency = question_goal_dependency;
  state->question_goal_revision = question_goal_revision;
  state->question_goal_lesion_revision = question_goal_lesion_revision;
  state->surface_revision = surface_revision;
  return valid(*state);
}

BCC32_PLAN_HD inline bool append_population_step(
    ResidentDiscoursePlanState* state,
    const std::uint32_t* population_references,
    std::uint32_t population_count, std::uint32_t dependency_mask,
    std::uint64_t evidence_revision,
    PlanReferenceKind reference_kind = PlanReferenceKind::opaque_population) {
  if (state == nullptr || population_references == nullptr ||
      population_count == 0u || reference_kind == PlanReferenceKind::none ||
      !valid_reference_kind(reference_kind) || !valid(*state) ||
      state->status != PlanStatus::settling ||
      state->step_count >= kMaxSteps ||
      population_count >
          kMaxPopulationReferences - state->population_reference_count)
    return false;
  const std::uint32_t step_index = state->step_count;
  const std::uint32_t begin = state->population_reference_count;
  PlanStep& step = state->steps[step_index];
  step.population_begin = begin;
  step.population_count = population_count;
  step.dependency_mask = dependency_mask;
  step.evidence_revision = evidence_revision;
  step.modality = state->modality;
  step.reference_kind = reference_kind;
  for (std::uint32_t index = 0u; index < population_count; ++index)
    state->population_references[begin + index] =
        population_references[index];
  state->population_reference_count += population_count;
  ++state->step_count;
  state->supporting_evidence_revision = evidence_revision;
  return valid(*state);
}

BCC32_PLAN_HD inline bool commit_population_plan(
    ResidentDiscoursePlanState* state) {
  if (state == nullptr || !valid(*state) ||
      state->status != PlanStatus::settling || state->step_count == 0u)
    return false;
  state->active_step = 0u;
  state->status = PlanStatus::committed;
  ++state->revision;
  return valid(*state);
}

// Temporary one-step adapter for the existing motor-completion seam.  The
// caller supplies the explicit unit count; this function never scans for a
// sentinel and never interprets an anchor as semantic content.
BCC32_PLAN_HD inline bool commit_single_step_from_units(ResidentDiscoursePlanState* state,
                                                        const std::uint32_t* unit_references,
                                                        std::uint32_t unit_count,
                                                        std::uint64_t evidence_revision,
                                                        std::uint32_t surface_revision,
                                                        PlanModality modality =
                                                            PlanModality::unlicensed,
                                                        std::uint64_t conservative_counterevidence_epoch =
                                                            0u) {
  if (state == nullptr || (unit_count != 0u && unit_references == nullptr) ||
      unit_count > kMaxAnchorReferences || !valid(*state) ||
      (state->status != PlanStatus::empty && state->status != PlanStatus::completed &&
       state->status != PlanStatus::invalidated)) {
    return false;
  }
  clear(state);
  state->status = PlanStatus::settling;
  state->modality = modality;
  state->step_count = 1u;
  state->active_step = 0u;
  state->anchor_reference_count = unit_count;
  state->supporting_evidence_revision = evidence_revision;
  state->conservative_counterevidence_epoch =
      conservative_counterevidence_epoch;
  state->surface_revision = surface_revision;
  state->steps[0].anchor_begin = 0u;
  state->steps[0].anchor_count = unit_count;
  state->steps[0].evidence_revision = evidence_revision;
  state->steps[0].modality = modality;
  for (std::uint32_t index = 0u; index < unit_count; ++index)
    state->anchor_references[index] = unit_references[index];
  state->status = PlanStatus::committed;
  return valid(*state);
}

BCC32_PLAN_HD inline bool stage_realization_span(
    ResidentDiscoursePlanState* state, std::uint32_t step,
    std::uint32_t step_span, const std::uint8_t* bytes,
    std::uint32_t byte_count, std::uint32_t surface_revision,
    PlanModality modality) {
  if (state == nullptr || !valid(*state) || state->status != PlanStatus::committed ||
      step != state->active_step || step >= state->step_count ||
      step_span == 0u || step_span > state->step_count - step ||
      byte_count == 0u || bytes == nullptr || byte_count > kMaxRealizedBytes ||
      modality != state->steps[step].modality) {
    return false;
  }
  std::uint32_t included_mask = 1u << step;
  for (std::uint32_t offset = 1u; offset < step_span; ++offset) {
    const std::uint32_t index = step + offset;
    const PlanStep& linked = state->steps[index];
    if (linked.modality != modality ||
        (linked.dependency_mask & included_mask) == 0u) {
      return false;
    }
    included_mask |= 1u << index;
  }
  for (std::uint32_t index = 0u; index < byte_count; ++index)
    state->realized_bytes[index] = bytes[index];
  state->realized_step = step;
  state->realized_step_span = step_span;
  state->realized_modality = modality;
  state->realized_byte_count = byte_count;
  state->emission_cursor = 0u;
  state->surface_revision = surface_revision;
  state->status = PlanStatus::emitting;
  return true;
}

BCC32_PLAN_HD inline bool stage_realization(ResidentDiscoursePlanState* state,
                                            std::uint32_t step,
                                            const std::uint8_t* bytes,
                                            std::uint32_t byte_count,
                                            std::uint32_t surface_revision,
                                            PlanModality modality) {
  return stage_realization_span(state, step, 1u, bytes, byte_count,
                                surface_revision, modality);
}

BCC32_PLAN_HD inline bool stage_realization(ResidentDiscoursePlanState* state,
                                            std::uint32_t step,
                                            const std::uint8_t* bytes,
                                            std::uint32_t byte_count,
                                            std::uint32_t surface_revision) {
  if (state == nullptr || step >= state->step_count)
    return false;
  return stage_realization(state, step, bytes, byte_count, surface_revision,
                           state->steps[step].modality);
}

BCC32_PLAN_HD inline std::uint32_t emit(ResidentDiscoursePlanState* state,
                                        std::uint8_t* destination,
                                        std::uint32_t capacity,
                                        PlanModality* emitted_modality) {
  if (state == nullptr || destination == nullptr || capacity == 0u || !valid(*state) ||
      state->status != PlanStatus::emitting) {
    return 0u;
  }
  const std::uint32_t available = state->realized_byte_count - state->emission_cursor;
  const std::uint32_t count = available < capacity ? available : capacity;
  if (emitted_modality != nullptr)
    *emitted_modality = state->realized_modality;
  for (std::uint32_t index = 0u; index < count; ++index) {
    destination[index] = state->realized_bytes[state->emission_cursor + index];
  }
  state->emission_cursor += count;
  if (state->emission_cursor == state->realized_byte_count) {
    state->active_step += state->realized_step_span;
    state->realized_step_span = 0u;
    state->realized_modality = PlanModality::unlicensed;
    state->realized_byte_count = 0u;
    state->emission_cursor = 0u;
    state->status =
        state->active_step == state->step_count ? PlanStatus::completed : PlanStatus::committed;
  }
  return count;
}

BCC32_PLAN_HD inline std::uint32_t emit(ResidentDiscoursePlanState* state,
                                        std::uint8_t* destination,
                                        std::uint32_t capacity) {
  return emit(state, destination, capacity, nullptr);
}

BCC32_PLAN_HD inline void invalidate(ResidentDiscoursePlanState* state,
                                     std::uint64_t evidence_revision) {
  if (state == nullptr || !valid(*state))
    return;
  state->invalidated_by_revision = evidence_revision;
  state->realized_step_span = 0u;
  state->realized_modality = PlanModality::unlicensed;
  state->realized_byte_count = 0u;
  state->emission_cursor = 0u;
  state->status = PlanStatus::invalidated;
  ++state->revision;
}

#undef BCC32_PLAN_HD

}  // namespace bcc32_cuda_resident_discourse_plan
