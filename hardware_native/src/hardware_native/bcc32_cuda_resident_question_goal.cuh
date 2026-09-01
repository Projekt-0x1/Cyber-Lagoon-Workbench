#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "bcc32_cuda_resident_discourse_plan.cuh"
#include "bcc32_cuda_resident_proposition_tissue.cuh"

// Persistent information-gap matter.  This organ never reads bytes and no
// array offset names a concept. It retains a known opaque topic only when
// proposition matter touches that topic but supplies no causally qualified
// bridge. The goal does not copy a weak relation and call it an unknown answer.
// Surface question statistics may later realize the goal; they are not allowed
// to choose its content.
namespace bcc32_cuda_resident_question_goal {

namespace tissue = bcc32_cuda_resident_proposition_tissue;
namespace plan = bcc32_cuda_resident_discourse_plan;

constexpr std::uint64_t kQuestionGoalMagic = 0x314c414f47514242ull;
constexpr std::uint32_t kQuestionGoalVersion = 2u;
constexpr std::uint32_t kMaximumQuestionCandidates = 64u;
constexpr std::uint32_t kComposedQuestionGoalCount = 2u;
constexpr std::uint32_t kPrioritizedQuestionGoalCount = 3u;
constexpr std::uint64_t kQuestionCompositionCheckpointMagic =
    0x31504d4f43514242ull;
constexpr std::uint32_t kQuestionCompositionCheckpointVersion = 1u;
constexpr std::uint64_t kThreeQuestionPriorityCheckpointMagic =
    0x3152495250514242ull;
constexpr std::uint32_t kThreeQuestionPriorityCheckpointVersion = 1u;

enum class GoalStatus : std::uint32_t {
  empty = 0u,
  open = 1u,
  planned = 2u,
  committed = planned,  // schema-v1 spelling retained for source compatibility
  discharged = 3u,
  invalidated = 4u,
  asked = 5u,
};

enum class GapKind : std::uint32_t {
  none = 0u,
  unsupported_topic_bridge = 1u,
};

struct ResidentQuestionGoalState {
  std::uint64_t magic = kQuestionGoalMagic;
  std::uint32_t version = kQuestionGoalVersion;
  std::uint32_t revision = 0u;
  GoalStatus status = GoalStatus::empty;
  GapKind kind = GapKind::none;
  std::uint32_t target_count = 0u;
  std::uint32_t target_cells[tissue::kMaximumPopulationCells]{};
  std::uint64_t supporting_evidence_revision = 0u;
  std::uint64_t tissue_revision = 0u;
  std::uint64_t positive_mass = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint64_t answered_evidence_revision = 0u;
  std::uint32_t qualifying_context_count = 0u;
  std::uint32_t plan_revision = 0u;
  std::uint32_t surface_revision = 0u;
  std::uint32_t opened_count = 0u;
  std::uint32_t discharged_count = 0u;
  std::uint32_t asked_count = 0u;
  std::uint32_t repeated_suppressed = 0u;
  std::uint32_t ambiguous_count = 0u;
  std::uint32_t lesion_revision = 0u;
};

// Exact schema-v1 payload retained only for deterministic checkpoint
// migration.  It never enters the live adult after load.
struct ResidentQuestionGoalStateV1 {
  std::uint64_t magic = kQuestionGoalMagic;
  std::uint32_t version = 1u;
  std::uint32_t revision = 0u;
  GoalStatus status = GoalStatus::empty;
  GapKind kind = GapKind::none;
  std::uint32_t target_count = 0u;
  std::uint32_t target_cells[tissue::kMaximumPopulationCells]{};
  std::uint64_t supporting_evidence_revision = 0u;
  std::uint64_t tissue_revision = 0u;
  std::uint64_t positive_mass = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint64_t answered_evidence_revision = 0u;
  std::uint32_t qualifying_context_count = 0u;
  std::uint32_t plan_revision = 0u;
  std::uint32_t surface_revision = 0u;
  std::uint32_t opened_count = 0u;
  std::uint32_t discharged_count = 0u;
  std::uint32_t repeated_suppressed = 0u;
  std::uint32_t ambiguous_count = 0u;
  std::uint32_t lesion_revision = 0u;
};

struct OriginationResult {
  std::uint32_t attempted = 0u;
  std::uint32_t opened = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t repeated_suppressed = 0u;
  std::uint32_t topic_matches = 0u;
  std::uint32_t unsupported_matches = 0u;
  std::uint64_t tissue_revision = 0u;
};

struct GapCandidateEvidence {
  std::uint32_t retained = 0u;
  std::uint32_t qualified = 0u;
  std::uint32_t matches = 0u;
  std::uint32_t unsupported_matches = 0u;
  std::uint64_t latest_evidence = 0u;
  std::uint64_t positive_mass = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint32_t widest_context = 0u;
};

enum class CompositionStatus : std::uint32_t {
  empty = 0u,
  active = 1u,
  completed = 2u,
  invalidated = 3u,
};

// A bounded composition retains only goal slots/revisions and public answer
// revisions. Goal content remains canonical in the two ResidentQuestionGoal
// objects; it is never merged or copied into a synthetic compound target.
struct ResidentQuestionCompositionState {
  std::uint32_t revision = 0u;
  CompositionStatus status = CompositionStatus::empty;
  std::uint32_t active_position = 0u;
  std::uint32_t goal_slots[kComposedQuestionGoalCount]{};
  std::uint32_t goal_revisions[kComposedQuestionGoalCount]{};
  std::uint64_t public_answer_revisions[kComposedQuestionGoalCount]{};
};

// This capsule is embedded matter for an authenticated outer checkpoint. It
// owns no live duplicate after restore and deliberately excludes the transient
// discourse Plan, whose un-emitted tail has no post-crash physical authority.
struct ResidentQuestionCompositionCheckpoint {
  std::uint64_t magic = kQuestionCompositionCheckpointMagic;
  std::uint32_t version = kQuestionCompositionCheckpointVersion;
  std::uint32_t reserved = 0u;
  ResidentQuestionGoalState goals[kComposedQuestionGoalCount]{};
  ResidentQuestionCompositionState composition{};
};

enum class PriorityStatus : std::uint32_t {
  empty = 0u,
  active = 1u,
  tied = 2u,
  completed = 3u,
  invalidated = 4u,
};

struct ResidentThreeQuestionPriorityState {
  std::uint32_t revision = 0u;
  PriorityStatus status = PriorityStatus::empty;
  std::uint32_t remaining_mask = 0u;
  std::uint32_t active_slot = 0xffffffffu;
  std::uint32_t answer_count = 0u;
  std::uint32_t goal_revisions[kPrioritizedQuestionGoalCount]{};
  std::uint64_t public_answer_revisions[kPrioritizedQuestionGoalCount]{};
};

struct ResidentThreeQuestionPriorityCheckpoint {
  std::uint64_t magic = kThreeQuestionPriorityCheckpointMagic;
  std::uint32_t version = kThreeQuestionPriorityCheckpointVersion;
  std::uint32_t reserved = 0u;
  ResidentQuestionGoalState goals[kPrioritizedQuestionGoalCount]{};
  ResidentThreeQuestionPriorityState priority{};
};

static_assert(std::is_trivially_copyable_v<ResidentQuestionGoalState>);
static_assert(std::is_trivially_copyable_v<ResidentQuestionGoalStateV1>);
static_assert(std::is_trivially_copyable_v<OriginationResult>);
static_assert(std::is_trivially_copyable_v<GapCandidateEvidence>);
static_assert(std::is_trivially_copyable_v<ResidentQuestionCompositionState>);
static_assert(
    std::is_trivially_copyable_v<ResidentQuestionCompositionCheckpoint>);
static_assert(std::is_trivially_copyable_v<ResidentThreeQuestionPriorityState>);
static_assert(
    std::is_trivially_copyable_v<ResidentThreeQuestionPriorityCheckpoint>);

__host__ __device__ inline bool valid_status(GoalStatus status) {
  return static_cast<std::uint32_t>(status) <=
         static_cast<std::uint32_t>(GoalStatus::asked);
}

__host__ __device__ inline bool valid_kind(GapKind kind) {
  return static_cast<std::uint32_t>(kind) <=
         static_cast<std::uint32_t>(GapKind::unsupported_topic_bridge);
}

__host__ __device__ inline bool valid(const ResidentQuestionGoalState& goal,
                                      std::uint32_t cell_capacity) {
  if (goal.magic != kQuestionGoalMagic ||
      goal.version != kQuestionGoalVersion || !valid_status(goal.status) ||
      !valid_kind(goal.kind) ||
      goal.target_count > tissue::kMaximumPopulationCells ||
      (goal.kind == GapKind::none) != (goal.target_count == 0u))
    return false;
  const bool active = goal.status == GoalStatus::open ||
                      goal.status == GoalStatus::planned ||
                      goal.status == GoalStatus::discharged ||
                      goal.status == GoalStatus::asked;
  if (active != (goal.target_count != 0u))
    return false;
  for (std::uint32_t index = 0u; index < goal.target_count; ++index) {
    if (goal.target_cells[index] >= cell_capacity)
      return false;
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (goal.target_cells[prior] == goal.target_cells[index])
        return false;
  }
  return true;
}

__host__ inline ResidentQuestionGoalState migrate_v1(
    const ResidentQuestionGoalStateV1& old) {
  ResidentQuestionGoalState goal{};
  if (old.magic != kQuestionGoalMagic || old.version != 1u)
    return goal;
  goal.revision = old.revision;
  goal.status = old.status;
  goal.kind = old.kind;
  goal.target_count = old.target_count;
  for (std::uint32_t index = 0u; index < tissue::kMaximumPopulationCells;
       ++index)
    goal.target_cells[index] = old.target_cells[index];
  goal.supporting_evidence_revision = old.supporting_evidence_revision;
  goal.tissue_revision = old.tissue_revision;
  goal.positive_mass = old.positive_mass;
  goal.counterevidence = old.counterevidence;
  goal.answered_evidence_revision = old.answered_evidence_revision;
  goal.qualifying_context_count = old.qualifying_context_count;
  goal.plan_revision = old.plan_revision;
  goal.surface_revision = old.surface_revision;
  goal.opened_count = old.opened_count;
  goal.discharged_count = old.discharged_count;
  goal.repeated_suppressed = old.repeated_suppressed;
  goal.ambiguous_count = old.ambiguous_count;
  goal.lesion_revision = old.lesion_revision;
  // V1's numeric `committed` state did not carry a checkpoint-verifiable
  // resident Plan dependency.  Treat it as an open gap during schema-only
  // migration; a later stream restore may stage a fresh Plan, but may not
  // pretend that an unproven historical tail is still committed.
  if (goal.status == GoalStatus::planned) {
    goal.status = GoalStatus::open;
    if (goal.revision != 0xffffffffu)
      ++goal.revision;
    goal.plan_revision = 0u;
    goal.surface_revision = 0u;
  }
  return goal;
}

__device__ inline bool topic_matches(
    tissue::SparsePopulationView topic,
    const tissue::OrderedRoleBindingEvidence& binding) {
  for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
       ++role) {
    const auto population = tissue::ordered_binding_role(binding, role);
    if (tissue::exact_population_equals(topic.cells, topic.count, population))
      return true;
  }
  return false;
}

__device__ inline std::uint64_t saturating_add(std::uint64_t left,
                                               std::uint64_t right) {
  return right > 0xffffffffffffffffull - left
             ? 0xffffffffffffffffull
             : left + right;
}

__device__ inline GapCandidateEvidence evaluate_gap_candidate(
    tissue::TissueView resident_tissue,
    tissue::SparsePopulationView topic) {
  GapCandidateEvidence evidence{};
  for (std::uint32_t index = 0u;
       index < resident_tissue.ordered_binding_capacity; ++index) {
    const auto& binding = resident_tissue.ordered_bindings[index];
    if (binding.claimed == 0u ||
        !tissue::ordered_binding_structurally_intact(
            binding, resident_tissue.cell_capacity) ||
        !topic_matches(topic, binding))
      continue;
    ++evidence.matches;
    if (tissue::ordered_binding_qualified(
            binding, tissue::OrderedBindingQualification::causal,
            resident_tissue.cell_capacity)) {
      evidence.qualified = 1u;
      continue;
    }
    if (tissue::ordered_binding_positive_mass(binding) == 0u &&
        binding.counterevidence == 0u)
      continue;
    ++evidence.unsupported_matches;
    evidence.retained = 1u;
    evidence.latest_evidence =
        binding.last_evidence_revision > evidence.latest_evidence
            ? binding.last_evidence_revision
            : evidence.latest_evidence;
    evidence.positive_mass = saturating_add(
        evidence.positive_mass,
        tissue::ordered_binding_positive_mass(binding));
    evidence.counterevidence =
        saturating_add(evidence.counterevidence, binding.counterevidence);
    evidence.widest_context =
        binding.qualifying_context_count > evidence.widest_context
            ? binding.qualifying_context_count
            : evidence.widest_context;
  }
  return evidence;
}

// Competing gap topics are ordered only by resident evidence available at
// origination. Recency is deliberately excluded: a later answer revision
// must remain held out, and numeric topic/cell identity never breaks a tie.
__device__ inline bool stronger_gap_evidence(
    const GapCandidateEvidence& candidate,
    const GapCandidateEvidence& incumbent) {
  if (candidate.unsupported_matches != incumbent.unsupported_matches)
    return candidate.unsupported_matches > incumbent.unsupported_matches;
  const std::uint64_t candidate_mass =
      saturating_add(candidate.positive_mass, candidate.counterevidence);
  const std::uint64_t incumbent_mass =
      saturating_add(incumbent.positive_mass, incumbent.counterevidence);
  if (candidate_mass != incumbent_mass)
    return candidate_mass > incumbent_mass;
  return candidate.widest_context > incumbent.widest_context;
}

__device__ inline bool equal_gap_evidence(
    const GapCandidateEvidence& left,
    const GapCandidateEvidence& right) {
  return left.unsupported_matches == right.unsupported_matches &&
         saturating_add(left.positive_mass, left.counterevidence) ==
             saturating_add(right.positive_mass, right.counterevidence) &&
         left.widest_context == right.widest_context;
}

__host__ __device__ inline bool valid_composition_status(
    CompositionStatus status) {
  return static_cast<std::uint32_t>(status) <=
         static_cast<std::uint32_t>(CompositionStatus::invalidated);
}

__host__ __device__ inline bool valid_composition(
    const ResidentQuestionCompositionState& composition) {
  if (!valid_composition_status(composition.status) ||
      composition.active_position > kComposedQuestionGoalCount ||
      composition.goal_slots[0] >= kComposedQuestionGoalCount ||
      composition.goal_slots[1] >= kComposedQuestionGoalCount ||
      composition.goal_slots[0] == composition.goal_slots[1])
    return composition.status == CompositionStatus::empty &&
           composition.revision == 0u;
  if (composition.status == CompositionStatus::active)
    return composition.active_position < kComposedQuestionGoalCount;
  if (composition.status == CompositionStatus::completed)
    return composition.active_position == kComposedQuestionGoalCount &&
           composition.public_answer_revisions[0] != 0u &&
           composition.public_answer_revisions[1] >
               composition.public_answer_revisions[0];
  return composition.status == CompositionStatus::invalidated;
}

__device__ inline std::uint64_t question_goal_evidence_mass(
    const ResidentQuestionGoalState& goal) {
  return saturating_add(goal.positive_mass, goal.counterevidence);
}

__device__ inline bool stronger_question_goal(
    const ResidentQuestionGoalState& candidate,
    const ResidentQuestionGoalState& incumbent) {
  const std::uint64_t candidate_mass = question_goal_evidence_mass(candidate);
  const std::uint64_t incumbent_mass = question_goal_evidence_mass(incumbent);
  if (candidate_mass != incumbent_mass)
    return candidate_mass > incumbent_mass;
  return candidate.qualifying_context_count >
         incumbent.qualifying_context_count;
}

__device__ inline bool equal_question_goal_priority(
    const ResidentQuestionGoalState& left,
    const ResidentQuestionGoalState& right) {
  return question_goal_evidence_mass(left) == question_goal_evidence_mass(right) &&
         left.qualifying_context_count == right.qualifying_context_count;
}

// Remaining open Goals may acquire new retained evidence while an earlier
// question waits for its answer. Refresh only the Goal's own exact topic; a
// qualified answer is handled by discharge, never converted into priority.
__device__ inline bool refresh_open_question_goal_evidence(
    tissue::TissueView resident_tissue, ResidentQuestionGoalState* goal) {
  if (goal == nullptr || resident_tissue.scalars == nullptr ||
      goal->status != GoalStatus::open ||
      !valid(*goal, resident_tissue.cell_capacity))
    return false;
  const tissue::SparsePopulationView topic{goal->target_cells,
                                           goal->target_count};
  const GapCandidateEvidence evidence =
      evaluate_gap_candidate(resident_tissue, topic);
  if (evidence.retained == 0u || evidence.qualified != 0u)
    return false;
  const bool changed =
      evidence.latest_evidence != goal->supporting_evidence_revision ||
      evidence.positive_mass != goal->positive_mass ||
      evidence.counterevidence != goal->counterevidence ||
      evidence.widest_context != goal->qualifying_context_count;
  if (!changed)
    return true;
  if (goal->revision == 0xffffffffu)
    return false;
  goal->supporting_evidence_revision = evidence.latest_evidence;
  goal->tissue_revision = resident_tissue.scalars->revision;
  goal->positive_mass = evidence.positive_mass;
  goal->counterevidence = evidence.counterevidence;
  goal->qualifying_context_count = evidence.widest_context;
  ++goal->revision;
  return true;
}

__global__ void refresh_open_question_goal_evidence_kernel(
    tissue::TissueView resident_tissue, ResidentQuestionGoalState* goal) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  refresh_open_question_goal_evidence(resident_tissue, goal);
}

__host__ __device__ inline bool valid_three_question_priority(
    const ResidentThreeQuestionPriorityState& priority) {
  if (static_cast<std::uint32_t>(priority.status) >
          static_cast<std::uint32_t>(PriorityStatus::invalidated) ||
      priority.answer_count > kPrioritizedQuestionGoalCount ||
      (priority.remaining_mask & ~0x7u) != 0u)
    return false;
  if (priority.status == PriorityStatus::empty)
    return priority.revision == 0u && priority.remaining_mask == 0u &&
           priority.answer_count == 0u;
  if (priority.status != PriorityStatus::invalidated) {
    std::uint32_t remaining_count = 0u;
    for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
      remaining_count += (priority.remaining_mask >> slot) & 1u;
    if (remaining_count + priority.answer_count !=
        kPrioritizedQuestionGoalCount)
      return false;
    for (std::uint32_t answer = 0u;
         answer < kPrioritizedQuestionGoalCount; ++answer) {
      if (answer < priority.answer_count) {
        if (priority.public_answer_revisions[answer] == 0u ||
            (answer != 0u && priority.public_answer_revisions[answer] <=
                                 priority.public_answer_revisions[answer - 1u]))
          return false;
      } else if (priority.public_answer_revisions[answer] != 0u) {
        return false;
      }
    }
  }
  if (priority.status == PriorityStatus::active)
    return priority.active_slot < kPrioritizedQuestionGoalCount &&
           (priority.remaining_mask & (1u << priority.active_slot)) != 0u &&
           priority.answer_count < kPrioritizedQuestionGoalCount;
  if (priority.status == PriorityStatus::tied)
    return priority.active_slot == 0xffffffffu &&
           (priority.remaining_mask & (priority.remaining_mask - 1u)) != 0u &&
           priority.answer_count < kPrioritizedQuestionGoalCount;
  if (priority.status == PriorityStatus::completed) {
    if (priority.remaining_mask != 0u ||
        priority.answer_count != kPrioritizedQuestionGoalCount)
      return false;
  }
  return true;
}

__device__ inline bool select_strongest_question_goal(
    const ResidentQuestionGoalState* goals, std::uint32_t remaining_mask,
    std::uint32_t cell_capacity, std::uint32_t* selected, bool* tied) {
  if (goals == nullptr || selected == nullptr || tied == nullptr ||
      remaining_mask == 0u || (remaining_mask & ~0x7u) != 0u)
    return false;
  *selected = 0xffffffffu;
  *tied = false;
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot) {
    if ((remaining_mask & (1u << slot)) == 0u)
      continue;
    if (!valid(goals[slot], cell_capacity) ||
        goals[slot].status != GoalStatus::open)
      return false;
    if (*selected == 0xffffffffu ||
        stronger_question_goal(goals[slot], goals[*selected])) {
      *selected = slot;
      *tied = false;
    } else if (equal_question_goal_priority(goals[slot], goals[*selected])) {
      *tied = true;
    }
  }
  return *selected != 0xffffffffu;
}

__device__ inline bool reprioritize_three_question_goals(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (goals == nullptr || priority == nullptr ||
      !valid_three_question_priority(*priority) ||
      (priority->status != PriorityStatus::tied &&
       priority->status != PriorityStatus::active) ||
      priority->revision == 0xffffffffu)
    return false;
  ResidentThreeQuestionPriorityState shadow = *priority;
  std::uint32_t selected = 0xffffffffu;
  bool tied = false;
  if (!select_strongest_question_goal(
          goals, shadow.remaining_mask, cell_capacity, &selected, &tied)) {
    shadow.status = PriorityStatus::invalidated;
    ++shadow.revision;
    *priority = shadow;
    return false;
  }
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    if ((shadow.remaining_mask & (1u << slot)) != 0u)
      shadow.goal_revisions[slot] = goals[slot].revision;
  shadow.active_slot = tied ? 0xffffffffu : selected;
  shadow.status = tied ? PriorityStatus::tied : PriorityStatus::active;
  ++shadow.revision;
  if (!valid_three_question_priority(shadow))
    return false;
  *priority = shadow;
  return !tied;
}

__device__ inline bool prioritize_three_question_goals(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (goals == nullptr || priority == nullptr || cell_capacity == 0u ||
      !valid_three_question_priority(*priority) ||
      priority->status != PriorityStatus::empty)
    return false;
  // Validate every resident Goal before using its population extent in exact
  // identity comparisons.  A corrupt count must fail closed without becoming
  // an out-of-bounds comparison authority.
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    if (!valid(goals[slot], cell_capacity) ||
        goals[slot].status != GoalStatus::open)
      return false;
  for (std::uint32_t left = 0u; left < kPrioritizedQuestionGoalCount; ++left)
    for (std::uint32_t right = left + 1u;
         right < kPrioritizedQuestionGoalCount; ++right)
      if (tissue::exact_population_equals(
              goals[left].target_cells, goals[left].target_count,
              {goals[right].target_cells, goals[right].target_count}))
        return false;
  ResidentThreeQuestionPriorityState shadow{};
  shadow.remaining_mask = 0x7u;
  shadow.status = PriorityStatus::tied;
  *priority = shadow;
  return reprioritize_three_question_goals(goals, cell_capacity, priority);
}

__global__ void prioritize_three_question_goals_kernel(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  prioritize_three_question_goals(goals, cell_capacity, priority);
}

__global__ void reprioritize_three_question_goals_kernel(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  reprioritize_three_question_goals(goals, cell_capacity, priority);
}

// A priority checkpoint is admitted only between public questions: at least
// one answer is already resident, a unique next Goal has been selected, and
// that Goal is either still open or carries an un-emitted transient Plan.
// Answer order is checked against the discharged Goal matter without treating
// slot identity as public sequence authority.
__device__ inline bool three_question_priority_checkpoint_consistent(
    const ResidentThreeQuestionPriorityCheckpoint& checkpoint,
    std::uint32_t cell_capacity) {
  if (checkpoint.magic != kThreeQuestionPriorityCheckpointMagic ||
      checkpoint.version != kThreeQuestionPriorityCheckpointVersion ||
      checkpoint.reserved != 0u || cell_capacity == 0u ||
      !valid_three_question_priority(checkpoint.priority) ||
      checkpoint.priority.status != PriorityStatus::active ||
      checkpoint.priority.answer_count == 0u)
    return false;
  ResidentQuestionGoalState ranking[kPrioritizedQuestionGoalCount]{};
  bool matched_answers[kPrioritizedQuestionGoalCount]{};
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot) {
    const ResidentQuestionGoalState& resident = checkpoint.goals[slot];
    if (!valid(resident, cell_capacity) ||
        resident.revision != checkpoint.priority.goal_revisions[slot])
      return false;
    ranking[slot] = resident;
    if ((checkpoint.priority.remaining_mask & (1u << slot)) != 0u) {
      if (resident.status != GoalStatus::open &&
          !(slot == checkpoint.priority.active_slot &&
            resident.status == GoalStatus::planned))
        return false;
      ranking[slot].status = GoalStatus::open;
    } else {
      if (resident.status != GoalStatus::discharged ||
          resident.answered_evidence_revision == 0u)
        return false;
      bool matched = false;
      for (std::uint32_t answer = 0u;
           answer < checkpoint.priority.answer_count; ++answer) {
        if (!matched_answers[answer] &&
            checkpoint.priority.public_answer_revisions[answer] ==
                resident.answered_evidence_revision) {
          matched_answers[answer] = true;
          matched = true;
          break;
        }
      }
      if (!matched)
        return false;
    }
  }
  for (std::uint32_t left = 0u; left < kPrioritizedQuestionGoalCount; ++left)
    for (std::uint32_t right = left + 1u;
         right < kPrioritizedQuestionGoalCount; ++right)
      if (tissue::exact_population_equals(
              checkpoint.goals[left].target_cells,
              checkpoint.goals[left].target_count,
              {checkpoint.goals[right].target_cells,
               checkpoint.goals[right].target_count}))
        return false;
  std::uint32_t selected = 0xffffffffu;
  bool tied = false;
  return select_strongest_question_goal(
             ranking, checkpoint.priority.remaining_mask, cell_capacity,
             &selected, &tied) &&
         !tied && selected == checkpoint.priority.active_slot;
}

__device__ inline bool capture_three_question_priority_checkpoint(
    const ResidentQuestionGoalState* goals,
    const ResidentThreeQuestionPriorityState* priority,
    std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityCheckpoint* checkpoint) {
  if (goals == nullptr || priority == nullptr || checkpoint == nullptr)
    return false;
  ResidentThreeQuestionPriorityCheckpoint shadow{};
  shadow.priority = *priority;
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    shadow.goals[slot] = goals[slot];
  if (!three_question_priority_checkpoint_consistent(shadow, cell_capacity))
    return false;
  *checkpoint = shadow;
  return true;
}

__global__ void capture_three_question_priority_checkpoint_kernel(
    const ResidentQuestionGoalState* goals,
    const ResidentThreeQuestionPriorityState* priority,
    std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityCheckpoint* checkpoint) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || checkpoint == nullptr)
    return;
  *checkpoint = ResidentThreeQuestionPriorityCheckpoint{};
  if (!capture_three_question_priority_checkpoint(
          goals, priority, cell_capacity, checkpoint))
    checkpoint->magic = 0u;
}

// Restart fences every unanswered Goal, clears contact-local Plan authority,
// and deliberately discards the saved active slot before ranking the stored
// resident evidence again. Thus neither a pre-crash Plan nor an old answer
// branch can choose or discharge the next public question.
__device__ inline bool restore_three_question_priority_checkpoint(
    const ResidentThreeQuestionPriorityCheckpoint& checkpoint,
    std::uint32_t cell_capacity, ResidentQuestionGoalState* goals,
    ResidentThreeQuestionPriorityState* priority,
    plan::ResidentDiscoursePlanState* transient_plan) {
  if (goals == nullptr || priority == nullptr || transient_plan == nullptr ||
      !three_question_priority_checkpoint_consistent(checkpoint, cell_capacity))
    return false;
  ResidentQuestionGoalState goal_shadows[kPrioritizedQuestionGoalCount]{};
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    goal_shadows[slot] = checkpoint.goals[slot];
  ResidentThreeQuestionPriorityState priority_shadow = checkpoint.priority;
  plan::ResidentDiscoursePlanState plan_shadow = *transient_plan;
  plan::clear(&plan_shadow);
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot) {
    if ((priority_shadow.remaining_mask & (1u << slot)) == 0u)
      continue;
    ResidentQuestionGoalState& resident = goal_shadows[slot];
    if (resident.revision == 0xffffffffu ||
        (resident.status == GoalStatus::planned &&
         resident.revision == 0xfffffffeu))
      return false;
    if (resident.status == GoalStatus::planned) {
      resident.status = GoalStatus::open;
      ++resident.revision;
      resident.plan_revision = 0u;
      resident.surface_revision = 0u;
    }
    ++resident.revision;
    priority_shadow.goal_revisions[slot] = resident.revision;
  }
  if (priority_shadow.revision > 0xfffffffdu)
    return false;
  std::uint32_t remaining_count = 0u;
  std::uint32_t remaining_slot = 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot) {
    if ((priority_shadow.remaining_mask & (1u << slot)) == 0u)
      continue;
    ++remaining_count;
    remaining_slot = slot;
  }
  // Reprioritization accepts a structurally valid intermediate state. With
  // two Goals that state is explicit abstention; with one it is the only
  // remaining mask member, never the saved pre-crash active slot.
  priority_shadow.active_slot = remaining_count == 1u
      ? remaining_slot : 0xffffffffu;
  priority_shadow.status = remaining_count == 1u
      ? PriorityStatus::active : PriorityStatus::tied;
  ++priority_shadow.revision;
  if (!reprioritize_three_question_goals(
          goal_shadows, cell_capacity, &priority_shadow))
    return false;
  ResidentThreeQuestionPriorityCheckpoint normalized = checkpoint;
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    normalized.goals[slot] = goal_shadows[slot];
  normalized.priority = priority_shadow;
  if (!three_question_priority_checkpoint_consistent(normalized, cell_capacity) ||
      !plan::valid(plan_shadow) ||
      plan_shadow.status != plan::PlanStatus::empty)
    return false;
  for (std::uint32_t slot = 0u; slot < kPrioritizedQuestionGoalCount; ++slot)
    goals[slot] = goal_shadows[slot];
  *priority = priority_shadow;
  *transient_plan = plan_shadow;
  return true;
}

__global__ void restore_three_question_priority_checkpoint_kernel(
    const ResidentThreeQuestionPriorityCheckpoint* checkpoint,
    std::uint32_t cell_capacity, ResidentQuestionGoalState* goals,
    ResidentThreeQuestionPriorityState* priority,
    plan::ResidentDiscoursePlanState* transient_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || checkpoint == nullptr)
    return;
  restore_three_question_priority_checkpoint(
      *checkpoint, cell_capacity, goals, priority, transient_plan);
}

// Compose exactly two independently originated resident goals. Evidence, not
// caller order, selects which is asked first; an exact priority tie abstains.
// Supporting/answer revisions are not ranking inputs, so a future answer
// cannot leak into composition.
__device__ inline bool compose_question_goals(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentQuestionCompositionState* composition) {
  if (goals == nullptr || composition == nullptr || cell_capacity == 0u ||
      !valid_composition(*composition) ||
      composition->status == CompositionStatus::active ||
      composition->status == CompositionStatus::invalidated ||
      !valid(goals[0], cell_capacity) || !valid(goals[1], cell_capacity) ||
      goals[0].status != GoalStatus::open ||
      goals[1].status != GoalStatus::open ||
      tissue::exact_population_equals(
          goals[0].target_cells, goals[0].target_count,
          {goals[1].target_cells, goals[1].target_count}) ||
      equal_question_goal_priority(goals[0], goals[1]) ||
      composition->revision == 0xffffffffu)
    return false;
  ResidentQuestionCompositionState shadow{};
  const std::uint32_t first = stronger_question_goal(goals[1], goals[0])
                                  ? 1u : 0u;
  shadow.status = CompositionStatus::active;
  shadow.revision = composition->revision + 1u;
  shadow.goal_slots[0] = first;
  shadow.goal_slots[1] = 1u - first;
  shadow.goal_revisions[0] = goals[first].revision;
  shadow.goal_revisions[1] = goals[1u - first].revision;
  if (!valid_composition(shadow))
    return false;
  *composition = shadow;
  return true;
}

__global__ void compose_question_goals_kernel(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentQuestionCompositionState* composition) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  compose_question_goals(goals, cell_capacity, composition);
}

__host__ __device__ inline bool question_composition_checkpoint_consistent(
    const ResidentQuestionCompositionCheckpoint& checkpoint,
    std::uint32_t cell_capacity) {
  if (checkpoint.magic != kQuestionCompositionCheckpointMagic ||
      checkpoint.version != kQuestionCompositionCheckpointVersion ||
      checkpoint.reserved != 0u || cell_capacity == 0u ||
      !valid_composition(checkpoint.composition) ||
      checkpoint.composition.status != CompositionStatus::active)
    return false;
  for (std::uint32_t position = 0u;
       position < kComposedQuestionGoalCount; ++position) {
    const std::uint32_t slot = checkpoint.composition.goal_slots[position];
    const ResidentQuestionGoalState& goal = checkpoint.goals[slot];
    if (!valid(goal, cell_capacity))
      return false;
    if (position < checkpoint.composition.active_position) {
      if (goal.status != GoalStatus::discharged ||
          goal.revision != checkpoint.composition.goal_revisions[position] ||
          goal.answered_evidence_revision !=
              checkpoint.composition.public_answer_revisions[position])
        return false;
    } else if (position == checkpoint.composition.active_position) {
      if ((goal.status != GoalStatus::open &&
           goal.status != GoalStatus::planned &&
           goal.status != GoalStatus::asked) ||
          goal.revision < checkpoint.composition.goal_revisions[position] ||
          checkpoint.composition.public_answer_revisions[position] != 0u)
        return false;
    } else if (goal.status != GoalStatus::open ||
               goal.revision !=
                   checkpoint.composition.goal_revisions[position] ||
               checkpoint.composition.public_answer_revisions[position] != 0u) {
      return false;
    }
  }
  return true;
}

__device__ inline bool capture_question_composition_checkpoint(
    const ResidentQuestionGoalState* goals,
    const ResidentQuestionCompositionState* composition,
    std::uint32_t cell_capacity,
    ResidentQuestionCompositionCheckpoint* checkpoint) {
  if (goals == nullptr || composition == nullptr || checkpoint == nullptr)
    return false;
  ResidentQuestionCompositionCheckpoint shadow{};
  shadow.composition = *composition;
  for (std::uint32_t slot = 0u; slot < kComposedQuestionGoalCount; ++slot)
    shadow.goals[slot] = goals[slot];
  if (shadow.composition.status == CompositionStatus::active) {
    const std::uint32_t position = shadow.composition.active_position;
    const std::uint32_t slot = shadow.composition.goal_slots[position];
    shadow.composition.goal_revisions[position] = shadow.goals[slot].revision;
  }
  if (!question_composition_checkpoint_consistent(shadow, cell_capacity))
    return false;
  *checkpoint = shadow;
  return true;
}

__global__ void capture_question_composition_checkpoint_kernel(
    const ResidentQuestionGoalState* goals,
    const ResidentQuestionCompositionState* composition,
    std::uint32_t cell_capacity,
    ResidentQuestionCompositionCheckpoint* checkpoint) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || checkpoint == nullptr)
    return;
  *checkpoint = ResidentQuestionCompositionCheckpoint{};
  if (!capture_question_composition_checkpoint(
          goals, composition, cell_capacity, checkpoint))
    checkpoint->magic = 0u;
}

// Restore is atomic and fences the active Goal revision. A planned but
// un-emitted question is reopened, while an already asked question remains
// asked and waits for its answer. In both cases the fence makes every stale
// pre-checkpoint Plan/answer branch older than the restored composition.
__device__ inline bool restore_question_composition_checkpoint(
    const ResidentQuestionCompositionCheckpoint& checkpoint,
    std::uint32_t cell_capacity, ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition,
    plan::ResidentDiscoursePlanState* transient_plan) {
  if (goals == nullptr || composition == nullptr || transient_plan == nullptr ||
      !question_composition_checkpoint_consistent(checkpoint, cell_capacity))
    return false;
  ResidentQuestionGoalState goal_shadows[kComposedQuestionGoalCount]{};
  for (std::uint32_t slot = 0u; slot < kComposedQuestionGoalCount; ++slot)
    goal_shadows[slot] = checkpoint.goals[slot];
  ResidentQuestionCompositionState composition_shadow =
      checkpoint.composition;
  plan::ResidentDiscoursePlanState plan_shadow = *transient_plan;
  plan::clear(&plan_shadow);
  const std::uint32_t position = composition_shadow.active_position;
  const std::uint32_t slot = composition_shadow.goal_slots[position];
  ResidentQuestionGoalState& active = goal_shadows[slot];
  if (active.revision == 0xffffffffu ||
      (active.status == GoalStatus::planned &&
       active.revision == 0xfffffffeu))
    return false;
  if (active.status == GoalStatus::planned) {
    active.status = GoalStatus::open;
    ++active.revision;
    active.plan_revision = 0u;
    active.surface_revision = 0u;
  }
  ++active.revision;
  composition_shadow.goal_revisions[position] = active.revision;
  if (composition_shadow.revision == 0xffffffffu)
    return false;
  ++composition_shadow.revision;
  ResidentQuestionCompositionCheckpoint normalized = checkpoint;
  for (std::uint32_t index = 0u; index < kComposedQuestionGoalCount; ++index)
    normalized.goals[index] = goal_shadows[index];
  normalized.composition = composition_shadow;
  if (!question_composition_checkpoint_consistent(normalized, cell_capacity) ||
      !plan::valid(plan_shadow) ||
      plan_shadow.status != plan::PlanStatus::empty)
    return false;
  for (std::uint32_t index = 0u; index < kComposedQuestionGoalCount; ++index)
    goals[index] = goal_shadows[index];
  *composition = composition_shadow;
  *transient_plan = plan_shadow;
  return true;
}

__global__ void restore_question_composition_checkpoint_kernel(
    const ResidentQuestionCompositionCheckpoint* checkpoint,
    std::uint32_t cell_capacity, ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition,
    plan::ResidentDiscoursePlanState* transient_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || checkpoint == nullptr)
    return;
  restore_question_composition_checkpoint(*checkpoint, cell_capacity, goals,
                                          composition, transient_plan);
}

__global__ void initialize_question_goal_kernel(
    ResidentQuestionGoalState* goal) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr)
    return;
  *goal = ResidentQuestionGoalState{};
}

// Originate only when a known opaque topic has retained proposition matter but
// no causally qualified outgoing bridge. The target is the known topic, not
// a guessed relation or answer. Multiple weak records for the same topic add
// evidence to the same gap and do not authorize a host-like winner.
__global__ void originate_information_gap_kernel(
    tissue::TissueView resident_tissue, tissue::SparsePopulationView topic,
    ResidentQuestionGoalState* goal, OriginationResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr ||
      resident_tissue.scalars == nullptr)
    return;
  OriginationResult local{};
  local.attempted = 1u;
  local.tissue_revision = resident_tissue.scalars->revision;
  if (result != nullptr)
    *result = local;
  if (!valid(*goal, resident_tissue.cell_capacity) ||
      goal->lesion_revision != 0u ||
      !tissue::valid_population(topic, resident_tissue.cell_capacity))
    return;
  // A question already selected or handed to planning is a persistent
  // commitment. New scans cannot silently replace its target; normal answer
  // assimilation must discharge it, or explicit lesion/revision must
  // invalidate it first.
  if (goal->status == GoalStatus::open ||
      goal->status == GoalStatus::planned ||
      goal->status == GoalStatus::asked)
    return;

  bool retained_gap_matter = false;
  std::uint64_t latest_evidence = 0u;
  std::uint64_t positive_mass = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint32_t widest_context = 0u;
  bool qualified_bridge = false;
  for (std::uint32_t index = 0u;
       index < resident_tissue.ordered_binding_capacity; ++index) {
    const auto& binding = resident_tissue.ordered_bindings[index];
    if (binding.claimed == 0u ||
        !tissue::ordered_binding_structurally_intact(
            binding, resident_tissue.cell_capacity) ||
        !topic_matches(topic, binding))
      continue;
    ++local.topic_matches;
    if (tissue::ordered_binding_qualified(
            binding, tissue::OrderedBindingQualification::causal,
            resident_tissue.cell_capacity)) {
      qualified_bridge = true;
      continue;
    }
    if (tissue::ordered_binding_positive_mass(binding) == 0u &&
        binding.counterevidence == 0u)
      continue;
    ++local.unsupported_matches;
    retained_gap_matter = true;
    latest_evidence = binding.last_evidence_revision > latest_evidence
                          ? binding.last_evidence_revision
                          : latest_evidence;
    positive_mass = saturating_add(
        positive_mass, tissue::ordered_binding_positive_mass(binding));
    counterevidence = saturating_add(counterevidence, binding.counterevidence);
    widest_context = binding.qualifying_context_count > widest_context
                         ? binding.qualifying_context_count
                         : widest_context;
  }
  if (qualified_bridge || !retained_gap_matter) {
    if (goal->status == GoalStatus::discharged && qualified_bridge) {
      ++goal->repeated_suppressed;
      ++goal->revision;
      local.repeated_suppressed = 1u;
    }
    if (result != nullptr)
      *result = local;
    return;
  }

  if (goal->status == GoalStatus::discharged &&
      tissue::exact_population_equals(goal->target_cells, goal->target_count,
                                      topic) &&
      latest_evidence <= goal->answered_evidence_revision) {
    ++goal->repeated_suppressed;
    ++goal->revision;
    local.repeated_suppressed = 1u;
    if (result != nullptr)
      *result = local;
    return;
  }

  goal->status = GoalStatus::open;
  ++goal->revision;
  ++goal->opened_count;
  goal->kind = GapKind::unsupported_topic_bridge;
  goal->target_count = topic.count;
  for (std::uint32_t cell = 0u; cell < tissue::kMaximumPopulationCells; ++cell)
    goal->target_cells[cell] = cell < topic.count ? topic.cells[cell] : 0u;
  goal->supporting_evidence_revision = latest_evidence;
  goal->tissue_revision = resident_tissue.scalars->revision;
  goal->positive_mass = positive_mass;
  goal->counterevidence = counterevidence;
  goal->qualifying_context_count = widest_context;
  goal->plan_revision = 0u;
  goal->surface_revision = 0u;
  local.opened = 1u;
  if (result != nullptr)
    *result = local;
}

// Production-facing bounded candidate settlement. Distinct opaque topics are
// never merged into one synthetic goal. Repeated views of exactly the same
// population count once. Multiple gaps compete by retained evidence; an exact
// evidence tie causes atomic abstention rather than a host-like winner.
__device__ inline void originate_information_gap_candidates(
    tissue::TissueView resident_tissue,
    const tissue::SparsePopulationView* candidate_topics,
    std::uint32_t candidate_count, ResidentQuestionGoalState* goal,
    OriginationResult* result) {
  if (goal == nullptr || resident_tissue.scalars == nullptr ||
      candidate_topics == nullptr ||
      candidate_count == 0u || candidate_count > kMaximumQuestionCandidates)
    return;
  OriginationResult local{};
  local.attempted = 1u;
  local.tissue_revision = resident_tissue.scalars->revision;
  if (result != nullptr)
    *result = local;
  if (!valid(*goal, resident_tissue.cell_capacity) ||
      goal->lesion_revision != 0u || goal->status == GoalStatus::open ||
      goal->status == GoalStatus::planned ||
      goal->status == GoalStatus::asked)
    return;

  tissue::SparsePopulationView selected_topic{};
  GapCandidateEvidence selected_evidence{};
  std::uint32_t retained_gap_count = 0u;
  bool selection_tied = false;
  bool qualified_answered_topic = false;
  for (std::uint32_t candidate = 0u; candidate < candidate_count;
       ++candidate) {
    const auto topic = candidate_topics[candidate];
    if (!tissue::valid_population(topic, resident_tissue.cell_capacity))
      continue;
    bool duplicate = false;
    for (std::uint32_t prior = 0u; prior < candidate; ++prior) {
      const auto earlier = candidate_topics[prior];
      duplicate = duplicate ||
                  (tissue::valid_population(
                       earlier, resident_tissue.cell_capacity) &&
                   tissue::exact_population_equals(
                       topic.cells, topic.count, earlier));
    }
    if (duplicate)
      continue;
    const GapCandidateEvidence evidence =
        evaluate_gap_candidate(resident_tissue, topic);
    local.topic_matches += evidence.matches;
    local.unsupported_matches += evidence.unsupported_matches;
    qualified_answered_topic =
        qualified_answered_topic ||
        (evidence.qualified != 0u &&
         goal->status == GoalStatus::discharged &&
         tissue::exact_population_equals(goal->target_cells,
                                         goal->target_count, topic));
    if (evidence.qualified != 0u || evidence.retained == 0u)
      continue;
    ++retained_gap_count;
    if (retained_gap_count == 1u ||
        stronger_gap_evidence(evidence, selected_evidence)) {
      selected_topic = topic;
      selected_evidence = evidence;
      selection_tied = false;
    } else if (equal_gap_evidence(evidence, selected_evidence)) {
      selection_tied = true;
    }
  }
  if (retained_gap_count == 0u || selection_tied) {
    if (selection_tied) {
      ++goal->ambiguous_count;
      ++goal->revision;
      local.ambiguous = 1u;
    } else if (qualified_answered_topic) {
      ++goal->repeated_suppressed;
      ++goal->revision;
      local.repeated_suppressed = 1u;
    }
    if (result != nullptr)
      *result = local;
    return;
  }
  if (goal->status == GoalStatus::discharged &&
      tissue::exact_population_equals(goal->target_cells, goal->target_count,
                                      selected_topic) &&
      selected_evidence.latest_evidence <=
          goal->answered_evidence_revision) {
    ++goal->repeated_suppressed;
    ++goal->revision;
    local.repeated_suppressed = 1u;
    if (result != nullptr)
      *result = local;
    return;
  }
  goal->status = GoalStatus::open;
  ++goal->revision;
  ++goal->opened_count;
  goal->kind = GapKind::unsupported_topic_bridge;
  goal->target_count = selected_topic.count;
  for (std::uint32_t cell = 0u; cell < tissue::kMaximumPopulationCells;
       ++cell)
    goal->target_cells[cell] =
        cell < selected_topic.count ? selected_topic.cells[cell] : 0u;
  goal->supporting_evidence_revision = selected_evidence.latest_evidence;
  goal->tissue_revision = resident_tissue.scalars->revision;
  goal->positive_mass = selected_evidence.positive_mass;
  goal->counterevidence = selected_evidence.counterevidence;
  goal->qualifying_context_count = selected_evidence.widest_context;
  goal->plan_revision = 0u;
  goal->surface_revision = 0u;
  local.opened = 1u;
  if (result != nullptr)
    *result = local;
}

__global__ void originate_information_gap_candidates_kernel(
    tissue::TissueView resident_tissue,
    const tissue::SparsePopulationView* candidate_topics,
    std::uint32_t candidate_count, ResidentQuestionGoalState* goal,
    OriginationResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  originate_information_gap_candidates(resident_tissue, candidate_topics,
                                        candidate_count, goal, result);
}

// This is the causal authorization read by the production population-surface
// path.  Observer receipts are deliberately absent: only the persistent goal
// and its committed resident Plan can authorize an interrogative motor frame.
// In particular, matching a slot and revision is insufficient when the Plan
// was selected from different resident evidence.
__host__ __device__ inline bool question_plan_matches_goal(
    const ResidentQuestionGoalState& goal,
    const plan::ResidentDiscoursePlanState& resident_plan) {
  const auto advances_by = [](std::uint32_t newer, std::uint32_t older,
                              std::uint32_t delta) {
    return older <= 0xffffffffu - delta && newer == older + delta;
  };
  const bool revision_matches =
      resident_plan.question_goal_revision == goal.revision ||
      (goal.status == GoalStatus::asked &&
       resident_plan.status == plan::PlanStatus::completed &&
       advances_by(goal.revision, resident_plan.question_goal_revision, 1u)) ||
      (goal.status == GoalStatus::discharged &&
       resident_plan.status == plan::PlanStatus::completed &&
       (advances_by(goal.revision, resident_plan.question_goal_revision, 1u) ||
        advances_by(goal.revision, resident_plan.question_goal_revision, 2u)));
  const bool plan_revision_matches =
      resident_plan.revision == goal.plan_revision ||
      (resident_plan.status == plan::PlanStatus::invalidated &&
       advances_by(resident_plan.revision, goal.plan_revision, 1u));
  return (goal.status == GoalStatus::planned ||
          goal.status == GoalStatus::asked ||
          goal.status == GoalStatus::discharged) &&
         goal.kind != GapKind::none && goal.target_count != 0u &&
         goal.plan_revision != 0u &&
         resident_plan.question_goal_dependency == 1u &&
         plan_revision_matches &&
         resident_plan.supporting_evidence_revision ==
             goal.supporting_evidence_revision &&
         resident_plan.modality == plan::PlanModality::interrogative &&
         resident_plan.step_count == 1u &&
         resident_plan.steps[0].reference_kind ==
             plan::PlanReferenceKind::question_goal &&
         resident_plan.population_reference_count == 1u &&
         resident_plan.population_references[0] == 0u &&
         revision_matches &&
         resident_plan.question_goal_lesion_revision == goal.lesion_revision;
}

// A GroundingResult is an observer projection, not authority.  Adult stream
// production admits an interrogative action only when the live Goal/Plan pair
// still matches and the canonical materializer committed its one exact anchor.
__host__ __device__ inline bool materialized_question_plan_is_authoritative(
    const ResidentQuestionGoalState& goal,
    const plan::ResidentDiscoursePlanState& resident_plan,
    std::uint32_t cell_capacity) {
  return valid(goal, cell_capacity) && plan::valid(resident_plan) &&
         question_plan_matches_goal(goal, resident_plan) &&
         resident_plan.status == plan::PlanStatus::committed &&
         resident_plan.active_step == 0u &&
         resident_plan.anchor_reference_count == 1u &&
         resident_plan.steps[0].anchor_begin == 0u &&
         resident_plan.steps[0].anchor_count == 1u;
}

// Repair the two-object lifecycle as one transaction boundary.  Production
// kernels call this in the same thread that completes or invalidates a Plan;
// checkpoint/adoption code calls the identical rule before exposing restored
// matter.  Thus no completed question remains merely `planned`, no interrupted
// question remains stranded behind an invalidated Plan, and no unmatched Plan
// is granted goal authority.
__host__ __device__ inline void normalize_plan_goal_pair(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan) {
  if (goal == nullptr || resident_plan == nullptr)
    return;
  const bool matched = question_plan_matches_goal(*goal, *resident_plan);
  if (goal->status == GoalStatus::planned && matched &&
      resident_plan->status == plan::PlanStatus::completed) {
    goal->status = GoalStatus::asked;
    if (goal->revision != 0xffffffffu)
      ++goal->revision;
    if (goal->asked_count != 0xffffffffu)
      ++goal->asked_count;
    return;
  }
  if (goal->status == GoalStatus::planned &&
      ((!matched && resident_plan->question_goal_dependency == 0u) ||
       (matched && resident_plan->status == plan::PlanStatus::invalidated))) {
    goal->status = GoalStatus::open;
    if (goal->revision != 0xffffffffu)
      ++goal->revision;
    goal->plan_revision = 0u;
    goal->surface_revision = 0u;
    return;
  }
  if (resident_plan->question_goal_dependency != 0u && !matched) {
    if (resident_plan->status != plan::PlanStatus::invalidated)
      plan::invalidate(resident_plan, goal->revision);
    if (goal->status == GoalStatus::planned) {
      goal->status = GoalStatus::open;
      if (goal->revision != 0xffffffffu)
        ++goal->revision;
      goal->plan_revision = 0u;
      goal->surface_revision = 0u;
    }
  }
}

// Stream checkpoints persist the Goal as learned adult matter but deliberately
// discard the contact-local Plan. Restore/adoption must therefore reopen a
// planned Goal against the freshly empty transport Plan instead of stranding
// an unprovable historical emission tail.
__global__ void normalize_checkpoint_question_goal_kernel(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  normalize_plan_goal_pair(goal, resident_plan);
}

// Stage the normal Plan ABI from one persistent goal slot.  The Plan stores
// only the slot reference and the goal revision; it never copies target cells
// as if they were a guessed predicate or answer.  Grounding dereferences the
// slot later, after content commitment.
__device__ inline bool stage_question_goal_plan(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (goal == nullptr || resident_plan == nullptr ||
      goal->status != GoalStatus::open ||
      !valid(*goal, cell_capacity) ||
      !plan::valid(*resident_plan) || goal->revision == 0xffffffffu)
    return false;
  plan::ResidentDiscoursePlanState shadow = *resident_plan;
  const std::uint32_t planned_goal_revision = goal->revision + 1u;
  const std::uint32_t goal_slot = 0u;
  if (!plan::begin_population_plan(
          &shadow, goal->supporting_evidence_revision, surface_revision,
          plan::PlanModality::interrogative, 0u, 0u, 0u, 1u,
          planned_goal_revision, goal->lesion_revision) ||
      !plan::append_population_step(
          &shadow, &goal_slot, 1u, 0u,
          goal->supporting_evidence_revision,
          plan::PlanReferenceKind::question_goal) ||
      !plan::commit_population_plan(&shadow))
    return false;
  *resident_plan = shadow;
  goal->status = GoalStatus::planned;
  goal->revision = planned_goal_revision;
  goal->plan_revision = shadow.revision;
  goal->surface_revision = surface_revision;
  return true;
}

__global__ void stage_question_goal_plan_kernel(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_question_goal_plan(goal, resident_plan, surface_revision,
                           cell_capacity);
}

__device__ inline bool stage_prioritized_question_plan(
    ResidentQuestionGoalState* goals,
    ResidentThreeQuestionPriorityState* priority,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (goals == nullptr || priority == nullptr || resident_plan == nullptr ||
      !valid_three_question_priority(*priority) ||
      priority->status != PriorityStatus::active ||
      priority->active_slot >= kPrioritizedQuestionGoalCount ||
      priority->revision == 0xffffffffu)
    return false;
  const std::uint32_t slot = priority->active_slot;
  if (goals[slot].revision != priority->goal_revisions[slot] ||
      goals[slot].status != GoalStatus::open ||
      !stage_question_goal_plan(&goals[slot], resident_plan,
                                surface_revision, cell_capacity)) {
    priority->status = PriorityStatus::invalidated;
    ++priority->revision;
    return false;
  }
  priority->goal_revisions[slot] = goals[slot].revision;
  ++priority->revision;
  return true;
}

__global__ void stage_prioritized_question_plan_kernel(
    ResidentQuestionGoalState* goals,
    ResidentThreeQuestionPriorityState* priority,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_prioritized_question_plan(goals, priority, resident_plan,
                                  surface_revision, cell_capacity);
}

__device__ inline bool advance_prioritized_question_after_answer(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (goals == nullptr || priority == nullptr ||
      !valid_three_question_priority(*priority) ||
      priority->status != PriorityStatus::active ||
      priority->active_slot >= kPrioritizedQuestionGoalCount ||
      priority->revision == 0xffffffffu)
    return false;
  const std::uint32_t slot = priority->active_slot;
  const ResidentQuestionGoalState& answered = goals[slot];
  const std::uint64_t prior_answer = priority->answer_count == 0u
      ? 0u : priority->public_answer_revisions[priority->answer_count - 1u];
  if (answered.status != GoalStatus::discharged ||
      answered.revision < priority->goal_revisions[slot] ||
      answered.answered_evidence_revision <=
          answered.supporting_evidence_revision ||
      answered.answered_evidence_revision <= prior_answer)
    return false;
  ResidentThreeQuestionPriorityState shadow = *priority;
  shadow.goal_revisions[slot] = answered.revision;
  shadow.public_answer_revisions[shadow.answer_count++] =
      answered.answered_evidence_revision;
  shadow.remaining_mask &= ~(1u << slot);
  shadow.active_slot = 0xffffffffu;
  ++shadow.revision;
  if (shadow.remaining_mask == 0u) {
    shadow.status = PriorityStatus::completed;
    if (!valid_three_question_priority(shadow))
      return false;
    *priority = shadow;
    return true;
  }
  // `tied` is a truthful state only while at least two Goals remain.  When
  // the accepted answer leaves one Goal, select that sole resident matter
  // directly instead of constructing an invalid transient tied state that
  // the common reprioritizer must reject.
  if ((shadow.remaining_mask & (shadow.remaining_mask - 1u)) == 0u) {
    std::uint32_t selected = 0u;
    while ((shadow.remaining_mask & (1u << selected)) == 0u)
      ++selected;
    if (!valid(goals[selected], cell_capacity) ||
        goals[selected].status != GoalStatus::open ||
        shadow.revision == 0xffffffffu)
      return false;
    shadow.goal_revisions[selected] = goals[selected].revision;
    shadow.active_slot = selected;
    shadow.status = PriorityStatus::active;
    ++shadow.revision;
    if (!valid_three_question_priority(shadow))
      return false;
    *priority = shadow;
    return true;
  }
  shadow.status = PriorityStatus::tied;
  *priority = shadow;
  // Recording the answer is successful even when the next strongest Goal is
  // tied and public questioning must abstain until later evidence separates it.
  reprioritize_three_question_goals(goals, cell_capacity, priority);
  return true;
}

__global__ void advance_prioritized_question_after_answer_kernel(
    const ResidentQuestionGoalState* goals, std::uint32_t cell_capacity,
    ResidentThreeQuestionPriorityState* priority) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  advance_prioritized_question_after_answer(goals, cell_capacity, priority);
}

// Expose only the active composed goal through the existing one-goal Plan
// ABI. The composition owns ordering; each public question is still a normal
// canonical Goal/Plan transaction with no copied target cells.
__device__ inline bool stage_composed_question_plan(
    ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (goals == nullptr || composition == nullptr || resident_plan == nullptr ||
      !valid_composition(*composition) ||
      composition->status != CompositionStatus::active ||
      composition->active_position >= kComposedQuestionGoalCount ||
      composition->revision == 0xffffffffu)
    return false;
  const std::uint32_t position = composition->active_position;
  const std::uint32_t slot = composition->goal_slots[position];
  ResidentQuestionGoalState& selected = goals[slot];
  if (selected.revision != composition->goal_revisions[position] ||
      selected.status != GoalStatus::open ||
      !stage_question_goal_plan(&selected, resident_plan, surface_revision,
                                cell_capacity)) {
    composition->status = CompositionStatus::invalidated;
    ++composition->revision;
    return false;
  }
  composition->goal_revisions[position] = selected.revision;
  ++composition->revision;
  return true;
}

__global__ void stage_composed_question_plan_kernel(
    ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition,
    plan::ResidentDiscoursePlanState* resident_plan,
    std::uint32_t surface_revision, std::uint32_t cell_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_composed_question_plan(goals, composition, resident_plan,
                               surface_revision, cell_capacity);
}

// Record only an actually discharged active Goal as a public answer. Answers
// must advance in strictly increasing evidence-revision order. The next Goal
// remains independently open and is staged through a fresh Plan transaction.
__device__ inline bool advance_composed_question_after_answer(
    const ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition) {
  if (goals == nullptr || composition == nullptr ||
      !valid_composition(*composition) ||
      composition->status != CompositionStatus::active ||
      composition->active_position >= kComposedQuestionGoalCount ||
      composition->revision == 0xffffffffu)
    return false;
  const std::uint32_t position = composition->active_position;
  const ResidentQuestionGoalState& answered =
      goals[composition->goal_slots[position]];
  const std::uint64_t prior_answer =
      position == 0u ? 0u : composition->public_answer_revisions[position - 1u];
  if (answered.status != GoalStatus::discharged ||
      answered.revision < composition->goal_revisions[position] ||
      answered.answered_evidence_revision <=
          answered.supporting_evidence_revision ||
      answered.answered_evidence_revision <= prior_answer) {
    return false;
  }
  ResidentQuestionCompositionState shadow = *composition;
  shadow.goal_revisions[position] = answered.revision;
  shadow.public_answer_revisions[position] =
      answered.answered_evidence_revision;
  ++shadow.active_position;
  shadow.status =
      shadow.active_position == kComposedQuestionGoalCount
          ? CompositionStatus::completed
          : CompositionStatus::active;
  ++shadow.revision;
  if (!valid_composition(shadow))
    return false;
  *composition = shadow;
  return true;
}

__global__ void advance_composed_question_after_answer_kernel(
    const ResidentQuestionGoalState* goals,
    ResidentQuestionCompositionState* composition) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  advance_composed_question_after_answer(goals, composition);
}

// Asking is a reafferent lifecycle event: selection and partial emission are
// not enough.  Only completion of the retained Plan frame latches the goal as
// asked and suppresses immediate quiet-tick repetition.
__global__ void mark_question_goal_asked_kernel(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr ||
      resident_plan == nullptr || goal->status != GoalStatus::planned ||
      resident_plan->status != plan::PlanStatus::completed ||
      !question_plan_matches_goal(*goal, *resident_plan))
    return;
  normalize_plan_goal_pair(goal, resident_plan);
}

__global__ void reopen_question_goal_after_interruption_kernel(
    ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr ||
      resident_plan == nullptr || goal->status != GoalStatus::planned ||
      resident_plan->status != plan::PlanStatus::invalidated ||
      !question_plan_matches_goal(*goal, *resident_plan))
    return;
  normalize_plan_goal_pair(goal, resident_plan);
}

__global__ void invalidate_question_plan_on_goal_change_kernel(
    const ResidentQuestionGoalState* goal,
    plan::ResidentDiscoursePlanState* resident_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr ||
      resident_plan == nullptr || !plan::valid(*resident_plan) ||
      resident_plan->question_goal_dependency == 0u ||
      question_plan_matches_goal(*goal, *resident_plan))
    return;
  plan::invalidate(resident_plan, goal->revision);
}

__device__ inline bool discharge_question_goal_from_tissue(
    tissue::TissueView resident_tissue, ResidentQuestionGoalState* goal) {
  if (goal == nullptr ||
      (goal->status != GoalStatus::open &&
       goal->status != GoalStatus::planned &&
       goal->status != GoalStatus::asked) ||
      resident_tissue.scalars == nullptr ||
      !valid(*goal, resident_tissue.cell_capacity))
    return false;
  tissue::SparsePopulationView target{goal->target_cells, goal->target_count};
  std::uint64_t answered_revision = 0u;
  std::uint64_t returned_counterevidence = 0u;
  std::uint64_t counterevidence_answer_revision = 0u;
  for (std::uint32_t index = 0u;
       index < resident_tissue.ordered_binding_capacity; ++index) {
    const auto& binding = resident_tissue.ordered_bindings[index];
    if (binding.claimed == 0u || !topic_matches(target, binding) ||
        !tissue::ordered_binding_structurally_intact(
            binding, resident_tissue.cell_capacity))
      continue;
    returned_counterevidence =
        saturating_add(returned_counterevidence, binding.counterevidence);
    if (tissue::ordered_binding_qualified(
            binding, tissue::OrderedBindingQualification::causal,
            resident_tissue.cell_capacity) &&
        binding.last_evidence_revision > answered_revision) {
      answered_revision = binding.last_evidence_revision;
    }
    // A discordant returned consequence is also an answer, but only after
    // intervention-strength counterevidence dominates positive intervention
    // support across independent contexts. Observational mass cannot promote
    // this branch, and the goal's snapshot proves the counterevidence arrived
    // strictly after origination rather than being relabelled from its input.
    if (binding.qualifying_context_count >=
            tissue::kMinimumIndependentContexts &&
        binding.counterevidence >= tissue::kMinimumInterventionMass &&
        binding.counterevidence > binding.intervention_support &&
        binding.last_evidence_revision > counterevidence_answer_revision) {
      counterevidence_answer_revision = binding.last_evidence_revision;
    }
  }
  if (returned_counterevidence > goal->counterevidence &&
      counterevidence_answer_revision > answered_revision)
    answered_revision = counterevidence_answer_revision;
  if (answered_revision <= goal->supporting_evidence_revision)
    return false;
  goal->status = GoalStatus::discharged;
  ++goal->revision;
  ++goal->discharged_count;
  goal->answered_evidence_revision = answered_revision;
  goal->counterevidence = returned_counterevidence;
  goal->tissue_revision = resident_tissue.scalars->revision;
  return true;
}

__global__ void discharge_question_goal_from_tissue_kernel(
    tissue::TissueView resident_tissue, ResidentQuestionGoalState* goal) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  (void)discharge_question_goal_from_tissue(resident_tissue, goal);
}

__global__ void lesion_question_goal_kernel(ResidentQuestionGoalState* goal,
                                            std::uint32_t lesion_revision) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || goal == nullptr ||
      lesion_revision == 0u)
    return;
  goal->status = GoalStatus::invalidated;
  ++goal->revision;
  goal->lesion_revision = lesion_revision;
  goal->target_count = 0u;
  goal->kind = GapKind::none;
  goal->plan_revision = 0u;
  goal->surface_revision = 0u;
}

}  // namespace bcc32_cuda_resident_question_goal
