#pragma once

#include <cstdint>
#include <type_traits>

#include "bcc32_cuda_resident_discourse_plan.cuh"
#include "bcc32_cuda_resident_question_goal.cuh"

// A question ticket is a resident causal return fence.  It does not contain a
// word, expected answer, or host route.  It records the exact Goal/Plan
// revisions that authorized one emitted action and consumes at most one later
// external contact while those revisions remain live.  The action hash is an
// observer receipt; the revisions and live Plan/Goal checks are the authority.
namespace bcc32_cuda_resident_question_ticket {

namespace goal = bcc32_cuda_resident_question_goal;
namespace plan = bcc32_cuda_resident_discourse_plan;

struct ResidentQuestionActionTicket {
  std::uint32_t issued = 0u;
  std::uint32_t active = 0u;
  std::uint32_t goal_revision = 0u;
  std::uint32_t plan_revision = 0u;
  std::uint32_t lesion_revision = 0u;
  std::uint32_t target_count = 0u;
  std::uint32_t action_hash = 0u;
  std::uint32_t accepted_returns = 0u;
  std::uint32_t rejected_returns = 0u;
  std::uint32_t last_return_hash = 0u;
};

static_assert(std::is_trivially_copyable_v<ResidentQuestionActionTicket>);

__host__ __device__ inline std::uint32_t hash_bytes(
    const std::uint8_t* bytes, std::uint32_t count) {
  if (bytes == nullptr || count == 0u)
    return 0u;
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t index = 0u; index < count; ++index) {
    hash ^= bytes[index];
    hash *= 16777619u;
  }
  return hash == 0u ? 1u : hash;
}

__host__ __device__ inline bool live_emitted_question(
    const ResidentQuestionActionTicket& ticket,
    const goal::ResidentQuestionGoalState& resident_goal,
    const plan::ResidentDiscoursePlanState& resident_plan,
    std::uint32_t cell_capacity) {
  if (ticket.issued == 0u || ticket.goal_revision != resident_goal.revision ||
      ticket.plan_revision != resident_plan.revision ||
      ticket.lesion_revision != resident_goal.lesion_revision ||
      (resident_goal.status != goal::GoalStatus::planned &&
       resident_goal.status != goal::GoalStatus::asked) ||
      (resident_plan.status != plan::PlanStatus::committed &&
       resident_plan.status != plan::PlanStatus::completed) ||
      !goal::valid(resident_goal, cell_capacity) ||
      !plan::valid(resident_plan) ||
      !goal::question_plan_matches_goal(resident_goal, resident_plan))
    return false;
  return resident_plan.question_goal_dependency != 0u;
}

__host__ __device__ inline void issue(
    ResidentQuestionActionTicket* ticket,
    const goal::ResidentQuestionGoalState& resident_goal,
    const plan::ResidentDiscoursePlanState& resident_plan) {
  if (ticket == nullptr)
    return;
  *ticket = ResidentQuestionActionTicket{};
  ticket->issued = 1u;
  ticket->goal_revision = resident_goal.revision;
  ticket->plan_revision = resident_plan.revision;
  ticket->lesion_revision = resident_goal.lesion_revision;
  ticket->target_count = resident_goal.target_count;
}

__host__ __device__ inline bool arm(
    ResidentQuestionActionTicket* ticket,
    const goal::ResidentQuestionGoalState& resident_goal,
    const plan::ResidentDiscoursePlanState& resident_plan,
    std::uint32_t cell_capacity, const std::uint8_t* action,
    std::uint32_t action_count) {
  if (ticket == nullptr || !live_emitted_question(
                                *ticket, resident_goal, resident_plan,
                                cell_capacity) || action == nullptr ||
      action_count == 0u)
    return false;
  ticket->active = 1u;
  ticket->action_hash = hash_bytes(action, action_count);
  return true;
}

__device__ inline bool consume_return(
    ResidentQuestionActionTicket* ticket,
    const goal::ResidentQuestionGoalState& resident_goal,
    const plan::ResidentDiscoursePlanState& resident_plan,
    std::uint32_t cell_capacity, const std::uint8_t* source,
    std::uint32_t source_count, const std::uint8_t* returned,
    std::uint32_t returned_count) {
  if (ticket == nullptr || ticket->active == 0u)
    return false;
  const bool current = live_emitted_question(
      *ticket, resident_goal, resident_plan, cell_capacity);
  const bool fresh = returned != nullptr && returned_count != 0u &&
      returned_count <= 0xffffffffu &&
      (source == nullptr || source_count != returned_count ||
       hash_bytes(source, source_count) != hash_bytes(returned, returned_count));
  ticket->active = 0u;
  if (!current || !fresh) {
    if (ticket->rejected_returns != 0xffffffffu)
      ++ticket->rejected_returns;
    return false;
  }
  ticket->last_return_hash = hash_bytes(returned, returned_count);
  if (ticket->accepted_returns != 0xffffffffu)
    ++ticket->accepted_returns;
  return true;
}

}  // namespace bcc32_cuda_resident_question_ticket
