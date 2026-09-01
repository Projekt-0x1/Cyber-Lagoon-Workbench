#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// This is an integer-only resident seam.  The references are opaque handles
// owned by the discourse-plan tissue; this organ never decodes their content.
namespace substrate::bcc32::resident_plan_eligibility {

inline constexpr std::uint32_t kInvalidResidentPlanReference = 0xffffffffu;
inline constexpr std::uint32_t kResidentPlanEligibilityCapacity = 64u;

struct OpaquePlanStepReference {
  std::uint32_t value = kInvalidResidentPlanReference;
};

struct OpaqueEvidenceRevisionReference {
  std::uint32_t value = kInvalidResidentPlanReference;
};

struct OpaqueOrderedBindingReference {
  std::uint32_t value = kInvalidResidentPlanReference;
};

struct ResidentPlanEligibilityTrace {
  OpaquePlanStepReference plan_step{};
  OpaqueEvidenceRevisionReference evidence_revision{};
  OpaqueOrderedBindingReference ordered_binding{};
  std::uint32_t eligibility_quanta = 0u;
  std::uint32_t positive_credit_quanta = 0u;
  std::uint32_t counter_credit_quanta = 0u;
  std::uint32_t last_attachment_id = kInvalidResidentPlanReference;
  std::uint32_t last_consequence_id = kInvalidResidentPlanReference;
};

struct ResidentPlanEligibilityState {
  ResidentPlanEligibilityTrace traces[kResidentPlanEligibilityCapacity]{};
  std::uint32_t trace_count = 0u;
  std::uint32_t free_quanta = 0u;
  std::uint32_t last_attachment_id = kInvalidResidentPlanReference;
  std::uint32_t last_consequence_id = kInvalidResidentPlanReference;
};

struct ResidentPlanTraceAttachment {
  std::uint32_t attachment_id = kInvalidResidentPlanReference;
  OpaquePlanStepReference plan_step{};
  OpaqueEvidenceRevisionReference evidence_revision{};
  OpaqueOrderedBindingReference ordered_binding{};
  std::uint32_t seed_quanta = 0u;
};

struct ResidentPlanConsequenceTarget {
  std::uint32_t consequence_id = kInvalidResidentPlanReference;
  OpaquePlanStepReference plan_step{};
  OpaqueEvidenceRevisionReference evidence_revision{};
  OpaqueOrderedBindingReference ordered_binding{};
};

__device__ inline bool valid(OpaquePlanStepReference reference) {
  return reference.value != kInvalidResidentPlanReference;
}

__device__ inline bool valid(OpaqueEvidenceRevisionReference reference) {
  return reference.value != kInvalidResidentPlanReference;
}

__device__ inline bool valid(OpaqueOrderedBindingReference reference) {
  return reference.value != kInvalidResidentPlanReference;
}

__device__ inline bool same_trace(const ResidentPlanEligibilityTrace& trace,
                                  const ResidentPlanTraceAttachment& attachment) {
  return trace.plan_step.value == attachment.plan_step.value &&
         trace.evidence_revision.value == attachment.evidence_revision.value &&
         trace.ordered_binding.value == attachment.ordered_binding.value;
}

__device__ inline bool matches_target(const ResidentPlanEligibilityTrace& trace,
                                      const ResidentPlanConsequenceTarget& target) {
  return trace.plan_step.value == target.plan_step.value &&
         trace.evidence_revision.value == target.evidence_revision.value &&
         trace.ordered_binding.value == target.ordered_binding.value;
}

// The right shift is the local integer physics: decayed eligibility returns to
// free resident material.  No credit is created by decay.
__device__ inline void decay_eligibility(ResidentPlanEligibilityState* state,
                                         std::uint32_t ticks) {
  for (std::uint32_t tick = 0u; tick < ticks; ++tick) {
    for (std::uint32_t index = 0u; index < state->trace_count; ++index) {
      ResidentPlanEligibilityTrace& trace = state->traces[index];
      const std::uint32_t released =
          (trace.eligibility_quanta + 1u) >> 1u;
      trace.eligibility_quanta -= released;
      state->free_quanta += released;
    }
  }
}

__global__ inline void initialize_resident_plan_eligibility_kernel(
    ResidentPlanEligibilityState* state, std::uint32_t total_quanta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *state = ResidentPlanEligibilityState{};
  state->free_quanta = total_quanta;
}

// Attaching the same resident attachment id is idempotent. The call transfers
// existing free quanta into a trace and never interprets the opaque handles.
__device__ inline void attach_resident_plan_trace(
    ResidentPlanEligibilityState* state,
    const ResidentPlanTraceAttachment& input) {
  if (input.attachment_id == kInvalidResidentPlanReference ||
      input.attachment_id == state->last_attachment_id || !valid(input.plan_step) ||
      !valid(input.evidence_revision) || !valid(input.ordered_binding) || input.seed_quanta == 0u)
    return;

  std::uint32_t index = state->trace_count;
  std::uint32_t reusable = kInvalidResidentPlanReference;
  for (std::uint32_t candidate = 0u; candidate < state->trace_count; ++candidate) {
    if (same_trace(state->traces[candidate], input)) {
      index = candidate;
      break;
    }
    const ResidentPlanEligibilityTrace& trace = state->traces[candidate];
    if (reusable == kInvalidResidentPlanReference &&
        trace.eligibility_quanta == 0u &&
        trace.positive_credit_quanta == 0u &&
        trace.counter_credit_quanta == 0u)
      reusable = candidate;
  }
  if (index == state->trace_count) {
    if (state->trace_count == kResidentPlanEligibilityCapacity) {
      if (reusable == kInvalidResidentPlanReference) return;
      index = reusable;
      state->traces[index] = ResidentPlanEligibilityTrace{};
    } else {
      ++state->trace_count;
    }
    state->traces[index].plan_step = input.plan_step;
    state->traces[index].evidence_revision = input.evidence_revision;
    state->traces[index].ordered_binding = input.ordered_binding;
  }

  if (state->traces[index].last_attachment_id == input.attachment_id)
    return;

  const std::uint32_t transferred = input.seed_quanta < state->free_quanta
                                        ? input.seed_quanta
                                        : state->free_quanta;
  state->free_quanta -= transferred;
  state->traces[index].eligibility_quanta += transferred;
  state->traces[index].last_attachment_id = input.attachment_id;
  state->last_attachment_id = input.attachment_id;
}

__global__ inline void attach_resident_plan_trace_kernel(
    ResidentPlanEligibilityState* state,
    const ResidentPlanTraceAttachment* attachment) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  attach_resident_plan_trace(state, *attachment);
}

__global__ inline void decay_resident_plan_eligibility_kernel(
    ResidentPlanEligibilityState* state, std::uint32_t ticks) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  decay_eligibility(state, ticks);
}

// signed_consequence is resident input.  Its sign and magnitude are not host
// labels: positive and negative amplitudes route quanta to separate local banks.
// Repeating a consequence id is a no-op, making a replayed delivery idempotent.
__device__ inline void apply_resident_plan_consequence(
    ResidentPlanEligibilityState* state,
    const ResidentPlanConsequenceTarget& input,
    std::int32_t signed_amount) {
  if (input.consequence_id == kInvalidResidentPlanReference ||
      input.consequence_id == state->last_consequence_id || !valid(input.plan_step) ||
      !valid(input.evidence_revision) || !valid(input.ordered_binding))
    return;

  const std::uint32_t magnitude = signed_amount >= 0
                                      ? static_cast<std::uint32_t>(signed_amount)
                                      : static_cast<std::uint32_t>(-(static_cast<std::int64_t>(signed_amount)));
  for (std::uint32_t index = 0u; index < state->trace_count; ++index) {
    ResidentPlanEligibilityTrace& trace = state->traces[index];
    if (!matches_target(trace, input) || trace.eligibility_quanta == 0u) continue;
    if (trace.last_consequence_id == input.consequence_id) break;
    const std::uint32_t transferred = magnitude < trace.eligibility_quanta
                                          ? magnitude
                                          : trace.eligibility_quanta;
    trace.eligibility_quanta -= transferred;
    if (signed_amount > 0)
      trace.positive_credit_quanta += transferred;
    else if (signed_amount < 0)
      trace.counter_credit_quanta += transferred;
    trace.last_consequence_id = input.consequence_id;
    break;
  }
  state->last_consequence_id = input.consequence_id;
}

__global__ inline void apply_resident_plan_consequence_kernel(
    ResidentPlanEligibilityState* state,
    const ResidentPlanConsequenceTarget* target,
    const std::int32_t* signed_consequence) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  apply_resident_plan_consequence(state, *target, *signed_consequence);
}

__device__ inline std::int64_t ordered_binding_credit(
    const ResidentPlanEligibilityState* state, std::uint32_t ordered_binding) {
  if (state == nullptr ||
      ordered_binding == kInvalidResidentPlanReference)
    return 0;
  std::int64_t credit = 0;
  for (std::uint32_t index = 0u; index < state->trace_count; ++index) {
    const ResidentPlanEligibilityTrace& trace = state->traces[index];
    if (trace.ordered_binding.value != ordered_binding)
      continue;
    credit += static_cast<std::int64_t>(trace.positive_credit_quanta);
    credit -= static_cast<std::int64_t>(trace.counter_credit_quanta);
  }
  return credit;
}

__global__ inline void read_ordered_binding_credit_kernel(
    const ResidentPlanEligibilityState* state, std::uint32_t ordered_binding,
    std::int64_t* credit) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || credit == nullptr) return;
  *credit = ordered_binding_credit(state, ordered_binding);
}

__device__ inline std::uint64_t resident_plan_eligibility_mass(
    const ResidentPlanEligibilityState& state) {
  std::uint64_t total = state.free_quanta;
  for (std::uint32_t index = 0u; index < state.trace_count; ++index) {
    const ResidentPlanEligibilityTrace& trace = state.traces[index];
    total += trace.eligibility_quanta;
    total += trace.positive_credit_quanta;
    total += trace.counter_credit_quanta;
  }
  return total;
}

__global__ inline void account_resident_plan_eligibility_mass_kernel(
    const ResidentPlanEligibilityState* state, std::uint64_t* total) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *total = resident_plan_eligibility_mass(*state);
}

}  // namespace substrate::bcc32::resident_plan_eligibility
