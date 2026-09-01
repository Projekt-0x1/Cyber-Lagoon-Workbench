#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_EXECUTOR_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_EXECUTOR_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_causal_program.cuh"

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_CAUSAL_EXECUTOR_HD __host__ __device__
#else
#define DIRECT_CAUSAL_EXECUTOR_HD
#endif

struct ProgramExecutionState {
  Program active_program;
  std::uint32_t cursor;
  std::uint32_t start_tick;
  std::uint32_t last_emission_tick;
  std::uint32_t active;
  std::uint32_t completed;
  std::uint32_t emitted_steps;
};
static_assert(std::is_standard_layout_v<ProgramExecutionState> &&
              std::is_trivial_v<ProgramExecutionState>);

struct DueProgramCandidate {
  ProgramStep step;
  std::uint64_t program_identity;
  std::uint64_t participation_identity;
  std::uint32_t program_depth;
  std::uint32_t step_index;
  std::uint32_t admitted;
};
static_assert(std::is_standard_layout_v<DueProgramCandidate> &&
              std::is_trivial_v<DueProgramCandidate>);

DIRECT_CAUSAL_EXECUTOR_HD inline bool begin_execution(
    ProgramExecutionState* state, const Program& program,
    std::uint64_t current_participation_identity, std::uint32_t current_tick) {
  if (state == nullptr || program.step_count == 0u ||
      program.step_count > kMaxProgramSteps ||
      !initiation_current(program, current_participation_identity, current_tick))
    return false;
  *state = ProgramExecutionState{};
  state->active_program = program;
  state->start_tick = current_tick;
  state->active = 1u;
  return true;
}

DIRECT_CAUSAL_EXECUTOR_HD inline DueProgramCandidate due_candidate(
    const ProgramExecutionState& state, std::uint32_t current_tick,
    std::uint64_t current_participation_identity) {
  DueProgramCandidate out{};
  if (state.active == 0u || state.completed != 0u ||
      state.cursor >= state.active_program.step_count ||
      !initiation_current(state.active_program, current_participation_identity,
                          current_tick) ||
      current_tick < state.start_tick)
    return out;
  const ProgramStep& step = state.active_program.steps[state.cursor];
  const std::uint32_t elapsed = current_tick - state.start_tick;
  if (step.due_offset != elapsed) return out;
  out.step = step;
  out.program_identity = state.active_program.identity;
  out.participation_identity =
      state.active_program.initiation_participation_identity;
  out.program_depth = state.active_program.depth;
  out.step_index = state.cursor;
  out.admitted = 1u;
  return out;
}

DIRECT_CAUSAL_EXECUTOR_HD inline bool confirm_emitted_step(
    ProgramExecutionState* state, const DueProgramCandidate& candidate,
    std::uint32_t emission_tick, std::uint32_t emitted_node,
    std::uint32_t emitted_channel) {
  if (state == nullptr || state->active == 0u || candidate.admitted == 0u ||
      candidate.program_identity != state->active_program.identity ||
      candidate.step_index != state->cursor ||
      candidate.participation_identity !=
          state->active_program.initiation_participation_identity ||
      candidate.step.node != emitted_node ||
      candidate.step.channel != emitted_channel ||
      emission_tick != state->start_tick + candidate.step.due_offset)
    return false;
  state->last_emission_tick = emission_tick;
  ++state->cursor;
  ++state->emitted_steps;
  if (state->cursor == state->active_program.step_count) {
    state->active = 0u;
    state->completed = 1u;
  }
  return true;
}

#undef DIRECT_CAUSAL_EXECUTOR_HD

}  // namespace substrate::direct_causal_program

#endif
