#include <cstdint>
#include <cstdio>
#include <cstring>
#if !defined(__CUDACC__)
#define __host__
#define __device__
#define __forceinline__ inline
template <typename T,typename U> inline T atomicAdd(T* address,U value){T old=*address;*address=static_cast<T>(old+static_cast<T>(value));return old;}
#endif
#include "hardware_native/direct_language_plan_causal_lowering.cuh"
#include "hardware_native/direct_causal_program_executor.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_causal_program;

static DirectLanguageMotorPlan plan(std::uint32_t word_bias=0) {
  DirectLanguageMotorPlan p{};
  p.admitted=1u; p.step_count=5u; p.supporting_transitions=4u;
  p.distinct_channels=2u; p.revision_identity=0xabcddcbaULL;
  for (std::uint32_t i=0;i<5u;++i) {
    p.steps[i].node=300u+i; p.steps[i].channel=20u+(i&1u);
    p.steps[i].word=0x41410000u+word_bias+i; p.steps[i].due_offset=i+1u;
  }
  return p;
}

static bool finish(ProgramExecutionState* state,std::uint32_t begin_index) {
  for (std::uint32_t i=begin_index;i<5u;++i) {
    const auto c=due_candidate(*state,101u+i,0x55u);
    if (!c.admitted || !confirm_emitted_step(state,c,101u+i,c.step.node,c.step.channel))
      return false;
  }
  return state->completed!=0u && state->emitted_steps==5u;
}

int main() {
  const auto source=plan();
  const auto word_changed=plan(0x10100u);
  const Program program=lower_language_plan_to_causal_program(
      source,0x55u,(7ull<<32)|3ull,140u);
  const Program word_free=lower_language_plan_to_causal_program(
      word_changed,0x55u,(7ull<<32)|3ull,140u);
  if (program.identity==0u || program.identity!=word_free.identity ||
      program.step_count!=5u || program.depth!=4u) return 2;

  ProgramExecutionState live{};
  const bool began=begin_execution(&live,program,0x55u,100u);
  bool prefix=began;
  for (std::uint32_t i=0u;i<2u && prefix;++i) {
    const auto c=due_candidate(live,101u+i,0x55u);
    prefix=c.admitted!=0u && confirm_emitted_step(&live,c,101u+i,c.step.node,c.step.channel);
  }

  // Program execution state is POD resident state.  A byte-exact copy models the
  // checkpoint payload boundary; no surface transcript or word is needed to resume.
  ProgramExecutionState restored{};
  std::memcpy(&restored,&live,sizeof(restored));
  const bool checkpoint_exact=std::memcmp(&restored,&live,sizeof(restored))==0;
  const bool original_finished=finish(&live,2u);
  const bool restored_finished=finish(&restored,2u);
  const bool deterministic_resume=original_finished && restored_finished &&
      std::memcmp(&restored,&live,sizeof(restored))==0;

  ProgramExecutionState forged{};
  const bool forged_begin=begin_execution(&forged,program,0x56u,100u);
  const bool green=began && prefix && checkpoint_exact && deterministic_resume &&
      !forged_begin && program.identity==word_free.identity;
  std::printf(
      "FOUNDRY_DIRECT_LANGUAGE_CAUSAL_EXECUTION_%s plan_steps=5 program_depth=%u "
      "public_steps=%u checkpoint_resume=%u frozen_surface=0 word_free=%u forged=0\n",
      green?"GREEN":"RED",program.depth,live.emitted_steps,
      deterministic_resume?1u:0u,program.identity==word_free.identity?1u:0u);
  return green?0:1;
}
