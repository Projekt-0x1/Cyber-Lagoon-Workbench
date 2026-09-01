#ifndef HARDWARE_NATIVE_DIRECT_LANGUAGE_PLAN_CAUSAL_LOWERING_CUH
#define HARDWARE_NATIVE_DIRECT_LANGUAGE_PLAN_CAUSAL_LOWERING_CUH
#include <cstdint>
#include "hardware_native/direct_adult_language_expression_motor.cuh"
#include "hardware_native/direct_causal_program.cuh"
namespace substrate::direct_causal_program {
#if defined(__CUDACC__)
#define DIRECT_LANGUAGE_CAUSAL_HD __host__ __device__
#else
#define DIRECT_LANGUAGE_CAUSAL_HD
#endif
DIRECT_LANGUAGE_CAUSAL_HD inline std::uint64_t language_program_fold(std::uint64_t h,std::uint64_t v){h^=v+0x9e3779b97f4a7c15ULL+(h<<6u)+(h>>2u);return h;}
DIRECT_LANGUAGE_CAUSAL_HD inline Program lower_language_plan_to_causal_program(
    const direct_network::DirectLanguageMotorPlan& plan,
    std::uint64_t participation_identity,
    std::uint64_t parent_eligibility_ref,
    std::uint32_t expiry_tick) {
  Program out{};
  if (plan.admitted==0u || plan.step_count==0u ||
      plan.step_count>kMaxProgramSteps || participation_identity==0u ||
      parent_eligibility_ref==0u || expiry_tick==0u)
    return out;
  std::uint64_t identity=language_program_fold(0x4c5043415553414cULL,plan.revision_identity);
  for(std::uint32_t i=0;i<plan.step_count;++i){
    const auto& step=plan.steps[i];
    if(step.node==direct_adult_core::kInvalidIndex)return {};
    out.steps[i].node=step.node;
    out.steps[i].channel=step.channel;
    out.steps[i].due_offset=step.due_offset;
    identity=language_program_fold(identity,step.node);
    identity=language_program_fold(identity,step.channel);
    identity=language_program_fold(identity,step.due_offset);
  }
  out.identity=identity==0u?1u:identity;
  out.initiation_participation_identity=participation_identity;
  out.initiation_parent_eligibility_ref=parent_eligibility_ref;
  out.initiation_expiry_tick=expiry_tick;
  out.depth=plan.supporting_transitions==0u?1u:plan.supporting_transitions;
  out.step_count=plan.step_count;
  return out;
}
#undef DIRECT_LANGUAGE_CAUSAL_HD
} // namespace substrate::direct_causal_program
#endif
