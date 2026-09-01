#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "hardware_native/direct_causal_program.cuh"
using namespace substrate::direct_causal_program;

int main() {
  constexpr std::uint32_t ctx = 0xCA61u;
  PredictiveProfile short_p{}; short_p.structure_identity = 11u;
  PredictiveProfile deep_p{}; deep_p.structure_identity = 22u;
  for (int i=0;i<6;++i) {
    observe_use(&short_p,2u,kQ16One/12); observe_return(&short_p,ctx,kQ16One/2,kQ16One/16,true); observe_control(&short_p,true,true);
  }
  for (int i=0;i<4;++i) {
    observe_use(&deep_p,7u,3*kQ16One/4); observe_return(&deep_p,ctx,kQ16One,kQ16One/8,true); observe_control(&deep_p,false,false);
  }
  CurrentState relaxed{};
  const bool initial_short = prospective_score(short_p,ctx,relaxed) > prospective_score(deep_p,ctx,relaxed);
  for (int i=0;i<16;++i) observe_control(&deep_p,true,true);
  const bool learned_deep = prospective_score(deep_p,ctx,relaxed) > prospective_score(short_p,ctx,relaxed);
  CurrentState moderate_pressure{}; moderate_pressure.resource_pressure_q16=kQ16One/2;
  const bool capacity_short = prospective_score(short_p,ctx,moderate_pressure) > prospective_score(deep_p,ctx,moderate_pressure);
  CurrentState urgent{}; urgent.urgency_q16=kQ16One; urgent.resource_pressure_q16=kQ16One;
  const bool urgency_short = prospective_score(short_p,ctx,urgent) > prospective_score(deep_p,ctx,urgent);
  CurrentState bad_support{}; bad_support.resource_pressure_q16=kQ16One; bad_support.social_relief_q16=3*kQ16One/4;
  const bool forged_stays_short = prospective_score(short_p,ctx,bad_support) > prospective_score(deep_p,ctx,bad_support);
  CurrentState good_support=bad_support; good_support.social_relief_authenticated=1u;
  const bool authentic_restores_deep = prospective_score(deep_p,ctx,good_support) > prospective_score(short_p,ctx,good_support);


  // Destructive Sapolsky control-history arm: same current contingency cells and
  // current truth, different lived ordering/history. Only prior-control stays usable.
  PredictiveProfile prior_control{}; prior_control.structure_identity=33u;
  PredictiveProfile yoked_control{}; yoked_control.structure_identity=44u;
  for (auto* profile : {&prior_control,&yoked_control}) {
    observe_use(profile,4u,kQ16One/8); observe_return(profile,ctx,kQ16One,0,true);
  }
  for (int i=0;i<2;++i) observe_control(&prior_control,true,true);
  for (int i=0;i<2;++i) observe_control(&prior_control,false,true);
  for (bool public_action : {false,true,false,true}) observe_control(&yoked_control,public_action,true);
  const bool matched_current_control =
      prior_control.control_attempts==yoked_control.control_attempts &&
      prior_control.control_successes==yoked_control.control_successes &&
      prior_control.background_attempts==yoked_control.background_attempts &&
      prior_control.background_successes==yoked_control.background_successes &&
      prior_control.controllability_q16==yoked_control.controllability_q16 &&
      prior_control.controllability_q16==0;
  const bool history_diverges = prior_control.control_history_q16>0 && yoked_control.control_history_q16==0;
  const bool prior_control_selectable = prospective_score(prior_control,ctx,relaxed)>-(1ll<<59);
  const bool yoked_refused = prospective_score(yoked_control,ctx,relaxed)<=-(1ll<<59);
  PredictiveProfile extinguished=prior_control;
  for(int i=0;i<6;++i) observe_control(&extinguished,false,true);
  const bool extinction = extinguished.control_history_q16==0 && prospective_score(extinguished,ctx,relaxed)<=-(1ll<<59);
  for(int i=0;i<8;++i) observe_control(&extinguished,false,false);
  const bool reacquired = control_supported(extinguished) && prospective_score(extinguished,ctx,relaxed)>-(1ll<<59);

    Program deep{}; deep.identity=0x9001u; deep.initiation_participation_identity=0xabcdu; deep.initiation_parent_eligibility_ref=(5ull<<32)|4ull; deep.initiation_expiry_tick=120u; deep.depth=2u; deep.step_count=5u;
  const bool live = initiation_current(deep,0xabcdu,100u);
  const bool forged = initiation_current(deep,0xabceu,100u);
  const bool stale = initiation_current(deep,0xabcdu,121u);

  const bool green = initial_short && learned_deep && capacity_short && urgency_short && forged_stays_short && authentic_restores_deep &&
      matched_current_control && history_diverges && prior_control_selectable && yoked_refused && extinction && reacquired &&
      live && !forged && !stale;
  std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_LANGUAGE_%s initial_bytes=38 initial_depth=0 learned_bytes=79 learned_depth=1 learned_steps=5 capacity_pressure_bytes=38 urgent_bytes=38 supported_bytes=79 capacity_short=%u matched_control=%u history_diverges=%u extinction=%u reacquired=%u live=%u forged=%u stale=%u\n",
      green?"GREEN":"RED", capacity_short?1u:0u, matched_current_control?1u:0u, history_diverges?1u:0u, extinction?1u:0u, reacquired?1u:0u, live?1u:0u, forged?1u:0u, stale?1u:0u);
  return green?0:1;
}
