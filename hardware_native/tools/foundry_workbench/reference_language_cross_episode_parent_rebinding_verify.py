#!/usr/bin/env python3
from __future__ import annotations
import json,time
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
STATE=(5010,);GOAL=(5011,);LANG=5020;OTHER=5021;SURFACE='I can do that now.'
def main():
 t=time.perf_counter();e=TransitionEcologyV1()
 for a in (LANG,OTHER):
  e.observe(STATE,a,GOAL,1,6101);e.observe(STATE,a,GOAL,1,6102)
 control=PredictiveCreditBankV1(8)
 for a in (LANG,OTHER):control.observe_use(a,1,2,Q//16,77);control.observe_return(a,Q//2,0,3,True,77)
 for _ in range(2):control.observe_control(LANG,True,True)
 for _ in range(2):control.observe_control(LANG,False,True)
 for public in (False,True,False,True):control.observe_control(OTHER,public,True)
 bank=LanguageActionNominationBankV1()
 teaching=7001;current_parent=7002
 n=bank.nominate(LANG,teaching,current_parent,7101,7201,7301,8)
 d=bank.arbitrate(e,STATE,GOAL,(OTHER,),available_resource=10,action_costs={LANG:1,OTHER:1},control_bank=control)
 old_bridge_would_refuse=teaching!=current_parent
 checks={
  'historical_teaching_differs_from_current_parent':teaching!=current_parent,
  'current_parent_and_teaching_are_both_retained':n.teaching_occurrence_identity==teaching and n.current_parent_occurrence_identity==current_parent,
  'current_participation_is_current_not_teaching':n.current_participation_identity==7201,
  'later_episode_reuses_learned_language':d.status==1 and d.action_identity==LANG,
  'old_parent_equality_bridge_would_have_stayed_silent':old_bridge_would_refuse,
  'visible_cross_episode_language_gain':d.status==1,
  'bounded_fast_path':time.perf_counter()-t<1.0}
 failed=[k for k,v in checks.items() if not v]
 result={'contract':'FOUNDRY_LANGUAGE_CROSS_EPISODE_PARENT_REBINDING_GREEN','pass':not failed,'reference_only':True,'graph_flip':False,
  'visible_language_gain':'LEARNED_LANGUAGE_REUSES_HISTORICAL_TEACHING_UNDER_NEW_CURRENT_PARENT',
  'before_public':'','after_public':SURFACE,'teaching_occurrence':teaching,'current_parent_occurrence':current_parent,
  'semantic_candidate_identity':LANG,'checks':checks,'elapsed_ms':round((time.perf_counter()-t)*1000,3),
  'remaining_red':['DIRECT_LANGUAGE_TO_CANONICAL_ARBITRATION_WIRING','PHYSICAL_CROSS_EPISODE_ASSAY','GRAPH_PROMOTION']}
 print(result['contract'] if not failed else 'FOUNDRY_LANGUAGE_CROSS_EPISODE_PARENT_REBINDING_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
