#!/usr/bin/env python3
import json,time
from reference_outcome_specific_action_credit_v1 import *
P_A,P_B=0xDA01,0xDB01;O_A,O_B=0xEA01,0xEB01
S_A=b'I will use the gauge.';S_B=b'I will use the tap.'
def seed(cls):
    m=cls();m.add(P_A,S_A);m.add(P_B,S_B)
    for _ in range(2):m.action_return(P_A,O_A);m.action_return(P_B,O_B)
    m.background_non_event(P_A,O_A);m.background_non_event(P_B,O_B)
    return m
def main():
    t=time.perf_counter();pre=seed(GenericOutcomePredecessor);cur=seed(OutcomeSpecificActionCreditV1)
    for _ in range(2):pre.background(O_A,True);cur.background(O_A,True)
    pre_choice=pre.choose();cur_choice=cur.choose()
    a=cur.programs[P_A].outcomes[O_A];b=cur.programs[P_B].outcomes[O_B];b_free_a=cur.programs[P_B].outcomes[O_A]
    checks={
      'same_reward_different_outcomes_remain_distinct':O_A!=O_B and a.value==b.value==Q,
      'free_outcome_a_degrades_only_a_control':not a.control() and b.control(),
      'counterfactual_a_row_does_not_rewrite_b_outcome':b_free_a.action==0 and b_free_a.background_success==2 and b.control(),
      'generic_predecessor_collapses_to_silence':pre_choice==0,
      'outcome_specific_credit_selects_other_language_action':cur_choice==P_B,
      'visible_discussion_improvement':pre_choice==0 and cur_choice==P_B,
      'bounded':time.perf_counter()-t<0.1,
    }
    failed=[k for k,v in checks.items() if not v]
    print('FOUNDRY_OUTCOME_SPECIFIC_ACTION_LANGUAGE_'+('GREEN' if not failed else 'RED'))
    print("before_public_sequence=['']")
    print("after_public_sequence=['I will use the tap.']")
    print(json.dumps({'checks':checks,'failed':failed,'language_phenotype_improved':not failed,'visible_language_gain':'OUTCOME_SPECIFIC_FREE_EVENT_SPARES_UNRELATED_LANGUAGE_ACTION','reference_only':True},indent=2,sort_keys=True))
    return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
