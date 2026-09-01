#!/usr/bin/env python3
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
LANG=9101; OTHER=9102; STATE=(9103,); GOAL=(9104,); SURFACE='I can do that now.'
# Current world/action evidence is matched; language planning is not itself evidence.
e=TransitionEcologyV1()
for a in (LANG,OTHER):
    e.observe(STATE,a,GOAL,1,9201); e.observe(STATE,a,GOAL,1,9202)
control=PredictiveCreditBankV1(8)
for a in (LANG,OTHER):
    control.observe_use(a,1,2,Q//16,77); control.observe_return(a,Q//2,0,3,True,77)
for _ in range(2): control.observe_control(LANG,True,True)
for _ in range(2): control.observe_control(LANG,False,True)
for public in (False,True,False,True): control.observe_control(OTHER,public,True)
# Predecessor after plan-only split: no owned candidate enters ordinary arbitration.
before=['']
# Challenger: current authenticated parent admits the candidate, still silently.
bank=LanguageActionNominationBankV1()
nom=bank.nominate(LANG,9301,9302,9303,9304,9305,8)
precommit=['']
decision=bank.arbitrate(e,STATE,GOAL,(OTHER,),available_resource=10,
                        action_costs={LANG:1,OTHER:1},control_bank=control)
after=precommit+([SURFACE] if decision.status==1 and decision.action_identity==LANG else [''])
checks={
 'candidate_identity_owned_before_selection':nom.action_identity==LANG,
 'candidate_admission_is_publicly_silent':precommit==[''],
 'ordinary_control_not_language_admission_selects':decision.status==1 and decision.action_identity==LANG,
 'visible_language_gain':before==[''] and after==['',SURFACE],
 'same_semantic_candidate':nom.action_identity==decision.action_identity==LANG,
}
failed=[k for k,v in checks.items() if not v]
print('FOUNDRY_LANGUAGE_OWNED_CANDIDATE_ADMISSION_GREEN' if not failed else 'FOUNDRY_LANGUAGE_OWNED_CANDIDATE_ADMISSION_RED '+','.join(failed))
print('visible_language_gain=OWNED_CANDIDATE_CAN_LATER_SPEAK_THROUGH_ORDINARY_CONTROL')
print('before_public_sequence='+repr(before));print('after_public_sequence='+repr(after))
raise SystemExit(0 if not failed else 1)
