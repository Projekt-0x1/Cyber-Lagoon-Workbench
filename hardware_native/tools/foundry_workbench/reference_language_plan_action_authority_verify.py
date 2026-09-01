#!/usr/bin/env python3
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
LANG=8101; OTHER=8102; STATE=(8103,); GOAL=(8104,); SURFACE='I can do that now.'
e=TransitionEcologyV1()
for a in (LANG,OTHER):
    e.observe(STATE,a,GOAL,1,8201); e.observe(STATE,a,GOAL,1,8202)
c=PredictiveCreditBankV1(8)
for a in (LANG,OTHER):
    c.observe_use(a,1,2,Q//16,77); c.observe_return(a,Q//2,0,3,True,77)
for _ in range(2): c.observe_control(LANG,True,True)
for _ in range(2): c.observe_control(LANG,False,True)
for public in (False,True,False,True): c.observe_control(OTHER,public,True)
b=LanguageActionNominationBankV1(); n=b.nominate(LANG,8301,8302,8303,8304,8305,8)
# Predecessor Direct tick coupled plan formation to node drive, so public intent
# effectively began at planning time. Challenger waits for ordinary arbitration.
before=[SURFACE]
precommit=['']
d=b.arbitrate(e,STATE,GOAL,(OTHER,),available_resource=10,
              action_costs={LANG:1,OTHER:1},control_bank=c)
after=precommit+([SURFACE] if d.status==1 and d.action_identity==LANG else [''])
checks={
 'plan_exists_before_commit':n.action_identity==LANG,
 'planning_alone_is_publicly_silent':precommit==[''],
 'ordinary_action_control_commits_candidate':d.status==1 and d.action_identity==LANG,
 'same_semantic_candidate_speaks_after_commit':after==['',SURFACE],
 'visible_discussion_improvement':before!=after and after[0]=='',
}
failed=[k for k,v in checks.items() if not v]
print('FOUNDRY_LANGUAGE_PLAN_ACTION_AUTHORITY_GREEN' if not failed else 'FOUNDRY_LANGUAGE_PLAN_ACTION_AUTHORITY_RED '+','.join(failed))
print('visible_language_gain=PLAN_FORMATION_WAITS_FOR_ACTION_AUTHORITY_BEFORE_PUBLIC_SPEECH')
print('before_public_sequence='+repr(before)); print('after_public_sequence='+repr(after))
raise SystemExit(0 if not failed else 1)
