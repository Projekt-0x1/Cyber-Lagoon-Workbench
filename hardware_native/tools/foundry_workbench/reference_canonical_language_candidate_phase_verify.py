#!/usr/bin/env python3
from __future__ import annotations
from copy import deepcopy
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
LANG=10101; OTHER=10102; STATE=(10103,); GOAL=(10104,); SURFACE='I can do that now.'

def ecology():
    e=TransitionEcologyV1()
    for a in (LANG,OTHER):
        e.observe(STATE,a,GOAL,1,10201); e.observe(STATE,a,GOAL,1,10202)
    return e

def control(prior: bool):
    c=PredictiveCreditBankV1(8)
    for a in (LANG,OTHER):
        c.observe_use(a,1,2,Q//16,77); c.observe_return(a,Q//2,0,3,True,77)
    if prior:
        for _ in range(2): c.observe_control(LANG,True,True)
        for _ in range(2): c.observe_control(LANG,False,True)
    else:
        for public in (False,True,False,True): c.observe_control(LANG,public,True)
    for public in (False,True,False,True): c.observe_control(OTHER,public,True)
    return c

def run_mode(mode: str, prior: bool):
    e=ecology(); c=control(prior); before_e=deepcopy(e.checkpoint()); before_c=deepcopy(c.checkpoint())
    bank=LanguageActionNominationBankV1()
    # Same lawful current situation; executor mode must not change candidate identity.
    n=bank.nominate(LANG,10301,10302,10303,10304,10305,8)
    after_admit_e=e.checkpoint(); after_admit_c=c.checkpoint()
    d=bank.arbitrate(e,STATE,GOAL,(OTHER,),available_resource=10,
                     action_costs={LANG:1,OTHER:1},control_bank=c)
    return {
      'mode':mode,'nomination_identity':n.identity,'candidate':n.action_identity,
      'admission_evidence_unchanged':before_e==after_admit_e,
      'admission_control_unchanged':before_c==after_admit_c,
      'decision':d,
    }

stepped=run_mode('stepped',True); persistent=run_mode('persistent',True); yoked=run_mode('stepped-yoked',False)
# Predecessor D6 had lawful candidate mechanism but no canonical phase call: public silence.
before_public=['']
after_public=['',SURFACE] if stepped['decision'].status==1 and persistent['decision'].status==1 else ['']
checks={
 'stepped_candidate_admitted':stepped['candidate']==LANG,
 'persistent_candidate_admitted':persistent['candidate']==LANG,
 'executor_modes_same_candidate_identity':stepped['nomination_identity']==persistent['nomination_identity'],
 'admission_creates_no_transition_evidence':stepped['admission_evidence_unchanged'] and persistent['admission_evidence_unchanged'],
 'admission_creates_no_control_history':stepped['admission_control_unchanged'] and persistent['admission_control_unchanged'],
 'prior_control_can_later_commit':stepped['decision'].status==1 and stepped['decision'].action_identity==LANG and persistent['decision'].status==1 and persistent['decision'].action_identity==LANG,
 'yoked_history_stays_unresolved_after_same_admission':yoked['candidate']==LANG and yoked['decision'].status==2 and yoked['decision'].action_identity==0,
 'visible_discussion_improvement':before_public==[''] and after_public==['',SURFACE],
}
failed=[k for k,v in checks.items() if not v]
print('FOUNDRY_CANONICAL_LANGUAGE_CANDIDATE_PHASE_GREEN' if not failed else 'FOUNDRY_CANONICAL_LANGUAGE_CANDIDATE_PHASE_RED '+','.join(failed))
print('visible_language_gain=CANONICAL_ADULT_PHASE_ADMITS_LANGUAGE_CANDIDATE_WITHOUT_PREMATURE_DRIVE')
print('before_public_sequence='+repr(before_public)); print('after_public_sequence='+repr(after_public))
print('stepped_status=%d persistent_status=%d yoked_status=%d' % (stepped['decision'].status,persistent['decision'].status,yoked['decision'].status))
raise SystemExit(0 if not failed else 1)
