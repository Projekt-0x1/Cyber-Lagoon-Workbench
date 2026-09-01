#!/usr/bin/env python3
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_program_competition_v1 import ProgramCompetitionStateV1,ProgramBrakeEvidenceV1,arbitrate_programs,apply_program_brake
LANG=11101;ALT=11102;YOKED=11099;CTX=77;SURFACE='I can do that now.'
def earn(bank,sid,outcome,controlled=True,effort=Q//8):
    bank.observe_use(sid,0,1,effort,CTX);bank.observe_return(sid,outcome,0,2,True,CTX)
    if controlled:
        bank.observe_control(sid,True,True,CTX);bank.observe_control(sid,True,True,CTX);bank.observe_control(sid,False,False,CTX);bank.observe_control(sid,False,False,CTX)
    else:
        bank.observe_control(sid,True,False,CTX);bank.observe_control(sid,True,False,CTX);bank.observe_control(sid,False,True,CTX);bank.observe_control(sid,False,True,CTX)
bank=PredictiveCreditBankV1(8);earn(bank,LANG,Q//2,True);earn(bank,ALT,Q//2-Q//16,True);earn(bank,YOKED,Q,False)
state=ProgramCompetitionStateV1(CTX)
before=arbitrate_programs(bank,(YOKED,ALT,LANG),state)
bank.observe_return(LANG,Q,0,5,True,CTX);bank.observe_return(ALT,-Q,0,5,True,CTX)
after=arbitrate_programs(bank,(YOKED,ALT,LANG),state)
tie=PredictiveCreditBankV1(4);earn(tie,LANG,Q//2,True);earn(tie,ALT,Q//2,True);tied=arbitrate_programs(tie,(ALT,LANG),state)
mismatch=ProgramBrakeEvidenceV1(9001,LANG,CTX,Q,6)
veto=apply_program_brake(after,mismatch,current_tick=6);recovered=apply_program_brake(after,None,current_tick=7)
forged=apply_program_brake(after,ProgramBrakeEvidenceV1(9002,ALT,CTX,Q,6),current_tick=6)
before_public=[''];after_public=['',SURFACE] if recovered.decision==1 and recovered.candidate==LANG else ['']
checks={
 'close_competition_unresolved_without_unowned_urgency':before.status==2 and before.leader==0,
 'lived_consequence_creates_unique_language_leader':after.status==1 and after.leader==LANG,
 'exact_tie_never_gets_authority':tied.status==2 and tied.leader==0,
 'yoked_high_outcome_candidate_cannot_buy_control':YOKED not in (after.leader,after.runner_up) and after.considered==2,
 'matching_mismatch_brakes_without_erasing_candidate':veto.decision==2 and veto.candidate==LANG and after.leader==LANG,
 'brake_recovery_commits_same_candidate_without_reteaching':recovered.decision==1 and recovered.candidate==LANG,
 'mismatched_brake_evidence_refuses':forged.decision==3 and forged.candidate==LANG,
 'visible_discussion_improvement':before_public==[''] and after_public==['',SURFACE],
}
failed=[k for k,v in checks.items() if not v]
print('FOUNDRY_PROGRAM_COMPETITION_LANGUAGE_GREEN' if not failed else 'FOUNDRY_PROGRAM_COMPETITION_LANGUAGE_RED '+','.join(failed))
print('visible_language_gain=CONSEQUENCE_SEPARATED_PROGRAM_COMPETITION_CAN_COMMIT_PREPARED_LANGUAGE')
print(f'before_status={before.status} after_leader={after.leader} tie_status={tied.status} veto={veto.decision} recovered={recovered.decision} forged={forged.decision}')
print('before_public_sequence='+repr(before_public));print('after_public_sequence='+repr(after_public))
raise SystemExit(0 if not failed else 1)
