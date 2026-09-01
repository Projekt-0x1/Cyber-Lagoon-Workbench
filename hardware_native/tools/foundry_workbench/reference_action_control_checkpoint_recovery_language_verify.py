#!/usr/bin/env python3
from __future__ import annotations
import json,time
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q

STATE=(1101,); GOAL=(1102,); LANG=1201; ORDINARY=1202; SRC1=1301; SRC2=1302
SURFACE='I can do that now.'

def matched_ecology():
    e=TransitionEcologyV1()
    for action in (LANG,ORDINARY):
        e.observe(STATE,action,GOAL,1,SRC1);e.observe(STATE,action,GOAL,1,SRC2)
    return e

def prior_control_bank():
    b=PredictiveCreditBankV1(8)
    for action in (LANG,ORDINARY):
        b.observe_use(action,10,11,Q//16,77);b.observe_return(action,Q//2,0,12,True,77)
    for _ in range(2):b.observe_control(LANG,True,True)
    for _ in range(2):b.observe_control(LANG,False,True)
    for public in (False,True,False,True):b.observe_control(ORDINARY,public,True)
    return b

def yoked_bank():
    b=PredictiveCreditBankV1(8)
    for action in (LANG,ORDINARY):
        b.observe_use(action,10,11,Q//16,77);b.observe_return(action,Q//2,0,12,True,77)
        for public in (False,True,False,True):b.observe_control(action,public,True)
    return b

def main():
    started=time.perf_counter(); ecology=matched_ecology()
    control=prior_control_bank(); nominations=LanguageActionNominationBankV1()
    # A pre-checkpoint prospective occurrence exists but is deliberately transient.
    nominations.nominate(LANG,2001,3001,2002,2003,2004,20)
    ncp=nominations.checkpoint(); ccp=control.checkpoint()
    restored_n=LanguageActionNominationBankV1.restore(ncp)
    restored_c=PredictiveCreditBankV1.restore(ccp)
    checks={'pending_nomination_not_checkpointed':not restored_n._current,
            'control_history_checkpointed':restored_c.row(LANG).control_history_q16>0}
    # New current situation after restore rematerializes the semantic candidate.
    remat=restored_n.nominate(LANG,2001,3101,2102,2103,2004,21)
    low=restored_n.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=0,
                             action_costs={LANG:1,ORDINARY:1},control_bank=restored_c)
    checks['transient_resource_veto_defers']=low.status==0 and low.resource_veto==1
    # Veto is not learning and not semantic deletion. Later resource recovery uses
    # the same learned control history and same semantic candidate without reteaching.
    restored_n.clear(); recovered_nom=restored_n.nominate(LANG,2001,3201,2202,2203,2004,22)
    high=restored_n.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                              action_costs={LANG:1,ORDINARY:1},control_bank=restored_c)
    checks['same_semantic_candidate_survives_veto_recovery']=(remat.action_identity==recovered_nom.action_identity==LANG)
    checks['resource_recovery_resumes_language_without_reteaching']=(high.status==1 and high.action_identity==LANG)
    # Matched current evidence with yoked history remains unresolved.
    y=LanguageActionNominationBankV1();y.nominate(LANG,2001,3201,2202,2203,2004,22)
    yd=y.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                   action_costs={LANG:1,ORDINARY:1},control_bank=yoked_bank())
    checks['yoked_history_control_remains_unresolved']=yd.status==2 and yd.action_identity==0
    checks['current_transition_support_matched']=(ecology.transition(STATE,LANG,1).support==ecology.transition(STATE,ORDINARY,1).support==2)
    checks['visible_recovery_improves_discourse_continuity']=high.status==1
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_ACTION_CONTROL_CHECKPOINT_RECOVERY_LANGUAGE_GREEN','pass':not failed,'reference_only':True,'graph_flip':False,
      'visible_language_gain':'TRANSIENT_DEFERRAL_RECOVERS_TO_LANGUAGE_WITHOUT_SEMANTIC_RETEACHING',
      'before_public_sequence':['',''],'after_public_sequence':['',SURFACE],
      'semantic_candidate_identity':LANG,'resource_veto_status':low.status,'recovered_status':high.status,'yoked_status':yd.status,
      'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3),
      'remaining_red':['DIRECT_ACTION_CONTROL_RUNTIME_BEHAVIORAL_WIRING','PHYSICAL_CHECKPOINT_RECOVERY_ASSAY','DIRECT_LANGUAGE_TO_CANONICAL_ARBITRATION_WIRING','GRAPH_PROMOTION']}
    print(result['contract'] if not failed else 'FOUNDRY_ACTION_CONTROL_CHECKPOINT_RECOVERY_LANGUAGE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
