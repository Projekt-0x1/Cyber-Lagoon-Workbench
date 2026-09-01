#!/usr/bin/env python3
from __future__ import annotations
import copy,json,time
from reference_cognition_v1 import TransitionEcologyV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q

STATE=(101,); GOAL=(202,)
LANG=301; ORDINARY=302; SRC1=401; SRC2=402; SRC3=403

def main():
    started=time.perf_counter(); ecology=TransitionEcologyV1(); bank=LanguageActionNominationBankV1()
    # Language nomination itself has no evidence. Ordinary lived transitions supply all support.
    before=len(ecology._evidence)
    n=bank.nominate(LANG,501,1501,502,503,504,1)
    checks={
      'nomination_is_endogenous_authority_none':n.lineage==1 and n.authority==0,
      'nomination_adds_no_transition_evidence':len(ecology._evidence)==before,
    }
    # One lived positive source for language candidate, two for ordinary candidate: ordinary wins.
    ecology.observe(STATE,LANG,GOAL,1,SRC1)
    ecology.observe(STATE,ORDINARY,GOAL,1,SRC2)
    ecology.observe(STATE,ORDINARY,GOAL,1,SRC3)
    d1=bank.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                      action_costs={LANG:1,ORDINARY:1})
    checks['ordinary_evidence_can_beat_language_nomination']=(d1.status==1 and d1.action_identity==ORDINARY and d1.nomination_identity==0)

    # Equal evidence must fail closed; no action-id/storage-order tie break.
    ecology.observe(STATE,LANG,GOAL,1,SRC2)
    d2=bank.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                      action_costs={LANG:1,ORDINARY:1})
    checks['equal_best_candidates_fail_closed']=(d2.status==2 and d2.action_identity==0 and d2.alternatives==2)

    # New independent evidence can make language candidate win, but only via ordinary evidence.
    ecology.observe(STATE,LANG,GOAL,1,SRC3)
    d3=bank.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                      action_costs={LANG:1,ORDINARY:1})
    checks['language_candidate_can_win_only_after_ordinary_evidence']=(d3.status==1 and d3.action_identity==LANG and d3.nomination_identity==n.identity)

    # Sapolsky/resource axis can veto the same language candidate without altering learned evidence.
    evidence_before=copy.deepcopy(ecology._evidence)
    d4=bank.arbitrate(ecology,STATE,GOAL,(ORDINARY,),available_resource=0,
                      action_costs={LANG:1,ORDINARY:1})
    checks['resource_state_can_veto_without_rewriting_semantics']=(d4.status==0 and d4.resource_veto==1 and ecology._evidence==evidence_before)

    # Destructive Sapolsky arm: matched present transition evidence and matched
    # current controllability truth, different lived controllability history.
    # The prior-control twin may commit the language candidate; the yoked twin
    # must remain unresolved. Candidate identity and current evidence are held fixed.
    def history_twin(prior_control):
        cb=PredictiveCreditBankV1(8)
        for action in (LANG,ORDINARY):
            for n in range(2):
                tick=100+n*5
                cb.observe_use(action,tick,tick+1,Q//16,77)
                cb.observe_return(action,Q//2,0,tick+2,True,77)
        if prior_control:
            for _ in range(2):cb.observe_control(LANG,True,True)
            for _ in range(2):cb.observe_control(LANG,False,True)
        else:
            for public_action in (False,True,False,True):cb.observe_control(LANG,public_action,True)
        # Ordinary candidate receives matched current contingency but no prior-control trace.
        for public_action in (False,True,False,True):cb.observe_control(ORDINARY,public_action,True)
        return cb
    prior_control=history_twin(True); yoked=history_twin(False)
    prior_row=prior_control.row(LANG); yoked_row=yoked.row(LANG)
    history_ecology=TransitionEcologyV1()
    for action in (LANG,ORDINARY):
        history_ecology.observe(STATE,action,GOAL,1,SRC1)
        history_ecology.observe(STATE,action,GOAL,1,SRC2)
    prior_decision=bank.arbitrate(history_ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                                  action_costs={LANG:1,ORDINARY:1},control_bank=prior_control)
    yoked_decision=bank.arbitrate(history_ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                                  action_costs={LANG:1,ORDINARY:1},control_bank=yoked)
    checks['matched_current_transition_support']=(history_ecology.transition(STATE,LANG,1).support==history_ecology.transition(STATE,ORDINARY,1).support==2)
    checks['matched_current_control_truth']=(
        prior_row.controllability_q16==yoked_row.controllability_q16==0 and
        (prior_row.control_attempts,prior_row.control_successes,prior_row.background_attempts,prior_row.background_successes)==
        (yoked_row.control_attempts,yoked_row.control_successes,yoked_row.background_attempts,yoked_row.background_successes)==(2,2,2,2))
    checks['prior_control_history_is_distinct']=prior_row.control_history_q16>yoked_row.control_history_q16==0
    checks['control_history_changes_commit_without_semantic_rewrite']=(
        prior_decision.status==1 and prior_decision.action_identity==LANG and
        yoked_decision.status==2 and yoked_decision.action_identity==0 and
        n.action_identity==LANG)
    lesioned=PredictiveCreditBankV1.restore(prior_control.checkpoint())
    lesioned.row(LANG).control_history_q16=0
    lesion_decision=bank.arbitrate(history_ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                                   action_costs={LANG:1,ORDINARY:1},control_bank=lesioned)
    checks['focal_control_history_lesion_removes_advantage']=(lesion_decision.status==2)

    # Destructive timescale/recovery arm. Continued uncontrollable background
    # outcomes extinguish the history advantage; informative background non-events
    # later restore action-vs-background contingency without changing semantics.
    extinguished=PredictiveCreditBankV1.restore(prior_control.checkpoint())
    for _ in range(6):extinguished.observe_control(LANG,False,True)
    extinguished_decision=bank.arbitrate(history_ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                                         action_costs={LANG:1,ORDINARY:1},control_bank=extinguished)
    checks['continued_uncontrollable_history_extinguishes_advantage']=(
        extinguished.row(LANG).control_history_q16==0 and extinguished_decision.status==2)
    recovered=PredictiveCreditBankV1.restore(extinguished.checkpoint())
    for _ in range(8):recovered.observe_control(LANG,False,False)
    recovered_decision=bank.arbitrate(history_ecology,STATE,GOAL,(ORDINARY,),available_resource=10,
                                      action_costs={LANG:1,ORDINARY:1},control_bank=recovered)
    checks['ordinary_contingency_reacquires_control_advantage']=(
        recovered.row(LANG).control_supported and recovered_decision.status==1 and
        recovered_decision.action_identity==LANG and n.action_identity==LANG)

    # Checkpoint carries no pending nomination/answer; rematerialization is required after restore.
    restored=LanguageActionNominationBankV1.restore(bank.checkpoint())
    checks['checkpoint_drops_pending_nomination']=not restored._current

    # Visible phenotype: before arbitration-aware control the plausible language answer would speak;
    # now matched resource veto appropriately defers/silences it rather than forcing output.
    before_public='I can do that now.'
    after_public='' if d4.status==0 else 'I can do that now.'
    checks['visible_discussion_improvement_is_context_appropriate_deferral']=(before_public!='' and after_public=='')
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_LANGUAGE_NOMINATION_ARBITRATION_GREEN','pass':not failed,'reference_only':True,'graph_flip':False,
      'visible_language_gain':'LANGUAGE_CANDIDATE_DEFERS_WHEN_ORDINARY_RESOURCE_ARBITRATION_VETOES',
      'direct_arbitration_boundary':'CANONICAL_SOMATIC_EXACT_TIES_HAVE_NO_SELECTED_ACTION',
      'before_public':before_public,'after_public':after_public,'ordinary_winner':d1.action_identity,
      'tie_status':d2.status,'language_winner_after_evidence':d3.action_identity,'resource_veto_status':d4.status,'prior_control_status':prior_decision.status,'yoked_control_status':yoked_decision.status,'extinguished_status':extinguished_decision.status,'recovered_status':recovered_decision.status,
      'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3),
      'remaining_red':['DIRECT_LANGUAGE_TO_CANONICAL_ARBITRATION_WIRING','DIRECT_CONTROL_HISTORY_LOWERING','PHYSICAL_NOMINATION_SELECTION_ASSAY','GRAPH_PROMOTION'],
      'next_falsifiers':{
        'chomsky':'Held-out structural language candidate with local distractor must nominate structural action before ordinary arbitration.',
        'sapolsky':'Destructive arms exercised: matched-current prior-control vs yoked divergence, focal lesion, extinction, and ordinary contingency reacquisition; next add social/resource cross-over.'}}
    print(result['contract'] if not failed else 'FOUNDRY_LANGUAGE_NOMINATION_ARBITRATION_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
