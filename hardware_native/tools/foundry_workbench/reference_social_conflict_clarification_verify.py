#!/usr/bin/env python3
"""Visible social-discourse ratchet: learned conflict asks a more informative question."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_predictive_credit_profile_v1 import Q

def main():
    started=time.perf_counter();adult=LanguageMasteryAdultV1()
    SAY=0x6101;GENERIC=0x6102;CONFLICT=0x6103
    for concept,text in ((GENERIC,'where do you think it is?'),(CONFLICT,"you've said both; which one do you mean?")):
        raw=tuple(text.encode())
        adult.observe_surface_item(concept,raw,10000+concept);adult.observe_surface_item(concept,raw,20000+concept)
        if not adult.observe_surface_construction(SAY,(concept,),raw,30000+concept):raise RuntimeError('social-conflict:surface')
    generic=adult.leaf(SAY,(GENERIC,));conflict=adult.leaf(SAY,(CONFLICT,))
    P_GENERIC=0xE201;P_CONFLICT=0xE202
    OBJ=0x7001;STATE=0x7103;STATE_NEW=0x7104;AA=0x7201;AB=0x7202;P_MIX=0x7303;P_NEW=0x7304
    for n,action in enumerate((AA,AB,AA,AB)):
        adult.observe_social_contact(P_MIX,OBJ,STATE,AB,0x8200+n);adult.observe_social_behavior(P_MIX,OBJ,action,True)
    adult.observe_social_contact(P_MIX,OBJ,STATE,AB,0x8303);ctx_conflict=adult._current_selection_context
    adult.observe_social_contact(P_NEW,OBJ,STATE_NEW,AB,0x8304);ctx_unknown=adult._current_selection_context
    for cycle in range(4):
        for index,(pid,root,outcome,partner,state) in enumerate((
            (P_CONFLICT,conflict,3*Q//4,P_MIX,STATE),(P_GENERIC,generic,-Q//4,P_MIX,STATE),
            (P_GENERIC,generic,3*Q//4,P_NEW,STATE_NEW),(P_CONFLICT,conflict,-Q//4,P_NEW,STATE_NEW))):
            adult.observe_social_contact(partner,OBJ,state,AB,0x880000+cycle*8+index)
            adult.experience_atomic_program(pid,root,Q//4,0,None,Q//8,True)
            adult.experience_choice(pid,outcome,Q//16 if outcome>0 else -Q//16,None,Q//8,2,True)
    adult.observe_social_contact(P_MIX,OBJ,STATE,AB,0x8313);mixed=adult.choose(AdultStateV1())
    adult.observe_social_contact(P_NEW,OBJ,STATE_NEW,AB,0x8314);unknown=adult.choose(AdultStateV1())
    checks={
        'conflict_and_unknown_have_distinct_resident_contexts':ctx_conflict!=ctx_unknown,
        'conflicting_history_asks_informative_clarification':mixed==P_CONFLICT,
        'no_history_keeps_generic_question':unknown==P_GENERIC,
        'visible_discussion_improvement':mixed==P_CONFLICT and bytes(adult.public_surface(mixed))==b"you've said both; which one do you mean?",
        'no_host_answer_selector':not hasattr(adult.prospection,'conflict_question') and not hasattr(adult,'prompt'),
    }
    failed=[k for k,v in checks.items() if not v]
    def shown(pid):
        if not pid:return ''
        surface=adult.public_surface(pid)
        return '' if surface is None else bytes(surface).decode()
    result={'contract':'FOUNDRY_SOCIAL_CONFLICT_CLARIFICATION_GREEN','pass':not failed,'checks':checks,
            'conversation':{'conflicting_history':shown(mixed),'no_history':shown(unknown)},
            'runtime_llm':False,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(('FOUNDRY_SOCIAL_CONFLICT_CLARIFICATION_GREEN' if not failed else 'FOUNDRY_SOCIAL_CONFLICT_CLARIFICATION_RED '+','.join(failed)))
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
