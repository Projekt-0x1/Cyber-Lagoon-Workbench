#!/usr/bin/env python3
"""N+1: identical reduced follow-up competes between acknowledgment and continuation."""
from __future__ import annotations
import copy,inspect,json,time
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE,CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_pragmatic_response_bridge_v1 import LivedWorldPragmaticResponseBridgeV1
from reference_predictive_credit_profile_v1 import Q
from reference_self_initiated_world_followup_verify import REFERENCE_SUCCESSOR,prepared
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
from reference_world_derived_proposition_frontier_verify import WORLD_A,SOURCE_A,world

RIGHT=0xF301;MARK=0xF303;RIGHT_CTX=0xF302;NAME=0xD700;ACK=0xF401;CONT=0xF402;CHANNEL=7
visible_language_gain='IDENTICAL_REDUCED_CONFIRMATION_CONTACT_SELECTS_ACKNOWLEDGMENT_OR_CONTINUATION_FROM_CONSEQUENCE_HISTORY'
language_phenotype_improved=True;future_update_authority_preserved=True

def teach_right(a):
    c=LanguageMasteryContactAdapterV1(a)
    for feature,units,base in ((RIGHT,b'right',0xA100),(MARK,b'?',0xA140)):
        for k in range(2):
            c.contact(CONTACT_SCENE,(NAME,feature),base+k)
            if not c.contact(CONTACT_SURFACE,units,base+0x10+k):raise RuntimeError('pragmatic:item')
    for k in range(2):
        c.contact(CONTACT_SCENE,(RIGHT_CTX,RIGHT,MARK),0xA120+k)
        if not c.contact(CONTACT_SURFACE,b'is that right?',0xA130+k):raise RuntimeError('pragmatic:construction')
    rows=a.language.invert_surface(tuple(b'is that right?'))
    if len(rows)!=1 or rows[0].context!=RIGHT_CTX or tuple(rows[0].atoms)!=(RIGHT,MARK):raise RuntimeError('pragmatic:inverse')

def register_responses(a,o):
    world(o,WORLD_A,SOURCE_A);frontier=WorldDiscourseSituationBridgeV1.frontier(a,o)
    successor=next(x for x in frontier if bytes(x.surface)==REFERENCE_SUCCESSOR)
    ack=a.leaf_surface(0xF310,1,tuple(b'okay.'))
    context=LivedWorldPragmaticResponseBridgeV1.context(o,RIGHT_CTX,(RIGHT,MARK))
    # Register both without giving either a large unilateral value advantage.
    a.experience_atomic_program(ACK,ack,Q//4,context=context,effort_q16=Q//32,controllable=True)
    a.experience_atomic_program(CONT,successor,Q//4,context=context,effort_q16=Q//32,controllable=True)
    return context

def settle_history(a,context,winner,loser,repetitions=4):
    for _ in range(repetitions):
        a.experience_choice(winner,Q,context=context,effort_q16=Q//32,duration=1,controllable=True,independent_return=True)
        a.experience_choice(loser,-Q//2,context=context,effort_q16=Q//32,duration=1,controllable=True,independent_return=True)

def ask(a,o,last,source):
    prior=a.last_completed_public_context();m=LanguageMasteryContactAdapterV1(a)
    sid=m.contact(CONTACT_UTTERANCE,tuple(b'is that right?'),source,CHANNEL);row=m.scenes.get(int(sid)) if int(sid)>0 else None
    if row is None:return int(sid),0,b'',prior
    _ctx,pid=LivedWorldPragmaticResponseBridgeV1.activate_program(a,o,row.context,row.atoms,last)
    return int(sid),int(pid),emit_choice(a,pid),prior

def branch(base,org_cp,winner,loser):
    a=type(base).restore(copy.deepcopy(base.checkpoint()));o=type(org_cp).restore(copy.deepcopy(org_cp.checkpoint()))
    world(o,WORLD_A,SOURCE_A);context=LivedWorldPragmaticResponseBridgeV1.context(o,RIGHT_CTX,(RIGHT,MARK));settle_history(a,context,winner,loser)
    learned=copy.deepcopy(a.checkpoint());a=type(a).restore(copy.deepcopy(learned));o=type(o).restore(copy.deepcopy(o.checkpoint()));world(o,WORLD_A,SOURCE_A);speech,last=quiet(a,o,0);return a,o,speech,last,learned

def main():
    started=time.perf_counter();checks={};base,org=prepared();teach_right(base);context=register_responses(base,org);base_cp=copy.deepcopy(base.checkpoint());org_cp=copy.deepcopy(org.checkpoint())
    ack=type(base).restore(copy.deepcopy(base_cp));ack_org=type(org).restore(copy.deepcopy(org_cp));settle_history(ack,context,ACK,CONT);ack_learned=copy.deepcopy(ack.checkpoint());ack=type(ack).restore(copy.deepcopy(ack_learned));ack_org=type(ack_org).restore(copy.deepcopy(org_cp));world(ack_org,WORLD_A,SOURCE_A);ack_speech,ack_last=quiet(ack,ack_org,0);asid,apid,aout,aprior=ask(ack,ack_org,ack_last,0xB100)
    cont=type(base).restore(copy.deepcopy(base_cp));cont_org=type(org).restore(copy.deepcopy(org_cp));settle_history(cont,context,CONT,ACK);cont_learned=copy.deepcopy(cont.checkpoint());cont=type(cont).restore(copy.deepcopy(cont_learned));cont_org=type(cont_org).restore(copy.deepcopy(org_cp));world(cont_org,WORLD_A,SOURCE_A);cont_speech,cont_last=quiet(cont,cont_org,0);csid,cpid,cout,cprior=ask(cont,cont_org,cont_last,0xB200)
    checks['same_current_world_and_same_reduced_contact_have_distinct_history_selected_public_actions']=(ack_speech==cont_speech and asid and csid and aprior==cprior==WorldDiscourseSituationBridgeV1.context(ack_org) and apid==ACK and aout==b'okay.' and cpid==CONT and cout==REFERENCE_SUCCESSOR)
    checks['same_query_structure_has_one_learned_binding']=(len(base.language.invert_surface(tuple(b'is that right?')))==1)
    # No completed self turn means no pragmatic authority even with learned response credit.
    early=type(ack).restore(copy.deepcopy(ack_learned));early_org=type(ack_org).restore(copy.deepcopy(org_cp));world(early_org,WORLD_A,SOURCE_A);eid,epid,eout,_=ask(early,early_org,int(early_org.world_state_occurrence),0xB300)
    checks['same_contact_without_completed_self_discourse_refuses']=(eid and epid==0 and eout==b'')
    # Checkpoint after the self turn preserves the opaque completed context and the same response.
    resumed=type(ack).restore(copy.deepcopy(ack_learned));res_org=type(ack_org).restore(copy.deepcopy(org_cp));world(res_org,WORLD_A,SOURCE_A);rspeech,rlast=quiet(resumed,res_org,0);mid=copy.deepcopy(resumed.checkpoint());rr=type(resumed).restore(copy.deepcopy(mid));rid,rpid,rout,rprior=ask(rr,res_org,rlast,0xB400)
    checks['checkpoint_between_self_turn_and_reduced_contact_preserves_history_selected_acknowledgment']=(rspeech==ack_speech and rpid==ACK and rout==b'okay.' and rprior==WorldDiscourseSituationBridgeV1.context(res_org))
    # Equal value/history must refuse arbitrary pragmatic interpretation.
    tied=type(base).restore(copy.deepcopy(base_cp));torg=type(org).restore(copy.deepcopy(org_cp));world(torg,WORLD_A,SOURCE_A);tspeech,tlast=quiet(tied,torg,0);tid,tpid,tout,_=ask(tied,torg,tlast,0xB500)
    checks['equal_response_history_refuses_arbitrary_confirmation_or_continuation']=(tid and tpid==0 and tout==b'')
    # Current pressure cannot invent an unsupported response in a naive context.
    naive=type(base).restore(copy.deepcopy(base_cp));norg=type(org).restore(copy.deepcopy(org_cp));world(norg,WORLD_A,SOURCE_A);nspeech,nlast=quiet(naive,norg,0);m=LanguageMasteryContactAdapterV1(naive);sid=m.contact(CONTACT_UTTERANCE,tuple(b'is that right?'),0xB600,CHANNEL);row=m.scenes.get(int(sid));ctx,pid=LivedWorldPragmaticResponseBridgeV1.activate_program(naive,norg,row.context,row.atoms,nlast)
    checks['unearned_history_stays_silent']=(nspeech and ctx==0 and pid==0)
    source=inspect.getsource(LivedWorldPragmaticResponseBridgeV1).lower()
    checks['bridge_has_no_confirmation_continue_yes_right_or_answer_policy']=(all(x not in source for x in ('confirm','continue','yes','right','answer','expected','okay')))
    blob=json.dumps(ack_learned,sort_keys=True)
    checks['checkpoint_has_no_confirmation_or_turn_policy_state']=(all(x not in blob for x in ('confirmation_mode','continue_mode','turn_policy','is that right?')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.pragmatic-confirmation-continuation.v1','contract':'FOUNDRY_PRAGMATIC_CONFIRMATION_CONTINUATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':future_update_authority_preserved,'visible_language_gain':visible_language_gain,'conversation':{'ack':[ack_speech.decode(),'is that right?',aout.decode() if aout else ''],'continue':[cont_speech.decode(),'is that right?',cout.decode() if cout else '']},'checks':checks,'failed':failed,'remaining_red':['REFERENCE_TO_PRIOR_NONCURRENT_EPISODE','OPEN_ENDED_CONVERSATION','DIRECT_PRAGMATIC_RESPONSE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print('visible_language_gain='+visible_language_gain);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
