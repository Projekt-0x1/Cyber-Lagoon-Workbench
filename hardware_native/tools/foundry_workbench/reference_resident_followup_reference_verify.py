#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=0xFA01;NAME=0xFA02;P=0xFA03;SRC=0xFA100
R1=(1101,1201,1301,1401);R2=(1102,1202,1302,1402);R3=(1103,1203,1303,1403)
C1=(1501,1601,1701,1401);C2=(1502,1602,1702,1402);C3=(1503,1603,1703,1403)
WORDS={1101:'quiet',1201:'technician',1301:'inspects',1401:'sensor',1102:'careful',1202:'engineer',1302:'tests',1402:'valve',1103:'swift',1203:'observer',1303:'tracks',1403:'gauge',1501:'calm',1601:'observer',1701:'tracks',1502:'bright',1602:'operator',1702:'checks',1503:'patient',1603:'analyst',1703:'examines'}
def u(s):return tuple(s.encode())
def full(r):return f"the {WORDS[r[0]]} {WORDS[r[1]]} {WORDS[r[2]]} the {WORDS[r[3]]}."
def compact(r):return f"the {WORDS[r[0]]} {WORDS[r[1]]} {WORDS[r[2]]} it."
def partner(o):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,P),SRC,True,True)
def scene(o,r,src):return o.contact(CONTACT_SCENE,(7,CTX,4,*r),src,True,True)
def surface(o,text,src):return o.contact(CONTACT_SURFACE,u(text),src,True,True)
def name(o,e,src):
    o.contact(CONTACT_SCENE,(7,NAME,1,e),src,True,True);surface(o,WORDS[e],src+1000)
def learn_base(o):
    for i,e in enumerate(tuple(dict.fromkeys((*R1,*R2,*R3,*C1,*C2,*C3)))):
        name(o,e,10000+i);name(o,e,20000+i)
    scene(o,R1,30001);surface(o,full(R1),31001)
    scene(o,R2,30002);surface(o,full(R2),31002)
    assert o.language.template(CTX,4) is not None
def teach_followup(o,r,compact_scene,base):
    partner(o);scene(o,r,base);first=o.tick();assert isinstance(first,ActionV2) and bytes(first.payload).decode()==full(r)
    assert not o._shared_reinstated(P,r[-1]);o.contact(CONTACT_CONSEQUENCE,(first.ticket,1),P,True,True);assert o._shared_reinstated(P,r[-1])
    # Learn generic reference form from a real shared referent; no condition is supplied.
    for k in range(2):
        o.contact(CONTACT_SCENE,(7,NAME,1,r[-1]),base+100+k,True,True);_ctx,conds=o._surface_context(o.current_scene);assert COND_REINSTATED in conds[-1];surface(o,'it',base+200+k)
    # Learn compact clause only under the residently acquired shared condition.
    for k in range(2):
        scene(o,compact_scene,base+300+k);_ctx,conds=o._surface_context(o.current_scene);assert all(COND_REINSTATED not in row for row in conds[:-1]) and COND_REINSTATED in conds[-1];surface(o,compact(compact_scene),base+400+k)
    return first
def heldout(o,settle_first=True,base=50000):
    partner(o);scene(o,R3,base);first=o.tick()
    if not isinstance(first,ActionV2):return first,None,False,()
    before=o._shared_reinstated(P,R3[-1])
    if settle_first:o.contact(CONTACT_CONSEQUENCE,(first.ticket,1),P,True,True)
    after=o._shared_reinstated(P,R3[-1])
    # If unsettled, retire the pending action branch by checkpointing is forbidden; do not request a second motor action.
    if not settle_first:return first,None,(not before and not after),()
    scene(o,C3,base+1);_ctx,conds=o._surface_context(o.current_scene);second=o.tick()
    return first,second,(not before and after),conds
def main():
    started=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));learn_base(o)
    teach_followup(o,R1,C1,40000);teach_followup(o,R2,C2,45000)
    generalized=o.language.condition_form_candidates((COND_REINSTATED,))
    checks['resident_dialogue_examples_generalize_compact_reference_form']=any(row[3]==u('it') and row[5]>=2 for row in generalized)
    cp=copy.deepcopy(o.checkpoint());first,second,acquired,conds=heldout(o,True,50000)
    checks['heldout_first_turn_uses_full_reference']=isinstance(first,ActionV2) and bytes(first.payload).decode()==full(R3)
    checks['positive_partner_return_acquires_reference_condition']=acquired and COND_REINSTATED in conds[-1]
    checks['heldout_second_turn_uses_compact_reference']=isinstance(second,ActionV2) and bytes(second.payload).decode()==compact(C3)
    checks['visible_followup_is_shorter_than_first_turn']=isinstance(first,ActionV2) and isinstance(second,ActionV2) and len(second.payload)<len(first.payload)
    # A matched fresh branch without settlement cannot acquire common ground.
    no_return=ReferenceOrganismV2.restore(copy.deepcopy(cp));nf,ns,not_acquired,_=heldout(no_return,False,60000)
    checks['withheld_first_turn_return_cannot_author_reference']=isinstance(nf,ActionV2) and ns is None and not_acquired and not no_return._shared_reinstated(P,R3[-1])
    # Negative independent consequence also cannot create shared reference.
    negative=ReferenceOrganismV2.restore(copy.deepcopy(cp));partner(negative);scene(negative,R3,61000);na=negative.tick();negative.contact(CONTACT_CONSEQUENCE,(na.ticket,-1),P,True,True)
    checks['negative_return_does_not_create_shared_reference']=isinstance(na,ActionV2) and not negative._shared_reinstated(P,R3[-1])
    # Checkpoint at the natural turn boundary: first answer has settled and common ground exists, follow-up has not yet been emitted.
    mid=ReferenceOrganismV2.restore(copy.deepcopy(cp));partner(mid);scene(mid,R3,62000);ma=mid.tick();mid.contact(CONTACT_CONSEQUENCE,(ma.ticket,1),P,True,True);mid_cp=copy.deepcopy(mid.checkpoint())
    restored=ReferenceOrganismV2.restore(mid_cp);partner(restored);scene(restored,C3,62001);ra=restored.tick()
    checks['checkpoint_reenters_compact_followup_without_transcript']=isinstance(ra,ActionV2) and bytes(ra.payload).decode()==compact(C3) and all(k not in mid_cp for k in ('transcript','conversation_buffer','context_window'))
    checks['shared_episode_is_load_bearing']=any(row.partner==P for row in o.shared_episode_relations)
    src=inspect.getsource(teach_followup)+inspect.getsource(heldout)
    checks['host_never_supplies_reference_condition_or_topic_id']=all(token not in src for token in ('CONTACT_CONDITION','context_id=','topic_id=','pronoun='))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_RESIDENT_FOLLOWUP_REFERENCE_GREEN','checks':checks,'failed':failed,'conversation':{'first':'' if first is None else bytes(first.payload).decode(),'followup':'' if second is None else bytes(second.payload).decode()},'elapsed_ms':round((time.perf_counter()-started)*1000,3),'remaining_red':['OPEN_QUESTION_FOLLOWUP_BINDING','ELLIPSIS_BEYOND_PRONOUN_REFERENCE','TOPIC_SHIFT_AND_RETURN','OPEN_ENDED_CONVERSATIONAL_GENERATION','HUMAN_LANGUAGE_MASTERY']}
    if failed:print('FOUNDRY_RESIDENT_FOLLOWUP_REFERENCE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
