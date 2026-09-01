#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

S=(101,102);G=(201,202);ALT=(301,302);A=7001;B=7002;T1=8001;T2=8002

def setup(o,world_source):
    o.contact(CONTACT_WORLD_STATE,S,world_source,True,True)
    o.contact(CONTACT_BODY_TARGET,G,9001,True,True)
    o.contact(CONTACT_AFFORDANCES,(A,B),9002,True,True)

def lived_relation(o,world_source,testimony_source,chosen):
    setup(o,world_source)
    base=o._exploration_candidate(); assert chosen!=base
    claim=o.contact(CONTACT_SOURCE_ASSERTION,(chosen,),testimony_source,True,True)
    action=o.tick(); assert isinstance(action,MotorActionV2) and action.action_id==chosen and claim in action.source_assertion_ids and action.source_counterfactual_action!=chosen
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(action.ticket,1,len(G),*G),world_source,True,True)
    return claim,action,learned

def observer_relation(ecology,state,action):
    edge=ecology.transition(state,action)
    if edge is None:return None
    logical=hashlib.sha256(b'transition-logical-v2\0'+json.dumps([list(edge.state),edge.action],separators=(',',':')).encode()).hexdigest()
    revision=hashlib.sha256(b'transition-revision-v2\0'+json.dumps({
        'state':list(edge.state),'action':edge.action,'next':list(edge.next_state),
        'effect':edge.effect,'support':edge.support,'sources':list(edge.sources)},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return edge,logical,revision

def focal_lesion(ecology,state,action):
    state=tuple(sorted(state)); action=int(action); removed=0
    for key in list(ecology._evidence):
        if key[0]==state and key[1]==action:
            del ecology._evidence[key];removed+=1
    return removed

def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(65536,2,3,42,8))
    setup(o,9101);base=o._exploration_candidate();chosen=B if base==A else A
    c1,a1,l1=lived_relation(o,9101,T1,chosen);c2,a2,l2=lived_relation(o,9102,T2,chosen)
    receipt=observer_relation(o.cognition,S,chosen);assert receipt is not None
    edge,logical,revision=receipt
    checks['consequence_earned_relation']=edge.support==2 and set(edge.sources)=={9101,9102} and edge.next_state==G and l1['source_credit']>0 and l2['source_credit']>0

    # Withdrawal removes testimony authority, not independently earned world evidence.
    o.contact(CONTACT_WITHDRAW_SOURCE,(T1,),9201,True,True);o.contact(CONTACT_WITHDRAW_SOURCE,(T2,),9202,True,True)
    checks['raw_testimony_withdrawn']=all(not r.active for r in o.source_assertions if r.identity in (c1,c2))
    after=observer_relation(o.cognition,S,chosen)
    checks['relation_survives_source_withdrawal']=after is not None and after[1]==logical and after[2]==revision

    # Checkpoint carries mathematical evidence without reviving testimony.
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));rr=observer_relation(r.cognition,S,chosen)
    checks['checkpoint_relation_exact']=r.digest()==o.digest() and rr is not None and rr[1]==logical and rr[2]==revision and all(not x.active for x in r.source_assertions if x.identity in (c1,c2))

    # Equal-type remote relation stays while focal mathematical support is removed.
    for src in (9301,9302):r.cognition.observe((501,),9009,(502,),1,src,True)
    remote_before=observer_relation(r.cognition,(501,),9009)
    removed=focal_lesion(r.cognition,S,chosen)
    checks['focal_lesion_destroys_reconstruction']=removed>0 and observer_relation(r.cognition,S,chosen) is None
    remote_after=observer_relation(r.cognition,(501,),9009)
    checks['remote_relation_spared']=remote_before is not None and remote_after==remote_before
    checks['withdrawn_assertion_rows_not_relation_authority']=any(x.identity in (c1,c2) for x in r.source_assertions) and observer_relation(r.cognition,S,chosen) is None

    # New lived consequences re-earn the same state/action relation under new evidence.
    for ws,ts in ((9401,8101),(9402,8102)):
        setup(r,ws);claim=r.contact(CONTACT_SOURCE_ASSERTION,(chosen,),ts,True,True);a=r.tick();assert isinstance(a,MotorActionV2) and a.action_id==chosen and claim in a.source_assertion_ids and a.source_counterfactual_action!=chosen;r.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,1,len(G),*G),ws,True,True)
    relearned=observer_relation(r.cognition,S,chosen)
    checks['reread_relearns_same_logical_relation']=relearned is not None and relearned[1]==logical and relearned[2]!=revision and set(relearned[0].sources)=={9401,9402}

    old_revision=relearned[2]
    focal_lesion(r.cognition,S,chosen)
    for src in (9501,9502):r.cognition.observe(S,chosen,ALT,-1,src,True)
    contradicted=observer_relation(r.cognition,S,chosen)
    checks['contradiction_new_revision']=contradicted is not None and contradicted[1]==logical and contradicted[2]!=old_revision and contradicted[0].next_state==ALT and edge.next_state==G
    checks['no_fact_module_or_llm']=all(not hasattr(r,n) for n in ('facts','knowledge','truth','answer','prompt','complete'))
    out={'schema':'0x1.reference-source-withdrawn-relation.v2','pass':all(checks.values()),'checks':checks,'logical_identity':logical,'initial_revision':revision,'relearned_revision':relearned[2] if relearned else '', 'contradicted_revision':contradicted[2] if contradicted else '', 'claim':'SOURCE_WITHDRAWN_CONSEQUENCE_EARNED_RELATION_REFERENCE_ONLY_NOT_DIRECT_KG1_GREEN','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_SOURCE_WITHDRAWN_RELATION '+('GREEN' if out['pass'] else 'RED')+' testimony_truth=0 consequence_relation=1 lesion=1 reread=1 revision=1')
    print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
