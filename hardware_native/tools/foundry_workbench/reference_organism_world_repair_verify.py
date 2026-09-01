#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;SEE=77001
ALICE,INSPECT,SENSOR=201,303,403
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)

def train(o):
    feat(o,ALICE,FA,8001);feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,(45,46),8016)
    for e,s in ((ALICE,'alice'),(INSPECT,'inspects'),(SENSOR,'sensor')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    assert o.language.template(CTX,3) is not None

def fault(o,src=41001):
    partner(o);o.contact(CONTACT_SCENE,(7,CTX,3,ALICE,INSPECT,SENSOR),src,True,True)
    o.inject_output_fault(0,ord('X'));return o.tick()

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    planned=u('alice inspects the sensor.')
    bad=fault(o)
    checks['fault_before_see_has_no_world']=isinstance(bad,ActionV2) and bad.payload!=planned and bad.planned_payload==planned and o._world_revisions.row_count==0
    see(o,ALICE,SEE,True)
    checks['see_after_fault_writes_world']=ALICE in o._world_marked_entities()
    o.contact(CONTACT_CONSEQUENCE,(bad.ticket,-1),P,True,True)
    repair=o.tick();rw=o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['repair_is_outer_language']=isinstance(repair,ActionV2) and repair.repair and repair.payload==planned and repair.source==P
    checks['repair_recruits_later_seen_world']=bool(rw) and {row[0] for row in rw}=={ALICE} and all(row[3] in repair.contributors for row in rw)
    checks['faulted_surface_did_not_already_carry_world']=all(row[3] not in bad.contributors for row in rw)
    yoked=ReferenceOrganismV2(spec);train(yoked);yb=fault(yoked,42001);see(yoked,ALICE,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2(spec);train(cut);cb=fault(cut,43001);see(cut,ALICE,SEE,True)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True);cut.contact(CONTACT_CONSEQUENCE,(cb.ticket,-1),P,True,True);cr=cut.tick()
    checks['world_withdrawal_keeps_repair_drops_world']=isinstance(cr,ActionV2) and cr.repair and cr.payload==planned and not cut._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    prior=ReferenceOrganismV2(spec);train(prior);see(prior,ALICE,SEE,True);pb=fault(prior,44001)
    prior_world=prior._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['see_before_fault_joins_faulted_surface']=bool(prior_world) and all(row[3] in pb.contributors for row in prior_world)
    prior.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88003,True,True);prior.contact(CONTACT_CONSEQUENCE,(pb.ticket,-1),P,True,True);pr=prior.tick()
    checks['withdrawn_see_drops_stale_world_from_repair']=isinstance(pr,ActionV2) and pr.repair and pr.payload==planned and not prior._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR)) and all(row[3] not in pr.contributors for row in prior_world)
    checks['no_oracle_opcode']=not hasattr(o,'correct_output') and not hasattr(o,'imagine') and not hasattr(o,'translate')
    result={'schema':'0x1.reference-organism-world-repair.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'fault':bytes(bad.payload).decode(errors='replace') if isinstance(bad,ActionV2) else '','repair':bytes(repair.payload).decode() if isinstance(repair,ActionV2) else '','claim':'REPAIR_SURFACE_RECRUITS_CURRENT_SEEN_WORLD_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_REPAIR '+('GREEN' if result['pass'] else 'RED')+' later_see_joins_repair=1 oracle=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
