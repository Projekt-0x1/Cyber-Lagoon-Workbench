#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;START=7;ACT=101;BODY=88001
ALICE,INSPECT,SENSOR=201,303,403
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)

def train(o):
    feat(o,ALICE,FA,8001);feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,(45,46),8016)
    for e,s in ((ALICE,'alice'),(INSPECT,'inspects'),(SENSOR,'sensor')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    assert o.language.template(CTX,3) is not None

def idle(o,src=60001):
    o.contact(CONTACT_WORLD_STATE,(START,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+1,True,True)

def mark(o,src=BODY,independent=True,effect=-1):
    o.contact(CONTACT_WORLD_STATE,(ALICE,),src,True,True)
    o.contact(CONTACT_BODY_STATE,(ALICE,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(ACT,),src+2,True,True)
    motor=o.tick()
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    nxt=tuple(sorted({ALICE,START}))
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,effect,len(nxt),*nxt),src,True,independent)
    idle(o,src+10)
    return learned

def fault(o,src=41001):
    partner(o);o.contact(CONTACT_SCENE,(7,CTX,3,ALICE,INSPECT,SENSOR),src,True,True)
    o.inject_output_fault(0,ord('X'));return o.tick()

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    planned=u('alice inspects the sensor.')
    learned=mark(o)
    checks['body_return_marks_alice']=learned.get('somatic_updates',0)>0 and ALICE in o._somatic_marked_entities()
    bad=fault(o)
    recruited=o._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['fault_already_carries_prior_soma']=isinstance(bad,ActionV2) and bad.payload!=planned and bad.planned_payload==planned and bool(recruited) and all(row[3] in bad.contributors for row in recruited) and all(row[3] in bad.somatic_occurrences for row in recruited)
    o.contact(CONTACT_CONSEQUENCE,(bad.ticket,-1),P,True,True)
    repair=o.tick();rs=o._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['repair_is_outer_language']=isinstance(repair,ActionV2) and repair.repair and repair.payload==planned and repair.source==P
    checks['repair_recruits_current_soma']=bool(rs) and {row[0] for row in rs}=={ALICE} and all(row[3] in repair.contributors for row in rs) and all(row[3] in repair.somatic_occurrences for row in rs)
    yoked=ReferenceOrganismV2(spec);train(yoked);mark(yoked,independent=False);yb=fault(yoked,42001)
    checks['yoked_body_cannot_write_soma']=yoked._somatic_revisions.row_count==0 and isinstance(yb,ActionV2) and not yb.somatic_occurrences
    cut=ReferenceOrganismV2(spec);train(cut);mark(cut);cb=fault(cut,43001)
    stale=tuple(cb.somatic_occurrences)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),88002,True,True)
    cut.contact(CONTACT_CONSEQUENCE,(cb.ticket,-1),P,True,True);cr=cut.tick()
    checks['body_withdrawal_keeps_repair_drops_soma']=isinstance(cr,ActionV2) and cr.repair and cr.payload==planned and not cut._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR)) and not cr.somatic_occurrences and bool(stale) and all(oid not in cr.contributors for oid in stale)
    rows=cut._somatic_revisions.row_count
    cut.contact(CONTACT_CONSEQUENCE,(cr.ticket,1),P,True,True)
    checks['withdrawn_body_cannot_rewrite_soma']=cut._somatic_revisions.row_count==rows
    swap=ReferenceOrganismV2(spec);train(swap);mark(swap);sb=fault(swap,44001);NEW=99001
    swap.contact(CONTACT_BODY_STATE,(202,),NEW,True,True)
    swap.contact(CONTACT_CONSEQUENCE,(sb.ticket,-1),P,True,True);sr=swap.tick()
    checks['repair_carries_current_body']=isinstance(sr,ActionV2) and sr.repair and sr.body_source==NEW and sr.body_occurrence==swap.body_state_occurrence and sr.body_occurrence in sr.contributors and sb.body_occurrence not in sr.contributors
    checks['no_oracle_opcode']=not hasattr(o,'correct_output') and not hasattr(o,'feel') and not hasattr(o,'imagine')
    result={'schema':'0x1.reference-organism-somatic-repair.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'fault':bytes(bad.payload).decode(errors='replace') if isinstance(bad,ActionV2) else '','repair':bytes(repair.payload).decode() if isinstance(repair,ActionV2) else '','claim':'REPAIR_RECRUITS_CURRENT_SOMA_NOT_STALE_FAULTED_COPY_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_REPAIR '+('GREEN' if result['pass'] else 'RED')+' current_soma=1 withdraw_drops=1 oracle=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
