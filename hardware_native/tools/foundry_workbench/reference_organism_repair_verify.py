#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=100;CTX=9001;P1=9101;P2=9102
def u(s):return tuple(s.encode())
def scene(o,c,a,src):return o.contact(CONTACT_SCENE,(7,c,len(a),*a),src,True,True)
def surface(o,s,src):return o.contact(CONTACT_SURFACE,u(s),src,True,True)
def partner(o,p):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def train(o):
    mapping={101:'careful',201:'engineer',301:'tests',401:'sensor',102:'quiet',202:'technician',302:'inspects',402:'valve'}
    for f,s in mapping.items():
        scene(o,NAME,(f,),10000+f);surface(o,s,20000+f);scene(o,NAME,(f,),11000+f);surface(o,s,21000+f)
    scene(o,CTX,(101,201,301,401),30001);surface(o,'the careful engineer tests the sensor.',31001)
    scene(o,CTX,(102,202,302,402),30002);surface(o,'the quiet technician inspects the valve.',31002)

def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));train(o);partner(o,P1)
    sid=scene(o,CTX,(102,201,301,402),40001);o.inject_output_fault(0,ord('X'));fault=o.tick();expected=u('the quiet engineer tests the valve.')
    checks['public_fault_differs_from_plan']=fault is not None and fault.payload!=expected and fault.planned_payload==expected and fault.payload[0]==ord('X') and not fault.repair
    checks['fault_does_not_create_common_ground_before_return']=P1 not in o.last_shared_episode_by_partner
    checks['pending_fault_blocks_repair']=o.tick() is None
    fault_learned=o.contact(CONTACT_CONSEQUENCE,(fault.ticket,-1),P1,True,True)
    checks['negative_fault_return_not_common_ground']=P1 not in o.last_shared_episode_by_partner
    checks['faulted_surface_does_not_credit_network']=fault_learned.get('selection_network_updates',0)==0
    # Current social context may move; corrective action remains bound to the original addressee.
    partner(o,P2);repair=o.tick()
    checks['endogenous_repair_to_frozen_partner']=repair is not None and repair.repair and repair.payload==expected and repair.planned_payload==expected and repair.source==P1
    checks['host_did_not_supply_corrected_surface']=o.pending_repair is None and not hasattr(o,'correct_output') and not hasattr(o,'answer')
    # Freeze a checkpoint with the repair pending and verify exact regeneration.
    r0=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));train(r0);partner(r0,P1);scene(r0,CTX,(102,201,301,402),41001);r0.inject_output_fault(0,ord('Y'));bad=r0.tick();r0.contact(CONTACT_CONSEQUENCE,(bad.ticket,-1),P1,True,True);cp=r0.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));rr=r.tick()
    checks['checkpoint_preserves_pending_repair']=rr is not None and rr.repair and rr.payload==bad.planned_payload and rr.source==P1
    checks['repair_carries_planned_selection_network']=isinstance(repair,ActionV2) and any(k==PREF_TEMPLATE for k,_,_,_ in repair.selection_occurrences) and repair.lexical_identities==fault.lexical_identities
    repair_learned=o.contact(CONTACT_CONSEQUENCE,(repair.ticket,1),P1,True,True)
    checks['successful_repair_credits_planned_network']=repair_learned.get('selection_network_updates',0)>=1
    checks['successful_repair_establishes_shared_history']=P1 in o.last_shared_episode_by_partner and o.last_shared_episode_by_partner[P1]==next(e.identity for e in o.episodes if e.scene_identity==sid)
    checks['terminal_no_repeat_repair']=o.tick() is None
    yoked=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));train(yoked);partner(yoked,P1)
    scene(yoked,CTX,(102,201,301,402),42001);yoked.inject_output_fault(0,ord('Z'));yf=yoked.tick();yoked.contact(CONTACT_CONSEQUENCE,(yf.ticket,-1),P1,True,True)
    yr=yoked.tick();yoked_learned=yoked.contact(CONTACT_CONSEQUENCE,(yr.ticket,1),P1,True,False)
    checks['yoked_repair_cannot_credit_network']=yoked_learned.get('selection_network_updates',0)==0
    dead=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));train(dead);partner(dead,P1)
    scene(dead,CTX,(102,201,301,402),43001);dead.inject_output_fault(0,ord('X'));df=dead.tick()
    dead.contact(CONTACT_CONSEQUENCE,(df.ticket,-1),P1,True,True)
    dead.contact(CONTACT_WITHDRAW_SOURCE,(P1,),88001,True,True)
    checks['withdrawn_addressee_drops_pending_repair']=dead.pending_repair is None and dead.tick() is None
    result={'schema':'0x1.reference-organism-repair.v1','pass':all(checks.values()),'checks':checks,'fault_surface':bytes(fault.payload).decode(errors='replace'),'planned_surface':bytes(expected).decode(),'repair_surface':bytes(repair.payload).decode() if repair else '', 'claim':'EAFFERENCE_REAFFERENCE_REPAIR_REFERENCE_NOT_LANGUAGE_ORACLE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_REPAIR '+('GREEN' if result['pass'] else 'RED')+' public_fault=1 endogenous_repair=1 host_corrected_surface=0 frozen_addressee=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
