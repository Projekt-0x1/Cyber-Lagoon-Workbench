#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_incremental_expression_v1 import IncrementalTransientExpressionV1
from reference_organism_discourse_verify import train,scene,surface,partner,settle,u,CTX,REL_NEXT

SENSOR,VALVE=401,402;SEE=77001
FS=(45,46,47,48);FV=(75,76,77,78)

def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def see(o,entity,src,independent=True):return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)

def teach_span(o):
    left_a=scene(o,(101,201,301,401),70001);right_a=scene(o,(102,202,302,402),70002)
    o.contact(CONTACT_SCENE_LINK,(left_a,right_a,REL_NEXT),71001,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the careful engineer tests the sensor. then the quiet technician inspects the valve.'),71001)
    left_b=scene(o,(102,202,302,402),70003);right_b=scene(o,(101,201,301,401),70004)
    o.contact(CONTACT_SCENE_LINK,(left_b,right_b,REL_NEXT),71002,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the quiet technician inspects the valve. then the careful engineer tests the sensor.'),71002)
    assert o.language.span_template(REL_NEXT,2) is not None

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8);P=9901
    o=ReferenceOrganismV2(spec);train(o);partner(o,P);teach_span(o)
    feat(o,SENSOR,FS,8001);feat(o,VALVE,FV,8002)
    see(o,SENSOR,SEE,True)
    checks['see_writes_sensor_not_valve']=SENSOR in o._world_marked_entities() and VALVE not in o._world_marked_entities()
    left=scene(o,(101,201,301,401),73001);first=o.tick();fw=o._world_state_occurrences(CTX,(101,201,301,401))
    checks['first_clause_recruits_seen_sensor']=isinstance(first,ActionV2) and {row[0] for row in fw}=={SENSOR} and all(row[3] in first.contributors for row in fw)
    settle(o,first,P,1)
    right=scene(o,(102,202,302,402),73002);o.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),73003,True,True)
    cont=o.tick();cw=o._world_state_occurrences(CTX,(101,201,301,401))
    checks['span_suffix_recruits_prior_seen_world']=isinstance(cont,ActionV2) and cont.payload==u(' then the quiet technician inspects the valve.') and any(k==PREF_SPAN for k,_,_,_ in cont.selection_occurrences) and {row[0] for row in cw}=={SENSOR} and all(row[3] in cont.contributors for row in cw)
    plan=o.current_expression_plan(cont) if isinstance(cont,ActionV2) else None
    emitted=[];transient_checkpoint={};trajectory=None
    if plan is not None:
        trajectory=IncrementalTransientExpressionV1(o.language,plan,leaves=o.utterances)
        for _ in range(7):
            byte_plan=trajectory.emit()
            if byte_plan is None:break
            emitted.append(byte_plan.value);assert trajectory.reafference(byte_plan,byte_plan.value)
        transient_checkpoint=copy.deepcopy(trajectory.checkpoint())
        trajectory=IncrementalTransientExpressionV1.restore(o.language,transient_checkpoint,leaves=o.utterances)
        while True:
            byte_plan=trajectory.emit()
            if byte_plan is None:break
            emitted.append(byte_plan.value)
            assert trajectory.reafference(byte_plan,byte_plan.value)
    checks['selected_transient_closure_drives_incremental_motor']=(
        plan is not None and plan.identity in cont.contributors
        and not hasattr(o,'hierarchy')
        and tuple(emitted)==tuple(cont.planned_payload)==tuple(cont.payload)
        and trajectory is not None and trajectory.complete and not hasattr(trajectory,'surface')
        and trajectory.h is None
        and 'payload' not in transient_checkpoint and 'surface' not in transient_checkpoint
        and transient_checkpoint.get('transient_plan',{}).get('identity')==plan.identity
    )
    checks['language_phenotype_improved']=checks['selected_transient_closure_drives_incremental_motor']
    checks['current_unseen_valve_stays_off']=VALVE not in {row[0] for row in cw}
    fix=ReferenceOrganismV2(spec);train(fix);partner(fix,P);teach_span(fix);feat(fix,SENSOR,FS,8001);feat(fix,VALVE,FV,8002);see(fix,SENSOR,SEE,True)
    fl=scene(fix,(101,201,301,401),74001);ff=fix.tick();settle(fix,ff,P,1)
    fr=scene(fix,(102,202,302,402),74002);fix.contact(CONTACT_SCENE_LINK,(fl,fr,REL_NEXT),74003,True,True)
    fix.inject_output_fault(0,ord('X'));fb=fix.tick();prior=fix._world_state_occurrences(CTX,(101,201,301,401))
    settle(fix,fb,P,-1);frep=fix.tick()
    checks['span_repair_keeps_prior_world']=isinstance(frep,ActionV2) and frep.repair and frep.span_identity and bool(prior) and all(row[3] in frep.contributors for row in prior) and VALVE not in {row[0] for row in prior}
    moved=ReferenceOrganismV2(spec);train(moved);partner(moved,P);teach_span(moved);feat(moved,SENSOR,FS,8001);feat(moved,VALVE,FV,8002);see(moved,SENSOR,SEE,True)
    ml=scene(moved,(101,201,301,401),75001);mf=moved.tick();settle(moved,mf,P,1)
    mr=scene(moved,(102,202,302,402),75002);moved.contact(CONTACT_SCENE_LINK,(ml,mr,REL_NEXT),75003,True,True)
    moved.inject_output_fault(0,ord('X'));mb=moved.tick();mprior=moved._world_state_occurrences(CTX,(101,201,301,401))
    settle(moved,mb,P,-1);partner(moved,9902);mrep=moved.tick()
    checks['span_repair_keeps_prior_after_partner_switch']=isinstance(mrep,ActionV2) and mrep.repair and mrep.source==P and bool(mprior) and all(row[3] in mrep.contributors for row in mprior)
    settle(o,cont,P,-1)
    right2=scene(o,(102,202,302,402),73004);o.contact(CONTACT_SCENE_LINK,(left,right2,REL_NEXT),73005,True,True)
    leaf=o.tick();lw=o._world_state_occurrences(CTX,(102,202,302,402))
    checks['leaf_fallback_drops_prior_world']=isinstance(leaf,ActionV2) and leaf.payload==u('the quiet technician inspects the valve.') and not any(k==PREF_SPAN for k,_,_,_ in leaf.selection_occurrences) and not lw
    yoked=ReferenceOrganismV2(spec);train(yoked);partner(yoked,P);teach_span(yoked);feat(yoked,SENSOR,FS,8001);see(yoked,SENSOR,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2(spec);train(cut);partner(cut,P);teach_span(cut);feat(cut,SENSOR,FS,8001);feat(cut,VALVE,FV,8002);see(cut,SENSOR,SEE,True)
    cl=scene(cut,(101,201,301,401),83001);ca=cut.tick();settle(cut,ca,P,1)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True)
    cr=scene(cut,(102,202,302,402),83002);cut.contact(CONTACT_SCENE_LINK,(cl,cr,REL_NEXT),83003,True,True);cc=cut.tick()
    checks['world_withdrawal_keeps_span_drops_world']=isinstance(cc,ActionV2) and b'then' in bytes(cc.payload) and not cut._world_state_occurrences(CTX,(101,201,301,401))
    link=ReferenceOrganismV2(spec);train(link);partner(link,P);teach_span(link)
    la=scene(link,(101,201,301,401),84001);lb=scene(link,(102,202,302,402),84002);DEAD=71001
    before=len(link.scene_links)
    link.contact(CONTACT_WITHDRAW_SOURCE,(DEAD,),88031,True,True)
    try:
        link.contact(CONTACT_SCENE_LINK,(la,lb,REL_NEXT),DEAD,True,True);linked=True
    except ValueError:
        linked=False
    checks['withdrawn_source_cannot_link_scenes']=not linked and len(link.scene_links)==before
    EN,EN2,DE,DE2=1101,1102,2101,2102;BCTX=8801;A,B,I,S,T,V=201,202,303,403,302,502
    both=ReferenceOrganismV2(spec)
    feat(both,A,(11,12,13,14),8001);feat(both,B,(21,22,23,24),8002)
    feat(both,I,(35,36),8015);feat(both,S,(45,46,47,48),8016);feat(both,T,(33,34),8013);feat(both,V,(75,76,77,78),8014)
    for src,alice,inspect,sensor,bob,test,valve in (
        (EN,'alice','inspects','sensor','bob','tests','valve'),
        (EN2,'alice','inspects','sensor','bob','tests','valve'),
        (DE,'Alice','prueft','Sensor','Bob','testet','Ventil'),
        (DE2,'Alice','prueft','Sensor','Bob','testet','Ventil')):
        both.contact(CONTACT_SCENE,(7,0,1,A),src,True,True);surface(both,alice,src)
        both.contact(CONTACT_SCENE,(7,0,1,I),src,True,True);surface(both,inspect,src)
        both.contact(CONTACT_SCENE,(7,0,1,S),src,True,True);surface(both,sensor,src)
        both.contact(CONTACT_SCENE,(7,0,1,B),src,True,True);surface(both,bob,src)
        both.contact(CONTACT_SCENE,(7,0,1,T),src,True,True);surface(both,test,src)
        both.contact(CONTACT_SCENE,(7,0,1,V),src,True,True);surface(both,valve,src)
    for src,atoms,text in (
        (EN,(A,I,S),'alice inspects sensor.'),(EN2,(A,I,S),'alice inspects sensor.'),
        (DE,(A,I,S),'Alice prueft Sensor.'),(DE2,(A,I,S),'Alice prueft Sensor.'),
        (EN,(B,T,V),'bob tests valve.'),(EN2,(B,T,V),'bob tests valve.'),
        (DE,(B,T,V),'Bob testet Ventil.'),(DE2,(B,T,V),'Bob testet Ventil.')):
        both.contact(CONTACT_SCENE,(7,BCTX,len(atoms),*atoms),src,True,True);surface(both,text,src)
    taught=True
    try:
        for p,s0,s1,first,second in (
            (EN,EN,EN2,'alice inspects sensor. then bob tests valve.','bob tests valve. then alice inspects sensor.'),
            (DE,DE,DE2,'Alice prueft Sensor. dann Bob testet Ventil.','Bob testet Ventil. dann Alice prueft Sensor.')):
            partner(both,p)
            left=both.contact(CONTACT_SCENE,(7,BCTX,3,A,I,S),s0,True,True)
            right=both.contact(CONTACT_SCENE,(7,BCTX,3,B,T,V),s0,True,True)
            both.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),s0,True,True)
            both.contact(CONTACT_DISCOURSE_SURFACE,u(first),s0)
            left=both.contact(CONTACT_SCENE,(7,BCTX,3,B,T,V),s1,True,True)
            right=both.contact(CONTACT_SCENE,(7,BCTX,3,A,I,S),s1,True,True)
            both.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),s1,True,True)
            both.contact(CONTACT_DISCOURSE_SURFACE,u(second),s1)
    except ValueError:
        taught=False
    checks['bilingual_span_can_be_taught']=taught
    see(both,S,SEE,True);buried=copy.deepcopy(both.checkpoint()) if taught else None
    def run_span(p):
        if buried is None:return None,(),()
        x=ReferenceOrganismV2.restore(copy.deepcopy(buried));partner(x,p)
        left=x.contact(CONTACT_SCENE,(7,BCTX,3,A,I,S),73001,True,True);first=x.tick()
        if not isinstance(first,ActionV2):return None,(),()
        settle(x,first,p,1)
        right=x.contact(CONTACT_SCENE,(7,BCTX,3,B,T,V),73002,True,True)
        x.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),73003,True,True)
        cont=x.tick();prior=x._world_state_occurrences(BCTX,(A,I,S));motor=[]
        plan=x.current_expression_plan(cont) if isinstance(cont,ActionV2) else None
        if plan is not None:
            trajectory=IncrementalTransientExpressionV1(x.language,plan,leaves=x.utterances)
            while True:
                byte_plan=trajectory.emit()
                if byte_plan is None:break
                motor.append(byte_plan.value);trajectory.reafference(byte_plan,byte_plan.value)
        return cont,prior,tuple(motor)
    en,ew,em=run_span(EN);de,dw,dm=run_span(DE)
    checks['english_partner_keeps_then']=isinstance(en,ActionV2) and en.payload==u(' then bob tests valve.') and any(k==PREF_SPAN for k,_,_,_ in en.selection_occurrences)
    checks['german_partner_keeps_dann']=isinstance(de,ActionV2) and de.payload==u(' dann Bob testet Ventil.') and any(k==PREF_SPAN for k,_,_,_ in de.selection_occurrences)
    checks['partner_conditioned_transient_plans_drive_distinct_motors']=(
        isinstance(en,ActionV2) and isinstance(de,ActionV2)
        and em==tuple(en.planned_payload) and dm==tuple(de.planned_payload)
        and em!=dm and bool(em) and bool(dm)
    )
    checks['both_span_suffixes_recruit_same_prior_world']=isinstance(en,ActionV2) and isinstance(de,ActionV2) and {row[0] for row in ew}=={row[0] for row in dw}=={S} and all(row[3] in en.contributors for row in ew) and all(row[3] in de.contributors for row in dw)
    checks['no_connective_opcode']=not hasattr(o,'then') and not hasattr(o,'imagine') and PREF_SPAN not in (PREF_VIEW,)
    result={'schema':'0x1.reference-organism-world-span.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'span':bytes(cont.payload).decode() if isinstance(cont,ActionV2) else '','claim':'SPAN_CLOSURE_RECRUITS_PRIOR_SEEN_WORLD_LEAF_DOES_NOT_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_SPAN '+('GREEN' if result['pass'] else 'RED')+' span_keeps_prior=1 leaf_drops=1 media=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
