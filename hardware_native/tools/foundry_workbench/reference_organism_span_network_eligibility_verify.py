#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_discourse_verify import train,scene,surface,partner,settle,u,CTX,REL_NEXT
from reference_organism_social_language_verify import train_base,establish,teach_conditioned_form

def cfg(o,action):
    return o._selection_configuration(action.template_identity,action.lexical_identities,action.form_slots,action.span_identity)

def teach_span(o,rel=REL_NEXT):
    left_a=scene(o,(101,201,301,401),70001);right_a=scene(o,(102,202,302,402),70002)
    o.contact(CONTACT_SCENE_LINK,(left_a,right_a,rel),71001,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the careful engineer tests the sensor. then the quiet technician inspects the valve.'),71001)
    left_b=scene(o,(102,202,302,402),70003);right_b=scene(o,(101,201,301,401),70004)
    o.contact(CONTACT_SCENE_LINK,(left_b,right_b,rel),71002,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the quiet technician inspects the valve. then the careful engineer tests the sensor.'),71002)
    assert o.language.span_template(rel,2) is not None

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8);P=9901
    o=ReferenceOrganismV2(spec);train(o);partner(o,P);teach_span(o)
    left=scene(o,(101,201,301,401),73001);first=o.tick();assert first is not None and first.scene_identity==left;settle(o,first,P,1)
    right=scene(o,(102,202,302,402),73002);o.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),73003,True,True)
    cont=o.tick()
    checks['untried_span_still_developmental']=isinstance(cont,ActionV2) and cont.payload==u(' then the quiet technician inspects the valve.') and any(k==PREF_SPAN for k,_,_,_ in cont.selection_occurrences)
    settle(o,cont,P,-1)
    value,evidence=o._selection_configuration_evidence(cont.selection_context,cfg(o,cont))
    checks['independent_negative_credits_span_network']=value<0 and evidence>0
    checks['span_template_still_present']=o.language.span_template(REL_NEXT,2) is not None
    right2=scene(o,(102,202,302,402),73004);o.contact(CONTACT_SCENE_LINK,(left,right2,REL_NEXT),73005,True,True)
    punished=copy.deepcopy(o.checkpoint())
    reopened=o.tick()
    checks['negative_span_network_falls_back_leaf']=isinstance(reopened,ActionV2) and reopened.payload==u('the quiet technician inspects the valve.') and not any(k==PREF_SPAN for k,_,_,_ in (reopened.selection_occurrences if reopened else ()))
    yoked=ReferenceOrganismV2(spec);train(yoked);partner(yoked,P);teach_span(yoked)
    yl=scene(yoked,(101,201,301,401),83001);ya=yoked.tick();settle(yoked,ya,P,1)
    yr=scene(yoked,(102,202,302,402),83002);yoked.contact(CONTACT_SCENE_LINK,(yl,yr,REL_NEXT),83003,True,True)
    yb=yoked.tick();yoked.contact(CONTACT_CONSEQUENCE,(yb.ticket,1),P,True,False)
    yr2=scene(yoked,(102,202,302,402),83004);yoked.contact(CONTACT_SCENE_LINK,(yl,yr2,REL_NEXT),83005,True,True)
    yc=yoked.tick()
    checks['yoked_return_cannot_punish_span_network']=isinstance(yc,ActionV2) and yc.payload==u(' then the quiet technician inspects the valve.')
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(punished));withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(P,),88001,True,True)
    partner(withdrawn,P);restored=withdrawn.tick()
    checks['withdrawal_cascades_span_credit_and_discourse_source']=isinstance(restored,ActionV2) and restored.payload==u('the quiet technician inspects the valve.') and not restored.span_identity
    r=ReferenceOrganismV2.restore(copy.deepcopy(punished));ra=r.tick()
    checks['checkpoint_keeps_span_network_veto']=isinstance(ra,ActionV2) and ra.payload==u('the quiet technician inspects the valve.')

    compact=ReferenceOrganismV2(spec);sensor_prior,valve_prior=train_base(compact)
    establish(compact,9001,sensor_prior,40001);teach_conditioned_form(compact,9001,401,41001)
    establish(compact,9002,valve_prior,40002);teach_conditioned_form(compact,9002,402,42001)
    partner(compact,9001);scene(compact,(102,202,302,401),43001);surface(compact,'the quiet technician inspects it.',43101)
    partner(compact,9002);scene(compact,(101,201,301,402),43002);surface(compact,'the careful engineer tests it.',43102)
    partner(compact,P);teach_span(compact)
    establish(compact,P,(101,202,301,401),44001)
    cl=scene(compact,(101,202,301,401),44010);ca=compact.tick();settle(compact,ca,P,1)
    cr=scene(compact,(102,201,302,401),44011);compact.contact(CONTACT_SCENE_LINK,(cl,cr,REL_NEXT),44013,True,True)
    cb=compact.tick()
    checks['compact_span_emits_suffix']=isinstance(cb,ActionV2) and cb.payload==u(' then the quiet engineer inspects it.') and any(k==PREF_FORM for k,_,_,_ in cb.selection_occurrences) and any(k==PREF_SPAN for k,_,_,_ in cb.selection_occurrences)
    settle(compact,cb,P,-1)
    cr2=scene(compact,(102,201,302,401),44012);compact.contact(CONTACT_SCENE_LINK,(cl,cr2,REL_NEXT),44014,True,True)
    cc=compact.tick()
    checks['punished_compact_span_keeps_form_drops_span']=isinstance(cc,ActionV2) and cc.payload==u('the quiet engineer inspects it.') and any(k==PREF_FORM for k,_,_,_ in (cc.selection_occurrences if cc else ())) and not any(k==PREF_SPAN for k,_,_,_ in (cc.selection_occurrences if cc else ()))
    checks['no_connective_opcode']=not hasattr(o,'then') and not hasattr(o,'discourse_word')
    result={'schema':'0x1.reference-organism-span-network-eligibility.v1','pass':all(checks.values()),'checks':checks,'reopened':bytes(reopened.payload).decode() if reopened else '','compact_reopened':bytes(cc.payload).decode() if cc else '','runtime_llm':False,'graph_flip':False,'claim':'DISCOURSE_SPAN_NOMINATION_USES_SELECTION_NETWORK_CREDIT_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SPAN_NETWORK_ELIGIBILITY '+('GREEN' if result['pass'] else 'RED')+' network_veto=1 leaf_fallback=1 compact_span=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
