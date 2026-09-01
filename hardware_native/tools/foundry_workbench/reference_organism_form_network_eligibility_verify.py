#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_social_language_verify import train_base,scene,surface,partner,settle,establish,teach_conditioned_form,u,CTX

def compact_cfg(o,action):
    return o._selection_configuration(action.template_identity,action.lexical_identities,action.form_slots,action.span_identity)

def primed(spec,p_sensor=9001,p_valve=9002):
    o=ReferenceOrganismV2(spec);sensor_prior,valve_prior=train_base(o)
    establish(o,p_sensor,sensor_prior,40001);teach_conditioned_form(o,p_sensor,401,41001)
    establish(o,p_valve,valve_prior,40002);teach_conditioned_form(o,p_valve,402,42001)
    partner(o,p_sensor);scene(o,CTX,(102,202,302,401),43001);surface(o,'the quiet technician inspects it.',43101)
    partner(o,p_valve);scene(o,CTX,(101,201,301,402),43002);surface(o,'the careful engineer tests it.',43102)
    return o

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8)
    P3,P4=9010,9011;novel=(102,201,302,401)
    o=primed(spec);establish(o,P3,(101,202,301,401),44001)
    partner(o,P3);scene(o,CTX,novel,44002);first=o.tick()
    checks['untried_compact_still_developmental']=isinstance(first,ActionV2) and first.payload==u('the quiet engineer inspects it.') and any(k==PREF_FORM for k,_,_,_ in first.selection_occurrences)
    settle(o,first,P3,-1,True)
    value,evidence=o._selection_configuration_evidence(first.selection_context,compact_cfg(o,first))
    checks['independent_negative_credits_form_network']=value<0 and evidence>0
    checks['developmental_form_still_present']=o.language.form(401,(COND_REINSTATED,),require_conditioned=True)==u('it')
    punished=copy.deepcopy(o.checkpoint())
    partner(o,P3);scene(o,CTX,novel,44003);_,cond=o._surface_context(o.current_scene)
    checks['common_ground_survives_negative_form_return']=cond==((),(),(),(COND_REINSTATED,))
    reopened=o.tick()
    checks['negative_form_network_falls_back_explicit']=isinstance(reopened,ActionV2) and reopened.payload==u('the quiet engineer inspects the sensor.') and not any(k==PREF_FORM for k,_,_,_ in (reopened.selection_occurrences if reopened else ()))
    yoked=primed(spec);establish(yoked,P4,(101,202,301,401),54001)
    partner(yoked,P4);scene(yoked,CTX,novel,54002);ya=yoked.tick();settle(yoked,ya,P4,1,False)
    partner(yoked,P4);scene(yoked,CTX,novel,54003);yb=yoked.tick()
    checks['yoked_return_cannot_punish_form_network']=isinstance(yb,ActionV2) and yb.payload==u('the quiet engineer inspects it.')
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(punished));withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(P3,),88001,True,True)
    partner(withdrawn,P3);scene(withdrawn,CTX,novel,44004);restored=withdrawn.tick()
    checks['withdrawal_cascades_form_credit_and_common_ground']=isinstance(restored,ActionV2) and restored.payload==u('the quiet engineer inspects the sensor.') and not restored.form_slots
    r=ReferenceOrganismV2.restore(copy.deepcopy(punished));partner(r,P3);scene(r,CTX,novel,44005);ra=r.tick()
    checks['checkpoint_keeps_form_network_veto']=isinstance(ra,ActionV2) and ra.payload==u('the quiet engineer inspects the sensor.')
    checks['no_pronoun_opcode']=not hasattr(o,'pronoun') and not hasattr(o,'fallback_explicit')
    result={'schema':'0x1.reference-organism-form-network-eligibility.v1','pass':all(checks.values()),'checks':checks,'reopened':bytes(reopened.payload).decode() if reopened else '','runtime_llm':False,'graph_flip':False,'claim':'CONDITIONED_FORM_NOMINATION_USES_SELECTION_NETWORK_CREDIT_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_FORM_NETWORK_ELIGIBILITY '+('GREEN' if result['pass'] else 'RED')+' network_veto=1 developmental_form_kept=1 yoked=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
