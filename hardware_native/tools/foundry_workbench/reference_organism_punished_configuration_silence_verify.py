#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_repair_verify import train,scene,surface,partner,u,CTX

P=9101;ATOMS=(102,201,301,402)

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o);partner(o,P)
    scene(o,CTX,ATOMS,40001);first=o.tick()
    checks['untried_single_configuration_emits']=isinstance(first,ActionV2) and first.payload==u('the quiet engineer tests the valve.')
    learned=o.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),P,True,True)
    checks['independent_negative_credits_network']=learned.get('selection_network_updates',0)>=1
    punished=copy.deepcopy(o.checkpoint())
    scene(o,CTX,ATOMS,40002);reopened=o.tick()
    checks['sole_punished_configuration_stays_silent']=reopened is None
    checks['developmental_template_still_present']=o.language.template(CTX,4) is not None
    yoked=ReferenceOrganismV2(spec);train(yoked);partner(yoked,P)
    scene(yoked,CTX,ATOMS,41001);ya=yoked.tick();yoked.contact(CONTACT_CONSEQUENCE,(ya.ticket,1),P,True,False)
    scene(yoked,CTX,ATOMS,41002);yb=yoked.tick()
    checks['yoked_return_cannot_silence']=isinstance(yb,ActionV2) and yb.payload==u('the quiet engineer tests the valve.')
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(punished));withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(P,),88001,True,True)
    partner(withdrawn,P);scene(withdrawn,CTX,ATOMS,40003)
    checks['withdrawal_restores_sole_configuration']=isinstance(withdrawn.tick(),ActionV2)
    r=ReferenceOrganismV2.restore(copy.deepcopy(punished));partner(r,P);scene(r,CTX,ATOMS,40004)
    checks['checkpoint_keeps_silence']=r.tick() is None
    checks['no_retry_opcode']=not hasattr(o,'retry') and not hasattr(o,'fallback_surface')
    result={'schema':'0x1.reference-organism-punished-configuration-silence.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'claim':'SOLE_NET_NONPOSITIVE_CONFIGURATION_REFUSES_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_PUNISHED_CONFIGURATION_SILENCE '+('GREEN' if result['pass'] else 'RED')+' sole_veto=1 yoked=1 withdrawal=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
