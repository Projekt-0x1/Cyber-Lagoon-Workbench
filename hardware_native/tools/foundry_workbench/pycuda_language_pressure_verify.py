#!/usr/bin/env python3
"""16K lived lexical bindings on the V2 organism over an 80B procedural GPU namespace."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pycuda_population_exec import GpuPopulationSpecV1,PyCudaPopulationExecV1
from reference_population_v1 import PopulationSpecV1
from reference_organism_v2 import *

ALIAS_COUNT=16_384;NAME=7001;REL_ALIAS=7002;PARTNER=4200000

def alias_surface(i):return (240,(i>>16)&255,(i>>8)&255,i&255)
def scene(o,context,atoms,source):return o.contact(CONTACT_SCENE,(7,context,len(atoms),*atoms),source,True,True)
def main():
    t=time.perf_counter();pop=PyCudaPopulationExecV1(GpuPopulationSpecV1(80_000_000_000,fanout=2,sites_per_feature=3,eligibility_horizon=8,site_delta_capacity=1<<20,edge_delta_capacity=1<<21))
    o=ReferenceOrganismV2(PopulationSpecV1(1024,2,3,42,8));o.population=pop
    start=time.perf_counter()
    for i in range(ALIAS_COUNT):
        feature=9_000_000+i
        for k in range(2):
            src=2_000_000+i*2+k;scene(o,NAME,(feature,),src);o.contact(CONTACT_SURFACE,alias_surface(i),src,True,True)
    acquisition_ms=(time.perf_counter()-start)*1000
    samples=[]
    for i in (0,ALIAS_COUNT//2,ALIAS_COUNT-1):
        feature=9_000_000+i;start=time.perf_counter();units=o.language.lexeme(feature);dt=(time.perf_counter()-start)*1000
        samples.append({'index':i,'correct':units==alias_surface(i),'lookup_touches':o.language.last_lookup_touches,'lookup_ms':round(dt,6)})
    # Two raw demonstrations induce one generic one-port construction.
    for k,i in enumerate((0,1)):
        f=9_000_000+i;src=3_100_000+k;scene(o,REL_ALIAS,(f,),src);o.contact(CONTACT_SURFACE,alias_surface(i),src,True,True)
    assert o.language.template(REL_ALIAS,1) is not None
    far=9_000_000+ALIAS_COUNT-1;o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),4_200_001,True,True);scene(o,REL_ALIAS,(far,),4_100_000)
    start=time.perf_counter();action=o.tick();expression_ms=(time.perf_counter()-start)*1000
    public_ok=isinstance(action,ActionV2) and action.payload==alias_surface(ALIAS_COUNT-1)
    learned=o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),PARTNER,True,True) if action else {'credit':0,'revisions':0}
    counts=pop.sparse_counts();checks={
      'exact_alias_population':len(o.language._lexeme_sources)==ALIAS_COUNT,
      'first_middle_last_retrieve':all(x['correct'] for x in samples),
      'bounded_lookup_touches':all(x['lookup_touches']==1 for x in samples),
      'far_end_public_expression':public_ok,
      'public_expression_credited':learned['credit']>0 and learned['revisions']>0,
      'lived_population_large':counts['site_deltas']>200_000 and counts['edge_deltas']>400_000,
      'no_sparse_overflow':counts['site_overflow']==0 and counts['edge_overflow']==0,
      'rapid_acquisition':acquisition_ms<5_000.0,
      'rapid_far_expression':expression_ms<20.0,
      'single_generic_module':pop.module_compile_count==1,
      'no_v1_or_external_seed':not hasattr(o,'xi'),
    }
    result={'schema':'0x1.pycuda-language-pressure.v2','pass':all(checks.values()),'resident_sites':pop.spec.site_count,'procedural_edges':pop.spec.site_count*pop.spec.fanout,'learned_aliases':ALIAS_COUNT,'life_changed_sites':counts['site_deltas'],'life_changed_edges':counts['edge_deltas'],'device_bytes':pop.required_device_bytes,'acquisition_ms':round(acquisition_ms,3),'aliases_per_second':round(ALIAS_COUNT/(acquisition_ms/1000),3),'far_expression_ms':round(expression_ms,3),'samples':samples,'checks':checks,'elapsed_ms':round((time.perf_counter()-t)*1000,3),'claim':'V2_16K_LIVED_LEXICAL_BINDINGS_80B_NAMESPACE_SCALE_GATE_NOT_LANGUAGE_MASTERY'}
    print('PYCUDA_LANGUAGE_PRESSURE_V2 '+('GREEN' if result['pass'] else 'RED')+f" aliases={ALIAS_COUNT} acquire_ms={result['acquisition_ms']} far_ms={result['far_expression_ms']}")
    print(json.dumps(result,indent=2,sort_keys=True));pop.close();raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
