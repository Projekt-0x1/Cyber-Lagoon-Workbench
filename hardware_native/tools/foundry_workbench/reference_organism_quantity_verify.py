#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=7001;PAIR=7002;PARTNER=99001

def surf(i:int):return (65,i&255,(i>>8)&255,(i>>16)&255)
def name(o,feature,source):
    o.contact(CONTACT_SCENE,(7,NAME,1,feature),source,True,True);o.contact(CONTACT_SURFACE,surf(feature),source+100000,True,True)
def pair_surface(a,b):return (*surf(a),32,*surf(b))
def pair(o,a,b,source):
    o.contact(CONTACT_SCENE,(7,PAIR,2,a,b),source,True,True);o.contact(CONTACT_SURFACE,pair_surface(a,b),source+100000,True,True)

def main():
    t=time.perf_counter();spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=3,eligibility_horizon=8);o=ReferenceOrganismV2(spec)
    # 1000 learned lexical identities, every developmental scene consumed immediately.
    for f in range(1,1001):name(o,f,10000+f)
    # 2000 complete episodic pair scenes under one construction. Each rare atom occurs only a few times.
    for i in range(2000):
        a=1+(i%1000);b=1+((i*37+101)%1000);pair(o,a,b,30000+i)
    history=len(o.episodes);heap_after_development=len(o._pending_heap)
    # Partial cue: exact known atom nominates only its small incidence posting, not 3000 episodes.
    target_atom=997;o.contact(CONTACT_SCENE,(7,PAIR,2,target_atom,0),80001,True,True);partial=o.current_scene;o._complete_scene(partial)
    episode_touches=o.last_episode_lookup_touches;partial.demonstrated=True
    # Fresh language scene after the entire developmental history should be selected at O(frontier), not by skipping history.
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),80002,True,True)
    fresh=o.contact(CONTACT_SCENE,(7,PAIR,2,13,17),80003,True,True);selected=o._select_pending_scene();pending_touches=o.last_pending_lookup_touches
    checks={
      'thousands_of_lived_episodes':history>=3000,
      'developmental_scenes_leave_no_stale_frontier':heap_after_development==0,
      'episodic_lookup_touched_not_history':0<episode_touches<=8 and episode_touches*100<history,
      'fresh_pending_selected':selected is not None and selected.identity==fresh,
      'pending_lookup_touched_not_history':pending_touches<=2,
      'derived_indices_not_checkpoint_authority':not any(k.startswith('_') for k in o.checkpoint()),
      'population_cold_exceeds_touched':o.population.spec.site_count>o.population.materialized_site_count(),
    }
    result={'schema':'0x1.reference-organism-quantity.v1','pass':all(checks.values()),'checks':checks,'episodes':history,'learned_lexemes':len(o.language._lexeme_sources),'episode_lookup_touches':episode_touches,'pending_lookup_touches':pending_touches,'pending_heap_after_development':heap_after_development,'resident_sites':o.population.spec.site_count,'experienced_sites':o.population.materialized_site_count(),'elapsed_ms':round((time.perf_counter()-t)*1000,3),'claim':'LIFETIME_HISTORY_TOUCHED_WORK_REFERENCE_NOT_HUMAN_SCALE_OR_DIRECT_PARITY'}
    print('FOUNDRY_REFERENCE_ORGANISM_QUANTITY '+('GREEN' if result['pass'] else 'RED')+f" episodes={history} episode_touches={episode_touches} pending_touches={pending_touches} history_scan=0")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
