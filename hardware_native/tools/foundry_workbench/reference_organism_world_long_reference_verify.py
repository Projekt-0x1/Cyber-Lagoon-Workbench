#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_social_language_verify import scene,partner,settle,u,CTX
from reference_organism_long_reference_verify import prepare,run_until,P

SENSOR=401;SEE=77001;FS=(45,46,47,48)

def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def see(o,entity,src,independent=True):return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(131072,2,4,42,8)
    o=ReferenceOrganismV2(spec);_ctx,words=prepare(o)
    feat(o,SENSOR,FS,8001)
    checks['talking_during_prepare_did_not_mint_world']=o._world_revisions.row_count==0
    see(o,SENSOR,SEE,True)
    checks['see_writes_sensor_world']=SENSOR in o._world_marked_entities()
    partner(o,P);scene(o,CTX,(101,201,301,401),70001);first=o.tick();assert first is not None;settle(o,first,P,1,True)
    run_until(o,words,0,12,71000)
    silent=[o.tick() for _ in range(5)]
    checks['distractors_and_silence_keep_world']=len(silent)==5 and all(x is None for x in silent) and SENSOR in o._world_marked_entities()
    buried=copy.deepcopy(o.checkpoint())
    final=(1012,2012,3012,401);partner(o,P);scene(o,CTX,final,73001);a=o.tick();aw=o._world_state_occurrences(CTX,final)
    checks['long_compact_still_recruits_seen_world']=isinstance(a,ActionV2) and a.payload==u('the m12 a12 v12 it.') and {row[0] for row in aw}=={SENSOR} and all(row[3] in a.contributors for row in aw)
    fresh=ReferenceOrganismV2.restore(copy.deepcopy(buried));partner(fresh,9552);scene(fresh,CTX,final,73002);b=fresh.tick();bw=fresh._world_state_occurrences(CTX,final)
    checks['fresh_partner_explicit_still_recruits_world']=isinstance(b,ActionV2) and b.payload==u('the m12 a12 v12 the sensor.') and {row[0] for row in bw}=={SENSOR} and all(row[3] in b.contributors for row in bw)
    yoked=ReferenceOrganismV2(spec);feat(yoked,SENSOR,FS,8001);see(yoked,SENSOR,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2.restore(copy.deepcopy(buried));cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True)
    partner(cut,P);scene(cut,CTX,final,73003);c=cut.tick()
    checks['world_withdrawal_keeps_compact_drops_world']=isinstance(c,ActionV2) and c.payload==u('the m12 a12 v12 it.') and not cut._world_state_occurrences(CTX,final)
    checks['no_transcript_window']=all(not hasattr(o,n) for n in ('transcript','context_window','imagine','prompt'))
    result={'schema':'0x1.reference-organism-world-long-reference.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'distractors':12,'compact':bytes(a.payload).decode() if isinstance(a,ActionV2) else '','claim':'SEEN_WORLD_SURVIVES_LONG_DISTRACTOR_BURIAL_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_LONG_REFERENCE '+('GREEN' if result['pass'] else 'RED')+' distractors=12 world_survives=1 transcript=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
