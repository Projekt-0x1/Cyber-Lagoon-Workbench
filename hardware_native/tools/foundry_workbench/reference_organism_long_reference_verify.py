#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_social_language_verify import train_base,scene,surface,partner,settle,establish,teach_conditioned_form,u,NAME,CTX
P=9551

def teach_name(o,f,text,base):
    scene(o,NAME,(f,),base);surface(o,text,base+1000)
    scene(o,NAME,(f,),base+1);surface(o,text,base+1001)

def prepare(o):
    sensor_prior,valve_prior=train_base(o)
    # Learn reusable object-position compact reference from two referents.
    establish(o,9001,sensor_prior,40001);teach_conditioned_form(o,9001,401,41001)
    establish(o,9002,valve_prior,40002);teach_conditioned_form(o,9002,402,42001)
    partner(o,9001);scene(o,CTX,(102,202,302,401),43001);ctx1,cond1=o._surface_context(o.current_scene);surface(o,'the quiet technician inspects it.',43101)
    partner(o,9002);scene(o,CTX,(101,201,301,402),43002);ctx2,cond2=o._surface_context(o.current_scene);surface(o,'the careful engineer tests it.',43102)
    assert ctx1==ctx2 and cond1[-1]==cond2[-1]==(COND_REINSTATED,) and o.language.template(ctx1,4) is not None
    # Add unique lexical atoms for distractors and final held-out recombination.
    words={}
    for i in range(13):
        ids=(1000+i,2000+i,3000+i,4000+i);texts=(f'm{i}',f'a{i}',f'v{i}',f'o{i}')
        for j,(f,text) in enumerate(zip(ids,texts)):teach_name(o,f,text,50000+i*20+j*2)
        words[i]=ids
    # Final non-object slots have never occurred in any shared episode before the final test.
    return ctx1,words

def run_until(o,words,start,stop,source_base):
    outputs=[]
    partner(o,P)
    for i in range(start,stop):
        atoms=words[i]
        scene(o,CTX,atoms,source_base+i);a=o.tick();assert isinstance(a,ActionV2);outputs.append(bytes(a.payload).decode());settle(o,a,P,1,True)
    return outputs

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(131072,2,4,42,8);o=ReferenceOrganismV2(spec);compact_ctx,words=prepare(o)
    # Establish sensor as shared with P, then bury it under unrelated episodes.
    partner(o,P);scene(o,CTX,(101,201,301,401),70001);first=o.tick();assert first is not None;settle(o,first,P,1,True);sensor_ep=o.last_shared_episode_by_partner[P]
    first_half=run_until(o,words,0,6,71000);cp=o.checkpoint();digest_mid=o.digest();second_half=run_until(o,words,6,12,71000)
    checks['twelve_unrelated_shared_distractors']=len(first_half)+len(second_half)==12 and all(' it.' not in x for x in first_half+second_half)
    checks['last_shared_is_distractor']=o.last_shared_episode_by_partner[P]!=sensor_ep
    partner_relations=[row for row in o.shared_episode_relations if row.partner==P]
    checks['sensor_remains_in_partner_relations']=any(row.episode_identity==sensor_ep for row in partner_relations) and len(partner_relations)>=13
    # Silence does not erase common ground.
    silent=[o.tick() for _ in range(5)];checks['silence_valid_and_non_destructive']=all(x is None for x in silent)
    final=(1012,2012,3012,401);partner(o,P);scene(o,CTX,final,73001);ctx,conditions=o._surface_context(o.current_scene);a=o.tick()
    checks['old_reference_reinstated_after_distractors']=ctx==compact_ctx and conditions==((),(),(),(COND_REINSTATED,))
    checks['long_reference_compacts']=isinstance(a,ActionV2) and a.payload==u('the m12 a12 v12 it.')
    if a:settle(o,a,P,1,True)
    # A fresh partner cannot borrow P's deep shared history.
    fresh=ReferenceOrganismV2.restore(copy.deepcopy(o.checkpoint()));partner(fresh,9552);scene(fresh,CTX,final,73002);b=fresh.tick();checks['fresh_partner_explicit']=isinstance(b,ActionV2) and b.payload==u('the m12 a12 v12 the sensor.')
    # Mid-distractor checkpoint reproduces the remaining history and final compact form.
    r=ReferenceOrganismV2.restore(copy.deepcopy(cp));checks['mid_checkpoint_digest']=r.digest()==digest_mid
    replay_second=run_until(r,words,6,12,71000);partner(r,P);scene(r,CTX,final,73001);ra=r.tick();checks['checkpoint_continuation_exact']=replay_second==second_half and isinstance(ra,ActionV2) and ra.payload==u('the m12 a12 v12 it.')
    # Derived incidence is not durable authority.
    final_checkpoint=o.checkpoint();dumped=json.dumps(final_checkpoint,sort_keys=True);checks['derived_shared_indices_not_checkpoint']='shared_episode_incidence' not in dumped
    checks['settled_language_computation_not_checkpoint']=not final_checkpoint['actions'] and not final_checkpoint['action_commitments'] and sum(1 for row in partner_relations if row.closure_identity)==1
    checks['no_transcript_lookup_api']=all(not hasattr(o,n) for n in ('transcript','history_text','context_window','prompt'))
    result={'schema':'0x1.reference-organism-long-reference.v1','pass':all(checks.values()),'checks':checks,'distractors':12,'silent_ticks':5,'partner_relations':len(partner_relations),'compact_surface':bytes(a.payload).decode() if a else '', 'claim':'LONG_PARTNER_REFERENCE_REINSTATEMENT_REFERENCE_ONLY_NOT_HUMAN_DISCOURSE_OR_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_LONG_REFERENCE '+('GREEN' if result['pass'] else 'RED')+' distractors=12 silence=5 transcript=0 deep_history=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
