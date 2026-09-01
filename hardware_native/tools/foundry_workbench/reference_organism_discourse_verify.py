#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=100;CTX=9001;REL_NEXT=50001
M={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
def u(s):return tuple(s.encode())
def scene(o,atoms,source,channel=7):return o.contact(CONTACT_SCENE,(channel,CTX,len(atoms),*atoms),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o,p,channel=7):return o.contact(CONTACT_PARTNER_CONTEXT,(1,channel,p),70000+p,True,True)
def settle(o,a,p,effect=1):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),p,True,True)
def train(o):
    for f,text in M.items():
        o.contact(CONTACT_SCENE,(7,NAME,1,f),10000+f,True,True);surface(o,text,20000+f)
        o.contact(CONTACT_SCENE,(7,NAME,1,f),11000+f,True,True);surface(o,text,21000+f)
    a=(101,201,301,401);b=(102,202,302,402)
    scene(o,a,30001);surface(o,'the careful engineer tests the sensor.',31001)
    scene(o,b,30002);surface(o,'the quiet technician inspects the valve.',31002)
    assert o.language.template(CTX,4) is not None

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8);P=9901
    o=ReferenceOrganismV2(spec);train(o);partner(o,P)
    atoms=((101,201,301,401),(102,201,302,402),(101,202,302,401),(102,202,301,402),(101,201,302,402),(102,202,302,401),(101,202,301,402),(102,201,301,401))
    ids=[scene(o,row,40000+i) for i,row in enumerate(atoms)]
    for i in range(len(ids)-1):o.contact(CONTACT_SCENE_LINK,(ids[i],ids[i+1],REL_NEXT),50000+i,True,True)
    before_actions=len(o.actions);emitted=[];emitted_actions=[];checkpoint=None
    for i in range(len(ids)):
        a=o.tick();assert isinstance(a,ActionV2);emitted_actions.append(a);emitted.append((a.scene_identity,bytes(a.payload).decode()))
        if i==2:checkpoint=o.checkpoint()
    checks['eight_scene_endogenous_chain']=[x[0] for x in emitted]==ids and len(emitted)==8
    checks['source_linked_chain_does_not_wait_on_public_turns']=(
        len(o.actions)-before_actions==8 and len([a for a in o.actions if not a.settled])==8)
    checks['no_new_request_between_actions']=o.tick() is None
    checks['unreturned_chain_is_not_promoted_to_common_ground']=P not in o.last_shared_episode_by_partner

    for action in reversed(emitted_actions):settle(o,action,P,1)
    checks['reverse_consequence_order_settles_exact_chain']=not any(not a.settled for a in emitted_actions)
    checks['latest_returned_episode_updates_partner_history']=(
        o.last_shared_episode_by_partner.get(P)==next(e.identity for e in o.episodes if e.scene_identity==ids[0]))

    # Mid-chain restore has the same exact next scene and surface.
    r=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));a=r.tick();checks['checkpoint_mid_discourse_exact']=a is not None and a.scene_identity==ids[3] and bytes(a.payload).decode()==emitted[3][1]

    # Link selection must override an earlier unrelated pending scene.
    x=ReferenceOrganismV2(spec);train(x);partner(x,P)
    first=scene(x,(101,201,301,401),60001);distractor=scene(x,(102,202,302,402),60002);linked=scene(x,(101,202,302,401),60003)
    x.contact(CONTACT_SCENE_LINK,(first,linked,REL_NEXT),61001,True,True)
    a0=x.tick();settle(x,a0,P,1);a1=x.tick();checks['link_overrides_earlier_distractor']=a0.scene_identity==first and a1.scene_identity==linked

    # With link source withdrawn, heap order cannot impersonate discourse relation.
    y=ReferenceOrganismV2(spec);train(y);partner(y,P)
    yf=scene(y,(101,201,301,401),62001);yd=scene(y,(102,202,302,402),62002);yl=scene(y,(101,202,302,401),62003)
    y.contact(CONTACT_SCENE_LINK,(yf,yl,REL_NEXT),63001,True,True);fa=y.tick();settle(y,fa,P,1);y.contact(CONTACT_WITHDRAW_SOURCE,(63001,),64001,True,True);ya=y.tick()
    checks['link_source_withdrawal_removes_continuation_authority']=ya is None

    z=ReferenceOrganismV2(spec);train(z);partner(z,P)
    left_a=scene(z,(101,201,301,401),70001);right_a=scene(z,(102,202,302,402),70002)
    z.contact(CONTACT_SCENE_LINK,(left_a,right_a,REL_NEXT),71001,True,True)
    z.contact(CONTACT_DISCOURSE_SURFACE,u('the careful engineer tests the sensor. then the quiet technician inspects the valve.'),71001)
    left_b=scene(z,(102,202,302,402),70003);right_b=scene(z,(101,201,301,401),70004)
    z.contact(CONTACT_SCENE_LINK,(left_b,right_b,REL_NEXT),71002,True,True)
    z.contact(CONTACT_DISCOURSE_SURFACE,u('the quiet technician inspects the valve. then the careful engineer tests the sensor.'),71002)
    checks['span_template_learned']=z.language.span_template(REL_NEXT,2) is not None
    partner(z,P);first=scene(z,(101,201,301,401),73001);second=scene(z,(102,202,302,402),73002)
    z.contact(CONTACT_SCENE_LINK,(first,second,REL_NEXT),73003,True,True)
    za=z.tick();assert za is not None;settle(z,za,P,1);zb=z.tick()
    checks['span_selection_occurrence']=isinstance(zb,ActionV2) and any(k==PREF_SPAN for k,_slot,_cid,_oid in zb.selection_occurrences)
    learned=settle(z,zb,P,1) if zb else {}
    checks['span_receives_network_credit']=learned.get('selection_credit',0)>0 and learned.get('selection_network_updates',0)>=1
    checks['span_emits_continuation_suffix']=isinstance(zb,ActionV2) and zb.payload==u(' then the quiet technician inspects the valve.')

    checks['no_prompt_goal_api']=not hasattr(o,'prompt') and not hasattr(o,'enqueue_goal') and not hasattr(o,'speak')
    result={'schema':'0x1.reference-organism-discourse.v1','pass':all(checks.values()),'checks':checks,'scene_count':8,'actions':len(emitted),'emitted':emitted,'claim':'CONSEQUENCE_GATED_RESIDENT_DISCOURSE_CONTINUATION_NOT_HUMAN_PROSE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_DISCOURSE '+('GREEN' if result['pass'] else 'RED')+' scenes=8 prompts_after_start=0 consequence_gated=1 resident_links=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
