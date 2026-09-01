#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_social_language_verify import settle,u
NAME=7701;CTX=7702;P_SENSOR=9101;P_VALVE=9102;P=9103
CATS={
 'quiet':(10,11,12),'careful':(10,13,14),'engineer':(20,21,22),'technician':(20,23,24),
 'inspects':(30,31,32),'tests':(30,33,34),'sensor':(40,41,42),'valve':(40,43,44),
 'steady':(10,15,16),'planner':(20,25,26),'observes':(30,35,36),
 'amber':(10,17,18),'analyst':(20,27,28),'reviews':(30,37,38),'result':(40,45,46),
 'swift':(10,19,20),'observer':(20,29,30),'tracks':(30,39,40)}

def entity(o,e,cat,uniq,src):
    f=(*CATS[cat],uniq);return o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)

def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,NAME,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)

def sentence(words):return f"the {words[0]} {words[1]} {words[2]} the {words[3]}."
def scene(o,atoms,src):return o.contact(CONTACT_SCENE,(7,CTX,4,*atoms),src,True,True)
def surface(o,text,src):return o.contact(CONTACT_SURFACE,u(text),src,True,True)
def partner(o,p):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),60000+p,True,True)

def add_named(o,e,cat,uniq,src):
    entity(o,e,cat,uniq,src);name(o,e,cat,src);name(o,e,cat,src+1)

def teach_form(o,p,obj,src):
    partner(o,p)
    for i in range(2):
        o.contact(CONTACT_SCENE,(7,NAME,1,obj),src+i,True,True);surface(o,'it',src+100+i)
    assert o.language.form(obj,(COND_REINSTATED,),require_conditioned=True)==u('it')

def establish(o,p,atoms,src):
    partner(o,p);scene(o,atoms,src);a=o.tick();assert a is not None;settle(o,a,p,1);return a

def train(o):
    rows=[];base=1000
    for ci,cat in enumerate(CATS):
        e1=base+ci*10+1;e2=base+ci*10+2
        add_named(o,e1,cat,9000+ci*20+1,10000+ci*100+1);add_named(o,e2,cat,9000+ci*20+2,10000+ci*100+2);rows.append((cat,e1,e2))
    ids={cat:(e1,e2) for cat,e1,e2 in rows}
    a=(ids['careful'][0],ids['engineer'][0],ids['tests'][0],ids['sensor'][0])
    b=(ids['quiet'][1],ids['technician'][1],ids['inspects'][1],ids['valve'][1])
    scene(o,a,30001);surface(o,sentence(('careful','engineer','tests','sensor')),31001)
    scene(o,b,30002);surface(o,sentence(('quiet','technician','inspects','valve')),31002)
    assert o.language.template(CTX,4) is not None
    return ids

def overlap(o,a,b):
    sa=set(o._atom_sites(a));sb=set(o._atom_sites(b))
    return (len(sa&sb)/len(sa) if sa else 0.0),(len(sa&sb)/len(sb) if sb else 0.0)

def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(131072,2,4,42,8));ids=train(o)
    sensor_prior=(ids['careful'][0],ids['engineer'][0],ids['tests'][0],ids['sensor'][0]);establish(o,P_SENSOR,sensor_prior,40001);teach_form(o,P_SENSOR,ids['sensor'][0],41001)
    valve_prior=(ids['quiet'][1],ids['technician'][1],ids['inspects'][1],ids['valve'][1]);establish(o,P_VALVE,valve_prior,40002);teach_form(o,P_VALVE,ids['valve'][1],42001)
    checks['conditioned_form_learned']=o.language.form(ids['sensor'][0],(COND_REINSTATED,),require_conditioned=True)==u('it') and o.language.form(ids['valve'][1],(COND_REINSTATED,),require_conditioned=True)==u('it')
    sensor_current=(ids['quiet'][0],ids['technician'][0],ids['inspects'][0],ids['sensor'][0]);partner(o,P_SENSOR);scene(o,sensor_current,43001);ctx1,cond1=o._surface_context(o.current_scene);surface(o,'the quiet technician inspects it.',43101)
    valve_current=(ids['careful'][1],ids['engineer'][1],ids['tests'][1],ids['valve'][1]);partner(o,P_VALVE);scene(o,valve_current,43002);ctx2,cond2=o._surface_context(o.current_scene);surface(o,'the careful engineer tests it.',43102)
    checks['distributed_compact_template']=ctx1==ctx2 and cond1[-1]==cond2[-1]==(COND_REINSTATED,) and o.language.template(ctx1,4) is not None
    old_sensor=88001;entity(o,old_sensor,'sensor',99001,50001)
    prior=(ids['careful'][1],ids['technician'][0],ids['tests'][1],old_sensor);establish(o,P,prior,50002);old_ep=o.last_shared_episode_by_partner[P]
    for i,(m,a,v,obj) in enumerate((('amber','analyst','reviews','result'),('quiet','engineer','inspects','valve'),('steady','planner','observes','result'))):
        atoms=(ids[m][i%2],ids[a][(i+1)%2],ids[v][i%2],ids[obj][(i+1)%2]);partner(o,P);scene(o,atoms,51000+i);x=o.tick();assert x is not None;settle(o,x,P,1)
    checks['old_sensor_not_last_shared']=o.last_shared_episode_by_partner[P]!=old_ep
    new_sensor=88002;entity(o,new_sensor,'sensor',99002,52001)
    frac_new,frac_old=overlap(o,new_sensor,old_sensor)
    checks['sparse_signature_overlap']=frac_new>=0.75 and frac_old>=0.75 and new_sensor!=old_sensor
    final=(ids['swift'][0],ids['observer'][0],ids['tracks'][0],new_sensor);partner(o,P);scene(o,final,52002);ctx,cond=o._surface_context(o.current_scene);a=o.tick()
    checks['new_identity_reinstated_from_coalition']=new_sensor!=old_sensor and cond==((),(),(),(COND_REINSTATED,)) and ctx==ctx1
    checks['distributed_reference_compacts']=isinstance(a,ActionV2) and a.payload==u('the swift observer tracks it.')
    checks['form_selection_occurrence']=a is not None and any(k==PREF_FORM for k,_slot,_cid,_oid in a.selection_occurrences)
    learned=settle(o,a,P,1) if a else {}
    checks['form_receives_network_credit']=learned.get('selection_credit',0)>0 and learned.get('selection_network_updates',0)>=1
    remote=ReferenceOrganismV2.restore(copy.deepcopy(o.checkpoint()));alien=88003
    remote.contact(CONTACT_ENTITY_FEATURES,(alien,4,777001,777002,777003,777004),53001,True,True)
    partner(remote,P);scene(remote,(ids['swift'][1],ids['observer'][1],ids['tracks'][1],alien),53002);_c,_d=remote._surface_context(remote.current_scene)
    checks['remote_unrelated_identity_not_reinstated']=_d[-1]==()
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['checkpoint_exact']=r.digest()==o.digest() and r._shared_reinstated(P,new_sensor)
    checks['derived_site_index_not_checkpoint']='shared_site_incidence' not in json.dumps(cp) and 'entity_site_index' not in json.dumps(cp)
    result={'schema':'0x1.reference-organism-distributed-reference.v2','pass':all(checks.values()),'checks':checks,'compact_surface':bytes(a.payload).decode() if a else '','overlap_new':round(frac_new,3),'overlap_old':round(frac_old,3),'history_depth':len([row for row in o.shared_episode_relations if row.partner==P]),'site_touches':o.last_shared_site_touches,'claim':'REMATERIALIZED_IDENTITY_FORM_NETWORK_CREDIT_REFERENCE_ONLY_NOT_DIRECT_PARITY_OR_HUMAN_DISCOURSE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_DISTRIBUTED_REFERENCE '+('GREEN' if result['pass'] else 'RED')+' rematerialized_identity=1 form_network=1 transcript=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
