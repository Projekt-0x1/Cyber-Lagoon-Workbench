#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CLAUSE=8801;EN1,EN2=1101,1102;DE1,DE2=2101,2102;MIX1,MIX2=3101,3102;NEUTRAL=9999
DOG=(100,200,201);CAT=(100,300,301);SEE=(400,500,501)
def u(s):return tuple(s.encode())
def entity(o,e,features,src):return o.contact(CONTACT_ENTITY_FEATURES,(e,len(features),*features),src,True,True)
def name(o,e,text,src):o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):o.contact(CONTACT_SCENE,(7,CLAUSE,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def partner(o,p):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)

def train(o):
    entities=((1001,(*DOG,9001)),(1002,(*DOG,9002)),(2001,(*CAT,9101)),(2002,(*CAT,9102)),(3001,(*SEE,9201)),(3002,(*SEE,9202)))
    for e,f in entities:entity(o,e,f,80000+e)
    for src,dog,cat,see,di,ci,si in (
        (EN1,'dog','cat','sees',1001,2001,3001),(EN2,'dog','cat','sees',1002,2002,3002),
        (DE1,'Hund','Katze','sieht',1001,2001,3001),(DE2,'Hund','Katze','sieht',1002,2002,3002)):
        name(o,di,dog,src);name(o,ci,cat,src);name(o,si,see,src)
    clause(o,(1001,3001,2001),'dog sees cat.',EN1);clause(o,(2002,3002,1002),'cat sees dog.',EN2)
    clause(o,(1001,3001,2001),'Hund sieht Katze.',DE1);clause(o,(2002,3002,1002),'Katze sieht Hund.',DE2)
    return o

def heldout(o):
    for e,f,s in ((1003,(*DOG,9991),90001),(2003,(*CAT,9992),90002),(3003,(*SEE,9993),90003)):entity(o,e,f,s)

def emit(o,p):
    partner(o,p);sid=o.contact(CONTACT_SCENE,(7,CLAUSE,3,1003,3003,2003),91000+p,True,True);a=o.tick();return sid,a

def main():
    t=time.perf_counter();checks={};base=train(ReferenceOrganismV2(PopulationSpecV1(65536,2,3,42,8)));heldout(base)
    checks['shared_world_one_entity_map']=len(base.entity_features)==9
    templates=base.language.template_candidates(CLAUSE,3)
    checks['parallel_lexicons_shared_construction']=len(base._lexeme_rows(1003))==2 and len(base._lexeme_rows(2003))==2 and len(base._lexeme_rows(3003))==2 and len(templates)==1 and templates[0].support==4
    en=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()));sid,a=emit(en,EN1)
    checks['english_partner_recruits_english']=isinstance(a,ActionV2) and a.scene_identity==sid and a.payload==u('dog sees cat.')
    de=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()));sid,b=emit(de,DE1)
    checks['german_partner_recruits_german']=isinstance(b,ActionV2) and b.scene_identity==sid and b.payload==u('Hund sieht Katze.')
    neutral=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()));sid,c=emit(neutral,NEUTRAL)
    checks['neutral_exact_tie_refuses']=c is None
    checks['no_language_router']=all(not hasattr(base,n) for n in ('language_id','language_router','english','german','translate'))
    checks['same_scene_relation_all_ecologies']=a is not None and b is not None and en._scene_by_id[a.scene_identity].context==de._scene_by_id[b.scene_identity].context==CLAUSE and en._scene_by_id[a.scene_identity].atoms==de._scene_by_id[b.scene_identity].atoms
    mix=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()))
    name(mix,1001,'dog',MIX1);name(mix,2001,'Katze',MIX1);name(mix,3001,'sieht',MIX1)
    name(mix,1002,'dog',MIX2);name(mix,2002,'Katze',MIX2);name(mix,3002,'sieht',MIX2)
    clause(mix,(1001,3001,2001),'dog sieht Katze.',MIX1);clause(mix,(2002,3002,1002),'Katze sieht dog.',MIX2)
    partner(mix,MIX1);sid=mix.contact(CONTACT_SCENE,(7,CLAUSE,3,1003,3003,2003),92001,True,True);m=mix.tick()
    checks['mixed_history_can_recruit_learned_codeswitch']=isinstance(m,ActionV2) and m.payload==u('dog sieht Katze.')
    checks['no_unlearned_codeswitch']=all(x not in (u('dog sieht Katze.'),u('Hund sees cat.')) for x in ((a.payload if a else ()),(b.payload if b else ())))
    learned=en.contact(CONTACT_CONSEQUENCE,(a.ticket,1),EN1,True,True)
    checks['selected_surface_causal_credit']=learned.get('selection_credit',0)>0 and len(a.selection_occurrences)==4
    cp=mix.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));checks['checkpoint_exact']=r.digest()==mix.digest()
    out={'schema':'0x1.reference-organism-multilingual.v2','pass':all(checks.values()),'checks':checks,'english':bytes(a.payload).decode() if a else '','german':bytes(b.payload).decode() if b else '','mixed':bytes(m.payload).decode() if m else '','surface_candidates_per_entity':2,'shared_construction_templates':len(base.language.template_candidates(CLAUSE,3)),'claim':'SHARED_RELATION_MULTILINGUAL_SURFACE_ECOLOGY_REFERENCE_ONLY_NOT_DIRECT_PARITY_OR_HUMAN_MULTILINGUAL_MASTERY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_MULTILINGUAL '+('GREEN' if out['pass'] else 'RED')+' shared_relations=1 language_router=0 partner_history=1 codeswitch_learned=1')
    print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)

if __name__=='__main__':main()
