#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import reference_organism_v2 as organism
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

R_LEFT=0xA11CE;R_RIGHT=0xB0B;R_META=0xFEE1;SRC=7711;PARTNER=8811
ATOMS=(101,201,102,202);WORDS=('a','x','b','y')

def u(s):return tuple(s.encode())
def singleton(o,atom,source):return o.contact(CONTACT_SCENE,(7,1,1,atom),source,True,True)
def link_scene(o,left,right,relation,source,independent=True):
    before=o.next_scene;o.contact(CONTACT_SCENE_LINK,(left,right,relation),source,True,independent)
    return o.next_scene-1 if o.next_scene>before else 0

def teach_names(o):
    for i,(atom,word) in enumerate(zip(ATOMS,WORDS)):
        for repeat in range(2):
            sid=9000+i*10+repeat;singleton(o,atom,sid);o.contact(CONTACT_SURFACE,u(word),sid+1000,True,True)

def make_rank2(o,base):
    a=singleton(o,ATOMS[0],base);x=singleton(o,ATOMS[1],base+1)
    b=singleton(o,ATOMS[2],base+2);y=singleton(o,ATOMS[3],base+3)
    left=link_scene(o,a,x,R_LEFT,SRC+base);right=link_scene(o,b,y,R_RIGHT,SRC+base)
    high=link_scene(o,left,right,R_META,SRC+base)
    return left,right,high

def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));teach_names(o)
    left,right,high=make_rank2(o,20000);l=o._scene_by_id[left];r=o._scene_by_id[right];h=o._scene_by_id.get(high)
    checks['rank1_relations_are_current_occurrence_computations']=left and right and l.binding_identity and r.binding_identity and len(l.relation_occurrences)==len(r.relation_occurrences)==1
    checks['rank2_relation_materializes_from_relation_endpoints']=h is not None and h.context==R_META and h.atoms==ATOMS and h.binding_identity not in (0,l.binding_identity,r.binding_identity)
    inherited=tuple(dict.fromkeys((*l.relation_occurrences,*r.relation_occurrences)))
    checks['rank2_carries_child_and_new_relation_occurrences']=h is not None and len(h.relation_occurrences)==3 and h.relation_occurrences[:2]==inherited and all(any(p.identity==oid for p in o.population.occurrences) for oid in h.relation_occurrences)
    checks['higher_rank_identity_depends_on_grouping']=int(organism._digest('relation-binding-v2',[R_META,l.binding_identity,r.binding_identity])[:15],16)==h.binding_identity and int(organism._digest('relation-binding-v2',[R_META,r.binding_identity,l.binding_identity])[:15],16)!=h.binding_identity

    # Language is a boundary projection of the higher-rank relation, not its owner.
    o.contact(CONTACT_SURFACE,u('a x b y'),30001,True,True)
    _l2,_r2,h2=make_rank2(o,21000);o.contact(CONTACT_SURFACE,u('a x b y'),30002,True,True)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),31000,True,True)
    _l3,_r3,h3=make_rank2(o,22000);active=o._scene_by_id[h3];action=o.tick()
    checks['language_can_project_higher_rank_relation']=isinstance(action,ActionV2) and action.payload==u('a x b y') and action.binding_identity==active.binding_identity
    checks['higher_rank_binding_is_selection_member']=isinstance(action,ActionV2) and any(row[0]==PREF_BINDING for row in action.selection_occurrences)
    checks['all_relation_occurrences_participate_in_language_action']=isinstance(action,ActionV2) and tuple(action.relation_occurrences)==tuple(active.relation_occurrences) and len(action.relation_occurrences)==3 and all(oid in action.contributors for oid in action.relation_occurrences)
    if action is not None:
        before=len(o.selection_configuration_revisions);o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),PARTNER,True,True)
        checks['consequence_credits_complete_higher_rank_configuration']=len(o.selection_configuration_revisions)==before+1 and o._selection_configuration_evidence(action.selection_context,tuple(row[:3] for row in action.selection_occurrences))[0]>0
    else:checks['consequence_credits_complete_higher_rank_configuration']=False

    cp=o.checkpoint();restored=ReferenceOrganismV2.restore(copy.deepcopy(cp));rh=restored._scene_by_id.get(high)
    checks['checkpoint_preserves_relation_of_relations']=restored.digest()==o.digest() and rh is not None and rh.binding_identity==h.binding_identity and rh.relation_occurrences==h.relation_occurrences
    yoked=ReferenceOrganismV2.restore(copy.deepcopy(cp));before_links=len(yoked.scene_links);before_scenes=yoked.next_scene
    try:yoked.contact(CONTACT_SCENE_LINK,(left,right,R_META+1),SRC+1,True,False)
    except ValueError as exc:checks['nonindependent_higher_rank_relation_refuses']=str(exc)=='organism:scene_relation_independence' and len(yoked.scene_links)==before_links and yoked.next_scene==before_scenes
    else:checks['nonindependent_higher_rank_relation_refuses']=False
    checks['no_tom_or_emotion_opcode']=not any(hasattr(o,name) for name in ('theory_of_mind','alice_feels','belief_state','emotion_label'))
    result={'schema':'0x1.reference-organism-higher-rank-relation.v2','pass':all(checks.values()),'checks':checks,'rank1_relation_occurrences':2,'rank2_relation_occurrences':0 if h is None else len(h.relation_occurrences),'runtime_llm':False,'reference_only':True,'physical_direct_parity':'NOT_RUN','claim':'GENERIC_RELATION_OF_RELATIONS_CAN_PROJECT_THROUGH_LANGUAGE_REFERENCE_ONLY_NOT_TOM_CAPABILITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_HIGHER_RANK_RELATION '+('GREEN' if result['pass'] else 'RED')+' rank2=1 relation_occurrences='+str(result['rank2_relation_occurrences'])+' language_projection=1 tom_opcode=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
