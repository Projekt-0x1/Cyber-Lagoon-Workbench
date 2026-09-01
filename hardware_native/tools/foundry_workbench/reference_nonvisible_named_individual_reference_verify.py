#!/usr/bin/env python3
"""N+1: a proper name retrieves a durable organism-discovered individual after it leaves current vision."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1
from reference_discovered_visual_individual_naming_verify import CLAUSE,MIRA,NORA
from reference_global_discourse_relevance_verify import fresh
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_remembered_named_individual_v1 import RememberedNamedIndividualV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_joint_attention_naming_v1 import VisualJointAttentionNamingV1
from reference_visual_object_file_tracker_v1 import VisualObjectFileTrackerV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_multiple_visual_object_files_verify import train_temporal

SOURCE=0xDE01

def obs(t,o,s,temp,level1,seq,image):
    return t.observe(o,s,temp,SOURCE,seq,image,__import__('reference_continuous_visual_sensor_ownership_verify').vf(level1,image),VisualSensorIngressV1.frame_digest(image))
def discover(t,o,s,temp,level1,start):
    e1,*_=obs(t,o,s,temp,level1,start,A1);e2,_r,same,_=obs(t,o,s,temp,level1,start+1,A2)
    if e1!=e2 or not same:raise RuntimeError('remembered_name:discover')
    return e1

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));s=VisualSensorIngressV1();t=VisualObjectFileTrackerV1()
    mira=discover(t,o,s,temporal,level1,1);VisualJointAttentionNamingV1.observe(adult,t,MIRA,0xDE11);VisualJointAttentionNamingV1.observe(adult,t,MIRA,0xDE12)
    # Give the ordinary clause family enough proper-name subject diversity.
    nora=discover(t,o,s,temporal,level1,4);VisualJointAttentionNamingV1.observe(adult,t,NORA,0xDE21);VisualJointAttentionNamingV1.observe(adult,t,NORA,0xDE22)
    for src in (0xDE31,0xDE32):adult.observe_surface_construction(CLAUSE,(101,mira,301,401),b'the careful mira tests the sensor.',src)
    for src in (0xDE33,0xDE34):adult.observe_surface_construction(CLAUSE,(102,nora,302,402),b'the quiet nora inspects the valve.',src)

    # End visual occurrence entirely; proper name must still retrieve durable individual memory.
    t.gap();remembered=RememberedNamedIndividualV1.resolve(adult,o,MIRA)
    speech=bytes(adult.leaf(CLAUSE,(101,remembered,301,401)).surface) if remembered else b''
    checks['name_retrieves_same_individual_when_no_visual_file_is_active']=(not t.active_entity and remembered==mira)
    checks['remembered_name_drives_ordinary_productive_language_about_absent_individual']=(speech==b'the careful mira tests the sensor.')

    # A new visible same-category individual is not the named individual and cannot steal reference.
    peer=discover(t,o,s,temporal,level1,7);peer_visible=t.active_entity==peer;with_peer=RememberedNamedIndividualV1.resolve(adult,o,MIRA)
    checks['visible_same_category_distractor_does_not_steal_absent_proper_name']=(peer_visible and peer not in (mira,nora) and with_peer==mira and adult.language.lexeme(peer) is None)

    # Checkpoint/restart keeps durable organism identity + learned name but no active visual file.
    adult_cp=copy.deepcopy(adult.checkpoint());org_cp=copy.deepcopy(o.checkpoint());ra=type(adult).restore(adult_cp);ro=ReferenceOrganismV2.restore(org_cp)
    replay=RememberedNamedIndividualV1.resolve(ra,ro,MIRA)
    checks['checkpoint_restart_preserves_absent_name_reference_without_visual_track']=(replay==mira)

    # Give peer the same name under independent sources: two durable homonyms must refuse.
    adult.observe_surface_item(peer,MIRA,0xDE41);adult.observe_surface_item(peer,MIRA,0xDE42)
    homonym=RememberedNamedIndividualV1.resolve(adult,o,MIRA)
    checks['two_durable_individuals_with_same_name_refuse_without_recency_tiebreak']=(homonym==0)
    adult.language.withdraw_source(0xDE42);restored_unique=RememberedNamedIndividualV1.resolve(adult,o,MIRA)
    checks['withdrawing_one_homonym_name_authority_restores_unique_remembered_referent']=(restored_unique==mira and o._active_entity_features(peer))

    # Withdraw original name authority too: entity remains remembered perceptually but cannot be named.
    adult.language.withdraw_source(0xDE12);withdrawn=RememberedNamedIndividualV1.resolve(adult,o,MIRA)
    checks['withdrawn_original_name_blocks_reference_without_erasing_individual_memory']=(withdrawn==0 and o._active_entity_features(mira))

    blob=json.dumps(adult_cp,sort_keys=True)
    checks['checkpoint_has_learned_name_relation_but_no_visual_track_or_query_response_cache']=(speech.decode() not in blob and all(token not in blob for token in ('active_entity','current_frame','query_response','referent_cache')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-nonvisible-named-individual-reference.v1','contract':'FOUNDRY_NONVISIBLE_NAMED_INDIVIDUAL_REFERENCE_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'A_PROPER_NAME_NOW_RETRIEVES_THE_SAME_ORGANISM_DISCOVERED_INDIVIDUAL_AFTER_IT_LEAVES_CURRENT_VISION_AND_SUPPORTS_NEW_LANGUAGE_ABOUT_THE_ABSENT_REFERENT','conversation':['mira',speech.decode() if speech else ''],'checks':checks,'failed':failed,'entities':{'mira':mira,'visible_same_category_peer':peer},'remaining_red':['DEICTIC_REFERENCE_TO_NONVISIBLE_UNNAMED_OBJECT','EVENT_MEMORY_FOR_ABSENT_OBJECT_STATE_CHANGE','DIRECT_ABSENT_NAME_REFERENCE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
