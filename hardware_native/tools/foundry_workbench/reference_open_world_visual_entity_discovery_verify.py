#!/usr/bin/env python3
"""N+1: organism mints novel visual entity identity from owned temporal continuity."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import (
    A1,A2,B1,B2,SRC_A,SRC_B,train_level1,vf,SURFACE,pair,DIRECT,
    RAW_ADJ,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE,
)
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM
from reference_visual_object_file_tracker_v1 import VisualObjectFileTrackerV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

KNOWN_A=0xAA01;KNOWN_B=0xAA02;TRAIN_FEATURE_SOURCE=0xAA10;TRACK_SOURCE=0xAA20;WORLD_SOURCE=0xAA30

def train_temporal(level1):
    temporal=TemporalVisualContinuityV1();sensor=VisualSensorIngressV1()
    for source,left,right,start in ((SRC_A,A1,A2,1),(SRC_B,B1,B2,100)):
        seq=start
        for _ in range(FEATURE_QUORUM):
            for image in (left,right):
                digest=VisualSensorIngressV1.frame_digest(image);frame,contiguous=sensor.ingest(source,seq,image,digest)
                if not contiguous:temporal.gap()
                temporal.observe_features((vf(level1,frame),));seq+=1
            seq+=1
    return temporal

def install_known(o,entity,feature):o.contact(CONTACT_ENTITY_FEATURES,(entity,1,feature),TRAIN_FEATURE_SOURCE,True,True)
def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)
def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xAB00+i*4+n)
def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)

def obs(tracker,o,sensor,temporal,level1,source,seq,image):
    return tracker.observe(o,sensor,temporal,source,seq,image,vf(level1,image),VisualSensorIngressV1.frame_digest(image))

def main():
    started=time.perf_counter();checks={};level1=train_level1();fa1,fa2,fb1,fb2=(vf(level1,x) for x in (A1,A2,B1,B2))
    trained_temporal=train_temporal(level1);ta=trained_temporal.relation(fa1,fa2);tb=trained_temporal.relation(fb1,fb2)
    if not ta or not tb or ta==tb:raise RuntimeError('entity_discovery:temporal_training')

    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    install_known(o,KNOWN_A,ta);install_known(o,KNOWN_B,tb);ground(g,adult,o,KNOWN_A,201,0xAC00);ground(g,adult,o,KNOWN_B,203,0xAC10)

    sensor=VisualSensorIngressV1();temporal=TemporalVisualContinuityV1.restore(copy.deepcopy(trained_temporal.checkpoint()));tracker=VisualObjectFileTrackerV1()
    e1,r1,same1,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,1,A1)
    e1b,r2,same2,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,2,A2)
    checks['organism_mints_entity_without_host_entity_token_and_continuity_preserves_it']=(e1>0 and e1b==e1 and not same1 and same2 and r2==ta)
    checks['completed_object_file_binds_temporal_invariant_as_durable_feature']=(o._active_entity_features(e1)==(ta,))

    # Gap with identical views creates a distinct individual.
    e2,_r3,same3,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,4,A1)
    e2b,r4,same4,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,5,A2)
    checks['sequence_gap_mints_distinct_entity_even_for_same_visual_category']=(e2!=e1 and e2b==e2 and not same3 and same4 and r4==ta)

    # Contiguous but unlearned cross-category transition splits object file; next
    # coherent B transition stabilizes the new identity under invariant tb.
    e3,r5,same5,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,6,B1)
    e3b,r6,same6,_=obs(tracker,o,sensor,temporal,level1,TRACK_SOURCE,7,B2)
    checks['incompatible_contiguous_transition_starts_new_entity_file']=(e3 not in (e1,e2) and not same5 and e3b==e3 and same6 and r6==tb)

    checks['discovered_entities_inherit_grounded_categories_without_language_pair']=(g.resolve_world_atom(adult,o,e1)==201 and g.resolve_world_atom(adult,o,e2)==201 and g.resolve_world_atom(adult,o,e3)==203)

    # Checkpoint keeps discovered identity/features but not active tracker/sensor frame.
    cp_o=copy.deepcopy(o.checkpoint());cp_sensor=copy.deepcopy(sensor.checkpoint());ro=ReferenceOrganismV2.restore(cp_o);rs=VisualSensorIngressV1.restore(cp_sensor);rt=VisualObjectFileTrackerV1.restore(tracker.checkpoint());rtemp=TemporalVisualContinuityV1.restore(copy.deepcopy(trained_temporal.checkpoint()))
    e4,_rr,_same,_=obs(rt,ro,rs,rtemp,level1,TRACK_SOURCE,8,A1)
    checks['restart_preserves_entity_memory_but_breaks_active_object_file_continuity']=(ro._active_entity_features(e1)==(ta,) and e4 not in (e1,e2,e3))

    # Strong language phenotype: discovered E1—not a fixture ID—enters the same world-derived discourse path.
    train_direct(g,adult,o);world(o,(RAW_ADJ,e1,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier:
        for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,e1,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));re_ctx,re_frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(re_frontier);selected=tuple(adult.last_discourse_selected)
    expected=tuple(x.identity for x in host_frontier if b'careful' in bytes(x.surface) and b'engineer' in bytes(x.surface))
    checks['organism_discovered_entity_drives_long_form_grounded_language']=(re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    tracker_source=inspect.getsource(VisualObjectFileTrackerV1);organism_source=inspect.getsource(ReferenceOrganismV2.mint_visual_entity)
    checks['tracker_and_mint_have_no_category_language_expected_or_reward_authority']=(all(token not in (tracker_source+organism_source).lower() for token in ('category','language','concept','expected','reward','engineer')))
    checks['discovered_entity_feature_provenance_is_tracker_source_not_host_training_contact']=(
        o.entity_feature_sources.get(e1)=={TRACK_SOURCE} and o.entity_feature_sources.get(e2)=={TRACK_SOURCE}
        and o.entity_feature_sources.get(e3)=={TRACK_SOURCE}
        and o.entity_feature_sources.get(KNOWN_A)=={TRAIN_FEATURE_SOURCE}
        and o.entity_feature_sources.get(KNOWN_B)=={TRAIN_FEATURE_SOURCE})
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-open-world-visual-entity-discovery.v1','contract':'FOUNDRY_OPEN_WORLD_VISUAL_ENTITY_DISCOVERY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'RAW_OWNED_VISUAL_CONTINUITY_MINTS_A_NEW_ORGANISM_ENTITY_ID_THAT_INHERITS_GROUNDED_CONCEPT_AND_DRIVES_LANGUAGE_WITHOUT_HOST_ENTITY_TOKEN','entities':{'first':e1,'gap_same_category':e2,'different_category':e3,'post_restart':e4},'checks':checks,'failed':failed,'remaining_red':['MULTIPLE_SIMULTANEOUS_OBJECT_FILES_FROM_ONE_FRAME','OCCLUSION_AND_REIDENTIFICATION','DIRECT_VISUAL_ENTITY_DISCOVERY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
