#!/usr/bin/env python3
"""N+1: a learned proper name selects the correct visible organism-discovered individual."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1,vf
from reference_discovered_visual_individual_naming_verify import (
    CLAUSE,NORA,LIAM,MIRA,discover,teach_name,
)
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_joint_attention_naming_v1 import VisualJointAttentionNamingV1
from reference_visual_named_individual_reference_v1 import VisualNamedIndividualReferenceV1
from reference_visual_object_file_tracker_v1 import VisualObjectFileTrackerV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xC301

def m_obs(t,o,s,temp,level1,seq,frame):
    return t.observe(o,s,temp,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    # Developmental named-subject productivity from two other discovered individuals.
    dev_o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));dev_s=VisualSensorIngressV1();dev_t=VisualObjectFileTrackerV1()
    nora=discover(dev_t,dev_o,dev_s,temporal,level1,1);teach_name(adult,dev_t,NORA,0xC311,0xC312)
    for src in (0xC313,0xC314):adult.observe_surface_construction(CLAUSE,(101,nora,301,401),b'the careful nora tests the sensor.',src)
    liam=discover(dev_t,dev_o,dev_s,temporal,level1,4);teach_name(adult,dev_t,LIAM,0xC315,0xC316)
    for src in (0xC317,0xC318):adult.observe_surface_construction(CLAUSE,(102,liam,302,402),b'the quiet liam inspects the valve.',src)

    # Test stream: Mira is discovered first while uniquely visible, then named by
    # joint attention. A second same-category individual appears afterward.
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));s=VisualSensorIngressV1();t=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    r1=m_obs(t,o,s,temp,level1,1,canvas(((A1,1,1),)));r2=m_obs(t,o,s,temp,level1,2,canvas(((A2,1,2),)))
    if len(r1)!=len(r2)!=1:raise RuntimeError('name_query:mira_seed')
    mira=r2[0][0]
    VisualJointAttentionNamingV1.observe(adult,t,MIRA,0xC321);VisualJointAttentionNamingV1.observe(adult,t,MIRA,0xC322)
    checks['mira_name_is_learned_while_visual_attention_has_one_organism_minted_referent']=(adult.language.lexeme(mira)==tuple(MIRA))

    r3=m_obs(t,o,s,temp,level1,3,canvas(((A1,1,3),(A1,1,14))))
    checks['second_same_category_individual_appears_without_changing_mira_identity']=(len(r3)==2 and r3[0][0]==mira and r3[1][0]!=mira)
    peer=r3[1][0]
    resolved=VisualNamedIndividualReferenceV1.resolve(adult,t,MIRA)
    checks['bare_name_selects_mira_not_same_category_peer']=(resolved==mira and adult.language.lexeme(peer) is None)
    response=bytes(adult.leaf(CLAUSE,(101,resolved,301,401)).surface) if resolved else b''
    checks['resolved_name_drives_ordinary_productive_language_response']=(response==b'the careful mira tests the sensor.')

    # Names of known but nonvisible individuals do not hijack visual reference.
    checks['known_nonvisible_name_does_not_select_current_visual_peer']=(VisualNamedIndividualReferenceV1.resolve(adult,t,LIAM)==0)
    checks['unknown_name_refuses']=(VisualNamedIndividualReferenceV1.resolve(adult,t,b'oscar')==0)

    # During a one-frame full occlusion, no track is currently visible, so bare name
    # cannot falsely claim a sensory referent. Reappearance rebinds the same identity.
    blank=canvas(())
    m_obs(t,o,s,temp,level1,4,blank)
    hidden=VisualNamedIndividualReferenceV1.resolve(adult,t,MIRA)
    r5=m_obs(t,o,s,temp,level1,5,canvas(((A2,1,5),(A2,1,14))))
    visible_again=VisualNamedIndividualReferenceV1.resolve(adult,t,MIRA)
    checks['name_reference_refuses_during_occlusion_then_recovers_same_identity_after_reidentification']=(
        hidden==0 and len(r5)==2 and r5[0][0]==mira and visible_again==mira)

    # Source withdrawal removes generated-name authority; visual individual remains.
    adult.language.withdraw_source(0xC322)
    withdrawn=VisualNamedIndividualReferenceV1.resolve(adult,t,MIRA)
    checks['withdrawn_name_authority_blocks_reference_without_erasing_visual_track']=(withdrawn==0 and mira in t.active and o._active_entity_features(mira))

    blob=json.dumps(adult.checkpoint(),sort_keys=True)
    checks['no_query_response_pair_or_visual_track_state_is_checkpointed']=(response.decode() not in blob and 'visible_entities' not in blob and 'active_entity' not in blob)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-named-visual-individual-query.v1','contract':'FOUNDRY_NAMED_VISUAL_INDIVIDUAL_QUERY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'A_BARE_LEARNED_NAME_NOW_SELECTS_THE_CORRECT_VISIBLE_ORGANISM_DISCOVERED_INDIVIDUAL_AMONG_SAME_CATEGORY_PEERS_AND_DRIVES_A_PRODUCTIVE_RESPONSE','conversation':['mira',response.decode() if response else ''],'checks':checks,'failed':failed,'entities':{'mira':mira,'same_category_peer':peer},'remaining_red':['DEICTIC_THIS_THAT_GROUNDING','NAME_REFERENCE_TO_NONVISIBLE_REMEMBERED_INDIVIDUAL','DIRECT_NAMED_VISUAL_REFERENCE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
