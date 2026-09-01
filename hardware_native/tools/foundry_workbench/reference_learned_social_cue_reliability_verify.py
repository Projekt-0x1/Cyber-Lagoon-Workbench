#!/usr/bin/env python3
"""N+1: independent social consequence calibrates gaze/pointing-like cue lanes and changes later reference."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,train_level1
from reference_discovered_visual_individual_naming_verify import CLAUSE
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_reliability_weighted_directional_fusion_v1 import ReliabilityWeightedDirectionalFusionV1
from reference_social_cue_reliability_v1 import SocialCueReliabilityV1,Q
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xDC01;LANE_A=0xDC11;LANE_B=0xDC12;SA=(0xDC21,0xDC22);SB=(0xDC31,0xDC32)

def motion(origin,endpoint):
    oy,ox=origin;ey,ex=endpoint;return ((oy,ox),((oy+ey)//2,(ox+ex)//2),(ey,ex))
def verbalize(adult,entity):return bytes(adult.leaf(CLAUSE,(101,entity,301,401)).surface) if entity else b''

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    rows=tracker.observe(o,sensor,temp,level1,SOURCE,1,canvas(((A1,1,1),(A1,1,8),(A1,1,15)),h=8,w=24),VisualSensorIngressV1.frame_digest(canvas(((A1,1,1),(A1,1,8),(A1,1,15)),h=8,w=24)))
    left,center,right=[row[0] for row in rows];pos=[(row[1],row[2]) for row in rows]
    # Names only make the selected identity legible in the public sentence.
    for entity,name,s1,s2 in ((left,b'mira',0xDC41,0xDC42),(center,b'nora',0xDC43,0xDC44),(right,b'liam',0xDC45,0xDC46)):
        adult.observe_surface_item(entity,name,s1);adult.observe_surface_item(entity,name,s2)
    for src in (0xDC51,0xDC52):adult.observe_surface_construction(CLAUSE,(101,left,301,401),b'the careful mira tests the sensor.',src)
    for src in (0xDC53,0xDC54):adult.observe_surface_construction(CLAUSE,(102,center,302,402),b'the quiet nora inspects the valve.',src)
    # Third concrete example gives all four slots sufficient diversity while retaining same family.
    for src in (0xDC55,0xDC56):adult.observe_surface_construction(CLAUSE,(103,right,303,403),b'the steady liam checks the panel.',src)

    origin=(147,21);cue_a=motion(origin,pos[0]);cue_b=motion(origin,pos[2]);neutral=SocialCueReliabilityV1()
    neutral_target=ReliabilityWeightedDirectionalFusionV1.resolve(tracker,cue_a,LANE_A,cue_b,LANE_B,neutral);neutral_speech=verbalize(adult,neutral_target)
    checks['equal_unlearned_reliability_preserves_geometric_center_fusion']=(neutral_target==center and b'nora' in neutral_speech)

    learned=SocialCueReliabilityV1()
    for source in SA:
        learned.observe(LANE_A,source,1,True);learned.observe(LANE_A,source,1,True)
    for source in SB:
        learned.observe(LANE_B,source,-1,True);learned.observe(LANE_B,source,-1,True)
    target_a=ReliabilityWeightedDirectionalFusionV1.resolve(tracker,cue_a,LANE_A,cue_b,LANE_B,learned);speech_a=verbalize(adult,target_a)
    checks['independent_consequence_calibration_shifts_same_raw_cues_to_reliable_lane_target']=(target_a==left and b'mira' in speech_a and learned.weight_q16(LANE_A)>learned.weight_q16(LANE_B))

    reverse=SocialCueReliabilityV1()
    for source in SA:
        reverse.observe(LANE_A,source,-1,True);reverse.observe(LANE_A,source,-1,True)
    for source in SB:
        reverse.observe(LANE_B,source,1,True);reverse.observe(LANE_B,source,1,True)
    target_b=ReliabilityWeightedDirectionalFusionV1.resolve(tracker,cue_a,LANE_A,cue_b,LANE_B,reverse);speech_b=verbalize(adult,target_b)
    checks['reversing_causal_reliability_history_reverses_spoken_referent']=(target_b==right and b'liam' in speech_b)

    yoked=SocialCueReliabilityV1()
    for source in SA:yoked.observe(LANE_A,source,1,False)
    for source in SB:yoked.observe(LANE_B,source,-1,False)
    yoked_target=ReliabilityWeightedDirectionalFusionV1.resolve(tracker,cue_a,LANE_A,cue_b,LANE_B,yoked)
    checks['yoked_outcomes_cannot_calibrate_social_cue_reliability']=(yoked_target==center and yoked.weight_q16(LANE_A)==yoked.weight_q16(LANE_B)==Q)

    restored=SocialCueReliabilityV1.restore(copy.deepcopy(learned.checkpoint()));replay=ReliabilityWeightedDirectionalFusionV1.resolve(tracker,cue_a,LANE_A,cue_b,LANE_B,restored)
    checks['checkpoint_preserves_source_qualified_reliability_not_visual_scene_or_target']=(replay==left and restored.evidence==learned.evidence and str(left) not in json.dumps(learned.checkpoint(),sort_keys=True))
    source=(inspect.getsource(SocialCueReliabilityV1)+inspect.getsource(ReliabilityWeightedDirectionalFusionV1)).lower()
    checks['reliability_ecology_has_no_modality_name_or_target_policy']=(all(token not in source for token in ("'gaze'","'pointing'",'mira','nora','liam','target_entity','expected','first_lane ==','second_lane ==')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-learned-social-cue-reliability.v1','contract':'FOUNDRY_LEARNED_SOCIAL_CUE_RELIABILITY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'THE_SAME_CONFLICTING_SOCIAL_DIRECTIONAL_CUES_NOW_CHANGE_THE_SPOKEN_REFERENT_AFTER_INDEPENDENT_CUE_RELIABILITY_EXPERIENCE_AND_REVERSE_WHEN_HISTORY_REVERSES','conversation':{'neutral':neutral_speech.decode(),'lane_a_reliable':speech_a.decode(),'lane_b_reliable':speech_b.decode()},'weights':{'a':learned.weight_q16(LANE_A),'b':learned.weight_q16(LANE_B)},'checks':checks,'failed':failed,'remaining_red':['NATURAL_PARTNER_RETURN_ATTACHMENT_TO_CUE_RELIABILITY','RAW_GAZE_SENSOR_OWNERSHIP','VERBAL_DEICTIC_PLUS_RELIABILITY_FUSION','DIRECT_SOCIAL_CUE_RELIABILITY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
