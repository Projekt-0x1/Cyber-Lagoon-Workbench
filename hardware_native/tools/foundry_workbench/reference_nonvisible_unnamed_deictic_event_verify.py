#!/usr/bin/env python3
"""N+1: reduced deictic recall retrieves an unnamed absent individual and its remembered grounded event."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1,vf,RAW_ADJ,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_joint_attention_episode_memory_v1 import JointAttentionEpisodeMemoryV1
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_open_world_visual_entity_discovery_verify import KNOWN_A,install_known,ground,train_direct,world
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xE101;THAT=0xE201;AGAIN=0xE202;RECALL_CTX=0xE203
CHANNEL=31;OTHER_CHANNEL=32
LEX_SOURCES=(0xE211,0xE212);AGAIN_SOURCES=(0xE221,0xE222);CONSTRUCTION_SOURCES=(0xE231,0xE232)
EP_A=(0xE241,0xE242);EP_B=(0xE251,0xE252)

def establish_language(adult):
    for source in LEX_SOURCES:adult.observe_surface_item(THAT,b'that',source)
    for source in AGAIN_SOURCES:adult.observe_surface_item(AGAIN,b'again',source)
    for source in CONSTRUCTION_SOURCES:
        if not adult.observe_surface_construction(RECALL_CTX,(THAT,AGAIN),b'that again',source):raise RuntimeError('unnamed_deictic:construction')
    rows=adult.language.invert_surface(tuple(b'that again'))
    contents={tuple(map(int,row.atoms)) for row in rows if row.atoms}
    if contents!={(THAT,AGAIN)}:raise RuntimeError('unnamed_deictic:inverse')

def setup():
    level1=train_level1();temporal=train_temporal(level1);ta=temporal.relation(vf(level1,A1),vf(level1,__import__('reference_continuous_visual_sensor_ownership_verify').A2))
    adult,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    install_known(o,KNOWN_A,ta);ground(g,adult,o,KNOWN_A,201,0xE300);train_direct(g,adult,o);establish_language(adult)
    sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    frame1=canvas(((A1,1,1),(A1,1,14)),h=8,w=22)
    first=tracker.observe(o,sensor,temp,level1,SOURCE,1,frame1,VisualSensorIngressV1.frame_digest(frame1))
    frame2=canvas(((A2,1,2),(A2,1,13)),h=8,w=22)
    rows=tracker.observe(o,sensor,temp,level1,SOURCE,2,frame2,VisualSensorIngressV1.frame_digest(frame2))
    if len(first)!=2 or len(rows)!=2 or [r[0] for r in first]!=[r[0] for r in rows]:raise RuntimeError('unnamed_deictic:visual')
    left,right=rows[0][0],rows[1][0];left_pos=(rows[0][1],rows[0][2]);right_pos=(rows[1][1],rows[1][2])
    if g.resolve_world_atom(adult,o,left)!=201 or g.resolve_world_atom(adult,o,right)!=201:raise RuntimeError('unnamed_deictic:ground')
    if adult.language.lexeme(left) is not None or adult.language.lexeme(right) is not None:raise RuntimeError('unnamed_deictic:named')
    return adult,o,g,tracker,left,right,left_pos,right_pos

def event_surface(adult,event):return bytes(adult._leaf_surface(int(event)))

def main():
    started=time.perf_counter();checks={};base,o,g,tracker,left,right,left_pos,right_pos=setup()
    base_cp=copy.deepcopy(base.checkpoint());org_cp=copy.deepcopy(o.checkpoint())

    # Partner A remembers only the unnamed left individual and its grounded event.
    a=type(base).restore(copy.deepcopy(base_cp));ao=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));ma=JointAttentionEpisodeMemoryV1();ta=copy.deepcopy(tracker)
    world(ao,(RAW_ADJ,left,RAW_TEST,RAW_SENSOR))
    for source in EP_A:row_a=ma.observe(a,ao,ta,g,b'that',THAT,*left_pos,CHANNEL,source)
    one_source=JointAttentionEpisodeMemoryV1();one_source.observe(a,ao,ta,g,b'that',THAT,*left_pos,CHANNEL,EP_A[0])
    ta.active={};resolved_a=ma.resolve(a,ao,b'that again',CHANNEL);answer_a=b'' if resolved_a is None else event_surface(a,resolved_a.event)

    # Matched partner memory instead established the other same-category individual/event.
    b=type(base).restore(copy.deepcopy(base_cp));bo=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));mb=JointAttentionEpisodeMemoryV1();tb=copy.deepcopy(tracker)
    world(bo,(RAW_ADJ,right,RAW_INSPECT,RAW_VALVE))
    for source in EP_B:row_b=mb.observe(b,bo,tb,g,b'that',THAT,*right_pos,CHANNEL,source)
    tb.active={};resolved_b=mb.resolve(b,bo,b'that again',CHANNEL);answer_b=b'' if resolved_b is None else event_surface(b,resolved_b.event)

    checks['two_unnamed_same_category_individuals_have_no_lexical_name_authority']=(a.language.lexeme(left) is None and a.language.lexeme(right) is None)
    checks['same_reduced_that_again_recovers_partner_established_absent_individual']=(resolved_a is not None and resolved_b is not None and resolved_a.entity==left and resolved_b.entity==right and resolved_a.entity!=resolved_b.entity)
    checks['remembered_entity_specific_event_changes_visible_language']=(answer_a==b'the careful engineer tests the sensor.' and answer_b==b'the careful engineer inspects the valve.' and answer_a!=answer_b)
    checks['one_source_cannot_install_joint_attention_episode']=(one_source.resolve(a,ao,b'that again',CHANNEL) is None)
    checks['wrong_partner_channel_cannot_claim_episode']=(ma.resolve(a,ao,b'that again',OTHER_CHANNEL) is None)

    # Two supported prior episodes with the same marker/channel are genuinely ambiguous.
    amb=JointAttentionEpisodeMemoryV1.restore(copy.deepcopy(ma.checkpoint()))
    world(ao,(RAW_ADJ,right,RAW_INSPECT,RAW_VALVE));ta2=copy.deepcopy(tracker)
    for source in EP_B:amb.observe(a,ao,ta2,g,b'that',THAT,*right_pos,CHANNEL,source)
    ambiguous=amb.resolve(a,ao,b'that again',CHANNEL)
    checks['two_matching_prior_joint_attention_episodes_refuse_without_recency_tiebreak']=(ambiguous is None)

    # Checkpoint memory persists IDs/support only; no active visual file is required after restart.
    mcp=copy.deepcopy(ma.checkpoint());restored=JointAttentionEpisodeMemoryV1.restore(mcp);ra=type(base).restore(copy.deepcopy(base_cp));ro=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));replay=restored.resolve(ra,ro,b'that again',CHANNEL)
    checks['checkpoint_restart_preserves_unnamed_absent_deictic_event_reference']=(replay is not None and replay.entity==left and event_surface(ra,replay.event)==answer_a)

    # Source withdrawal weakens episode authority, not organism identity/event existence.
    a.language.withdraw_source(EP_A[1]);withdrawn=ma.resolve(a,ao,b'that again',CHANNEL)
    checks['source_withdrawal_blocks_episode_reference_without_erasing_entity_memory']=(withdrawn is None and ao._active_entity_features(left) and a._has_leaf(row_a.event))

    blob=json.dumps(mcp,sort_keys=True);source=inspect.getsource(JointAttentionEpisodeMemoryV1).lower()
    checks['memory_checkpoint_contains_only_opaque_ids_and_sources_not_surface_gesture_or_transcript']=(all(token not in blob for token in ('that again','tests the sensor','inspects the valve','point_y2','point_x2','transcript','current_frame')))
    checks['memory_has_no_last_referent_recency_name_or_expected_answer_policy']=(all(token not in source for token in ('last_referent','most_recent','proper_name','expected_answer','mira','nora')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-nonvisible-unnamed-deictic-event.v1','contract':'FOUNDRY_NONVISIBLE_UNNAMED_DEICTIC_EVENT_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'REDUCED_DEICTIC_RECALL_NOW_RETRIEVES_AN_UNNAMED_ABSENT_ORGANISM_INDIVIDUAL_AND_ITS_PARTNER_ESTABLISHED_EVENT_WITHOUT_TRANSCRIPT_OR_LAST_REFERENT_STATE','conversation':{'left':['that again',answer_a.decode() if answer_a else ''],'right':['that again',answer_b.decode() if answer_b else '']},'checks':checks,'failed':failed,'entities':{'left':left,'right':right},'remaining_red':['NATURAL_DEICTIC_EPISODE_RETENTION_FROM_PARTNER_CONSEQUENCE','MULTIPLE_MARKER_EPISODE_DISAMBIGUATION_BY_EVENT_CUE','DIRECT_UNNAMED_ABSENT_DEICTIC_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
