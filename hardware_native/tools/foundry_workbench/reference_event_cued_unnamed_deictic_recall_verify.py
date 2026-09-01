#!/usr/bin/env python3
"""N+1: event-content cue disambiguates two unnamed absent deictic episodes without recency."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import RAW_ADJ,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE
from reference_joint_attention_episode_memory_v1 import JointAttentionEpisodeMemoryV1
from reference_nonvisible_unnamed_deictic_event_verify import (
    setup,world,THAT,AGAIN,RECALL_CTX,CHANNEL,EP_A,EP_B,event_surface,
)

ONE=0xE301;CUE_CTX=0xE302;ONE_SOURCES=(0xE311,0xE312);CUE_SOURCES=(0xE321,0xE322)

def teach_event_cues(adult):
    for source in ONE_SOURCES:adult.observe_surface_item(ONE,b'one',source)
    for source in CUE_SOURCES:
        if not adult.observe_surface_construction(CUE_CTX,(THAT,401,ONE),b'that sensor one',source):raise RuntimeError('event_cue:sensor')
        if not adult.observe_surface_construction(CUE_CTX,(THAT,402,ONE),b'that valve one',source+0x10):raise RuntimeError('event_cue:valve')
    for raw,expected in ((b'that sensor one',(THAT,401,ONE)),(b'that valve one',(THAT,402,ONE))):
        contents={tuple(map(int,row.atoms)) for row in adult.language.invert_surface(tuple(raw)) if row.atoms}
        if contents!={expected}:raise RuntimeError('event_cue:inverse')

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,right,left_pos,right_pos=setup();teach_event_cues(adult);memory=JointAttentionEpisodeMemoryV1()
    world(o,(RAW_ADJ,left,RAW_TEST,RAW_SENSOR))
    for source in EP_A:left_episode=memory.observe(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,source)
    world(o,(RAW_ADJ,right,RAW_INSPECT,RAW_VALVE))
    for source in EP_B:right_episode=memory.observe(adult,o,tracker,g,b'that',THAT,*right_pos,CHANNEL,source)
    tracker.active={}

    plain=memory.resolve(adult,o,b'that again',CHANNEL)
    sensor=memory.resolve(adult,o,b'that sensor one',CHANNEL)
    valve=memory.resolve(adult,o,b'that valve one',CHANNEL)
    sensor_speech=b'' if sensor is None else event_surface(adult,sensor.event)
    valve_speech=b'' if valve is None else event_surface(adult,valve.event)
    checks['plain_reduced_deictic_remains_ambiguous_with_two_matching_old_episodes']=(plain is None)
    checks['event_content_cue_selects_correct_absent_individual_without_recency']=(sensor is not None and valve is not None and sensor.entity==left and valve.entity==right and sensor.entity!=valve.entity)
    checks['event_cued_recall_changes_visible_language_to_each_old_event']=(sensor_speech==b'the careful engineer tests the sensor.' and valve_speech==b'the careful engineer inspects the valve.')

    # Replace cue with the other event atom while leaving marker/channel/history fixed.
    checks['same_marker_channel_and_history_switch_only_on_current_event_content']=(sensor.channel==valve.channel==CHANNEL and sensor.marker==valve.marker==THAT and sensor.event==left_episode.event and valve.event==right_episode.event)

    # A cue shared by both event propositions cannot force a winner. `engineer` is
    # resident feature 201 in both events; teach a structurally identical recall form.
    for source in (0xE341,0xE342):
        adult.observe_surface_construction(CUE_CTX,(THAT,201,ONE),b'that engineer one',source)
    tied=memory.resolve(adult,o,b'that engineer one',CHANNEL)
    checks['event_cue_shared_by_both_memories_refuses_without_id_or_age_tiebreak']=(tied is None)

    cp=copy.deepcopy(memory.checkpoint());restored=JointAttentionEpisodeMemoryV1.restore(cp)
    replay=restored.resolve(adult,o,b'that sensor one',CHANNEL)
    checks['checkpoint_restart_preserves_content_cued_disambiguation']=(replay is not None and replay.entity==left)
    blob=json.dumps(cp,sort_keys=True);source=inspect.getsource(JointAttentionEpisodeMemoryV1).lower()
    checks['memory_stores_no_cue_word_event_slot_recency_or_surface']=(all(token not in blob for token in ('sensor','valve','engineer','that','one')) and all(token not in source for token in ('event_slot','object_slot','most_recent','sensor','valve','engineer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-event-cued-unnamed-deictic-recall.v1','contract':'FOUNDRY_EVENT_CUED_UNNAMED_DEICTIC_RECALL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'EVENT_CONTENT_IN_A_REDUCED_DEICTIC_NOW_SELECTS_BETWEEN_TWO_UNNAMED_ABSENT_SHARED_ATTENTION_EPISODES_WITHOUT_RECENCY_OR_ROLE_LABELS','conversation':[['that sensor one',sensor_speech.decode() if sensor_speech else ''],['that valve one',valve_speech.decode() if valve_speech else '']],'checks':checks,'failed':failed,'remaining_red':['NATURAL_DEICTIC_EPISODE_RETENTION_FROM_PARTNER_CONSEQUENCE','COMPOSITIONAL_MULTI_FEATURE_EPISODE_CUE','DIRECT_EVENT_CUED_DEICTIC_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
