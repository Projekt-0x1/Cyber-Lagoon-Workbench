#!/usr/bin/env python3
"""N+1: multiple learned event features compose to disambiguate unnamed absent deictic memories."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import RAW_ADJ,RAW_TEST,RAW_INSPECT,RAW_SENSOR
from reference_event_cued_unnamed_deictic_recall_verify import ONE
from reference_joint_attention_episode_memory_v1 import JointAttentionEpisodeMemoryV1
from reference_nonvisible_unnamed_deictic_event_verify import setup,world,THAT,CHANNEL,EP_A,EP_B,event_surface

CTX=0xE401;SOURCES=(0xE411,0xE412)

def teach(adult):
    # ONE is not assumed from another assay's developmental state.
    adult.observe_surface_item(ONE,b'one',0xE421);adult.observe_surface_item(ONE,b'one',0xE422)
    rows=(
        ((THAT,401,ONE),b'that sensor one',0xE430),
        ((THAT,301,401,ONE),b'that tests sensor one',0xE440),
        ((THAT,302,401,ONE),b'that inspects sensor one',0xE450),
    )
    for atoms,raw,base in rows:
        for k in range(2):
            if not adult.observe_surface_construction(CTX,atoms,raw,base+k):raise RuntimeError('compositional_event_cue:construction')
        contents={tuple(map(int,row.atoms)) for row in adult.language.invert_surface(tuple(raw)) if row.atoms}
        if contents!={atoms}:raise RuntimeError('compositional_event_cue:inverse')

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,right,left_pos,right_pos=setup();teach(adult);memory=JointAttentionEpisodeMemoryV1()
    world(o,(RAW_ADJ,left,RAW_TEST,RAW_SENSOR))
    for source in EP_A:a=memory.observe(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,source)
    world(o,(RAW_ADJ,right,RAW_INSPECT,RAW_SENSOR))
    for source in EP_B:b=memory.observe(adult,o,tracker,g,b'that',THAT,*right_pos,CHANNEL,source)
    tracker.active={}

    one_feature=memory.resolve(adult,o,b'that sensor one',CHANNEL)
    test_cue=memory.resolve(adult,o,b'that tests sensor one',CHANNEL)
    inspect_cue=memory.resolve(adult,o,b'that inspects sensor one',CHANNEL)
    test_speech=b'' if test_cue is None else event_surface(adult,test_cue.event)
    inspect_speech=b'' if inspect_cue is None else event_surface(adult,inspect_cue.event)

    checks['shared_single_event_feature_remains_ambiguous']=(one_feature is None)
    checks['two_current_event_features_jointly_select_first_absent_episode']=(test_cue is not None and test_cue.entity==left and test_cue.event==a.event)
    checks['alternative_two_feature_conjunction_selects_second_absent_episode']=(inspect_cue is not None and inspect_cue.entity==right and inspect_cue.event==b.event)
    checks['composed_event_cues_change_visible_recalled_language']=(test_speech==b'the careful engineer tests the sensor.' and inspect_speech==b'the careful engineer inspects the sensor.')

    # Mixed cue shares one feature with each event; scores tie 1-1 and must refuse.
    adult.observe_surface_construction(CTX,(THAT,301,302,ONE),b'that tests inspects one',0xE461)
    adult.observe_surface_construction(CTX,(THAT,301,302,ONE),b'that tests inspects one',0xE462)
    mixed=memory.resolve(adult,o,b'that tests inspects one',CHANNEL)
    checks['mixed_features_split_across_memories_refuse_equal_overlap']=(mixed is None)

    cp=copy.deepcopy(memory.checkpoint());restored=JointAttentionEpisodeMemoryV1.restore(cp);replay=restored.resolve(adult,o,b'that tests sensor one',CHANNEL)
    checks['checkpoint_preserves_compositional_content_recall_without_cue_cache']=(replay is not None and replay.entity==left and 'tests sensor one' not in json.dumps(cp,sort_keys=True))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-compositional-event-cued-deictic-recall.v1','contract':'FOUNDRY_COMPOSITIONAL_EVENT_CUED_DEICTIC_RECALL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'MULTIPLE_CURRENT_EVENT_FEATURES_NOW_COMPOSE_TO_SELECT_BETWEEN_UNNAMED_ABSENT_DEICTIC_MEMORIES_WHILE_SINGLE_OR_SPLIT_CUES_REFUSE','conversation':[['that tests sensor one',test_speech.decode() if test_speech else ''],['that inspects sensor one',inspect_speech.decode() if inspect_speech else '']],'checks':checks,'failed':failed,'remaining_red':['NATURAL_DEICTIC_EPISODE_RETENTION_FROM_PARTNER_CONSEQUENCE','RELATIONAL_CUE_BEYOND_FLAT_FEATURE_OVERLAP','DIRECT_COMPOSITIONAL_DEICTIC_MEMORY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
