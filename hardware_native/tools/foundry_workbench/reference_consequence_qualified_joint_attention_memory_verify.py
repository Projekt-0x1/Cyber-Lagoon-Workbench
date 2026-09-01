#!/usr/bin/env python3
"""N+1: unnamed shared-attention episode recall is earned by actual Adult partner consequence, not observation count."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_continuous_visual_sensor_ownership_verify import RAW_ADJ,RAW_TEST,RAW_SENSOR
from reference_language_mastery_terminal_v1 import emit_choice
from reference_nonvisible_unnamed_deictic_event_verify import setup,world,THAT,CHANNEL,event_surface
from reference_predictive_credit_profile_v1 import Q
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

P=0xE501;S1=0xE511;S2=0xE512

def prepare():
    adult,o,g,tracker,left,_right,left_pos,_right_pos=setup();world(o,(RAW_ADJ,left,RAW_TEST,RAW_SENSOR))
    frontier=WorldDiscourseSituationBridgeV1.frontier(adult,o,g)
    if len(frontier)!=1:raise RuntimeError('consequence_joint:frontier')
    event=frontier[0]
    for _ in range(3):adult.experience_atomic_program(P,event,Q,context=event.identity,effort_q16=Q//32,controllable=True)
    adult.experience_program_background(P,False,context=event.identity)
    return adult,o,g,tracker,left,left_pos,event

def act(memory,adult,o,g,tracker,left_pos,source,outcome,independent=True):
    staged=memory.stage(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,source)
    if staged is None:return None,b'',False
    ticket,response,episode=staged;spoken=emit_choice(adult,response)
    settled=memory.settle_partner_return(adult,ticket,response,outcome,independent)
    return episode,spoken,settled

def main():
    started=time.perf_counter();checks={};base,o,g,tracker,left,left_pos,event=prepare();base_cp=copy.deepcopy(base.checkpoint());org_cp=copy.deepcopy(o.checkpoint())

    memory=ConsequenceQualifiedJointAttentionMemoryV1()
    staged_only=memory.stage(base,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,S1)
    before_return=memory.resolve(base,o,b'that again',CHANNEL)
    checks['joint_attention_observation_and_selected_response_do_not_self_install_memory']=(staged_only is not None and before_return is None and not memory.evidence)
    # Drop this unreturned transient eligibility; pending state is not durable authority.
    memory.pending.clear()

    ep1,spoken1,settled1=act(memory,base,o,g,tracker,left_pos,S1,Q,True)
    after_one=memory.resolve(base,o,b'that again',CHANNEL)
    ep1b,spoken1b,settled1b=act(memory,base,o,g,tracker,left_pos,S1,Q,True)
    after_repeat=memory.resolve(base,o,b'that again',CHANNEL)
    checks['one_independent_source_or_repeated_same_source_remains_below_memory_quorum']=(settled1 and settled1b and after_one is None and after_repeat is None)

    ep2,spoken2,settled2=act(memory,base,o,g,tracker,left_pos,S2,Q,True)
    learned=memory.resolve(base,o,b'that again',CHANNEL);answer=b'' if learned is None else event_surface(base,learned.event)
    checks['second_independent_successful_partner_return_installs_unnamed_episode_recall']=(settled2 and learned is not None and learned.entity==left and answer==b'the careful engineer tests the sensor.')
    checks['all_supporting_public_responses_use_normal_reafferent_program_surface']=(spoken1==spoken1b==spoken2==b'the careful engineer tests the sensor.')

    # Yoked second source: Adult gets a partner-use occurrence, but memory evidence cannot promote it.
    yoked_adult=type(base).restore(copy.deepcopy(base_cp));yoked_org=type(o).restore(copy.deepcopy(org_cp));yoked=ConsequenceQualifiedJointAttentionMemoryV1()
    y1,_ys1,_=act(yoked,yoked_adult,yoked_org,g,tracker,left_pos,S1,Q,True)
    y2,_ys2,yoked_settle=act(yoked,yoked_adult,yoked_org,g,tracker,left_pos,S2,Q,False)
    checks['yoked_second_source_cannot_install_shared_episode_memory']=(not yoked_settle and yoked.resolve(yoked_adult,yoked_org,b'that again',CHANNEL) is None)

    # Adverse second source produces negative local evidence, not support.
    adverse_adult=type(base).restore(copy.deepcopy(base_cp));adverse_org=type(o).restore(copy.deepcopy(org_cp));adverse=ConsequenceQualifiedJointAttentionMemoryV1()
    act(adverse,adverse_adult,adverse_org,g,tracker,left_pos,S1,Q,True)
    _a2,_as2,adverse_settle=act(adverse,adverse_adult,adverse_org,g,tracker,left_pos,S2,-Q,True)
    adverse_key=next(iter(adverse.evidence));negative=adverse.evidence[adverse_key].get(S2)
    checks['independent_adverse_return_records_negative_source_evidence_without_installing_recall']=(adverse_settle and negative==-1 and adverse.resolve(adverse_adult,adverse_org,b'that again',CHANNEL) is None)
    # Two later positive returns move S2 -1 -> 0 -> +1 and reacquire quorum.
    act(adverse,adverse_adult,adverse_org,g,tracker,left_pos,S2,Q,True)
    reacquired_mid=adverse.resolve(adverse_adult,adverse_org,b'that again',CHANNEL)
    act(adverse,adverse_adult,adverse_org,g,tracker,left_pos,S2,Q,True)
    reacquired=adverse.resolve(adverse_adult,adverse_org,b'that again',CHANNEL)
    checks['later_independent_success_reacquires_previously_adverse_episode_source']=(reacquired_mid is None and reacquired is not None and reacquired.entity==left)

    # Wrong response identity consumes no Adult partner consequence and no memory evidence.
    wrong_adult=type(base).restore(copy.deepcopy(base_cp));wrong_org=type(o).restore(copy.deepcopy(org_cp));wrong=ConsequenceQualifiedJointAttentionMemoryV1();st=wrong.stage(wrong_adult,wrong_org,tracker,g,b'that',THAT,*left_pos,CHANNEL,S1)
    wrong_result=wrong.settle_partner_return(wrong_adult,st[0],st[1]+1,Q,True) if st else True
    checks['wrong_selected_response_cannot_claim_episode_return']=(not wrong_result and not wrong.evidence)

    cp=copy.deepcopy(memory.checkpoint());restored=ConsequenceQualifiedJointAttentionMemoryV1.restore(cp);replay=restored.resolve(base,o,b'that again',CHANNEL)
    checks['checkpoint_keeps_consequence_evidence_and_explicitly_empty_pending_ticket_bank']=(replay is not None and not restored.pending and cp.get('pending')==[])
    blob=json.dumps(cp,sort_keys=True)
    checks['checkpoint_contains_only_ids_signed_evidence_and_sources_not_surface_or_transcript']=(all(token not in blob for token in ('that again','tests the sensor','point_y2','point_x2','transcript','expected_answer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-consequence-qualified-joint-attention-memory.v1','contract':'FOUNDRY_CONSEQUENCE_QUALIFIED_JOINT_ATTENTION_MEMORY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'UNNAMED_SHARED_ATTENTION_MEMORY_NOW_REQUIRES_INDEPENDENT_PARTNER_CONSEQUENCE_ON_THE_ACTUAL_PUBLIC_RESPONSE_AND_CAN_REVERSE_THEN_REACQUIRE','conversation':['that again',answer.decode() if answer else ''],'checks':checks,'failed':failed,'remaining_red':['DIRECT_ADULT_OWNERSHIP_OF_JOINT_ATTENTION_MEMORY','DELAYED_PARTNER_RETURN_ACROSS_INTERVENING_TURNS','DIRECT_CONSEQUENCE_QUALIFIED_JOINT_MEMORY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
