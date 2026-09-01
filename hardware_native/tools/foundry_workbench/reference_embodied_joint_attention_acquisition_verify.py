#!/usr/bin/env python3
"""N+1: embodied session acquires unnamed shared-attention memory from raw demonstrative text + raw hand motion."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_embodied_conversation_terminal_v1 import joint_attention_respond,respond,settle_joint_attention_return
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q
from reference_raw_pointing_motion_v1 import RawPointingMotionV1

S1=0xEB11;S2=0xEB12;ORIGIN=(5,0)

def motion_to(pos):
    y,x=map(int,pos);oy,ox=ORIGIN
    return ((oy,ox),((oy+y)//2,(ox+x)//2),(y,x))

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1()
    visible=sorted((int(e),int(row[0]),int(row[1])) for e,row in tracker.active.items() if row and int(row[-1])==0)
    right=next(row for row in visible if row[0]!=left);right_pos=(right[1],right[2])
    left_motion=motion_to(left_pos);right_motion=motion_to(right_pos);static=(ORIGIN,ORIGIN,ORIGIN)
    checks['raw_left_motion_derives_expected_visual_target_without_host_entity_input']=(RawPointingMotionV1.target(tracker,left_motion)==left)

    speech1,t1,p1=joint_attention_respond(adult,o,memory,tracker,g,b'that',left_motion,THAT,CHANNEL,S1)
    pre1=memory.resolve(adult,o,b'that again',CHANNEL)
    settled1=settle_joint_attention_return(adult,memory,t1,p1,Q,True)
    after1=memory.resolve(adult,o,b'that again',CHANNEL)
    speech2,t2,p2=joint_attention_respond(adult,o,memory,tracker,g,b'that',left_motion,THAT,CHANNEL,S2)
    settled2=settle_joint_attention_return(adult,memory,t2,p2,Q,True)
    learned=memory.resolve(adult,o,b'that again',CHANNEL)
    checks['raw_multimodal_contact_stages_real_adult_action_ticket_but_not_memory_before_return']=(bool(speech1) and t1>0 and p1>0 and pre1 is None)
    checks['one_return_stays_below_quorum_second_independent_return_installs_memory']=(settled1 and after1 is None and settled2 and learned is not None and learned.entity==left)
    checks['both_acquisition_responses_use_same_grounded_event_surface']=(speech1==speech2==b'the careful engineer tests the sensor.' and t2>t1)

    tracker.active={};recall=respond(adult,o,memory,b'that again',CHANNEL,int(o.world_state_occurrence))
    checks['later_raw_text_recall_uses_memory_acquired_only_through_multimodal_ingress']=(recall==b'the careful engineer tests the sensor.')

    # Wrong visible target conflicts with current world event membership and cannot even mint a response ticket.
    wrong_adult=type(adult).restore(copy.deepcopy(adult.checkpoint()));wrong_memory=ConsequenceQualifiedJointAttentionMemoryV1();tracker2=copy.deepcopy(tracker)
    # Restore visible tracker from pre-clear fixture by using the original current rows saved above.
    tracker2.active={e:copy.deepcopy(row) for e,row in prepare()[3].active.items()}
    wrong_before=int(wrong_adult._next_partner_action_ticket)
    wrong_speech,wrong_ticket,wrong_plan=joint_attention_respond(wrong_adult,o,wrong_memory,tracker2,g,b'that',right_motion,THAT,CHANNEL,0xEB21)
    checks['gesture_to_visible_distractor_not_in_current_event_refuses_before_action_ticket']=(wrong_speech==b'' and wrong_ticket==0 and wrong_plan==0 and wrong_adult._next_partner_action_ticket==wrong_before and not wrong_memory.pending)

    static_adult=type(adult).restore(copy.deepcopy(adult.checkpoint()));static_memory=ConsequenceQualifiedJointAttentionMemoryV1();static_before=static_adult._next_partner_action_ticket
    ss,st,sp=joint_attention_respond(static_adult,o,static_memory,tracker2,g,b'that',static,THAT,CHANNEL,0xEB22)
    checks['static_hand_plus_deictic_text_refuses_before_any_action_or_memory_eligibility']=(ss==b'' and st==sp==0 and static_adult._next_partner_action_ticket==static_before and not static_memory.pending)

    # Yoked return from otherwise valid multimodal contact cannot install source support.
    yoked_adult=type(adult).restore(copy.deepcopy(adult.checkpoint()));yoked_memory=ConsequenceQualifiedJointAttentionMemoryV1();ys,yt,yp=joint_attention_respond(yoked_adult,o,yoked_memory,tracker2,g,b'that',left_motion,THAT,CHANNEL,0xEB31);ysettle=settle_joint_attention_return(yoked_adult,yoked_memory,yt,yp,Q,False)
    checks['yoked_partner_return_cannot_turn_valid_multimodal_contact_into_memory_evidence']=(bool(ys) and not ysettle and not yoked_memory.evidence)

    blob=json.dumps(memory.checkpoint(),sort_keys=True)
    checks['memory_checkpoint_contains_no_gesture_path_raw_demonstrative_or_visual_track']=(all(token not in blob for token in ('that','point','trajectory','current_frame',str(left_motion))))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-embodied-joint-attention-acquisition.v1','contract':'FOUNDRY_EMBODIED_JOINT_ATTENTION_ACQUISITION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'RAW_DEMONSTRATIVE_TEXT_PLUS_RAW_HAND_MOTION_NOW_ACQUIRES_AN_UNNAMED_SHARED_ATTENTION_MEMORY_THROUGH_ADULT_ACTION_AND_PARTNER_RETURN_WITH_NO_HOST_TARGET','conversation':[['that+point',speech1.decode() if speech1 else ''],['that again',recall.decode() if recall else '']],'checks':checks,'failed':failed,'remaining_red':['AUTHENTICATED_GESTURE_STREAM_OWNERSHIP_AND_SEQUENCE','GAZE_PLUS_TEXT_ACQUISITION_IN_SAME_SESSION','DIRECT_EMBODIED_MULTIMODAL_ACQUISITION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
