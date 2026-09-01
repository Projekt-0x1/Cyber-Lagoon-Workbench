#!/usr/bin/env python3
"""N+1: authenticated ordered gesture ingress owns multimodal shared-attention acquisition before Adult action."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_embodied_conversation_terminal_v1 import owned_joint_attention_respond,respond,settle_joint_attention_return
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q

S1=0xEC11;S2=0xEC12;ORIGIN=(5,0)
def motion_to(pos):
    y,x=map(int,pos);oy,ox=ORIGIN
    return ((oy,ox),((oy+y)//2,(ox+x)//2),(y,x))

def acquire(adult,o,g,tracker,memory,sensor,source,sequence,motion):
    digest=GestureSensorIngressV1.motion_digest(motion)
    speech,ticket,plan=owned_joint_attention_respond(adult,o,memory,tracker,g,sensor,b'that',motion,THAT,CHANNEL,source,sequence,digest)
    settled=settle_joint_attention_return(adult,memory,ticket,plan,Q,True) if ticket else False
    return speech,ticket,plan,settled

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1();sensor=GestureSensorIngressV1();motion=motion_to(left_pos)
    speech1,t1,p1,ok1=acquire(adult,o,g,tracker,memory,sensor,S1,1,motion)
    checks['first_owned_gesture_contact_reaches_normal_adult_action_and_return']=(speech1==b'the careful engineer tests the sensor.' and t1>0 and p1>0 and ok1)
    checks['one_authenticated_gesture_source_stays_below_memory_quorum']=(memory.resolve(adult,o,b'that again',CHANNEL) is None)

    # Duplicate replay must fail before changing gesture cursor, Adult or memory.
    gcp=copy.deepcopy(sensor.checkpoint());acp=copy.deepcopy(adult.checkpoint());mcp=copy.deepcopy(memory.checkpoint());replay_refused=False
    try:owned_joint_attention_respond(adult,o,memory,tracker,g,sensor,b'that',motion,THAT,CHANNEL,S1,1,GestureSensorIngressV1.motion_digest(motion))
    except ValueError:replay_refused=True
    checks['duplicate_gesture_replay_refuses_atomically_before_adult_or_memory_mutation']=(replay_refused and sensor.checkpoint()==gcp and adult.checkpoint()==acp and memory.checkpoint()==mcp)

    # Forged digest on a fresh sequence also refuses atomically.
    forged_refused=False
    try:owned_joint_attention_respond(adult,o,memory,tracker,g,sensor,b'that',motion,THAT,CHANNEL,S1,2,'0'*64)
    except ValueError:forged_refused=True
    checks['forged_gesture_digest_refuses_atomically']=(forged_refused and sensor.checkpoint()==gcp and adult.checkpoint()==acp and memory.checkpoint()==mcp)

    # A second independent authenticated social source establishes quorum.
    speech2,t2,p2,ok2=acquire(adult,o,g,tracker,memory,sensor,S2,1,motion)
    learned=memory.resolve(adult,o,b'that again',CHANNEL)
    tracker.active={};recall=respond(adult,o,memory,b'that again',CHANNEL,int(o.world_state_occurrence))
    checks['second_authenticated_independent_gesture_source_installs_later_recall']=(ok2 and t2>t1 and learned is not None and learned.entity==left and recall==speech1==speech2)

    # Source withdrawal prevents future gesture ingress but does not erase already earned memory.
    sensor.withdraw_source(S2);withdraw_before=(copy.deepcopy(adult.checkpoint()),copy.deepcopy(memory.checkpoint()),copy.deepcopy(sensor.checkpoint()));withdraw_refused=False
    try:owned_joint_attention_respond(adult,o,memory,tracker,g,sensor,b'that',motion,THAT,CHANNEL,S2,2,GestureSensorIngressV1.motion_digest(motion))
    except ValueError:withdraw_refused=True
    checks['withdrawn_gesture_source_cannot_reenter_or_mutate_cognition']=(withdraw_refused and withdraw_before==(adult.checkpoint(),memory.checkpoint(),sensor.checkpoint()))
    checks['earned_memory_survives_gesture_source_withdrawal_as_separate_causal_state']=(memory.resolve(adult,o,b'that again',CHANNEL) is not None)

    cp=copy.deepcopy(sensor.checkpoint());restored=GestureSensorIngressV1.restore(cp)
    checks['gesture_checkpoint_preserves_provenance_but_not_raw_motion']=(restored.last_sequence==sensor.last_sequence and restored.withdrawn_sources==sensor.withdrawn_sources and restored.current_motion is None and restored.active_source==0)
    blob=json.dumps(cp,sort_keys=True)
    checks['gesture_checkpoint_contains_no_coordinates_raw_motion_or_deictic_text']=(all(token not in blob for token in ('current_motion','that',str(motion),'coordinate')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-owned-gesture-embodied-acquisition.v1','contract':'FOUNDRY_OWNED_GESTURE_EMBODIED_ACQUISITION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'AUTHENTICATED_ORDERED_GESTURE_SENSOR_CONTACT_NOW_OWNS_THE_HAND_MOTION_THAT_ACQUIRES_SHARED_ATTENTION_MEMORY_BEFORE_ADULT_ACTION','conversation':[['owned that+gesture',speech1.decode() if speech1 else ''],['that again',recall.decode() if recall else '']],'checks':checks,'failed':failed,'remaining_red':['GAZE_SENSOR_OWNERSHIP_AND_MULTIMODAL_FUSION_IN_SESSION','MULTI_SAMPLE_CONTINUOUS_GESTURE_SEGMENTATION','DIRECT_OWNED_GESTURE_ACQUISITION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
