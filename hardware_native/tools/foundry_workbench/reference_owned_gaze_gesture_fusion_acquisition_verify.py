#!/usr/bin/env python3
"""N+1: separately authenticated gesture + gaze streams atomically fuse before shared-attention acquisition."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_embodied_conversation_terminal_v1 import owned_fused_joint_attention_respond,respond,settle_joint_attention_return
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q

S1=0xED11;S2=0xED12;ORIGIN=(5,0)
def motion_to(pos):
    y,x=map(int,pos);oy,ox=ORIGIN
    return ((oy,ox),((oy+y)//2,(ox+x)//2),(y,x))

def acquire(adult,o,g,tracker,memory,gesture,gaze,source,seq,motion):
    gs=GestureSensorIngressV1.motion_digest(motion);zs=GazeSensorIngressV1.motion_digest(motion)
    speech,ticket,plan=owned_fused_joint_attention_respond(adult,o,memory,tracker,g,gesture,gaze,b'that',motion,motion,THAT,CHANNEL,source,seq,gs,seq,zs)
    settled=settle_joint_attention_return(adult,memory,ticket,plan,Q,True) if ticket else False
    return speech,ticket,plan,settled

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1();gesture=GestureSensorIngressV1();gaze=GazeSensorIngressV1();motion=motion_to(left_pos)
    speech1,t1,p1,ok1=acquire(adult,o,g,tracker,memory,gesture,gaze,S1,1,motion)
    checks['owned_gesture_and_gaze_jointly_reach_one_adult_action_occurrence']=(speech1==b'the careful engineer tests the sensor.' and t1>0 and p1>0 and ok1 and gesture.last_sequence.get(S1)==gaze.last_sequence.get(S1)==1)
    checks['one_multimodal_social_source_remains_below_memory_quorum']=(memory.resolve(adult,o,b'that again',CHANNEL) is None)

    # Both previews must succeed before either stream commits. Gesture packet is valid,
    # gaze packet has forged digest: no provenance or cognition may move.
    gcp=copy.deepcopy(gesture.checkpoint());zcp=copy.deepcopy(gaze.checkpoint());acp=copy.deepcopy(adult.checkpoint());mcp=copy.deepcopy(memory.checkpoint());forged=False
    try:
        owned_fused_joint_attention_respond(adult,o,memory,tracker,g,gesture,gaze,b'that',motion,motion,THAT,CHANNEL,S1,2,GestureSensorIngressV1.motion_digest(motion),2,'0'*64)
    except ValueError:forged=True
    checks['forged_gaze_packet_cannot_partially_advance_valid_gesture_or_cognition']=(forged and gesture.checkpoint()==gcp and gaze.checkpoint()==zcp and adult.checkpoint()==acp and memory.checkpoint()==mcp)

    # Duplicate gaze replay with a fresh valid gesture preview is equally atomic.
    replay=False
    try:
        owned_fused_joint_attention_respond(adult,o,memory,tracker,g,gesture,gaze,b'that',motion,motion,THAT,CHANNEL,S1,2,GestureSensorIngressV1.motion_digest(motion),1,GazeSensorIngressV1.motion_digest(motion))
    except ValueError:replay=True
    checks['gaze_replay_refusal_is_atomic_across_both_directional_streams']=(replay and gesture.checkpoint()==gcp and gaze.checkpoint()==zcp and adult.checkpoint()==acp and memory.checkpoint()==mcp)

    speech2,t2,p2,ok2=acquire(adult,o,g,tracker,memory,gesture,gaze,S2,1,motion)
    learned=memory.resolve(adult,o,b'that again',CHANNEL);tracker.active={};recall=respond(adult,o,memory,b'that again',CHANNEL,int(o.world_state_occurrence))
    checks['second_independent_owned_multimodal_source_installs_later_text_recall']=(ok2 and t2>t1 and learned is not None and learned.entity==left and recall==speech1==speech2)

    # Invalid static gaze is a sensory rejection, not a memory event. Sensor owner accepts
    # shape/order, but fusion rejects it before any Adult action ticket is minted.
    static=(ORIGIN,ORIGIN,ORIGIN);sa=type(adult).restore(copy.deepcopy(adult.checkpoint()));sm=ConsequenceQualifiedJointAttentionMemoryV1();sg=GestureSensorIngressV1();sz=GazeSensorIngressV1();before=sa._next_partner_action_ticket
    ss,st,sp=owned_fused_joint_attention_respond(sa,o,sm,prepare()[3],g,sg,sz,b'that',motion,static,THAT,CHANNEL,0xED21,1,GestureSensorIngressV1.motion_digest(motion),1,GazeSensorIngressV1.motion_digest(static))
    checks['static_gaze_plus_valid_gesture_cannot_create_shared_attention_action']=(ss==b'' and st==sp==0 and sa._next_partner_action_ticket==before and not sm.pending)

    gblob=json.dumps(gesture.checkpoint(),sort_keys=True);zblob=json.dumps(gaze.checkpoint(),sort_keys=True)
    rg=GestureSensorIngressV1.restore(copy.deepcopy(gesture.checkpoint()));rz=GazeSensorIngressV1.restore(copy.deepcopy(gaze.checkpoint()))
    checks['both_directional_checkpoints_keep_provenance_not_raw_motion']=(rg.current_motion is None and rz.current_motion is None and all(token not in gblob+zblob for token in ('current_motion','that',str(motion))))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-owned-gaze-gesture-fusion-acquisition.v1','contract':'FOUNDRY_OWNED_GAZE_GESTURE_FUSION_ACQUISITION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'SEPARATELY_AUTHENTICATED_GAZE_AND_GESTURE_STREAMS_NOW_COMMIT_ATOMICALLY_THEN_FUSE_BEFORE_ADULT_SHARED_ATTENTION_ACTION_AND_MEMORY','conversation':[['owned gaze+gesture+that',speech1.decode() if speech1 else ''],['that again',recall.decode() if recall else '']],'checks':checks,'failed':failed,'remaining_red':['LEARNED_MODALITY_RELIABILITY_IN_EMBODIED_ACQUISITION','CONTINUOUS_DIRECTIONAL_STREAM_SEGMENTATION','DIRECT_OWNED_MULTIMODAL_FUSION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
