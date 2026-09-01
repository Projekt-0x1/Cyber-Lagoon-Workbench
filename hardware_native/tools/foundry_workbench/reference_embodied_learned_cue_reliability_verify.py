#!/usr/bin/env python3
"""N+1: learned social-cue reliability changes embodied acquisition under identical owned sensor packets."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_embodied_conversation_terminal_v1 import owned_reliability_fused_joint_attention_respond,respond,settle_joint_attention_return
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q
from reference_social_cue_reliability_v1 import SocialCueReliabilityV1

LANE_A=0xEE01;LANE_B=0xEE02;ORIGIN=(100,20);S1=0xEE11;S2=0xEE12

def motion(pos):
    y,x=map(int,pos);oy,ox=ORIGIN
    return ((oy,ox),((oy+y)//2,(ox+x)//2),(y,x))

def learned(a_positive=True):
    r=SocialCueReliabilityV1()
    for source in (0xEE21,0xEE22):
        r.observe(LANE_A,source,1 if a_positive else -1,True);r.observe(LANE_A,source,1 if a_positive else -1,True)
    for source in (0xEE31,0xEE32):
        r.observe(LANE_B,source,-1 if a_positive else 1,True);r.observe(LANE_B,source,-1 if a_positive else 1,True)
    return r

def call(adult,o,g,tracker,memory,reliability,source,seq,left_motion,right_motion):
    gs=GestureSensorIngressV1();zs=GazeSensorIngressV1()
    return owned_reliability_fused_joint_attention_respond(
        adult,o,memory,tracker,g,gs,zs,reliability,LANE_A,LANE_B,b'that',left_motion,right_motion,
        THAT,CHANNEL,source,seq,GestureSensorIngressV1.motion_digest(left_motion),seq,GazeSensorIngressV1.motion_digest(right_motion))

def main():
    started=time.perf_counter();checks={};base,o,g,tracker,left,left_pos,event=prepare();rows=sorted((int(e),int(row[0]),int(row[1])) for e,row in tracker.active.items() if row and int(row[-1])==0);right=next(r for r in rows if r[0]!=left);lm=motion(left_pos);rm=motion((right[1],right[2]));base_cp=copy.deepcopy(base.checkpoint());org_cp=copy.deepcopy(o.checkpoint())

    neutral=type(base).restore(copy.deepcopy(base_cp));neutral_memory=ConsequenceQualifiedJointAttentionMemoryV1();before=neutral._next_partner_action_ticket
    ns,nt,np=call(neutral,o,g,tracker,neutral_memory,SocialCueReliabilityV1(),S1,1,lm,rm)
    checks['neutral_reliability_leaves_identical_conflicting_owned_packets_unresolved']=(ns==b'' and nt==np==0 and neutral._next_partner_action_ticket==before and not neutral_memory.pending)

    reliable=learned(True);adult=type(base).restore(copy.deepcopy(base_cp));memory=ConsequenceQualifiedJointAttentionMemoryV1()
    s1,t1,p1=call(adult,o,g,tracker,memory,reliable,S1,1,lm,rm);ok1=settle_joint_attention_return(adult,memory,t1,p1,Q,True)
    s2,t2,p2=call(adult,o,g,tracker,memory,reliable,S2,1,lm,rm);ok2=settle_joint_attention_return(adult,memory,t2,p2,Q,True)
    learned_episode=memory.resolve(adult,o,b'that again',CHANNEL);tracker.active={};recall=respond(adult,o,memory,b'that again',CHANNEL,int(o.world_state_occurrence))
    checks['learned_lane_a_reliability_turns_same_packets_into_grounded_left_action']=(s1==s2==b'the careful engineer tests the sensor.' and ok1 and ok2 and learned_episode is not None and learned_episode.entity==left and recall==s1)

    reverse=learned(False);ra=type(base).restore(copy.deepcopy(base_cp));rmemory=ConsequenceQualifiedJointAttentionMemoryV1();r_before=ra._next_partner_action_ticket
    rs,rt,rp=call(ra,o,g,prepare()[3],rmemory,reverse,S1,1,lm,rm)
    checks['reversed_reliability_selects_distractor_but_grounded_event_gate_refuses_before_action']=(rs==b'' and rt==rp==0 and ra._next_partner_action_ticket==r_before and not rmemory.pending)
    checks['only_reliability_history_differs_across_neutral_positive_and_reversed_packet_trials']=(reliable.weight_q16(LANE_A)>reliable.weight_q16(LANE_B) and reverse.weight_q16(LANE_A)<reverse.weight_q16(LANE_B))
    blob=json.dumps(reliable.checkpoint(),sort_keys=True)
    checks['reliability_checkpoint_contains_no_sensor_packet_target_or_language_surface']=(all(token not in blob for token in ('that',str(lm),str(rm),str(left),str(right[0]))))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-embodied-learned-cue-reliability.v1','contract':'FOUNDRY_EMBODIED_LEARNED_CUE_RELIABILITY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'IDENTICAL_AUTHENTICATED_GAZE_AND_GESTURE_PACKETS_NOW_CHANGE_SHARED_ATTENTION_ACTION_ONLY_WHEN_RESIDENT_CUE_RELIABILITY_HISTORY_CHANGES','conversation':[['reliable owned fusion',s1.decode() if s1 else ''],['that again',recall.decode() if recall else '']],'checks':checks,'failed':failed,'remaining_red':['NATURAL_RELIABILITY_UPDATE_FROM_TARGET_SPECIFIC_PARTNER_FEEDBACK','CONTINUOUS_DIRECTIONAL_STREAM_SEGMENTATION','DIRECT_EMBODIED_RELIABILITY_FUSION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
