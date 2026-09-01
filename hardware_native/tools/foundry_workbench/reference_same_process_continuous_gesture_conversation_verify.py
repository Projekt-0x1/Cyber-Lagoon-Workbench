#!/usr/bin/env python3
"""N+1: same process converts continuous gesture samples into shared-attention acquisition without gesture boundary labels."""
from __future__ import annotations
import copy,json,time
from reference_body_event_ingress_v1 import BodyEventIngressV1
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_continuous_gesture_segmenter_v1 import ContinuousGestureSegmenterV1
from reference_continuous_visual_sensor_ownership_verify import A1,A2,RAW_ADJ,RAW_TEST,RAW_SENSOR,train_level1
from reference_embodied_multimodal_process_v1 import EmbodiedMultimodalRuntimeV1
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

VSRC=0xFD01;S1=0xFD11;S2=0xFD12;WSRC=0xFD21;TSRC=0xFD31;ORIGIN=(3,0)

def visual(seq,frame):return {'kind':'visual','source':VSRC,'sequence':seq,'frame':[list(r) for r in frame],'digest':VisualSensorIngressV1.frame_digest(frame)}
def sample(source,seq,p):
    y,x=map(int,p);return {'kind':'gesture_sample','source':source,'sequence':seq,'y':y,'x':x,'digest':ContinuousGestureSegmenterV1.sample_digest(y,x)}
def line(a,b,steps=3):
    ay,ax=a;by,bx=b;return tuple((round(ay+(by-ay)*i/steps),round(ax+(bx-ax)*i/steps)) for i in range(steps+1))
def gaze_motion(pos):
    y,x=map(int,pos);return (ORIGIN,((ORIGIN[0]+y)//2,(ORIGIN[1]+x)//2),(y,x))
def joint_stream(source,world_seq,pos):
    gaze=gaze_motion(pos);tp={'text':'that','channel':CHANNEL,'marker_feature':THAT};wp={'world_features':[RAW_ADJ,RAW_TEST,RAW_SENSOR]}
    return {'kind':'joint_stream','source':source,'gaze_sequence':1,'gaze':[list(x) for x in gaze],'gaze_digest':GazeSensorIngressV1.motion_digest(gaze),'text':'that','marker_feature':THAT,'channel':CHANNEL,
            'text_source':source,'text_sequence':1,'text_digest':BodyEventIngressV1.payload_digest('text',tp),'world_source':WSRC,'world_sequence':world_seq,'world_digest':BodyEventIngressV1.payload_digest('world',wp),'world_features':wp['world_features']}
def ret(source,ticket,plan):
    payload={'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True}
    return {'kind':'return','source':source,'sequence':1,'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True,'digest':BodyEventIngressV1.payload_digest('return',payload)}
def text_event(seq):
    payload={'text':'that again','channel':CHANNEL};return {'kind':'text','text':'that again','channel':CHANNEL,'source':TSRC,'sequence':seq,'digest':BodyEventIngressV1.payload_digest('text',payload)}
def feed_stroke(rt,source,pos):
    stream=[ORIGIN,ORIGIN,*line(ORIGIN,pos)[1:],pos,pos];out=[]
    for seq,p in enumerate(stream,1):out.append(rt.event(sample(source,seq,p)))
    return out

def main():
    started=time.perf_counter();checks={};adult,o,g,_tracker,_left,_left_pos,_event=prepare();level1=train_level1();temporal=train_temporal(level1)
    rt=EmbodiedMultimodalRuntimeV1(adult,o,ConsequenceQualifiedJointAttentionMemoryV1(),g,level1,temporal,VisualSensorIngressV1(),GestureSensorIngressV1(),GazeSensorIngressV1(),int(o.world_state_occurrence))
    rt.event(visual(1,A1));rt.event(visual(2,A2));visible=sorted((int(e),int(row[0]),int(row[1])) for e,row in rt.tracker.active.items() if row and int(row[-1])==0);target=visible[0];pos=(target[1],target[2])

    outputs=feed_stroke(rt,S1,pos);checks['continuous_sample_events_emit_no_public_output_or_host_boundary_signal']=(all(row is None for row in outputs) and S1 in rt.completed_gesture)
    row1=joint_stream(S1,1,pos);checks['joint_stream_event_contains_no_gesture_trajectory_or_entity_id']=('gesture' not in row1 and 'entity' not in row1 and row1['source']==S1)
    a1=rt.event(row1);r1=rt.event(ret(S1,a1['ticket'],a1['plan']))
    checks['internally_segmented_first_stroke_reaches_grounded_ticketed_response']=(a1['speech']=='the careful engineer tests the sensor.' and r1=={'settled':True} and S1 not in rt.completed_gesture)
    checks['one_segmented_social_source_stays_below_joint_memory_quorum']=(rt.memory.resolve(rt.adult,rt.organism,b'that again',CHANNEL) is None)

    feed_stroke(rt,S2,pos);a2=rt.event(joint_stream(S2,2,pos));r2=rt.event(ret(S2,a2['ticket'],a2['plan']));learned=rt.memory.resolve(rt.adult,rt.organism,b'that again',CHANNEL)
    checks['second_independent_segmented_source_installs_shared_attention_memory']=(a2['speech']==a1['speech'] and r2=={'settled':True} and learned is not None and learned.entity==target[0])

    blank=tuple(tuple(0 for _ in row) for row in A1);rt.event(visual(3,blank));rt.event(visual(4,blank));recall=rt.event(text_event(1))
    checks['after_visual_disappearance_text_only_recall_uses_memory_acquired_from_segmented_streams']=(recall['speech']=='the careful engineer tests the sensor.')

    cp=copy.deepcopy(rt.checkpoint());restored=EmbodiedMultimodalRuntimeV1.restore(cp)
    checks['checkpoint_drops_completed_and_partial_gesture_motion_but_preserves_sample_provenance']=(not restored.completed_gesture and not restored.gesture_segmenter.active and restored.gesture_segmenter.ingress.last_sequence==rt.gesture_segmenter.ingress.last_sequence)
    blob=json.dumps(cp,sort_keys=True)
    checks['checkpoint_contains_no_segment_points_or_boundary_labels']=(all(token not in blob for token in ('completed_gesture','gesture_start','gesture_end','points','trajectory')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-same-process-continuous-gesture-conversation.v1','contract':'FOUNDRY_SAME_PROCESS_CONTINUOUS_GESTURE_CONVERSATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'visible_language_gain':'ONE_CONTINUOUS_AUTHENTICATED_HAND_SAMPLE_STREAM_NOW_SELF_SEGMENTS_INSIDE_THE_RUNNING_BODY_PROCESS_AND_DRIVES_SHARED_ATTENTION_MEMORY_WITHOUT_GESTURE_START_END_LABELS',
            'conversation':[['segmented that',a1.get('speech','')],['that again',recall.get('speech','')]],'checks':checks,'failed':failed,'remaining_red':['LEARNED_GESTURE_STYLE_VARIATION','CONTINUOUS_GAZE_SEGMENTATION','DIRECT_CONTINUOUS_GESTURE_CONVERSATION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
