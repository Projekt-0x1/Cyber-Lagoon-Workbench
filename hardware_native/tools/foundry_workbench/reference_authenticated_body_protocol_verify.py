#!/usr/bin/env python3
"""N+1: text/world body records have owned source/order/integrity and joint multimodal validation is atomic."""
from __future__ import annotations
import copy,json,time
from reference_body_event_ingress_v1 import BodyEventIngressV1
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_continuous_visual_sensor_ownership_verify import A1,A2,RAW_ADJ,RAW_TEST,RAW_SENSOR,train_level1
from reference_embodied_memory_conversation_verify import install_memory
from reference_embodied_multimodal_process_v1 import EmbodiedMultimodalRuntimeV1
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

VSRC=0xFB01;DSRC=0xFB11;TSRC=0xFB21;WSRC=0xFB31;ORIGIN=(3,0)
def motion_to(y,x):return (ORIGIN,((ORIGIN[0]+y)//2,(ORIGIN[1]+x)//2),(y,x))
def visual(seq,frame):return {'kind':'visual','source':VSRC,'sequence':seq,'frame':[list(r) for r in frame],'digest':VisualSensorIngressV1.frame_digest(frame)}
def joint(motion,text_digest=None,world_digest=None):
    tp={'text':'that','channel':CHANNEL,'marker_feature':THAT};wp={'world_features':[RAW_ADJ,RAW_TEST,RAW_SENSOR]}
    return {'kind':'joint','source':DSRC,'gesture_sequence':1,'gaze_sequence':1,'gesture':[list(x) for x in motion],'gaze':[list(x) for x in motion],
            'gesture_digest':GestureSensorIngressV1.motion_digest(motion),'gaze_digest':GazeSensorIngressV1.motion_digest(motion),
            'text':'that','marker_feature':THAT,'channel':CHANNEL,'text_source':TSRC,'text_sequence':1,'text_digest':text_digest or BodyEventIngressV1.payload_digest('text',tp),
            'world_source':WSRC,'world_sequence':1,'world_digest':world_digest or BodyEventIngressV1.payload_digest('world',wp),'world_features':wp['world_features']}
def text_event(text,seq,digest=None):
    payload={'text':text,'channel':CHANNEL}
    return {'kind':'text','text':text,'channel':CHANNEL,'source':TSRC,'sequence':seq,'digest':digest or BodyEventIngressV1.payload_digest('text',payload)}
def snapshot(rt):return copy.deepcopy(rt.checkpoint())

def main():
    started=time.perf_counter();checks={};adult,o,g,_tracker,left,left_pos,event=prepare();level1=train_level1();temporal=train_temporal(level1)
    rt=EmbodiedMultimodalRuntimeV1(adult,o,ConsequenceQualifiedJointAttentionMemoryV1(),g,level1,temporal,VisualSensorIngressV1(),GestureSensorIngressV1(),GazeSensorIngressV1(),int(o.world_state_occurrence))
    rt.event(visual(1,A1));rt.event(visual(2,A2));visible=sorted((e,row[0],row[1]) for e,row in rt.tracker.active.items() if row and int(row[-1])==0);target=visible[0];motion=motion_to(target[1],target[2])

    before=snapshot(rt);forged_text=False
    try:rt.event(joint(motion,text_digest='0'*64))
    except ValueError:forged_text=True
    checks['forged_joint_text_digest_refuses_before_any_lane_or_cognition_mutates']=(forged_text and snapshot(rt)==before)

    before=snapshot(rt);forged_world=False
    try:rt.event(joint(motion,world_digest='0'*64))
    except ValueError:forged_world=True
    checks['forged_joint_world_digest_refuses_atomically_across_text_gesture_gaze_and_world']=(forged_world and snapshot(rt)==before)

    # A valid authenticated joint contact commits all four provenance lanes together.
    result=rt.event(joint(motion));after_joint=snapshot(rt)
    checks['valid_joint_contact_commits_text_world_gesture_and_gaze_provenance_together']=(
        result is not None and result['ticket']>0
        and rt.body.last_sequence.get(('text',TSRC))==1 and rt.body.last_sequence.get(('world',WSRC))==1
        and rt.gesture.last_sequence.get(DSRC)==1 and rt.gaze.last_sequence.get(DSRC)==1)

    # Use a separate durable-memory runtime to test authenticated text replay without
    # depending on the still-unsettled joint action above.
    ta,to,tg,ttracker,tleft,tpos,tevent=prepare();tm=ConsequenceQualifiedJointAttentionMemoryV1();install_memory(ta,to,tg,ttracker,tpos,tm);ttracker.active={}
    trt=EmbodiedMultimodalRuntimeV1(ta,to,tm,tg,level1,temporal,VisualSensorIngressV1(),GestureSensorIngressV1(),GazeSensorIngressV1(),int(to.world_state_occurrence))
    first=trt.event(text_event('that again',1));post_first=snapshot(trt)
    replay=False
    try:trt.event(text_event('that again',1))
    except ValueError:replay=True
    checks['authenticated_text_duplicate_sequence_refuses_before_conversation_state_changes']=(replay and snapshot(trt)==post_first and first['speech']=='the careful engineer tests the sensor.')

    forged=False
    try:trt.event(text_event('that again',2,'f'*64))
    except ValueError:forged=True
    checks['authenticated_text_forgery_on_fresh_sequence_refuses_atomically']=(forged and snapshot(trt)==post_first)

    cp=copy.deepcopy(trt.checkpoint());restored=EmbodiedMultimodalRuntimeV1.restore(cp)
    checks['text_and_world_provenance_cursors_survive_checkpoint_without_payloads']=(restored.body.last_sequence==trt.body.last_sequence and cp['body']['last_sequence']==[{'lane':'text','source':TSRC,'sequence':1}])
    blob=json.dumps(cp['body'],sort_keys=True)
    checks['body_envelope_checkpoint_contains_no_text_world_features_or_digest_payload']=(all(token not in blob for token in ('that again','world_features','careful engineer','digest')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-authenticated-body-protocol.v1','contract':'FOUNDRY_AUTHENTICATED_BODY_PROTOCOL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'visible_language_gain':'TEXT_AND_WORLD_CONTACTS_NOW_HAVE_THE_SAME_AUTHENTICATED_SOURCE_SEQUENCE_INTEGRITY_OWNERSHIP_AS_VISUAL_GAZE_AND_GESTURE_AND_MULTIMODAL_JOINT_VALIDATION_IS_ATOMIC',
            'checks':checks,'failed':failed,'remaining_red':['CONTINUOUS_STREAM_GESTURE_SEGMENTATION','DIRECT_AUTHENTICATED_BODY_PROTOCOL_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
