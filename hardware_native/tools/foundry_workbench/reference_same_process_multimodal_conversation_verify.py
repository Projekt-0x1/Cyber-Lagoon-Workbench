#!/usr/bin/env python3
"""N+1: one physical process discovers a visual individual, acquires shared attention through gaze+gesture, then recalls it by text."""
from __future__ import annotations
import json,subprocess,sys,tempfile,time
from pathlib import Path
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_body_event_ingress_v1 import BodyEventIngressV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_continuous_visual_sensor_ownership_verify import A1,A2,RAW_ADJ,RAW_TEST,RAW_SENSOR,train_level1
from reference_embodied_multimodal_process_v1 import EmbodiedMultimodalRuntimeV1
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

PROCESS=Path(__file__).with_name('reference_embodied_multimodal_process_v1.py')
VSRC=0xFA01;S1=0xFA11;S2=0xFA12;WORLD_SOURCE=0xFA21;TEXT_SOURCE=0xFA31;ORIGIN=(3,0)

def motion_to(y,x):return (ORIGIN,((ORIGIN[0]+y)//2,(ORIGIN[1]+x)//2),(y,x))
def send(proc,row,expect=False):
    proc.stdin.write((json.dumps(row,separators=(',',':'))+'\n').encode());proc.stdin.flush()
    if not expect:return None
    line=proc.stdout.readline()
    if not line:raise RuntimeError('same_process:no_output')
    return json.loads(line)
def visual(source,seq,frame):return {'kind':'visual','source':source,'sequence':seq,'frame':[list(r) for r in frame],'digest':VisualSensorIngressV1.frame_digest(frame)}
def joint(source,motion,world_sequence):
    text_payload={'text':'that','channel':CHANNEL,'marker_feature':THAT};world_payload={'world_features':[RAW_ADJ,RAW_TEST,RAW_SENSOR]}
    return {'kind':'joint','source':source,'gesture_sequence':1,'gaze_sequence':1,'gesture':[list(x) for x in motion],'gaze':[list(x) for x in motion],'gesture_digest':GestureSensorIngressV1.motion_digest(motion),'gaze_digest':GazeSensorIngressV1.motion_digest(motion),'text':'that','marker_feature':THAT,'channel':CHANNEL,
            'text_source':source,'text_sequence':1,'text_digest':BodyEventIngressV1.payload_digest('text',text_payload),
            'world_source':WORLD_SOURCE,'world_sequence':world_sequence,'world_digest':BodyEventIngressV1.payload_digest('world',world_payload),'world_features':[RAW_ADJ,RAW_TEST,RAW_SENSOR]}
def text_event(text,sequence):
    payload={'text':text,'channel':CHANNEL}
    return {'kind':'text','text':text,'channel':CHANNEL,'source':TEXT_SOURCE,'sequence':sequence,'digest':BodyEventIngressV1.payload_digest('text',payload)}
def return_event(source,ticket,plan,sequence):
    payload={'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True}
    return {'kind':'return','source':int(source),'sequence':int(sequence),'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True,'digest':BodyEventIngressV1.payload_digest('return',payload)}

def main():
    started=time.perf_counter();checks={};adult,o,g,_tracker,_left,_left_pos,_event=prepare();level1=train_level1();temporal=train_temporal(level1)
    runtime=EmbodiedMultimodalRuntimeV1(adult,o,ConsequenceQualifiedJointAttentionMemoryV1(),g,level1,temporal,VisualSensorIngressV1(),GestureSensorIngressV1(),GazeSensorIngressV1(),int(o.world_state_occurrence))
    initial_entities=set(map(int,o.entity_features));blank=tuple(tuple(0 for _ in row) for row in A1);target_motion=motion_to(5,5)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'runtime.json';path.write_text(json.dumps(runtime.checkpoint(),sort_keys=True))
        proc=subprocess.Popen([sys.executable,str(PROCESS),'--resume',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            send(proc,visual(VSRC,1,A1));send(proc,visual(VSRC,2,A2))
            a1=send(proc,joint(S1,target_motion,1),True)
            r1=send(proc,return_event(S1,a1['ticket'],a1['plan'],1),True)
            a2=send(proc,joint(S2,target_motion,2),True)
            r2=send(proc,return_event(S2,a2['ticket'],a2['plan'],1),True)
            send(proc,visual(VSRC,3,blank));send(proc,visual(VSRC,4,blank))
            recall=send(proc,text_event('that again',1),True)
            proc.stdin.close();proc.wait(timeout=5);stderr=proc.stderr.read()
        finally:
            if proc.poll() is None:proc.kill()
        saved=json.loads(path.read_text());saved_runtime=EmbodiedMultimodalRuntimeV1.restore(saved);new_entities=set(map(int,saved_runtime.organism.entity_features))-initial_entities
        checks['same_process_raw_visual_sequence_mints_new_internal_individual_without_protocol_entity_id']=(len(new_entities)>=1 and 'entity' not in joint(S1,target_motion,1) and 'entity' not in visual(VSRC,1,A1))
        checks['two_owned_multimodal_joint_events_emit_same_grounded_public_response_with_distinct_tickets']=(a1['speech']==a2['speech']=='the careful engineer tests the sensor.' and a1['ticket']>0 and a2['ticket']>a1['ticket'] and r1==r2=={'settled':True})
        checks['after_two_blank_frames_same_process_text_only_recall_uses_acquired_memory']=(recall['speech']=='the careful engineer tests the sensor.' and recall['ticket']==0 and recall['plan']==0)
        checks['process_protocol_stderr_is_empty_and_exits_cleanly']=(proc.returncode==0 and stderr==b'')
        checks['saved_checkpoint_has_no_active_object_file_or_raw_sensor_samples']=(saved_runtime.tracker.active=={} and saved_runtime.visual.current_frame is None and saved_runtime.gesture.current_motion is None and saved_runtime.gaze.current_motion is None)
        checks['durable_memory_survives_process_exit_without_live_visual_tracker']=(saved_runtime.memory.resolve(saved_runtime.adult,saved_runtime.organism,b'that again',CHANNEL) is not None)

        # Restart from durable file: no visual bootstrap on second process, text recall still works.
        again=subprocess.run([sys.executable,str(PROCESS),'--resume',str(path)],input=(json.dumps(text_event('that again',2))+'\n').encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=True)
        rows=[json.loads(line) for line in again.stdout.splitlines()]
        checks['restart_without_visual_frames_still_recalls_prior_unnamed_shared_attention_event']=(len(rows)==1 and rows[0]['speech']=='the careful engineer tests the sensor.' and again.stderr==b'')
        blob=path.read_text();checks['checkpoint_protocol_has_no_raw_frame_gesture_gaze_or_transcript_fields']=(all(token not in blob for token in ('current_frame','current_motion','tracker','transcript','that again')))
    checks['bounded_fast_path']=time.perf_counter()-started<3.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-same-process-multimodal-conversation.v1','contract':'FOUNDRY_SAME_PROCESS_MULTIMODAL_CONVERSATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'ONE_PERSISTENT_BODY_PROCESS_NOW_DISCOVERS_A_VISUAL_INDIVIDUAL_ACQUIRES_IT_THROUGH_AUTHENTICATED_GAZE_GESTURE_AND_DEMONSTRATIVE_CONTACT_THEN_RECALLS_IT_FROM_RAW_TEXT_AFTER_VISUAL_DISAPPEARANCE_AND_RESTART','conversation':[['joint',a1.get('speech','')],['that again',recall.get('speech','')]],'checks':checks,'failed':failed,'remaining_red':['CONTINUOUS_STREAM_GESTURE_SEGMENTATION','DIRECT_SAME_PROCESS_MULTIMODAL_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
