#!/usr/bin/env python3
"""JSON-line physical event runtime for continuous embodied visual/social/language conversation."""
from __future__ import annotations
import argparse,json,sys,tempfile,os
from pathlib import Path
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_continuous_gesture_segmenter_v1 import ContinuousGestureSegmenterV1
from reference_body_event_ingress_v1 import BodyEventIngressV1
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_embodied_conversation_terminal_v1 import respond,settle_joint_attention_return
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_social_directional_cue_fusion_v1 import SocialDirectionalCueFusionV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_unsupervised_perceptual_features_v1 import UnsupervisedPerceptualFeaturesV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

class EmbodiedMultimodalRuntimeV1:
    def __init__(self,adult,organism,memory,grounding,level1,temporal,visual,gesture,gaze,last_world=0,body=None,gesture_segmenter=None):
        self.adult=adult;self.organism=organism;self.memory=memory;self.grounding=grounding;self.level1=level1;self.temporal=temporal
        self.visual=visual;self.gesture=gesture;self.gaze=gaze;self.body=body if body is not None else BodyEventIngressV1();self.gesture_segmenter=gesture_segmenter if gesture_segmenter is not None else ContinuousGestureSegmenterV1();self.completed_gesture={};self.tracker=MultiVisualObjectFileTrackerV1();self.last_world=max(0,int(last_world))

    def checkpoint(self):
        self.memory.synchronize_pending(self.adult)
        return {'schema':3,'adult':self.adult.checkpoint(),'organism':self.organism.checkpoint(),'joint_memory':self.memory.checkpoint(),
                'grounding':self.grounding.checkpoint(),'level1':self.level1.checkpoint(),'temporal':self.temporal.checkpoint(),
                'visual':self.visual.checkpoint(),'gesture':self.gesture.checkpoint(),'gaze':self.gaze.checkpoint(),'body':self.body.checkpoint(),'gesture_segmenter':self.gesture_segmenter.checkpoint(),'last_world':self.last_world}

    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2,3):raise ValueError('embodied_multimodal:checkpoint')
        adult=LanguageMasteryAdultV1.restore(data['adult']);organism=ReferenceOrganismV2.restore(data['organism']);memory=ConsequenceQualifiedJointAttentionMemoryV1.restore(data['joint_memory']);memory.synchronize_pending(adult)
        body=BodyEventIngressV1.restore(data['body']) if schema>=2 else BodyEventIngressV1();segmenter=ContinuousGestureSegmenterV1.restore(data['gesture_segmenter']) if schema>=3 else ContinuousGestureSegmenterV1()
        return cls(adult,organism,memory,CrossmodalConceptGroundingV1.restore(data['grounding']),UnsupervisedPerceptualFeaturesV1.restore(data['level1']),TemporalVisualContinuityV1.restore(data['temporal']),VisualSensorIngressV1.restore(data['visual']),GestureSensorIngressV1.restore(data['gesture']),GazeSensorIngressV1.restore(data['gaze']),data.get('last_world',0),body,segmenter)

    def _emit(self,speech,ticket=0,plan=0):
        if not speech:return None
        return {'speech':bytes(speech).decode('utf-8'),'ticket':int(ticket),'plan':int(plan)}

    def event(self,row):
        kind=str(row.get('kind',''))
        if kind=='visual':
            frame=tuple(tuple(int(x) for x in line) for line in row.get('frame',()))
            self.tracker.observe(self.organism,self.visual,self.temporal,self.level1,int(row['source']),int(row['sequence']),frame,str(row['digest']))
            return None
        if kind=='gesture_sample':
            source=int(row['source']);sequence=int(row['sequence']);y=int(row['y']);x=int(row['x']);segment=self.gesture_segmenter.ingest(source,sequence,y,x,str(row['digest']))
            if segment:self.completed_gesture[source]=tuple(segment)
            return None
        if kind=='joint_stream':
            direction_source=int(row['source']);gm=self.completed_gesture.get(direction_source);zm=tuple((int(y),int(x)) for y,x in row.get('gaze',()))
            if not gm:return None
            raw_text=str(row.get('text',''));marker=int(row['marker_feature']);channel=max(0,int(row.get('channel',0)))
            text_source=int(row['text_source']);text_payload={'text':raw_text,'channel':channel,'marker_feature':marker}
            world_features=tuple(int(x) for x in row.get('world_features',()) if int(x)>0);world_source=int(row['world_source']);world_payload={'world_features':list(world_features)}
            if not world_features:return None
            tp=self.body.preview('text',text_source,int(row['text_sequence']),text_payload,str(row['text_digest']));wp=self.body.preview('world',world_source,int(row['world_sequence']),world_payload,str(row['world_digest']));zp=self.gaze.preview(direction_source,int(row['gaze_sequence']),zm,str(row['gaze_digest']))
            self.body.commit(tp);self.body.commit(wp);accepted_gaze,_=self.gaze.commit(zp)
            target=int(SocialDirectionalCueFusionV1.resolve(self.tracker,gm,accepted_gaze));track=self.tracker.active.get(target)
            if target<=0 or not track or int(track[-1])!=0:return None
            self.organism.contact(CONTACT_WORLD_STATE,(*world_features,target),world_source,True,True);self.last_world=int(self.organism.world_state_occurrence)
            staged=self.memory.stage(self.adult,self.organism,self.tracker,self.grounding,raw_text.encode('utf-8'),marker,int(track[0]),int(track[1]),channel,text_source)
            if staged is None:return None
            self.completed_gesture.pop(direction_source,None)
            ticket,plan,_episode=staged;speech=emit_choice(self.adult,plan)
            return self._emit(speech,ticket,plan)
        if kind=='joint':
            direction_source=int(row['source']);gm=tuple((int(y),int(x)) for y,x in row.get('gesture',()));zm=tuple((int(y),int(x)) for y,x in row.get('gaze',()))
            raw_text=str(row.get('text',''));marker=int(row['marker_feature']);channel=max(0,int(row.get('channel',0)))
            text_source=int(row['text_source']);text_payload={'text':raw_text,'channel':channel,'marker_feature':marker}
            world_features=tuple(int(x) for x in row.get('world_features',()) if int(x)>0);world_source=int(row['world_source']);world_payload={'world_features':list(world_features)}
            if not world_features:return None
            tp=self.body.preview('text',text_source,int(row['text_sequence']),text_payload,str(row['text_digest']))
            wp=self.body.preview('world',world_source,int(row['world_sequence']),world_payload,str(row['world_digest']))
            gp=self.gesture.preview(direction_source,int(row['gesture_sequence']),gm,str(row['gesture_digest']))
            zp=self.gaze.preview(direction_source,int(row['gaze_sequence']),zm,str(row['gaze_digest']))
            self.body.commit(tp);self.body.commit(wp);accepted_gesture,_=self.gesture.commit(gp);accepted_gaze,_=self.gaze.commit(zp)
            target=int(SocialDirectionalCueFusionV1.resolve(self.tracker,accepted_gesture,accepted_gaze));track=self.tracker.active.get(target)
            if target<=0 or not track or int(track[-1])!=0:return None
            self.organism.contact(CONTACT_WORLD_STATE,(*world_features,target),world_source,True,True);self.last_world=int(self.organism.world_state_occurrence)
            raw=raw_text.encode('utf-8')
            staged=self.memory.stage(self.adult,self.organism,self.tracker,self.grounding,raw,marker,int(track[0]),int(track[1]),channel,text_source)
            if staged is None:return None
            ticket,plan,_episode=staged;speech=emit_choice(self.adult,plan)
            return self._emit(speech,ticket,plan)
        if kind=='return':
            ticket=int(row['ticket']);plan=int(row['plan']);outcome=int(row['outcome']);independent=bool(row.get('independent',True));source=int(row['source'])
            self.memory.synchronize_pending(self.adult);pending=self.memory.pending.get(ticket)
            if pending is None:raise ValueError('embodied_multimodal:return_ticket')
            _episode,pending_source,pending_plan,_partner_context=pending
            if int(pending_source)!=source or int(pending_plan)!=plan:raise ValueError('embodied_multimodal:return_lineage')
            payload={'ticket':ticket,'plan':plan,'outcome':outcome,'independent':independent}
            preview=self.body.preview('return',source,int(row['sequence']),payload,str(row['digest']));self.body.commit(preview)
            settled=settle_joint_attention_return(self.adult,self.memory,ticket,plan,outcome,independent)
            return {'settled':bool(settled)}
        if kind=='text':
            raw_text=str(row.get('text',''));channel=max(0,int(row.get('channel',0)));source=int(row['source']);payload={'text':raw_text,'channel':channel}
            preview=self.body.preview('text',source,int(row['sequence']),payload,str(row['digest']));self.body.commit(preview)
            speech=respond(self.adult,self.organism,self.memory,raw_text.encode('utf-8'),channel,self.last_world,source=source)
            return self._emit(speech)
        raise ValueError('embodied_multimodal:event_kind')

def save(path,runtime):
    path=Path(path)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as out:
        json.dump(runtime.checkpoint(),out,separators=(',',':'),sort_keys=True);out.flush();os.fsync(out.fileno());tmp=Path(out.name)
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',required=True,type=Path);args=ap.parse_args();runtime=EmbodiedMultimodalRuntimeV1.restore(json.loads(args.resume.read_text()))
    try:
        for raw in sys.stdin.buffer:
            raw=raw.strip()
            if not raw:continue
            result=runtime.event(json.loads(raw))
            if result is not None:sys.stdout.write(json.dumps(result,separators=(',',':'),sort_keys=True)+'\n');sys.stdout.flush()
    finally:save(args.resume,runtime)
if __name__=='__main__':main()
