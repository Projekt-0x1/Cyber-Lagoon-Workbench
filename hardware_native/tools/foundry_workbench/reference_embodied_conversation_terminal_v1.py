#!/usr/bin/env python3
"""Persistent embodied conversation session: Adult + world organism + joint-attention memory."""
from __future__ import annotations
import argparse,json,os,selectors,sys,tempfile
from pathlib import Path
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_organism_v2 import ReferenceOrganismV2
from reference_raw_pointing_motion_v1 import RawPointingMotionV1
from reference_social_directional_cue_fusion_v1 import SocialDirectionalCueFusionV1
from reference_reliability_weighted_directional_fusion_v1 import ReliabilityWeightedDirectionalFusionV1
from reference_resident_conversation_affordance_v1 import ConversationProposalV1,ResidentConversationAffordanceV1

MEMORY_KIND=7

class EmbodiedConversationAffordanceV1:
    @staticmethod
    def remembered(adult,organism,memory,raw,channel,state=AdultStateV1()):
        episode=memory.resolve(adult,organism,raw,channel)
        if episode is None:return ()
        plan=int(adult._select(int(episode.event),state))
        if not plan:return ()
        return (ConversationProposalV1(int(episode.event),plan,MEMORY_KIND,int(episode.entity)),)

    @classmethod
    def proposals(cls,adult,organism,memory,raw,channel,last_world,state=AdultStateV1()):
        rows=list(ResidentConversationAffordanceV1.proposals(adult,organism,raw,channel,last_world,state))
        rows.extend(cls.remembered(adult,organism,memory,raw,channel,state))
        unique={(int(r.context),int(r.plan),int(r.kind),int(r.aux)):r for r in rows if r.context and r.plan}
        return tuple(unique[k] for k in sorted(unique))

    @classmethod
    def activate(cls,adult,organism,memory,raw,channel,last_world,state=AdultStateV1()):
        rows=cls.proposals(adult,organism,memory,raw,channel,last_world,state)
        if len(rows)!=1:return None
        row=rows[0]
        if int(row.kind)!=MEMORY_KIND:
            return ResidentConversationAffordanceV1.activate(adult,organism,raw,channel,last_world,state)
        adult._clear_current_occurrence();adult._current_selection_context=int(row.context)
        adult._current_language_channel=max(0,int(channel));adult._current_partner_context=adult._repair_partner_context(channel,row.context)
        chosen=int(adult.choose(state))
        return row if chosen==int(row.plan) else None


def joint_attention_respond(adult,organism,memory,tracker,grounding,raw_marker,motion,marker_feature,channel,source,state=AdultStateV1()):
    """Raw text + raw hand motion establish one candidate shared-attention response."""
    target=int(RawPointingMotionV1.target(tracker,motion))
    row=getattr(tracker,'active',{}).get(target)
    if target<=0 or not row or int(row[-1])!=0:return b'',0,0
    staged=memory.stage(adult,organism,tracker,grounding,raw_marker,marker_feature,
                        int(row[0]),int(row[1]),channel,source)
    if staged is None:return b'',0,0
    ticket,response,_episode=staged;spoken=emit_choice(adult,response)
    if not spoken:return b'',0,0
    return spoken,int(ticket),int(response)


def owned_joint_attention_respond(adult,organism,memory,tracker,grounding,gesture_sensor,
                                  raw_marker,motion,marker_feature,channel,source,sequence,digest,state=AdultStateV1()):
    accepted,_contiguous=gesture_sensor.ingest(source,sequence,motion,digest)
    return joint_attention_respond(adult,organism,memory,tracker,grounding,raw_marker,
                                   accepted,marker_feature,channel,source,state)


def owned_fused_joint_attention_respond(adult,organism,memory,tracker,grounding,
                                        gesture_sensor,gaze_sensor,raw_marker,
                                        gesture_motion,gaze_motion,marker_feature,channel,source,
                                        gesture_sequence,gesture_digest,gaze_sequence,gaze_digest,
                                        state=AdultStateV1()):
    """Atomically accept two authenticated directional streams before fusion/cognition."""
    gp=gesture_sensor.preview(source,gesture_sequence,gesture_motion,gesture_digest)
    zp=gaze_sensor.preview(source,gaze_sequence,gaze_motion,gaze_digest)
    accepted_gesture,_=gesture_sensor.commit(gp);accepted_gaze,_=gaze_sensor.commit(zp)
    target=int(SocialDirectionalCueFusionV1.resolve(tracker,accepted_gesture,accepted_gaze))
    row=getattr(tracker,'active',{}).get(target)
    if target<=0 or not row or int(row[-1])!=0:return b'',0,0
    staged=memory.stage(adult,organism,tracker,grounding,raw_marker,marker_feature,
                        int(row[0]),int(row[1]),channel,source)
    if staged is None:return b'',0,0
    ticket,response,_episode=staged;spoken=emit_choice(adult,response)
    if not spoken:return b'',0,0
    return spoken,int(ticket),int(response)


def owned_reliability_fused_joint_attention_respond(adult,organism,memory,tracker,grounding,
                                                    gesture_sensor,gaze_sensor,reliability,
                                                    gesture_lane,gaze_lane,raw_marker,
                                                    gesture_motion,gaze_motion,marker_feature,channel,source,
                                                    gesture_sequence,gesture_digest,gaze_sequence,gaze_digest,
                                                    state=AdultStateV1()):
    gp=gesture_sensor.preview(source,gesture_sequence,gesture_motion,gesture_digest)
    zp=gaze_sensor.preview(source,gaze_sequence,gaze_motion,gaze_digest)
    accepted_gesture,_=gesture_sensor.commit(gp);accepted_gaze,_=gaze_sensor.commit(zp)
    target=int(ReliabilityWeightedDirectionalFusionV1.resolve(
        tracker,accepted_gesture,gesture_lane,accepted_gaze,gaze_lane,reliability))
    row=getattr(tracker,'active',{}).get(target)
    if target<=0 or not row or int(row[-1])!=0:return b'',0,0
    staged=memory.stage(adult,organism,tracker,grounding,raw_marker,marker_feature,
                        int(row[0]),int(row[1]),channel,source)
    if staged is None:return b'',0,0
    ticket,response,_episode=staged;spoken=emit_choice(adult,response)
    if not spoken:return b'',0,0
    return spoken,int(ticket),int(response)


def settle_joint_attention_return(adult,memory,ticket,response,outcome_q16,independent=True):
    return memory.settle_partner_return(adult,int(ticket),int(response),int(outcome_q16),bool(independent))


def respond(adult,organism,memory,raw,channel=0,last_world=0,state=AdultStateV1(),source=None):
    rows=EmbodiedConversationAffordanceV1.proposals(adult,organism,memory,raw,channel,last_world,state)
    if len(rows)>1:return b''
    if len(rows)==1:
        row=EmbodiedConversationAffordanceV1.activate(adult,organism,memory,raw,channel,last_world,state)
        if row is None:return b''
        plan=int(row.plan) if int(row.kind) in (2,MEMORY_KIND) else adult.choose_public_plan(state)
        return emit_choice(adult,plan)
    contact=LanguageMasteryContactAdapterV1(adult);contact_source=adult._advance() if source is None else int(source);contact.contact(CONTACT_UTTERANCE,tuple(raw),contact_source,channel)
    return emit_choice(adult,adult.choose_public_plan(state))


def checkpoint(adult,organism,memory,last_world):
    memory.synchronize_pending(adult)
    return {'schema':1,'adult':adult.checkpoint(),'organism':organism.checkpoint(),'joint_memory':memory.checkpoint(),'last_world':max(0,int(last_world))}

def restore(data):
    if int(data.get('schema',0))!=1:raise ValueError('embodied_terminal:checkpoint')
    adult=LanguageMasteryAdultV1.restore(data['adult']);organism=ReferenceOrganismV2.restore(data['organism']);memory=ConsequenceQualifiedJointAttentionMemoryV1.restore(data['joint_memory']);memory.synchronize_pending(adult)
    return adult,organism,memory,max(0,int(data.get('last_world',0)))
def save(path,adult,organism,memory,last_world):
    payload=checkpoint(adult,organism,memory,last_world);path=Path(path)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as out:
        json.dump(payload,out,separators=(',',':'),sort_keys=True);out.flush();os.fsync(out.fileno());tmp=Path(out.name)
    os.replace(tmp,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',required=True,type=Path);parser.add_argument('--channel',type=int,default=0);parser.add_argument('--idle-ms',type=int,default=25)
    args=parser.parse_args()
    if args.channel<0 or not 1<=args.idle_ms<=1000:raise SystemExit('embodied-terminal:arguments')
    adult,organism,memory,last=restore(json.loads(args.resume.read_text()))
    readiness=selectors.DefaultSelector();readiness.register(sys.stdin.buffer,selectors.EVENT_READ)
    try:
        from reference_lived_world_conversation_terminal_v1 import quiet as world_quiet
        while True:
            events=readiness.select(args.idle_ms/1000.0)
            if events:
                raw=sys.stdin.buffer.readline()
                if not raw:break
                motor=respond(adult,organism,memory,raw.rstrip(b'\r\n'),args.channel,last)
            else:
                motor,last=world_quiet(adult,organism,last)
            if motor:sys.stdout.buffer.write(motor+b'\n');sys.stdout.buffer.flush()
    finally:
        readiness.close();save(args.resume,adult,organism,memory,last)

if __name__=='__main__':main()
