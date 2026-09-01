#!/usr/bin/env python3
"""Transient kinematic segmentation of an authenticated continuous hand-position stream."""
from __future__ import annotations
from reference_body_event_ingress_v1 import BodyEventIngressV1

MAX_ACTIVE_GESTURE_SOURCES=16
MAX_SEGMENT_SAMPLES=32
START_SPEED2=4
HOLD_SPEED2=1
HOLD_TRANSITIONS=2

class ContinuousGestureSegmenterV1:
    def __init__(self,ingress=None):
        self.ingress=ingress if ingress is not None else BodyEventIngressV1();self.active={}

    @staticmethod
    def sample_payload(y,x):return {'y':int(y),'x':int(x)}
    @classmethod
    def sample_digest(cls,y,x):return BodyEventIngressV1.payload_digest('gesture-sample',cls.sample_payload(y,x))

    def ingest(self,source:int,sequence:int,y:int,x:int,digest:str):
        source=int(source);sequence=int(sequence);y=int(y);x=int(x)
        preview=self.ingress.preview('gesture-sample',source,sequence,self.sample_payload(y,x),digest)
        prior=self.active.get(source)
        if prior is None:
            if len(self.active)>=MAX_ACTIVE_GESTURE_SOURCES:raise ValueError('gesture_segment:source_capacity')
            self.ingress.commit(preview);self.active[source]=((y,x),[],0,False);return ()
        last,points,hold,moving=prior;dy=y-int(last[0]);dx=x-int(last[1]);speed2=dy*dy+dx*dx
        completed=()
        if not moving:
            if speed2>=START_SPEED2:
                points=[last,(y,x)];moving=True;hold=0
            else:
                points=[];hold=0
        else:
            if len(points)>=MAX_SEGMENT_SAMPLES:
                # Capacity refuses the current partial stroke rather than emitting a truncated gesture.
                points=[];moving=False;hold=0
            else:
                points=[*points,(y,x)]
                if speed2<=HOLD_SPEED2:hold+=1
                else:hold=0
                if hold>=HOLD_TRANSITIONS:
                    # Remove the final repeated hold samples but retain the first hold position as endpoint.
                    endpoint_index=max(3,len(points)-HOLD_TRANSITIONS)
                    candidate=tuple(points[:endpoint_index])
                    if len(candidate)>=3:completed=candidate
                    points=[];moving=False;hold=0
        self.ingress.commit(preview);self.active[source]=((y,x),points,hold,moving)
        return completed

    def withdraw_source(self,source:int):
        source=int(source);self.ingress.withdraw_source(source);self.active.pop(source,None)

    def checkpoint(self):
        return {'schema':1,'ingress':self.ingress.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('gesture_segment:checkpoint')
        return cls(BodyEventIngressV1.restore(data['ingress']))
