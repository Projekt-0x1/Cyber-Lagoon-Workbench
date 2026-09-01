#!/usr/bin/env python3
"""Content-neutral owned social hand-trajectory stream boundary."""
from __future__ import annotations
import hashlib

MAX_GESTURE_SAMPLES=32

class GestureSensorIngressV1:
    def __init__(self):
        self.last_sequence={};self.withdrawn_sources=set();self.active_source=0;self.active_sequence=0;self.current_motion=None

    @staticmethod
    def motion_digest(samples)->str:
        rows=tuple((int(y),int(x)) for y,x in samples)
        if not 3<=len(rows)<=MAX_GESTURE_SAMPLES:raise ValueError('gesture_sensor:shape')
        h=hashlib.sha256(b'gesture-sensor-motion-v1\0')
        for y,x in rows:
            if not -(1<<31)<=y<(1<<31) or not -(1<<31)<=x<(1<<31):raise ValueError('gesture_sensor:coordinate')
            h.update(int(y).to_bytes(4,'little',signed=True));h.update(int(x).to_bytes(4,'little',signed=True))
        h.update(len(rows).to_bytes(2,'little'));return h.hexdigest()

    def preview(self,source:int,sequence:int,samples,digest:str):
        source=int(source);sequence=int(sequence)
        if source<=0 or sequence<=0 or source in self.withdrawn_sources:raise ValueError('gesture_sensor:source')
        computed=self.motion_digest(samples)
        if str(digest)!=computed:raise ValueError('gesture_sensor:digest')
        prior=int(self.last_sequence.get(source,0))
        if sequence<=prior:raise ValueError('gesture_sensor:sequence')
        rows=tuple((int(y),int(x)) for y,x in samples)
        contiguous=(self.active_source==source and self.active_sequence>0 and sequence==self.active_sequence+1)
        return source,sequence,rows,bool(contiguous)

    def commit(self,preview):
        source,sequence,rows,contiguous=preview
        self.last_sequence[int(source)]=int(sequence);self.active_source=int(source);self.active_sequence=int(sequence);self.current_motion=tuple(rows)
        return tuple(rows),bool(contiguous)

    def ingest(self,source:int,sequence:int,samples,digest:str):
        return self.commit(self.preview(source,sequence,samples,digest))

    def withdraw_source(self,source:int):
        source=int(source)
        if source<=0:raise ValueError('gesture_sensor:source')
        self.withdrawn_sources.add(source)
        if self.active_source==source:self.active_source=0;self.active_sequence=0;self.current_motion=None

    def checkpoint(self):
        return {'schema':1,'last_sequence':[[s,q] for s,q in sorted(self.last_sequence.items())],'withdrawn_sources':sorted(self.withdrawn_sources)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('gesture_sensor:checkpoint')
        out=cls();out.last_sequence={int(s):int(q) for s,q in data.get('last_sequence',())};out.withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        if any(s<=0 or q<=0 for s,q in out.last_sequence.items()) or any(s<=0 for s in out.withdrawn_sources):raise ValueError('gesture_sensor:checkpoint')
        return out
