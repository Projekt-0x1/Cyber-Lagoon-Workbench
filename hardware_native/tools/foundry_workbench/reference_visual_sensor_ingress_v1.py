#!/usr/bin/env python3
"""Content-neutral owned visual frame stream boundary."""
from __future__ import annotations
import hashlib


class VisualSensorIngressV1:
    """Own source/order/integrity only; raw frames remain transient."""
    def __init__(self):
        self.last_sequence={}
        self.withdrawn_sources=set()
        self.active_source=0
        self.active_sequence=0
        self.current_frame=None

    @staticmethod
    def frame_digest(frame)->str:
        h=hashlib.sha256(b'visual-sensor-frame-v1\0')
        rows=tuple(tuple(int(v) for v in row) for row in frame)
        if not rows or not rows[0] or any(len(row)!=len(rows[0]) for row in rows):raise ValueError('visual_sensor:shape')
        for row in rows:
            for value in row:
                if value<0 or value>255:raise ValueError('visual_sensor:pixel')
                h.update(bytes((value,)))
        h.update(len(rows).to_bytes(4,'little'));h.update(len(rows[0]).to_bytes(4,'little'))
        return h.hexdigest()

    def ingest(self,source:int,sequence:int,frame,digest:str):
        source=int(source);sequence=int(sequence)
        if source<=0 or sequence<=0 or source in self.withdrawn_sources:raise ValueError('visual_sensor:source')
        computed=self.frame_digest(frame)
        if str(digest)!=computed:raise ValueError('visual_sensor:digest')
        prior=int(self.last_sequence.get(source,0))
        if sequence<=prior:raise ValueError('visual_sensor:sequence')
        contiguous=(self.active_source==source and self.active_sequence>0 and sequence==self.active_sequence+1)
        rows=tuple(tuple(int(v) for v in row) for row in frame)
        self.last_sequence[source]=sequence
        self.active_source=source;self.active_sequence=sequence;self.current_frame=rows
        return rows,bool(contiguous)

    def withdraw_source(self,source:int):
        source=int(source)
        if source<=0:raise ValueError('visual_sensor:source')
        self.withdrawn_sources.add(source)
        if self.active_source==source:
            self.active_source=0;self.active_sequence=0;self.current_frame=None

    def checkpoint(self):
        return {'schema':1,'last_sequence':[[k,v] for k,v in sorted(self.last_sequence.items())],
                'withdrawn_sources':sorted(self.withdrawn_sources)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('visual_sensor:checkpoint')
        out=cls();out.last_sequence={int(k):int(v) for k,v in data.get('last_sequence',())}
        if any(k<=0 or v<=0 for k,v in out.last_sequence.items()):raise ValueError('visual_sensor:checkpoint')
        out.withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        if any(x<=0 for x in out.withdrawn_sources):raise ValueError('visual_sensor:checkpoint')
        return out
