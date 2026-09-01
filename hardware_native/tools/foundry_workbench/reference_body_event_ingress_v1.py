#!/usr/bin/env python3
"""Content-neutral authenticated source/order/integrity envelope for non-sensor body events."""
from __future__ import annotations
import hashlib,json

MAX_BODY_LANES=16
MAX_BODY_SOURCES=64

class BodyEventIngressV1:
    def __init__(self):
        self.last_sequence={};self.withdrawn_sources=set()

    @staticmethod
    def payload_digest(lane,payload)->str:
        lane=str(lane)
        if not lane or len(lane)>64:raise ValueError('body_event:lane')
        raw=json.dumps(payload,separators=(',',':'),sort_keys=True,ensure_ascii=True)
        h=hashlib.sha256(b'body-event-ingress-v1\0');h.update(lane.encode('utf-8'));h.update(b'\0');h.update(raw.encode('ascii'))
        return h.hexdigest()

    def preview(self,lane,source:int,sequence:int,payload,digest:str):
        lane=str(lane);source=int(source);sequence=int(sequence)
        if not lane or len(lane)>64:raise ValueError('body_event:lane')
        if source<=0 or sequence<=0 or source in self.withdrawn_sources:raise ValueError('body_event:source')
        if str(digest)!=self.payload_digest(lane,payload):raise ValueError('body_event:digest')
        key=(lane,source);prior=int(self.last_sequence.get(key,0))
        if sequence<=prior:raise ValueError('body_event:sequence')
        if key not in self.last_sequence and len({k[0] for k in self.last_sequence}|{lane})>MAX_BODY_LANES:raise ValueError('body_event:lane_capacity')
        if key not in self.last_sequence and len({k[1] for k in self.last_sequence}|{source})>MAX_BODY_SOURCES:raise ValueError('body_event:source_capacity')
        return lane,source,sequence

    def commit(self,preview):
        lane,source,sequence=preview;self.last_sequence[(str(lane),int(source))]=int(sequence);return True

    def ingest(self,lane,source:int,sequence:int,payload,digest:str):
        return self.commit(self.preview(lane,source,sequence,payload,digest))

    def withdraw_source(self,source:int):
        source=int(source)
        if source<=0:raise ValueError('body_event:source')
        self.withdrawn_sources.add(source)

    def checkpoint(self):
        return {'schema':1,'last_sequence':[{'lane':lane,'source':source,'sequence':seq} for (lane,source),seq in sorted(self.last_sequence.items())],'withdrawn_sources':sorted(self.withdrawn_sources)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('body_event:checkpoint')
        out=cls()
        for row in data.get('last_sequence',()):
            lane=str(row.get('lane',''));source=int(row.get('source',0));seq=int(row.get('sequence',0));key=(lane,source)
            if not lane or len(lane)>64 or source<=0 or seq<=0 or key in out.last_sequence:raise ValueError('body_event:checkpoint_row')
            out.last_sequence[key]=seq
        out.withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        if any(s<=0 for s in out.withdrawn_sources) or len({k[0] for k in out.last_sequence})>MAX_BODY_LANES or len({k[1] for k in out.last_sequence})>MAX_BODY_SOURCES:raise ValueError('body_event:checkpoint_capacity')
        return out
