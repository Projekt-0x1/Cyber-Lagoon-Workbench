#!/usr/bin/env python3
"""Owned raw vestibular gravito-inertial stream boundary."""
from __future__ import annotations
import hashlib


class VestibularSensorIngressV1:
    """Own source/order/integrity only; raw inertial samples remain transient."""
    def __init__(self):
        self.last_sequence={}
        self.withdrawn_sources=set()
        self.active_source=0
        self.active_sequence=0
        self.current_sample=None

    @staticmethod
    def sample_digest(sample)->str:
        row=tuple(int(x) for x in sample)
        if len(row)!=2 or any(abs(x)>4096 for x in row):raise ValueError('vestibular_sensor:sample')
        h=hashlib.sha256(b'vestibular-sensor-sample-v1\0')
        for value in row:h.update(int(value).to_bytes(4,'little',signed=True))
        return h.hexdigest()

    def ingest(self,source:int,sequence:int,sample,digest:str):
        source=int(source);sequence=int(sequence)
        if source<=0 or sequence<=0 or source in self.withdrawn_sources:raise ValueError('vestibular_sensor:source')
        computed=self.sample_digest(sample)
        if str(digest)!=computed:raise ValueError('vestibular_sensor:digest')
        prior=int(self.last_sequence.get(source,0))
        if sequence<=prior:raise ValueError('vestibular_sensor:sequence')
        contiguous=(self.active_source==source and self.active_sequence>0 and sequence==self.active_sequence+1)
        row=tuple(int(x) for x in sample)
        self.last_sequence[source]=sequence
        self.active_source=source;self.active_sequence=sequence;self.current_sample=row
        return row,bool(contiguous)

    def withdraw_source(self,source:int):
        source=int(source)
        if source<=0:raise ValueError('vestibular_sensor:source')
        self.withdrawn_sources.add(source)
        if self.active_source==source:
            self.active_source=0;self.active_sequence=0;self.current_sample=None

    def checkpoint(self):
        return {'schema':1,'last_sequence':[[k,v] for k,v in sorted(self.last_sequence.items())],
                'withdrawn_sources':sorted(self.withdrawn_sources)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('vestibular_sensor:checkpoint')
        out=cls();out.last_sequence={int(k):int(v) for k,v in data.get('last_sequence',())}
        if any(k<=0 or v<=0 for k,v in out.last_sequence.items()):raise ValueError('vestibular_sensor:checkpoint')
        out.withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        if any(x<=0 for x in out.withdrawn_sources):raise ValueError('vestibular_sensor:checkpoint')
        return out
