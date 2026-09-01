#!/usr/bin/env python3
"""Current-context source prediction calibration, isolated from recommendation valence.

This owner records only explicit grounded source prediction matches/mismatches. Historical
core SourceCalibrationV2 rows may contain reward-based updates and are therefore not imported.
"""
from __future__ import annotations

Q=1<<16
MAX_ROWS=4096

class SourcePredictionCalibrationV1:
    def __init__(self):self._support={};self._counter={}
    @property
    def evidence_count(self):return sum(self._support.values())+sum(self._counter.values())
    def observe(self,source,context,matched,independent=True):
        source=int(source);context=int(context)
        if not independent or source<=0 or context<=0:return False
        key=(source,context);bucket=self._support if bool(matched) else self._counter
        if key not in bucket and len(set((*self._support.keys(),*self._counter.keys())))>=MAX_ROWS:raise RuntimeError('source-prediction:capacity')
        bucket[key]=int(bucket.get(key,0))+1;return True
    def calibration(self,source,context):
        key=(int(source),int(context));return int(self._support.get(key,0))-int(self._counter.get(key,0))
    def evidence_for(self,source,context):
        key=(int(source),int(context));return int(self._support.get(key,0))+int(self._counter.get(key,0))
    def quality_q16(self,source,context):
        key=(int(source),int(context));good=int(self._support.get(key,0));bad=int(self._counter.get(key,0))
        if good+bad==0:return 0
        # Signed evidence around neutral 0; no global reputation prior.
        return ((good-bad)*Q)//(good+bad+2)
    def checkpoint(self):
        return {'schema':1,
            'support':[[s,c,n] for (s,c),n in sorted(self._support.items())],
            'counter':[[s,c,n] for (s,c),n in sorted(self._counter.items())]}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('source-prediction:checkpoint-schema')
        out=cls();out._support={(int(s),int(c)):int(n) for s,c,n in data.get('support',())};out._counter={(int(s),int(c)):int(n) for s,c,n in data.get('counter',())}
        if len(set((*out._support.keys(),*out._counter.keys())))>MAX_ROWS:raise RuntimeError('source-prediction:capacity')
        return out
