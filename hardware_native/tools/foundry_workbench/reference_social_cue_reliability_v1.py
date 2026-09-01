#!/usr/bin/env python3
"""Bounded source-qualified reliability for opaque social directional cue lanes."""
from __future__ import annotations
Q=1<<16;MAX_LANES=8;MAX_SOURCES=16
class SocialCueReliabilityV1:
    def __init__(self):self.evidence={}
    def observe(self,lane,source,effect,independent=True):
        lane=int(lane);source=int(source);effect=int(effect)
        if not independent or lane<=0 or source<=0 or effect==0:return False
        if lane not in self.evidence and len(self.evidence)>=MAX_LANES:return False
        rows=self.evidence.setdefault(lane,{})
        if source not in rows and len(rows)>=MAX_SOURCES:return False
        prior=int(rows.get(source,0));rows[source]=max(-8,min(8,prior+(1 if effect>0 else -1)));return True
    def weight_q16(self,lane):
        rows=self.evidence.get(int(lane),{})
        if len(rows)<2:return Q
        strength=max(-8,min(8,sum(int(v) for v in rows.values())))
        return max(Q//8,min(8*Q,(Q*(9+strength))//max(1,9-strength)))
    def checkpoint(self):return {'schema':1,'evidence':[{'lane':l,'rows':[[s,e] for s,e in sorted(rows.items())]} for l,rows in sorted(self.evidence.items())]}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('cue_reliability:checkpoint')
        out=cls()
        for row in data.get('evidence',()):
            lane=int(row.get('lane',0));rows={int(s):int(e) for s,e in row.get('rows',())}
            if lane<=0 or lane in out.evidence or len(rows)>MAX_SOURCES or any(s<=0 or e==0 or not -8<=e<=8 for s,e in rows.items()):raise ValueError('cue_reliability:checkpoint')
            out.evidence[lane]=rows
        if len(out.evidence)>MAX_LANES:raise ValueError('cue_reliability:capacity')
        return out
