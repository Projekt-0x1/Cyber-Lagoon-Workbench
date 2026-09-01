#!/usr/bin/env python3
"""Source-qualified deictic applicability over target distance relative to current action extent."""
from __future__ import annotations
from reference_visual_deictic_distance_ecology_v1 import VisualDeicticDistanceEcologyV1,MAX_DEICTIC_MARKERS,MAX_DEICTIC_SOURCES,MIN_DEICTIC_SOURCES

Q=1<<16
class VisualDeicticActionSpaceV1:
    def __init__(self):self.support={}
    @staticmethod
    def _ratio_q16(distance,extent):
        distance=int(distance);extent=int(extent)
        if distance<0 or extent<=0:return 0
        return max(1,(distance*Q)//extent)
    def observe(self,adult,tracker,raw,point_y2,point_x2,speaker_y2,speaker_x2,action_extent,source):
        source=int(source);features=tuple(adult.language.lexical_features(tuple(raw)))
        if source<=0 or len(features)!=1:return 0
        entity=VisualDeicticDistanceEcologyV1._pointed(tracker,point_y2,point_x2)[0]
        if not entity:return 0
        row=tracker.active.get(entity);distance=abs(int(row[0])-int(speaker_y2))+abs(int(row[1])-int(speaker_x2));ratio=self._ratio_q16(distance,action_extent)
        feature=int(features[0])
        if feature not in self.support and len(self.support)>=MAX_DEICTIC_MARKERS:return 0
        bucket=self.support.setdefault(feature,{})
        if source not in bucket and len(bucket)>=MAX_DEICTIC_SOURCES:return 0
        bucket[source]=ratio;return entity
    def profile(self,adult,feature):
        rows=self.support.get(int(feature),{});values=[int(v) for s,v in rows.items() if int(s) not in adult.language._withdrawn]
        if len(values)<MIN_DEICTIC_SOURCES:return None
        return sum(values)//len(values)
    def resolve(self,adult,tracker,raw,point_y2,point_x2,speaker_y2,speaker_x2,action_extent):
        features=tuple(adult.language.lexical_features(tuple(raw)))
        if len(features)!=1:return 0
        feature=int(features[0]);target=VisualDeicticDistanceEcologyV1._pointed(tracker,point_y2,point_x2)[0]
        if not target:return 0
        row=tracker.active.get(target);distance=abs(int(row[0])-int(speaker_y2))+abs(int(row[1])-int(speaker_x2));ratio=self._ratio_q16(distance,action_extent)
        profiles={f:self.profile(adult,f) for f in self.support};profiles={f:p for f,p in profiles.items() if p is not None}
        if feature not in profiles or len(profiles)<2:return 0
        errors=sorted((abs(int(p)-ratio),int(f)) for f,p in profiles.items())
        if errors[0][0]==errors[1][0] or errors[0][1]!=feature:return 0
        return int(target)
    def checkpoint(self):return {'schema':1,'support':[{'feature':f,'rows':[[s,v] for s,v in sorted(rows.items())]} for f,rows in sorted(self.support.items())]}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('deictic_action_space:checkpoint')
        out=cls()
        for row in data.get('support',()):
            f=int(row.get('feature',0));rows={int(s):int(v) for s,v in row.get('rows',())}
            if f<=0 or f in out.support or not rows or len(rows)>MAX_DEICTIC_SOURCES or min(rows)<=0 or min(rows.values())<=0:raise ValueError('deictic_action_space:checkpoint')
            out.support[f]=rows
        if len(out.support)>MAX_DEICTIC_MARKERS:raise ValueError('deictic_action_space:capacity')
        return out
