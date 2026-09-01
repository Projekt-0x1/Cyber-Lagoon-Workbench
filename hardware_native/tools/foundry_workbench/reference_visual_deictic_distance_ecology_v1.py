#!/usr/bin/env python3
"""Source-qualified learned spatial applicability for opaque deictic lexemes."""
from __future__ import annotations

MAX_DEICTIC_MARKERS=8
MAX_DEICTIC_SOURCES=16
MIN_DEICTIC_SOURCES=2

class VisualDeicticDistanceEcologyV1:
    def __init__(self):self.support={}

    @staticmethod
    def _visible(tracker):
        out=[]
        for entity,row in getattr(tracker,'active',{}).items():
            if row and int(row[-1])==0:out.append((int(entity),int(row[0]),int(row[1])))
        return tuple(out)

    @classmethod
    def _pointed(cls,tracker,point_y2,point_x2):
        rows=cls._visible(tracker)
        if not rows:return 0,0
        ranked=[(abs(y-int(point_y2))+abs(x-int(point_x2)),entity,y,x) for entity,y,x in rows]
        best=min(row[0] for row in ranked);wins=[row for row in ranked if row[0]==best]
        if len(wins)!=1:return 0,0
        _d,entity,y,x=wins[0];return int(entity),abs(y)+abs(x)

    def observe(self,adult,tracker,raw,point_y2,point_x2,speaker_y2,speaker_x2,source):
        source=int(source);features=tuple(adult.language.lexical_features(tuple(raw)))
        if source<=0 or len(features)!=1:return 0
        entity,_=self._pointed(tracker,point_y2,point_x2)
        if not entity:return 0
        row=getattr(tracker,'active',{}).get(entity);y2,x2=int(row[0]),int(row[1])
        distance=abs(y2-int(speaker_y2))+abs(x2-int(speaker_x2))
        feature=int(features[0])
        if feature not in self.support and len(self.support)>=MAX_DEICTIC_MARKERS:return 0
        bucket=self.support.setdefault(feature,{})
        if source not in bucket and len(bucket)>=MAX_DEICTIC_SOURCES:return 0
        bucket[source]=int(distance);return entity

    def profile(self,adult,feature):
        feature=int(feature);rows=self.support.get(feature,{})
        active=[int(distance) for source,distance in rows.items() if int(source) not in adult.language._withdrawn]
        if len(active)<MIN_DEICTIC_SOURCES:return None
        return sum(active)/len(active)

    def resolve(self,adult,tracker,raw,point_y2,point_x2,speaker_y2,speaker_x2):
        features=tuple(adult.language.lexical_features(tuple(raw)))
        if len(features)!=1:return 0
        feature=int(features[0]);target=self._pointed(tracker,point_y2,point_x2)[0]
        if not target:return 0
        row=getattr(tracker,'active',{}).get(target);distance=abs(int(row[0])-int(speaker_y2))+abs(int(row[1])-int(speaker_x2))
        profiles={f:self.profile(adult,f) for f in self.support};profiles={f:p for f,p in profiles.items() if p is not None}
        if feature not in profiles or len(profiles)<2:return 0
        errors=sorted((abs(float(p)-distance),int(f)) for f,p in profiles.items())
        if len(errors)<2 or errors[0][0]==errors[1][0] or errors[0][1]!=feature:return 0
        return int(target)

    def checkpoint(self):
        return {'schema':1,'support':[{'feature':f,'rows':[[s,d] for s,d in sorted(rows.items())]} for f,rows in sorted(self.support.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('deictic_distance:checkpoint')
        out=cls()
        for row in data.get('support',()):
            feature=int(row.get('feature',0));rows={int(s):int(d) for s,d in row.get('rows',())}
            if feature<=0 or feature in out.support or not rows or len(rows)>MAX_DEICTIC_SOURCES or min(rows)<=0 or min(rows.values())<0:raise ValueError('deictic_distance:checkpoint')
            out.support[feature]=rows
        if len(out.support)>MAX_DEICTIC_MARKERS:raise ValueError('deictic_distance:capacity')
        return out
