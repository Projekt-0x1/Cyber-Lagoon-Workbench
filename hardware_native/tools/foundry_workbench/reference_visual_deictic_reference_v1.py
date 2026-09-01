#!/usr/bin/env python3
"""Resolve a learned deictic marker against visible object files and a pointing location."""
from __future__ import annotations

class VisualDeicticReferenceV1:
    @staticmethod
    def resolve(adult,tracker,raw,marker_feature,point_y2,point_x2):
        marker_feature=int(marker_feature);units=tuple(int(x) for x in raw)
        if marker_feature<=0 or adult.language.lexeme(marker_feature)!=units:return 0
        rows=getattr(tracker,'active',{})
        visible=[]
        for entity,row in rows.items():
            if not row or int(row[-1])!=0:continue
            visible.append((int(entity),int(row[0]),int(row[1])))
        if not visible:return 0
        distances=[(abs(y2-int(point_y2))+abs(x2-int(point_x2)),entity) for entity,y2,x2 in visible]
        best=min(row[0] for row in distances);winners=[entity for distance,entity in distances if distance==best]
        return winners[0] if len(winners)==1 else 0
