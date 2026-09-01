#!/usr/bin/env python3
"""Source-qualified lexical naming of the uniquely attended visual individual."""
from __future__ import annotations

class VisualJointAttentionNamingV1:
    @staticmethod
    def visible_entities(tracker):
        if hasattr(tracker,'active_entity'):
            entity=int(getattr(tracker,'active_entity',0))
            return () if entity<=0 else (entity,)
        rows=getattr(tracker,'active',{})
        out=[]
        for entity,row in rows.items():
            # Multi-object row ends in miss count; only currently visible files nameable.
            if row and int(row[-1])==0:out.append(int(entity))
        return tuple(sorted(out))

    @classmethod
    def observe(cls,adult,tracker,raw,source:int):
        entities=cls.visible_entities(tracker);source=int(source);surface=tuple(int(x) for x in raw)
        if len(entities)!=1 or source<=0 or not surface:return 0
        entity=entities[0];adult.observe_surface_item(entity,surface,source);return entity
