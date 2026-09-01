#!/usr/bin/env python3
"""Transient object-file tracker over owned visual sensor continuity."""
from __future__ import annotations

class VisualObjectFileTrackerV1:
    """Continuity owns individuation; downstream semantics never enter this adapter."""
    def __init__(self):
        self.active_entity=0;self.active_source=0;self.previous_feature=0

    def gap(self):
        self.active_entity=0;self.active_source=0;self.previous_feature=0

    def observe(self,organism,sensor,temporal,source,sequence,frame,feature,digest):
        rows,contiguous=sensor.ingest(source,sequence,frame,digest)
        source=int(source);sequence=int(sequence);feature=int(feature)
        if feature<=0:raise ValueError('object_file:feature')
        if not contiguous:
            temporal.gap();self.gap()
        relation=temporal.observe_features((feature,))
        same=(bool(contiguous) and self.active_entity>0 and self.active_source==source
              and self.previous_feature>0 and relation>0)
        if same:
            entity=int(self.active_entity);organism.update_visual_entity(entity,(relation,),source)
        else:
            entity=organism.mint_visual_entity((feature,),source,sequence)
        self.active_entity=entity;self.active_source=source;self.previous_feature=feature
        return entity,int(relation),bool(same),rows

    def checkpoint(self):
        # Active object files are perceptual occurrence state; restart breaks continuity.
        return {'schema':1}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('object_file:checkpoint')
        return cls()
