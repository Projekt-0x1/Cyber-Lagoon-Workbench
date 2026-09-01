#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import time
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_pathway_stage_occurrences_v1 import VECTOR_STAGE

@dataclass(frozen=True)
class RecurrentHeadingOccurrenceV1:
    sequence:int;organism_tick:int;heading:tuple;vector_occurrences:int;provenance_ns:int

class RecurrentVisualHeadingV1:
    def __init__(self):
        self.sequence=0;self.tick=None;self.rows=[];self.shape=None;self.sources=[];self.current=None
    @staticmethod
    def _tick(o):
        t=int(getattr(o,'tick_count',-1))
        if t<0:raise ValueError('recurrent_heading:organism')
        return t
    def _clear(self):
        self.tick=None;self.rows=[];self.shape=None;self.sources=[]
    def observe_vector(self,o,vector_occurrence):
        if vector_occurrence is None or int(getattr(vector_occurrence,'stage',0))!=VECTOR_STAGE:raise ValueError('recurrent_heading:vector')
        tick=self._tick(o)
        if int(vector_occurrence.organism_tick)!=tick:raise ValueError('recurrent_heading:stale_tick')
        rows,shape=vector_occurrence.representation
        rows=tuple(tuple(int(v) for v in row) for row in rows);shape=tuple(map(int,shape))
        if self.tick is not None and self.tick!=tick:self._clear()
        if self.shape is not None and self.shape!=shape:self._clear()
        if self.tick is None:self.tick=tick;self.shape=shape
        source=(int(vector_occurrence.sequence),tuple(rows))
        if source not in self.sources:self.sources.append(source)
        for row in rows:
            if row not in self.rows:self.rows.append(row)
        width,height=self.shape
        heading=VisualHeadingCenterV1.estimate_rows(tuple(self.rows),width,height)
        if heading is None:return None
        # All accumulated subsets must be geometrically consistent with the final solution.
        if len(self.rows)>=3:
            for i in range(len(self.rows)):
                subset=tuple(r for j,r in enumerate(self.rows) if j!=i)
                if len(subset)>=3:
                    candidate=VisualHeadingCenterV1.estimate_rows(subset,width,height)
                    if candidate is not None and candidate!=heading:
                        self._clear();return None
        self.sequence+=1
        occ=RecurrentHeadingOccurrenceV1(self.sequence,tick,heading,len(self.sources),int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        self.current=occ;self._clear();return occ
    def checkpoint(self):return {'schema':1,'sequence':self.sequence}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('recurrent_heading:checkpoint')
        out=cls();out.sequence=int(d.get('sequence',0))
        if out.sequence<0:raise ValueError('recurrent_heading:checkpoint')
        return out
