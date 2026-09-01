#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import time
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1

VISUAL_PATH=0x915A1;VESTIBULAR_PATH=0x915A2
@dataclass(frozen=True)
class ReadyOccurrenceV1:
    pathway:int;sequence:int;organism_tick:int;representation:object;provenance_ns:int

class PathwayReadyOccurrenceV1:
    def __init__(self):self.sequences={VISUAL_PATH:0,VESTIBULAR_PATH:0};self.current=None
    @staticmethod
    def _tick(o):
        t=int(getattr(o,'tick_count',-1))
        if t<0:raise ValueError('pathway_ready:organism')
        return t
    def _emit(self,pathway,o,representation):
        if representation is None:return None
        seq=self.sequences[pathway]+1;self.sequences[pathway]=seq
        occ=ReadyOccurrenceV1(pathway,seq,self._tick(o),representation,int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        self.current=occ;return occ
    def process_visual_heading(self,o,previous_frame,current_frame):
        return self._emit(VISUAL_PATH,o,VisualHeadingCenterV1().estimate(previous_frame,current_frame))
    def process_vestibular_heading(self,o,resolution):
        return self._emit(VESTIBULAR_PATH,o,VestibularTranslationHeadingV1.from_resolution(resolution))
    def checkpoint(self):return {'schema':1,'sequences':[[k,v] for k,v in sorted(self.sequences.items())]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('pathway_ready:checkpoint')
        out=cls();out.sequences={int(k):int(v) for k,v in d.get('sequences',())}
        if set(out.sequences)!={VISUAL_PATH,VESTIBULAR_PATH} or any(v<0 for v in out.sequences.values()):raise ValueError('pathway_ready:checkpoint')
        return out
