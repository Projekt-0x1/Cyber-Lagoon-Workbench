#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import time
from reference_local_optic_flow_v1 import LocalOpticFlowV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1

COMPONENT_STAGE=0xA501;VECTOR_STAGE=0xA502;HEADING_STAGE=0xA503
@dataclass(frozen=True)
class VisualStageOccurrenceV1:
    stage:int;sequence:int;organism_tick:int;predecessor_sequence:int;representation:object;provenance_ns:int

class VisualPathwayStageOccurrencesV1:
    def __init__(self):
        self.sequences={COMPONENT_STAGE:0,VECTOR_STAGE:0,HEADING_STAGE:0};self.current={}
    @staticmethod
    def _tick(o):
        t=int(getattr(o,'tick_count',-1))
        if t<0:raise ValueError('visual_stage:organism')
        return t
    def _emit(self,stage,o,pred,rep):
        if rep is None:return None
        seq=self.sequences[stage]+1;self.sequences[stage]=seq
        occ=VisualStageOccurrenceV1(stage,seq,self._tick(o),0 if pred is None else pred.sequence,rep,int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        self.current[stage]=occ;return occ
    def process_components(self,o,previous_frame,current_frame):
        previous=LocalOpticFlowV1._components(previous_frame);current=LocalOpticFlowV1._components(current_frame)
        if not previous or not current:return None
        shape=(len(previous_frame[0]),len(previous_frame))
        self.current.pop(VECTOR_STAGE,None);self.current.pop(HEADING_STAGE,None)
        return self._emit(COMPONENT_STAGE,o,None,(previous,current,shape))
    def process_vectors(self,o,component_occurrence):
        if component_occurrence is None or component_occurrence.stage!=COMPONENT_STAGE or self.current.get(COMPONENT_STAGE)!=component_occurrence:raise ValueError('visual_stage:component_predecessor')
        previous,current,shape=component_occurrence.representation;rows=[]
        for prior,now in LocalOpticFlowV1._match(previous,current):
            dx=now[0]-prior[0];dy=now[1]-prior[1]
            if dx or dy:rows.append((int(prior[0]),int(prior[1]),int(dx),int(dy)))
        if not rows:return None
        self.current.pop(HEADING_STAGE,None)
        return self._emit(VECTOR_STAGE,o,component_occurrence,(tuple(rows),shape))
    def process_heading(self,o,vector_occurrence):
        if vector_occurrence is None or vector_occurrence.stage!=VECTOR_STAGE or self.current.get(VECTOR_STAGE)!=vector_occurrence:raise ValueError('visual_stage:vector_predecessor')
        rows,(width,height)=vector_occurrence.representation
        center=VisualHeadingCenterV1.estimate_rows(rows,width,height)
        return self._emit(HEADING_STAGE,o,vector_occurrence,center)
    def checkpoint(self):return {'schema':1,'sequences':[[k,v] for k,v in sorted(self.sequences.items())]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('visual_stage:checkpoint')
        out=cls();out.sequences={int(k):int(v) for k,v in d.get('sequences',())}
        if set(out.sequences)!={COMPONENT_STAGE,VECTOR_STAGE,HEADING_STAGE} or any(v<0 for v in out.sequences.values()):raise ValueError('visual_stage:checkpoint')
        return out
