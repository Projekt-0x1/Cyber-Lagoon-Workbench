#!/usr/bin/env python3
from __future__ import annotations
from reference_visual_pathway_stage_occurrences_v1 import VECTOR_STAGE
HISTORY=8;QUORUM=3
class VisualVectorPredictionV1:
    def __init__(self):self.history=[]
    @staticmethod
    def rows(vector_occurrence):
        if vector_occurrence is None or int(getattr(vector_occurrence,'stage',0))!=VECTOR_STAGE:raise ValueError('visual_prediction:vector')
        rows,_shape=vector_occurrence.representation
        return tuple(sorted(set(tuple(int(v) for v in row) for row in rows)))
    def observe(self,vector_occurrence):
        rowset=self.rows(vector_occurrence);self.history.append(rowset)
        if len(self.history)>HISTORY:del self.history[:-HISTORY]
        return self.prediction()
    def prediction(self):
        counts={}
        for rows in self.history:counts[rows]=counts.get(rows,0)+1
        if not counts:return None
        best=max(counts.values());w=[rows for rows,n in counts.items() if n==best]
        return w[0] if best>=QUORUM and len(w)==1 else None
    def residual(self,vector_occurrence):
        pred=self.prediction();obs=self.rows(vector_occurrence)
        if pred is None:return None
        ps,os=set(pred),set(obs)
        return {'missing':tuple(sorted(ps-os)),'unexpected':tuple(sorted(os-ps))}
    def checkpoint(self):return {'schema':1,'history':[[list(r) for r in rows] for rows in self.history]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('visual_prediction:checkpoint')
        out=cls();out.history=[tuple(tuple(int(v) for v in r) for r in rows) for rows in d.get('history',())]
        if len(out.history)>HISTORY:raise ValueError('visual_prediction:checkpoint')
        return out
