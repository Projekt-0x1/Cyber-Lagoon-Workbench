#!/usr/bin/env python3
from __future__ import annotations
from reference_population_v1 import mix64
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1
HISTORY=8;QUORUM=3;MAX_CONTEXTS=4;CONTEXT_TAG=0xC017EC7
class ContextConditionalVisualPredictionV1:
    def __init__(self):self.histories={};self.context_order=[]
    @staticmethod
    def context(organism):
        state=getattr(organism,'world_state',None)
        if not state:return None
        value=CONTEXT_TAG
        for atom in state:value=mix64(value^mix64(int(atom)))
        return int(value&((1<<63)-1) or 1)
    def _touch(self,ctx):
        if ctx in self.context_order:self.context_order.remove(ctx)
        self.context_order.append(ctx)
        while len(self.context_order)>MAX_CONTEXTS:
            dead=self.context_order.pop(0);self.histories.pop(dead,None)
    def observe(self,organism,vector_occurrence):
        ctx=self.context(organism)
        if ctx is None:raise ValueError('context_visual_prediction:world')
        rows=VisualVectorPredictionV1.rows(vector_occurrence);hist=self.histories.setdefault(ctx,[]);hist.append(rows)
        if len(hist)>HISTORY:del hist[:-HISTORY]
        self._touch(ctx);return self.prediction(organism)
    def prediction(self,organism):
        ctx=self.context(organism)
        if ctx is None:return None
        hist=self.histories.get(ctx,())
        counts={}
        for rows in hist:counts[rows]=counts.get(rows,0)+1
        if not counts:return None
        best=max(counts.values());w=[rows for rows,n in counts.items() if n==best]
        return w[0] if best>=QUORUM and len(w)==1 else None
    def residual(self,organism,vector_occurrence):
        pred=self.prediction(organism);obs=VisualVectorPredictionV1.rows(vector_occurrence)
        if pred is None:return None
        ps,os=set(pred),set(obs);return {'missing':tuple(sorted(ps-os)),'unexpected':tuple(sorted(os-ps))}
    def checkpoint(self):
        return {'schema':2,'context_order':list(self.context_order),'histories':[[ctx,[[list(r) for r in rows] for rows in self.histories.get(ctx,())]] for ctx in self.context_order]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=2:raise ValueError('context_visual_prediction:checkpoint')
        out=cls();out.context_order=[int(x) for x in d.get('context_order',())]
        if len(out.context_order)>MAX_CONTEXTS or len(set(out.context_order))!=len(out.context_order):raise ValueError('context_visual_prediction:checkpoint')
        for ctx,hist in d.get('histories',()):
            ctx=int(ctx);rows=[tuple(tuple(int(v) for v in r) for r in rowset) for rowset in hist]
            if len(rows)>HISTORY:raise ValueError('context_visual_prediction:checkpoint')
            out.histories[ctx]=rows
        if set(out.histories)!=set(out.context_order):raise ValueError('context_visual_prediction:checkpoint')
        return out
