#!/usr/bin/env python3
from __future__ import annotations
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1
HISTORY=8;QUORUM=3;MAX_CONTEXTS=8
class RelationalContextVisualPredictionV1:
    def __init__(self):self.histories={};self.order=[]
    @staticmethod
    def state(o):
        s=getattr(o,'world_state',None)
        return None if not s else tuple(sorted(set(int(x) for x in s)))
    @staticmethod
    def related(a,b):
        a=set(a);b=set(b);den=max(len(a),len(b))
        return bool(den and len(a&b)*4>=den*3)
    def _touch(self,state):
        if state in self.order:self.order.remove(state)
        self.order.append(state)
        while len(self.order)>MAX_CONTEXTS:
            dead=self.order.pop(0);self.histories.pop(dead,None)
    @staticmethod
    def _winner(rows):
        counts={}
        for r in rows:counts[r]=counts.get(r,0)+1
        if not counts:return None
        best=max(counts.values());w=[r for r,n in counts.items() if n==best]
        return w[0] if best>=QUORUM and len(w)==1 else None
    def observe(self,o,v):
        state=self.state(o)
        if state is None:raise ValueError('rel_context_prediction:world')
        rows=VisualVectorPredictionV1.rows(v);hist=self.histories.setdefault(state,[]);hist.append(rows)
        if len(hist)>HISTORY:del hist[:-HISTORY]
        self._touch(state);return self.prediction(o)
    def prediction(self,o):
        state=self.state(o)
        if state is None:return None
        exact=self.histories.get(state)
        if exact:
            winner=self._winner(exact)
            if winner is not None:return winner
            # exact ambiguous experience blocks donor substitution
            if len(exact)>=QUORUM:return None
        pooled=[]
        for donor,hist in self.histories.items():
            if donor!=state and self.related(state,donor):
                winner=self._winner(hist)
                if winner is not None:pooled.extend([winner]*sum(1 for x in hist if x==winner))
        return self._winner(pooled)
    def residual(self,o,v):
        pred=self.prediction(o);obs=VisualVectorPredictionV1.rows(v)
        if pred is None:return None
        ps,os=set(pred),set(obs);return {'missing':tuple(sorted(ps-os)),'unexpected':tuple(sorted(os-ps))}
    def checkpoint(self):return {'schema':1,'order':[list(s) for s in self.order],'histories':[[list(s),[[list(r) for r in rows] for rows in self.histories[s]]] for s in self.order]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('rel_context_prediction:checkpoint')
        out=cls();out.order=[tuple(int(x) for x in s) for s in d.get('order',())]
        if len(out.order)>MAX_CONTEXTS or len(set(out.order))!=len(out.order):raise ValueError('rel_context_prediction:checkpoint')
        for s,hist in d.get('histories',()):
            state=tuple(int(x) for x in s);rows=[tuple(tuple(int(v) for v in r) for r in rowset) for rowset in hist]
            if len(rows)>HISTORY:raise ValueError('rel_context_prediction:checkpoint')
            out.histories[state]=rows
        if set(out.histories)!=set(out.order):raise ValueError('rel_context_prediction:checkpoint')
        return out
