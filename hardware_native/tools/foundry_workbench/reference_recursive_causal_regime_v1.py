#!/usr/bin/env python3
"""Latent causal-regime inference for contextual competence routing.

This owner does not learn world transitions, self success, or reason truth. It groups bounded
resident causal cue receipts into latent regimes and may split a regime after an independent
contradictory outcome only when a cue difference supplies a causal basis for separation.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_REGIMES=512
MAX_CUES=32
MIN_REUSE_SIMILARITY=Q//2
MIN_SPLIT_NOVELTY=Q//4
MIN_SPLIT_EVIDENCE=2

@dataclass(frozen=True)
class CausalRegimeV1:
    identity:int
    parent:int
    created_tick:int
    last_tick:int
    observations:int
    authority:int=0

class RecursiveCausalRegimeV1:
    def __init__(self):
        self._regimes=[]
        self._cue_counts={}
        self._next_regime=1
        self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('causal-regime:time-reversal')
        self._tick=tick

    @staticmethod
    def normalize_cues(cues):
        return tuple(sorted(set(int(x) for x in cues if int(x)>0)))[:MAX_CUES]

    def prototype(self,identity):
        rows=self._cue_counts.get(int(identity),{})
        return tuple(cue for cue,_count in sorted(rows.items(),key=lambda item:(-item[1],item[0]))[:MAX_CUES])

    @staticmethod
    def similarity_q16(left,right):
        left=set(map(int,left));right=set(map(int,right))
        if not left and not right:return Q
        if not left or not right:return 0
        return (2*len(left&right)*Q)//(len(left)+len(right))

    def _create(self,cues,tick,parent=0):
        if len(self._regimes)>=MAX_REGIMES:raise RuntimeError('causal-regime:capacity')
        identity=self._next_regime;self._next_regime+=1
        row=CausalRegimeV1(identity,int(parent),int(tick),int(tick),0,0)
        self._regimes.append(row);self._cue_counts[identity]={cue:1 for cue in self.normalize_cues(cues)}
        return identity

    def infer(self,cues,tick,recent_regime=0):
        self._advance(tick);cues=self.normalize_cues(cues)
        if not self._regimes:return self._create(cues,tick,0)
        ranked=[]
        for row in self._regimes:
            similarity=self.similarity_q16(cues,self.prototype(row.identity))
            continuity=Q//16 if int(row.identity)==int(recent_regime) else 0
            ranked.append((min(Q,similarity+continuity),similarity,-int(row.identity),int(row.identity)))
        ranked.sort(reverse=True);_score,similarity,_neg,identity=ranked[0]
        return identity if similarity>=MIN_REUSE_SIMILARITY else self._create(cues,tick,0)

    def _commit_cues(self,identity,cues,tick):
        identity=int(identity);cues=self.normalize_cues(cues);rows=self._cue_counts.setdefault(identity,{})
        for cue in cues:rows[cue]=int(rows.get(cue,0))+1
        regime=next((r for r in self._regimes if int(r.identity)==identity),None)
        if regime is None:raise RuntimeError('causal-regime:missing')
        revised=CausalRegimeV1(regime.identity,regime.parent,regime.created_tick,int(tick),regime.observations+1,0)
        self._regimes[self._regimes.index(regime)]=revised

    def resolve_after_outcome(self,identity,cues,predicted_success_q16,success,
                              independent,prior_action_evidence,tick):
        self._advance(tick);identity=int(identity);cues=self.normalize_cues(cues)
        regime=next((r for r in self._regimes if int(r.identity)==identity),None)
        if regime is None:raise RuntimeError('causal-regime:resolve-missing')
        prediction=max(0,min(Q,int(predicted_success_q16)));success=bool(success)
        strong_contradiction=((prediction>=3*Q//4 and not success) or
                              (prediction<=Q//4 and success))
        novelty=Q-self.similarity_q16(cues,self.prototype(identity))
        if (bool(independent) and int(prior_action_evidence)>=MIN_SPLIT_EVIDENCE
                and strong_contradiction and novelty>=MIN_SPLIT_NOVELTY):
            child=self._create(cues,tick,identity);self._commit_cues(child,cues,tick);return child
        self._commit_cues(identity,cues,tick);return identity

    def regime(self,identity):return next((r for r in self._regimes if int(r.identity)==int(identity)),None)
    @property
    def regime_count(self):return len(self._regimes)

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'next_regime':self._next_regime,
            'regimes':[{'identity':r.identity,'parent':r.parent,'created_tick':r.created_tick,
                'last_tick':r.last_tick,'observations':r.observations,'authority':0} for r in self._regimes],
            'cue_counts':[{'regime':int(identity),'rows':[[int(cue),int(count)] for cue,count in sorted(rows.items())]}
                for identity,rows in sorted(self._cue_counts.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('causal-regime:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._next_regime=int(data.get('next_regime',1))
        out._regimes=[CausalRegimeV1(int(r['identity']),int(r.get('parent',0)),int(r['created_tick']),
            int(r['last_tick']),int(r.get('observations',0)),0) for r in data.get('regimes',())]
        out._cue_counts={int(row['regime']):{int(c):int(n) for c,n in row.get('rows',())}
            for row in data.get('cue_counts',())}
        if len(out._regimes)>MAX_REGIMES:raise RuntimeError('causal-regime:capacity')
        return out
