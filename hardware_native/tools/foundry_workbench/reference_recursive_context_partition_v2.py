#!/usr/bin/env python3
"""Explicit memory-routing contexts without treating context identity as world ontology.

WORLD_CONTEXT groups exteroceptive world cues and may split only from a supported pre-outcome
transition mismatch plus world-cue novelty. Selecting a WORLD_CONTEXT before outcome does not
write its prototype; the lived transition settles merge/split first. CONTROL_CONTEXT groups
performance/control cues and may update immediately because it is a retrieval/control index,
not a hypothesis about an external latent cause. All rows have authority zero.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
WORLD_CONTEXT=1
CONTROL_CONTEXT=2
MAX_CONTEXTS=1024
MAX_CUES=32
MIN_REUSE_SIMILARITY=Q//2
MIN_SPLIT_NOVELTY=Q//4
MIN_WORLD_SUPPORT=2

@dataclass(frozen=True)
class ContextPartitionV2:
    identity:int;kind:int;parent:int;created_tick:int;last_tick:int;observations:int;authority:int=0

class RecursiveContextPartitionV2:
    def __init__(self):self._rows=[];self._cue_counts={};self._next=1;self._tick=-1
    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('context-partition:time-reversal')
        self._tick=tick
    @staticmethod
    def normalize(cues):return tuple(sorted(set(int(x) for x in cues if int(x)>0)))[:MAX_CUES]
    @staticmethod
    def similarity_q16(left,right):
        left=set(map(int,left));right=set(map(int,right))
        if not left and not right:return Q
        if not left or not right:return 0
        return (2*len(left&right)*Q)//(len(left)+len(right))
    def prototype(self,identity):
        rows=self._cue_counts.get(int(identity),{})
        return tuple(c for c,_n in sorted(rows.items(),key=lambda item:(-item[1],item[0]))[:MAX_CUES])
    def _create(self,kind,cues,tick,parent=0):
        if len(self._rows)>=MAX_CONTEXTS:raise RuntimeError('context-partition:capacity')
        identity=self._next;self._next+=1
        self._rows.append(ContextPartitionV2(identity,int(kind),int(parent),int(tick),int(tick),0,0))
        self._cue_counts[identity]={c:1 for c in self.normalize(cues)};return identity
    def _commit(self,identity,cues,tick):
        identity=int(identity);cues=self.normalize(cues);bucket=self._cue_counts.setdefault(identity,{})
        for cue in cues:bucket[cue]=int(bucket.get(cue,0))+1
        row=next((r for r in self._rows if r.identity==identity),None)
        if row is None:raise RuntimeError('context-partition:missing')
        self._rows[self._rows.index(row)]=ContextPartitionV2(row.identity,row.kind,row.parent,row.created_tick,int(tick),row.observations+1,0)
    def infer(self,kind,cues,tick,recent=0):
        self._advance(tick);kind=int(kind);cues=self.normalize(cues)
        candidates=[r for r in self._rows if int(r.kind)==kind]
        if not candidates:return self._create(kind,cues,tick,0)
        ranked=[]
        for row in candidates:
            sim=self.similarity_q16(cues,self.prototype(row.identity));continuity=Q//16 if row.identity==int(recent) else 0
            ranked.append((min(Q,sim+continuity),sim,-row.identity,row.identity))
        ranked.sort(reverse=True);_score,sim,_neg,identity=ranked[0]
        if sim<MIN_REUSE_SIMILARITY:return self._create(kind,cues,tick,0)
        # External world cues are only provisional classification before their transition
        # settles. Committing them here would erase novelty before mismatch can be assessed.
        if kind==CONTROL_CONTEXT:self._commit(identity,cues,tick)
        return identity
    def resolve_world_after_transition(self,identity,cues,predicted_state,actual_state,prediction_support,independent,tick):
        self._advance(tick);identity=int(identity);cues=self.normalize(cues)
        row=next((r for r in self._rows if r.identity==identity and r.kind==WORLD_CONTEXT),None)
        if row is None:raise RuntimeError('context-partition:world-missing')
        predicted=tuple(sorted(set(map(int,predicted_state or ()))))
        actual=tuple(sorted(set(map(int,actual_state or ()))))
        mismatch=bool(predicted) and predicted!=actual
        novelty=Q-self.similarity_q16(cues,self.prototype(identity))
        if bool(independent) and int(prediction_support)>=MIN_WORLD_SUPPORT and mismatch and novelty>=MIN_SPLIT_NOVELTY:
            child=self._create(WORLD_CONTEXT,cues,tick,identity);self._commit(child,cues,tick);return child
        self._commit(identity,cues,tick);return identity
    def row(self,identity):return next((r for r in self._rows if r.identity==int(identity)),None)
    @property
    def context_count(self):return len(self._rows)
    def checkpoint(self):
        return {'schema':2,'tick':self._tick,'next':self._next,
            'rows':[{'identity':r.identity,'kind':r.kind,'parent':r.parent,'created_tick':r.created_tick,'last_tick':r.last_tick,'observations':r.observations,'authority':0} for r in self._rows],
            'cue_counts':[{'identity':i,'rows':[[c,n] for c,n in sorted(rows.items())]} for i,rows in sorted(self._cue_counts.items())]}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=2:raise RuntimeError('context-partition:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._next=int(data.get('next',1))
        out._rows=[ContextPartitionV2(int(r['identity']),int(r['kind']),int(r.get('parent',0)),int(r['created_tick']),int(r['last_tick']),int(r.get('observations',0)),0) for r in data.get('rows',())]
        out._cue_counts={int(x['identity']):{int(c):int(n) for c,n in x.get('rows',())} for x in data.get('cue_counts',())}
        if len(out._rows)>MAX_CONTEXTS:raise RuntimeError('context-partition:capacity')
        return out
