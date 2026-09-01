#!/usr/bin/env python3
"""Learned competence for organism-selected causal probe geometries.

This owner never learns world truth and never relaxes present-state probe safety. It learns
only whether a structural probe geometry, when self-selected, actually increased certainty
of the procedural predictions exposed by that probe after an independent lived consequence.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_STRATEGIES=1024
MAX_TRACES=2048


def _clip(value,lo=0,hi=Q):return max(lo,min(hi,int(value)))


@dataclass(frozen=True)
class ExperimentStrategyKeyV1:
    competitors:int
    reason_count:int
    gap_rank:int
    risk_rank:int
    control_class:int


@dataclass(frozen=True)
class ExperimentStrategyTraceV1:
    intervention:int
    key:ExperimentStrategyKeyV1
    before_certainty_q16:int
    self_selected:bool
    settled:bool=False
    independent:bool=False
    realized_diagnosticity_q16:int=0
    authority:int=0


class RecursiveExperimentStrategyV1:
    """Acquired learn-to-learn state; authority is control priority only."""
    def __init__(self):
        self._useful={};self._uninformative={};self._traces=[]

    @staticmethod
    def structural_key(branch,branches,reason_count,controllability_q16):
        branches=tuple(branches);action=int(branch.action)
        others=[row for row in branches if int(row.action)!=action]
        gap_order=sorted(set(int(row.epistemic_gap_q16) for row in branches),reverse=True)
        risk_order=sorted(set(int(row.consequence_risk_q16) for row in branches))
        gap_rank=gap_order.index(int(branch.epistemic_gap_q16)) if int(branch.epistemic_gap_q16) in gap_order else len(gap_order)
        risk_rank=risk_order.index(int(branch.consequence_risk_q16)) if int(branch.consequence_risk_q16) in risk_order else len(risk_order)
        return ExperimentStrategyKeyV1(min(8,len(others)),min(8,max(0,int(reason_count))),
            min(8,gap_rank),min(8,risk_rank),1 if int(controllability_q16)>=3*Q//4 else 0)

    @staticmethod
    def _tuple(key):return (key.competitors,key.reason_count,key.gap_rank,key.risk_rank,key.control_class)

    @property
    def evidence_count(self):return sum(self._useful.values())+sum(self._uninformative.values())

    def competence_q16(self,key):
        k=self._tuple(key);good=int(self._useful.get(k,0));bad=int(self._uninformative.get(k,0))
        return ((good+1)*Q)//(good+bad+2)

    def permits(self,key):
        k=self._tuple(key);good=int(self._useful.get(k,0));bad=int(self._uninformative.get(k,0))
        # No prior evidence blocks nothing. Strong repeated failure can suppress this
        # structural strategy, but never makes a currently unsafe probe safe.
        return not (good==0 and bad>=2 and self.competence_q16(key)<Q//3)

    def begin(self,intervention,key,before_certainty_q16,self_selected=True):
        intervention=int(intervention)
        if intervention<=0:return False
        row=ExperimentStrategyTraceV1(intervention,key,_clip(before_certainty_q16),bool(self_selected),False,False,0,0)
        self._traces=[x for x in self._traces if int(x.intervention)!=intervention]
        if len(self._traces)>=MAX_TRACES:self._traces.pop(0)
        self._traces.append(row);return True

    def settle(self,intervention,after_certainty_q16,independent=True,self_selected=True):
        intervention=int(intervention);row=next((x for x in self._traces if int(x.intervention)==intervention),None)
        if row is None or row.settled:return False
        independent=bool(independent);self_selected=bool(self_selected and row.self_selected)
        diagnosticity=max(0,_clip(after_certainty_q16)-int(row.before_certainty_q16))
        revised=ExperimentStrategyTraceV1(row.intervention,row.key,row.before_certainty_q16,row.self_selected,True,independent,diagnosticity,0)
        self._traces[self._traces.index(row)]=revised
        if not independent or not self_selected:return False
        k=self._tuple(row.key);bucket=self._useful if diagnosticity>0 else self._uninformative
        bucket[k]=int(bucket.get(k,0))+1
        if len(self._useful)+len(self._uninformative)>2*MAX_STRATEGIES:raise RuntimeError('experiment-strategy:capacity')
        return diagnosticity>0

    def trace(self,intervention):return next((x for x in self._traces if int(x.intervention)==int(intervention)),None)

    def checkpoint(self):
        def rows(values):return [[*key,int(value)] for key,value in sorted(values.items())]
        return {'schema':1,'useful':rows(self._useful),'uninformative':rows(self._uninformative),
            'traces':[{'intervention':x.intervention,'key':list(self._tuple(x.key)),'before':x.before_certainty_q16,
                'self_selected':x.self_selected,'settled':x.settled,'independent':x.independent,
                'diagnosticity':x.realized_diagnosticity_q16,'authority':0} for x in self._traces]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('experiment-strategy:checkpoint-schema')
        out=cls()
        def load(rows):return {tuple(map(int,row[:5])):int(row[5]) for row in rows}
        out._useful=load(data.get('useful',()));out._uninformative=load(data.get('uninformative',()))
        out._traces=[]
        for row in data.get('traces',()):
            key=ExperimentStrategyKeyV1(*map(int,row['key']))
            out._traces.append(ExperimentStrategyTraceV1(int(row['intervention']),key,int(row['before']),
                bool(row['self_selected']),bool(row['settled']),bool(row['independent']),int(row['diagnosticity']),0))
        if len(out._traces)>MAX_TRACES:raise RuntimeError('experiment-strategy:trace-capacity')
        return out
