#!/usr/bin/env python3
"""Context-conditional learned mixture of safe experiment policies.

This owner learns only which zero-authority search policy has produced realized diagnosticity
in a bounded causal-control context. It never enumerates actions, never relaxes present-state
safety, and never writes world/self/reason evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_recursive_causal_experiment_v2 import (
    POLICY_DISCRIMINATE,POLICY_CONFIRM,Q,
)

MAX_POLICY_CONTEXTS=2048
MAX_POLICY_TRACES=4096

@dataclass(frozen=True)
class ExperimentPolicyContextV2:
    candidates:int
    live_reason_class:int
    ambiguity_class:int
    control_class:int
    resource_class:int
    pressure_class:int

@dataclass(frozen=True)
class ExperimentPolicyTraceV2:
    intervention:int
    policy:int
    context:ExperimentPolicyContextV2
    self_selected:bool
    before_certainty_q16:int
    settled:bool=False
    independent:bool=False
    realized_diagnosticity_q16:int=0
    authority:int=0

class RecursiveExperimentPolicyV2:
    def __init__(self):
        self._useful={};self._uninformative={};self._traces=[]

    @staticmethod
    def context(candidates,resource_q16,controllability_q16):
        candidates=tuple(candidates)
        live=sum(1 for row in candidates if int(row.reason_count)>0)
        max_gap=max((int(row.epistemic_gap_q16) for row in candidates),default=0)
        resource=int(resource_q16);control=int(controllability_q16)
        return ExperimentPolicyContextV2(
            min(8,len(candidates)),
            0 if live==0 else (1 if live==1 else 2),
            0 if max_gap<Q//3 else (1 if max_gap<2*Q//3 else 2),
            0 if control<Q//2 else (1 if control<3*Q//4 else 2),
            0 if resource<Q//3 else (1 if resource<2*Q//3 else 2),
            1 if resource<Q//2 else 0,
        )

    @staticmethod
    def _ctx(context):
        return (context.candidates,context.live_reason_class,context.ambiguity_class,
            context.control_class,context.resource_class,context.pressure_class)

    @property
    def evidence_count(self):return sum(self._useful.values())+sum(self._uninformative.values())

    def competence_q16(self,policy,context):
        key=(int(policy),*self._ctx(context));good=int(self._useful.get(key,0));bad=int(self._uninformative.get(key,0))
        return ((good+1)*Q)//(good+bad+2)

    def evidence_for(self,policy,context):
        key=(int(policy),*self._ctx(context));return int(self._useful.get(key,0))+int(self._uninformative.get(key,0))

    def choose_policy(self,candidates,resource_q16,controllability_q16):
        candidates=tuple(candidates)
        if not candidates:return 0,None
        context=self.context(candidates,resource_q16,controllability_q16)
        # Prior policy preference is contextual, not global. With no evidence, high ambiguity
        # defaults to discrimination while lower ambiguity uses the cheaper confirmatory path.
        d_e=self.evidence_for(POLICY_DISCRIMINATE,context);c_e=self.evidence_for(POLICY_CONFIRM,context)
        d=self.competence_q16(POLICY_DISCRIMINATE,context);c=self.competence_q16(POLICY_CONFIRM,context)
        if d_e==0 and c_e==0:
            policy=POLICY_DISCRIMINATE if context.ambiguity_class>=2 or context.live_reason_class>=2 else POLICY_CONFIRM
        elif d==c:
            policy=POLICY_DISCRIMINATE if d_e<=c_e else POLICY_CONFIRM
        else:policy=POLICY_DISCRIMINATE if d>c else POLICY_CONFIRM
        return int(policy),context

    def permits(self,policy,context):
        evidence=self.evidence_for(policy,context)
        return not (evidence>=2 and self.competence_q16(policy,context)<Q//3)

    def begin(self,intervention,policy,context,before_certainty_q16,self_selected=True):
        intervention=int(intervention);policy=int(policy)
        if intervention<=0 or policy not in (POLICY_DISCRIMINATE,POLICY_CONFIRM):return False
        row=ExperimentPolicyTraceV2(intervention,policy,context,bool(self_selected),int(before_certainty_q16),False,False,0,0)
        self._traces=[x for x in self._traces if int(x.intervention)!=intervention]
        if len(self._traces)>=MAX_POLICY_TRACES:self._traces.pop(0)
        self._traces.append(row);return True

    def settle(self,intervention,after_certainty_q16,independent=True,self_selected=True):
        intervention=int(intervention);row=next((x for x in self._traces if int(x.intervention)==intervention),None)
        if row is None or row.settled:return False
        independent=bool(independent);self_selected=bool(self_selected and row.self_selected)
        diagnosticity=max(0,min(Q,int(after_certainty_q16))-min(Q,max(0,int(row.before_certainty_q16))))
        revised=ExperimentPolicyTraceV2(row.intervention,row.policy,row.context,row.self_selected,True,independent,diagnosticity,0)
        self._traces[self._traces.index(row)]=revised
        if not independent or not self_selected:return False
        key=(int(row.policy),*self._ctx(row.context));bucket=self._useful if diagnosticity>0 else self._uninformative
        bucket[key]=int(bucket.get(key,0))+1
        if len(self._useful)+len(self._uninformative)>4*MAX_POLICY_CONTEXTS:raise RuntimeError('experiment-policy:capacity')
        return diagnosticity>0

    def trace(self,intervention):return next((x for x in self._traces if int(x.intervention)==int(intervention)),None)

    def checkpoint(self):
        def rows(values):return [[*key,int(value)] for key,value in sorted(values.items())]
        return {'schema':1,'useful':rows(self._useful),'uninformative':rows(self._uninformative),
            'traces':[{'intervention':x.intervention,'policy':x.policy,'context':list(self._ctx(x.context)),
                'self_selected':x.self_selected,'before':x.before_certainty_q16,'settled':x.settled,
                'independent':x.independent,'diagnosticity':x.realized_diagnosticity_q16,'authority':0}
                for x in self._traces]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('experiment-policy:checkpoint-schema')
        out=cls()
        def load(rows):return {tuple(map(int,row[:-1])):int(row[-1]) for row in rows}
        out._useful=load(data.get('useful',()));out._uninformative=load(data.get('uninformative',()))
        for row in data.get('traces',()):
            context=ExperimentPolicyContextV2(*map(int,row['context']))
            out._traces.append(ExperimentPolicyTraceV2(int(row['intervention']),int(row['policy']),context,
                bool(row['self_selected']),int(row['before']),bool(row['settled']),bool(row['independent']),
                int(row['diagnosticity']),0))
        if len(out._traces)>MAX_POLICY_TRACES:raise RuntimeError('experiment-policy:trace-capacity')
        return out
