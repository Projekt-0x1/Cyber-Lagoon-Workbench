#!/usr/bin/env python3
"""Generic other-referenced prospective learning for fast Adult experiments.

This is deliberately not a Theory-of-Mind ontology. It stores agent-indexed
observation traces and learned successor statistics. Terms such as belief,
knowledge, desire, deception, or emotion never enter resident state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1


def _identity(tag, values):
    raw=json.dumps(values,sort_keys=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(tag.encode()+b'\0'+raw).digest()[:8],'little') or 1


@dataclass(frozen=True)
class AgentObservationTraceV1:
    agent_identity:int
    subject_identity:int
    observed_state_identity:int
    observation_identity:int
    tick:int


class SocialProspectionV1:
    """Predict another source's next behavior from that source's own observed history."""
    def __init__(self,capacity=256,trace_capacity=None):
        self.predictive=PredictiveCreditBankV1(capacity)
        self.trace_capacity=int(capacity if trace_capacity is None else trace_capacity)
        if self.trace_capacity<=0:raise ValueError('social_prospection:trace_capacity')
        self._traces={}
        self._tick=1
        self.trace_evictions=0

    def _trace_predictive_support(self,trace):
        pid=self.trace_program_identity(trace);row=self.predictive.rows.get(pid)
        if row is None:return 0
        return sum(int(s.count) for s in row.successors.values())

    def _trace_retention_key(self,item):
        key,trace=item
        # Predictively supported histories survive one-off strangers. Ties evict
        # older traces first, then use stable source identity for determinism.
        return (self._trace_predictive_support(trace),int(trace.tick),int(key[0]),int(key[1]))

    def observe_contact(self,agent_identity,subject_identity,state_identity,observation_identity):
        agent=int(agent_identity);subject=int(subject_identity);state=int(state_identity);obs=int(observation_identity)
        if min(agent,subject,state,obs)<=0:raise ValueError('social_prospection:observation')
        self._tick+=1;key=(agent,subject)
        if key not in self._traces and len(self._traces)>=self.trace_capacity:
            victim=min(self._traces.items(),key=self._trace_retention_key)[0]
            del self._traces[victim];self.trace_evictions+=1
        self._traces[key]=AgentObservationTraceV1(agent,subject,state,obs,self._tick)
        return self._traces[key]

    def trace(self,agent_identity,subject_identity):
        return self._traces.get((int(agent_identity),int(subject_identity)))

    @staticmethod
    def trace_program_identity(trace:AgentObservationTraceV1):
        # Agent identity is excluded from the learned program body so an action
        # regularity can transfer across agents; current individuality remains in
        # the agent-indexed live trace.
        return _identity('social-trace-program-v1',(
            int(trace.subject_identity),int(trace.observed_state_identity)))

    def observe_behavior(self,agent_identity,subject_identity,action_identity,independent=True):
        trace=self.trace(agent_identity,subject_identity)
        action=int(action_identity)
        if trace is None or action<=0 or not independent:return False
        pid=self.trace_program_identity(trace)
        self.predictive.observe_successor(pid,action,1)
        return True

    def predict_action(self,agent_identity,subject_identity):
        trace=self.trace(agent_identity,subject_identity)
        if trace is None:return 0
        pid=self.trace_program_identity(trace);row=self.predictive.rows.get(pid)
        # Prediction is observation, never an admission operation. An evicted or
        # never-learned program stays unknown rather than allocating state by read.
        return 0 if row is None else row.expected_successor()

    def prediction_error(self,agent_identity,subject_identity,actual_action_identity):
        predicted=self.predict_action(agent_identity,subject_identity)
        actual=int(actual_action_identity)
        if predicted==0 or actual<=0:return 0
        return 0 if predicted==actual else 1

    def pragmatic_context(self,agent_identity,subject_identity,current_state_action_identity):
        """Opaque relation context; relation values are mechanics, not mental labels.

        Distinguish absence of learned predictive matter from a learned predictor
        whose successor competition remains unresolved.  Both produce no unique
        prediction, but they are different lived histories and can lawfully support
        different later discourse without storing a duplicate social-support cache.
        """
        trace=self.trace(agent_identity,subject_identity)
        row=None if trace is None else self.predictive.rows.get(self.trace_program_identity(trace))
        predicted=0 if row is None else row.expected_successor()
        actual=int(current_state_action_identity)
        if actual<=0:raise ValueError('social_prospection:current_action')
        relation=0 if row is None else (3 if predicted==0 else (1 if predicted==actual else 2))
        return _identity('social-pragmatic-relation-v1',(relation,))

    def snapshot(self):
        traces=tuple((k[0],k[1],v.observed_state_identity,v.observation_identity,v.tick)
                     for k,v in sorted(self._traces.items()))
        return traces,self.trace_capacity,self.trace_evictions,self.predictive.snapshot()

    def checkpoint(self):
        return {'schema':1,'trace_capacity':self.trace_capacity,'tick':self._tick,
                'trace_evictions':self.trace_evictions,
                'traces':[{'agent':k[0],'subject':k[1],'state':v.observed_state_identity,
                           'observation':v.observation_identity,'tick':v.tick}
                          for k,v in sorted(self._traces.items())],
                'predictive':self.predictive.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('social_prospection:checkpoint_schema')
        predictive=PredictiveCreditBankV1.restore(data.get('predictive',{}))
        trace_capacity=int(data.get('trace_capacity',0));bank=cls(predictive.capacity,trace_capacity)
        bank.predictive=predictive;bank._tick=int(data.get('tick',0));bank.trace_evictions=int(data.get('trace_evictions',0))
        if bank._tick<1 or bank.trace_evictions<0:raise ValueError('social_prospection:checkpoint_counter')
        for row in data.get('traces',()):
            trace=AgentObservationTraceV1(int(row.get('agent',0)),int(row.get('subject',0)),
                                          int(row.get('state',0)),int(row.get('observation',0)),
                                          int(row.get('tick',0)))
            key=(trace.agent_identity,trace.subject_identity)
            if (min(trace.agent_identity,trace.subject_identity,trace.observed_state_identity,
                    trace.observation_identity,trace.tick)<=0 or key in bank._traces or
                    trace.tick>bank._tick or len(bank._traces)>=bank.trace_capacity):
                raise ValueError('social_prospection:checkpoint_trace')
            bank._traces[key]=trace
        return bank
