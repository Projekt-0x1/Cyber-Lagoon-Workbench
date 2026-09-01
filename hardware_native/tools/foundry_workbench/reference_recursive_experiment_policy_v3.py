#!/usr/bin/env python3
"""Regime-conditioned adaptive experiment-policy mixture over V2.

V2 global/context-bin history remains compatibility state. V3 conditions current policy
competence on the inferred latent causal regime so learned search style cannot leak across
regimes with otherwise similar resource/control/ambiguity bins.
"""
from __future__ import annotations
from reference_recursive_experiment_policy_v2 import (
    RecursiveExperimentPolicyV2,ExperimentPolicyTraceV2,
    POLICY_DISCRIMINATE,POLICY_CONFIRM,Q,MAX_POLICY_CONTEXTS,
)

class RecursiveExperimentPolicyV3(RecursiveExperimentPolicyV2):
    def __init__(self):
        super().__init__();self._regime_useful={};self._regime_uninformative={};self._intervention_regime={}

    @property
    def regime_evidence_count(self):return sum(self._regime_useful.values())+sum(self._regime_uninformative.values())

    def _rkey(self,policy,context,regime):return (int(regime),int(policy),*self._ctx(context))

    def competence_q16(self,policy,context,regime=0):
        regime=int(regime)
        if regime<=0:return super().competence_q16(policy,context)
        key=self._rkey(policy,context,regime);good=int(self._regime_useful.get(key,0));bad=int(self._regime_uninformative.get(key,0))
        return ((good+1)*Q)//(good+bad+2)

    def evidence_for(self,policy,context,regime=0):
        regime=int(regime)
        if regime<=0:return super().evidence_for(policy,context)
        key=self._rkey(policy,context,regime);return int(self._regime_useful.get(key,0))+int(self._regime_uninformative.get(key,0))

    def choose_policy(self,candidates,resource_q16,controllability_q16,regime=0):
        candidates=tuple(candidates)
        if not candidates:return 0,None
        context=self.context(candidates,resource_q16,controllability_q16);regime=int(regime)
        if regime<=0:return super().choose_policy(candidates,resource_q16,controllability_q16)
        d_e=self.evidence_for(POLICY_DISCRIMINATE,context,regime);c_e=self.evidence_for(POLICY_CONFIRM,context,regime)
        d=self.competence_q16(POLICY_DISCRIMINATE,context,regime);c=self.competence_q16(POLICY_CONFIRM,context,regime)
        if d_e==0 and c_e==0:policy=POLICY_DISCRIMINATE if context.ambiguity_class>=2 or context.live_reason_class>=2 else POLICY_CONFIRM
        elif d==c:policy=POLICY_DISCRIMINATE if d_e<=c_e else POLICY_CONFIRM
        else:policy=POLICY_DISCRIMINATE if d>c else POLICY_CONFIRM
        return int(policy),context

    def permits(self,policy,context,regime=0):
        regime=int(regime)
        if regime<=0:return super().permits(policy,context)
        evidence=self.evidence_for(policy,context,regime)
        return not (evidence>=2 and self.competence_q16(policy,context,regime)<Q//3)

    def begin(self,intervention,policy,context,before_certainty_q16,self_selected=True,regime=0):
        ok=super().begin(intervention,policy,context,before_certainty_q16,self_selected)
        if ok:self._intervention_regime[int(intervention)]=int(regime)
        return ok

    def settle(self,intervention,after_certainty_q16,independent=True,self_selected=True,resolved_regime=0):
        intervention=int(intervention);row=self.trace(intervention)
        if row is None or row.settled:return False
        result=super().settle(intervention,after_certainty_q16,independent,self_selected)
        if not bool(independent) or not bool(self_selected and row.self_selected):return result
        regime=int(resolved_regime or self._intervention_regime.get(intervention,0));self._intervention_regime[intervention]=regime
        if regime>0:
            diagnosticity=max(0,min(Q,int(after_certainty_q16))-min(Q,max(0,int(row.before_certainty_q16))))
            key=self._rkey(row.policy,row.context,regime);bucket=self._regime_useful if diagnosticity>0 else self._regime_uninformative
            bucket[key]=int(bucket.get(key,0))+1
            if len(self._regime_useful)+len(self._regime_uninformative)>8*MAX_POLICY_CONTEXTS:raise RuntimeError('experiment-policy-v3:capacity')
        return result

    def checkpoint(self):
        data=super().checkpoint()
        def rows(values):return [[*key,int(value)] for key,value in sorted(values.items())]
        data['v3_regime_policy']={'useful':rows(self._regime_useful),'uninformative':rows(self._regime_uninformative),
            'intervention_regime':[[int(i),int(g)] for i,g in sorted(self._intervention_regime.items())]};return data

    @classmethod
    def restore(cls,data):
        prior=RecursiveExperimentPolicyV2.restore(data);out=cls();out.__dict__.update(prior.__dict__)
        row=data.get('v3_regime_policy',{})
        def load(rows):return {tuple(map(int,x[:-1])):int(x[-1]) for x in rows}
        out._regime_useful=load(row.get('useful',()));out._regime_uninformative=load(row.get('uninformative',()))
        out._intervention_regime={int(i):int(g) for i,g in row.get('intervention_regime',())};return out
