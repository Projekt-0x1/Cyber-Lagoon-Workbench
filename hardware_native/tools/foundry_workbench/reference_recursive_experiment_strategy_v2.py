#!/usr/bin/env python3
"""Regime-conditioned learned experiment-geometry competence over V1.

V1 global history is retained for compatibility. V2 routes current learned suppression and
rehabilitation through `(regime, structural strategy key)` so competence does not leak across
latent causal regimes.
"""
from __future__ import annotations
from reference_recursive_experiment_strategy_v1 import (
    RecursiveExperimentStrategyV1,Q,MAX_STRATEGIES,
)

class RecursiveExperimentStrategyV2(RecursiveExperimentStrategyV1):
    def __init__(self):
        super().__init__();self._regime_useful={};self._regime_uninformative={};self._intervention_regime={}

    @property
    def regime_evidence_count(self):return sum(self._regime_useful.values())+sum(self._regime_uninformative.values())

    def _rkey(self,key,regime):return (int(regime),*self._tuple(key))

    def competence_q16(self,key,regime=0):
        regime=int(regime)
        if regime<=0:return super().competence_q16(key)
        k=self._rkey(key,regime);good=int(self._regime_useful.get(k,0));bad=int(self._regime_uninformative.get(k,0))
        return ((good+1)*Q)//(good+bad+2)

    def permits(self,key,regime=0):
        regime=int(regime)
        if regime<=0:return super().permits(key)
        k=self._rkey(key,regime);good=int(self._regime_useful.get(k,0));bad=int(self._regime_uninformative.get(k,0))
        return not (good==0 and bad>=2 and self.competence_q16(key,regime)<Q//3)

    def begin(self,intervention,key,before_certainty_q16,self_selected=True,regime=0):
        ok=super().begin(intervention,key,before_certainty_q16,self_selected)
        if ok:self._intervention_regime[int(intervention)]=int(regime)
        return ok

    def settle(self,intervention,after_certainty_q16,independent=True,self_selected=True,resolved_regime=0):
        intervention=int(intervention);row=self.trace(intervention)
        if row is None or row.settled:return False
        result=super().settle(intervention,after_certainty_q16,independent,self_selected)
        if not bool(independent) or not bool(self_selected and row.self_selected):return result
        regime=int(resolved_regime or self._intervention_regime.get(intervention,0));self._intervention_regime[intervention]=regime
        if regime>0:
            diagnosticity=max(0,min(Q,int(after_certainty_q16))-int(row.before_certainty_q16))
            key=self._rkey(row.key,regime);bucket=self._regime_useful if diagnosticity>0 else self._regime_uninformative
            bucket[key]=int(bucket.get(key,0))+1
            if len(self._regime_useful)+len(self._regime_uninformative)>2*MAX_STRATEGIES*4:raise RuntimeError('experiment-strategy-v2:capacity')
        return result

    def checkpoint(self):
        data=super().checkpoint()
        def rows(values):return [[*key,int(value)] for key,value in sorted(values.items())]
        data['v2_regime_strategy']={'useful':rows(self._regime_useful),'uninformative':rows(self._regime_uninformative),
            'intervention_regime':[[int(i),int(g)] for i,g in sorted(self._intervention_regime.items())]};return data

    @classmethod
    def restore(cls,data):
        prior=RecursiveExperimentStrategyV1.restore(data);out=cls();out.__dict__.update(prior.__dict__)
        row=data.get('v2_regime_strategy',{})
        def load(rows):return {tuple(map(int,x[:-1])):int(x[-1]) for x in rows}
        out._regime_useful=load(row.get('useful',()));out._regime_uninformative=load(row.get('uninformative',()))
        out._intervention_regime={int(i):int(g) for i,g in row.get('intervention_regime',())};return out
