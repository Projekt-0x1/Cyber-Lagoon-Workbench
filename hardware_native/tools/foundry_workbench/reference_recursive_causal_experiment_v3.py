#!/usr/bin/env python3
"""Regime-conditioned procedural reason calibration over the V2 safe frontier.

V2 retains global compatibility history and present-state safety. V3 adds
`(regime, reason, source, action)` evidence. Context-specific evidence can be deferred until
the latent-regime owner resolves merge versus split after the ordinary world consequence.
"""
from __future__ import annotations
from reference_recursive_causal_experiment_v2 import (
    RecursiveCausalExperimentV2,SafeProbeCandidateV2,Q,_clip,
)

class RecursiveCausalExperimentV3(RecursiveCausalExperimentV2):
    def __init__(self):
        super().__init__();self._regime_reason_success={};self._regime_reason_failure={}
        self._intervention_regime={};self._deferred_regime=set()

    @property
    def regime_reason_outcome_count(self):return sum(self._regime_reason_success.values())+sum(self._regime_reason_failure.values())

    def reason_evidence_count(self,reason,source,action,regime=0):
        regime=int(regime)
        if regime<=0:return super().reason_evidence_count(reason,source,action)
        key=(regime,int(reason),int(source),int(action))
        return int(self._regime_reason_success.get(key,0))+int(self._regime_reason_failure.get(key,0))

    def reason_reliability_q16(self,reason,source,action,regime=0):
        regime=int(regime)
        if regime<=0:return super().reason_reliability_q16(reason,source,action)
        key=(regime,int(reason),int(source),int(action));good=int(self._regime_reason_success.get(key,0));bad=int(self._regime_reason_failure.get(key,0))
        return ((good+1)*Q)//(good+bad+2)

    def reason_calibration(self,action,predictions,regime=0):
        regime=int(regime)
        if regime<=0:return super().reason_calibration(action,predictions)
        deltas=[];evidence=0;action=int(action)
        for prediction in predictions:
            if int(prediction.source) in self._withdrawn_sources:continue
            count=self.reason_evidence_count(prediction.reason,prediction.source,action,regime)
            if count<=0:continue
            reliability=self.reason_reliability_q16(prediction.reason,prediction.source,action,regime)
            direction=1 if bool(prediction.predicts_success) else -1
            deltas.append(direction*(int(reliability)-Q//2));evidence+=count
        if not deltas:return 0,0
        return max(-Q//2,min(Q//2,sum(deltas)//len(deltas))),min(Q,evidence*(Q//4))

    def _uncertainty_q16(self,action,predictions,regime=0):
        rows=[self.reason_reliability_q16(p.reason,p.source,action,regime) for p in predictions
            if int(p.source) not in self._withdrawn_sources]
        if not rows:return Q//2
        return sum(Q-abs(2*value-Q) for value in rows)//len(rows)

    def information_gain_q16(self,branch,all_branches,predictions,regime=0):
        competitors=[b for b in all_branches if int(b.action)!=int(branch.action)]
        ambiguity=_clip(branch.epistemic_gap_q16)
        competition=(Q-_clip(min(abs(int(branch.support_q16)-int(other.support_q16)) for other in competitors))) if competitors else 0
        return _clip((2*ambiguity+competition+self._uncertainty_q16(int(branch.action),predictions,regime))//4)

    def eligible_probes(self,branches,reason_map,resource_q16,controllability_q16,regime=0):
        resource=_clip(resource_q16);control=_clip(controllability_q16)
        if resource<Q//3 or control<Q//2:return ()
        branches=tuple(branches);rows=[]
        for branch in branches:
            action=int(branch.action);risk=_clip(branch.consequence_risk_q16);gap=_clip(branch.epistemic_gap_q16)
            if action<=0 or risk>Q//4 or gap<Q//3:continue
            predictions=tuple(reason_map.get(action,()))
            discrimination=self.information_gain_q16(branch,branches,predictions,regime)
            confirmation=_clip((2*_clip(branch.support_q16)+_clip(branch.counterfactual_q16)+(Q-risk))//4)
            rows.append(SafeProbeCandidateV2(action,discrimination,confirmation,risk,gap,min(8,len(predictions)),0))
        return tuple(sorted(rows,key=lambda row:row.action))

    def begin(self,action,decision,predictions,alternatives,information_gain_q16,tick,regime=0,
              defer_regime=False):
        identity=super().begin(action,decision,predictions,alternatives,information_gain_q16,tick)
        if identity:
            self._intervention_regime[int(identity)]=int(regime)
            if bool(defer_regime):self._deferred_regime.add(int(identity))
        return identity

    def intervention_regime(self,identity):return int(self._intervention_regime.get(int(identity),0))

    def _record_regime_reason(self,row,regime,success):
        regime=int(regime);success=bool(success)
        if regime<=0:return False
        wrote=False
        for prediction in row.reasons:
            if int(prediction.source) in self._withdrawn_sources:continue
            matched=(success==bool(prediction.predicts_success));bucket=self._regime_reason_success if matched else self._regime_reason_failure
            key=(regime,int(prediction.reason),int(prediction.source),int(row.action));bucket[key]=int(bucket.get(key,0))+1;wrote=True
        return wrote

    def settle(self,identity,success,independent,tick):
        identity=int(identity);row=self.intervention(identity);regime=self.intervention_regime(identity)
        if row is None or row.settled:return False
        settled=super().settle(identity,success,independent,tick)
        if not bool(independent) or regime<=0 or identity in self._deferred_regime:return settled
        self._record_regime_reason(row,regime,success);return settled

    def settle_regime_evidence(self,identity,resolved_regime):
        identity=int(identity);row=self.intervention(identity)
        if row is None or not row.settled or not row.independent:return False
        self._deferred_regime.discard(identity)
        self._intervention_regime[identity]=int(resolved_regime)
        return self._record_regime_reason(row,int(resolved_regime),bool(row.success))

    def checkpoint(self):
        data=super().checkpoint()
        def rows(values):return [[g,r,s,a,v] for (g,r,s,a),v in sorted(values.items())]
        data['v3_regime_reason']={'success':rows(self._regime_reason_success),'failure':rows(self._regime_reason_failure),
            'intervention_regime':[[int(i),int(g)] for i,g in sorted(self._intervention_regime.items())],
            'deferred_regime':sorted(self._deferred_regime)}
        return data

    @classmethod
    def restore(cls,data):
        prior=RecursiveCausalExperimentV2.restore(data);out=cls();out.__dict__.update(prior.__dict__)
        row=data.get('v3_regime_reason',{})
        out._regime_reason_success={(int(g),int(r),int(s),int(a)):int(v) for g,r,s,a,v in row.get('success',())}
        out._regime_reason_failure={(int(g),int(r),int(s),int(a)):int(v) for g,r,s,a,v in row.get('failure',())}
        out._intervention_regime={int(i):int(g) for i,g in row.get('intervention_regime',())}
        out._deferred_regime=set(map(int,row.get('deferred_regime',())));return out
